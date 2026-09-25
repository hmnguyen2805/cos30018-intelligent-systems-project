"""
Detection Subagent — decides whether a TrafficEvent looks anomalous.

The baseline RandomForest (classifier.py) is a tool the agent calls, not the
agent itself. On a borderline confidence score, the agent re-examines via
per-tree vote spread (a second, finer-grained tool call) before finalizing —
that borderline-handling loop is what makes this an agent rather than a
single classifier call.

For every truly anomalous event, code ALSO decides an attack category —
classifier.predict_attack_category (the multiclass model, trained on
anomalous rows only) gives per-category probabilities; the top class is used
if its probability clears CATEGORY_CONFIDENCE_THRESHOLD, else "Unknown". This
is deterministic and independent of use_llm — same principle as
is_anomalous/confidence, computed by code, never the LLM. Optionally
(`use_llm=True`), a bounded LLM loop then writes a short, grounded
explanation of that already-chosen category for detector_notes; the LLM's
output schema has no category field at all, so it structurally cannot
override the decision. Any invalid, ungrounded, timed-out, or errored LLM
output falls back to a template explanation (the category tag is unaffected
either way). See README.md's guardrails table for the full list.

The MCP connection (a subprocess that loads the model artifact) is opened
lazily on the first event that needs it and reused across events — spawning
it per event would reload the artifact every time. Call warmup() once before
a run (opens the connection and primes the model outside any per-event
timeout) and close() (or use the subagent as a context manager) when done.
If the connection dies mid-run, the current event falls back to a template
note; repeated failures trip a circuit breaker that skips the LLM entirely
for the rest of the run, rather than reconnecting (and paying the model's
slow cold-start again) on every following event.

Owned by the Detection Manager (manager.py), which delegates each event here.
"""
import os
import sys
import threading
import time
import uuid
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from src.detection import classifier
from src.detection.llm_notes import (
    ANSWER_JSON_SCHEMA, REASON_INVALID_JSON, SINGLE_SHOT_SYSTEM_PROMPT, SYSTEM_PROMPT,
    build_template_note, extract_json_object, parse_and_validate,
)
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent

load_dotenv()

BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6
DISAGREEMENT_THRESHOLD = 0.15  # tree-vote std above this = low ensemble consensus

# Below this, classifier.predict_attack_category's top class isn't trusted and code reports
# "Unknown" instead of a possibly-wrong specific category. Same role as the borderline band
# above, but for the category decision rather than is_anomalous.
DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD = 0.6
CATEGORY_TOP_K = 3

MAX_LLM_TOOL_CALLS = 3
MAX_LLM_OUTPUT_TOKENS = 300  # caps generation length — the main lever on a slow local CPU model
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LLM_MODEL = "ollama_chat/qwen2.5:7b"
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 3

# "agent": ToolCallingAgent loop (default) — for models that reliably honor tool_choice, e.g.
# hosted ones like Gemini. "single_shot": code calls top_features (and tree_vote_spread, if
# borderline) via MCP directly, then makes ONE LLM call with response_format's json_schema —
# the reliable path for small local models. See litellm's ollama transformation, which drops
# tool_choice entirely ("causes ollama requests to hang"): nothing can force those models
# through the agent loop's "every step is a tool call" requirement, but response_format's
# json_schema is honored (verified against ollama_chat/qwen2.5:3b — Ollama constrains output
# to the schema via grammar-constrained decoding).
DEFAULT_LLM_MODE = "agent"
GROUNDING_TOP_K = 5  # feature count used both to ground the LLM and to check its explanation
RAW_OUTPUT_LOG_CHARS = 500  # truncation length for raw LLM text logged into the trace/CSV

MCP_SERVER_MODULE = "src.detection.mcp_server"

# Trace actions logged on LLM failure, each naming a distinct reason.
# evaluate.py reports these as detector_notes' fallback_reason per event, and
# _record_llm_failure/_record_llm_success use them to drive the circuit breaker.
FALLBACK_REASON_BY_ACTION = {
    "llm_timeout": "timeout",
    "llm_error": "exception",
    "llm_rate_limited": "rate_limited",
    "llm_provider_unavailable": "provider_unavailable",
    "llm_invalid_json": "invalid_json",
    "llm_ungrounded": "ungrounded",
    "llm_circuit_open": "circuit_open",
    # Not a failure — the answer was used — but reported like one so evaluate.py can count
    # how often the salvage path (extract_json_object) had to kick in, see _run_llm_layer.
    "llm_validation_salvaged": "salvaged",
}

# Tools the LLM agent is allowed to call. predict_proba_anomalous is deliberately excluded:
# code already computes p_anomalous and puts it in the task, so exposing it just wastes a
# tool-call step on a small/slow model; it stays on the MCP server for completeness/tests.
# predict_attack_category IS exposed (agent mode only — single_shot never calls MCP tools
# itself): the LLM may investigate it for its own explanation, but code has already made the
# actual category decision directly via classifier.predict_attack_category before the LLM
# ever runs, so calling this tool cannot change detector_notes' category.
# tree_vote_spread is only offered when the event is borderline (vote_std is not None) —
# it's not meaningful otherwise, since the subagent itself never computed it.
LLM_HIDDEN_TOOL_NAMES = {"register_event", "clear_event", "predict_proba_anomalous"}

# Retry-with-backoff schedule for transient LLM-provider errors (rate limits, 5xx,
# connection/timeout issues) — used by both warmup() and per-event calls. 3 retries
# at 2s/4s/8s, i.e. up to 4 attempts total.
LLM_RETRY_DELAYS_SECONDS = (2.0, 4.0, 8.0)

_RATE_LIMIT_STATUS_CODES = {429}
_PROVIDER_UNAVAILABLE_STATUS_CODES = {408, 500, 502, 503, 504}
_MAX_ERROR_CHAIN_DEPTH = 5  # backstop against a pathological/circular __cause__/__context__ chain


def _resolve_llm_timeout(llm_timeout_seconds: Optional[float]) -> float:
    """Explicit constructor arg wins; otherwise DETECTION_LLM_TIMEOUT env var;
    otherwise the hardcoded default."""
    if llm_timeout_seconds is not None:
        return llm_timeout_seconds
    return float(os.environ.get("DETECTION_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT_SECONDS))


def resolve_llm_model_id() -> str:
    """The LiteLLM model id _get_llm_model will actually use — DETECTION_LLM_MODEL
    env var, or the default. Exposed so evaluate.py can print it at startup without
    duplicating the resolution logic."""
    return os.environ.get("DETECTION_LLM_MODEL", DEFAULT_LLM_MODEL)


def resolve_llm_mode() -> str:
    """DETECTION_LLM_MODE env var ("agent" or "single_shot"), or the default.
    An unrecognized value falls back to the default rather than raising —
    resolving config must never be a way for the LLM layer to break run()."""
    mode = os.environ.get("DETECTION_LLM_MODE", DEFAULT_LLM_MODE)
    return mode if mode in ("agent", "single_shot") else DEFAULT_LLM_MODE


def _classify_llm_error(exc: BaseException) -> Optional[str]:
    """Classify an exception raised by an LLM call as a transient, retryable
    provider-side failure: "rate_limited" (HTTP 429), "provider_unavailable"
    (5xx / connection / timeout), or None (not recognized as transient — e.g.
    a bad API key, an MCP-side failure, or a programming error).

    smolagents wraps whatever the model call raises in its own
    AgentGenerationError (`raise AgentGenerationError(...) from e` in
    smolagents/agents.py) — the wrapper itself has no status_code and its
    name matches none of our keywords, so classifying it directly always
    returned None even for a real, retryable provider error. The actual
    litellm/HTTP exception (with its status_code) is one level down, in
    `__cause__`. So: walk the __cause__ / __context__ chain — bounded by
    _MAX_ERROR_CHAIN_DEPTH and a seen-ids guard against a circular chain —
    and classify on the first link that isn't None."""
    current = exc
    seen_ids = set()
    for _ in range(_MAX_ERROR_CHAIN_DEPTH):
        if current is None or id(current) in seen_ids:
            break
        seen_ids.add(id(current))

        reason = _classify_single_llm_error(current)
        if reason is not None:
            return reason

        current = current.__cause__ or current.__context__
    return None


def _classify_single_llm_error(exc: BaseException) -> Optional[str]:
    """Classify one exception in isolation, ignoring __cause__/__context__ —
    the actual status_code / exception-name check. litellm's exceptions
    (which mirror openai's) all carry a status_code for HTTP-backed errors;
    the exception class name is a fallback for connection-style errors that
    might not set one consistently across providers/versions."""
    status_code = getattr(exc, "status_code", None)
    if status_code in _RATE_LIMIT_STATUS_CODES:
        return "rate_limited"
    if status_code in _PROVIDER_UNAVAILABLE_STATUS_CODES:
        return "provider_unavailable"

    exc_name = type(exc).__name__
    if "RateLimit" in exc_name:
        return "rate_limited"
    if any(keyword in exc_name for keyword in
           ("ServiceUnavailable", "InternalServerError", "APIConnectionError", "Timeout", "BadGateway")):
        return "provider_unavailable"
    return None


def _call_with_retry(fn, on_retry=None, delays=LLM_RETRY_DELAYS_SECONDS):
    """Call fn(), retrying with exponential backoff on a transient LLM-provider
    error (see _classify_llm_error), up to len(delays) retries (len(delays) + 1
    attempts total). Raises the last exception once retries are exhausted, or
    immediately if the error isn't classified as transient. on_retry(attempt_
    number, delay_seconds, reason), if given, fires right before each sleep —
    callers use it to log without assuming which thread this runs on."""
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as exc:
            reason = _classify_llm_error(exc)
            is_last_attempt = attempt == len(delays)
            if reason is None or is_last_attempt:
                raise
            delay = delays[attempt]
            if on_retry is not None:
                on_retry(attempt + 1, delay, reason)
            time.sleep(delay)


class DetectionSubagent(BaseAgent):
    name = "detection_subagent"

    def __init__(
        self,
        model_path: Optional[str] = None,
        category_model_path: Optional[str] = None,
        use_llm: bool = False,
        llm_timeout_seconds: Optional[float] = None,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        category_confidence_threshold: float = DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self._artifact = classifier.load_artifact(model_path or classifier.DEFAULT_BINARY_MODEL_PATH)
        self._category_artifact = self._load_category_artifact(category_model_path)
        if self._category_artifact is not None:
            classifier.assert_feature_names_match(self._artifact, self._category_artifact)
        self._category_confidence_threshold = category_confidence_threshold

        self.use_llm = use_llm
        self._llm_timeout_seconds = _resolve_llm_timeout(llm_timeout_seconds)
        self._circuit_breaker_threshold = circuit_breaker_threshold

        # Persistent MCP connection state — lazily opened by _ensure_llm_connection(),
        # torn down by close(). None means "not connected right now". register_event/
        # clear_event tools are kept separate from _llm_tools so the LLM agent never
        # sees them (only code calls them directly).
        self._mcp_client = None
        self._llm_tools_by_name = None
        self._register_event_tool = None
        self._clear_event_tool = None
        self._llm_model = None

        # Connections abandoned mid-call after a timeout — not safe to disconnect from
        # this thread (a background thread may still be using them), so they're queued
        # here for close() (or the abandoned thread itself) to clean up later.
        self._abandoned_mcp_clients: List = []

        # Circuit breaker: after this many consecutive LLM failures, stop attempting the
        # LLM (and stop reconnecting) for the rest of this subagent's life.
        self._consecutive_llm_failures = 0
        self._circuit_open = False

    @staticmethod
    def _load_category_artifact(category_model_path: Optional[str]) -> Optional[dict]:
        """Best-effort: the category model is optional, for backward
        compatibility with an install that only ever ran train_binary.py (or
        an artifact saved before train_category.py existed). When it's
        unavailable, _choose_category always falls back to "Unknown" rather
        than raising — the same "degrade gracefully" pattern as
        feature_medians/feature_mad on the binary artifact."""
        try:
            artifact = classifier.load_artifact(category_model_path or classifier.DEFAULT_CATEGORY_MODEL_PATH)
        except FileNotFoundError:
            return None
        if "category_model" not in artifact:
            return None
        return artifact

    def __enter__(self) -> "DetectionSubagent":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the persistent MCP connection, if one is open, plus any
        connections abandoned after a timeout. Safe to call multiple times,
        or when the LLM layer was never used."""
        clients_to_close = list(self._abandoned_mcp_clients)
        self._abandoned_mcp_clients.clear()
        if self._mcp_client is not None:
            clients_to_close.append(self._mcp_client)
            self._mcp_client = None
            self._llm_tools_by_name = None
            self._register_event_tool = None
            self._clear_event_tool = None

        for client in clients_to_close:
            self._disconnect_client(client)

    def _disconnect_client(self, client) -> None:
        try:
            client.disconnect()
        except Exception as exc:  # noqa: BLE001 - closing must never raise
            self.log_step(
                thought="Error while closing an MCP connection.",
                action="llm_close_error",
                observation=f"{type(exc).__name__}: {exc}",
            )

    def warmup(self) -> dict:
        """Open the MCP connection and send one tiny prompt to the LLM,
        outside of any per-event timeout — so the first real event doesn't
        have to absorb connection setup plus the model's (often slow, on a
        local CPU model or a rate-limited hosted one) first inference. Call
        this once before a run.

        Retries a transient provider error (rate limit / 5xx / connection)
        with backoff (see _call_with_retry), but never raises: returns
        {"ok": bool, "elapsed_seconds": float, "reason": str | None} either
        way, so a broken LLM/provider can't crash the caller — it can only
        report warmup as failed and let the caller decide to skip the LLM
        arm. No-op success ({"ok": True, "elapsed_seconds": 0.0, "reason":
        None}) when use_llm is False."""
        if not self.use_llm:
            return {"ok": True, "elapsed_seconds": 0.0, "reason": None}

        start = time.perf_counter()

        def attempt():
            from smolagents import ToolCallingAgent

            self._ensure_llm_connection()
            agent = ToolCallingAgent(
                tools=[], model=self._get_llm_model(), max_steps=1,
                instructions="Reply with exactly one word: OK.",
            )
            agent.run("Reply with OK.")

        def on_retry(attempt_number, delay, reason):
            self.log_step(
                thought=f"Warmup attempt {attempt_number} failed ({reason}); retrying in {delay:.0f}s.",
                action="llm_warmup_retry",
                observation=reason,
            )

        try:
            _call_with_retry(attempt, on_retry=on_retry)
        except Exception as exc:  # noqa: BLE001 - warmup must never raise
            elapsed = time.perf_counter() - start
            reason = _classify_llm_error(exc) or "exception"
            detail = f"{reason}: {type(exc).__name__}: {exc}"
            self.log_step(
                thought="LLM warmup failed after retries.",
                action="llm_warmup_failed",
                observation=detail,
            )
            return {"ok": False, "elapsed_seconds": elapsed, "reason": detail}

        elapsed = time.perf_counter() - start
        self.log_step(
            thought="Warmed up the MCP connection and LLM before starting the run.",
            action="llm_warmup",
            observation=f"elapsed_seconds={elapsed:.2f}",
        )
        return {"ok": True, "elapsed_seconds": elapsed, "reason": None}

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        event = input_data
        self._trace = []  # fresh trace per event

        p_anomalous = classifier.predict_proba_anomalous(self._artifact, event.features)
        self.log_step(
            thought="Run baseline RandomForest classifier on event features.",
            action="call_classifier",
            tool_input={"n_features": len(event.features)},
            observation=f"p_anomalous={p_anomalous:.3f}",
        )

        final_p = p_anomalous
        notes = None
        vote_std = None

        if BORDERLINE_LOW <= p_anomalous <= BORDERLINE_HIGH:
            vote_frac, vote_std = classifier.tree_vote_spread(self._artifact, event.features)
            self.log_step(
                thought=f"Confidence borderline (p={p_anomalous:.3f}). Re-examine via per-tree "
                        "vote spread before deciding.",
                action="inspect_tree_votes",
                tool_input={"n_features": len(event.features)},
                observation=f"tree_vote_frac={vote_frac:.3f}, tree_vote_std={vote_std:.3f}",
            )
            final_p = vote_frac
            if vote_std >= DISAGREEMENT_THRESHOLD:
                notes = (
                    f"Borderline call, high tree disagreement (std={vote_std:.3f}) — "
                    "flagged low-confidence for downstream correlation/response."
                )
            else:
                notes = f"Borderline call, trees agree (std={vote_std:.3f}) — trusting vote fraction."

        # is_anomalous and confidence are final here — computed purely from classifier.py's
        # outputs. Nothing below this line may change either value; the category decision and
        # the LLM (if it runs) only affect detector_notes.
        is_anomalous = final_p >= 0.5
        confidence = final_p if is_anomalous else 1.0 - final_p

        chosen_category = None
        category_probability = None
        if is_anomalous:
            chosen_category, category_probability = self._choose_category(event)

        if is_anomalous and self.use_llm:
            if self._circuit_open:
                self.log_step(
                    thought="Circuit breaker is open after repeated LLM failures — skipping the "
                            "LLM layer for this event.",
                    action="llm_circuit_open",
                )
                notes = self._tag_with_category(chosen_category, build_template_note(vote_std))
            else:
                notes = self._apply_llm_layer(
                    event, p_anomalous, vote_std, notes, chosen_category, category_probability,
                )
        elif chosen_category is not None:
            notes = self._tag_with_category(chosen_category, notes)

        self.log_step(
            thought="Finalize decision.",
            action="finalize",
            observation=f"is_anomalous={is_anomalous}, confidence={confidence:.3f}",
        )

        return DetectionResult(
            event=event,
            is_anomalous=is_anomalous,
            confidence=confidence,
            detector_notes=notes,
            trace=self.get_trace(),
        )

    def _choose_category(self, event: TrafficEvent) -> Tuple[str, Optional[float]]:
        """Deterministic category decision — same principle as
        is_anomalous/confidence: code, not the LLM, decides. Returns
        (category, top_probability) where top_probability is the category
        model's own confidence in its raw top class (None if no category
        model is loaded). Below CATEGORY_CONFIDENCE_THRESHOLD, the top class
        isn't trusted and "Unknown" is reported instead of a possibly-wrong
        specific category — top_probability is still returned either way, so
        callers (evaluate.py) can compare the thresholded decision against
        the classifier's raw top-1 accuracy."""
        if self._category_artifact is None:
            self.log_step(
                thought="No category model loaded — reporting category as Unknown.",
                action="category_model_unavailable",
            )
            return "Unknown", None

        ranked = classifier.predict_attack_category(self._category_artifact, event.features, top_k=CATEGORY_TOP_K)
        top_category, top_probability = ranked[0]["category"], ranked[0]["probability"]
        chosen = top_category if top_probability >= self._category_confidence_threshold else "Unknown"

        self.log_step(
            thought=f"Category model top class: {top_category} (probability={top_probability:.3f}). "
                    f"{'Above' if chosen == top_category else 'Below'} the "
                    f"{self._category_confidence_threshold:.2f} confidence threshold.",
            action="category_decision",
            tool_input={
                "chosen_category": chosen, "raw_top_category": top_category,
                "raw_top_probability": top_probability, "threshold": self._category_confidence_threshold,
            },
            observation=f"top3={ranked}",
        )
        return chosen, top_probability

    @staticmethod
    def _tag_with_category(category: str, base_notes: Optional[str]) -> str:
        tag = f"[category={category}]"
        return f"{tag} {base_notes}" if base_notes else tag

    def _apply_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        deterministic_notes: Optional[str], chosen_category: str, category_probability: Optional[float],
    ) -> str:
        """Return the detector_notes text after attempting the LLM layer:
        `[category=X] explanation` (plus the deterministic borderline note,
        if any) either way — X is always `chosen_category` (code's decision,
        made before this runs); only the explanation text depends on the
        LLM succeeding. Never raises."""
        llm_result = self._run_llm_layer(event, p_anomalous, vote_std, chosen_category, category_probability)
        explanation = llm_result["explanation"] if llm_result is not None else build_template_note(vote_std)

        tag = self._tag_with_category(chosen_category, explanation)
        if deterministic_notes:
            return f"{tag} {deterministic_notes}"
        return tag

    def _run_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float],
    ) -> Optional[dict]:
        """Run the bounded LLM tool-calling loop (max MAX_LLM_TOOL_CALLS
        steps, max self._llm_timeout_seconds wall clock) and validate its
        output. Returns a validated {"explanation", "tools_used"} dict, or
        None on any failure — the caller falls back to a template note.
        Never raises: a broken LLM/MCP server must not break run().

        Runs on a plain daemon thread rather than a ThreadPoolExecutor: on
        timeout we can't force a blocking third-party call to stop, and a
        pool's shutdown() joins its worker on exit, which would silently
        turn our timeout into a wait. Abandoning a daemon thread instead
        means a hung call never blocks the caller (or process exit).

        step_buffer collects this call's log_step() calls locally instead of
        writing straight into self._trace: an abandoned (timed-out) thread
        keeps running after this method returns for the *next* event, and if
        it called self.log_step directly it would write into that next
        event's trace. Steps are only replayed into self._trace below once
        we know this call actually finished within the timeout.

        The "llm_dispatch" step below is logged directly on this (the main)
        thread, unlike "llm_layer_start" (buffered, only replayed on
        success/timely-failure) — evaluate.py counts *invocations* from
        llm_dispatch specifically, so a timed-out event still counts as
        "sent to the LLM" even though its buffered steps never get merged.
        It also records which DETECTION_LLM_MODE was used, resolved once
        here (not inside the worker) so it's visible even on a timeout."""
        mode = resolve_llm_mode()
        self.log_step(
            thought=f"Event is anomalous (category={chosen_category}) — dispatching to the "
                    f"LLM layer ({mode} mode) to explain it.",
            action="llm_dispatch",
            tool_input={"max_tool_calls": MAX_LLM_TOOL_CALLS, "timeout_seconds": self._llm_timeout_seconds,
                        "mode": mode},
        )

        outcome: dict = {}
        step_buffer: List[dict] = []

        def worker():
            try:
                outcome["raw_output"] = self._call_llm_agent(
                    event, p_anomalous, vote_std, chosen_category, category_probability, step_buffer, mode,
                )
            except Exception as exc:  # noqa: BLE001 - captured for the main thread to log
                outcome["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=self._llm_timeout_seconds)

        if thread.is_alive():
            self.log_step(
                thought="LLM layer exceeded its time budget.",
                action="llm_timeout",
                observation=f"timeout_seconds={self._llm_timeout_seconds}",
            )
            self._abandon_current_connection()
            self._record_llm_failure("timeout")
            return None

        if "error" in outcome:
            exc = outcome["error"]
            provider_reason = _classify_llm_error(exc)
            action = {
                "rate_limited": "llm_rate_limited",
                "provider_unavailable": "llm_provider_unavailable",
                None: "llm_error",
            }[provider_reason]
            self.log_step(
                thought="LLM layer raised an exception"
                        + (" — MCP connection assumed dead." if provider_reason is None else
                           " — provider-side, not the MCP connection."),
                action=action,
                observation=f"{type(exc).__name__}: {exc}",
            )
            if provider_reason is None:
                # Not a recognized provider error (rate limit / 5xx) — could well be the MCP
                # connection itself (e.g. a broken pipe), so don't risk reusing it.
                self._drop_and_disconnect_connection()
            self._record_llm_failure(provider_reason or "exception")
            return None

        # Finished within the timeout: safe to replay this call's steps into the real trace.
        for step in step_buffer:
            self.log_step(**step)

        raw_output = outcome["raw_output"]
        known_feature_names = self._artifact["feature_names"]
        grounded_feature_names = [
            f["name"] for f in classifier.top_features(self._artifact, event.features, k=GROUNDING_TOP_K)
        ]
        validated, reason = parse_and_validate(raw_output, known_feature_names, grounded_feature_names)

        # Salvage path: ToolCallingAgent only reaches this point with clean text when the
        # model correctly called final_answer. If it instead exhausted max_steps without ever
        # calling a tool (e.g. it just typed its answer as a chat message), smolagents itself
        # falls back to one direct, un-tooled generation and returns that raw text — which is
        # often the right JSON with some stray prose around it, not garbage. Re-run the exact
        # same guardrails (parse_and_validate) on just the extracted object before giving up.
        salvaged = False
        if validated is None and reason == REASON_INVALID_JSON:
            extracted = extract_json_object(raw_output)
            if extracted is not None:
                validated, reason = parse_and_validate(extracted, known_feature_names, grounded_feature_names)
                salvaged = validated is not None

        if validated is None:
            action = {
                "invalid_json": "llm_invalid_json",
                "ungrounded": "llm_ungrounded",
            }[reason]
            self.log_step(
                thought="LLM output failed schema/grounding validation.",
                action=action,
                observation=str(raw_output)[:RAW_OUTPUT_LOG_CHARS],
            )
            self._record_llm_failure(reason)
            return None

        self.log_step(
            thought="LLM output passed schema and grounding validation."
                    + (" (salvaged: extracted a JSON object from surrounding text)" if salvaged else ""),
            action="llm_validation_salvaged" if salvaged else "llm_validation_passed",
            observation=f"tools_used={validated['tools_used']}",
        )
        self._record_llm_success()
        return validated

    def _record_llm_failure(self, reason: str) -> None:
        self._consecutive_llm_failures += 1
        if self._consecutive_llm_failures >= self._circuit_breaker_threshold and not self._circuit_open:
            self._circuit_open = True
            self.log_step(
                thought=f"{self._consecutive_llm_failures} consecutive LLM failures — disabling "
                        "the LLM layer for the rest of this run instead of continuing to retry.",
                action="llm_circuit_breaker_tripped",
                observation=f"last_failure_reason={reason}",
            )

    def _record_llm_success(self) -> None:
        self._consecutive_llm_failures = 0

    def _call_llm_agent(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float], step_buffer: List[dict], mode: str,
    ) -> str:
        """Register this event's features under a short event_id, then run
        either the agent tool-calling loop or a single direct LLM call,
        depending on `mode` (see DEFAULT_LLM_MODE). Runs on a worker thread
        (see _run_llm_layer) so its caller can enforce a wall-clock timeout
        around it. Returns raw final-answer text, unvalidated, either way —
        the caller (_run_llm_layer) doesn't need to know which mode produced
        it.

        self-cleanup: if, by the time this call finishes, the main thread has
        already abandoned the connection we started with (a timeout on this
        very call), we disconnect it ourselves — the main thread couldn't,
        since we might still have been using it."""
        client, tools_by_name, register_tool, clear_tool = self._ensure_llm_connection()
        event_id = uuid.uuid4().hex[:12]
        plain_features = {name: float(value) for name, value in event.features.items()}

        register_tool(event_id=event_id, features=plain_features)
        try:
            if mode == "single_shot":
                return self._call_llm_single_shot(
                    event_id, p_anomalous, vote_std, chosen_category, category_probability,
                    tools_by_name, step_buffer,
                )
            return self._call_llm_agent_loop(
                event_id, p_anomalous, vote_std, chosen_category, category_probability,
                tools_by_name, step_buffer,
            )
        finally:
            try:
                clear_tool(event_id=event_id)
            except Exception:  # noqa: BLE001 - best-effort; a dead connection has nothing to clear
                pass
            if client is not self._mcp_client:
                self._disconnect_client(client)

    def _call_llm_agent_loop(
        self, event_id: str, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float],
        tools_by_name: dict, step_buffer: List[dict],
    ) -> str:
        """DETECTION_LLM_MODE=agent (default): build a fresh smolagents
        ToolCallingAgent over a subset of the persistent MCP connection's
        tools, and run the bounded reasoning loop. The task prompt carries
        only event_id, the already-computed p_anomalous/tree_vote_std, and
        the already-chosen category+probability — never the raw feature
        dict — since that's most of a local model's prompt-processing time
        on ~78 features.

        Tool subset: top_features and predict_attack_category always;
        tree_vote_spread only when the event is borderline (vote_std is not
        None) — offering a tool with nothing meaningful for it to report
        just tempts a small model into a wasted step. register_event/
        clear_event/predict_proba_anomalous are never offered (see
        LLM_HIDDEN_TOOL_NAMES); calling predict_attack_category here is for
        the LLM's own investigation only — it never changes chosen_category.

        A new ToolCallingAgent per call is cheap (no I/O) and keeps one
        event's tool-call history from leaking into the next's — only the
        underlying MCP connection (the expensive part) persists."""
        from smolagents import ToolCallingAgent

        tool_names = ["top_features", "predict_attack_category"] + (
            ["tree_vote_spread"] if vote_std is not None else []
        )
        tools = [tools_by_name[name] for name in tool_names if name in tools_by_name]

        step_buffer.append({
            "thought": f"p_anomalous={p_anomalous:.3f}, category={chosen_category} — invoke LLM "
                       "layer to explain the already-chosen category.",
            "action": "llm_layer_start",
            "tool_input": {"max_tool_calls": MAX_LLM_TOOL_CALLS, "timeout_seconds": self._llm_timeout_seconds,
                           "tools": tool_names},
        })

        agent = ToolCallingAgent(tools=tools, model=self._get_llm_model(), max_steps=MAX_LLM_TOOL_CALLS,
                                  instructions=SYSTEM_PROMPT)
        task = (
            f"event_id={event_id}. Baseline classifier already computed p_anomalous={p_anomalous:.3f}"
            + (f", tree_vote_std={vote_std:.3f}." if vote_std is not None else ".")
            + f" Code has already determined this event's category: {chosen_category}"
            + (f" (probability={category_probability:.3f})." if category_probability is not None else ".")
            + " Investigate with the available tools, then call final_answer with your JSON "
              "answer (explanation and tools_used only — do not state a category)."
        )

        def on_retry(attempt_number, delay, reason):
            step_buffer.append({
                "thought": f"LLM call attempt {attempt_number} failed ({reason}); "
                           f"retrying in {delay:.0f}s.",
                "action": "llm_retry",
                "observation": reason,
            })

        # Retries stay inside this call's overall wall-clock budget — _run_llm_layer's
        # thread.join(timeout=...) bounds this whole method, retries and backoff included.
        raw_output = _call_with_retry(lambda: agent.run(task), on_retry=on_retry)

        self._log_agent_memory_diagnostics(agent, raw_output, step_buffer)

        usage = agent.monitor.get_total_token_counts()
        step_buffer.append({
            "action": "llm_token_usage",
            "tool_input": {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens},
        })
        step_buffer.append({"action": "llm_layer_end", "observation": str(raw_output)[:RAW_OUTPUT_LOG_CHARS]})
        return raw_output

    def _call_llm_single_shot(
        self, event_id: str, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float],
        tools_by_name: dict, step_buffer: List[dict],
    ) -> str:
        """DETECTION_LLM_MODE=single_shot: code calls top_features (and
        tree_vote_spread, if borderline) directly via the MCP connection —
        no tool-calling loop, no LLM decision about which tools to use —
        then makes exactly one LLM call with the results (plus the already-
        chosen category+probability) embedded in the prompt and
        `response_format` set to ANSWER_JSON_SCHEMA. litellm forwards a
        json_schema response_format to Ollama as its `format` parameter,
        which Ollama enforces with grammar-constrained decoding — unlike
        tool_choice, which litellm's own ollama transformation drops
        outright ("causes ollama requests to hang"), so this is the reliable
        structured-output path for small/local models that the agent loop's
        tool-call requirement can't be forced onto. predict_attack_category
        is not called here at all — code already has chosen_category from
        its own direct classifier call before this method runs.

        Returns the raw JSON text, same contract as _call_llm_agent_loop —
        it still goes through the normal parse_and_validate/salvage pipeline
        in _run_llm_layer, so no guardrail is bypassed by this mode."""
        import litellm

        step_buffer.append({
            "thought": f"p_anomalous={p_anomalous:.3f}, category={chosen_category} — single_shot "
                       "mode: code calls top_features (and tree_vote_spread, if borderline) "
                       "directly, then one LLM call with a JSON-schema response format.",
            "action": "llm_layer_start",
            "tool_input": {"timeout_seconds": self._llm_timeout_seconds, "mode": "single_shot"},
        })

        features_summary = self._unwrap_mcp_result(tools_by_name["top_features"](event_id=event_id, k=GROUNDING_TOP_K))
        vote_summary = None
        if vote_std is not None and "tree_vote_spread" in tools_by_name:
            vote_summary = self._unwrap_mcp_result(tools_by_name["tree_vote_spread"](event_id=event_id))

        user_message = (
            f"Baseline classifier already computed p_anomalous={p_anomalous:.3f}"
            + (f", tree_vote_std={vote_std:.3f}." if vote_std is not None else ".")
            + f" This event's category has already been determined: {chosen_category}"
            + (f" (probability={category_probability:.3f})." if category_probability is not None else ".")
            + f" Most relevant features (top_features result): {features_summary}."
            + (f" tree_vote_spread result: {vote_summary}." if vote_summary is not None else "")
        )

        model_id = resolve_llm_model_id()
        api_key = os.environ.get("DETECTION_LLM_API_KEY")

        def attempt():
            return litellm.completion(
                model=model_id, api_key=api_key,
                messages=[
                    {"role": "system", "content": SINGLE_SHOT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "detection_answer",
                                                                          "schema": ANSWER_JSON_SCHEMA}},
                max_tokens=MAX_LLM_OUTPUT_TOKENS,
            )

        def on_retry(attempt_number, delay, reason):
            step_buffer.append({
                "thought": f"LLM call attempt {attempt_number} failed ({reason}); "
                           f"retrying in {delay:.0f}s.",
                "action": "llm_retry",
                "observation": reason,
            })

        response = _call_with_retry(attempt, on_retry=on_retry)
        raw_output = response.choices[0].message.content

        step_buffer.append({
            "action": "llm_token_usage",
            "tool_input": {"prompt_tokens": response.usage.prompt_tokens,
                           "completion_tokens": response.usage.completion_tokens},
        })
        step_buffer.append({"action": "llm_layer_end", "observation": str(raw_output)[:RAW_OUTPUT_LOG_CHARS]})
        return raw_output

    @staticmethod
    def _unwrap_mcp_result(value):
        """smolagents' MCPClient (structured_output=True) wraps a non-object
        tool return (e.g. top_features' list) as {"result": ...}; a tool
        returning an object (e.g. tree_vote_spread's {vote_fraction,
        vote_std}) comes back as-is. Normalizes either shape to the actual
        payload for embedding in the single_shot prompt."""
        if isinstance(value, dict) and set(value.keys()) == {"result"}:
            return value["result"]
        return value

    @staticmethod
    def _log_agent_memory_diagnostics(agent, raw_output: str, step_buffer: List[dict]) -> None:
        """Log the raw text of every step that failed to parse as a tool
        call, plus whether the run only produced an answer because
        smolagents exhausted max_steps and fell back to one direct,
        un-tooled generation (`_handle_max_steps_reached` in
        smolagents/agents.py) — this is exactly the "model output does not
        contain any JSON blob" failure mode small local models hit: it
        wastes steps replying in plain text instead of calling a tool, then
        that final coerced answer is what parse_and_validate actually sees.
        Without this, all we could see was the final raw_output; this makes
        the earlier failed attempts visible in the trace/CSV too. Best-
        effort: this is diagnostic logging, never allowed to break the main
        flow (e.g. in tests where ToolCallingAgent is mocked and .memory
        isn't a real Memory object), so any failure here is swallowed
        silently."""
        try:
            from smolagents.memory import ActionStep

            action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]

            for step in action_steps:
                if step.error is not None:
                    step_buffer.append({
                        "action": "llm_step_failed",
                        "tool_input": {"step_number": step.step_number},
                        "observation": f"error={step.error}; raw_text="
                                       f"{str(step.model_output)[:RAW_OUTPUT_LOG_CHARS]}",
                    })

            if action_steps and not any(step.is_final_answer for step in action_steps):
                step_buffer.append({
                    "action": "llm_forced_final_answer",
                    "observation": str(raw_output)[:RAW_OUTPUT_LOG_CHARS],
                })
        except Exception:  # noqa: BLE001 - diagnostics are best-effort, never fatal
            pass

    def _ensure_llm_connection(self):
        """Open the MCP client (spawns the tool-server subprocess, which
        loads the model artifact) on first use, and reuse it across events.
        Returns (client, tools_by_name, register_tool, clear_tool) — tools_by_name
        maps name -> Tool for everything the LLM agent may call (register_event/
        clear_event/predict_proba_anomalous excluded, see LLM_HIDDEN_TOOL_NAMES);
        _call_llm_agent picks the subset to actually hand to ToolCallingAgent.
        Called from the worker thread started by _run_llm_layer (or from
        warmup(), before any timed loop starts)."""
        if self._mcp_client is None:
            from mcp import StdioServerParameters
            from smolagents import MCPClient

            server_params = StdioServerParameters(command=sys.executable, args=["-m", MCP_SERVER_MODULE])
            client = MCPClient(server_params, structured_output=True)
            all_tools = client.get_tools()

            self._register_event_tool = next(t for t in all_tools if t.name == "register_event")
            self._clear_event_tool = next(t for t in all_tools if t.name == "clear_event")
            self._llm_tools_by_name = {t.name: t for t in all_tools if t.name not in LLM_HIDDEN_TOOL_NAMES}
            self._mcp_client = client

        return self._mcp_client, self._llm_tools_by_name, self._register_event_tool, self._clear_event_tool

    def _abandon_current_connection(self) -> None:
        """After a timeout: stop referencing the current connection without
        disconnecting it here — the timed-out worker thread may still be
        using it, and it isn't safe to touch from this (the main) thread.
        Queue it for close() to clean up, and let the worker thread's own
        finally block disconnect it as soon as it notices (via _call_llm_agent)
        that it's no longer self._mcp_client. This is also what avoids
        reconnecting on every following event: only the connection changes
        (to None, forcing a respawn on next use); repeated timeouts still
        count toward the circuit breaker, which stops that respawn-then-
        timeout-again cascade after a few consecutive failures."""
        if self._mcp_client is not None:
            self._abandoned_mcp_clients.append(self._mcp_client)
        self._mcp_client = None
        self._llm_tools_by_name = None
        self._register_event_tool = None
        self._clear_event_tool = None

    def _drop_and_disconnect_connection(self) -> None:
        """After an exception: the worker thread has already finished (it's
        not abandoned/still-running), so it's safe to disconnect directly
        from here."""
        client, self._mcp_client = self._mcp_client, None
        self._llm_tools_by_name = None
        self._register_event_tool = None
        self._clear_event_tool = None
        if client is not None:
            self._disconnect_client(client)

    def _get_llm_model(self):
        """DETECTION_LLM_MODEL picks the LiteLLM model id (default: a local
        Ollama model). DETECTION_LLM_API_KEY is only needed for a hosted
        model (e.g. Gemini) — see .env.example. Output length is capped
        (MAX_LLM_OUTPUT_TOKENS) since generation time dominates on a slow
        local model. Built once and cached.

        retry=False: smolagents' own Model has a built-in retryer for what it
        recognizes as rate-limit errors (`is_rate_limit_error` — a substring
        match on the message, so it also fires on some non-429 phrasing),
        with defaults RETRY_MAX_ATTEMPTS=3 and RETRY_WAIT=60s exponential
        (i.e. up to ~60s + ~120s of internal sleeping before it ever raises).
        That's larger than our own default llm_timeout_seconds (20s), so with
        it left on, a 429 would blow our per-event timeout while still
        inside smolagents' own retry loop — we'd record "timeout" and never
        get a chance to run _call_with_retry's faster (2s/4s/8s) backoff at
        all. It also doesn't retry 5xx/ServiceUnavailable (only rate-limit-
        shaped messages), so it wasn't helping there either. We disable it
        so _call_with_retry (bounded by the per-event timeout, see
        _run_llm_layer) is the single, predictable retry authority for both
        rate limits and provider-unavailable errors."""
        if self._llm_model is None:
            from smolagents import LiteLLMModel

            model_id = resolve_llm_model_id()
            api_key = os.environ.get("DETECTION_LLM_API_KEY")
            self._llm_model = LiteLLMModel(
                model_id=model_id, api_key=api_key, max_tokens=MAX_LLM_OUTPUT_TOKENS, retry=False,
            )
        return self._llm_model
