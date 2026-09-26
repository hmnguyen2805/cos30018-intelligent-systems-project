"""
LLM explanation layer — everything that talks to an LLM to explain an
already-decided (by code) anomalous event: the persistent MCP connection,
the agent/single_shot dispatch paths, retry/backoff, the circuit breaker,
and the validation/salvage pipeline. See docs/llm-layer.md for guardrails,
config, and troubleshooting, and docs/architecture.md for the flow diagram.

explain() returns a validated {"explanation", "tools_used"} dict, or None on
any failure; the caller (subagent.py) falls back to a template note either
way. It can never change is_anomalous, confidence, or category — see
llm.notes.ANSWER_JSON_SCHEMA, which has no fields for any of those.

The MCP connection is opened lazily and reused across events — call
warmup() once before a run and close() (or use as a context manager) when
done.
"""
import os
import sys
import threading
import time
import uuid
from typing import Callable, List, Optional

from dotenv import load_dotenv

from src.detection import classifier
from src.detection.llm.notes import (
    ANSWER_JSON_SCHEMA, REASON_INVALID_JSON, SINGLE_SHOT_SYSTEM_PROMPT, SYSTEM_PROMPT,
    extract_json_object, parse_and_validate,
)
from src.shared.schemas import TrafficEvent

load_dotenv()

MAX_LLM_TOOL_CALLS = 3
MAX_LLM_OUTPUT_TOKENS = 300  # caps generation length — the main lever on a slow local CPU model
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MODEL = "ollama_chat/qwen2.5:3b"
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 3

# "agent": ToolCallingAgent loop (default, for models with real tool_choice support).
# "single_shot": one direct call with a JSON-schema response_format instead — litellm's ollama
# transformation drops tool_choice entirely, so this is the reliable path for local Ollama
# models. resolve_llm_mode() defaults to single_shot for ollama_chat/* regardless of this
# constant. See docs/design-decisions.md (agent vs single_shot).
DEFAULT_LLM_MODE = "agent"
GROUNDING_TOP_K = 5  # feature count used both to ground the LLM and to check its explanation
RAW_OUTPUT_LOG_CHARS = 500  # truncation length for raw LLM text logged into the trace/CSV

MCP_SERVER_MODULE = "src.detection.llm.tool_server"

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

# Tools never offered to the LLM agent: register_event/clear_event are code-only, and
# predict_proba_anomalous is redundant (code already has p_anomalous). See docs/architecture.md
# (MCP server's role) for the full tool list and why each is or isn't exposed.
LLM_HIDDEN_TOOL_NAMES = {"register_event", "clear_event", "predict_proba_anomalous"}

# Backoff schedule for transient LLM-provider errors — used by warmup() and per-event calls.
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
    """DETECTION_LLM_MODE env var when recognized; otherwise single_shot for
    an ollama_chat/* model, else DEFAULT_LLM_MODE. Never raises on an
    unrecognized value — falls back to the same model-based default."""
    mode = os.environ.get("DETECTION_LLM_MODE")
    if mode in ("agent", "single_shot"):
        return mode
    return "single_shot" if resolve_llm_model_id().startswith("ollama_chat/") else DEFAULT_LLM_MODE


def _classify_llm_error(exc: BaseException) -> Optional[str]:
    """Classify as "rate_limited", "provider_unavailable", or None (not a
    recognized transient error). smolagents re-raises the real litellm/HTTP
    error as its own AgentGenerationError, so the status_code we need is in
    __cause__, not on `exc` itself — walk that chain (depth-limited,
    cycle-guarded) instead of only checking `exc`."""
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
    """Classify one exception in isolation (no __cause__/__context__ walk) by
    status_code, falling back to a class-name keyword match for
    connection-style errors that don't set one consistently."""
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
    """Call fn(), retrying with backoff on a transient error (see
    _classify_llm_error); raises immediately on a non-transient error or once
    retries are exhausted. on_retry(attempt_number, delay_seconds, reason)
    fires before each sleep, if given."""
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


class LLMExplanationLayer:
    """Owns the MCP connection, the agent/single_shot dispatch, retries, and
    the circuit breaker. Constructed once per DetectionSubagent and reused
    across events. `artifact` is the binary classifier artifact, used for
    top_features' grounding check in _run_llm_layer."""

    def __init__(
        self,
        artifact: dict,
        log_step: Callable[..., None],
        llm_timeout_seconds: Optional[float] = None,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    ):
        self._artifact = artifact
        self._log_step = log_step
        self._llm_timeout_seconds = _resolve_llm_timeout(llm_timeout_seconds)
        self._circuit_breaker_threshold = circuit_breaker_threshold

        # Persistent MCP connection state, lazily opened by _ensure_llm_connection(). Tools are
        # split so register_event/clear_event (code-only) never reach the LLM agent.
        self._mcp_client = None
        self._llm_tools_by_name = None
        self._register_event_tool = None
        self._clear_event_tool = None
        self._llm_model = None

        # Connections abandoned after a timeout — not safe to disconnect here, since a
        # background thread may still be using them; queued for close() to clean up.
        self._abandoned_mcp_clients: List = []

        # Circuit breaker: stop attempting the LLM after this many consecutive failures.
        self._consecutive_llm_failures = 0
        self._circuit_open = False

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
            self._log_step(
                thought="Error while closing an MCP connection.",
                action="llm_close_error",
                observation=f"{type(exc).__name__}: {exc}",
            )

    def warmup(self) -> dict:
        """Open the MCP connection and send one tiny prompt, outside any
        per-event timeout, so the first real event skips that cold-start
        cost. Never raises — returns {"ok", "elapsed_seconds", "reason"}."""
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
            self._log_step(
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
            self._log_step(
                thought="LLM warmup failed after retries.",
                action="llm_warmup_failed",
                observation=detail,
            )
            return {"ok": False, "elapsed_seconds": elapsed, "reason": detail}

        elapsed = time.perf_counter() - start
        self._log_step(
            thought="Warmed up the MCP connection and LLM before starting the run.",
            action="llm_warmup",
            observation=f"elapsed_seconds={elapsed:.2f}",
        )
        return {"ok": True, "elapsed_seconds": elapsed, "reason": None}

    def explain(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float],
    ) -> Optional[dict]:
        """Explain an already-decided anomalous event. Returns a validated
        {"explanation", "tools_used"} dict, or None on any failure — circuit
        open, timeout, error, invalid/ungrounded output — the caller
        (DetectionSubagent._apply_llm_layer) falls back to a template note
        either way. Never raises."""
        if self._circuit_open:
            self._log_step(
                thought="Circuit breaker is open after repeated LLM failures — skipping the "
                        "LLM layer for this event.",
                action="llm_circuit_open",
            )
            return None
        return self._run_llm_layer(event, p_anomalous, vote_std, chosen_category, category_probability)

    def _run_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        chosen_category: str, category_probability: Optional[float],
    ) -> Optional[dict]:
        """Run the bounded LLM call with a wall-clock timeout and validate
        its output. Returns a validated {"explanation", "tools_used"} dict,
        or None on any failure. Never raises.

        Uses a daemon thread, not a ThreadPoolExecutor: a pool's shutdown()
        joins its worker, which would turn our timeout into a wait; a
        daemon thread can be abandoned outright on timeout.

        step_buffer holds this call's log_step() calls locally so an
        abandoned (timed-out) thread can't write into a later event's trace
        — replayed below only once we know this call finished in time.

        "llm_dispatch" is logged directly (not buffered) so a timed-out
        event still counts as sent to the LLM even though its buffered
        steps never get merged."""
        mode = resolve_llm_mode()
        self._log_step(
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
            self._log_step(
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
            self._log_step(
                thought="LLM layer raised an exception"
                        + (" — MCP connection assumed dead." if provider_reason is None else
                           " — provider-side, not the MCP connection."),
                action=action,
                observation=f"{type(exc).__name__}: {exc}",
            )
            if provider_reason is None:
                # Not a recognized provider error — could be the MCP connection itself.
                self._drop_and_disconnect_connection()
            self._record_llm_failure(provider_reason or "exception")
            return None

        # Finished within the timeout: safe to replay this call's steps into the real trace.
        for step in step_buffer:
            self._log_step(**step)

        raw_output = outcome["raw_output"]
        known_feature_names = self._artifact["feature_names"]
        grounded_feature_names = [
            f["name"] for f in classifier.top_features(self._artifact, event.features, k=GROUNDING_TOP_K)
        ]
        validated, reason = parse_and_validate(raw_output, known_feature_names, grounded_feature_names)

        # Salvage: a model that never called final_answer still gets one un-tooled generation
        # from smolagents, often valid JSON with stray prose — re-run validation on just that.
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
            self._log_step(
                thought="LLM output failed schema/grounding validation.",
                action=action,
                observation=str(raw_output)[:RAW_OUTPUT_LOG_CHARS],
            )
            self._record_llm_failure(reason)
            return None

        self._log_step(
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
            self._log_step(
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
        """Register the event under a short event_id, dispatch to the agent
        loop or single_shot per `mode`, and return raw (unvalidated)
        final-answer text. Runs on a worker thread (see _run_llm_layer).

        If the main thread already abandoned our connection (a timeout on
        this call), we disconnect it ourselves — the main thread couldn't,
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
        """DETECTION_LLM_MODE=agent: run smolagents' ToolCallingAgent over a
        subset of the MCP tools (never the raw feature dict — see
        LLM_HIDDEN_TOOL_NAMES and docs/architecture.md for which tools and
        why). A fresh agent per call is cheap and keeps tool-call history
        from leaking between events; only the MCP connection persists."""
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

        # Retries stay inside _run_llm_layer's overall timeout — it bounds this whole method.
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
        """DETECTION_LLM_MODE=single_shot: code calls the needed MCP tools
        directly, then one litellm call with response_format=ANSWER_JSON_SCHEMA
        — the reliable structured-output path on Ollama, where tool_choice
        doesn't work (see docs/design-decisions.md). Same raw-text contract
        and validation pipeline as _call_llm_agent_loop."""
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
        """Log each step that failed to parse as a tool call, and whether
        smolagents had to fall back to an un-tooled generation after
        exhausting max_steps. See docs/llm-layer.md (Diagnostics). Best-
        effort — any failure here is swallowed, never breaks the main flow."""
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
        """Open the MCP client on first use and reuse it across events.
        Returns (client, tools_by_name, register_tool, clear_tool) —
        tools_by_name excludes LLM_HIDDEN_TOOL_NAMES."""
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
        """After a timeout: drop the connection reference without
        disconnecting — the timed-out worker thread may still be using it,
        so only it (via _call_llm_agent) or close() may touch it now."""
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
        """Build (and cache) the LiteLLMModel for DETECTION_LLM_MODEL.
        retry=False: smolagents' own retryer can sleep longer than our
        per-event timeout and doesn't handle 5xx at all — _call_with_retry
        is the single retry authority instead. See docs/llm-layer.md."""
        if self._llm_model is None:
            from smolagents import LiteLLMModel

            model_id = resolve_llm_model_id()
            api_key = os.environ.get("DETECTION_LLM_API_KEY")
            self._llm_model = LiteLLMModel(
                model_id=model_id, api_key=api_key, max_tokens=MAX_LLM_OUTPUT_TOKENS, retry=False,
            )
        return self._llm_model
