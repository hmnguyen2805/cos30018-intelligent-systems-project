"""
Detection Subagent — decides whether a TrafficEvent looks anomalous.

The baseline RandomForest (classifier.py) is a tool the agent calls, not the
agent itself. On a borderline confidence score, the agent re-examines via
per-tree vote spread (a second, finer-grained tool call) before finalizing —
that borderline-handling loop is what makes this an agent rather than a
single classifier call.

Optionally (`use_llm=True`), a bounded LLM tool-calling loop investigates
borderline/anomalous events over MCP (mcp_server.py) and proposes an attack
category + explanation for detector_notes. The LLM never sets or edits
is_anomalous or confidence — those are always the deterministic values
computed below from classifier.py's outputs — and any invalid, ungrounded,
timed-out, or errored LLM output falls back to a template note. See
README.md's guardrails table for the full list.

The MCP connection (a subprocess that loads the model artifact) is opened
lazily on the first event that needs it and reused across events — spawning
it per event would reload the artifact every time. Call close() (or use the
subagent as a context manager) to shut it down; if it dies mid-run, the
current event falls back to a template note and the next one that needs the
LLM reconnects.

Owned by the Detection Manager (manager.py), which delegates each event here.
"""
import os
import sys
import threading
from typing import Optional

from dotenv import load_dotenv

from src.detection import classifier
from src.detection.llm_notes import SYSTEM_PROMPT, build_template_note, parse_and_validate
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent

load_dotenv()

BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6
DISAGREEMENT_THRESHOLD = 0.15  # tree-vote std above this = low ensemble consensus

LLM_TRIGGER_THRESHOLD = BORDERLINE_LOW  # only invoke the LLM at p_anomalous >= this
MAX_LLM_TOOL_CALLS = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LLM_MODEL = "ollama_chat/qwen2.5:7b"
GROUNDING_TOP_K = 5  # feature count used both to ground the LLM and to check its explanation

MCP_SERVER_MODULE = "src.detection.mcp_server"


class DetectionSubagent(BaseAgent):
    name = "detection_subagent"

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_llm: bool = False,
        llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ):
        super().__init__()
        self._artifact = classifier.load_artifact(model_path or classifier.DEFAULT_MODEL_PATH)
        self.use_llm = use_llm
        self._llm_timeout_seconds = llm_timeout_seconds

        # Persistent MCP connection state — lazily opened by _ensure_llm_connection(),
        # torn down by close(). None means "not connected right now".
        self._mcp_client = None
        self._llm_tools = None
        self._llm_model = None

    def __enter__(self) -> "DetectionSubagent":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the persistent MCP connection, if one is open. Safe to
        call multiple times, or when the LLM layer was never used."""
        if self._mcp_client is None:
            return
        try:
            self._mcp_client.disconnect()
        except Exception as exc:  # noqa: BLE001 - closing must never raise
            self.log_step(
                thought="Error while closing the MCP connection.",
                action="llm_close_error",
                observation=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._mcp_client = None
            self._llm_tools = None

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
        # outputs. Nothing below this line may change either value; the LLM (if it runs) only
        # affects detector_notes.
        is_anomalous = final_p >= 0.5
        confidence = final_p if is_anomalous else 1.0 - final_p

        if self.use_llm and p_anomalous >= LLM_TRIGGER_THRESHOLD:
            notes = self._apply_llm_layer(event, p_anomalous, vote_std, notes)

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

    def _apply_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        deterministic_notes: Optional[str],
    ) -> str:
        """Return the detector_notes text after attempting the LLM layer:
        `[category=X] explanation` (plus the deterministic borderline note,
        if any) on success, or a template note on any failure. Never raises."""
        llm_result = self._run_llm_layer(event, p_anomalous, vote_std)
        if llm_result is None:
            return build_template_note(vote_std)

        tag = f"[category={llm_result['category']}] {llm_result['explanation']}"
        if deterministic_notes:
            return f"{tag} {deterministic_notes}"
        return tag

    def _run_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float]
    ) -> Optional[dict]:
        """Run the bounded LLM tool-calling loop (max MAX_LLM_TOOL_CALLS
        steps, max self._llm_timeout_seconds wall clock) and validate its
        output. Returns a validated {"category", "explanation", "tools_used"}
        dict, or None on any failure — the caller falls back to a template
        note. Never raises: a broken LLM/MCP server must not break run().

        Runs on a plain daemon thread rather than a ThreadPoolExecutor: on
        timeout we can't force a blocking third-party call to stop, and a
        pool's shutdown() joins its worker on exit, which would silently
        turn our timeout into a wait. Abandoning a daemon thread instead
        means a hung call never blocks the caller (or process exit) — see
        _reset_llm_connection for why we also drop the connection then."""
        outcome: dict = {}

        def worker():
            try:
                outcome["raw_output"] = self._call_llm_agent(event, p_anomalous, vote_std)
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
            self._reset_llm_connection()
            return None

        if "error" in outcome:
            exc = outcome["error"]
            self.log_step(
                thought="LLM layer raised an exception — MCP connection assumed dead.",
                action="llm_error",
                observation=f"{type(exc).__name__}: {exc}",
            )
            self._reset_llm_connection()
            return None

        raw_output = outcome["raw_output"]
        known_feature_names = self._artifact["feature_names"]
        grounded_feature_names = [
            f["name"] for f in classifier.top_features(self._artifact, event.features, k=GROUNDING_TOP_K)
        ]
        validated = parse_and_validate(raw_output, known_feature_names, grounded_feature_names)

        if validated is None:
            self.log_step(
                thought="LLM output failed schema/category/grounding validation.",
                action="llm_validation_failed",
                observation=str(raw_output)[:300],
            )
            return None

        self.log_step(
            thought="LLM output passed schema and grounding validation.",
            action="llm_validation_passed",
            observation=f"category={validated['category']}, tools_used={validated['tools_used']}",
        )
        return validated

    def _call_llm_agent(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float]
    ) -> str:
        """Build a fresh smolagents ToolCallingAgent over the persistent MCP
        connection's tools, and run the bounded reasoning loop. Runs on a
        worker thread (see _run_llm_layer) so its caller can enforce a
        wall-clock timeout around it. Returns the agent's raw final-answer
        text, unvalidated. A new ToolCallingAgent per call is cheap (no I/O)
        and keeps one event's tool-call history from leaking into the next's
        — only the underlying MCP connection (the expensive part) persists."""
        from smolagents import ToolCallingAgent

        tools = self._ensure_llm_connection()

        self.log_step(
            thought=f"p_anomalous={p_anomalous:.3f} is borderline/anomalous — invoke LLM layer "
                    "to investigate and propose a category + explanation.",
            action="llm_layer_start",
            tool_input={"max_tool_calls": MAX_LLM_TOOL_CALLS, "timeout_seconds": self._llm_timeout_seconds},
        )

        agent = ToolCallingAgent(tools=tools, model=self._get_llm_model(), max_steps=MAX_LLM_TOOL_CALLS,
                                  instructions=SYSTEM_PROMPT)
        task = (
            f"Event features: {event.features}. "
            f"Baseline classifier p_anomalous={p_anomalous:.3f}"
            + (f", tree_vote_std={vote_std:.3f}." if vote_std is not None else ".")
            + " Investigate with the available tools, then output only the required JSON."
        )
        raw_output = agent.run(task)

        self.log_step(action="llm_layer_end", observation=str(raw_output)[:300])
        return raw_output

    def _ensure_llm_connection(self) -> list:
        """Open the MCP client (spawns the tool-server subprocess, which
        loads the model artifact) on first use, and reuse it across events.
        Called from the worker thread started by _run_llm_layer."""
        if self._mcp_client is None:
            from mcp import StdioServerParameters
            from smolagents import MCPClient

            server_params = StdioServerParameters(command=sys.executable, args=["-m", MCP_SERVER_MODULE])
            self._mcp_client = MCPClient(server_params, structured_output=True)
            self._llm_tools = self._mcp_client.get_tools()
        return self._llm_tools

    def _reset_llm_connection(self) -> None:
        """Drop the current MCP connection after a timeout or error, so the
        next event that needs the LLM opens a fresh one. We don't call
        disconnect() here: the failure may have come from a still-running
        abandoned thread (timeout) or a subprocess that already died
        (error), so touching that same client again isn't safe — we just
        stop referencing it and let it get cleaned up on its own."""
        self._mcp_client = None
        self._llm_tools = None

    def _get_llm_model(self):
        """DETECTION_LLM_MODEL picks the LiteLLM model id (default: a local
        Ollama model). DETECTION_LLM_API_KEY is only needed for a hosted
        model (e.g. Gemini) — see .env.example. Built once and cached."""
        if self._llm_model is None:
            from smolagents import LiteLLMModel

            model_id = os.environ.get("DETECTION_LLM_MODEL", DEFAULT_LLM_MODEL)
            api_key = os.environ.get("DETECTION_LLM_API_KEY")
            self._llm_model = LiteLLMModel(model_id=model_id, api_key=api_key)
        return self._llm_model
