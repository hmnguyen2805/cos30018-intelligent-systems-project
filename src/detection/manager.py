"""
Detection Manager — owns the top-level Detection contract:

    DetectionManager.run(event: TrafficEvent) -> DetectionResult

Delegates the actual classification work to DetectionSubagent. Also the
place for any future manager-level oversight (e.g. deciding whether to
trust the subagent's result as-is, or dispatch to a second detection
subagent if one gets added).

With use_llm=True, the subagent holds a persistent MCP connection — call
warmup() once before a run, and close() (or use the manager as a context
manager) to shut it down when done.
"""
from typing import Optional

from src.detection.llm.layer import DEFAULT_CIRCUIT_BREAKER_THRESHOLD
from src.detection.subagent import DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD, DetectionSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent


class DetectionManager(BaseAgent):
    name = "detection_manager"

    def __init__(
        self,
        use_llm: bool = False,
        llm_timeout_seconds: Optional[float] = None,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        category_confidence_threshold: float = DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self._subagent = DetectionSubagent(
            use_llm=use_llm,
            llm_timeout_seconds=llm_timeout_seconds,
            circuit_breaker_threshold=circuit_breaker_threshold,
            category_confidence_threshold=category_confidence_threshold,
        )

    def __enter__(self) -> "DetectionManager":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the subagent's persistent MCP connection, if any."""
        self._subagent.close()

    def warmup(self) -> dict:
        """Open the MCP connection and prime the LLM before a run, outside any
        per-event timeout. Never raises; no-op success when use_llm is False."""
        return self._subagent.warmup()

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        """Delegate to DetectionSubagent and return its result."""
        self._trace = []
        self.log_step(thought="Delegate to Detection Subagent.", action="delegate_to_subagent")
        return self._subagent.run(input_data)
