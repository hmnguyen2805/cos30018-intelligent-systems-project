"""
Detection Manager — owns the top-level Detection contract:

    DetectionManager.run(event: TrafficEvent) -> DetectionResult

Delegates the actual classification work to DetectionSubagent. Also the
place for any future manager-level oversight (e.g. deciding whether to
trust the subagent's result as-is, or dispatch to a second detection
subagent if one gets added).
"""
from src.detection.subagent import DetectionSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent


class DetectionManager(BaseAgent):
    name = "detection_manager"

    def __init__(self):
        super().__init__()
        self._subagent = DetectionSubagent()

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        self._trace = []
        self.log_step(thought="Delegate to Detection Subagent.", action="delegate_to_subagent")
        return self._subagent.run(input_data)
