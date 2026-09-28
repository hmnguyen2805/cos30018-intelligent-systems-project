from src.correlation.subagent import CorrelationSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, MitigationRecommendation

ACTION_BY_TECHNIQUE = {
    "T1110": "Lock account after repeated failed logins; alert analyst.",
    "T1110.001": "Lock account after repeated failed logins; alert analyst.",
    "T1566": "Quarantine the message/attachment; warn the user.",
    "T1190": "Patch/isolate the vulnerable service; block source IP.",
    "T1041": "Block outbound C2 traffic; isolate host.",
    "T1567": "Block the destination web service; isolate host.",
}


class MitigationManager(BaseAgent):
    name = "mitigation_manager"

    def __init__(self):
        super().__init__()
        self._subagent = CorrelationSubagent()

    def run(self, input_data: DetectionResult) -> MitigationRecommendation:
        self._trace = []
        self.log_step(thought="Delegate to Correlation Subagent.", action="delegate_to_subagent")

        correlation = self._subagent.run(input_data)

        if correlation.matched_technique_ids:
            technique = correlation.matched_technique_ids[0]
            action = ACTION_BY_TECHNIQUE.get(technique, "Escalate — unrecognised technique.")
        else:
            technique = None
            action = "No confident technique match — escalate for manual review."

        self.log_step(
            thought=f"Reason over correlation result (technique={technique}).",
            action="decide_action",
            observation=action,
        )

        return MitigationRecommendation(
            correlation=correlation,
            proposed_action=action,
            confidence=correlation.confidence,
            trace=self.get_trace(),
        )