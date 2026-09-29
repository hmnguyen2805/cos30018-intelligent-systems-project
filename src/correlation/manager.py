from typing import Optional

from src.correlation.subagent import CorrelationSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, MitigationRecommendation

# Every technique in the catalog needs an action here (tests/test_mitigation.py checks this).
ACTION_BY_TECHNIQUE = {
    "T1110": "Lock account after repeated failed logins; alert analyst.",
    "T1110.001": "Lock account after repeated failed logins; alert analyst.",
    "T1566": "Quarantine the message/attachment; warn the user.",
    "T1190": "Patch/isolate the vulnerable service; block source IP.",
    "T1041": "Block outbound C2 traffic; isolate host.",
    "T1567": "Block the destination web service; isolate host.",
    "T1498": "Rate-limit or block the flooding sources upstream; alert analyst.",
    "T1499": "Block or rate-limit the source; protect the targeted service.",
    "T1046": "Block the scanning source IP; watch for follow-up attacks.",
}

NO_THREAT_ACTION = "No action: detection says the traffic is benign."
NO_MATCH_ACTION = "No confident technique match: escalate for manual review."
UNKNOWN_TECHNIQUE_ACTION = "Escalate: unrecognised technique."


class MitigationManager(BaseAgent):
    name = "mitigation_manager"

    def __init__(self, subagent: Optional[CorrelationSubagent] = None):
        super().__init__()
        self._subagent = subagent or CorrelationSubagent()

    def run(self, input_data: DetectionResult) -> MitigationRecommendation:
        self._trace = []
        self.log_step(thought="Delegate to Correlation Subagent.", action="delegate_to_subagent")

        correlation = self._subagent.run(input_data)

        if correlation.matched_technique_ids:
            # The subagent lists the most likely technique first.
            technique = correlation.matched_technique_ids[0]
            action = ACTION_BY_TECHNIQUE.get(technique, UNKNOWN_TECHNIQUE_ACTION)
        else:
            technique = None
            action = NO_MATCH_ACTION if input_data.is_anomalous else NO_THREAT_ACTION

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
