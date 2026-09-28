"""
Judge Agent: arbitrates between the Detection Manager's and the Mitigation
Manager's conclusions and produces the pipeline's final ResponseRecommendation.

    JudgeAgent.run(JudgeInput) -> ResponseRecommendation

Week 7 version: fully rule-based (see rules.py). Each step is logged with
log_step so the trace already has the shape the LLM version will produce:

    compare_conclusions -> apply_decision_rule -> finalize | escalate_to_human

Week 8 replaces the middle step with a smolagents LLM loop that chooses its
own tool calls; rules.decide() stays as the fallback when that loop fails.
"""
import logging

from src.response import rules
from src.shared.base import BaseAgent
from src.shared.schemas import JudgeInput, ResponseRecommendation

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    name = "judge"

    def run(self, input_data: JudgeInput) -> ResponseRecommendation:
        if input_data is None or input_data.detection is None:
            raise ValueError("JudgeAgent needs a JudgeInput with a DetectionResult.")

        self._trace = []

        comparison = rules.compare_conclusions(input_data)
        self.log_step(
            thought="Compare the Detection Manager's and Mitigation Manager's conclusions.",
            action="compare_conclusions",
            tool_input={"mitigation_available": comparison.mitigation_available},
            observation=comparison.summary(),
        )

        decision = rules.decide(comparison)
        self.log_step(
            thought=decision.reasoning,
            action="apply_decision_rule",
            tool_input={"case": decision.case},
            observation=f"action={decision.action}, escalate={decision.escalate}",
        )

        self.log_step(
            thought="Escalate to a human analyst." if decision.escalate else "Finalize the response.",
            action="escalate_to_human" if decision.escalate else "finalize",
            observation=f"recommended_action={decision.action}",
        )

        logger.info("Judge decision: case=%s action=%s escalate=%s",
                    decision.case, decision.action, decision.escalate)

        return ResponseRecommendation(
            detection=input_data.detection,
            mitigation=input_data.mitigation,
            recommended_action=decision.action,
            agents_agree=decision.agents_agree,
            escalated_to_human=decision.escalate,
            case=decision.case,
            reasoning=decision.reasoning,
            trace=self.get_trace(),
        )
