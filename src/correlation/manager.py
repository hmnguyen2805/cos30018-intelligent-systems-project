"""
Mitigation Manager: turns Detection's result into a MITRE ATT&CK technique and a
proposed response action.

    MitigationManager.run(DetectionResult, recheck_reason=None) -> MitigationRecommendation

Two modes with the same input and output:

    "rules"  (default) Delegate to the Correlation Subagent: benign skip, category
             lookup, or embedding search. No LLM.
    "agent"  An LLM agent loop (agent_loop.py) chooses its own steps with read-only
             tools. Code enforces the rules: only real techniques, evidence before
             choosing, the action text comes from ACTION_BY_TECHNIQUE, and the
             confidence comes from the evidence. Falls back to the rules path on an
             LLM error, a timeout, 2 invalid replies in a row, or the step limit.

Benign traffic never reaches the LLM in either mode.
"""
import os
from typing import List, Optional

from src.correlation import agent_loop
from src.correlation.subagent import CorrelationSubagent
from src.response import llm
from src.shared.base import BaseAgent
from src.shared.schemas import CorrelationResult, DetectionResult, MitigationRecommendation

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

MODE_RULES = "rules"
MODE_AGENT = "agent"
MODES = (MODE_RULES, MODE_AGENT)


class MitigationManager(BaseAgent):
    name = "mitigation_manager"

    def __init__(self, subagent: Optional[CorrelationSubagent] = None, mode: str = MODE_RULES,
                 llm_client: Optional[llm.LLMClient] = None, config: Optional[llm.LLMConfig] = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown Mitigation mode {mode!r}; expected one of {MODES}.")
        self.mode = mode
        self._subagent = subagent or CorrelationSubagent()
        self._config = config
        self._llm_client = llm_client
        self.last_usage: dict = {}   # LLM calls/tokens/latency of the latest agent run (evaluation)
        self._last_choice = None     # (detection, [technique acted on]) of the latest run, so a recheck can name the first answer

    @property
    def config(self) -> llm.LLMConfig:
        if self._config is None:  # read the MITIGATION_LLM_* variables only when the agent needs them
            self._config = agent_loop.mitigation_llm_config()
        return self._config

    @property
    def llm_client(self) -> llm.LLMClient:
        if self._llm_client is None:  # created on first use, so rules mode never needs litellm
            self._llm_client = llm.LiteLLMClient(self.config)
        return self._llm_client

    def run(self, input_data: DetectionResult, recheck_reason: Optional[str] = None) -> MitigationRecommendation:
        """Decide a technique and action for one DetectionResult (see _decide). The manager
        remembers what it chose, so a recheck of the same detection can tell the LLM what the
        first pass answered."""
        previous_ids = None
        if recheck_reason and self._last_choice and self._last_choice[0] is input_data:
            previous_ids = self._last_choice[1] or None
        result = self._decide(input_data, recheck_reason, previous_ids)
        # Only the first technique is acted on (its action is the one proposed); the rest are just candidates.
        self._last_choice = (input_data, list(result.correlation.matched_technique_ids)[:1])
        return result

    def _decide(self, input_data: DetectionResult, recheck_reason: Optional[str],
                previous_ids: Optional[List[str]]) -> MitigationRecommendation:
        """recheck_reason is set when the Judge sends the case back ("look
        again, because ..."). It is recorded in the trace in both modes. In agent
        mode it is also put in the LLM's first message, together with an instruction
        to try something different from the first pass."""
        self._trace = []
        self.last_usage = {}
        if recheck_reason is not None:
            self.log_step(
                thought=("Judge requested a recheck. The agent is told the reason and asked to try something different."
                         if self.mode == MODE_AGENT and input_data.is_anomalous else
                         "Judge requested a recheck. Recorded; the analysis below runs as normal for now."),
                action="recheck",
                tool_input={"reason": recheck_reason},
            )

        if self.mode == MODE_AGENT and input_data.is_anomalous:  # benign traffic skips the LLM
            try:
                return self._run_agent(input_data, recheck_reason, previous_ids)
            except agent_loop.MitigationFallback as fallback:
                self.last_usage = {"fallback_reason": fallback.reason, **self.last_usage}
                self.log_step(
                    thought=f"LLM path failed ({fallback.reason}); the rules path decides.",
                    action="fallback_to_rules",
                    tool_input={"reason": fallback.reason},
                )
        return self._run_rules(input_data)

    # --- rules mode ------------------------------------------------------------

    def _run_rules(self, input_data: DetectionResult) -> MitigationRecommendation:
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

    # --- agent mode ------------------------------------------------------------

    def _run_agent(self, input_data: DetectionResult, recheck_reason: Optional[str],
                   previous_ids: Optional[List[str]] = None) -> MitigationRecommendation:
        loop = agent_loop.MitigationAgentLoop(
            detection=input_data, subagent=self._subagent, llm_client=self.llm_client,
            config=self.config, log_step=self.log_step, recheck_reason=recheck_reason,
            previous_ids=previous_ids,
        )
        try:
            outcome = loop.run()
        finally:
            self.last_usage = dict(loop.usage)  # kept even when the loop falls back

        if outcome.technique_ids:
            technique = outcome.technique_ids[0]
            # The LLM never writes the action: it comes from the fixed table.
            action = ACTION_BY_TECHNIQUE.get(technique, UNKNOWN_TECHNIQUE_ACTION)
            notes = (f"LLM agent chose {outcome.technique_ids} (confidence {outcome.confidence:.2f}, "
                     f"from {outcome.source}): {outcome.reasoning}")
        else:
            action = NO_MATCH_ACTION
            notes = f"LLM agent found no matching technique: {outcome.reasoning}"

        self.log_step(
            thought=f"Reason over the agent's result (technique={outcome.technique_ids[0] if outcome.technique_ids else None}).",
            action="decide_action",
            observation=action,
        )

        correlation = CorrelationResult(
            detection=input_data,
            matched_technique_ids=list(outcome.technique_ids),
            confidence=outcome.confidence,
            correlation_notes=notes,
            trace=[],
        )
        return MitigationRecommendation(
            correlation=correlation,
            proposed_action=action,
            confidence=outcome.confidence,
            trace=self.get_trace(),
        )


def mitigation_from_env() -> MitigationManager:
    """Mitigation Manager configured from the environment: MITIGATION_MODE
    (rules | agent, default rules) plus the MITIGATION_LLM_* settings documented
    in agent_loop.py."""
    return MitigationManager(mode=os.environ.get("MITIGATION_MODE", MODE_RULES))