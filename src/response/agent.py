"""
Judge Agent: arbitrates between the Detection Manager's and the Mitigation
Manager's conclusions and produces the pipeline's final ResponseRecommendation.

    JudgeAgent.run(JudgeInput) -> ResponseRecommendation

Three modes with the same input and output, so they can be compared in the
evaluation (assignment section 6 requires a simpler baseline):

    "rules"        Decision table only (rules.py), no LLM.
                   Baseline: a fixed workflow without adaptive coordination.
    "single_shot"  Code gathers the evidence with the Judge's tools, then ONE
                   LLM call decides. Baseline: a single LLM call.
    "agent"        The LLM Judge agent. An iterative loop: each turn the LLM
                   picks the next action (a tool from tools.py, or a final
                   decision), code runs it and feeds the result back, until the
                   LLM finalizes or escalates, or the step limit is reached.

Only the agent can send a case back to a manager ("request_recheck"): the
pipeline puts a recheck callback in JudgeInput, the Judge calls it with the
manager and a reason, and the loop carries on with the managers' updated
conclusions. Each manager can be rechecked at most once per case, enforced
here in code. The rules and single-shot baselines never recheck.

In both LLM modes the rule decision is still computed and serves as
    - the fallback when the LLM errors, times out, gives invalid replies
      twice in a row, or runs out of steps; and
    - the guardrail: the LLM may only be more cautious than the rules
      (see guardrails.py).

Every step is logged with log_step, so the trace shows exactly what the Judge
did, and llm_usage records calls, tokens, latency and iterations for the
evaluation metrics.
"""
import json
import logging
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from src.response import guardrails, llm, rules, tools
from src.response.guardrails import Verdict
from src.shared.base import BaseAgent
from src.shared.schemas import JudgeInput, RecheckRecord, ResponseRecommendation

logger = logging.getLogger(__name__)

MODE_RULES = "rules"
MODE_SINGLE_SHOT = "single_shot"
MODE_AGENT = "agent"
MODES = (MODE_RULES, MODE_SINGLE_SHOT, MODE_AGENT)

MAX_CONSECUTIVE_INVALID = 2  # invalid LLM replies in a row before falling back to the rules
OBSERVATION_LOG_CHARS = 500


def _clip(text: Any) -> str:
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    return text if len(text) <= OBSERVATION_LOG_CHARS else text[:OBSERVATION_LOG_CHARS] + "..."


class LLMFallback(Exception):
    """The LLM path couldn't produce a usable verdict; the rules decide instead."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class JudgeAgent(BaseAgent):
    name = "judge"

    def __init__(self, mode: str = MODE_RULES, llm_client: Optional[llm.LLMClient] = None,
                 config: Optional[llm.LLMConfig] = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown Judge mode {mode!r}; expected one of {MODES}.")
        self.mode = mode
        self.config = config or llm.LLMConfig.from_env()
        self._llm_client = llm_client
        self._usage: Dict[str, Any] = {}
        self._ctx: Optional[tools.JudgeContext] = None   # latest case evidence (replaced after a recheck)
        self._rechecks: List[RecheckRecord] = []

    @property
    def llm_client(self) -> llm.LLMClient:
        if self._llm_client is None:  # created on first use, so rules mode never needs litellm
            self._llm_client = llm.LiteLLMClient(self.config)
        return self._llm_client

    def run(self, input_data: JudgeInput) -> ResponseRecommendation:
        if input_data is None or input_data.detection is None:
            raise ValueError("JudgeAgent needs a JudgeInput with a DetectionResult.")

        self._trace = []
        self._usage = {
            "mode": self.mode, "model": None if self.mode == MODE_RULES else self.config.model,
            "llm_calls": 0, "iterations": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "llm_latency_ms": 0.0, "invalid_replies": 0, "recovered_replies": 0, "rechecks": 0,
        }
        self._rechecks = []
        self._ctx = tools.JudgeContext(input_data, allow_rechecks=self.mode == MODE_AGENT)

        if self.mode == MODE_RULES:
            rule_decision = rules.decide(self._ctx.comparison)
            final = self._run_rules(self._ctx, rule_decision)
        else:
            final, rule_decision = self._run_llm()
        case_input = self._ctx.judge_input  # after any recheck, the managers' latest conclusions

        self.log_step(
            thought="Escalate to a human analyst." if final.escalate else "Finalize the response.",
            action="escalate_to_human" if final.escalate else "finalize",
            observation=f"recommended_action={final.action}",
        )
        logger.info("Judge decision: mode=%s case=%s action=%s escalate=%s decided_by=%s",
                    self.mode, rule_decision.case, final.action, final.escalate, final.decided_by)

        return ResponseRecommendation(
            detection=case_input.detection,
            mitigation=case_input.mitigation,
            recommended_action=final.action,
            agents_agree=rule_decision.agents_agree,
            escalated_to_human=final.escalate,
            case=rule_decision.case,
            reasoning=final.reasoning,
            trace=self.get_trace(),
            decided_by=final.decided_by,
            llm_usage={} if self.mode == MODE_RULES else dict(self._usage),
            rechecks=list(self._rechecks),
        )

    # --- rules mode ------------------------------------------------------------

    def _run_rules(self, ctx: tools.JudgeContext, decision: rules.Decision) -> guardrails.FinalDecision:
        self.log_step(
            thought="Compare the Detection Manager's and Mitigation Manager's conclusions.",
            action="compare_conclusions",
            tool_input={"mitigation_available": ctx.comparison.mitigation_available},
            observation=ctx.comparison.summary(),
        )
        self.log_step(
            thought=decision.reasoning,
            action="apply_decision_rule",
            tool_input={"case": decision.case},
            observation=f"action={decision.action}, escalate={decision.escalate}",
        )
        return guardrails.from_rules(decision)

    # --- LLM modes -------------------------------------------------------------

    def _run_llm(self) -> Tuple[guardrails.FinalDecision, rules.Decision]:
        """Returns the final decision and the rule decision it was checked against.
        The rule decision is computed from the LATEST evidence: if the agent
        rechecked a manager, the rules judge the updated conclusions too."""
        try:
            verdict = self._run_agent_loop() if self.mode == MODE_AGENT else self._run_single_shot(self._ctx)
        except LLMFallback as fallback:
            rule_decision = rules.decide(self._ctx.comparison)
            self._usage["fallback_reason"] = fallback.reason
            self.log_step(
                thought=f"LLM path failed ({fallback.reason}); the rule table decides.",
                action="fallback_to_rules",
                tool_input={"case": rule_decision.case},
                observation=f"action={rule_decision.action}, escalate={rule_decision.escalate}",
            )
            return guardrails.from_rules(rule_decision, guardrails.DECIDED_BY_FALLBACK,
                                         note=f"LLM unavailable ({fallback.reason}); rule-based decision:"), rule_decision

        rule_decision = rules.decide(self._ctx.comparison)
        final = guardrails.apply_guardrails(verdict, rule_decision)
        if final.decided_by == guardrails.DECIDED_BY_OVERRIDE:
            self.log_step(
                thought=final.reasoning,
                action="guardrail_override",
                tool_input={"case": rule_decision.case},
                observation=f"action={final.action}, escalate={final.escalate}",
            )
        return final, rule_decision

    def _run_agent_loop(self) -> Verdict:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": llm.agent_system_prompt(bool(self._ctx.rechecks_available))},
            {"role": "user", "content": llm.AGENT_TASK_MESSAGE},
        ]
        consecutive_invalid = 0
        tools_called = set()

        for _ in range(self.config.max_steps):
            ctx = self._ctx  # replaced after a recheck
            recheck_managers = ctx.rechecks_available
            self._usage["iterations"] += 1
            response = self._call_llm(messages, llm.agent_step_schema(bool(recheck_managers)), "judge_step")
            messages.append({"role": "assistant", "content": response.content})

            try:
                step = llm.parse_agent_step(response.content, ctx.allowed_final_actions, recheck_managers)
                self._check_preconditions(step, tools_called)
            except llm.LLMOutputError as err:
                consecutive_invalid += 1
                self._usage["invalid_replies"] += 1
                self.log_step(thought="LLM reply rejected.", action="invalid_llm_reply",
                              tool_input={"reason": err.reason}, observation=_clip(response.content))
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    raise LLMFallback(f"invalid_reply:{err.reason}")
                messages.append(llm.invalid_reply_message(str(err), bool(recheck_managers)))
                continue

            if consecutive_invalid:
                self._usage["recovered_replies"] += 1
            consecutive_invalid = 0

            if step.action in (llm.FINALIZE, llm.ESCALATE):
                self.log_step(thought=step.thought, action=f"llm_{step.action}",
                              tool_input={"final_action": step.final_action} if step.final_action else None,
                              observation=step.reasoning)
                return Verdict(escalate=step.action == llm.ESCALATE, action=step.final_action,
                               reasoning=step.reasoning or "")

            if step.action == llm.REQUEST_RECHECK:
                result = self._recheck(step.manager, step.recheck_reason)
                # The evidence has changed: the old consistency check no longer counts.
                # The observation includes the updated conclusions, so that counts as compared.
                tools_called = {"compare_conclusions"}
                self.log_step(thought=step.thought, action=llm.REQUEST_RECHECK,
                              tool_input={"manager": step.manager, "reason": step.recheck_reason},
                              observation=_clip(result))
                messages.append(llm.observation_message(llm.REQUEST_RECHECK, result))
                continue

            kwargs = {"technique_id": step.technique_id} if step.technique_id else {}
            result = tools.run_tool(step.action, ctx, **kwargs)
            tools_called.add(step.action)
            self.log_step(thought=step.thought, action=step.action, tool_input=kwargs or None,
                          observation=_clip(result))
            messages.append(llm.observation_message(step.action, result))

        raise LLMFallback("max_steps")

    def _recheck(self, manager: str, reason: str) -> dict:
        """Send the case back to one manager through the pipeline's callback,
        then carry on with the updated conclusions. Never raises: a failed
        recheck is reported to the LLM and the earlier conclusions stay."""
        ctx = self._ctx
        record = RecheckRecord(manager=manager, reason=reason, before=ctx.comparison.summary())
        start = time.perf_counter()
        try:
            new_input = ctx.judge_input.recheck(manager, reason)
        except Exception as exc:  # the manager crashed during its second look
            record.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Recheck of %s failed: %s", manager, record.error)
            new_input = ctx.judge_input
        record.duration_ms = (time.perf_counter() - start) * 1000

        # Once per manager per case, whatever the callback returns.
        remaining = [m for m in new_input.rechecks_available if m != manager]
        new_input = replace(new_input, rechecks_available=remaining)
        self._ctx = tools.JudgeContext(new_input, allow_rechecks=True)
        record.after = self._ctx.comparison.summary()
        self._rechecks.append(record)
        self._usage["rechecks"] += 1

        result: Dict[str, Any] = {"manager": manager, "recheck_done": record.error is None}
        if record.error:
            result["error"] = f"The recheck failed ({record.error}); the earlier conclusions still stand."
        result["updated_conclusions"] = tools.compare_conclusions(self._ctx)
        return result

    @staticmethod
    def _check_preconditions(step: llm.AgentStep, tools_called: set) -> None:
        """Evidence the agent must have gathered before it may finalize.

        Enforced in code, not left to the prompt: a 3B model skipped the
        consistency check in one Ollama run and finalized a DoS-vs-brute-force
        mismatch. Escalating is always allowed without preconditions."""
        if step.action != llm.FINALIZE:
            return
        if "compare_conclusions" not in tools_called:
            raise llm.LLMOutputError("no_evidence", "call compare_conclusions before finalizing")
        if step.final_action != rules.NO_ACTION and "check_category_consistency" not in tools_called:
            raise llm.LLMOutputError(
                "consistency_not_checked",
                "call check_category_consistency before finalizing with a response action")

    def _run_single_shot(self, ctx: tools.JudgeContext) -> Verdict:
        evidence: Dict[str, Any] = {"conclusions": tools.run_tool("compare_conclusions", ctx)}
        self.log_step(thought="Gather the evidence in code before the single LLM call.",
                      action="compare_conclusions", observation=_clip(evidence["conclusions"]))

        if ctx.comparison.technique_ids:
            evidence["playbook"] = {t: tools.run_tool("lookup_playbook", ctx, technique_id=t)
                                    for t in ctx.comparison.technique_ids}
            self.log_step(action="lookup_playbook", tool_input={"technique_ids": list(ctx.comparison.technique_ids)},
                          observation=_clip(evidence["playbook"]))

        evidence["category_consistency"] = tools.run_tool("check_category_consistency", ctx)
        self.log_step(action="check_category_consistency", observation=_clip(evidence["category_consistency"]))

        messages = [
            {"role": "system", "content": llm.SINGLE_SHOT_SYSTEM_PROMPT},
            llm.single_shot_user_message(evidence),
        ]
        self._usage["iterations"] = 1
        response = self._call_llm(messages, llm.SINGLE_SHOT_SCHEMA, "judge_decision")
        try:
            step = llm.parse_single_shot(response.content, ctx.allowed_final_actions)
        except llm.LLMOutputError as err:
            self._usage["invalid_replies"] += 1
            self.log_step(thought="LLM reply rejected.", action="invalid_llm_reply",
                          tool_input={"reason": err.reason}, observation=_clip(response.content))
            raise LLMFallback(f"invalid_reply:{err.reason}")

        self.log_step(thought=step.thought, action=f"llm_{step.action}",
                      tool_input={"final_action": step.final_action} if step.final_action else None,
                      observation=step.reasoning)
        return Verdict(escalate=step.action == llm.ESCALATE, action=step.final_action,
                       reasoning=step.reasoning or "")

    def _call_llm(self, messages: List[Dict[str, str]], schema: dict, schema_name: str) -> llm.LLMResponse:
        try:
            response = self.llm_client(messages, schema, schema_name)
        except Exception as exc:  # timeouts, connection errors, provider errors
            self.log_step(thought="LLM call failed.", action="llm_error",
                          observation=_clip(f"{type(exc).__name__}: {exc}"))
            raise LLMFallback(f"llm_error:{type(exc).__name__}") from exc
        self._usage["llm_calls"] += 1
        self._usage["prompt_tokens"] += response.prompt_tokens
        self._usage["completion_tokens"] += response.completion_tokens
        self._usage["llm_latency_ms"] += response.latency_ms
        return response


def judge_from_env() -> JudgeAgent:
    """Judge configured from the environment: JUDGE_MODE (rules | single_shot | agent,
    default rules) plus the JUDGE_LLM_* settings documented in llm.py."""
    return JudgeAgent(mode=os.environ.get("JUDGE_MODE", MODE_RULES))
