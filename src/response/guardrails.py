"""
Guardrails applied to the LLM Judge's verdict before it becomes the final
result.

The rule table (rules.decide) is always computed alongside the LLM. The LLM
may only be *more cautious* than the rules, never less:

    LLM escalates                          -> escalate            (decided_by = "llm")
    rules escalate, LLM wants to finalize  -> escalate            (decided_by = "guardrail_override")
    both finalize with the same action     -> that action         (decided_by = "llm")
    both finalize, different actions       -> the rules' action   (decided_by = "guardrail_override")

So the LLM can catch problems the rules can't see (e.g. Detection's attack
category contradicting the matched technique) and escalate them, but it can
never trigger an automated action the rules didn't sanction.
"""
from dataclasses import dataclass
from typing import Optional

from src.response import rules

DECIDED_BY_RULES = "rules"
DECIDED_BY_LLM = "llm"
DECIDED_BY_OVERRIDE = "guardrail_override"
DECIDED_BY_FALLBACK = "rules_fallback"


@dataclass(frozen=True)
class Verdict:
    """What the LLM concluded."""
    escalate: bool
    action: Optional[str]
    reasoning: str


@dataclass(frozen=True)
class FinalDecision:
    action: str
    escalate: bool
    reasoning: str
    decided_by: str


def from_rules(decision: rules.Decision, decided_by: str = DECIDED_BY_RULES, note: str = "") -> FinalDecision:
    reasoning = f"{note} {decision.reasoning}".strip()
    return FinalDecision(decision.action, decision.escalate, reasoning, decided_by)


def apply_guardrails(verdict: Verdict, rule_decision: rules.Decision) -> FinalDecision:
    if verdict.escalate:
        return FinalDecision(rules.ESCALATE, True, verdict.reasoning, DECIDED_BY_LLM)

    if rule_decision.escalate:
        return from_rules(
            rule_decision, DECIDED_BY_OVERRIDE,
            note=f"Guardrail: the LLM wanted to finalize with {verdict.action!r} "
                 f"({verdict.reasoning}), but the rules require escalation.",
        )

    if verdict.action == rule_decision.action:
        return FinalDecision(verdict.action, False, verdict.reasoning, DECIDED_BY_LLM)

    return from_rules(
        rule_decision, DECIDED_BY_OVERRIDE,
        note=f"Guardrail: the LLM chose {verdict.action!r} ({verdict.reasoning}), "
             f"but the rules only sanction {rule_decision.action!r}.",
    )
