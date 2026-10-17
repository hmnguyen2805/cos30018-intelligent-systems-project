"""
Tests for the Judge's LLM modes (src/response/agent.py, llm.py, tools.py,
guardrails.py). The LLM is replaced by ScriptedLLM, which returns pre-written
replies in order, so these run without Ollama or any network access.
"""
import copy
import json

import pytest

from src.response import guardrails, rules, tools
from src.response.agent import MODE_AGENT, MODE_RULES, MODE_SINGLE_SHOT, JudgeAgent, judge_from_env
from src.response.llm import LLMConfig, LLMOutputError, LLMResponse, parse_agent_step
from src.shared.schemas import DetectionResult, JudgeInput
from tests.fakes import make_event, make_mitigation


class ScriptedLLM:
    """Stands in for the model: returns scripted replies (dicts are JSON-encoded,
    exceptions are raised) and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, schema, schema_name):
        self.calls.append({"messages": copy.deepcopy(messages), "schema_name": schema_name})
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(content=content, prompt_tokens=100, completion_tokens=20, latency_ms=50.0)


def det(anomalous=True, confidence=0.9, notes=None):
    return DetectionResult(event=make_event(), is_anomalous=anomalous, confidence=confidence,
                           detector_notes=notes)


def case(notes="[category=BruteForce] explanation", confidence=0.9, techniques=("T1110",), mit_conf=0.85):
    d = det(confidence=confidence, notes=notes)
    return JudgeInput(detection=d, mitigation=make_mitigation(d, techniques, mit_conf, "block_source_ip"))


def judge(mode, llm, max_steps=6):
    return JudgeAgent(mode=mode, llm_client=llm, config=LLMConfig(max_steps=max_steps))


def step(action, **extra):
    return {"thought": f"next: {action}", "action": action, **extra}


COMPARE = step("compare_conclusions")
CHECK = step("check_category_consistency")
FINALIZE_BLOCK = step("finalize", final_action="block_source_ip", reasoning="Consistent, confident brute force.")


def actions(result):
    return [s.action for s in result.trace]


# --- tools -----------------------------------------------------------------------

def test_parse_category():
    assert tools.parse_category("[category=DoS] Borderline call") == "DoS"
    assert tools.parse_category("no tag here") is None
    assert tools.parse_category(None) is None


def test_lookup_playbook_known_and_unknown():
    ctx = tools.JudgeContext(case())
    assert tools.lookup_playbook(ctx, "T1110")["found"] is True
    assert tools.lookup_playbook(ctx, "T9999")["found"] is False


def test_lookup_playbook_defaults_to_matched_technique():
    ctx = tools.JudgeContext(case())
    result = tools.lookup_playbook(ctx)
    assert result["technique_id"] == "T1110" and "note" in result
    no_match = tools.JudgeContext(case(techniques=()))
    assert "error" in tools.lookup_playbook(no_match)


def test_compare_conclusions_spells_out_verdict_and_confidence():
    facts = tools.compare_conclusions(tools.JudgeContext(case(mit_conf=0.41)))
    assert facts["detection_verdict"] == "anomalous" and facts["detection_confident"] is True
    assert facts["mitigation_confident"] is False
    benign = tools.compare_conclusions(tools.JudgeContext(JudgeInput(detection=det(anomalous=False, confidence=0.97))))
    assert benign["detection_verdict"] == "benign"


def test_irrelevant_mitigation_confidence_is_not_applicable_when_nothing_matched():
    d = det(anomalous=False, confidence=0.97)
    ctx = tools.JudgeContext(JudgeInput(d, make_mitigation(d, (), 0.2, "none")))
    facts = tools.compare_conclusions(ctx)
    assert facts["mitigation_confident"] is None and facts["proposed_action"] is None
    assert facts["mitigation_confidence"] is None
    assert "mitigation_note" in facts
    assert tools.check_category_consistency(ctx)["consistent"] == "not_applicable"


@pytest.mark.parametrize("notes, expected", [
    ("[category=BruteForce]", True),
    ("[category=DoS]", False),
    ("no category tag", "unknown"),
    ("[category=Unknown]", "unknown"),
])
def test_category_consistency(notes, expected):
    ctx = tools.JudgeContext(case(notes=notes))
    assert tools.check_category_consistency(ctx)["consistent"] == expected


def test_allowed_final_actions():
    assert tools.JudgeContext(case()).allowed_final_actions == ["block_source_ip", rules.NO_ACTION]
    no_mitigation = JudgeInput(detection=det(), mitigation_error="crashed")
    assert tools.JudgeContext(no_mitigation).allowed_final_actions == [rules.NO_ACTION]


# --- guardrails ------------------------------------------------------------------

def rule_decision_for(judge_input):
    return rules.decide(rules.compare_conclusions(judge_input))


def test_guardrail_rules_escalate_beats_llm_finalize():
    decision = rule_decision_for(case(confidence=0.55))  # LOW_DETECTION_CONFIDENCE
    final = guardrails.apply_guardrails(guardrails.Verdict(False, "block_source_ip", "looks fine"), decision)
    assert final.escalate and final.decided_by == guardrails.DECIDED_BY_OVERRIDE


def test_guardrail_llm_may_escalate_when_rules_finalize():
    decision = rule_decision_for(case())  # AGREED_THREAT
    final = guardrails.apply_guardrails(guardrails.Verdict(True, None, "category mismatch"), decision)
    assert final.escalate and final.decided_by == guardrails.DECIDED_BY_LLM


def test_guardrail_same_action_is_llm_decision():
    decision = rule_decision_for(case())
    final = guardrails.apply_guardrails(guardrails.Verdict(False, "block_source_ip", "ok"), decision)
    assert final.action == "block_source_ip" and final.decided_by == guardrails.DECIDED_BY_LLM


def test_guardrail_different_action_falls_back_to_rules_action():
    decision = rule_decision_for(case())
    final = guardrails.apply_guardrails(guardrails.Verdict(False, rules.NO_ACTION, "ignore it"), decision)
    assert final.action == "block_source_ip" and final.decided_by == guardrails.DECIDED_BY_OVERRIDE


# --- agent mode: the reasoning-action loop -----------------------------------------

def test_agent_loop_gathers_evidence_then_finalizes():
    llm = ScriptedLLM(COMPARE, step("check_category_consistency"),
                      step("lookup_playbook", technique_id="T1110"), FINALIZE_BLOCK)
    result = judge(MODE_AGENT, llm).run(case())

    assert result.recommended_action == "block_source_ip" and not result.escalated_to_human
    assert result.decided_by == guardrails.DECIDED_BY_LLM
    assert actions(result) == ["compare_conclusions", "check_category_consistency",
                               "lookup_playbook", "llm_finalize", "finalize"]
    usage = result.llm_usage
    assert usage["iterations"] == 4 and usage["llm_calls"] == 4
    assert usage["prompt_tokens"] == 400 and usage["completion_tokens"] == 80
    # each tool result is fed back to the model as an observation
    last_messages = llm.calls[-1]["messages"]
    assert any("Observation (lookup_playbook)" in m["content"] for m in last_messages)
    assert llm.calls[0]["schema_name"] == "judge_step"


def test_agent_escalates_category_mismatch_the_rules_would_act_on():
    llm = ScriptedLLM(COMPARE, step("check_category_consistency"),
                      step("escalate", reasoning="Detection says DoS but the matched technique is brute force."))
    result = judge(MODE_AGENT, llm).run(case(notes="[category=DoS] flood-like traffic"))

    assert result.case == rules.AGREED_THREAT  # the rules alone would have acted automatically
    assert result.escalated_to_human and result.recommended_action == rules.ESCALATE
    assert result.decided_by == guardrails.DECIDED_BY_LLM


def test_agent_cannot_finalize_what_the_rules_escalate():
    result = judge(MODE_AGENT, ScriptedLLM(COMPARE, CHECK, FINALIZE_BLOCK)).run(case(confidence=0.55))
    assert result.escalated_to_human
    assert result.decided_by == guardrails.DECIDED_BY_OVERRIDE
    assert "guardrail_override" in actions(result)


def test_agent_recovers_from_one_invalid_reply():
    llm = ScriptedLLM("this is not json", COMPARE, CHECK, FINALIZE_BLOCK)
    result = judge(MODE_AGENT, llm).run(case())

    assert result.decided_by == guardrails.DECIDED_BY_LLM
    assert result.llm_usage["invalid_replies"] == 1 and result.llm_usage["recovered_replies"] == 1
    assert "not valid" in llm.calls[1]["messages"][-1]["content"]


def test_agent_falls_back_after_two_invalid_replies():
    judge_input = case()
    result = judge(MODE_AGENT, ScriptedLLM("oops", step("dance"))).run(judge_input)
    expected = rule_decision_for(judge_input)

    assert result.decided_by == guardrails.DECIDED_BY_FALLBACK
    assert result.llm_usage["fallback_reason"].startswith("invalid_reply")
    assert (result.recommended_action, result.escalated_to_human) == (expected.action, expected.escalate)
    assert "fallback_to_rules" in actions(result)


def test_agent_falls_back_when_llm_call_fails():
    result = judge(MODE_AGENT, ScriptedLLM(TimeoutError("timed out"))).run(case())
    assert result.decided_by == guardrails.DECIDED_BY_FALLBACK
    assert result.llm_usage["fallback_reason"] == "llm_error:TimeoutError"
    assert "llm_error" in actions(result)


def test_agent_falls_back_at_step_limit():
    result = judge(MODE_AGENT, ScriptedLLM(COMPARE, COMPARE, COMPARE), max_steps=3).run(case())
    assert result.decided_by == guardrails.DECIDED_BY_FALLBACK
    assert result.llm_usage["fallback_reason"] == "max_steps"
    assert result.llm_usage["iterations"] == 3


def test_agent_must_look_at_evidence_before_finalizing():
    result = judge(MODE_AGENT, ScriptedLLM(FINALIZE_BLOCK, COMPARE, CHECK, FINALIZE_BLOCK)).run(case())
    assert result.decided_by == guardrails.DECIDED_BY_LLM
    assert result.llm_usage["invalid_replies"] == 1


def test_agent_must_check_consistency_before_acting():
    # The regression seen on Ollama: compare -> finalize, skipping the category check.
    llm = ScriptedLLM(COMPARE, FINALIZE_BLOCK, CHECK,
                      step("escalate", reasoning="Detection says DoS but the technique is brute force."))
    result = judge(MODE_AGENT, llm).run(case(notes="[category=DoS] flood-like traffic"))
    assert result.escalated_to_human and result.decided_by == guardrails.DECIDED_BY_LLM
    assert result.llm_usage["invalid_replies"] == 1 and result.llm_usage["recovered_replies"] == 1


def test_agent_may_finalize_no_action_without_consistency_check():
    d = det(anomalous=False, confidence=0.97)
    judge_input = JudgeInput(d, make_mitigation(d, (), 0.2, "none"))
    llm = ScriptedLLM(COMPARE, step("finalize", final_action="no_action", reasoning="benign, nothing matched"))
    result = judge(MODE_AGENT, llm).run(judge_input)
    assert result.recommended_action == "no_action" and result.decided_by == guardrails.DECIDED_BY_LLM


def test_agent_cannot_invent_an_action():
    llm = ScriptedLLM(COMPARE, CHECK, step("finalize", final_action="delete_all_logs", reasoning="x"), FINALIZE_BLOCK)
    result = judge(MODE_AGENT, llm).run(case())
    assert result.recommended_action == "block_source_ip"
    assert result.llm_usage["invalid_replies"] == 1


# --- single_shot mode (baseline: one LLM call) --------------------------------------

def test_single_shot_makes_one_call_with_all_evidence():
    llm = ScriptedLLM({"decision": "finalize", "final_action": "block_source_ip", "reasoning": "consistent"})
    result = judge(MODE_SINGLE_SHOT, llm).run(case())

    assert result.decided_by == guardrails.DECIDED_BY_LLM and result.llm_usage["llm_calls"] == 1
    assert actions(result) == ["compare_conclusions", "lookup_playbook", "check_category_consistency",
                               "llm_finalize", "finalize"]
    user_message = llm.calls[0]["messages"][-1]["content"]
    assert "category_consistency" in user_message and "playbook" in user_message


def test_single_shot_invalid_reply_falls_back():
    result = judge(MODE_SINGLE_SHOT, ScriptedLLM("nope")).run(case())
    assert result.decided_by == guardrails.DECIDED_BY_FALLBACK


# --- rules mode and configuration ----------------------------------------------------

def test_rules_mode_marks_decisions_and_uses_no_llm():
    result = JudgeAgent(mode=MODE_RULES).run(case())
    assert result.decided_by == guardrails.DECIDED_BY_RULES and result.llm_usage == {}


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        JudgeAgent(mode="magic")


def test_judge_from_env(monkeypatch):
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    assert judge_from_env().mode == MODE_RULES
    monkeypatch.setenv("JUDGE_MODE", MODE_AGENT)
    assert judge_from_env().mode == MODE_AGENT


def test_parse_accepts_fenced_json_and_rejects_unknown_actions():
    parsed = parse_agent_step('```json\n{"thought": "t", "action": "compare_conclusions"}\n```', [])
    assert parsed.action == "compare_conclusions"
    with pytest.raises(LLMOutputError):
        parse_agent_step('{"thought": "t", "action": "rm -rf"}', [])
