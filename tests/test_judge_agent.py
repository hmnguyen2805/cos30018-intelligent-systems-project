"""
Unit tests for JudgeAgent (src/response/agent.py): output contract and trace.
"""
import pytest

from src.response import rules
from src.response.agent import JudgeAgent
from src.shared.schemas import JudgeInput, ResponseRecommendation
from tests.fakes import make_detection, make_mitigation


def test_agreed_threat_returns_full_recommendation():
    det = make_detection(anomalous=True, confidence=0.9)
    mit = make_mitigation(det, ("T1110",), 0.85, "block_source_ip")

    result = JudgeAgent().run(JudgeInput(detection=det, mitigation=mit))

    assert isinstance(result, ResponseRecommendation)
    assert result.detection is det and result.mitigation is mit
    assert result.recommended_action == "block_source_ip"
    assert result.agents_agree and not result.escalated_to_human
    assert result.case == rules.AGREED_THREAT
    assert result.reasoning


def test_trace_follows_compare_decide_finalize():
    det = make_detection(anomalous=True, confidence=0.9)
    result = JudgeAgent().run(JudgeInput(detection=det, mitigation=make_mitigation(det)))
    assert [s.action for s in result.trace] == ["compare_conclusions", "apply_decision_rule", "finalize"]
    assert [s.step_number for s in result.trace] == [1, 2, 3]


def test_escalation_ends_with_escalate_step():
    result = JudgeAgent().run(JudgeInput(detection=make_detection(), mitigation_error="crashed"))
    assert result.escalated_to_human
    assert result.recommended_action == rules.ESCALATE
    assert result.trace[-1].action == "escalate_to_human"


def test_trace_resets_between_runs():
    judge = JudgeAgent()
    det = make_detection()
    judge.run(JudgeInput(detection=det, mitigation=make_mitigation(det)))
    second = judge.run(JudgeInput(detection=det, mitigation=make_mitigation(det)))
    assert len(second.trace) == 3


def test_rejects_input_without_detection():
    with pytest.raises(ValueError):
        JudgeAgent().run(JudgeInput(detection=None))
