"""
Unit tests for the Judge's decision table (src/response/rules.py).
One test per case, plus the agreement flag and the comparison summary.
"""
import pytest

from src.response import rules
from src.shared.schemas import JudgeInput
from tests.fakes import make_detection, make_mitigation


def decide_for(detection, mitigation=None, error=None):
    comparison = rules.compare_conclusions(
        JudgeInput(detection=detection, mitigation=mitigation, mitigation_error=error)
    )
    return rules.decide(comparison)


def test_low_detection_confidence_escalates_even_if_managers_agree():
    det = make_detection(anomalous=True, confidence=0.55)
    decision = decide_for(det, make_mitigation(det, ("T1110",), 0.9))
    assert decision.case == rules.LOW_DETECTION_CONFIDENCE
    assert decision.escalate and decision.action == rules.ESCALATE


def test_missing_mitigation_with_threat_escalates():
    decision = decide_for(make_detection(anomalous=True, confidence=0.9), error="RuntimeError: boom")
    assert decision.case == rules.MITIGATION_MISSING
    assert decision.escalate
    assert "boom" in decision.reasoning
    assert decision.agents_agree is False


def test_missing_mitigation_with_confident_benign_is_no_action_but_flagged_incomplete():
    decision = decide_for(make_detection(anomalous=False, confidence=0.95))
    assert decision.case == rules.MITIGATION_MISSING
    assert decision.action == rules.NO_ACTION and not decision.escalate
    assert "incomplete" in decision.reasoning


def test_both_benign_is_no_action_and_agreement():
    det = make_detection(anomalous=False, confidence=0.9)
    decision = decide_for(det, make_mitigation(det, (), 0.9, "none"))
    assert decision.case == rules.BOTH_BENIGN
    assert decision.action == rules.NO_ACTION
    assert decision.agents_agree is True


def test_benign_but_technique_matched_is_a_conflict():
    det = make_detection(anomalous=False, confidence=0.9)
    decision = decide_for(det, make_mitigation(det, ("T1566",), 0.9))
    assert decision.case == rules.BENIGN_BUT_ACTION
    assert decision.escalate
    assert decision.agents_agree is False


def test_anomalous_with_no_technique_escalates_as_possible_unknown_attack():
    det = make_detection(anomalous=True, confidence=0.9)
    decision = decide_for(det, make_mitigation(det, (), 0.3, "none"))
    assert decision.case == rules.ANOMALOUS_NO_MATCH
    assert decision.escalate
    assert decision.agents_agree is False


def test_low_mitigation_confidence_escalates():
    det = make_detection(anomalous=True, confidence=0.9)
    decision = decide_for(det, make_mitigation(det, ("T1190",), 0.4))
    assert decision.case == rules.LOW_MITIGATION_CONFIDENCE
    assert decision.escalate
    assert decision.agents_agree is True  # they agree on "threat", mitigation just isn't sure which


def test_agreed_confident_threat_applies_proposed_action():
    det = make_detection(anomalous=True, confidence=0.9)
    decision = decide_for(det, make_mitigation(det, ("T1110",), 0.85, "block_source_ip"))
    assert decision.case == rules.AGREED_THREAT
    assert decision.action == "block_source_ip"
    assert not decision.escalate
    assert decision.agents_agree is True


@pytest.mark.parametrize("confidence, expected_case", [
    (rules.DETECTION_MIN_CONFIDENCE - 0.01, rules.LOW_DETECTION_CONFIDENCE),
    (rules.DETECTION_MIN_CONFIDENCE, rules.AGREED_THREAT),
])
def test_detection_threshold_boundary(confidence, expected_case):
    det = make_detection(anomalous=True, confidence=confidence)
    assert decide_for(det, make_mitigation(det, ("T1110",), 0.9)).case == expected_case


def test_summary_mentions_both_sides():
    det = make_detection(anomalous=True, confidence=0.9)
    comparison = rules.compare_conclusions(
        JudgeInput(detection=det, mitigation=make_mitigation(det, ("T1110",), 0.8))
    )
    summary = comparison.summary()
    assert "anomalous" in summary and "T1110" in summary


def test_summary_when_mitigation_unavailable():
    comparison = rules.compare_conclusions(
        JudgeInput(detection=make_detection(), mitigation_error="not built yet")
    )
    assert "unavailable" in comparison.summary()
    assert "not built yet" in comparison.summary()
