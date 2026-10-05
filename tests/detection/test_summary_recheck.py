"""
Deterministic traffic summary (classifier.describe_flow), the final notes
format, optional DetectionResult fields, and the Judge recheck argument.
"""
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from src.detection import classifier
from src.detection.llm.layer import _recheck_clause
from src.detection.manager import DetectionManager
from src.shared import tags
from src.shared.schemas import DetectionResult, TrafficEvent
from tests.detection.conftest import make_llm_agent
from tests.detection.test_subagent import make_agent_with_category_model

ARTIFACT = {
    "feature_medians": {"Flow Duration": 5e4, "Total Fwd Packets": 3.0, "Total Backward Packets": 3.0},
    "feature_mad": {"Flow Duration": 1e4, "Total Fwd Packets": 1.0, "Total Backward Packets": 1.0},
}
SSH_FLOW = {"Destination Port": 22.0, "Flow Duration": 1.2e6, "Total Fwd Packets": 3.0,
            "Total Backward Packets": 0.0, "SYN Flag Count": 2.0, "FIN Flag Count": 0.0, "RST Flag Count": 1.0}
CLASSES = [("DoS Hulk", 0.62), ("DoS GoldenEye", 0.31), ("Benign", 0.07)]  # group DoS = 0.93
UNUSUAL = [{"name": "Flow IAT Mean", "value": 9.0, "direction": "above", "source": "top_k"},
           {"name": "Destination Port", "value": 22.0, "direction": None, "source": "context"}]


# --- describe_flow ----------------------------------------------------------------

def test_summary_names_ssh_service_and_uses_directions():
    text = classifier.describe_flow(ARTIFACT, SSH_FLOW)
    assert text.startswith("Single flow to destination port 22 (SSH):")
    assert "duration 1.20 s (above the training median)" in text
    assert "forward packets 3 (near the training median)" in text
    assert "backward packets 0 (below the training median)" in text
    assert "SYN flags 2" in text and "FIN flags 0" in text and "RST flags 1" in text
    assert text.endswith(".") and text.count(".") == 2  # one sentence ("1.20" is the other dot)


def test_summary_names_http_and_https_and_leaves_unknown_ports_bare():
    assert "port 80 (HTTP)" in classifier.describe_flow({}, {"Destination Port": 80.0})
    assert "port 443 (HTTPS)" in classifier.describe_flow({}, {"Destination Port": 443.0})
    assert classifier.describe_flow({}, {"Destination Port": 8081.0}) == "Single flow to destination port 8081."


def test_summary_omits_missing_features_and_is_none_when_nothing_is_known():
    text = classifier.describe_flow(ARTIFACT, {"Destination Port": 21.0, "Flow Duration": 500.0})
    assert "port 21 (FTP)" in text and "duration 500 µs" in text
    assert "packets" not in text and "SYN" not in text
    assert classifier.describe_flow(ARTIFACT, {"Some Other Feature": 1.0}) is None


def test_summary_without_medians_has_no_direction_text():
    assert "median" not in classifier.describe_flow({}, SSH_FLOW)


def test_summary_never_claims_other_connections_or_hosts():
    text = classifier.describe_flow(ARTIFACT, SSH_FLOW).lower()
    for word in ("connections", "source", "scan", "repeated", "multiple", "ip"):
        assert word not in text.replace("single flow", "")


@patch("src.detection.classifier.top_features", return_value=[
    {"name": "Idle Min", "value": 8.48e7, "direction": "above", "source": "top_k"},
    {"name": "Fwd IAT Std", "value": 2500.0, "direction": None, "source": "top_k"},
    {"name": "Active Mean", "value": 800.0, "direction": None, "source": "top_k"},
    {"name": "Bwd Packet Length Std", "value": 2118.23, "direction": None, "source": "top_k"},
])
def test_unusual_time_features_are_shown_with_units_not_raw_microseconds(_mock):
    text = classifier.describe_flow(ARTIFACT, SSH_FLOW, top_k=4)
    assert "Idle Min 84.80 s (above the training median)" in text
    assert "Fwd IAT Std 2.5 ms" in text
    assert "Active Mean 800 µs" in text
    assert "Bwd Packet Length Std 2118.23" in text  # non-time features stay as plain numbers
    assert "e+07" not in text


@patch("src.detection.classifier.top_features", return_value=UNUSUAL)
def test_summary_with_top_k_appends_only_top_k_entries(mock_top):
    text = classifier.describe_flow(ARTIFACT, SSH_FLOW, top_k=10)
    assert mock_top.call_args.kwargs["k"] == 10
    assert "most unusual features: Flow IAT Mean 9 µs (above the training median)" in text
    assert text.count("Destination Port") == 0  # context-source entries are not "unusual"


# --- notes format, parsed by src/shared/tags -----------------------------------------

def _agent(**kwargs):
    agent = make_agent_with_category_model(CLASSES, category_confidence_threshold=0.8, **kwargs)
    agent._artifact.update(ARTIFACT)
    return agent


@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
def test_notes_put_tags_first_then_summary_and_tags_module_parses_them(_mock):
    notes = _agent().run(TrafficEvent(features=SSH_FLOW)).detector_notes
    assert notes.startswith("[category=DoS] [label=Unknown] Single flow to destination port 22 (SSH)")
    assert tags.parse_category(notes) == "DoS"
    assert tags.strip_category_tag(notes).startswith("[label=Unknown] Single flow to destination port 22")


@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
def test_no_summary_when_the_event_has_no_context_features(_mock):
    notes = _agent().run(TrafficEvent(features={"duration": 1.0})).detector_notes
    assert notes == "[category=DoS] [label=Unknown]"


@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.05)
def test_benign_event_has_no_notes_and_no_summary(_mock):
    assert _agent().run(TrafficEvent(features=SSH_FLOW)).detector_notes is None


# --- optional DetectionResult fields -------------------------------------------------

@dataclass
class _ExtendedResult(DetectionResult):
    attack_label: Optional[str] = None
    attack_category: Optional[str] = None
    category_confidence: Optional[float] = None
    traffic_summary: Optional[str] = None


@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
def test_fields_are_populated_when_the_schema_has_them(_mock):
    with patch("src.detection.subagent.DetectionResult", _ExtendedResult):
        result = _agent().run(TrafficEvent(features=SSH_FLOW))
    assert (result.attack_label, result.attack_category) == ("Unknown", "DoS")
    assert result.category_confidence == pytest.approx(0.93)
    assert result.traffic_summary.startswith("Single flow to destination port 22 (SSH)")


@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
def test_fields_are_not_invented_when_the_schema_lacks_them(_mock):
    result = _agent().run(TrafficEvent(features=SSH_FLOW))
    for name in ("attack_label", "attack_category", "category_confidence", "traffic_summary"):
        assert not hasattr(result, name)


# --- recheck --------------------------------------------------------------------------

def _run(p, recheck_reason=None, agent=None, features=SSH_FLOW):
    agent = agent or _agent()
    with patch("src.detection.classifier.predict_proba_anomalous", return_value=p), \
         patch("src.detection.classifier.tree_vote_spread", return_value=(0.9, 0.05)) as votes, \
         patch("src.detection.classifier.top_features", return_value=UNUSUAL) as top:
        kwargs = {} if recheck_reason is None else {"recheck_reason": recheck_reason}
        return agent.run(TrafficEvent(features=features), **kwargs), votes, top


@pytest.mark.parametrize("p", [0.05, 0.3, 0.45, 0.55, 0.7, 0.95])
def test_recheck_never_changes_is_anomalous_or_confidence(p):
    normal, _, _ = _run(p)
    rechecked, _, _ = _run(p, "Mitigation disagrees")
    assert rechecked.is_anomalous == normal.is_anomalous
    assert rechecked.confidence == normal.confidence


def test_recheck_adds_evidence_to_notes_and_trace():
    result, votes, top = _run(0.95, "Mitigation saw a port scan")
    votes.assert_called_once()  # tree_vote_spread even though 0.95 is not borderline
    assert top.call_args.kwargs["k"] == 10
    notes = result.detector_notes
    assert tags.parse_category(notes) == "DoS"
    assert "Recheck evidence: Top labels:" in notes
    assert "port scan" not in notes  # the Judge's reason stays out of the notes
    assert "Top labels: DoS Hulk 0.62, DoS GoldenEye 0.31, Benign 0.07." in notes
    assert "Groups: DoS 0.93, Benign 0.07." in notes
    assert "Tree votes: fraction=0.90, std=0.05." in notes
    assert "most unusual features: Flow IAT Mean" in notes
    step = next(s for s in result.trace if s.action == "recheck")
    assert step.tool_input == {"reason": "Mitigation saw a port scan"}


def test_default_call_is_unchanged_no_recheck_evidence():
    result, votes, top = _run(0.95)
    votes.assert_not_called()
    top.assert_not_called()
    assert "Recheck" not in result.detector_notes and "most unusual" not in result.detector_notes
    assert not any(s.action == "recheck" for s in result.trace)


@pytest.mark.parametrize("p", [0.05, 0.95])
def test_recheck_reason_text_never_appears_in_detector_notes(p):
    reason = "Mitigation insists this is a port-scan [category=PortScan]"
    result, _, _ = _run(p, reason)
    notes = result.detector_notes
    assert "port-scan" not in notes and "port scan" not in notes.lower()
    assert "Mitigation" not in notes and "PortScan" not in notes
    assert tags.parse_category(notes) in ("DoS", None)
    step = next(s for s in result.trace if s.action == "recheck")  # still recorded here
    assert "port-scan" in step.tool_input["reason"]


def test_recheck_on_a_benign_event_returns_evidence_but_same_decision():
    normal, _, _ = _run(0.05)
    result, votes, _ = _run(0.05, "Judge doubts it")
    votes.assert_called_once()
    assert (result.is_anomalous, result.confidence) == (normal.is_anomalous, normal.confidence)
    assert result.detector_notes.startswith("Recheck evidence: Tree votes:")
    assert tags.parse_category(result.detector_notes) is None


def test_manager_forwards_recheck_reason_only_when_given():
    with patch("src.detection.manager.DetectionSubagent") as sub:
        manager = DetectionManager()
        event = TrafficEvent(features={})
        manager.run(event)
        sub.return_value.run.assert_called_with(event)
        manager.run(event, recheck_reason="why")
        sub.return_value.run.assert_called_with(event, recheck_reason="why")


# --- recheck + LLM ---------------------------------------------------------------------

def _valid_json():
    return json.dumps({"explanation": "Flow looks like a DoS.", "tools_used": []})


@patch("src.detection.classifier.tree_vote_spread", return_value=(0.9, 0.05))
@patch("src.detection.classifier.top_features", return_value=UNUSUAL + [{"name": "duration", "value": 1.0, "source": "context"}])
@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=_valid_json())
def test_recheck_passes_reason_to_the_llm_and_default_does_not(mock_call, _p, _t, _v):
    agent = make_llm_agent()
    agent.run(TrafficEvent(features=SSH_FLOW))
    assert "recheck_reason" not in mock_call.call_args.kwargs

    result = agent.run(TrafficEvent(features=SSH_FLOW), recheck_reason="Judge doubts it")
    assert mock_call.call_args.kwargs["recheck_reason"] == "Judge doubts it"
    assert "Flow looks like a DoS." in result.detector_notes
    assert result.detector_notes.startswith("[category=Unknown] [label=Unknown]")


@patch("src.detection.classifier.tree_vote_spread", return_value=(0.9, 0.05))
@patch("src.detection.classifier.top_features", return_value=UNUSUAL + [{"name": "duration", "value": 1.0, "source": "context"}])
@patch("src.detection.classifier.predict_proba_anomalous", return_value=0.95)
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=_valid_json())
def test_recheck_forces_the_llm_even_with_the_circuit_breaker_open(mock_call, _p, _t, _v):
    agent = make_llm_agent()
    agent._llm._circuit_open = True
    agent.run(TrafficEvent(features=SSH_FLOW))
    mock_call.assert_not_called()
    agent.run(TrafficEvent(features=SSH_FLOW), recheck_reason="Judge doubts it")
    mock_call.assert_called_once()


def test_recheck_clause_is_empty_without_a_reason_and_carries_it_otherwise():
    assert _recheck_clause(None) == ""
    assert "Judge asked for a recheck: because X" in _recheck_clause("because X")
