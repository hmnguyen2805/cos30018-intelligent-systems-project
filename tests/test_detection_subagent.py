"""
Unit tests for DetectionSubagent — the borderline-handling loop that makes
Detection an agent rather than a bare classifier call. classifier.py's
functions are monkeypatched so no trained model artifact is needed.
"""
from unittest.mock import patch

from src.detection.subagent import DetectionSubagent
from src.shared.schemas import TrafficEvent


def make_event():
    return TrafficEvent(features={"duration": 1.0, "packet_count": 2.0})


@patch("src.detection.classifier.load_artifact", return_value={"model": None, "feature_names": []})
def make_agent(_mock_load):
    return DetectionSubagent(model_path="unused")


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_anomalous_score_skips_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.95
    agent = make_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95
    mock_votes.assert_not_called()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_benign_score_skips_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.05
    agent = make_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is False
    assert result.confidence == 0.95
    mock_votes.assert_not_called()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_borderline_score_triggers_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.5  # inside [0.4, 0.6]
    mock_votes.return_value = (0.7, 0.05)  # trees agree, low std

    result = make_agent().run(make_event())

    mock_votes.assert_called_once()
    assert result.is_anomalous is True
    assert result.confidence == 0.7
    assert "trust" in result.detector_notes.lower()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_borderline_score_with_high_tree_disagreement_is_flagged(mock_predict, mock_votes):
    mock_predict.return_value = 0.5
    mock_votes.return_value = (0.55, 0.3)  # high std, disagreement

    result = make_agent().run(make_event())

    assert "disagreement" in result.detector_notes.lower()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_run_produces_a_trace_of_its_steps(mock_predict, mock_votes):
    mock_predict.return_value = 0.9
    result = make_agent().run(make_event())

    assert len(result.trace) >= 2
    assert result.trace[0].action == "call_classifier"
