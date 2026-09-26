"""
Unit tests for DetectionSubagent's core flow (borderline handling, tracing)
and the deterministic category decision. classifier.py's functions are
monkeypatched so no trained model artifact is needed.

Split out of the old test_detection_subagent.py — see llm/test_layer.py
(connection/timeout/retry/circuit-breaker) and llm/test_modes.py (agent vs
single_shot, salvage) for the rest.
"""
from unittest.mock import MagicMock, patch

from src.detection.subagent import DetectionSubagent
from tests.detection.conftest import make_agent, make_event


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


# --- _choose_category: the deterministic category decision -------------------
# Same principle as is_anomalous/confidence: code, not the LLM, decides the
# category. These tests configure a real category artifact (unlike the rest of
# this file, which always falls back to "Unknown" with no artifact loaded).

def _make_category_artifact(classes_and_probas):
    """A fake category_model whose predict_proba always returns the given
    (class, probability) pairs, in that order, regardless of input."""
    classes = [c for c, _ in classes_and_probas]
    probas = [p for _, p in classes_and_probas]
    model = MagicMock()
    model.classes_ = classes
    model.predict_proba.return_value = [probas]
    return {"category_model": model, "classes": classes, "feature_names": ["duration", "packet_count"]}


@patch("src.detection.classifier.load_artifact")
def make_agent_with_category_model(classes_and_probas, mock_load, **kwargs):
    # patch() appends the mock as the last positional arg (after ones the
    # caller passes explicitly), so classes_and_probas must come first here.
    binary_artifact = {"model": None, "feature_names": ["duration", "packet_count"]}
    category_artifact = _make_category_artifact(classes_and_probas)
    mock_load.side_effect = [binary_artifact, category_artifact]
    return DetectionSubagent(model_path="unused", **kwargs)


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_uses_top_class_when_above_threshold(mock_predict):
    mock_predict.return_value = 0.95
    # Explicit threshold: this test's story (a probability clearly above the threshold is
    # trusted) shouldn't depend on whatever DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD happens to be.
    agent = make_agent_with_category_model(
        [("DDoS", 0.9), ("PortScan", 0.1)], category_confidence_threshold=0.6,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_falls_back_to_unknown_below_threshold(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("DDoS", 0.55), ("PortScan", 0.45)], category_confidence_threshold=0.6,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=Unknown]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_threshold_is_configurable(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("DDoS", 0.55), ("PortScan", 0.45)], category_confidence_threshold=0.5,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"  # 0.55 clears a lowered 0.5 threshold


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_at_exactly_the_threshold_is_accepted(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("DDoS", 0.6), ("PortScan", 0.4)], category_confidence_threshold=0.6,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_event_gets_no_category_tag_even_with_a_category_model_loaded(mock_predict):
    mock_predict.return_value = 0.05  # clearly benign
    agent = make_agent_with_category_model([("DDoS", 0.9), ("PortScan", 0.1)])

    result = agent.run(make_event())

    assert result.is_anomalous is False
    assert result.detector_notes is None


@patch("src.detection.classifier.predict_proba_anomalous")
def test_category_decision_trace_step_records_chosen_and_raw_top_and_threshold(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("DDoS", 0.55), ("PortScan", 0.45)], category_confidence_threshold=0.6,
    )

    result = agent.run(make_event())

    step = next(s for s in result.trace if s.action == "category_decision")
    assert step.tool_input == {
        "chosen_category": "Unknown",
        "raw_top_category": "DDoS",
        "raw_top_probability": 0.55,
        "threshold": 0.6,
    }


@patch("src.detection.classifier.predict_proba_anomalous")
def test_category_model_unavailable_is_logged_as_its_own_trace_step(mock_predict):
    # make_agent() (no category artifact) hits the FileNotFoundError branch of
    # _load_category_artifact, so _choose_category takes the "no model" path.
    mock_predict.return_value = 0.95
    result = make_agent().run(make_event())

    assert any(s.action == "category_model_unavailable" for s in result.trace)
    assert not any(s.action == "category_decision" for s in result.trace)


# --- Benign class / binary-category model disagreement -----------------------
# train_category.py now trains an explicit "Benign" class alongside the attack
# categories (see its module docstring): when it's the category model's top
# vote for an event the BINARY model already called anomalous, that's a
# disagreement between the two models — reported as "Unknown" (never
# "Benign", which would contradict is_anomalous=True) and logged as its own
# "model_disagreement" trace step, distinct from an ordinary below-threshold
# "Unknown" (see test_choose_category_falls_back_to_unknown_below_threshold).

@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_reports_unknown_when_category_model_votes_benign(mock_predict):
    mock_predict.return_value = 0.95  # binary model: anomalous
    agent = make_agent_with_category_model([("Benign", 0.99), ("DDoS", 0.01)])

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.detector_notes == "[category=Unknown]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_top_vote_is_logged_as_model_disagreement_not_category_decision(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("Benign", 0.99), ("DDoS", 0.01)], category_confidence_threshold=0.6,
    )

    result = agent.run(make_event())

    disagreement_steps = [s for s in result.trace if s.action == "model_disagreement"]
    assert len(disagreement_steps) == 1
    assert disagreement_steps[0].tool_input == {
        "chosen_category": "Unknown", "raw_top_category": "Benign",
        "raw_top_probability": 0.99, "threshold": 0.6,
    }
    assert not any(s.action == "category_decision" for s in result.trace)


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_top_vote_disagreement_is_reported_even_above_the_confidence_threshold(mock_predict):
    # A high-confidence Benign vote is still a disagreement, not a trustworthy category —
    # the confidence threshold only governs which ATTACK category to trust, it never makes
    # "Benign" an acceptable answer for an event the binary model called anomalous.
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("Benign", 0.99), ("DDoS", 0.01)], category_confidence_threshold=0.1,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=Unknown]"
    assert any(s.action == "model_disagreement" for s in result.trace)


# --- assert_feature_names_match is enforced at construction time -------------

@patch(
    "src.detection.classifier.load_artifact",
    side_effect=[
        {"model": None, "feature_names": ["duration", "packet_count"]},
        {"category_model": MagicMock(), "classes": ["DDoS"], "feature_names": ["duration"]},
    ],
)
def test_mismatched_feature_names_between_artifacts_raises_on_construction(_mock_load):
    try:
        DetectionSubagent(model_path="unused")
        assert False, "expected a ValueError for mismatched feature_names"
    except ValueError as exc:
        assert "feature_names" in str(exc) or "feature set" in str(exc).lower()
