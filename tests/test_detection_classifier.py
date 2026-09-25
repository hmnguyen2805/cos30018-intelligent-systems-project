"""
Unit tests for src.detection.classifier — the tool layer the Detection
Subagent calls. Uses a tiny hand-fit RandomForest fixture so tests run
without the real CICIDS2017 dataset or a trained model artifact on disk.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.detection import classifier

FEATURE_NAMES = ["duration", "packet_count"]


def make_artifact(n_estimators=5, random_state=0):
    """A tiny RandomForest fit on separable synthetic data, wrapped as an
    artifact the same shape classifier.load_artifact would produce."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],   # class 0 (benign)
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1],  # class 1 (anomalous)
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return {"model": model, "feature_names": FEATURE_NAMES}


def test_predict_proba_anomalous_clearly_benign():
    artifact = make_artifact()
    p = classifier.predict_proba_anomalous(artifact, {"duration": 0.0, "packet_count": 0.0})
    assert p < 0.5


def test_predict_proba_anomalous_clearly_anomalous():
    artifact = make_artifact()
    p = classifier.predict_proba_anomalous(artifact, {"duration": 10.0, "packet_count": 10.0})
    assert p > 0.5


def test_predict_proba_anomalous_aligns_features_by_name_not_dict_order():
    artifact = make_artifact()
    # Same point, keys given in reversed / different order than feature_names.
    ordered = classifier.predict_proba_anomalous(artifact, {"duration": 10.0, "packet_count": 10.0})
    reordered = classifier.predict_proba_anomalous(artifact, {"packet_count": 10.0, "duration": 10.0})
    assert ordered == reordered


def test_predict_proba_anomalous_missing_feature_defaults_to_zero():
    artifact = make_artifact()
    # Omitting packet_count should behave like packet_count=0.0.
    p_missing = classifier.predict_proba_anomalous(artifact, {"duration": 0.0})
    p_explicit_zero = classifier.predict_proba_anomalous(artifact, {"duration": 0.0, "packet_count": 0.0})
    assert p_missing == p_explicit_zero


def test_tree_vote_spread_all_trees_agree_has_zero_std():
    artifact = make_artifact()
    vote_frac, vote_std = classifier.tree_vote_spread(artifact, {"duration": 10.0, "packet_count": 10.0})
    assert vote_frac == 1.0
    assert vote_std == 0.0


def test_tree_vote_spread_matches_manual_per_tree_vote_count():
    artifact = make_artifact(n_estimators=5)
    features = {"duration": 5.0, "packet_count": 5.0}  # near the boundary
    X = np.array([[features[name] for name in FEATURE_NAMES]])
    manual_votes = np.array([tree.predict(X)[0] for tree in artifact["model"].estimators_])

    vote_frac, vote_std = classifier.tree_vote_spread(artifact, features)

    assert vote_frac == pytest.approx(manual_votes.mean())
    assert vote_std == pytest.approx(manual_votes.std())


def test_top_features_returns_k_items_without_medians():
    artifact = make_artifact()
    features = {"duration": 10.0, "packet_count": 10.0}
    result = classifier.top_features(artifact, features, k=1)
    assert len(result) == 1
    assert result[0]["name"] in FEATURE_NAMES
    assert result[0]["value"] == features[result[0]["name"]]
    assert "median" not in result[0]


def test_top_features_includes_median_when_artifact_has_it():
    artifact = make_artifact()
    artifact["feature_medians"] = {"duration": 1.0, "packet_count": 2.0}
    result = classifier.top_features(artifact, {"duration": 10.0, "packet_count": 10.0}, k=2)
    assert len(result) == 2
    for entry in result:
        assert entry["median"] == artifact["feature_medians"][entry["name"]]


def test_top_features_caps_at_number_of_available_features():
    artifact = make_artifact()
    result = classifier.top_features(artifact, {"duration": 1.0, "packet_count": 1.0}, k=10)
    assert len(result) == len(FEATURE_NAMES)


def test_top_features_missing_feature_value_defaults_to_zero():
    artifact = make_artifact()
    result = classifier.top_features(artifact, {}, k=2)
    assert all(entry["value"] == 0.0 for entry in result)


def test_load_artifact_missing_path_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="train.py"):
        classifier.load_artifact(str(missing))
