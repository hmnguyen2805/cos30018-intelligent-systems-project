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


def test_top_features_top_k_entries_have_source_top_k():
    artifact = make_artifact()
    result = classifier.top_features(artifact, {"duration": 1.0, "packet_count": 1.0}, k=2)
    assert all(entry["source"] == "top_k" for entry in result)


def test_top_features_ranking_field_is_global_fallback_without_stats():
    artifact = make_artifact()
    result = classifier.top_features(artifact, {"duration": 1.0, "packet_count": 1.0}, k=2)
    assert all(entry["ranking"] == "global_importance_fallback" for entry in result)
    assert all("median" not in entry and "direction" not in entry for entry in result)


def test_top_features_ranking_field_is_global_fallback_when_only_mad_missing():
    # feature_medians alone (no feature_mad) is what an artifact saved before this feature
    # existed would have — still falls back to global ranking, but medians/direction still show.
    artifact = make_artifact()
    artifact["feature_medians"] = {"duration": 1.0, "packet_count": 2.0}
    result = classifier.top_features(artifact, {"duration": 10.0, "packet_count": 10.0}, k=2)
    assert all(entry["ranking"] == "global_importance_fallback" for entry in result)
    assert all("median" in entry and "direction" in entry for entry in result)


def test_top_features_ranks_by_per_event_deviation_when_medians_and_mad_present():
    """Two different events with the same fixed global feature_importances_ must get
    DIFFERENT top-1 features when their values deviate from the median differently —
    proving the ranking is per-event, not the same k features every time."""
    artifact = make_artifact()
    artifact["feature_medians"] = {"duration": 5.0, "packet_count": 5.0}
    artifact["feature_mad"] = {"duration": 1.0, "packet_count": 1.0}

    duration_is_unusual = classifier.top_features(artifact, {"duration": 50.0, "packet_count": 5.0}, k=1)
    packet_count_is_unusual = classifier.top_features(artifact, {"duration": 5.0, "packet_count": 50.0}, k=1)

    assert duration_is_unusual[0]["name"] == "duration"
    assert packet_count_is_unusual[0]["name"] == "packet_count"
    assert duration_is_unusual[0]["ranking"] == "per_event_deviation"


def test_top_features_direction_above_below_near():
    assert classifier._direction(10.0, 5.0, 1.0) == "above"
    assert classifier._direction(0.0, 5.0, 1.0) == "below"
    assert classifier._direction(5.2, 5.0, 1.0) == "near"


def test_top_features_direction_is_none_without_a_median():
    assert classifier._direction(5.0, None, 1.0) is None


def test_top_features_direction_without_scale_treats_any_nonzero_diff_as_above_below():
    assert classifier._direction(5.0, 5.0, None) == "near"
    assert classifier._direction(5.1, 5.0, None) == "above"
    assert classifier._direction(4.9, 5.0, None) == "below"


def make_artifact_with_low_importance_context_feature():
    """A "Destination Port" feature that's constant across every training row —
    RandomForest can't split on a constant, so it gets ~zero feature_importances_
    and would never make a small top-k on its own."""
    feature_names = ["duration", "packet_count", "Destination Port"]
    X = np.array([
        [0.0, 0.0, 80.0], [0.1, 0.0, 80.0], [0.0, 0.1, 80.0], [0.1, 0.1, 80.0],
        [10.0, 10.0, 80.0], [10.1, 10.0, 80.0], [10.0, 10.1, 80.0], [10.1, 10.1, 80.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, y)
    return {"model": model, "feature_names": feature_names}


def test_top_features_always_includes_context_features_even_when_not_top_k():
    artifact = make_artifact_with_low_importance_context_feature()
    features = {"duration": 10.0, "packet_count": 10.0, "Destination Port": 443.0}

    result = classifier.top_features(artifact, features, k=1)

    names = [entry["name"] for entry in result]
    assert "Destination Port" in names
    context_entry = next(entry for entry in result if entry["name"] == "Destination Port")
    assert context_entry["source"] == "context"
    assert context_entry["value"] == 443.0


def test_top_features_context_feature_not_duplicated_if_already_in_top_k():
    artifact = make_artifact_with_low_importance_context_feature()
    features = {"duration": 10.0, "packet_count": 10.0, "Destination Port": 443.0}

    result = classifier.top_features(artifact, features, k=3)  # k big enough to include everything

    names = [entry["name"] for entry in result]
    assert names.count("Destination Port") == 1


def test_top_features_context_feature_absent_from_artifact_is_skipped():
    # make_artifact()'s feature set doesn't include any CONTEXT_FEATURE_CANDIDATES name.
    artifact = make_artifact()
    result = classifier.top_features(artifact, {"duration": 1.0, "packet_count": 1.0}, k=1)
    assert len(result) == 1  # no context features to append


# --- assert_feature_names_match ----------------------------------------------

def test_assert_feature_names_match_passes_when_identical():
    binary_artifact = {"feature_names": ["a", "b", "c"]}
    category_artifact = {"feature_names": ["a", "b", "c"]}
    classifier.assert_feature_names_match(binary_artifact, category_artifact)  # must not raise


def test_assert_feature_names_match_raises_on_mismatch():
    binary_artifact = {"feature_names": ["a", "b", "c"]}
    category_artifact = {"feature_names": ["a", "b"]}
    with pytest.raises(ValueError, match="different feature sets"):
        classifier.assert_feature_names_match(binary_artifact, category_artifact)


def test_assert_feature_names_match_raises_on_reordered_names():
    # Same set, different order — a TrafficEvent's feature dict is turned into a row using
    # this order, so a reorder is just as broken as a genuinely different set.
    binary_artifact = {"feature_names": ["a", "b", "c"]}
    category_artifact = {"feature_names": ["b", "a", "c"]}
    with pytest.raises(ValueError):
        classifier.assert_feature_names_match(binary_artifact, category_artifact)


# --- predict_attack_category --------------------------------------------------

def make_category_artifact(n_estimators=5, random_state=0):
    """A tiny multiclass RandomForest, wrapped as train_category.py's
    artifact shape."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],       # PortScan-ish
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],  # DDoS-ish
        [5.0, 0.0], [5.1, 0.0],                    # BruteForce-ish
    ])
    y = np.array(["PortScan", "PortScan", "PortScan", "DDoS", "DDoS", "DDoS",
                  "BruteForce", "BruteForce"])
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return {"category_model": model, "classes": list(model.classes_), "feature_names": FEATURE_NAMES}


def test_predict_attack_category_returns_top_k_sorted_descending():
    artifact = make_category_artifact()
    result = classifier.predict_attack_category(
        artifact, {"duration": 10.0, "packet_count": 10.0}, top_k=2,
    )
    assert len(result) == 2
    assert result[0]["probability"] >= result[1]["probability"]
    assert all(0.0 <= entry["probability"] <= 1.0 for entry in result)
    assert all(isinstance(entry["category"], str) for entry in result)


def test_predict_attack_category_top_class_matches_the_closest_training_cluster():
    artifact = make_category_artifact()
    result = classifier.predict_attack_category(
        artifact, {"duration": 10.0, "packet_count": 10.0}, top_k=1,
    )
    assert result[0]["category"] == "DDoS"


def test_predict_attack_category_probabilities_sum_to_one_across_all_classes():
    artifact = make_category_artifact()
    result = classifier.predict_attack_category(
        artifact, {"duration": 5.0, "packet_count": 0.0}, top_k=len(artifact["classes"]),
    )
    assert sum(entry["probability"] for entry in result) == pytest.approx(1.0)


def test_predict_attack_category_falls_back_to_model_classes_without_classes_key():
    artifact = make_category_artifact()
    del artifact["classes"]
    result = classifier.predict_attack_category(artifact, {"duration": 10.0, "packet_count": 10.0}, top_k=1)
    assert result[0]["category"] in ["PortScan", "DDoS", "BruteForce"]


def test_load_artifact_missing_path_raises_clear_error(tmp_path, monkeypatch):
    # Point the legacy-path check somewhere that doesn't exist, so this test doesn't depend on
    # whether the real repo happens to have an old models/detection_rf.joblib lying around.
    monkeypatch.setattr(classifier, "_LEGACY_MODEL_PATH", str(tmp_path / "no_legacy_here.joblib"))
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="src.detection.train"):
        classifier.load_artifact(str(missing))


def test_load_artifact_missing_path_mentions_migration_when_legacy_artifact_exists(tmp_path):
    legacy_path = tmp_path / "detection_rf.joblib"
    legacy_path.write_bytes(b"not a real artifact, just needs to exist")

    with pytest.raises(FileNotFoundError) as exc_info:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(classifier, "_LEGACY_MODEL_PATH", str(legacy_path))
            classifier.load_artifact(str(tmp_path / "does_not_exist.joblib"))

    assert "no longer used" in str(exc_info.value)
    assert "python -m src.detection.train" in str(exc_info.value)
