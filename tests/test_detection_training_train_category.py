"""Unit tests for src.detection.training.train_category's pure helpers."""
import numpy as np
import pandas as pd

from src.detection import classifier
from src.detection.training import train_category

FEATURE_NAMES = ["x"]


def _make_df():
    return pd.DataFrame({
        "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS", "PortScan", "Some Future Attack"],
        "x": [0.0, 1.0, 10.0, 11.0, 20.0, 30.0],
    })


def test_build_category_training_set_excludes_benign_rows():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES)
    assert "BENIGN" not in y  # map_cicids_label_to_category("BENIGN") is None -> excluded
    assert len(X) == len(y)


def test_build_category_training_set_excludes_unrecognized_labels():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES)
    # "Some Future Attack" doesn't map to anything -> excluded, so only DDoS/PortScan remain.
    assert set(y) == {"DDoS", "PortScan"}
    assert len(X) == 3  # 2 DDoS rows + 1 PortScan row


def test_build_category_training_set_maps_labels_to_categories():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES)
    assert list(y).count("DDoS") == 2
    assert list(y).count("PortScan") == 1


def test_build_category_training_set_features_align_with_feature_names():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES)
    # Rows are DDoS (x=10,11) and PortScan (x=20) in that order.
    assert sorted(X.flatten().tolist()) == [10.0, 11.0, 20.0]


def test_train_category_model_fits_and_predicts():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["PortScan", "PortScan", "DDoS", "DDoS"])
    model = train_category.train_category_model(X, y)
    assert list(model.predict([[0.5], [10.5]])) == ["PortScan", "DDoS"]


# --- include_benign: an explicit "this looks benign" class --------------------
# See train_category.py's module docstring: without a Benign option, the category
# model had no way to push back on a binary-model false positive, and a real run
# found benign events wrongly flagged anomalous all got a confident attack label.

def test_build_category_training_set_default_still_excludes_benign():
    # include_benign defaults to False — identical to the pre-Benign behavior, so
    # existing callers (and the TEST-split call with no ratio) are unaffected.
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES)
    assert classifier.BENIGN_CATEGORY not in y


def test_build_category_training_set_include_benign_keeps_benign_rows():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES, include_benign=True)
    assert list(y).count(classifier.BENIGN_CATEGORY) == 2  # both BENIGN rows kept, no ratio given
    assert set(y) == {classifier.BENIGN_CATEGORY, "DDoS", "PortScan"}
    assert len(X) == len(y) == 5  # 2 benign + 2 DDoS + 1 PortScan; "Some Future Attack" still excluded


def test_build_category_training_set_include_benign_still_excludes_unrecognized_labels():
    X, y = train_category.build_category_training_set(_make_df(), FEATURE_NAMES, include_benign=True)
    assert "Some Future Attack" not in y


def _make_imbalanced_df(n_benign: int, n_attack: int):
    return pd.DataFrame({
        "Label": ["BENIGN"] * n_benign + ["DDoS"] * n_attack,
        "x": list(range(n_benign + n_attack)),
    })


def test_build_category_training_set_downsamples_benign_to_the_given_ratio():
    df = _make_imbalanced_df(n_benign=100, n_attack=10)
    X, y = train_category.build_category_training_set(
        df, FEATURE_NAMES, include_benign=True, benign_to_attack_ratio=2.0,
    )
    y = list(y)
    assert y.count("DDoS") == 10  # attack rows never downsampled
    assert y.count(classifier.BENIGN_CATEGORY) == 20  # capped at ratio * n_attack
    assert len(X) == len(y) == 30


def test_build_category_training_set_ratio_is_a_no_op_when_benign_is_already_below_it():
    df = _make_imbalanced_df(n_benign=5, n_attack=10)
    X, y = train_category.build_category_training_set(
        df, FEATURE_NAMES, include_benign=True, benign_to_attack_ratio=2.0,
    )
    y = list(y)
    assert y.count(classifier.BENIGN_CATEGORY) == 5  # already under the cap (20) — nothing dropped
    assert y.count("DDoS") == 10


def test_build_category_training_set_downsampling_is_deterministic_given_a_random_state():
    df = _make_imbalanced_df(n_benign=100, n_attack=10)
    _, y1 = train_category.build_category_training_set(
        df, FEATURE_NAMES, include_benign=True, benign_to_attack_ratio=2.0, random_state=7,
    )
    _, y2 = train_category.build_category_training_set(
        df, FEATURE_NAMES, include_benign=True, benign_to_attack_ratio=2.0, random_state=7,
    )
    assert list(y1) == list(y2)
