"""Unit tests for src.detection.training.train_category's pure helpers."""
import numpy as np
import pandas as pd

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
