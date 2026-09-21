"""
Unit tests for the pure data-prep / training helpers in src.detection.train.
Runs on small synthetic frames — no CICIDS2017 download needed. The
kagglehub-fetch-and-orchestrate main() is thin I/O glue and isn't unit tested
here.
"""
import numpy as np
import pandas as pd

from src.detection import train


def test_clean_dataframe_strips_whitespace_from_column_names():
    df = pd.DataFrame({" Label": ["BENIGN"], " Flow Duration": [1.0]})
    cleaned = train.clean_dataframe(df)
    assert list(cleaned.columns) == ["Label", "Flow Duration"]


def test_clean_dataframe_drops_rows_with_inf_or_nan():
    df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN", "BENIGN"],
        "Flow Duration": [1.0, np.inf, np.nan],
    })
    cleaned = train.clean_dataframe(df)
    assert len(cleaned) == 1


def test_clean_dataframe_drops_duplicate_rows():
    df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN"],
        "Flow Duration": [1.0, 1.0],
    })
    cleaned = train.clean_dataframe(df)
    assert len(cleaned) == 1


def test_binarize_labels_benign_is_zero():
    labels = pd.Series(["BENIGN", "BENIGN"])
    assert list(train.binarize_labels(labels)) == [0, 0]


def test_binarize_labels_attack_is_one():
    labels = pd.Series(["DoS Hulk", "PortScan"])
    assert list(train.binarize_labels(labels)) == [1, 1]


def test_binarize_labels_is_case_and_whitespace_insensitive():
    labels = pd.Series([" benign ", "Benign"])
    assert list(train.binarize_labels(labels)) == [0, 0]


def test_select_feature_names_excludes_label_and_non_numeric_columns():
    df = pd.DataFrame({
        "Label": ["BENIGN"],
        "Flow Duration": [1.0],
        "Some Text Column": ["x"],
    })
    assert train.select_feature_names(df, label_col="Label") == ["Flow Duration"]


def test_train_baseline_model_fits_and_predicts():
    X = np.array([[0.0], [0.0], [10.0], [10.0]])
    y = np.array([0, 0, 1, 1])
    model = train.train_baseline_model(X, y)
    assert list(model.predict([[0.0], [10.0]])) == [0, 1]
