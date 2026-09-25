"""
Unit tests for src.detection.offline_eval's batch (no-LLM, no per-event agent
loop) metrics and the CATEGORY_CONFIDENCE_THRESHOLD sweep. All synthetic,
small, in-memory data — no real dataset, no trained artifact files.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection import classifier, offline_eval


def _fake_binary_artifact(proba_anomalous):
    """A fake binary artifact whose predict_proba returns a fixed [P(benign),
    P(anomalous)] matrix built from the given per-row P(anomalous) values."""
    model = MagicMock()
    model.classes_ = [0, 1]
    proba_anomalous = np.asarray(proba_anomalous, dtype=float)
    model.predict_proba.return_value = np.column_stack([1 - proba_anomalous, proba_anomalous])
    return {"model": model, "feature_names": ["x"]}


def _fake_category_artifact(classes, proba_matrix):
    model = MagicMock()
    model.classes_ = classes
    model.predict_proba.return_value = np.asarray(proba_matrix, dtype=float)
    return {"category_model": model, "classes": classes, "feature_names": ["x"]}


# --- batch_predict_binary / compute_binary_metrics ----------------------------

def test_batch_predict_binary_thresholds_at_half():
    artifact = _fake_binary_artifact([0.1, 0.49, 0.5, 0.51, 0.9])
    pred, proba = offline_eval.batch_predict_binary(artifact, np.zeros((5, 1)))
    assert list(pred) == [0, 0, 1, 1, 1]
    assert list(proba) == pytest.approx([0.1, 0.49, 0.5, 0.51, 0.9])


def test_compute_binary_metrics_matches_hand_computed_values():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])  # 1 TP, 1 FN, 1 TN, 1 FP
    y_proba = np.array([0.9, 0.4, 0.3, 0.6])
    report = offline_eval.compute_binary_metrics(y_true, y_pred, y_proba)
    assert report["n"] == 4
    assert report["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
    assert report["recall"] == pytest.approx(0.5)      # 1 TP / (1 TP + 1 FN)
    assert report["f1"] == pytest.approx(0.5)
    assert report["roc_auc"] is not None


def test_compute_binary_metrics_roc_auc_is_none_with_a_single_class():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.4])
    report = offline_eval.compute_binary_metrics(y_true, y_pred, y_proba)
    assert report["roc_auc"] is None


# --- batch_predict_category ----------------------------------------------------

CATEGORY_CLASSES = [classifier.BENIGN_CATEGORY, "DDoS", "PortScan"]


def test_batch_predict_category_picks_the_argmax_class():
    artifact = _fake_category_artifact(CATEGORY_CLASSES, [
        [0.1, 0.8, 0.1],  # DDoS
        [0.1, 0.2, 0.7],  # PortScan
    ])
    chosen, raw_top, raw_prob, disagreement = offline_eval.batch_predict_category(
        artifact, np.zeros((2, 1)), threshold=0.6,
    )
    assert list(raw_top) == ["DDoS", "PortScan"]
    assert list(raw_prob) == pytest.approx([0.8, 0.7])
    assert list(chosen) == ["DDoS", "PortScan"]
    assert list(disagreement) == [False, False]


def test_batch_predict_category_below_threshold_becomes_unknown():
    artifact = _fake_category_artifact(CATEGORY_CLASSES, [[0.1, 0.55, 0.35]])  # top=DDoS @ 0.55
    chosen, raw_top, raw_prob, disagreement = offline_eval.batch_predict_category(
        artifact, np.zeros((1, 1)), threshold=0.6,
    )
    assert raw_top[0] == "DDoS"
    assert chosen[0] == "Unknown"
    assert disagreement[0] == False  # noqa: E712 - numpy bool, not a template-Y bool


def test_batch_predict_category_at_exactly_the_threshold_is_accepted():
    artifact = _fake_category_artifact(CATEGORY_CLASSES, [[0.1, 0.6, 0.3]])
    chosen, _, _, _ = offline_eval.batch_predict_category(artifact, np.zeros((1, 1)), threshold=0.6)
    assert chosen[0] == "DDoS"


def test_batch_predict_category_benign_top_vote_is_always_unknown_and_flagged():
    # High confidence Benign vote — must be "Unknown", not "Benign", regardless of threshold.
    artifact = _fake_category_artifact(CATEGORY_CLASSES, [[0.99, 0.005, 0.005]])
    chosen, raw_top, raw_prob, disagreement = offline_eval.batch_predict_category(
        artifact, np.zeros((1, 1)), threshold=0.1,  # low threshold: would otherwise accept 0.99
    )
    assert raw_top[0] == classifier.BENIGN_CATEGORY
    assert raw_prob[0] == pytest.approx(0.99)
    assert chosen[0] == "Unknown"
    assert disagreement[0] == True  # noqa: E712


# --- scored_attack_mask --------------------------------------------------------

def test_scored_attack_mask_excludes_benign_false_negatives_and_unknown_true_category():
    true_label = np.array([1, 1, 1, 0])
    binary_pred = np.array([1, 0, 1, 1])  # index 1: false negative (binary missed it)
    true_category = np.array(["DDoS", "DDoS", None, "PortScan"], dtype=object)
    mask = offline_eval.scored_attack_mask(true_label, binary_pred, true_category)
    assert list(mask) == [True, False, False, False]


# --- compute_offline_category_report -------------------------------------------

def test_compute_offline_category_report_empty_when_nothing_scored():
    mask = np.array([False, False])
    report = offline_eval.compute_offline_category_report(
        np.array(["DDoS", "DoS"], dtype=object), np.array(["DDoS", "Unknown"], dtype=object),
        np.array(["DDoS", "DoS"], dtype=object), mask,
    )
    assert report["n"] == 0
    assert report["accuracy"] is None


def test_compute_offline_category_report_computes_accuracy_and_per_class_metrics():
    true_category = np.array(["DDoS", "DDoS", "PortScan"], dtype=object)
    chosen_category = np.array(["DDoS", "PortScan", "PortScan"], dtype=object)  # 1 wrong (DDoS->PortScan)
    raw_top_category = np.array(["DDoS", "DDoS", "PortScan"], dtype=object)     # raw would've been all correct
    mask = np.array([True, True, True])

    report = offline_eval.compute_offline_category_report(true_category, chosen_category, raw_top_category, mask)

    assert report["n"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["raw_accuracy"] == pytest.approx(1.0)
    assert report["pct_unknown"] == 0.0
    assert report["support"]["DDoS"] == 2
    assert report["support"]["PortScan"] == 1
    # DDoS: 1 true, chosen correctly once -> recall 0.5; PortScan: predicted twice, true once -> precision 0.5
    assert report["recall"]["DDoS"] == pytest.approx(0.5)
    assert report["precision"]["PortScan"] == pytest.approx(0.5)
    assert report["macro_f1"] == pytest.approx(np.mean(list(report["f1"].values())))


def test_compute_offline_category_report_counts_unknown_predictions():
    true_category = np.array(["DDoS", "DoS"], dtype=object)
    chosen_category = np.array(["Unknown", "Unknown"], dtype=object)
    raw_top_category = np.array(["DDoS", "DoS"], dtype=object)
    mask = np.array([True, True])

    report = offline_eval.compute_offline_category_report(true_category, chosen_category, raw_top_category, mask)
    assert report["pct_unknown"] == 100.0
    assert report["accuracy"] == 0.0
    assert report["raw_accuracy"] == 1.0


# --- compute_offline_false_positive_report -------------------------------------

def test_false_positive_report_empty_with_no_false_positives():
    true_label = np.array([1, 0])
    binary_pred = np.array([1, 0])  # the one benign row was correctly called benign
    chosen_category = np.array(["DDoS", "Unknown"], dtype=object)
    report = offline_eval.compute_offline_false_positive_report(true_label, binary_pred, chosen_category)
    assert report["n"] == 0
    assert report["pct_given_specific_category"] is None


def test_false_positive_report_counts_specific_categories_given_to_false_positives():
    true_label = np.array([0, 0, 0])
    binary_pred = np.array([1, 1, 1])  # all three are false positives
    chosen_category = np.array(["Botnet", "Unknown", "DoS"], dtype=object)
    report = offline_eval.compute_offline_false_positive_report(true_label, binary_pred, chosen_category)
    assert report["n"] == 3
    assert report["n_given_specific_category"] == 2
    assert report["pct_given_specific_category"] == pytest.approx(2 / 3 * 100)


# --- compute_model_disagreement_count ------------------------------------------

def test_model_disagreement_count_only_counts_binary_anomalous_rows():
    binary_pred = np.array([1, 0, 1])
    disagreement = np.array([True, True, False])  # index 1 disagreed but binary said benign -> not counted
    assert offline_eval.compute_model_disagreement_count(binary_pred, disagreement) == 1


# --- sweep_category_thresholds / recommend_category_threshold ------------------

def test_sweep_category_thresholds_returns_one_row_per_threshold():
    artifact = _fake_category_artifact(CATEGORY_CLASSES, [
        [0.05, 0.75, 0.20],  # DDoS @ 0.75, true=DDoS, binary caught it
        [0.05, 0.20, 0.75],  # PortScan @ 0.75, true=None (benign, false positive)
    ])
    true_category = np.array(["DDoS", None], dtype=object)
    binary_pred = np.array([1, 1])
    true_label = np.array([1, 0])

    rows = offline_eval.sweep_category_thresholds(
        artifact, np.zeros((2, 1)), true_category, binary_pred, true_label, thresholds=(0.5, 0.8),
    )

    assert [r["threshold"] for r in rows] == [0.5, 0.8]
    # threshold=0.5: both top votes (0.75) clear it -> the true attack scores correct,
    # and the benign false positive still gets a specific category (PortScan).
    assert rows[0]["category_accuracy"] == pytest.approx(1.0)
    assert rows[0]["fp_categorisation_rate"] == pytest.approx(100.0)
    # threshold=0.8: neither 0.75 vote clears it -> both become Unknown.
    assert rows[1]["category_accuracy"] == pytest.approx(0.0)
    assert rows[1]["pct_unknown"] == pytest.approx(100.0)
    assert rows[1]["fp_categorisation_rate"] == pytest.approx(0.0)


def test_recommend_category_threshold_picks_highest_accuracy():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.6, "fp_categorisation_rate": 10.0},
        {"threshold": 0.6, "category_accuracy": 0.9, "fp_categorisation_rate": 5.0},
        {"threshold": 0.7, "category_accuracy": 0.7, "fp_categorisation_rate": 0.0},
    ]
    recommended = offline_eval.recommend_category_threshold(rows)
    assert recommended["threshold"] == 0.6


def test_recommend_category_threshold_breaks_ties_by_lower_fp_rate_then_higher_threshold():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.8, "fp_categorisation_rate": 20.0},
        {"threshold": 0.6, "category_accuracy": 0.8, "fp_categorisation_rate": 5.0},
        {"threshold": 0.7, "category_accuracy": 0.8, "fp_categorisation_rate": 5.0},
    ]
    recommended = offline_eval.recommend_category_threshold(rows)
    assert recommended["threshold"] == 0.7  # tied accuracy+fp_rate with 0.6 -> higher threshold wins


def test_recommend_category_threshold_returns_none_when_nothing_scored():
    rows = [{"threshold": 0.5, "category_accuracy": None, "fp_categorisation_rate": None}]
    assert offline_eval.recommend_category_threshold(rows) is None
