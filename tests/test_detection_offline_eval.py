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


# --- is_heartbleed_label / scored_attack_mask ----------------------------------

def test_is_heartbleed_label_matches_case_insensitively():
    raw = np.array(["Heartbleed", "heartbleed", "DDoS", "BENIGN"], dtype=object)
    assert list(offline_eval.is_heartbleed_label(raw)) == [True, True, False, False]


def test_scored_attack_mask_excludes_benign_false_negatives_and_unknown_true_category():
    true_label = np.array([1, 1, 1, 0])
    binary_pred = np.array([1, 0, 1, 1])  # index 1: false negative (binary missed it)
    true_category = np.array(["DDoS", "DDoS", None, "PortScan"], dtype=object)
    mask = offline_eval.scored_attack_mask(true_label, binary_pred, true_category)
    assert list(mask) == [True, False, False, False]


def test_scored_attack_mask_excludes_heartbleed_when_given():
    true_label = np.array([1, 1])
    binary_pred = np.array([1, 1])
    true_category = np.array(["DDoS", "Unknown"], dtype=object)  # index 1: Heartbleed's mapped category
    is_heartbleed = np.array([False, True])
    mask = offline_eval.scored_attack_mask(true_label, binary_pred, true_category, is_heartbleed)
    assert list(mask) == [True, False]


# --- compute_offline_category_report -------------------------------------------
# table_mask = attack_mask | fp_mask (every flagged flow); attack_mask alone drives
# accuracy/raw_accuracy/coverage/macro_f1 — a false positive isn't a category to "get
# right", and macro F1 must average real attack classes only, never the Benign row.

def test_compute_offline_category_report_empty_when_nothing_scored():
    mask = np.array([False, False])
    report = offline_eval.compute_offline_category_report(
        np.array(["DDoS", "DoS"], dtype=object), np.array(["DDoS", "Unknown"], dtype=object),
        np.array(["DDoS", "DoS"], dtype=object), mask, mask,
    )
    assert report["n"] == 0
    assert report["accuracy"] is None


def test_compute_offline_category_report_computes_accuracy_and_per_class_metrics_no_fps():
    true_category = np.array(["DDoS", "DDoS", "PortScan"], dtype=object)
    chosen_category = np.array(["DDoS", "PortScan", "PortScan"], dtype=object)  # 1 wrong (DDoS->PortScan)
    raw_top_category = np.array(["DDoS", "DDoS", "PortScan"], dtype=object)     # raw would've been all correct
    mask = np.array([True, True, True])  # no false positives in this table

    report = offline_eval.compute_offline_category_report(
        true_category, chosen_category, raw_top_category, mask, mask,
    )

    assert report["n"] == 3
    assert report["n_table"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["raw_accuracy"] == pytest.approx(1.0)
    assert report["coverage"] == 100.0  # nothing was punted to Unknown
    assert report["support"]["DDoS"] == 2
    assert report["support"]["PortScan"] == 1
    # DDoS: 1 true, chosen correctly once -> recall 0.5; PortScan: predicted twice, true once -> precision 0.5
    assert report["recall"]["DDoS"] == pytest.approx(0.5)
    assert report["precision"]["PortScan"] == pytest.approx(0.5)
    assert "Unknown" not in report["labels"]  # never its own row (see fix for Heartbleed conflation)


def test_compute_offline_category_report_coverage_replaces_pct_unknown():
    true_category = np.array(["DDoS", "DoS"], dtype=object)
    chosen_category = np.array(["Unknown", "Unknown"], dtype=object)
    raw_top_category = np.array(["DDoS", "DoS"], dtype=object)
    mask = np.array([True, True])

    report = offline_eval.compute_offline_category_report(
        true_category, chosen_category, raw_top_category, mask, mask,
    )
    assert report["coverage"] == 0.0  # nothing got a specific category
    assert report["accuracy"] == 0.0
    assert report["raw_accuracy"] == 1.0
    assert "pct_unknown" not in report


def test_compute_offline_category_report_includes_benign_row_for_false_positives():
    # 2 true attacks (both correctly labeled) + 1 binary false positive wrongly labeled DDoS.
    true_category_for_table = np.array(["DDoS", "PortScan", classifier.BENIGN_CATEGORY], dtype=object)
    chosen_category = np.array(["DDoS", "PortScan", "DDoS"], dtype=object)
    raw_top_category = np.array(["DDoS", "PortScan", "DDoS"], dtype=object)
    table_mask = np.array([True, True, True])
    attack_mask = np.array([True, True, False])  # the false positive is NOT a true attack

    report = offline_eval.compute_offline_category_report(
        true_category_for_table, chosen_category, raw_top_category, table_mask, attack_mask,
    )

    assert report["n"] == 2       # only the 2 true attacks
    assert report["n_table"] == 3  # attacks + the 1 false positive
    assert report["accuracy"] == 1.0       # both true attacks were correct
    assert classifier.BENIGN_CATEGORY in report["labels"]
    assert report["support"][classifier.BENIGN_CATEGORY] == 1
    # DDoS precision must now reflect the false alarm: 1 correct DDoS + 1 false-positive-as-DDoS
    # predicted, so precision = 1/2, even though accuracy over true attacks alone is 1.0.
    assert report["precision"]["DDoS"] == pytest.approx(0.5)
    assert report["recall"]["DDoS"] == pytest.approx(1.0)  # the one true DDoS was still found


def test_compute_offline_category_report_macro_f1_excludes_the_benign_row():
    true_category_for_table = np.array(["DDoS", "PortScan", classifier.BENIGN_CATEGORY], dtype=object)
    chosen_category = np.array(["DDoS", "PortScan", "DDoS"], dtype=object)
    raw_top_category = np.array(["DDoS", "PortScan", "DDoS"], dtype=object)
    table_mask = np.array([True, True, True])
    attack_mask = np.array([True, True, False])

    report = offline_eval.compute_offline_category_report(
        true_category_for_table, chosen_category, raw_top_category, table_mask, attack_mask,
    )

    attack_only_f1 = [report["f1"][label] for label in report["labels"] if label != classifier.BENIGN_CATEGORY]
    assert report["macro_f1"] == pytest.approx(np.mean(attack_only_f1))
    assert classifier.BENIGN_CATEGORY in report["f1"]  # Benign still has its OWN row/score...
    # ...but macro_f1 must differ from a naive mean over every label whenever Benign's own f1
    # differs from the attack classes' mean (it does here: precision/recall=0 since "Benign" is
    # never a valid chosen_category value).
    all_labels_f1 = list(report["f1"].values())
    assert report["macro_f1"] != pytest.approx(np.mean(all_labels_f1))


def test_compute_offline_category_report_benign_row_is_never_a_hit():
    # "Benign" can never be `chosen_category` (see subagent.py._choose_category — a Benign
    # top vote is always reported as "Unknown"), so its own precision/recall must always be 0,
    # regardless of how many false positives got punted to "Unknown" instead of a wrong label.
    true_category_for_table = np.array(["DDoS", classifier.BENIGN_CATEGORY], dtype=object)
    chosen_category = np.array(["DDoS", "Unknown"], dtype=object)  # FP correctly punted
    raw_top_category = np.array(["DDoS", classifier.BENIGN_CATEGORY], dtype=object)
    table_mask = np.array([True, True])
    attack_mask = np.array([True, False])

    report = offline_eval.compute_offline_category_report(
        true_category_for_table, chosen_category, raw_top_category, table_mask, attack_mask,
    )
    assert report["precision"][classifier.BENIGN_CATEGORY] == 0.0
    assert report["recall"][classifier.BENIGN_CATEGORY] == 0.0


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


# --- recommend_category_threshold: HIGHEST threshold clearing min_accuracy --
# Primary rule: among sweep rows whose validation category accuracy clears
# min_accuracy, pick the one with the HIGHEST threshold (most conservative).
# Only when NONE clear the bar does it fall back to the old
# max-accuracy/tie-break rule.

def test_recommend_category_threshold_picks_highest_threshold_clearing_the_bar():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.999, "fp_categorisation_rate": 10.0},
        {"threshold": 0.6, "category_accuracy": 0.995, "fp_categorisation_rate": 5.0},
        {"threshold": 0.7, "category_accuracy": 0.991, "fp_categorisation_rate": 0.0},
        {"threshold": 0.8, "category_accuracy": 0.980, "fp_categorisation_rate": 0.0},  # below 0.99 bar
    ]
    recommended = offline_eval.recommend_category_threshold(rows, min_accuracy=0.99)
    assert recommended["threshold"] == 0.7  # highest threshold still >= 0.99, not the highest accuracy
    assert recommended["meets_min_accuracy"] is True
    assert recommended["min_accuracy"] == 0.99


def test_recommend_category_threshold_min_accuracy_is_configurable():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.97, "fp_categorisation_rate": 10.0},
        {"threshold": 0.6, "category_accuracy": 0.95, "fp_categorisation_rate": 5.0},
    ]
    recommended = offline_eval.recommend_category_threshold(rows, min_accuracy=0.96)
    assert recommended["threshold"] == 0.5  # only 0.97 clears a 0.96 bar
    assert recommended["meets_min_accuracy"] is True


def test_recommend_category_threshold_falls_back_when_no_threshold_clears_the_bar():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.6, "fp_categorisation_rate": 10.0},
        {"threshold": 0.6, "category_accuracy": 0.9, "fp_categorisation_rate": 5.0},
        {"threshold": 0.7, "category_accuracy": 0.7, "fp_categorisation_rate": 0.0},
    ]
    recommended = offline_eval.recommend_category_threshold(rows, min_accuracy=0.99)
    assert recommended["threshold"] == 0.6  # fallback: highest accuracy among all rows
    assert recommended["meets_min_accuracy"] is False


def test_recommend_category_threshold_fallback_breaks_ties_by_lower_fp_rate_then_higher_threshold():
    rows = [
        {"threshold": 0.5, "category_accuracy": 0.8, "fp_categorisation_rate": 20.0},
        {"threshold": 0.6, "category_accuracy": 0.8, "fp_categorisation_rate": 5.0},
        {"threshold": 0.7, "category_accuracy": 0.8, "fp_categorisation_rate": 5.0},
    ]
    recommended = offline_eval.recommend_category_threshold(rows, min_accuracy=0.99)
    assert recommended["threshold"] == 0.7  # tied accuracy+fp_rate with 0.6 -> higher threshold wins
    assert recommended["meets_min_accuracy"] is False


def test_recommend_category_threshold_returns_none_when_nothing_scored():
    rows = [{"threshold": 0.5, "category_accuracy": None, "fp_categorisation_rate": None}]
    assert offline_eval.recommend_category_threshold(rows) is None


def test_recommend_category_threshold_default_min_accuracy_is_point_99():
    rows = [{"threshold": 0.5, "category_accuracy": 0.999, "fp_categorisation_rate": 0.0}]
    recommended = offline_eval.recommend_category_threshold(rows)  # no min_accuracy passed
    assert recommended["min_accuracy"] == offline_eval.DEFAULT_MIN_CATEGORY_ACCURACY == 0.99


def test_recommend_category_threshold_real_looking_values_all_clear_the_bar_no_fallback():
    # A real --offline sweep: every swept threshold's validation accuracy (0.999..0.992) is
    # already >= 0.99, so 0.9 (the highest) must be picked directly — meets_min_accuracy True,
    # no fallback — even though 0.9's own accuracy (0.992) is the LOWEST of the five.
    rows = [
        {"threshold": 0.5, "n_scored": 68119, "category_accuracy": 0.999, "pct_unknown": 0.1,
         "n_fp": 190, "fp_categorisation_rate": 61.1},
        {"threshold": 0.6, "n_scored": 68119, "category_accuracy": 0.998, "pct_unknown": 0.2,
         "n_fp": 190, "fp_categorisation_rate": 49.5},
        {"threshold": 0.7, "n_scored": 68119, "category_accuracy": 0.997, "pct_unknown": 0.3,
         "n_fp": 190, "fp_categorisation_rate": 46.8},
        {"threshold": 0.8, "n_scored": 68119, "category_accuracy": 0.995, "pct_unknown": 0.5,
         "n_fp": 190, "fp_categorisation_rate": 45.8},
        {"threshold": 0.9, "n_scored": 68119, "category_accuracy": 0.992, "pct_unknown": 0.8,
         "n_fp": 190, "fp_categorisation_rate": 43.7},
    ]
    recommended = offline_eval.recommend_category_threshold(rows, min_accuracy=0.99)
    assert recommended["threshold"] == 0.9
    assert recommended["meets_min_accuracy"] is True
    rule_text = offline_eval.describe_recommendation_rule(recommended)
    assert "fell back" not in rule_text.lower()
    assert "fallback" not in rule_text.lower()


# --- describe_recommendation_rule: the printed rule text ----------------------

def test_describe_recommendation_rule_when_bar_is_met():
    recommended = {"threshold": 0.7, "min_accuracy": 0.99, "meets_min_accuracy": True}
    text = offline_eval.describe_recommendation_rule(recommended)
    assert "0.99" in text
    assert "highest threshold" in text.lower()


def test_describe_recommendation_rule_when_bar_is_not_met():
    recommended = {"threshold": 0.6, "min_accuracy": 0.99, "meets_min_accuracy": False}
    text = offline_eval.describe_recommendation_rule(recommended)
    assert "0.99" in text
    assert "fell back" in text.lower() or "fallback" in text.lower()
