"""
Offline, no-LLM evaluation: batch-predicts both classifiers directly over a
whole split (the full TEST split, or a VALIDATION split carved from TRAIN for
threshold selection) — no DetectionManager/DetectionSubagent, no per-event
agent loop, no borderline tree-vote recheck. That per-event machinery is what
evaluate.py's default (sampled, RF-only vs RF+LLM) mode exercises; this module
answers a different question — "how good are the two trained classifiers,
across the entire held-out data, on their own" — for which batch
model.predict_proba() over a numpy matrix is both simpler and orders of
magnitude faster than looping DetectionManager.run() over hundreds of
thousands of events.

Two things live here:
1. A full-test-split report: binary precision/recall/F1/ROC-AUC; category
   per-class precision/recall/F1 (+macro F1, row counts) on true attacks the
   binary model also caught; false-positive categorisation rate (+count);
   model-disagreement count; % Unknown among true attacks.
2. A CATEGORY_CONFIDENCE_THRESHOLD sweep, evaluated on a VALIDATION split
   carved from TRAIN (data.split_validation) rather than TEST — so choosing a
   threshold never tunes on the same data the headline numbers above are
   reported on. Prints a table, saves it as CSV, and prints a *recommended*
   threshold — subagent.DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD is never
   changed automatically.
"""
import csv
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
)

from src.detection import classifier
from src.detection.training import data

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CATEGORY_THRESHOLD_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)


def batch_predict_binary(binary_artifact: dict, X: np.ndarray):
    """Vectorized P(anomalous) + thresholded prediction for a whole matrix of
    rows — the batch equivalent of classifier.predict_proba_anomalous, called
    once per row in the online path."""
    model = binary_artifact["model"]
    anomalous_idx = list(model.classes_).index(1)
    proba = model.predict_proba(X)[:, anomalous_idx]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """precision/recall/F1/ROC-AUC over a whole split. roc_auc is None when
    y_true has only one class present (undefined otherwise)."""
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(set(y_true.tolist())) > 1 else None
    return {
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def print_binary_report(report: dict) -> None:
    print(f"binary classifier (n={report['n']}):")
    print(f"  precision={report['precision']:.4f}  recall={report['recall']:.4f}  "
          f"f1={report['f1']:.4f}  roc_auc="
          + (f"{report['roc_auc']:.4f}" if report["roc_auc"] is not None else "n/a"))


def batch_predict_category(category_artifact: dict, X: np.ndarray, threshold: float):
    """Vectorized equivalent of subagent.DetectionSubagent._choose_category
    over a whole matrix of rows. Returns four same-length arrays:
    - chosen: "Unknown" wherever the top vote is classifier.BENIGN_CATEGORY
      (a binary/category model disagreement — never trusted as a specific
      category, regardless of its probability) OR its probability is below
      `threshold`; otherwise the raw top category.
    - raw_top_category / raw_top_probability: the category model's own
      unthresholded top-1 vote, straight from predict_proba's argmax.
    - disagreement: True wherever raw_top_category is classifier.BENIGN_CATEGORY.
    """
    model = category_artifact["category_model"]
    classes = np.array(category_artifact.get("classes") or list(model.classes_))
    proba = model.predict_proba(X)
    top_idx = np.argmax(proba, axis=1)
    raw_top_category = classes[top_idx]
    raw_top_probability = proba[np.arange(len(X)), top_idx]

    disagreement = raw_top_category == classifier.BENIGN_CATEGORY
    below_threshold = raw_top_probability < threshold
    chosen = np.where(disagreement | below_threshold, "Unknown", raw_top_category)
    return chosen, raw_top_category, raw_top_probability, disagreement


def scored_attack_mask(true_label: np.ndarray, binary_pred: np.ndarray, true_category: np.ndarray) -> np.ndarray:
    """The same scoring rule as evaluate.compute_category_report: a true
    attack (true_label == 1) that the binary model ALSO caught (binary_pred
    == 1) and whose true_category is known (excludes BENIGN/unrecognized). A
    false negative (binary_pred == 0) never had a category computed online,
    so it's excluded here the same way."""
    has_true_category = np.array([c is not None for c in true_category])
    return (true_label == 1) & (binary_pred == 1) & has_true_category


def compute_offline_category_report(
    true_category: np.ndarray, chosen_category: np.ndarray, raw_top_category: np.ndarray,
    scored_mask: np.ndarray,
) -> dict:
    """Per-class precision/recall/F1 (+ macro F1, row counts/support) of the
    thresholded `chosen_category` against `true_category`, plus overall
    accuracy/raw_accuracy/pct_unknown — all over the scored subset (true
    attacks the binary model also caught, see scored_attack_mask)."""
    n = int(scored_mask.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "raw_accuracy": None, "pct_unknown": None,
                "labels": [], "support": {}, "precision": {}, "recall": {}, "f1": {}, "macro_f1": None}

    true_scored = true_category[scored_mask]
    chosen_scored = chosen_category[scored_mask]
    raw_scored = raw_top_category[scored_mask]

    labels = sorted(set(true_scored.tolist()) | set(chosen_scored.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_scored, chosen_scored, labels=labels, zero_division=0,
    )

    return {
        "n": n,
        "accuracy": float((chosen_scored == true_scored).mean()),
        "raw_accuracy": float((raw_scored == true_scored).mean()),
        "pct_unknown": float((chosen_scored == "Unknown").mean() * 100),
        "labels": labels,
        "support": {label: int(s) for label, s in zip(labels, support)},
        "precision": {label: float(p) for label, p in zip(labels, precision)},
        "recall": {label: float(r) for label, r in zip(labels, recall)},
        "f1": {label: float(x) for label, x in zip(labels, f1)},
        "macro_f1": float(np.mean(f1)),
    }


def print_offline_category_report(report: dict) -> None:
    if report["n"] == 0:
        print("category classifier: no truly anomalous events with a known true category in this split")
        return
    print(f"category classifier (n={report['n']} true attacks the binary model also caught):")
    print(f"  accuracy (thresholded)={report['accuracy']:.4f}  "
          f"raw top-1 accuracy={report['raw_accuracy']:.4f}  "
          f"% Unknown={report['pct_unknown']:.1f}  macro F1={report['macro_f1']:.4f}")
    label_col = "category"
    header = f"    {label_col:<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    print(header)
    for label in report["labels"]:
        print(f"    {label:<14}{report['precision'][label]:>10.3f}{report['recall'][label]:>10.3f}"
              f"{report['f1'][label]:>10.3f}{report['support'][label]:>10d}")


def compute_offline_false_positive_report(
    true_label: np.ndarray, binary_pred: np.ndarray, chosen_category: np.ndarray,
) -> dict:
    """Among events the binary model wrongly flagged anomalous (true_label ==
    0, binary_pred == 1), how many/what percentage still got a specific
    attack category (not "Unknown") from the thresholded category decision.
    See train_category.py's module docstring for the bug this metric exists
    to catch: before a Benign class existed, this was consistently well above
    0%."""
    fp_mask = (true_label == 0) & (binary_pred == 1)
    n = int(fp_mask.sum())
    if n == 0:
        return {"n": 0, "n_given_specific_category": 0, "pct_given_specific_category": None}
    n_given = int(np.sum(chosen_category[fp_mask] != "Unknown"))
    return {"n": n, "n_given_specific_category": n_given, "pct_given_specific_category": n_given / n * 100}


def print_false_positive_report(report: dict) -> None:
    if report["n"] == 0:
        print("false-positive categorisation: no binary-model false positives in this split")
        return
    print(f"false-positive categorisation (n={report['n']} events wrongly flagged anomalous by "
          "the binary model):")
    print(f"  given a specific attack category: {report['n_given_specific_category']}/{report['n']} "
          f"({report['pct_given_specific_category']:.1f}%)")


def compute_model_disagreement_count(binary_pred: np.ndarray, disagreement: np.ndarray) -> int:
    """How many events the binary model called anomalous where the category
    model's top vote was Benign — see subagent.py._choose_category."""
    return int(np.sum((binary_pred == 1) & disagreement))


def sweep_category_thresholds(
    category_artifact: dict, X: np.ndarray, true_category: np.ndarray,
    binary_pred: np.ndarray, true_label: np.ndarray,
    thresholds: Sequence[float] = CATEGORY_THRESHOLD_SWEEP,
) -> list:
    """One row per threshold: category accuracy and % Unknown on the scored
    true-attack subset, plus the false-positive categorisation rate — same
    three metrics compute_offline_category_report/compute_offline_false_positive_report
    report, recomputed at each candidate threshold. Callers pass a VALIDATION
    split's X/true_category/binary_pred/true_label (never TEST's) so
    selecting a threshold from this sweep never tunes on test data."""
    mask = scored_attack_mask(true_label, binary_pred, true_category)
    rows = []
    for threshold in thresholds:
        chosen, _, _, _ = batch_predict_category(category_artifact, X, threshold)
        n_scored = int(mask.sum())
        if n_scored:
            scored_true = true_category[mask]
            scored_chosen = chosen[mask]
            category_accuracy = float((scored_chosen == scored_true).mean())
            pct_unknown = float((scored_chosen == "Unknown").mean() * 100)
        else:
            category_accuracy = None
            pct_unknown = None
        fp_report = compute_offline_false_positive_report(true_label, binary_pred, chosen)
        rows.append({
            "threshold": threshold,
            "n_scored": n_scored,
            "category_accuracy": category_accuracy,
            "pct_unknown": pct_unknown,
            "n_fp": fp_report["n"],
            "fp_categorisation_rate": fp_report["pct_given_specific_category"],
        })
    return rows


def recommend_category_threshold(sweep_rows: list) -> Optional[dict]:
    """Pick the sweep row maximizing category accuracy; ties broken by the
    lowest false-positive categorisation rate, then by the highest threshold
    (more conservative — prefers punting to Unknown over a wrong specific
    guess). Returns None if no row has a scored true-attack subset at all.
    This never changes subagent.DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD — it's
    a printed recommendation only; the caller decides whether to act on it."""
    scored = [r for r in sweep_rows if r["category_accuracy"] is not None]
    if not scored:
        return None
    return max(
        scored,
        key=lambda r: (r["category_accuracy"], -(r["fp_categorisation_rate"] or 0.0), r["threshold"]),
    )


def print_threshold_sweep(sweep_rows: list) -> None:
    print("CATEGORY_CONFIDENCE_THRESHOLD sweep (on a VALIDATION split carved from TRAIN):")
    header = f"    {'threshold':>9}{'n_scored':>10}{'accuracy':>10}{'%unknown':>10}{'n_fp':>7}{'fp_rate':>9}"
    print(header)
    for row in sweep_rows:
        accuracy_text = f"{row['category_accuracy']:.3f}" if row["category_accuracy"] is not None else "n/a"
        unknown_text = f"{row['pct_unknown']:.1f}" if row["pct_unknown"] is not None else "n/a"
        fp_text = f"{row['fp_categorisation_rate']:.1f}" if row["fp_categorisation_rate"] is not None else "n/a"
        print(f"    {row['threshold']:>9.2f}{row['n_scored']:>10d}{accuracy_text:>10}"
              f"{unknown_text:>10}{row['n_fp']:>7d}{fp_text:>9}")


def save_threshold_sweep_csv(path: Path, sweep_rows: list) -> None:
    if not sweep_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)


def run_offline_evaluation(default_threshold: float) -> None:
    """Loads the full cleaned dataset, evaluates both classifiers in batch
    over the whole TEST split at `default_threshold`, then sweeps
    CATEGORY_CONFIDENCE_THRESHOLD on a VALIDATION split carved from TRAIN and
    prints a recommendation. Prints everything; saves the sweep to
    results/category_threshold_sweep.csv. Never modifies `default_threshold`
    or any training artifact."""
    binary_artifact = classifier.load_artifact(classifier.DEFAULT_BINARY_MODEL_PATH)
    category_artifact = classifier.load_artifact(classifier.DEFAULT_CATEGORY_MODEL_PATH)

    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    print(f"=== Offline evaluation (full test split, n={len(test_df)}, no LLM) ===")

    X_test = test_df[feature_names].to_numpy()
    y_test = data.binarize_labels(test_df[data.LABEL_COL])
    true_category_test = np.array(
        [data.map_cicids_label_to_category(raw) for raw in test_df[data.LABEL_COL].to_numpy()], dtype=object,
    )

    binary_pred_test, binary_proba_test = batch_predict_binary(binary_artifact, X_test)
    print_binary_report(compute_binary_metrics(y_test, binary_pred_test, binary_proba_test))

    chosen_test, raw_top_test, _, disagreement_test = batch_predict_category(
        category_artifact, X_test, default_threshold,
    )
    scored_mask_test = scored_attack_mask(y_test, binary_pred_test, true_category_test)
    print(f"(category decisions at threshold={default_threshold:.2f}, the current default)")
    print_offline_category_report(
        compute_offline_category_report(true_category_test, chosen_test, raw_top_test, scored_mask_test)
    )
    print_false_positive_report(compute_offline_false_positive_report(y_test, binary_pred_test, chosen_test))
    print(f"model disagreement count (binary anomalous, category top vote Benign): "
          f"{compute_model_disagreement_count(binary_pred_test, disagreement_test)}")

    print()
    _, validation_df = data.split_validation(train_df)
    X_val = validation_df[feature_names].to_numpy()
    y_val = data.binarize_labels(validation_df[data.LABEL_COL])
    true_category_val = np.array(
        [data.map_cicids_label_to_category(raw) for raw in validation_df[data.LABEL_COL].to_numpy()], dtype=object,
    )
    binary_pred_val, _ = batch_predict_binary(binary_artifact, X_val)

    sweep_rows = sweep_category_thresholds(category_artifact, X_val, true_category_val, binary_pred_val, y_val)
    print_threshold_sweep(sweep_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sweep_csv_path = RESULTS_DIR / "category_threshold_sweep.csv"
    save_threshold_sweep_csv(sweep_csv_path, sweep_rows)
    print(f"Saved threshold sweep to {sweep_csv_path}")

    recommended = recommend_category_threshold(sweep_rows)
    if recommended is None:
        print("No recommendation: no scored true attacks in the validation split.")
        return

    print(f"Recommended CATEGORY_CONFIDENCE_THRESHOLD={recommended['threshold']:.2f} "
          f"(maximizes validation category accuracy, ties -> lower FP rate, then higher threshold). "
          f"Current default is {default_threshold:.2f} — NOT changed automatically.")

    chosen_recommended_test, _, _, _ = batch_predict_category(
        category_artifact, X_test, recommended["threshold"],
    )
    fp_recommended_test = compute_offline_false_positive_report(y_test, binary_pred_test, chosen_recommended_test)
    if scored_mask_test.sum():
        scored_true_test = true_category_test[scored_mask_test]
        scored_chosen_test = chosen_recommended_test[scored_mask_test]
        accuracy_recommended_test = float((scored_chosen_test == scored_true_test).mean())
        pct_unknown_recommended_test = float((scored_chosen_test == "Unknown").mean() * 100)
    else:
        accuracy_recommended_test = None
        pct_unknown_recommended_test = None

    print(f"Recommended threshold's results on TEST (threshold={recommended['threshold']:.2f}):")
    print(f"  category accuracy="
          + (f"{accuracy_recommended_test:.3f}" if accuracy_recommended_test is not None else "n/a")
          + "  % Unknown="
          + (f"{pct_unknown_recommended_test:.1f}" if pct_unknown_recommended_test is not None else "n/a")
          + f"  FP categorisation rate="
          + (f"{fp_recommended_test['pct_given_specific_category']:.1f}"
             if fp_recommended_test["pct_given_specific_category"] is not None else "n/a"))
