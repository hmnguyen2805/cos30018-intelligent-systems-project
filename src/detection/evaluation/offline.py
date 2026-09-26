"""
Offline, no-LLM batch evaluation over a whole split (TEST, or a VALIDATION
split carved from TRAIN for threshold selection) — no DetectionManager, no
per-event loop. batch model.predict_proba() is simpler and far faster than
looping DetectionManager.run() over hundreds of thousands of events.

Reports binary precision/recall/F1/ROC-AUC, per-class category
precision/recall/F1, false-positive categorisation rate, and a
CATEGORY_CONFIDENCE_THRESHOLD sweep with a recommended value (never applied
automatically). See docs/evaluation.md for exactly what's scored, why
Heartbleed and Benign are handled specially, and current result tables.
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
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
    """Vectorized equivalent of subagent.DetectionSubagent._choose_category.
    Returns (chosen, raw_top_category, raw_top_probability, disagreement) —
    chosen is "Unknown" wherever the top vote is BENIGN_CATEGORY or below
    threshold; disagreement flags a BENIGN_CATEGORY top vote."""
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


def is_heartbleed_label(raw_labels) -> np.ndarray:
    """True wherever the raw CICIDS2017 Label is Heartbleed — the one true
    label that maps to the same "Unknown" string a low-confidence punt uses.
    Excluded from per-class scoring (see docs/evaluation.md) rather than
    given its own class, since the model was never trained to tell them
    apart."""
    return np.array([isinstance(r, str) and r.strip().lower() == "heartbleed" for r in raw_labels])


def scored_attack_mask(
    true_label: np.ndarray, binary_pred: np.ndarray, true_category: np.ndarray,
    is_heartbleed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Same scoring rule as evaluate.compute_category_report: a true attack
    the binary model also caught, with a known true_category. Excludes
    Heartbleed rows too when `is_heartbleed` is given (see
    is_heartbleed_label)."""
    has_true_category = np.array([c is not None for c in true_category])
    mask = (true_label == 1) & (binary_pred == 1) & has_true_category
    if is_heartbleed is not None:
        mask = mask & ~is_heartbleed
    return mask


def compute_offline_category_report(
    true_category_for_table: np.ndarray, chosen_category: np.ndarray, raw_top_category: np.ndarray,
    table_mask: np.ndarray, attack_mask: np.ndarray,
) -> dict:
    """Per-class precision/recall/F1 over table_mask (true attacks caught
    plus false positives scored as BENIGN_CATEGORY, so precision reflects
    false alarms). accuracy/raw_accuracy/coverage/macro_f1 are computed over
    attack_mask only (true attacks, Heartbleed excluded) — a false positive
    is never a category to "get right". BENIGN_CATEGORY's own row is always
    precision=recall=0, since it can never itself be predicted; its purpose
    is only to penalize other classes' precision. See docs/evaluation.md."""
    n = int(attack_mask.sum())
    n_table = int(table_mask.sum())
    if n == 0:
        return {"n": 0, "n_table": n_table, "accuracy": None, "raw_accuracy": None, "coverage": None,
                "labels": [], "support": {}, "precision": {}, "recall": {}, "f1": {}, "macro_f1": None}

    true_attack = true_category_for_table[attack_mask]
    chosen_attack = chosen_category[attack_mask]
    raw_attack = raw_top_category[attack_mask]

    true_table = true_category_for_table[table_mask]
    chosen_table = chosen_category[table_mask]

    # "Unknown" is never its own label row — it still counts as a miss, just not its own row.
    attack_labels = sorted((set(true_attack.tolist()) | set(chosen_attack.tolist())) - {"Unknown"})
    table_labels = sorted(set(attack_labels) | {classifier.BENIGN_CATEGORY})

    precision, recall, f1, support = precision_recall_fscore_support(
        true_table, chosen_table, labels=table_labels, zero_division=0,
    )
    f1_by_label = dict(zip(table_labels, f1))
    macro_f1 = float(np.mean([f1_by_label[label] for label in attack_labels])) if attack_labels else None

    return {
        "n": n,
        "n_table": n_table,
        "accuracy": float((chosen_attack == true_attack).mean()),
        "raw_accuracy": float((raw_attack == true_attack).mean()),
        "coverage": float((chosen_attack != "Unknown").mean() * 100),
        "labels": table_labels,
        "support": {label: int(s) for label, s in zip(table_labels, support)},
        "precision": {label: float(p) for label, p in zip(table_labels, precision)},
        "recall": {label: float(r) for label, r in zip(table_labels, recall)},
        "f1": f1_by_label,
        "macro_f1": macro_f1,
    }


def print_offline_category_report(report: dict) -> None:
    if report["n"] == 0:
        print("category classifier: no truly anomalous events with a known true category in this split")
        return
    print(f"category classifier (n={report['n']} true attacks the binary model also caught, "
          f"n_table={report['n_table']} incl. binary false positives as Benign):")
    print(f"  accuracy (thresholded)={report['accuracy']:.4f}  "
          f"raw top-1 accuracy={report['raw_accuracy']:.4f}  "
          f"coverage={report['coverage']:.1f}%  macro F1 (attack classes only)="
          + (f"{report['macro_f1']:.4f}" if report["macro_f1"] is not None else "n/a"))
    label_col = "category"
    header = f"    {label_col:<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    print(header)
    for label in report["labels"]:
        print(f"    {label:<14}{report['precision'][label]:>10.3f}{report['recall'][label]:>10.3f}"
              f"{report['f1'][label]:>10.3f}{report['support'][label]:>10d}")


def compute_offline_false_positive_report(
    true_label: np.ndarray, binary_pred: np.ndarray, chosen_category: np.ndarray,
) -> dict:
    """Among the binary model's false positives, how many/what percentage
    still got a specific attack category instead of "Unknown". See
    docs/design-decisions.md for why this metric exists."""
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
    is_heartbleed: Optional[np.ndarray] = None,
) -> list:
    """One row per threshold: category accuracy, % Unknown, and false-
    positive categorisation rate, recomputed at each candidate threshold.
    Callers pass a VALIDATION split (never TEST) so selecting a threshold
    never tunes on test data."""
    mask = scored_attack_mask(true_label, binary_pred, true_category, is_heartbleed)
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


DEFAULT_MIN_CATEGORY_ACCURACY = 0.99


def recommend_category_threshold(
    sweep_rows: list, min_accuracy: float = DEFAULT_MIN_CATEGORY_ACCURACY,
) -> Optional[dict]:
    """Recommend the highest threshold whose validation accuracy clears
    min_accuracy, falling back to the single highest-accuracy row (ties:
    lower FP rate, then higher threshold) with meets_min_accuracy=False.
    Returns None if no row has a scored subset. Never changes
    subagent.DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD — recommendation only."""
    scored = [r for r in sweep_rows if r["category_accuracy"] is not None]
    if not scored:
        return None

    qualifying = [r for r in scored if r["category_accuracy"] >= min_accuracy]
    if qualifying:
        recommended = max(qualifying, key=lambda r: r["threshold"])
        meets_min_accuracy = True
    else:
        recommended = max(
            scored,
            key=lambda r: (r["category_accuracy"], -(r["fp_categorisation_rate"] or 0.0), r["threshold"]),
        )
        meets_min_accuracy = False

    return {**recommended, "min_accuracy": min_accuracy, "meets_min_accuracy": meets_min_accuracy}


def describe_recommendation_rule(recommended: dict) -> str:
    """The exact rule that produced `recommended`, for printing alongside it."""
    if recommended["meets_min_accuracy"]:
        return (f"highest threshold with validation category accuracy >= "
                f"{recommended['min_accuracy']:.2f}")
    return (f"no threshold reached >= {recommended['min_accuracy']:.2f} validation category "
            "accuracy; fell back to the highest-accuracy threshold (ties -> lower FP rate, "
            "then higher threshold)")


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


def run_offline_evaluation(
    default_threshold: float, min_category_accuracy: float = DEFAULT_MIN_CATEGORY_ACCURACY,
) -> None:
    """Evaluate both classifiers over the whole TEST split at
    default_threshold, then sweep CATEGORY_CONFIDENCE_THRESHOLD on a
    VALIDATION split and print a recommendation. Saves the sweep to
    results/category_threshold_sweep.csv. Never modifies any argument or
    training artifact."""
    binary_artifact = classifier.load_artifact(classifier.DEFAULT_BINARY_MODEL_PATH)
    category_artifact = classifier.load_artifact(classifier.DEFAULT_CATEGORY_MODEL_PATH)

    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    print(f"=== Offline evaluation (full test split, n={len(test_df)}, no LLM) ===")

    X_test = test_df[feature_names].to_numpy()
    y_test = data.binarize_labels(test_df[data.LABEL_COL])
    raw_labels_test = test_df[data.LABEL_COL].to_numpy()
    true_category_test = np.array(
        [data.map_cicids_label_to_category(raw) for raw in raw_labels_test], dtype=object,
    )
    is_heartbleed_test = is_heartbleed_label(raw_labels_test)

    binary_pred_test, binary_proba_test = batch_predict_binary(binary_artifact, X_test)
    print_binary_report(compute_binary_metrics(y_test, binary_pred_test, binary_proba_test))

    chosen_test, raw_top_test, _, disagreement_test = batch_predict_category(
        category_artifact, X_test, default_threshold,
    )

    attack_mask_test = scored_attack_mask(y_test, binary_pred_test, true_category_test, is_heartbleed_test)
    heartbleed_excluded_test = int(
        (scored_attack_mask(y_test, binary_pred_test, true_category_test) & is_heartbleed_test).sum()
    )
    fp_mask_test = (y_test == 0) & (binary_pred_test == 1)
    table_mask_test = attack_mask_test | fp_mask_test
    true_for_table_test = np.where(fp_mask_test, classifier.BENIGN_CATEGORY, true_category_test)

    print(f"(category decisions at threshold={default_threshold:.2f}, the current default; "
          f"excluded {heartbleed_excluded_test} Heartbleed event(s) — true label is the same "
          "\"Unknown\" string a low-confidence punt uses, so it isn't scored — see README)")
    print_offline_category_report(
        compute_offline_category_report(true_for_table_test, chosen_test, raw_top_test,
                                         table_mask_test, attack_mask_test)
    )
    print_false_positive_report(compute_offline_false_positive_report(y_test, binary_pred_test, chosen_test))
    print(f"model disagreement count (binary anomalous, category top vote Benign): "
          f"{compute_model_disagreement_count(binary_pred_test, disagreement_test)}")

    print()
    _, validation_df = data.split_validation(train_df)
    X_val = validation_df[feature_names].to_numpy()
    y_val = data.binarize_labels(validation_df[data.LABEL_COL])
    raw_labels_val = validation_df[data.LABEL_COL].to_numpy()
    true_category_val = np.array(
        [data.map_cicids_label_to_category(raw) for raw in raw_labels_val], dtype=object,
    )
    is_heartbleed_val = is_heartbleed_label(raw_labels_val)
    binary_pred_val, _ = batch_predict_binary(binary_artifact, X_val)

    sweep_rows = sweep_category_thresholds(
        category_artifact, X_val, true_category_val, binary_pred_val, y_val, is_heartbleed=is_heartbleed_val,
    )
    print_threshold_sweep(sweep_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sweep_csv_path = RESULTS_DIR / "category_threshold_sweep.csv"
    save_threshold_sweep_csv(sweep_csv_path, sweep_rows)
    print(f"Saved threshold sweep to {sweep_csv_path}")

    recommended = recommend_category_threshold(sweep_rows, min_accuracy=min_category_accuracy)
    if recommended is None:
        print("No recommendation: no scored true attacks in the validation split.")
        return

    print(f"Recommendation rule: {describe_recommendation_rule(recommended)}")
    print(f"Recommended CATEGORY_CONFIDENCE_THRESHOLD={recommended['threshold']:.2f} "
          f"(validation category accuracy={recommended['category_accuracy']:.3f}). "
          f"Current default is {default_threshold:.2f} — NOT changed automatically.")

    print(f"Recommended threshold's results on TEST (threshold={recommended['threshold']:.2f}):")
    chosen_recommended_test, raw_top_recommended_test, _, _ = batch_predict_category(
        category_artifact, X_test, recommended["threshold"],
    )
    print_offline_category_report(
        compute_offline_category_report(
            true_for_table_test, chosen_recommended_test, raw_top_recommended_test,
            table_mask_test, attack_mask_test,
        )
    )
    print_false_positive_report(
        compute_offline_false_positive_report(y_test, binary_pred_test, chosen_recommended_test)
    )
