"""
Offline, no-LLM batch evaluation over a whole split (TEST, or a VALIDATION
split carved from TRAIN for threshold selection) — no DetectionManager, no
per-event loop. batch model.predict_proba() is simpler and far faster than
looping DetectionManager.run() over hundreds of thousands of events.

Reports binary precision/recall/F1/ROC-AUC, per-fine-label AND per-coarse-
category precision/recall/F1 (coarse = the category the three-way decision in
subagent.decide_category picked), false-positive categorisation rate, and a
CATEGORY_CONFIDENCE_THRESHOLD sweep with a recommended value (never applied
automatically). Classes with support < MIN_EVAL_SUPPORT are flagged "too small
to evaluate". See docs/evaluation.md for exactly what's scored and why Benign
is handled specially.
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
MIN_EVAL_SUPPORT = 50


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


def batch_category_probabilities(category_artifact: dict, X: np.ndarray) -> dict:
    """One predict_proba pass for a whole matrix: fine-label probabilities and
    the coarse-group probabilities aggregated from them (sum of each group's
    fine labels, as classifier.coarse_probabilities does per event)."""
    model = category_artifact["category_model"]
    labels = np.array(category_artifact.get("classes") or list(model.classes_))
    fine = np.asarray(model.predict_proba(X))
    groups = np.array([classifier.coarse_category(label) for label in labels])
    categories = np.unique(groups)
    coarse = np.stack([fine[:, groups == c].sum(axis=1) for c in categories], axis=1)
    return {"labels": labels, "fine": fine, "categories": categories, "coarse": coarse}


def decide_batch(probs: dict, threshold: float) -> dict:
    """Vectorized subagent.decide_category over batch_category_probabilities'
    output. Returns arrays: label, category (the decision), raw_label,
    raw_category (top-1, unthresholded), disagreement (a Benign top vote)."""
    n = len(probs["fine"])
    fine_idx = probs["fine"].argmax(axis=1)
    coarse_idx = probs["coarse"].argmax(axis=1)
    raw_label = probs["labels"][fine_idx]
    raw_category = probs["categories"][coarse_idx]
    fine_p = probs["fine"][np.arange(n), fine_idx]
    coarse_p = probs["coarse"][np.arange(n), coarse_idx]

    disagreement = (raw_label == classifier.BENIGN_CATEGORY) | (raw_category == classifier.BENIGN_CATEGORY)
    label_ok = ~disagreement & (fine_p >= threshold)
    category_ok = ~disagreement & (coarse_p >= threshold)
    raw_label_group = np.array([classifier.coarse_category(label) for label in raw_label])
    label = np.where(label_ok, raw_label, "Unknown")
    category = np.where(label_ok, raw_label_group, np.where(category_ok, raw_category, "Unknown"))
    return {"label": label, "category": category, "raw_label": raw_label,
            "raw_category": raw_category, "disagreement": disagreement}


def batch_predict_category(category_artifact: dict, X: np.ndarray, threshold: float) -> dict:
    """Vectorized equivalent of subagent.DetectionSubagent._choose_category."""
    return decide_batch(batch_category_probabilities(category_artifact, X), threshold)


def scored_attack_mask(true_label: np.ndarray, binary_pred: np.ndarray, true_fine: np.ndarray) -> np.ndarray:
    """Same scoring rule as evaluate.compute_category_report: a true attack
    the binary model also caught, with a known true fine label."""
    has_true_label = np.array([c is not None for c in true_fine])
    return (true_label == 1) & (binary_pred == 1) & has_true_label


def compute_offline_category_report(
    true_for_table: np.ndarray, chosen: np.ndarray, raw_top: np.ndarray,
    table_mask: np.ndarray, attack_mask: np.ndarray,
) -> dict:
    """Per-class precision/recall/F1 over table_mask (true attacks caught
    plus false positives scored as BENIGN_CATEGORY, so precision reflects
    false alarms). Level-agnostic: pass fine labels or coarse categories.
    accuracy/raw_accuracy/coverage/macro_f1 are computed over attack_mask only
    (true attacks) — a false positive is never a label to "get right".
    BENIGN_CATEGORY's own row is always precision=recall=0, since it can never
    itself be predicted; its purpose is only to penalize other classes'
    precision. Classes with support < MIN_EVAL_SUPPORT are listed in
    "too_small"; macro_f1 averages every attack class, macro_f1_evaluable
    only the ones large enough to evaluate. See docs/evaluation.md."""
    n = int(attack_mask.sum())
    n_table = int(table_mask.sum())
    if n == 0:
        return {"n": 0, "n_table": n_table, "accuracy": None, "raw_accuracy": None, "coverage": None,
                "labels": [], "support": {}, "precision": {}, "recall": {}, "f1": {}, "macro_f1": None,
                "macro_f1_evaluable": None, "too_small": []}

    true_attack = true_for_table[attack_mask]
    chosen_attack = chosen[attack_mask]
    raw_attack = raw_top[attack_mask]

    true_table = true_for_table[table_mask]
    chosen_table = chosen[table_mask]

    # "Unknown" is never its own label row — it still counts as a miss, just not its own row.
    attack_labels = sorted((set(true_attack.tolist()) | set(chosen_attack.tolist())) - {"Unknown"})
    table_labels = sorted(set(attack_labels) | {classifier.BENIGN_CATEGORY})

    precision, recall, f1, support = precision_recall_fscore_support(
        true_table, chosen_table, labels=table_labels, zero_division=0,
    )
    f1_by_label = dict(zip(table_labels, f1))
    support_by_label = {label: int(s) for label, s in zip(table_labels, support)}
    too_small = [label for label in attack_labels if support_by_label[label] < MIN_EVAL_SUPPORT]
    evaluable = [label for label in attack_labels if label not in too_small]

    return {
        "n": n,
        "n_table": n_table,
        "accuracy": float((chosen_attack == true_attack).mean()),
        "raw_accuracy": float((raw_attack == true_attack).mean()),
        "coverage": float((chosen_attack != "Unknown").mean() * 100),
        "labels": table_labels,
        "support": support_by_label,
        "precision": {label: float(p) for label, p in zip(table_labels, precision)},
        "recall": {label: float(r) for label, r in zip(table_labels, recall)},
        "f1": f1_by_label,
        "macro_f1": float(np.mean([f1_by_label[label] for label in attack_labels])) if attack_labels else None,
        "macro_f1_evaluable": float(np.mean([f1_by_label[label] for label in evaluable])) if evaluable else None,
        "too_small": too_small,
    }


def print_offline_category_report(report: dict, level: str = "category") -> None:
    if report["n"] == 0:
        print(f"{level} classifier: no truly anomalous events with a known true label in this split")
        return
    macro = lambda v: f"{v:.4f}" if v is not None else "n/a"  # noqa: E731
    print(f"{level} classifier (n={report['n']} true attacks the binary model also caught, "
          f"n_table={report['n_table']} incl. binary false positives as Benign):")
    print(f"  accuracy (thresholded)={report['accuracy']:.4f}  "
          f"raw top-1 accuracy={report['raw_accuracy']:.4f}  "
          f"coverage={report['coverage']:.1f}%")
    print(f"  macro F1 (attack classes only)={macro(report['macro_f1'])}  "
          f"macro F1 excluding classes with support < {MIN_EVAL_SUPPORT}={macro(report['macro_f1_evaluable'])}")
    print(f"    {level:<28}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in report["labels"]:
        flag = "  too small to evaluate" if label in report["too_small"] else ""
        print(f"    {label:<28}{report['precision'][label]:>10.3f}{report['recall'][label]:>10.3f}"
              f"{report['f1'][label]:>10.3f}{report['support'][label]:>10d}{flag}")


def web_attack_confusion(true_fine: np.ndarray, chosen_label: np.ndarray, attack_mask: np.ndarray) -> dict:
    """True-vs-chosen-label counts for the Web Attack fine labels over the
    scored attacks, plus how many got a WRONG specific label (chosen is
    neither the truth nor "Unknown", i.e. the wrong answer cleared the
    threshold)."""
    web = attack_mask & np.array([isinstance(t, str) and t.startswith("Web Attack") for t in true_fine])
    rows = sorted(set(true_fine[web].tolist()))
    cols = sorted(set(chosen_label[web].tolist()) | set(rows))
    counts = {t: {c: int(((true_fine == t) & web & (chosen_label == c)).sum()) for c in cols} for t in rows}
    wrong = int((web & (chosen_label != true_fine) & (chosen_label != "Unknown")).sum())
    return {"rows": rows, "cols": cols, "counts": counts, "n": int(web.sum()), "n_wrong_confident": wrong}


def print_web_attack_confusion(report: dict, threshold: float) -> None:
    print(f"web attack confusion (true row vs predicted column, threshold={threshold:.2f}):")
    print("    " + f"{'true / predicted':<28}" + "".join(f"{c:>28}" for c in report["cols"]))
    for t in report["rows"]:
        print("    " + f"{t:<28}" + "".join(f"{report['counts'][t][c]:>28d}" for c in report["cols"]))
    print(f"  wrong fine label at confidence >= {threshold:.2f}: {report['n_wrong_confident']}/{report['n']} "
          "web-attack flows")


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
    category_artifact: dict, X: np.ndarray, true_fine: np.ndarray, true_category: np.ndarray,
    binary_pred: np.ndarray, true_label: np.ndarray,
    thresholds: Sequence[float] = CATEGORY_THRESHOLD_SWEEP,
) -> list:
    """One row per threshold: coarse-category accuracy (the quantity the
    recommendation rule uses), fine-label accuracy, % Unknown at each level,
    and false-positive categorisation rate, recomputed at each candidate
    threshold. Callers pass a VALIDATION split (never TEST) so selecting a
    threshold never tunes on test data."""
    mask = scored_attack_mask(true_label, binary_pred, true_fine)
    n_scored = int(mask.sum())
    probs = batch_category_probabilities(category_artifact, X)
    rows = []
    for threshold in thresholds:
        decided = decide_batch(probs, threshold)
        stats = {"category_accuracy": None, "pct_unknown": None, "label_accuracy": None, "pct_label_unknown": None}
        if n_scored:
            chosen_category = decided["category"][mask]
            chosen_label = decided["label"][mask]
            stats = {
                "category_accuracy": float((chosen_category == true_category[mask]).mean()),
                "pct_unknown": float((chosen_category == "Unknown").mean() * 100),
                "label_accuracy": float((chosen_label == true_fine[mask]).mean()),
                "pct_label_unknown": float((chosen_label == "Unknown").mean() * 100),
            }
        fp_report = compute_offline_false_positive_report(true_label, binary_pred, decided["category"])
        rows.append({
            "threshold": threshold,
            "n_scored": n_scored,
            **stats,
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
    print(f"    {'threshold':>9}{'n_scored':>10}{'cat_acc':>9}{'%cat_unk':>10}{'label_acc':>11}"
          f"{'%lbl_unk':>10}{'n_fp':>7}{'fp_rate':>9}")
    fmt = lambda v, spec: format(v, spec) if v is not None else "n/a"  # noqa: E731
    for row in sweep_rows:
        print(f"    {row['threshold']:>9.2f}{row['n_scored']:>10d}{fmt(row['category_accuracy'], '.3f'):>9}"
              f"{fmt(row['pct_unknown'], '.1f'):>10}{fmt(row['label_accuracy'], '.3f'):>11}"
              f"{fmt(row['pct_label_unknown'], '.1f'):>10}{row['n_fp']:>7d}"
              f"{fmt(row['fp_categorisation_rate'], '.1f'):>9}")


def save_threshold_sweep_csv(path: Path, sweep_rows: list) -> None:
    if not sweep_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)


def run_offline_evaluation(
    threshold: float, min_category_accuracy: float = DEFAULT_MIN_CATEGORY_ACCURACY,
) -> None:
    """Evaluate both classifiers over the whole TEST split at
    `threshold`, then sweep CATEGORY_CONFIDENCE_THRESHOLD on a
    VALIDATION split and print a recommendation. Saves the sweep to
    results/category_threshold_sweep.csv. Never modifies any argument or
    training artifact."""
    binary_artifact = classifier.load_artifact(classifier.DEFAULT_BINARY_MODEL_PATH)
    category_artifact = classifier.load_artifact(classifier.DEFAULT_CATEGORY_MODEL_PATH)

    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    print(f"=== Offline evaluation (full test split, n={len(test_df)}, no LLM) ===")

    def truth(frame):
        """(X, binary y, fine labels, coarse categories) for a split."""
        raw = frame[data.LABEL_COL].to_numpy()
        return (
            frame[feature_names].to_numpy(), data.binarize_labels(frame[data.LABEL_COL]),
            np.array([data.map_cicids_label_to_fine(r) for r in raw], dtype=object),
            np.array([data.map_cicids_label_to_category(r) for r in raw], dtype=object),
        )

    X_test, y_test, true_fine_test, true_category_test = truth(test_df)

    binary_pred_test, binary_proba_test = batch_predict_binary(binary_artifact, X_test)
    print_binary_report(compute_binary_metrics(y_test, binary_pred_test, binary_proba_test))

    attack_mask_test = scored_attack_mask(y_test, binary_pred_test, true_fine_test)
    fp_mask_test = (y_test == 0) & (binary_pred_test == 1)
    table_mask_test = attack_mask_test | fp_mask_test
    fine_for_table = np.where(fp_mask_test, classifier.BENIGN_CATEGORY, true_fine_test)
    category_for_table = np.where(fp_mask_test, classifier.BENIGN_CATEGORY, true_category_test)
    test_probs = batch_category_probabilities(category_artifact, X_test)

    def report_test(threshold: float) -> dict:
        decided = decide_batch(test_probs, threshold)
        print_offline_category_report(
            compute_offline_category_report(fine_for_table, decided["label"], decided["raw_label"],
                                            table_mask_test, attack_mask_test),
            level="fine label",
        )
        print_offline_category_report(
            compute_offline_category_report(category_for_table, decided["category"], decided["raw_category"],
                                            table_mask_test, attack_mask_test),
            level="coarse category",
        )
        print_false_positive_report(
            compute_offline_false_positive_report(y_test, binary_pred_test, decided["category"])
        )
        print_web_attack_confusion(web_attack_confusion(true_fine_test, decided["label"], attack_mask_test), threshold)
        return decided

    from src.detection.subagent import DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD as default_threshold

    used = f"{threshold:.2f}" + (" (default)" if threshold == default_threshold else "")
    print(f"(category decisions at threshold={used}; "
          "coverage = % of true attacks given a non-Unknown answer at that level)")
    decided_default = report_test(threshold)
    print(f"model disagreement count (binary anomalous, category top vote Benign): "
          f"{compute_model_disagreement_count(binary_pred_test, decided_default['disagreement'])}")

    print()
    _, validation_df = data.split_validation(train_df)
    X_val, y_val, true_fine_val, true_category_val = truth(validation_df)
    binary_pred_val, _ = batch_predict_binary(binary_artifact, X_val)

    sweep_rows = sweep_category_thresholds(
        category_artifact, X_val, true_fine_val, true_category_val, binary_pred_val, y_val,
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

    print(f"Recommendation rule: {describe_recommendation_rule(recommended)} (coarse category accuracy)")
    print(f"Recommended CATEGORY_CONFIDENCE_THRESHOLD={recommended['threshold']:.2f} "
          f"(validation category accuracy={recommended['category_accuracy']:.3f}). "
          f"Current default is {default_threshold:.2f} — NOT changed automatically.")

    print(f"Recommended threshold's results on TEST (threshold={recommended['threshold']:.2f}):")
    report_test(recommended["threshold"])
