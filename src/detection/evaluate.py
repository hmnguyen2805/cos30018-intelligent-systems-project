"""
Ablation: RF-only vs RF+LLM (DetectionManager with use_llm=False vs True) on
a held-out sample. The point is to show the LLM layer never changes the
decision (accuracy/F1 should match exactly) while reporting its latency and
validation cost, and to eyeball the category distribution it proposes.

Needs a trained artifact (see train.py) and, for the +LLM arm, whatever
DETECTION_LLM_MODEL points at (default: `ollama pull qwen2.5:7b` first, then
`ollama serve`).

Usage:
    python -m src.detection.evaluate [--sample-size 200] [--random-state 42]
"""
import argparse
import csv
import os
import time
from pathlib import Path

import kagglehub
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.detection import classifier, train
from src.detection.manager import DetectionManager
from src.shared.schemas import TrafficEvent

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BORDERLINE_LOW, BORDERLINE_HIGH = 0.3, 0.7  # slightly wider than the subagent's band, for sampling


def build_holdout_sample(sample_size: int, random_state: int, artifact: dict):
    """Reproduces train.py's train/test split, then takes a stratified
    sample of `sample_size` test rows — with up to a quarter of them chosen
    from the borderline p_anomalous band, so the sample actually exercises
    the LLM layer's trigger condition rather than being all clear-cut cases."""
    dataset_dir = kagglehub.dataset_download(train.KAGGLE_DATASET)
    df = train.clean_dataframe(train.load_dataset(dataset_dir))
    feature_names = train.select_feature_names(df, label_col=train.LABEL_COL)
    X = df[feature_names].to_numpy()
    y = train.binarize_labels(df[train.LABEL_COL])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    model = artifact["model"]
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    is_borderline = (proba >= BORDERLINE_LOW) & (proba <= BORDERLINE_HIGH)

    rng = np.random.RandomState(random_state)
    n_borderline = min(sample_size // 4, int(is_borderline.sum()))
    borderline_idx = rng.choice(np.flatnonzero(is_borderline), size=n_borderline, replace=False)

    remaining_pool = np.setdiff1d(np.arange(len(y_test)), borderline_idx)
    n_remaining = sample_size - n_borderline
    _, remaining_idx = train_test_split(
        remaining_pool, test_size=n_remaining, stratify=y_test[remaining_pool], random_state=random_state,
    )

    idx = np.concatenate([borderline_idx, remaining_idx])
    events = [TrafficEvent(features=dict(zip(feature_names, X_test[i]))) for i in idx]
    return events, y_test[idx]


def run_arm(manager: DetectionManager, events, labels):
    """Runs `manager` over every event, timing each call and recording
    whether the LLM layer fired and, if so, whether its output validated."""
    predictions, latencies_ms = [], []
    llm_invoked = llm_validated = 0
    categories, rows = [], []

    for event, label in zip(events, labels):
        start = time.perf_counter()
        result = manager.run(event)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        predictions.append(int(result.is_anomalous))

        trace_actions = [step.action for step in result.trace]
        invoked = "llm_layer_start" in trace_actions
        validated = "llm_validation_passed" in trace_actions
        llm_invoked += int(invoked)
        llm_validated += int(validated)

        category = None
        if result.detector_notes and result.detector_notes.startswith("[category="):
            category = result.detector_notes.split("]", 1)[0].removeprefix("[category=")
        categories.append(category)

        rows.append({
            "true_label": int(label), "predicted": int(result.is_anomalous),
            "confidence": result.confidence, "llm_invoked": invoked, "llm_validated": validated,
            "category": category, "latency_ms": latencies_ms[-1],
        })

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "pct_llm_invoked": llm_invoked / len(events) * 100,
        "pct_llm_validated": (llm_validated / llm_invoked * 100) if llm_invoked else 0.0,
        "category_distribution": {c: categories.count(c) for c in set(categories)},
        "rows": rows,
    }


def print_summary(rf_only: dict, rf_llm: dict):
    print(f"{'metric':<22}{'RF-only':>15}{'RF+LLM':>15}")
    for key, label in [
        ("accuracy", "accuracy"), ("f1", "f1"),
        ("mean_latency_ms", "mean latency ms"), ("p95_latency_ms", "p95 latency ms"),
    ]:
        print(f"{label:<22}{rf_only[key]:>15.4f}{rf_llm[key]:>15.4f}")
    print(f"{'% events -> LLM':<22}{'-':>15}{rf_llm['pct_llm_invoked']:>15.1f}")
    print(f"{'% LLM outputs valid':<22}{'-':>15}{rf_llm['pct_llm_validated']:>15.1f}")
    print(f"category distribution (RF+LLM): {rf_llm['category_distribution']}")


def save_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    artifact = classifier.load_artifact(classifier.DEFAULT_MODEL_PATH)
    events, labels = build_holdout_sample(args.sample_size, args.random_state, artifact)
    print(f"Evaluating on {len(events)} held-out events "
          f"({int(labels.sum())} anomalous, {len(labels) - int(labels.sum())} benign).")

    with DetectionManager(use_llm=False) as rf_only_manager:
        rf_only = run_arm(rf_only_manager, events, labels)

    # One persistent MCP connection (and one loaded model artifact) for the whole
    # +LLM arm, not one per event — close() shuts down its subprocess when done.
    with DetectionManager(use_llm=True) as rf_llm_manager:
        rf_llm = run_arm(rf_llm_manager, events, labels)

    print_summary(rf_only, rf_llm)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_csv(RESULTS_DIR / "rf_only.csv", rf_only["rows"])
    save_csv(RESULTS_DIR / "rf_llm.csv", rf_llm["rows"])
    print(f"Saved per-event results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
