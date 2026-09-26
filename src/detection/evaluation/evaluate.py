"""
Ablation: RF-only vs RF+LLM (DetectionManager with use_llm=False vs True) on
a held-out sample. The point is to show the LLM layer never changes the
decision (accuracy/F1 should match exactly) or the category (code decides
that too, deterministically — see subagent.py._choose_category) while
reporting the LLM's explanation validity/latency cost.

Uses src/detection/training/data.py's split_train_test — the SAME train/test
split train_binary.py and train_category.py use — so every sampled event is
guaranteed to have been in neither model's training data.

Needs both trained artifacts (see train.py) and, for the +LLM arm, whatever
DETECTION_LLM_MODEL points at (default: `ollama pull qwen2.5:7b` first, then
`ollama serve`). If warmup fails (bad model id/API key, provider down), the
+LLM arm is skipped and only the RF-only results are printed/saved.

Usage:
    python -m src.detection.evaluation.evaluate [--sample-size 200] [--random-state 42]
                                      [--sampling random|borderline] [--llm-timeout 20]
                                      [--llm-delay 0]

    python -m src.detection.evaluation.evaluate --offline [--category-threshold 0.9] [--min-category-accuracy 0.99]
        Offline, no-LLM batch evaluation over the entire test split, plus a
        CATEGORY_CONFIDENCE_THRESHOLD sweep on a validation split carved from
        train — see evaluation/offline.py. Ignores every other flag above.
"""
import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.detection import classifier
from src.detection.evaluation import offline
from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION, resolve_llm_mode, resolve_llm_model_id
from src.detection.manager import DetectionManager
from src.detection.subagent import DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD
from src.detection.training import data
from src.shared.schemas import TrafficEvent

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BORDERLINE_LOW, BORDERLINE_HIGH = 0.3, 0.7  # slightly wider than the subagent's band, for sampling


def build_holdout_sample(sample_size: int, random_state: int, artifact: dict, sampling: str = "random"):
    """Draws `sample_size` rows from the TEST half of data.split_train_test —
    never the train half either model was fit on.

    sampling="random": a plain stratified-by-label random sample.
    sampling="borderline": the same, but up to a quarter of the sample is
    swapped for events in the borderline p_anomalous band, so the sample
    actually exercises the LLM layer's trigger condition rather than being
    all clear-cut cases — useful when you specifically want to evaluate the
    LLM path, at the cost of the sample no longer being representative of
    the real class/confidence distribution. `random_state` only controls
    this sampling step, not the underlying train/test split (which is fixed
    — see data.split_train_test).

    Returns (events, labels, true_categories) — true_categories is the
    original multiclass CICIDS2017 Label mapped via
    data.map_cicids_label_to_category (None for BENIGN/unrecognized).
    """
    df = data.load_clean_dataframe()
    _, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    X_test = test_df[feature_names].to_numpy()
    y_test = data.binarize_labels(test_df[data.LABEL_COL])
    raw_labels_test = test_df[data.LABEL_COL].to_numpy()

    if sampling == "borderline":
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
    else:
        _, idx = train_test_split(
            np.arange(len(y_test)), test_size=sample_size, stratify=y_test, random_state=random_state,
        )

    # Plain Python floats, not numpy scalars — these end up serialized over the MCP
    # wire (register_event) when the LLM layer runs, which numpy floats can't be.
    events = [
        TrafficEvent(features={name: float(v) for name, v in zip(feature_names, X_test[i])})
        for i in idx
    ]
    true_categories = [data.map_cicids_label_to_category(raw_labels_test[i]) for i in idx]
    return events, y_test[idx], true_categories


def _fallback_reason(trace) -> Optional[str]:
    for step in trace:
        if step.action in FALLBACK_REASON_BY_ACTION:
            return FALLBACK_REASON_BY_ACTION[step.action]
    return None


def _token_usage(trace):
    for step in trace:
        if step.action == "llm_token_usage":
            return step.tool_input.get("prompt_tokens"), step.tool_input.get("completion_tokens")
    return None, None


def _llm_mode(trace) -> Optional[str]:
    for step in trace:
        if step.action == "llm_dispatch":
            return step.tool_input.get("mode")
    return None


def _category_decision(trace):
    """The classifier's raw top-1 category + probability, from whichever
    step _choose_category logged for a truly anomalous event —
    "category_decision" (ordinary threshold decision) or
    "model_disagreement" (category model's top vote was Benign, contradicting
    the binary model) — independent of use_llm/llm_invoked entirely, since
    the category decision is deterministic and made by code either way."""
    for step in trace:
        if step.action in ("category_decision", "model_disagreement"):
            ti = step.tool_input or {}
            return ti.get("raw_top_category"), ti.get("raw_top_probability")
    return None, None


def _model_disagreement(trace) -> bool:
    """True when _choose_category found the category model's top vote was
    Benign for an event the binary model already called anomalous — a
    disagreement between the two models, distinct from an ordinary
    below-threshold "Unknown" (see subagent.py._choose_category)."""
    return any(step.action == "model_disagreement" for step in trace)


def _llm_diagnostics(trace):
    """Raw text of every step that failed to parse as a tool call, plus the
    forced final-answer text if smolagents exhausted max_steps without the
    model ever calling final_answer — see
    DetectionSubagent._log_agent_memory_diagnostics. Lets a CSV row show
    exactly what a small local model wrote, not just whether it passed."""
    step_failures = [step.observation for step in trace if step.action == "llm_step_failed"]
    forced_final_answer = next(
        (step.observation for step in trace if step.action == "llm_forced_final_answer"), None,
    )
    return " || ".join(step_failures) if step_failures else None, forced_final_answer


def compute_category_report(rows: list) -> dict:
    """Category accuracy over ALL truly anomalous events in the sample —
    NOT only ones where the LLM ran. The category decision
    (subagent.py._choose_category) is a deterministic classifier.
    predict_attack_category call code makes for every anomalous event, same
    principle as is_anomalous/confidence, so it's available whether or not
    use_llm is even True; the RF-only arm reports it exactly like the +LLM
    arm does.

    Reports both:
    - "accuracy": the code-chosen category (top class if its probability
      clears CATEGORY_CONFIDENCE_THRESHOLD, else "Unknown")
    - "raw_accuracy": the classifier's raw top-1 class, no threshold applied
    plus a naive "always guess the most common true category" baseline over
    the same events, and a confusion table for the code-chosen category.

    Scored only over rows with true_label == 1 (an actual attack the binary
    model also called anomalous) and a known true_category (excludes
    BENIGN/unrecognized labels) — a row the binary model got wrong isn't a
    category-accuracy question, that's what the binary accuracy/F1 are for.
    A false negative (true_label == 1 but the binary model called it benign)
    never had _choose_category run at all, so its "category" is None, not
    "Unknown" — excluded here by the same "model also called anomalous" rule,
    not folded into "Unknown" (which means something different: the category
    model *did* run but wasn't confident enough).
    """
    scored = [
        r for r in rows
        if r["true_label"] == 1 and r["true_category"] is not None and r["category"] is not None
    ]
    n = len(scored)
    if n == 0:
        return {"n": 0, "accuracy": None, "raw_accuracy": None, "pct_unknown": None,
                "confusion": {}, "baseline_category": None, "baseline_accuracy": None}

    correct = sum(1 for r in scored if r["category"] == r["true_category"])
    raw_correct = sum(1 for r in scored if r["raw_top_category"] == r["true_category"])
    unknown_count = sum(1 for r in scored if r["category"] == "Unknown")

    confusion: dict = {}
    for r in scored:
        confusion.setdefault(r["true_category"], Counter())[r["category"]] += 1

    baseline_category = Counter(r["true_category"] for r in scored).most_common(1)[0][0]
    baseline_correct = sum(1 for r in scored if r["true_category"] == baseline_category)

    return {
        "n": n,
        "accuracy": correct / n,
        "raw_accuracy": raw_correct / n,
        "pct_unknown": unknown_count / n * 100,
        "confusion": {true_cat: dict(counts) for true_cat, counts in confusion.items()},
        "baseline_category": baseline_category,
        "baseline_accuracy": baseline_correct / n,
    }


def print_category_report(report: dict) -> None:
    if report["n"] == 0:
        print("category accuracy: no truly anomalous events with a known true category in this sample")
        return

    print(f"category accuracy (n={report['n']} truly anomalous events with a known true category):")
    print(f"  code-chosen (thresholded): {report['accuracy']:.3f}")
    print(f"  classifier raw top-1:      {report['raw_accuracy']:.3f}")
    print(f"  % predicted Unknown (code-chosen): {report['pct_unknown']:.1f}")
    print(f"  trivial baseline (\"always guess {report['baseline_category']}\"): "
          f"{report['baseline_accuracy']:.3f}")

    predicted_categories = sorted({p for counts in report["confusion"].values() for p in counts})
    print("  confusion (rows=true category, columns=code-chosen predicted):")
    true_vs_predicted_label = "true vs predicted"
    header = "    " + f"{true_vs_predicted_label:<16}" + "".join(f"{p:>12}" for p in predicted_categories)
    print(header)
    for true_cat in sorted(report["confusion"]):
        counts = report["confusion"][true_cat]
        row = "    " + f"{true_cat:<16}" + "".join(f"{counts.get(p, 0):>12}" for p in predicted_categories)
        print(row)


def compute_false_positive_categorization(rows: list) -> dict:
    """Among events the binary model wrongly flagged anomalous
    (true_label == 0, predicted == 1), how often the category decision still
    handed out a specific attack category rather than "Unknown".

    Before train_category.py trained on a Benign class, this was exactly the
    gap a real run surfaced: 5 benign events the binary model false-positived
    all got a confident attack category (Botnet 0.97-1.0 x4, DoS 0.99),
    because the category model had no benign option to vote for and
    compute_category_report never scores benign ground truth at all — this
    metric is the one that would have caught it. Should be ~0% once the
    Benign class (and model_disagreement) are in place.
    """
    false_positives = [r for r in rows if r["true_label"] == 0 and r["predicted"] == 1]
    n = len(false_positives)
    if n == 0:
        return {"n": 0, "pct_given_specific_category": None}
    given_specific_category = sum(
        1 for r in false_positives if r["category"] not in (None, "Unknown")
    )
    return {"n": n, "pct_given_specific_category": given_specific_category / n * 100}


def print_false_positive_report(report: dict) -> None:
    if report["n"] == 0:
        print("false-positive categorisation: no binary-model false positives in this sample")
        return
    print(f"false-positive categorisation (n={report['n']} events wrongly flagged anomalous by "
          "the binary model):")
    print(f"  % given a specific attack category: {report['pct_given_specific_category']:.1f}")


def run_arm(manager: DetectionManager, events, labels, true_categories, llm_delay_seconds: float = 0.0):
    """Runs `manager` over every event, timing each call and recording
    whether the LLM layer fired, whether its output validated, and (when it
    didn't) why. "Fired" is counted from the main-thread "llm_dispatch" step
    (logged unconditionally before the timed call, not the buffered
    "llm_layer_start" that a timed-out call never gets to merge into the
    trace) — otherwise a timeout would silently undercount invocations.

    true_categories is the per-event ground-truth attack category from
    build_holdout_sample (None for benign/unrecognized), aligned with
    events/labels — used for compute_category_report and recorded as each
    row's true_category. category/raw_top_category are read from the trace's
    "category_decision" step, which _choose_category logs for every truly
    anomalous event regardless of use_llm.

    llm_delay_seconds, if set, sleeps after every event that dispatched to
    the LLM (not after RF-only events) — for free-tier rate limits."""
    predictions, latencies_ms = [], []
    llm_invoked = llm_validated = 0
    categories, fallback_reasons, rows = [], [], []

    for i, (event, label, true_category) in enumerate(zip(events, labels, true_categories)):
        start = time.perf_counter()
        result = manager.run(event)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        predictions.append(int(result.is_anomalous))

        trace_actions = [step.action for step in result.trace]
        invoked = "llm_dispatch" in trace_actions
        # "valid" = clean pass OR salvaged (extract_json_object recovered a usable answer) —
        # both mean detector_notes carries the LLM's real explanation, not a template.
        validated = "llm_validation_passed" in trace_actions or "llm_validation_salvaged" in trace_actions
        llm_invoked += int(invoked)
        llm_validated += int(validated)

        category = None
        if result.detector_notes and result.detector_notes.startswith("[category="):
            category = result.detector_notes.split("]", 1)[0].removeprefix("[category=")
        categories.append(category)
        raw_top_category, raw_top_probability = _category_decision(result.trace)

        fallback_reason = _fallback_reason(result.trace)
        fallback_reasons.append(fallback_reason)
        prompt_tokens, completion_tokens = _token_usage(result.trace)
        step_failures, forced_final_answer = _llm_diagnostics(result.trace)
        mode = _llm_mode(result.trace)

        disagreement = _model_disagreement(result.trace)

        rows.append({
            "true_label": int(label), "predicted": int(result.is_anomalous),
            "confidence": result.confidence, "llm_invoked": invoked, "llm_validated": validated,
            "llm_mode": mode, "category": category, "true_category": true_category,
            "raw_top_category": raw_top_category, "raw_top_probability": raw_top_probability,
            "model_disagreement": disagreement,
            "fallback_reason": fallback_reason,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "latency_ms": latencies_ms[-1],
            "llm_step_failures": step_failures, "llm_forced_final_answer": forced_final_answer,
        })

        if invoked and llm_delay_seconds > 0 and i < len(events) - 1:
            time.sleep(llm_delay_seconds)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "pct_llm_invoked": llm_invoked / len(events) * 100,
        "pct_llm_validated": (llm_validated / llm_invoked * 100) if llm_invoked else 0.0,
        "category_distribution": {c: categories.count(c) for c in set(categories)},
        "fallback_reason_counts": dict(Counter(r for r in fallback_reasons if r)),
        "category_report": compute_category_report(rows),
        "false_positive_report": compute_false_positive_categorization(rows),
        "model_disagreement_count": sum(1 for r in rows if r["model_disagreement"]),
        "rows": rows,
    }


def print_summary(rf_only: dict, rf_llm: Optional[dict], sampling: str):
    print(f"sampling: {sampling}")
    if rf_llm is None:
        print(f"{'metric':<22}{'RF-only':>15}")
        for key, label in [
            ("accuracy", "accuracy"), ("f1", "f1"),
            ("mean_latency_ms", "mean latency ms"), ("p95_latency_ms", "p95 latency ms"),
        ]:
            print(f"{label:<22}{rf_only[key]:>15.4f}")
        print("(+LLM arm skipped — warmup failed, see above)")
    else:
        print(f"{'metric':<22}{'RF-only':>15}{'RF+LLM':>15}")
        for key, label in [
            ("accuracy", "accuracy"), ("f1", "f1"),
            ("mean_latency_ms", "mean latency ms"), ("p95_latency_ms", "p95 latency ms"),
        ]:
            print(f"{label:<22}{rf_only[key]:>15.4f}{rf_llm[key]:>15.4f}")
        print(f"{'% events -> LLM':<22}{'-':>15}{rf_llm['pct_llm_invoked']:>15.1f}")
        print(f"{'% LLM valid (+salvaged)':<22}{'-':>15}{rf_llm['pct_llm_validated']:>15.1f}")
        print(f"category distribution (RF+LLM detector_notes): {rf_llm['category_distribution']}")
        print(f"fallback reason counts (RF+LLM): {rf_llm['fallback_reason_counts']}")

    # The category decision is deterministic and independent of use_llm (see
    # subagent.py._choose_category), so this is identical whether or not the LLM ran —
    # report it once, from the arm that's always available.
    print_category_report(rf_only["category_report"])
    print_false_positive_report(rf_only["false_positive_report"])
    print(f"model disagreement count (binary anomalous, category top vote Benign): "
          f"{rf_only['model_disagreement_count']}")


def save_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                         help="Offline, no-LLM batch evaluation over the ENTIRE test split "
                              "(binary + category classifier metrics, false-positive "
                              "categorisation, model disagreement), plus a "
                              "CATEGORY_CONFIDENCE_THRESHOLD sweep on a validation split carved "
                              "from train. Ignores every other flag below.")
    parser.add_argument("--category-threshold", type=float, default=DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD,
                         help="Only with --offline: threshold used for the main test-split report "
                              f"(default: subagent.DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD = "
                              f"{DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD}). The sweep itself always "
                              "covers offline.CATEGORY_THRESHOLD_SWEEP regardless of this flag.")
    parser.add_argument("--min-category-accuracy", type=float,
                         default=offline.DEFAULT_MIN_CATEGORY_ACCURACY,
                         help="Only with --offline: recommend the HIGHEST swept threshold whose "
                              "validation category accuracy (on true attacks) is >= this value "
                              f"(default {offline.DEFAULT_MIN_CATEGORY_ACCURACY}).")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42,
                         help="Controls only the held-out SAMPLING step (which rows of the fixed "
                              "test split get drawn), not the train/test split itself.")
    parser.add_argument("--sampling", choices=["random", "borderline"], default="random")
    parser.add_argument("--llm-timeout", type=float, default=None,
                         help="Per-event LLM timeout in seconds. Defaults to DETECTION_LLM_TIMEOUT "
                              "or subagent.py's built-in default if unset.")
    parser.add_argument("--llm-delay", type=float, default=0.0,
                         help="Seconds to sleep after each event dispatched to the LLM — for "
                              "free-tier hosted-provider rate limits.")
    args = parser.parse_args()

    if args.offline:
        offline.run_offline_evaluation(args.category_threshold, args.min_category_accuracy)
        return

    print(f"DETECTION_LLM_MODEL={resolve_llm_model_id()}  DETECTION_LLM_MODE={resolve_llm_mode()}")

    artifact = classifier.load_artifact(classifier.DEFAULT_BINARY_MODEL_PATH)
    events, labels, true_categories = build_holdout_sample(
        args.sample_size, args.random_state, artifact, args.sampling,
    )
    print(f"Evaluating on {len(events)} held-out events ({args.sampling} sampling), "
          f"({int(labels.sum())} anomalous, {len(labels) - int(labels.sum())} benign).")

    with DetectionManager(use_llm=False) as rf_only_manager:
        rf_only = run_arm(rf_only_manager, events, labels, true_categories)

    # One persistent MCP connection (and one loaded model artifact) for the whole
    # +LLM arm, not one per event — close() shuts down its subprocess when done.
    rf_llm = None
    with DetectionManager(use_llm=True, llm_timeout_seconds=args.llm_timeout) as rf_llm_manager:
        warmup_result = rf_llm_manager.warmup()
        if not warmup_result["ok"]:
            print(f"LLM warmup failed: {warmup_result['reason']}. Check DETECTION_LLM_MODEL / "
                  "API key / provider status.")
        else:
            print(f"LLM warmup: {warmup_result['elapsed_seconds']:.2f}s (connection + first "
                  "inference, outside any per-event timeout)")
            rf_llm = run_arm(rf_llm_manager, events, labels, true_categories, llm_delay_seconds=args.llm_delay)

    print_summary(rf_only, rf_llm, args.sampling)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_csv(RESULTS_DIR / f"rf_only_{args.sampling}.csv", rf_only["rows"])
    if rf_llm is not None:
        save_csv(RESULTS_DIR / f"rf_llm_{args.sampling}.csv", rf_llm["rows"])
    print(f"Saved per-event results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
