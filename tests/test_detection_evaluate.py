"""
Unit tests for src.detection.evaluate's pure per-event reporting helpers —
_fallback_reason, _token_usage, _llm_diagnostics, _llm_mode, _category_decision,
and run_arm's "valid" definition (clean pass OR salvaged). These operate on a
DetectionResult trace (a list of TraceStep), so a fake manager/event stands in
for the real pipeline — no model, no MCP, no network.

map_cicids_label_to_category has moved to src.detection.training.data — see
tests/test_detection_training_data.py.
"""
from unittest.mock import MagicMock

import pytest

from src.detection import evaluate
from src.shared.schemas import DetectionResult, TraceStep, TrafficEvent


def make_trace(*steps):
    return [
        TraceStep(step_number=i + 1, action=action, tool_input=tool_input, observation=observation)
        for i, (action, tool_input, observation) in enumerate(steps)
    ]


def test_fallback_reason_returns_none_when_llm_never_invoked():
    assert evaluate._fallback_reason(make_trace(("call_classifier", None, None))) is None


def test_fallback_reason_maps_known_action_to_its_reason():
    trace = make_trace(("llm_dispatch", {}, None), ("llm_timeout", None, None))
    assert evaluate._fallback_reason(trace) == "timeout"


def test_fallback_reason_salvaged_is_reported_too():
    trace = make_trace(("llm_dispatch", {}, None), ("llm_validation_salvaged", None, "explanation"))
    assert evaluate._fallback_reason(trace) == "salvaged"


def test_token_usage_extracts_prompt_and_completion_tokens():
    trace = make_trace(("llm_token_usage", {"prompt_tokens": 12, "completion_tokens": 34}, None))
    assert evaluate._token_usage(trace) == (12, 34)


def test_token_usage_returns_none_pair_when_absent():
    assert evaluate._token_usage(make_trace(("call_classifier", None, None))) == (None, None)


def test_llm_diagnostics_collects_step_failures_and_forced_final_answer():
    trace = make_trace(
        ("llm_step_failed", {"step_number": 2}, "error=x; raw_text=hi"),
        ("llm_step_failed", {"step_number": 3}, "error=y; raw_text=None"),
        ("llm_forced_final_answer", None, "here is my answer"),
    )
    step_failures, forced = evaluate._llm_diagnostics(trace)
    assert step_failures == "error=x; raw_text=hi || error=y; raw_text=None"
    assert forced == "here is my answer"


def test_llm_diagnostics_returns_none_when_nothing_to_report():
    step_failures, forced = evaluate._llm_diagnostics(make_trace(("llm_validation_passed", None, None)))
    assert step_failures is None
    assert forced is None


def test_llm_mode_reads_it_from_the_dispatch_step():
    trace = make_trace(("llm_dispatch", {"mode": "single_shot"}, None))
    assert evaluate._llm_mode(trace) == "single_shot"


def test_llm_mode_returns_none_when_llm_never_dispatched():
    assert evaluate._llm_mode(make_trace(("call_classifier", None, None))) is None


def test_category_decision_reads_raw_top_category_and_probability():
    trace = make_trace((
        "category_decision",
        {"chosen_category": "Unknown", "raw_top_category": "DDoS", "raw_top_probability": 0.42},
        None,
    ))
    assert evaluate._category_decision(trace) == ("DDoS", 0.42)


def test_category_decision_returns_none_pair_when_absent():
    assert evaluate._category_decision(make_trace(("call_classifier", None, None))) == (None, None)


def _make_event():
    return TrafficEvent(features={"duration": 1.0})


def test_run_arm_counts_salvaged_output_as_valid():
    """run_arm's `validated` (and hence pct_llm_validated) must count a
    salvaged answer as valid, not just a clean llm_validation_passed —
    otherwise a real salvage success would be reported as if it failed."""
    trace = make_trace(
        ("llm_dispatch", {"mode": "agent"}, None),
        ("llm_validation_salvaged", None, "explanation, tools_used=[]"),
    )
    result = DetectionResult(event=_make_event(), is_anomalous=True, confidence=0.9,
                              detector_notes="[category=PortScan] x", trace=trace)
    manager = MagicMock()
    manager.run.return_value = result

    summary = evaluate.run_arm(manager, [_make_event()], [1], [None])

    assert summary["pct_llm_validated"] == 100.0
    assert summary["rows"][0]["llm_validated"] is True


# --- compute_category_report -------------------------------------------------
# Scored over ALL truly anomalous events (true_label == 1 and a known true_category) —
# not gated on llm_invoked/llm_validated at all, since the category decision is now a
# deterministic classifier call code makes for every anomalous event regardless of use_llm.

def _row(true_label=1, category=None, raw_top_category=None, true_category=None):
    return {"true_label": true_label, "category": category,
            "raw_top_category": raw_top_category, "true_category": true_category}


def test_category_report_excludes_benign_predictions():
    # true_label=0: the binary model got this one wrong (or it's genuinely benign) — not a
    # category-accuracy question either way.
    rows = [_row(true_label=0, category="DDoS", true_category="DDoS")]
    report = evaluate.compute_category_report(rows)
    assert report["n"] == 0
    assert report["accuracy"] is None


def test_category_report_excludes_events_with_no_true_category():
    # BENIGN/unrecognized traffic (true_category is None) isn't scoreable.
    rows = [_row(category="DDoS", true_category=None)]
    report = evaluate.compute_category_report(rows)
    assert report["n"] == 0


def test_category_report_excludes_false_negatives_of_the_binary_model():
    # true_label=1 (a real attack) but category=None means the binary model called this
    # event benign, so _choose_category never ran (it's gated on is_anomalous) — this must
    # not be scored as a category miss, nor crash the confusion table on a None key.
    rows = [
        _row(category=None, true_category="DDoS"),          # false negative: excluded
        _row(category="DDoS", true_category="DDoS"),         # correctly caught: scored
    ]
    report = evaluate.compute_category_report(rows)
    assert report["n"] == 1
    assert report["accuracy"] == 1.0
    assert None not in report["confusion"].get("DDoS", {})


def test_category_report_does_not_require_llm_invocation():
    # No llm_invoked/llm_validated keys at all — RF-only rows never have them, and the
    # category report must still work (this is exactly item 5's ask: report over ALL
    # truly anomalous events, not only ones where the LLM ran).
    rows = [_row(category="DDoS", true_category="DDoS")]
    report = evaluate.compute_category_report(rows)
    assert report["n"] == 1
    assert report["accuracy"] == 1.0


def test_category_report_computes_accuracy_and_pct_unknown():
    rows = [
        _row(category="DDoS", true_category="DDoS"),      # correct
        _row(category="PortScan", true_category="DDoS"),  # wrong
        _row(category="Unknown", true_category="DoS"),    # wrong, and a punt
    ]
    report = evaluate.compute_category_report(rows)
    assert report["n"] == 3
    assert report["accuracy"] == 1 / 3
    assert report["pct_unknown"] == pytest.approx(100 / 3)


def test_category_report_raw_accuracy_differs_from_thresholded_accuracy():
    # code-chosen category is "Unknown" (below threshold) but the classifier's raw top-1
    # would have been correct — accuracy and raw_accuracy must diverge here.
    rows = [_row(category="Unknown", raw_top_category="DDoS", true_category="DDoS")]
    report = evaluate.compute_category_report(rows)
    assert report["accuracy"] == 0.0
    assert report["raw_accuracy"] == 1.0


def test_category_report_confusion_table_counts_true_vs_predicted():
    rows = [
        _row(category="DDoS", true_category="DDoS"),
        _row(category="PortScan", true_category="DDoS"),
        _row(category="PortScan", true_category="PortScan"),
    ]
    report = evaluate.compute_category_report(rows)
    assert report["confusion"] == {
        "DDoS": {"DDoS": 1, "PortScan": 1},
        "PortScan": {"PortScan": 1},
    }


def test_category_report_baseline_guesses_the_most_common_true_category():
    rows = [
        _row(category="Unknown", true_category="DDoS"),
        _row(category="Unknown", true_category="DDoS"),
        _row(category="Unknown", true_category="PortScan"),
    ]
    report = evaluate.compute_category_report(rows)
    assert report["baseline_category"] == "DDoS"
    assert report["baseline_accuracy"] == 2 / 3


def test_category_report_can_beat_a_bad_baseline():
    rows = [
        _row(category="DDoS", true_category="DDoS"),
        _row(category="DDoS", true_category="DDoS"),
        _row(category="PortScan", true_category="PortScan"),
    ]
    report = evaluate.compute_category_report(rows)
    assert report["accuracy"] == 1.0
    assert report["baseline_accuracy"] == 2 / 3
    assert report["accuracy"] > report["baseline_accuracy"]


def test_run_arm_records_true_category_and_raw_top_category_per_row():
    trace = make_trace(
        ("category_decision",
         {"chosen_category": "DDoS", "raw_top_category": "DDoS", "raw_top_probability": 0.9}, None),
    )
    result = DetectionResult(event=_make_event(), is_anomalous=True, confidence=0.9,
                              detector_notes="[category=DDoS] x", trace=trace)
    manager = MagicMock()
    manager.run.return_value = result

    summary = evaluate.run_arm(manager, [_make_event()], [1], ["DDoS"])

    row = summary["rows"][0]
    assert row["true_category"] == "DDoS"
    assert row["category"] == "DDoS"
    assert row["raw_top_category"] == "DDoS"
    assert row["raw_top_probability"] == 0.9
    assert summary["category_report"]["n"] == 1
    assert summary["category_report"]["accuracy"] == 1.0
    assert summary["category_report"]["raw_accuracy"] == 1.0


def test_run_arm_category_report_works_without_any_llm_dispatch():
    # No "llm_dispatch"/"llm_validation_*" steps at all — this is the RF-only arm's shape,
    # and it must still populate category/raw_top_category from category_decision alone.
    trace = make_trace(
        ("category_decision",
         {"chosen_category": "Unknown", "raw_top_category": "PortScan", "raw_top_probability": 0.4}, None),
    )
    result = DetectionResult(event=_make_event(), is_anomalous=True, confidence=0.7,
                              detector_notes="[category=Unknown]", trace=trace)
    manager = MagicMock()
    manager.run.return_value = result

    summary = evaluate.run_arm(manager, [_make_event()], [1], ["PortScan"])

    assert summary["rows"][0]["llm_invoked"] is False
    assert summary["category_report"]["n"] == 1
    assert summary["category_report"]["accuracy"] == 0.0
    assert summary["category_report"]["raw_accuracy"] == 1.0
