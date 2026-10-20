"""
Tests for the evaluation harness (src/evaluation/): the task file, scoring,
running each Judge mode (the LLM is the ScriptedLLM from test_judge_llm.py,
so no Ollama), and the report files.
"""
import csv
import json
from pathlib import Path

import pytest

from src.correlation.manager import MitigationManager
from src.correlation.subagent import CorrelationSubagent
from src.evaluation import runner
from src.evaluation.report import summary_markdown, write_results
from src.evaluation.runner import MitigationUsage, RunRecord, run_evaluation, run_task, score
from src.evaluation.tasks import DIFFICULTIES, EvalTask, ScriptedMitigation, load_tasks, task_from_dict
from src.response.llm import LLMConfig
from tests.test_judge_llm import CHECK, COMPARE, ScriptedLLM, step
from tests.test_mitigation_llm import EVIDENCE, choose
from tests.test_mitigation_llm import ScriptedLLM as MitigationScriptedLLM
from tests.test_mitigation_llm import detection as make_detection

DEV_TASKS = Path(__file__).resolve().parents[1] / "evaluation" / "tasks" / "dev_tasks.json"


@pytest.fixture(scope="module")
def tasks():
    return {t.id: t for t in load_tasks(DEV_TASKS)}


# --- the task file -------------------------------------------------------------------

def test_dev_tasks_meet_the_spec_minimum(tasks):
    assert len(tasks) >= 30
    assert {t.difficulty for t in tasks.values()} == set(DIFFICULTIES)  # simple, moderate, difficult, edge


def test_dev_tasks_need_no_trained_model(tasks):
    assert all(not t.uses_real_detection for t in tasks.values())


@pytest.mark.parametrize("bad, message", [
    ({"expected": ["maybe"]}, "expected"),
    ({"difficulty": "hard"}, "difficulty"),
    ({"split": "test"}, "split"),
    ({"detection": None, "features": {}}, "needs either"),
    ({"mitigation": 3}, "mitigation"),
])
def test_bad_tasks_are_rejected(bad, message):
    good = {"id": "X1", "difficulty": "simple", "expected": ["act"], "detection": {"anomalous": True, "confidence": 0.9}}
    with pytest.raises(ValueError, match=message):
        task_from_dict({**good, **bad})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "t.json"
    task = {"id": "X1", "difficulty": "simple", "expected": "act", "detection": {"anomalous": False, "confidence": 0.9}}
    path.write_text(json.dumps([task, task]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_tasks(path)


# --- scoring -------------------------------------------------------------------------

def _record(outcome, technique=None):
    return RunRecord(task_id="X", difficulty="simple", split="dev", mode="rules", repeat=0, expected=[],
                     outcome=outcome, technique=technique)


def _task(expected, true_category=None):
    return EvalTask(id="X", difficulty="simple", split="dev", description="", expected=expected,
                    detection={"anomalous": True, "confidence": 0.9}, true_category=true_category)


@pytest.mark.parametrize("expected, outcome, technique, correct, unsafe, over_cautious", [
    (["act"], "act", "T1110", True, False, False),
    (["act"], "act", "T1498", False, True, False),        # acted on the wrong technique
    (["act"], "escalate", None, False, False, True),      # safe but over-cautious
    (["escalate"], "act", "T1110", False, True, False),   # acted when a human should check
    (["escalate"], "no_action", None, False, True, False),  # treated a threat as benign
    (["no_action"], "no_action", None, True, False, False),
    (["failed"], "failed", None, True, False, False),
])
def test_scoring(expected, outcome, technique, correct, unsafe, over_cautious):
    record = _record(outcome, technique)
    score(record, _task(expected, true_category="BruteForce"))
    assert (record.correct, record.unsafe, record.over_cautious) == (correct, unsafe, over_cautious)


# --- running the modes -----------------------------------------------------------------

def test_rules_mode_on_the_dev_tasks(tasks):
    records = {r.task_id: r for r in run_evaluation(list(tasks.values()), modes=["rules"], keep_trace=False)}
    assert len(records) == len(tasks)
    assert records["S01"].correct and records["S01"].outcome == "act"
    assert records["S06"].outcome == "no_action"
    assert records["E03"].outcome == "failed" and records["E03"].correct  # Detection crash
    assert records["E01"].outcome == "escalate" and "mitigation" in records["E01"].errors
    # The category mismatch the rules can't see: they act, so it's unsafe.
    assert records["D01"].outcome == "act" and records["D01"].unsafe


def test_rules_mode_runs_once_even_with_repeats(tasks):
    records = run_evaluation([tasks["S01"]], modes=["rules"], repeats=3, keep_trace=False)
    assert len(records) == 1


def test_agent_mode_catches_the_mismatch_and_records_metrics(tasks):
    llm = ScriptedLLM(COMPARE, CHECK, step("escalate", reasoning="DoS does not fit brute force."))
    record = run_task(tasks["D01"], "agent", llm_client=llm)
    assert record.outcome == "escalate" and record.correct and not record.unsafe
    assert (record.iterations, record.llm_calls, record.tokens) == (3, 3, 360)
    assert (record.tool_steps, record.tool_ok) == (2, 2)
    assert record.decided_by == "llm"
    assert any(row["action"] == "check_category_consistency" for row in record.trace)


def test_agent_recheck_happens_inside_the_harness(tasks):
    llm = ScriptedLLM(COMPARE, step("request_recheck", manager="detection", reason="Category is Unknown."),
                      step("escalate", reasoning="Still no technique."))
    record = run_task(tasks["M09"], "agent", llm_client=llm)
    assert record.rechecks == 1 and record.correct


def test_llm_failure_falls_back_to_the_rules(tasks):
    llm = ScriptedLLM(TimeoutError("model too slow"))
    record = run_task(tasks["S01"], "agent", llm_client=llm)
    assert record.decided_by == "rules_fallback" and record.correct
    assert record.fallback_reason.startswith("llm_error")


class FakeUsageManager:
    """Stands in for a Mitigation Manager: every run() leaves 2 LLM calls (120 tokens) in last_usage."""
    mode = "agent"

    def __init__(self):
        self.last_usage = {}

    def run(self, input_data, recheck_reason=None):
        self.last_usage = {"llm_calls": 2, "prompt_tokens": 100, "completion_tokens": 20, "llm_latency_ms": 50.0}
        if recheck_reason:
            self.last_usage["fallback_reason"] = "max_steps"
        return "result"


def test_mitigation_usage_adds_up_every_run_including_a_recheck():
    tracker = MitigationUsage(FakeUsageManager())
    assert tracker.run("detection") == "result"
    tracker.run("detection", recheck_reason="Category does not fit the matched technique.")

    assert (tracker.llm_calls, tracker.tokens, tracker.llm_latency_ms) == (4, 240, 100.0)
    assert tracker.fallback_reason == "max_steps"


def test_scripted_mitigation_reports_no_usage():
    tracker = MitigationUsage(ScriptedMitigation({"techniques": ["T1190"], "confidence": 0.9}))
    tracker.run(make_detection())

    assert tracker.mode == "scripted"
    assert (tracker.llm_calls, tracker.tokens, tracker.llm_latency_ms, tracker.fallback_reason) == (0, 0, 0.0, None)


def test_mitigation_agent_usage_lands_in_the_run_record(tasks, monkeypatch):
    mitigation_llm = MitigationScriptedLLM(EVIDENCE, choose("T1190"))
    agent = MitigationManager(subagent=CorrelationSubagent(), mode="agent", llm_client=mitigation_llm,
                              config=LLMConfig())
    monkeypatch.setattr("src.correlation.manager.mitigation_from_env", lambda: agent)
    record = run_task(tasks["D05"], "rules")

    assert record.mitigation_mode == "agent" and record.technique == "T1190"
    assert (record.mit_llm_calls, record.mit_tokens, record.mit_llm_latency_ms) == (2, 240, 100.0)
    assert record.tokens == 0  # the Judge ran in rules mode, so only Mitigation used the LLM


def test_rules_mode_mitigation_records_its_mode_and_no_llm_usage(tasks, monkeypatch):
    monkeypatch.delenv("MITIGATION_MODE", raising=False)
    record = run_task(tasks["D05"], "rules")

    assert record.mitigation_mode == "rules"
    assert (record.mit_llm_calls, record.mit_tokens, record.mit_fallback_reason) == (0, 0, None)


def test_missing_trained_model_skips_real_detection_tasks(monkeypatch):
    def no_model():
        raise FileNotFoundError("No trained model")
    monkeypatch.setattr(runner, "_real_detection_manager", no_model)
    task = task_from_dict({"id": "U1", "difficulty": "simple", "split": "unseen", "expected": "no_action",
                           "features": {"Destination Port": 443}})
    record = run_task(task, "rules")
    assert record.outcome == "skipped" and "No trained model" in record.skip_reason


# --- the report ------------------------------------------------------------------------

def test_report_files(tasks, tmp_path):
    chosen = [tasks[i] for i in ("S01", "D01", "E03")]
    records = run_evaluation(chosen, modes=["rules"])
    llm = ScriptedLLM(COMPARE, CHECK, step("escalate", reasoning="Mismatch."))
    records.append(run_task(tasks["D01"], "agent", llm_client=llm))

    paths = write_results(records, tmp_path)
    assert json.loads(paths["json"].read_text())[0]["trace"]  # traces are kept
    with open(paths["csv"], newline="") as f:
        assert len(list(csv.DictReader(f))) == 4
    summary = paths["summary"].read_text()
    for heading in ("Metrics per Judge mode", "Success by difficulty", "Per task", "Tasks where the modes differ"):
        assert heading in summary
    assert "| D01 |" in summary


def test_summary_and_csv_include_the_mitigation_columns(tasks, tmp_path):
    rules_only = run_evaluation([tasks["D05"]], modes=["rules"], keep_trace=False)
    summary = summary_markdown(rules_only)
    assert "| Mitigation agent runs | 0/1 (0%) |" in summary
    assert "| Mitigation avg tokens | - |" in summary

    agent_run = RunRecord(task_id="X1", difficulty="simple", split="dev", mode="agent", repeat=0, expected=["act"],
                          outcome="act", correct=True, tokens=700, mitigation_mode="agent",
                          mit_llm_calls=2, mit_tokens=300, mit_llm_latency_ms=4000.0)
    summary = summary_markdown([agent_run])
    assert "| Mitigation avg tokens | 300 |" in summary
    assert "| avg tokens (Judge + Mitigation) | 1000 |" in summary

    paths = write_results([agent_run], tmp_path)
    with open(paths["csv"], newline="") as f:
        header = next(csv.reader(f))
    assert {"mitigation_mode", "mit_llm_calls", "mit_tokens", "mit_llm_latency_ms", "mit_fallback_reason"} <= set(header)


def test_summary_counts_success(tasks):
    records = run_evaluation([tasks["S01"], tasks["D01"]], modes=["rules"], keep_trace=False)
    assert "| success | 1/2 (50%) |" in summary_markdown(records)