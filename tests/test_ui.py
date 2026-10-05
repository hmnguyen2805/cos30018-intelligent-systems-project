"""
Tests for the UI (src/ui/): the demo scenarios run end to end through the real
Mitigation Manager and Judge, and the helpers that turn a PipelineRun into
the UI's tables report results and errors correctly. No Gradio server is
started; build_app() is only checked to construct.
"""
import json

import pytest

from src.correlation.manager import MitigationManager
from src.pipeline import Pipeline
from src.response import rules
from src.response.agent import MODE_AGENT
from src.ui import app
from src.ui.scenarios import SCENARIOS, ScriptedDetectionManager
from tests.test_judge_llm import COMPARE, ScriptedLLM, judge, step


def run_demo(title, mode="rules"):
    return app.execute(app.SOURCE_DEMO, title, "", mode)


EXPECTED = {
    "SSH brute force (SSH-Patator)": rules.AGREED_THREAT,
    "DoS Hulk on a web server": rules.AGREED_THREAT,
    "Port scan": rules.AGREED_THREAT,
    "Heartbleed against a TLS service": rules.AGREED_THREAT,
    "Normal web browsing (benign)": rules.BOTH_BENIGN,
    "Borderline detection": rules.LOW_DETECTION_CONFIDENCE,
    "Mitigation Manager crashes": rules.MITIGATION_MISSING,
}


@pytest.mark.parametrize("title, case", EXPECTED.items())
def test_demo_scenarios_reach_the_expected_case(title, case):
    run = run_demo(title)
    assert run.succeeded
    assert run.response.case == case


def test_every_scenario_is_covered_by_a_test():
    assert set(SCENARIOS) == set(EXPECTED) | {"Unknown attack type", "Detection Manager crashes"}


def test_unknown_attack_type_escalates_without_the_embedding_model(monkeypatch):
    # No sentence-transformers in the test environment: Mitigation reports no
    # match and the Judge escalates.
    monkeypatch.setattr("src.correlation.subagent._load_default_encoder",
                        lambda: (_ for _ in ()).throw(ImportError("not installed")))
    run = run_demo("Unknown attack type")
    assert run.response.escalated_to_human
    assert run.response.case == rules.ANOMALOUS_NO_MATCH


def test_successful_run_is_shown_as_succeeded():
    run = run_demo("SSH brute force (SSH-Patator)")
    assert "run succeeded" in app.status_markdown(run)
    assert "Automated response" in app.result_markdown(run)
    assert [row[1] for row in app.stage_rows(run)] == ["OK", "OK", "OK"]
    agents = {row[0] for row in app.trace_rows(run)}
    assert agents == {"Detection Manager", "Correlation Subagent", "Mitigation Manager", "Judge"}


def test_failed_mitigation_is_shown_as_an_error_but_the_run_completes():
    run = run_demo("Mitigation Manager crashes")
    status = app.status_markdown(run)
    assert "completed with errors" in status and "simulated failure" in status
    assert app.stage_rows(run)[1][1] == "Failed"
    assert "Escalated" in app.result_markdown(run)


def test_failed_detection_is_an_unsuccessful_run():
    run = run_demo("Detection Manager crashes")
    assert not run.succeeded
    assert "run failed" in app.status_markdown(run)
    assert [row[1] for row in app.stage_rows(run)] == ["Failed", "Skipped", "Skipped"]
    assert "None" in app.result_markdown(run)


def test_bad_custom_features_are_reported_not_raised():
    run = app.execute(app.SOURCE_CUSTOM, "", "{not json", "rules")
    assert not run.succeeded
    assert "Features must be a JSON object" in app.status_markdown(run)


@pytest.mark.parametrize("text", ['[1, 2]', '{}', '{"Destination Port": "abc"}'])
def test_parse_features_rejects_bad_input(text):
    with pytest.raises(ValueError):
        app.parse_features(text)


def test_parse_features_converts_to_floats():
    assert app.parse_features('{"Destination Port": 22}') == {"Destination Port": 22.0}


def test_agent_recheck_shows_in_the_rechecks_table_and_trace():
    scenario = SCENARIOS["Unknown attack type"]
    llm = ScriptedLLM(COMPARE, step("request_recheck", manager="detection", reason="Category is Unknown."),
                      step("escalate", reasoning="Still no known technique."))
    pipeline = Pipeline(ScriptedDetectionManager(scenario), judge(MODE_AGENT, llm),
                        MitigationManager(_no_embedding_subagent()))
    run = pipeline.run(app.make_event(app.SOURCE_DEMO, scenario.title, ""))

    (row,) = app.recheck_rows(run)
    assert row[0] == "detection" and row[1] == "Category is Unknown."
    assert "Recheck: Detection" in [r[0] for r in app.stage_rows(run)]
    assert "Recheck evidence" in run.detection.detector_notes
    assert "request_recheck" in [r[2] for r in app.trace_rows(run)]
    assert app.metrics(run)["judge_llm_usage"]["rechecks"] == 1


def test_progress_callback_sees_each_stage():
    seen = []
    app.execute(app.SOURCE_DEMO, "Port scan", "", "rules", on_stage=seen.append)
    assert seen == ["detection", "mitigation", "judge"]


def test_raw_json_is_valid():
    assert json.loads(app.raw_json(run_demo("Port scan")))["response"]["case"] == rules.AGREED_THREAT


def test_app_builds():
    pytest.importorskip("gradio")
    assert app.build_app() is not None


def _no_embedding_subagent():
    """A Correlation Subagent that behaves as if sentence-transformers isn't installed."""
    from src.correlation.subagent import CorrelationSubagent

    subagent = CorrelationSubagent(encoder=None)
    subagent._encoder_error = "not installed (test)"
    return subagent
