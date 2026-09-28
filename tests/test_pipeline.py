"""
Tests for src/pipeline.py: stage ordering, failure handling, timings.
Uses the real JudgeAgent, a mocked Detection Manager and the fake Mitigation
Manager, so no trained model or LLM is needed.
"""
from unittest.mock import MagicMock

from src.pipeline import Pipeline
from src.response import rules
from src.response.agent import JudgeAgent
from tests.fakes import FakeMitigationManager, make_detection, make_event


def detection_manager_returning(detection):
    manager = MagicMock()
    manager.run.return_value = detection
    return manager


def test_happy_path_runs_all_three_stages_in_order():
    detection = make_detection(anomalous=True, confidence=0.9)
    mitigation = FakeMitigationManager("match")
    pipeline = Pipeline(detection_manager_returning(detection), JudgeAgent(), mitigation)

    run = pipeline.run(make_event())

    assert run.succeeded and not run.errors
    assert mitigation.calls == [detection]  # mitigation received detection's output
    assert run.response.case == rules.AGREED_THREAT
    assert run.response.recommended_action == "block_source_ip"
    assert set(run.timings_ms) == {"detection", "mitigation", "judge"}


def test_mitigation_failure_is_recorded_and_judge_still_decides():
    pipeline = Pipeline(
        detection_manager_returning(make_detection(anomalous=True, confidence=0.9)),
        JudgeAgent(),
        FakeMitigationManager("fail"),
    )

    run = pipeline.run(make_event())

    assert "vector DB unreachable" in run.errors["mitigation"]
    assert run.mitigation is None
    assert run.succeeded
    assert run.response.case == rules.MITIGATION_MISSING
    assert run.response.escalated_to_human


def test_missing_mitigation_manager_is_handled_as_incomplete():
    pipeline = Pipeline(
        detection_manager_returning(make_detection(anomalous=False, confidence=0.95)),
        JudgeAgent(),
        mitigation_manager=None,
    )

    run = pipeline.run(make_event())

    assert run.errors["mitigation"] == "Mitigation Manager not configured"
    assert run.response.case == rules.MITIGATION_MISSING
    assert run.response.recommended_action == rules.NO_ACTION


def test_detection_failure_stops_the_run():
    detection_manager = MagicMock()
    detection_manager.run.side_effect = FileNotFoundError("no trained model")
    mitigation = FakeMitigationManager("match")
    judge = MagicMock()

    run = Pipeline(detection_manager, judge, mitigation).run(make_event())

    assert not run.succeeded
    assert "FileNotFoundError" in run.errors["detection"]
    assert mitigation.calls == []
    judge.run.assert_not_called()
    assert "detection" in run.timings_ms


def test_judge_failure_is_recorded():
    judge = MagicMock()
    judge.run.side_effect = RuntimeError("judge crashed")
    pipeline = Pipeline(
        detection_manager_returning(make_detection()), judge, FakeMitigationManager("match")
    )

    run = pipeline.run(make_event())

    assert not run.succeeded
    assert "judge crashed" in run.errors["judge"]
    assert run.detection is not None and run.mitigation is not None


def test_each_stage_trace_is_available_for_the_ui():
    detection = make_detection()
    pipeline = Pipeline(detection_manager_returning(detection), JudgeAgent(), FakeMitigationManager("match"))
    run = pipeline.run(make_event())
    assert run.response.trace  # Judge's steps are inspectable
