"""
Unit tests for DetectionManager — owns the top-level Detection contract and
delegates classification work to DetectionSubagent.
"""
from unittest.mock import patch

from src.detection.manager import DetectionManager
from src.shared.schemas import DetectionResult, TrafficEvent


def make_event():
    return TrafficEvent(features={"duration": 1.0})


@patch("src.detection.manager.DetectionSubagent")
def test_run_delegates_to_subagent_and_returns_its_result(mock_subagent_cls):
    expected = DetectionResult(event=make_event(), is_anomalous=True, confidence=0.9)
    mock_subagent_cls.return_value.run.return_value = expected

    manager = DetectionManager()
    result = manager.run(make_event())

    assert result is expected
    mock_subagent_cls.return_value.run.assert_called_once_with(make_event())


@patch("src.detection.manager.DetectionSubagent")
def test_run_logs_a_delegation_step(mock_subagent_cls):
    mock_subagent_cls.return_value.run.return_value = DetectionResult(
        event=make_event(), is_anomalous=False, confidence=0.8
    )

    manager = DetectionManager()
    manager.run(make_event())

    trace = manager.get_trace()
    assert any(step.action == "delegate_to_subagent" for step in trace)
