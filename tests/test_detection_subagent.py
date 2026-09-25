"""
Unit tests for DetectionSubagent — the borderline-handling loop that makes
Detection an agent rather than a bare classifier call. classifier.py's
functions are monkeypatched so no trained model artifact is needed.

The LLM-layer tests further down mock _call_llm_agent (the piece that talks
to the MCP subprocess/smolagents) so they run offline and fast, while
exercising the real validation/fallback/timeout wiring in subagent.py.
"""
import json
import time
from unittest.mock import patch

from src.detection.subagent import DetectionSubagent
from src.shared.schemas import TrafficEvent


def make_event():
    return TrafficEvent(features={"duration": 1.0, "packet_count": 2.0})


FAKE_ARTIFACT = {"model": None, "feature_names": ["duration", "packet_count"]}


@patch("src.detection.classifier.load_artifact", return_value={"model": None, "feature_names": []})
def make_agent(_mock_load, **kwargs):
    return DetectionSubagent(model_path="unused", **kwargs)


@patch("src.detection.classifier.load_artifact", return_value=FAKE_ARTIFACT)
def make_llm_agent(_mock_load, **kwargs):
    """Like make_agent, but with a non-empty feature_names list so
    top_features (used for the LLM grounding check) has something to rank."""
    return DetectionSubagent(model_path="unused", use_llm=True, **kwargs)


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_anomalous_score_skips_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.95
    agent = make_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95
    mock_votes.assert_not_called()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_benign_score_skips_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.05
    agent = make_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is False
    assert result.confidence == 0.95
    mock_votes.assert_not_called()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_borderline_score_triggers_tree_vote_check(mock_predict, mock_votes):
    mock_predict.return_value = 0.5  # inside [0.4, 0.6]
    mock_votes.return_value = (0.7, 0.05)  # trees agree, low std

    result = make_agent().run(make_event())

    mock_votes.assert_called_once()
    assert result.is_anomalous is True
    assert result.confidence == 0.7
    assert "trust" in result.detector_notes.lower()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_borderline_score_with_high_tree_disagreement_is_flagged(mock_predict, mock_votes):
    mock_predict.return_value = 0.5
    mock_votes.return_value = (0.55, 0.3)  # high std, disagreement

    result = make_agent().run(make_event())

    assert "disagreement" in result.detector_notes.lower()


@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_run_produces_a_trace_of_its_steps(mock_predict, mock_votes):
    mock_predict.return_value = 0.9
    result = make_agent().run(make_event())

    assert len(result.trace) >= 2
    assert result.trace[0].action == "call_classifier"


# --- LLM layer -----------------------------------------------------------
# _call_llm_agent is mocked throughout: it's the only piece that talks to the
# MCP subprocess/smolagents, so mocking it keeps these tests offline and fast
# while exercising subagent.py's real gating, validation, and fallback logic.

def valid_llm_json():
    return json.dumps({
        "category": "PortScan",
        "explanation": "duration and packet_count both look elevated for this event.",
        "tools_used": ["top_features"],
    })


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.subagent.DetectionSubagent._call_llm_agent")
@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_use_llm_false_is_unaffected_even_if_llm_would_be_called(
    mock_predict, mock_votes, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.95
    agent = make_agent()  # use_llm defaults to False

    result = agent.run(make_event())

    mock_call_llm.assert_not_called()
    assert result.is_anomalous is True
    assert result.confidence == 0.95
    assert result.detector_notes is None


@patch("src.detection.subagent.DetectionSubagent._call_llm_agent")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_benign_event_with_use_llm_true_skips_the_llm(mock_predict, mock_call_llm):
    mock_predict.return_value = 0.05  # below LLM_TRIGGER_THRESHOLD (0.4)
    agent = make_llm_agent()

    result = agent.run(make_event())

    mock_call_llm.assert_not_called()
    assert result.is_anomalous is False
    assert result.detector_notes is None


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.subagent.DetectionSubagent._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_valid_json_produces_categorized_notes_without_changing_numbers(
    mock_predict, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.95  # clearly anomalous, but still >= trigger threshold
    agent = make_llm_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95  # identical to the deterministic (use_llm=False) path
    assert result.detector_notes.startswith("[category=PortScan]")
    assert "duration and packet_count" in result.detector_notes


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.subagent.DetectionSubagent._call_llm_agent", return_value="not valid json")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_invalid_json_falls_back_to_template_note(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95
    assert result.detector_notes == "[category=Unknown] Anomalous event — LLM explanation unavailable."


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.subagent.DetectionSubagent._call_llm_agent",
    return_value=json.dumps({"category": "Ransomware", "explanation": "x", "tools_used": []}),
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_category_outside_fixed_list_falls_back_to_template_note(
    mock_predict, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert "category=Unknown" in result.detector_notes
    assert result.is_anomalous is True
    assert result.confidence == 0.95


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.subagent.DetectionSubagent._call_llm_agent", side_effect=RuntimeError("MCP server crashed"))
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_exception_falls_back_and_does_not_raise(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    agent = make_llm_agent()
    agent._mcp_client = "sentinel-connection"  # pretend a connection was already open

    result = agent.run(make_event())  # must not raise

    assert result.is_anomalous is True
    assert "category=Unknown" in result.detector_notes
    assert agent._mcp_client is None  # dropped so the next event reconnects


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_timeout_falls_back_and_does_not_raise(mock_predict, mock_top_features):
    def slow_call(self, event, p_anomalous, vote_std):
        time.sleep(0.3)
        return valid_llm_json()

    mock_predict.return_value = 0.95
    with patch("src.detection.subagent.DetectionSubagent._call_llm_agent", slow_call):
        agent = make_llm_agent(llm_timeout_seconds=0.05)
        agent._mcp_client = "sentinel-connection"
        result = agent.run(make_event())

    assert result.is_anomalous is True
    assert "category=Unknown" in result.detector_notes
    assert agent._mcp_client is None  # dropped so the next event reconnects


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.subagent.DetectionSubagent._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_success_on_borderline_event_keeps_the_tree_disagreement_note(
    mock_predict, mock_votes, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.5  # borderline
    mock_votes.return_value = (0.55, 0.3)  # high disagreement
    result = make_llm_agent().run(make_event())

    assert result.detector_notes.startswith("[category=PortScan]")
    assert "disagreement" in result.detector_notes.lower()


# --- MCP connection lifecycle ---------------------------------------------
# Here _call_llm_agent runs for real (not mocked) so _ensure_llm_connection is
# exercised; only the MCP/smolagents boundary (MCPClient, ToolCallingAgent) is
# mocked, so no subprocess or real model is involved.

@patch("src.detection.subagent.DetectionSubagent._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_mcp_connection_is_opened_once_and_reused_across_events(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = ["fake_tool"]
    mock_agent_cls.return_value.run.return_value = valid_llm_json()

    agent = make_llm_agent()
    agent.run(make_event())
    agent.run(make_event())

    mock_mcp_client_cls.assert_called_once()  # one subprocess/connection for both events
    assert mock_agent_cls.call_count == 2  # a fresh ToolCallingAgent per event, same tools
    agent.close()


def test_close_is_a_noop_when_never_connected():
    make_llm_agent().close()  # must not raise


@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_close_disconnects_an_open_mcp_client(mock_server_params, mock_mcp_client_cls):
    agent = make_llm_agent()
    agent._ensure_llm_connection()

    agent.close()

    mock_mcp_client_cls.return_value.disconnect.assert_called_once()
    assert agent._mcp_client is None
    assert agent._llm_tools is None


@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_context_manager_closes_connection_on_exit(mock_server_params, mock_mcp_client_cls):
    with make_llm_agent() as agent:
        agent._ensure_llm_connection()

    mock_mcp_client_cls.return_value.disconnect.assert_called_once()


@patch("src.detection.subagent.DetectionSubagent._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_connection_reconnects_on_the_next_event_after_a_failure(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_agent_cls.return_value.run.side_effect = [RuntimeError("subprocess died"), valid_llm_json()]

    agent = make_llm_agent()
    first = agent.run(make_event())   # MCP call raises -> template note, connection dropped
    second = agent.run(make_event())  # succeeds on a freshly-opened connection

    assert "category=Unknown" in first.detector_notes
    assert second.detector_notes.startswith("[category=PortScan]")
    assert mock_mcp_client_cls.call_count == 2
