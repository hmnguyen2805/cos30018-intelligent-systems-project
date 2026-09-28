"""
Unit tests for LLMExplanationLayer's connection/timeout/retry/circuit-breaker
plumbing (src.detection.llm.layer), exercised through DetectionSubagent.
_call_llm_agent is mocked throughout (it's the only piece that talks to the
MCP subprocess/smolagents) so these run offline and fast, while exercising
subagent.py's real gating, validation, and fallback logic.

Split out of the old test_detection_subagent.py — see test_subagent.py
(core flow + category decision) and test_modes.py (agent vs single_shot,
salvage) for the rest.
"""
import json
import time
from unittest.mock import MagicMock, patch

from src.shared.schemas import TrafficEvent
from tests.detection.conftest import (
    fake_mcp_tools,
    make_agent,
    make_event,
    make_llm_agent,
    _make_provider_unavailable_error,
    _make_rate_limit_error,
    valid_llm_json,
)


# --- LLM layer -----------------------------------------------------------
# _call_llm_agent is mocked throughout: it's the only piece that talks to the
# MCP subprocess/smolagents, so mocking it keeps these tests offline and fast
# while exercising subagent.py's real gating, validation, and fallback logic.

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent")
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
    # Category is computed/tagged for every truly-anomalous event regardless of use_llm; no
    # category model is loaded here, so it falls back to "Unknown".
    assert result.detector_notes == "[category=Unknown]"


@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_benign_event_with_use_llm_true_skips_the_llm(mock_predict, mock_call_llm):
    mock_predict.return_value = 0.05  # below LLM_TRIGGER_THRESHOLD (0.4)
    agent = make_llm_agent()

    result = agent.run(make_event())

    mock_call_llm.assert_not_called()
    assert result.is_anomalous is False
    assert result.detector_notes is None


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_valid_json_produces_categorized_notes_without_changing_numbers(
    mock_predict, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.95  # clearly anomalous, but still >= trigger threshold
    agent = make_llm_agent()

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95  # identical to the deterministic (use_llm=False) path
    # No category model loaded in this fixture, so the code-side decision is "Unknown" — the
    # LLM's JSON has no category slot at all now, only explanation/tools_used.
    assert result.detector_notes.startswith("[category=Unknown]")
    assert "duration and packet_count" in result.detector_notes


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value="not valid json")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_invalid_json_falls_back_to_template_note(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95
    assert result.detector_notes == "[category=Unknown] Anomalous event — LLM explanation unavailable."


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm.layer.LLMExplanationLayer._call_llm_agent",
    return_value=json.dumps({"category": "Ransomware", "explanation": "x", "tools_used": []}),
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_stray_category_field_in_json_is_ignored_not_validated(
    mock_predict, mock_call_llm, mock_top_features
):
    # The LLM's schema has no category slot; a stray "category" key in its raw JSON (e.g. from
    # a stale cached prompt) must simply be ignored, not trigger any validation failure — there
    # is no "bad category" check left, since code (not the LLM) decides the category.
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert "category=Unknown" in result.detector_notes  # from the code-side decision, not the LLM
    assert "x" in result.detector_notes  # the LLM's explanation was accepted, not discarded
    assert result.is_anomalous is True
    assert result.confidence == 0.95


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("MCP server crashed"))
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_exception_falls_back_and_does_not_raise(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    agent = make_llm_agent()
    sentinel = MagicMock()
    agent._llm._mcp_client = sentinel  # pretend a connection was already open

    result = agent.run(make_event())  # must not raise

    assert result.is_anomalous is True
    assert "category=Unknown" in result.detector_notes
    assert agent._llm._mcp_client is None  # dropped so the next event reconnects
    sentinel.disconnect.assert_called_once()  # thread had already finished — safe to disconnect now
    assert sentinel not in agent._llm._abandoned_mcp_clients  # disconnected directly, not queued


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_timeout_falls_back_and_does_not_raise(mock_predict, mock_top_features):
    def slow_call(self, event, p_anomalous, vote_std, chosen_category, category_probability, step_buffer, mode):
        time.sleep(0.3)
        return valid_llm_json()

    mock_predict.return_value = 0.95
    with patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", slow_call):
        agent = make_llm_agent(llm_timeout_seconds=0.05)
        sentinel = MagicMock()
        agent._llm._mcp_client = sentinel
        result = agent.run(make_event())

    assert result.is_anomalous is True
    assert "category=Unknown" in result.detector_notes
    assert agent._llm._mcp_client is None  # dropped so the next event doesn't immediately reconnect
    assert sentinel in agent._llm._abandoned_mcp_clients  # queued for close() / self-cleanup, not leaked
    sentinel.disconnect.assert_not_called()  # not safe to touch directly — may still be in use


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_success_on_borderline_event_keeps_the_tree_disagreement_note(
    mock_predict, mock_votes, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.5  # borderline
    mock_votes.return_value = (0.55, 0.3)  # high disagreement
    result = make_llm_agent().run(make_event())

    assert result.detector_notes.startswith("[category=Unknown]")
    assert "disagreement" in result.detector_notes.lower()


# --- MCP connection lifecycle ---------------------------------------------
# Here _call_llm_agent runs for real (not mocked) so _ensure_llm_connection is
# exercised; only the MCP/smolagents boundary (MCPClient, ToolCallingAgent) is
# mocked, so no subprocess or real model is involved.

@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_mcp_connection_is_opened_once_and_reused_across_events(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
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
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    agent = make_llm_agent()
    agent._llm._ensure_llm_connection()

    agent.close()

    mock_mcp_client_cls.return_value.disconnect.assert_called_once()
    assert agent._llm._mcp_client is None
    assert agent._llm._llm_tools_by_name is None


@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_context_manager_closes_connection_on_exit(mock_server_params, mock_mcp_client_cls):
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    with make_llm_agent() as agent:
        agent._llm._ensure_llm_connection()

    mock_mcp_client_cls.return_value.disconnect.assert_called_once()


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_connection_reconnects_on_the_next_event_after_a_failure(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = [RuntimeError("subprocess died"), valid_llm_json()]

    agent = make_llm_agent()
    first = agent.run(make_event())   # MCP call raises -> template note, connection dropped
    second = agent.run(make_event())  # succeeds on a freshly-opened connection

    assert "category=Unknown" in first.detector_notes
    assert second.detector_notes.startswith("[category=Unknown]")
    assert mock_mcp_client_cls.call_count == 2


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_close_disconnects_a_client_abandoned_after_a_timeout(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()

    def hang(*args, **kwargs):
        time.sleep(0.3)
        return valid_llm_json()

    mock_agent_cls.return_value.run.side_effect = hang

    agent = make_llm_agent(llm_timeout_seconds=0.05)
    result = agent.run(make_event())  # times out; connection queued, not disconnected yet

    assert "category=Unknown" in result.detector_notes
    assert len(agent._llm._abandoned_mcp_clients) == 1
    abandoned_client = agent._llm._abandoned_mcp_clients[0]
    abandoned_client.disconnect.assert_not_called()

    agent.close()

    abandoned_client.disconnect.assert_called_once()


# --- event_id-based tools --------------------------------------------------
# The task prompt and the LLM-facing tool list must never carry raw features
# (only p_anomalous/tree_vote_std/event_id do) — register_event/clear_event are
# called directly by code, never exposed to the ToolCallingAgent, and neither is
# predict_proba_anomalous (code already computed p_anomalous and put it in the
# task — offering the tool just wastes a step). tree_vote_spread is offered only
# on a borderline event, since it's not meaningful otherwise. predict_attack_category
# is always offered in agent mode: the LLM may consult it for its own investigation
# even though code has already made the final category decision from the same tool.

@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_anomalous_event_only_exposes_top_features(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95  # clearly anomalous, not borderline -> vote_std stays None
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()

    make_llm_agent().run(make_event())

    agent_tool_names = {t.name for t in mock_agent_cls.call_args.kwargs["tools"]}
    assert agent_tool_names == {"top_features", "predict_attack_category"}


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_borderline_event_also_exposes_tree_vote_spread(
    mock_predict, mock_votes, mock_server_params, mock_mcp_client_cls, mock_agent_cls,
    mock_top_features, mock_get_model,
):
    mock_predict.return_value = 0.5  # borderline -> vote_std is not None
    mock_votes.return_value = (0.55, 0.1)
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()

    make_llm_agent().run(make_event())

    agent_tool_names = {t.name for t in mock_agent_cls.call_args.kwargs["tools"]}
    assert agent_tool_names == {"top_features", "tree_vote_spread", "predict_attack_category"}


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_agent_mode_exposes_predict_attack_category_for_the_llms_own_use(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()

    make_llm_agent().run(make_event())

    agent_tools_by_name = {t.name: t for t in mock_agent_cls.call_args.kwargs["tools"]}
    assert "predict_attack_category" in agent_tools_by_name


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_register_event_is_called_with_plain_python_floats_and_task_omits_features(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    import numpy as np

    mock_predict.return_value = 0.95
    tools = fake_mcp_tools()
    mock_mcp_client_cls.return_value.get_tools.return_value = tools
    mock_agent_cls.return_value.run.return_value = valid_llm_json()
    register_tool = next(t for t in tools if t.name == "register_event")
    clear_tool = next(t for t in tools if t.name == "clear_event")

    event = TrafficEvent(features={"duration": np.float64(1.0), "packet_count": np.float64(2.0)})
    make_llm_agent().run(event)

    register_tool.assert_called_once()
    passed_features = register_tool.call_args.kwargs["features"]
    assert all(type(v) is float for v in passed_features.values())  # not numpy.float64

    run_call_args = mock_agent_cls.return_value.run.call_args
    task_text = run_call_args.args[0] if run_call_args.args else run_call_args.kwargs["task"]
    assert "duration" not in task_text and "packet_count" not in task_text
    assert "event_id=" in task_text

    clear_tool.assert_called_once()
    assert clear_tool.call_args.kwargs["event_id"] == register_tool.call_args.kwargs["event_id"]


# --- warmup() ---------------------------------------------------------------

def test_warmup_is_a_noop_when_use_llm_is_false():
    agent = make_agent()
    result = agent.warmup()
    assert result == {"ok": True, "elapsed_seconds": 0.0, "reason": None}


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_warmup_opens_the_connection_and_returns_elapsed_seconds(
    mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_get_model
):
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = "OK"

    agent = make_llm_agent()
    result = agent.warmup()

    assert result["ok"] is True
    assert result["elapsed_seconds"] >= 0.0
    assert result["reason"] is None
    mock_mcp_client_cls.assert_called_once()  # connection opened by warmup...
    assert agent._llm._mcp_client is not None      # ...and left open for the run that follows


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_warmup_fails_cleanly_after_a_non_transient_error(
    mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_get_model
):
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = ValueError("bad API key")

    agent = make_llm_agent()
    result = agent.warmup()  # must not raise

    assert result["ok"] is False
    assert "bad API key" in result["reason"]
    mock_agent_cls.return_value.run.assert_called_once()  # non-transient: no retries


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.llm.layer.time.sleep")  # skip real backoff delays
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_warmup_retries_a_transient_error_then_succeeds(
    mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep, mock_get_model
):
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    rate_limit_error = _make_rate_limit_error()
    mock_agent_cls.return_value.run.side_effect = [rate_limit_error, "OK"]

    agent = make_llm_agent()
    result = agent.warmup()

    assert result["ok"] is True
    assert mock_agent_cls.return_value.run.call_count == 2
    mock_sleep.assert_called_once_with(2.0)  # first backoff delay
    assert any(step.action == "llm_warmup_retry" for step in agent.get_trace())


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
def test_warmup_fails_after_exhausting_retries_on_persistent_transient_errors(
    mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep, mock_get_model
):
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = [_make_rate_limit_error() for _ in range(4)]

    agent = make_llm_agent()
    result = agent.warmup()

    assert result["ok"] is False
    assert "rate_limited" in result["reason"]
    assert mock_agent_cls.return_value.run.call_count == 4  # 1 + 3 retries
    assert mock_sleep.call_args_list == [((2.0,),), ((4.0,),), ((8.0,),)]


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_warmup_means_the_first_event_does_not_open_a_second_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()

    agent = make_llm_agent()
    agent.warmup()
    agent.run(make_event())

    mock_mcp_client_cls.assert_called_once()


# --- trace isolation after a timeout ---------------------------------------
# _call_llm_agent buffers its log_step() calls in a local list (step_buffer)
# instead of writing straight into self._trace; _run_llm_layer only replays
# that buffer once it knows the call finished within the timeout. An
# abandoned (timed-out) call's buffer is simply never read again.

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.classifier.predict_proba_anomalous")
def test_timed_out_calls_steps_are_not_merged_into_the_trace(mock_predict, mock_top_features):
    def slow_call(self, event, p_anomalous, vote_std, chosen_category, category_probability, step_buffer, mode):
        step_buffer.append({"action": "llm_layer_start", "thought": "started"})
        time.sleep(0.2)
        return valid_llm_json()

    mock_predict.return_value = 0.95
    with patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", slow_call):
        agent = make_llm_agent(llm_timeout_seconds=0.02)
        result = agent.run(make_event())

    assert not any(step.action == "llm_layer_start" for step in result.trace)


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.classifier.predict_proba_anomalous")
def test_abandoned_threads_steps_never_appear_in_a_later_events_trace(mock_predict, mock_top_features):
    def slow_call(self, event, p_anomalous, vote_std, chosen_category, category_probability, step_buffer, mode):
        step_buffer.append({"action": "should_never_leak"})
        time.sleep(0.2)
        return valid_llm_json()

    mock_predict.return_value = 0.95
    agent = make_llm_agent(llm_timeout_seconds=0.02)

    with patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", slow_call):
        first = agent.run(make_event())  # times out

    with patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json()):
        second = agent.run(make_event())  # fresh event, fast success

    assert not any(step.action == "should_never_leak" for step in first.trace)
    assert not any(step.action == "should_never_leak" for step in second.trace)


# --- circuit breaker ---------------------------------------------------------

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("dead"))
@patch("src.detection.classifier.predict_proba_anomalous")
def test_circuit_opens_after_threshold_consecutive_failures_and_skips_further_llm_calls(
    mock_predict, mock_call_llm, mock_top_features
):
    mock_predict.return_value = 0.95
    agent = make_llm_agent(circuit_breaker_threshold=3)

    for _ in range(3):
        agent.run(make_event())
    assert agent._llm._circuit_open is True
    assert mock_call_llm.call_count == 3

    result = agent.run(make_event())  # circuit open: must not call the LLM at all

    assert mock_call_llm.call_count == 3  # unchanged
    assert "category=Unknown" in result.detector_notes
    assert any(step.action == "llm_circuit_open" for step in result.trace)


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_a_success_resets_the_consecutive_failure_count(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    mock_call_llm.side_effect = [RuntimeError("x"), RuntimeError("x"), valid_llm_json(), RuntimeError("x")]
    agent = make_llm_agent(circuit_breaker_threshold=3)

    for _ in range(4):
        agent.run(make_event())

    assert agent._llm._circuit_open is False  # never hit 3 *consecutive* failures (a success reset it)
    assert mock_call_llm.call_count == 4


# --- per-event retry on transient LLM-provider errors -----------------------
# ToolCallingAgent.run is mocked directly (not _call_llm_agent) so the real
# retry loop in _call_llm_agent runs; time.sleep is mocked so backoff delays
# don't slow the test down.

@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_per_event_call_retries_a_rate_limit_then_succeeds_without_dropping_the_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = [_make_rate_limit_error(), valid_llm_json()]

    agent = make_llm_agent()
    result = agent.run(make_event())

    assert result.detector_notes.startswith("[category=Unknown]")
    assert mock_agent_cls.return_value.run.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    assert any(step.action == "llm_retry" for step in result.trace)
    mock_mcp_client_cls.assert_called_once()  # provider-side error: MCP connection untouched
    assert agent._llm._mcp_client is not None


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_per_event_rate_limit_exhausting_retries_reports_rate_limited_fallback_reason(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = [_make_rate_limit_error() for _ in range(4)]

    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    agent = make_llm_agent()
    result = agent.run(make_event())

    assert "category=Unknown" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["rate_limited"]
    assert agent._llm._mcp_client is not None  # provider-side: connection kept, not disconnected


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_per_event_service_unavailable_reports_provider_unavailable_fallback_reason(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.side_effect = [_make_provider_unavailable_error() for _ in range(4)]

    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    agent = make_llm_agent()
    result = agent.run(make_event())

    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["provider_unavailable"]


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("broken pipe"))
@patch("src.detection.classifier.predict_proba_anomalous")
def test_non_transient_exception_still_disconnects_the_connection(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    agent = make_llm_agent()
    sentinel = MagicMock()
    agent._llm._mcp_client = sentinel

    result = agent.run(make_event())

    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["exception"]
    sentinel.disconnect.assert_called_once()  # not classified as provider-side: MCP assumed dead


# --- reporting: llm_dispatch counted even on timeout, token usage recorded --

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_dispatch_is_logged_even_when_the_call_times_out(mock_predict, mock_top_features):
    def slow_call(self, event, p_anomalous, vote_std, chosen_category, category_probability, step_buffer, mode):
        time.sleep(0.2)
        return valid_llm_json()

    mock_predict.return_value = 0.95
    with patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", slow_call):
        agent = make_llm_agent(llm_timeout_seconds=0.02)
        result = agent.run(make_event())

    # llm_dispatch (unbuffered, logged before the timed call) proves the event was sent to
    # the LLM even though llm_layer_start (buffered) never gets merged in on a timeout.
    assert any(step.action == "llm_dispatch" for step in result.trace)
    assert not any(step.action == "llm_layer_start" for step in result.trace)


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_token_usage_is_recorded_after_a_successful_llm_call(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    from types import SimpleNamespace

    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()
    mock_agent_cls.return_value.monitor.get_total_token_counts.return_value = SimpleNamespace(
        input_tokens=123, output_tokens=45,
    )

    result = make_llm_agent().run(make_event())

    usage_step = next(step for step in result.trace if step.action == "llm_token_usage")
    assert usage_step.tool_input == {"prompt_tokens": 123, "completion_tokens": 45}


# --- exception-chain unwrapping (smolagents wraps model errors) -------------
# smolagents' agent step re-raises whatever the model call raises as its own
# AgentGenerationError, `from e` — so the real litellm error (with its
# status_code) is one level down in __cause__, not on the exception our code
# is handed. _classify_llm_error must walk that chain to classify correctly.

def _wrapped_in_agent_generation_error(inner_exc):
    """Reproduces how smolagents actually surfaces a model-call failure: its
    agent step catches whatever the model raises and re-raises it as
    AgentGenerationError with `from e` (smolagents/agents.py), so the real
    litellm/HTTP error ends up in __cause__, not on the exception our code
    sees directly. AgentGenerationError's __init__ calls logger.log_error,
    so it needs a (mock) logger, not a real message-only construction."""
    from smolagents.utils import AgentGenerationError

    fake_logger = MagicMock()
    try:
        raise inner_exc
    except Exception as e:
        try:
            raise AgentGenerationError("Error while generating output", fake_logger) from e
        except AgentGenerationError as wrapped:
            return wrapped


def test_classify_llm_error_unwraps_rate_limit_through_agent_generation_error():
    import litellm

    from src.detection.llm.layer import _classify_llm_error

    inner = litellm.exceptions.RateLimitError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) == "rate_limited"


def test_classify_llm_error_unwraps_service_unavailable_through_agent_generation_error():
    import litellm

    from src.detection.llm.layer import _classify_llm_error

    inner = litellm.exceptions.ServiceUnavailableError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) == "provider_unavailable"


def test_classify_llm_error_unwraps_authentication_error_as_not_transient():
    import litellm

    from src.detection.llm.layer import _classify_llm_error

    inner = litellm.exceptions.AuthenticationError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) is None


# --- same, end-to-end through a per-event call ------------------------------

@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_rate_limit_is_retried_and_keeps_the_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    inner = litellm.exceptions.RateLimitError(message="x", llm_provider="gemini", model="m")
    mock_agent_cls.return_value.run.side_effect = [_wrapped_in_agent_generation_error(inner), valid_llm_json()]

    agent = make_llm_agent()
    result = agent.run(make_event())

    assert result.detector_notes.startswith("[category=Unknown]")
    assert mock_agent_cls.return_value.run.call_count == 2
    mock_sleep.assert_called_once_with(2.0)
    assert not any(a in FALLBACK_REASON_BY_ACTION for a in (s.action for s in result.trace))  # no fallback: it succeeded
    mock_mcp_client_cls.assert_called_once()  # provider-side error: connection untouched
    assert agent._llm._mcp_client is not None


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_service_unavailable_exhausting_retries_keeps_the_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    make_wrapped = lambda: _wrapped_in_agent_generation_error(
        litellm.exceptions.ServiceUnavailableError(message="x", llm_provider="gemini", model="m")
    )
    mock_agent_cls.return_value.run.side_effect = [make_wrapped() for _ in range(4)]

    agent = make_llm_agent()
    result = agent.run(make_event())

    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["provider_unavailable"]
    assert mock_agent_cls.return_value.run.call_count == 4  # 1 + 3 retries
    mock_mcp_client_cls.assert_called_once()  # provider-side: connection kept, not disconnected
    assert agent._llm._mcp_client is not None


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm.layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_authentication_error_is_not_retried_and_disconnects(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    inner = litellm.exceptions.AuthenticationError(message="bad key", llm_provider="gemini", model="m")
    mock_agent_cls.return_value.run.side_effect = _wrapped_in_agent_generation_error(inner)

    agent = make_llm_agent()
    result = agent.run(make_event())  # must not raise

    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["exception"]  # not transient -> no special reason, no retry
    assert mock_agent_cls.return_value.run.call_count == 1  # not retried
    mock_sleep.assert_not_called()
    assert agent._llm._mcp_client is None  # not a recognized provider error: connection dropped


# --- diagnostics: raw text of failed steps + forced final answer ------------
# _log_agent_memory_diagnostics inspects agent.memory.steps (real ActionStep
# objects, not the mocked agent itself) after agent.run() returns, so a real
# smolagents Memory/ActionStep is used here rather than mocking .memory too.

def _make_action_step(step_number, error=None, model_output=None, is_final_answer=False):
    from smolagents.memory import ActionStep, Timing

    return ActionStep(
        step_number=step_number, timing=Timing(start_time=0.0, end_time=0.1),
        error=error, model_output=model_output, is_final_answer=is_final_answer,
    )


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_failed_steps_and_forced_final_answer_are_logged(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    from smolagents.utils import AgentMaxStepsError, AgentParsingError

    fake_logger = MagicMock()
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    forced_answer = 'Sure, here you go: ' + valid_llm_json()
    mock_agent_cls.return_value.run.return_value = forced_answer
    mock_agent_cls.return_value.memory.steps = [
        _make_action_step(1, model_output='{"name": "top_features", "arguments": {}}'),
        _make_action_step(
            2, error=AgentParsingError("no JSON blob", fake_logger), model_output="I think this is fine.",
        ),
        _make_action_step(3, error=AgentMaxStepsError("Reached max steps.", fake_logger), model_output=None),
    ]

    result = make_llm_agent().run(make_event())

    step_failed = [s for s in result.trace if s.action == "llm_step_failed"]
    # step 1 succeeded (no error, excluded); step 2 failed to parse; step 3 is the synthetic
    # max-steps wrap-up (also carries an error, with no model_output of its own).
    assert [s.tool_input for s in step_failed] == [{"step_number": 2}, {"step_number": 3}]
    assert "I think this is fine." in step_failed[0].observation
    assert "raw_text=None" in step_failed[1].observation

    forced = next(s for s in result.trace if s.action == "llm_forced_final_answer")
    assert "Sure, here you go" in forced.observation
    assert result.detector_notes.startswith("[category=Unknown]")  # salvaged from the forced answer's prose


@patch("src.detection.llm.layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_no_diagnostics_logged_when_final_answer_tool_was_used_cleanly(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_top_features, mock_get_model
):
    mock_predict.return_value = 0.95
    mock_mcp_client_cls.return_value.get_tools.return_value = fake_mcp_tools()
    mock_agent_cls.return_value.run.return_value = valid_llm_json()
    mock_agent_cls.return_value.memory.steps = [
        _make_action_step(1, model_output='{"name": "top_features", "arguments": {}}'),
        _make_action_step(2, model_output=valid_llm_json(), is_final_answer=True),
    ]

    result = make_llm_agent().run(make_event())

    assert not any(s.action in ("llm_step_failed", "llm_forced_final_answer") for s in result.trace)
    assert any(s.action == "llm_validation_passed" for s in result.trace)
