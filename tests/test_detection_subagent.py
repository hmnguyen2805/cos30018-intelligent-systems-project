"""
Unit tests for DetectionSubagent — the borderline-handling loop that makes
Detection an agent rather than a bare classifier call. classifier.py's
functions are monkeypatched so no trained model artifact is needed.

The LLM-layer tests further down mock LLMExplanationLayer._call_llm_agent (the
piece that talks to the MCP subprocess/smolagents, now in llm_layer.py — see
that module for the connection/retry/salvage plumbing) so they run offline
and fast, while exercising the real validation/fallback/timeout wiring that
DetectionSubagent delegates to it.
"""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.detection.llm_layer import LLMExplanationLayer
from src.detection.subagent import DetectionSubagent
from src.shared.schemas import TrafficEvent


@pytest.fixture(autouse=True)
def _default_llm_env(monkeypatch):
    """Tests must not depend on the real .env's DETECTION_LLM_MODEL/_MODE — many
    exercise the real dispatch path (LLMExplanationLayer._call_llm_agent is mocked,
    but resolve_llm_mode()/resolve_llm_model_id() run for real), and mode now
    defaults based on the model id (ollama_chat/* -> single_shot, else agent — see
    llm_layer.resolve_llm_mode). Force a deterministic non-Ollama placeholder here
    so "agent mode" tests actually get agent mode unless a test explicitly
    overrides DETECTION_LLM_MODEL and/or DETECTION_LLM_MODE itself."""
    monkeypatch.setenv("DETECTION_LLM_MODEL", "test-provider/test-model")
    monkeypatch.delenv("DETECTION_LLM_MODE", raising=False)


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
        "explanation": "duration and packet_count both look elevated for this event.",
        "tools_used": ["top_features"],
    })


def _make_rate_limit_error():
    class FakeRateLimitError(Exception):
        status_code = 429
    return FakeRateLimitError("rate limited")


def _make_provider_unavailable_error():
    class FakeServiceUnavailableError(Exception):
        status_code = 503
    return FakeServiceUnavailableError("service unavailable")


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


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_clear_benign_event_with_use_llm_true_skips_the_llm(mock_predict, mock_call_llm):
    mock_predict.return_value = 0.05  # below LLM_TRIGGER_THRESHOLD (0.4)
    agent = make_llm_agent()

    result = agent.run(make_event())

    mock_call_llm.assert_not_called()
    assert result.is_anomalous is False
    assert result.detector_notes is None


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
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
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", return_value="not valid json")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_invalid_json_falls_back_to_template_note(mock_predict, mock_call_llm, mock_top_features):
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95
    assert result.detector_notes == "[category=Unknown] Anomalous event — LLM explanation unavailable."


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm_layer.LLMExplanationLayer._call_llm_agent",
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
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("MCP server crashed"))
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
    with patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", slow_call):
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
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
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


def fake_mcp_tools():
    """Fake tool objects shaped like the 6 real MCP tools, for tests where
    MCPClient itself is mocked. Plain MagicMock() attributes, so each is both
    callable and inspectable via .name (Mock's own `name=` kwarg is reserved
    for repr, not the attribute, hence setting it after construction)."""
    tools = []
    for tool_name in ("register_event", "clear_event", "predict_proba_anomalous",
                       "tree_vote_spread", "top_features", "predict_attack_category"):
        tool = MagicMock()
        tool.name = tool_name
        tools.append(tool)
    return tools


# --- MCP connection lifecycle ---------------------------------------------
# Here _call_llm_agent runs for real (not mocked) so _ensure_llm_connection is
# exercised; only the MCP/smolagents boundary (MCPClient, ToolCallingAgent) is
# mocked, so no subprocess or real model is involved.

@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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

@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.llm_layer.time.sleep")  # skip real backoff delays
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.llm_layer.time.sleep")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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
    with patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", slow_call):
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

    with patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", slow_call):
        first = agent.run(make_event())  # times out

    with patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json()):
        second = agent.run(make_event())  # fresh event, fast success

    assert not any(step.action == "should_never_leak" for step in first.trace)
    assert not any(step.action == "should_never_leak" for step in second.trace)


# --- circuit breaker ---------------------------------------------------------

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("dead"))
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
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent")
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

@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
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

    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

    agent = make_llm_agent()
    result = agent.run(make_event())

    assert "category=Unknown" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["rate_limited"]
    assert agent._llm._mcp_client is not None  # provider-side: connection kept, not disconnected


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
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

    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

    agent = make_llm_agent()
    result = agent.run(make_event())

    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["provider_unavailable"]


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", side_effect=RuntimeError("broken pipe"))
@patch("src.detection.classifier.predict_proba_anomalous")
def test_non_transient_exception_still_disconnects_the_connection(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

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
    with patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", slow_call):
        agent = make_llm_agent(llm_timeout_seconds=0.02)
        result = agent.run(make_event())

    # llm_dispatch (unbuffered, logged before the timed call) proves the event was sent to
    # the LLM even though llm_layer_start (buffered) never gets merged in on a timeout.
    assert any(step.action == "llm_dispatch" for step in result.trace)
    assert not any(step.action == "llm_layer_start" for step in result.trace)


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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

def test_classify_llm_error_unwraps_rate_limit_through_agent_generation_error():
    import litellm

    from src.detection.llm_layer import _classify_llm_error

    inner = litellm.exceptions.RateLimitError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) == "rate_limited"


def test_classify_llm_error_unwraps_service_unavailable_through_agent_generation_error():
    import litellm

    from src.detection.llm_layer import _classify_llm_error

    inner = litellm.exceptions.ServiceUnavailableError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) == "provider_unavailable"


def test_classify_llm_error_unwraps_authentication_error_as_not_transient():
    import litellm

    from src.detection.llm_layer import _classify_llm_error

    inner = litellm.exceptions.AuthenticationError(message="x", llm_provider="gemini", model="m")
    wrapped = _wrapped_in_agent_generation_error(inner)

    assert _classify_llm_error(wrapped) is None


# --- same, end-to-end through a per-event call ------------------------------

@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_rate_limit_is_retried_and_keeps_the_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_service_unavailable_exhausting_retries_keeps_the_connection(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
@patch("smolagents.ToolCallingAgent")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_wrapped_authentication_error_is_not_retried_and_disconnects(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_agent_cls, mock_sleep,
    mock_top_features, mock_get_model,
):
    import litellm

    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

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


# --- salvage path: model answered in plain text instead of via final_answer -
# When ToolCallingAgent exhausts max_steps without ever calling final_answer
# (e.g. a small model just typed its JSON answer as a chat message), smolagents
# itself falls back to one direct generation and returns that raw text as-is —
# often the right JSON with some stray prose around it. _run_llm_layer re-runs
# parse_and_validate on an extracted {...} substring before giving up.

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm_layer.LLMExplanationLayer._call_llm_agent",
    return_value='Sure, based on my investigation, here is the result: ' + valid_llm_json() + ' Hope that helps!',
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_extracts_json_from_surrounding_prose(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert result.is_anomalous is True
    assert result.confidence == 0.95  # unchanged by the salvage path
    assert result.detector_notes.startswith("[category=Unknown]")
    assert any(step.action == "llm_validation_salvaged" for step in result.trace)
    assert not any(step.action == "llm_validation_passed" for step in result.trace)
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["salvaged"]


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm_layer.LLMExplanationLayer._call_llm_agent",
    return_value="I'm not sure how to answer that question.",
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_gives_up_when_no_json_is_present(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert "category=Unknown" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["invalid_json"]  # extraction found nothing to salvage


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm_layer.LLMExplanationLayer._call_llm_agent",
    return_value='My answer: {"category": "Ransomware", "explanation": "x", "tools_used": []}',
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_ignores_a_stray_category_field_in_the_salvaged_json(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm_layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    # A stray "category" in the salvaged content has no schema slot to violate — it's simply
    # ignored, same as it would be in a clean (non-salvaged) answer. The code-side decision
    # ("Unknown", no category model loaded) still wins, and the salvage itself succeeds.
    assert "category=Unknown" in result.detector_notes
    assert "x" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["salvaged"]


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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


@patch("src.detection.llm_layer.LLMExplanationLayer._get_llm_model", return_value="fake-model")
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


# --- DETECTION_LLM_MODE=single_shot -----------------------------------------
# litellm's own ollama transformation drops tool_choice entirely ("causes ollama
# requests to hang" — see subagent.py's DEFAULT_LLM_MODE comment), so the agent
# loop's "every step must be a tool call" requirement can't be forced onto a
# small local model. single_shot instead has code call top_features (and
# tree_vote_spread, if borderline) directly, then makes one litellm.completion
# call with response_format's json_schema — verified for real against Ollama.

def _fake_litellm_response(content, prompt_tokens=10, completion_tokens=20):
    from types import SimpleNamespace

    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_resolve_llm_mode_defaults_to_agent(monkeypatch):
    from src.detection.llm_layer import resolve_llm_mode

    monkeypatch.delenv("DETECTION_LLM_MODE", raising=False)
    assert resolve_llm_mode() == "agent"


def test_resolve_llm_mode_reads_env_var(monkeypatch):
    from src.detection.llm_layer import resolve_llm_mode

    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    assert resolve_llm_mode() == "single_shot"


def test_resolve_llm_mode_falls_back_to_agent_on_invalid_value(monkeypatch):
    from src.detection.llm_layer import resolve_llm_mode

    monkeypatch.setenv("DETECTION_LLM_MODE", "bogus")
    assert resolve_llm_mode() == "agent"


def test_resolve_llm_mode_defaults_to_single_shot_for_an_ollama_model(monkeypatch):
    from src.detection.llm_layer import resolve_llm_mode

    monkeypatch.delenv("DETECTION_LLM_MODE", raising=False)
    monkeypatch.setenv("DETECTION_LLM_MODEL", "ollama_chat/qwen2.5:3b")
    assert resolve_llm_mode() == "single_shot"


def test_resolve_llm_mode_env_var_override_wins_even_for_an_ollama_model(monkeypatch):
    from src.detection.llm_layer import resolve_llm_mode

    monkeypatch.setenv("DETECTION_LLM_MODEL", "ollama_chat/qwen2.5:3b")
    monkeypatch.setenv("DETECTION_LLM_MODE", "agent")
    assert resolve_llm_mode() == "agent"


def test_unwrap_mcp_result_unwraps_the_result_key():
    assert LLMExplanationLayer._unwrap_mcp_result({"result": [1, 2, 3]}) == [1, 2, 3]


def test_unwrap_mcp_result_passes_through_an_object_shaped_result():
    value = {"vote_fraction": 0.5, "vote_std": 0.1}
    assert LLMExplanationLayer._unwrap_mcp_result(value) is value


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("litellm.completion")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_single_shot_mode_calls_top_features_directly_with_no_agent_loop(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_completion, mock_top_features, monkeypatch
):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.95
    tools = fake_mcp_tools()
    top_features_tool = next(t for t in tools if t.name == "top_features")
    top_features_tool.return_value = {"result": [{"name": "duration", "value": 1.0, "median": None}]}
    mock_mcp_client_cls.return_value.get_tools.return_value = tools
    mock_completion.return_value = _fake_litellm_response(valid_llm_json())

    with patch("smolagents.ToolCallingAgent") as mock_agent_cls:
        result = make_llm_agent().run(make_event())
        mock_agent_cls.assert_not_called()  # single_shot never builds a ToolCallingAgent

    assert result.detector_notes.startswith("[category=Unknown]")
    top_features_tool.assert_called_once()
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert "duration" in str(call_kwargs["messages"])  # feature data embedded in the prompt


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("litellm.completion")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.tree_vote_spread")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_single_shot_mode_includes_tree_vote_spread_when_borderline(
    mock_predict, mock_votes, mock_server_params, mock_mcp_client_cls, mock_completion,
    mock_top_features, monkeypatch,
):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.5
    mock_votes.return_value = (0.55, 0.1)
    tools = fake_mcp_tools()
    next(t for t in tools if t.name == "top_features").return_value = {"result": []}
    vote_tool = next(t for t in tools if t.name == "tree_vote_spread")
    vote_tool.return_value = {"vote_fraction": 0.55, "vote_std": 0.1}
    mock_mcp_client_cls.return_value.get_tools.return_value = tools
    mock_completion.return_value = _fake_litellm_response(valid_llm_json())

    make_llm_agent().run(make_event())

    vote_tool.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert "vote_fraction" in str(call_kwargs["messages"])


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("litellm.completion")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_single_shot_mode_records_token_usage(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_completion, mock_top_features, monkeypatch
):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.95
    tools = fake_mcp_tools()
    next(t for t in tools if t.name == "top_features").return_value = {"result": []}
    mock_mcp_client_cls.return_value.get_tools.return_value = tools
    mock_completion.return_value = _fake_litellm_response(valid_llm_json(), prompt_tokens=42, completion_tokens=17)

    result = make_llm_agent().run(make_event())

    usage_step = next(s for s in result.trace if s.action == "llm_token_usage")
    assert usage_step.tool_input == {"prompt_tokens": 42, "completion_tokens": 17}


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.time.sleep")
@patch("litellm.completion")
@patch("smolagents.MCPClient")
@patch("mcp.StdioServerParameters")
@patch("src.detection.classifier.predict_proba_anomalous")
def test_single_shot_mode_retries_a_transient_error_then_succeeds(
    mock_predict, mock_server_params, mock_mcp_client_cls, mock_completion, mock_sleep,
    mock_top_features, monkeypatch,
):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.95
    tools = fake_mcp_tools()
    next(t for t in tools if t.name == "top_features").return_value = {"result": []}
    mock_mcp_client_cls.return_value.get_tools.return_value = tools
    mock_completion.side_effect = [_make_rate_limit_error(), _fake_litellm_response(valid_llm_json())]

    result = make_llm_agent().run(make_event())

    assert result.detector_notes.startswith("[category=Unknown]")
    assert mock_completion.call_count == 2
    mock_sleep.assert_called_once_with(2.0)


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch("src.detection.llm_layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_dispatch_step_records_the_resolved_mode(mock_predict, mock_call_llm, mock_top_features, monkeypatch):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    dispatch_step = next(s for s in result.trace if s.action == "llm_dispatch")
    assert dispatch_step.tool_input["mode"] == "single_shot"


# --- _choose_category: the deterministic category decision -------------------
# Same principle as is_anomalous/confidence: code, not the LLM, decides the
# category. These tests configure a real category artifact (unlike the rest of
# this file, which always falls back to "Unknown" with no artifact loaded).

def _make_category_artifact(classes_and_probas):
    """A fake category_model whose predict_proba always returns the given
    (class, probability) pairs, in that order, regardless of input."""
    classes = [c for c, _ in classes_and_probas]
    probas = [p for _, p in classes_and_probas]
    model = MagicMock()
    model.classes_ = classes
    model.predict_proba.return_value = [probas]
    return {"category_model": model, "classes": classes, "feature_names": ["duration", "packet_count"]}


@patch("src.detection.classifier.load_artifact")
def make_agent_with_category_model(classes_and_probas, mock_load, **kwargs):
    # patch() appends the mock as the last positional arg (after ones the
    # caller passes explicitly), so classes_and_probas must come first here.
    binary_artifact = {"model": None, "feature_names": ["duration", "packet_count"]}
    category_artifact = _make_category_artifact(classes_and_probas)
    mock_load.side_effect = [binary_artifact, category_artifact]
    return DetectionSubagent(model_path="unused", **kwargs)


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_uses_top_class_when_above_threshold(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model([("DDoS", 0.9), ("PortScan", 0.1)])

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_falls_back_to_unknown_below_threshold(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model([("DDoS", 0.55), ("PortScan", 0.45)])

    result = agent.run(make_event())

    assert result.detector_notes == "[category=Unknown]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_threshold_is_configurable(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("DDoS", 0.55), ("PortScan", 0.45)], category_confidence_threshold=0.5,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"  # 0.55 clears a lowered 0.5 threshold


@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_at_exactly_the_threshold_is_accepted(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model([("DDoS", 0.6), ("PortScan", 0.4)])  # default threshold is 0.6

    result = agent.run(make_event())

    assert result.detector_notes == "[category=DDoS]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_event_gets_no_category_tag_even_with_a_category_model_loaded(mock_predict):
    mock_predict.return_value = 0.05  # clearly benign
    agent = make_agent_with_category_model([("DDoS", 0.9), ("PortScan", 0.1)])

    result = agent.run(make_event())

    assert result.is_anomalous is False
    assert result.detector_notes is None


@patch("src.detection.classifier.predict_proba_anomalous")
def test_category_decision_trace_step_records_chosen_and_raw_top_and_threshold(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model([("DDoS", 0.55), ("PortScan", 0.45)])

    result = agent.run(make_event())

    step = next(s for s in result.trace if s.action == "category_decision")
    assert step.tool_input == {
        "chosen_category": "Unknown",
        "raw_top_category": "DDoS",
        "raw_top_probability": 0.55,
        "threshold": 0.6,
    }


@patch("src.detection.classifier.predict_proba_anomalous")
def test_category_model_unavailable_is_logged_as_its_own_trace_step(mock_predict):
    # make_agent() (no category artifact) hits the FileNotFoundError branch of
    # _load_category_artifact, so _choose_category takes the "no model" path.
    mock_predict.return_value = 0.95
    result = make_agent().run(make_event())

    assert any(s.action == "category_model_unavailable" for s in result.trace)
    assert not any(s.action == "category_decision" for s in result.trace)


# --- Benign class / binary-category model disagreement -----------------------
# train_category.py now trains an explicit "Benign" class alongside the attack
# categories (see its module docstring): when it's the category model's top
# vote for an event the BINARY model already called anomalous, that's a
# disagreement between the two models — reported as "Unknown" (never
# "Benign", which would contradict is_anomalous=True) and logged as its own
# "model_disagreement" trace step, distinct from an ordinary below-threshold
# "Unknown" (see test_choose_category_falls_back_to_unknown_below_threshold).

@patch("src.detection.classifier.predict_proba_anomalous")
def test_choose_category_reports_unknown_when_category_model_votes_benign(mock_predict):
    mock_predict.return_value = 0.95  # binary model: anomalous
    agent = make_agent_with_category_model([("Benign", 0.99), ("DDoS", 0.01)])

    result = agent.run(make_event())

    assert result.is_anomalous is True
    assert result.detector_notes == "[category=Unknown]"


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_top_vote_is_logged_as_model_disagreement_not_category_decision(mock_predict):
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model([("Benign", 0.99), ("DDoS", 0.01)])

    result = agent.run(make_event())

    disagreement_steps = [s for s in result.trace if s.action == "model_disagreement"]
    assert len(disagreement_steps) == 1
    assert disagreement_steps[0].tool_input == {
        "chosen_category": "Unknown", "raw_top_category": "Benign",
        "raw_top_probability": 0.99, "threshold": 0.6,
    }
    assert not any(s.action == "category_decision" for s in result.trace)


@patch("src.detection.classifier.predict_proba_anomalous")
def test_benign_top_vote_disagreement_is_reported_even_above_the_confidence_threshold(mock_predict):
    # A high-confidence Benign vote is still a disagreement, not a trustworthy category —
    # the confidence threshold only governs which ATTACK category to trust, it never makes
    # "Benign" an acceptable answer for an event the binary model called anomalous.
    mock_predict.return_value = 0.95
    agent = make_agent_with_category_model(
        [("Benign", 0.99), ("DDoS", 0.01)], category_confidence_threshold=0.1,
    )

    result = agent.run(make_event())

    assert result.detector_notes == "[category=Unknown]"
    assert any(s.action == "model_disagreement" for s in result.trace)


# --- assert_feature_names_match is enforced at construction time -------------

@patch(
    "src.detection.classifier.load_artifact",
    side_effect=[
        {"model": None, "feature_names": ["duration", "packet_count"]},
        {"category_model": MagicMock(), "classes": ["DDoS"], "feature_names": ["duration"]},
    ],
)
def test_mismatched_feature_names_between_artifacts_raises_on_construction(_mock_load):
    try:
        DetectionSubagent(model_path="unused")
        assert False, "expected a ValueError for mismatched feature_names"
    except ValueError as exc:
        assert "feature_names" in str(exc) or "feature set" in str(exc).lower()
