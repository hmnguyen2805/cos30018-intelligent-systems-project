"""
Unit tests for the two LLM interaction modes (src.detection.llm.layer's
DETECTION_LLM_MODE=agent vs single_shot) and the salvage path that recovers
JSON from a model's un-tooled, prose-wrapped answer.

Split out of the old test_detection_subagent.py — see test_subagent.py
(core flow + category decision) and test_layer.py (connection/timeout/
retry/circuit-breaker) for the rest.
"""
from unittest.mock import patch

from src.detection.llm.layer import LLMExplanationLayer
from tests.detection.conftest import (
    fake_mcp_tools,
    make_event,
    make_llm_agent,
    _make_rate_limit_error,
    valid_llm_json,
)


# --- salvage path: model answered in plain text instead of via final_answer -
# When ToolCallingAgent exhausts max_steps without ever calling final_answer
# (e.g. a small model just typed its JSON answer as a chat message), smolagents
# itself falls back to one direct generation and returns that raw text as-is —
# often the right JSON with some stray prose around it. _run_llm_layer re-runs
# parse_and_validate on an extracted {...} substring before giving up.

@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm.layer.LLMExplanationLayer._call_llm_agent",
    return_value='Sure, based on my investigation, here is the result: ' + valid_llm_json() + ' Hope that helps!',
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_extracts_json_from_surrounding_prose(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

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
    "src.detection.llm.layer.LLMExplanationLayer._call_llm_agent",
    return_value="I'm not sure how to answer that question.",
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_gives_up_when_no_json_is_present(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    assert "category=Unknown" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["invalid_json"]  # extraction found nothing to salvage


@patch("src.detection.classifier.top_features", return_value=[{"name": "duration"}, {"name": "packet_count"}])
@patch(
    "src.detection.llm.layer.LLMExplanationLayer._call_llm_agent",
    return_value='My answer: {"category": "Ransomware", "explanation": "x", "tools_used": []}',
)
@patch("src.detection.classifier.predict_proba_anomalous")
def test_salvage_path_ignores_a_stray_category_field_in_the_salvaged_json(mock_predict, mock_call_llm, mock_top_features):
    from src.detection.llm.layer import FALLBACK_REASON_BY_ACTION

    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    # A stray "category" in the salvaged content has no schema slot to violate — it's simply
    # ignored, same as it would be in a clean (non-salvaged) answer. The code-side decision
    # ("Unknown", no category model loaded) still wins, and the salvage itself succeeds.
    assert "category=Unknown" in result.detector_notes
    assert "x" in result.detector_notes
    fallback_actions = [FALLBACK_REASON_BY_ACTION[s.action] for s in result.trace if s.action in FALLBACK_REASON_BY_ACTION]
    assert fallback_actions == ["salvaged"]


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
    from src.detection.llm.layer import resolve_llm_mode

    monkeypatch.delenv("DETECTION_LLM_MODE", raising=False)
    assert resolve_llm_mode() == "agent"


def test_resolve_llm_mode_reads_env_var(monkeypatch):
    from src.detection.llm.layer import resolve_llm_mode

    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    assert resolve_llm_mode() == "single_shot"


def test_resolve_llm_mode_falls_back_to_agent_on_invalid_value(monkeypatch):
    from src.detection.llm.layer import resolve_llm_mode

    monkeypatch.setenv("DETECTION_LLM_MODE", "bogus")
    assert resolve_llm_mode() == "agent"


def test_resolve_llm_mode_defaults_to_single_shot_for_an_ollama_model(monkeypatch):
    from src.detection.llm.layer import resolve_llm_mode

    monkeypatch.delenv("DETECTION_LLM_MODE", raising=False)
    monkeypatch.setenv("DETECTION_LLM_MODEL", "ollama_chat/qwen2.5:3b")
    assert resolve_llm_mode() == "single_shot"


def test_resolve_llm_mode_env_var_override_wins_even_for_an_ollama_model(monkeypatch):
    from src.detection.llm.layer import resolve_llm_mode

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
@patch("src.detection.llm.layer.time.sleep")
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
@patch("src.detection.llm.layer.LLMExplanationLayer._call_llm_agent", return_value=valid_llm_json())
@patch("src.detection.classifier.predict_proba_anomalous")
def test_llm_dispatch_step_records_the_resolved_mode(mock_predict, mock_call_llm, mock_top_features, monkeypatch):
    monkeypatch.setenv("DETECTION_LLM_MODE", "single_shot")
    mock_predict.return_value = 0.95
    result = make_llm_agent().run(make_event())

    dispatch_step = next(s for s in result.trace if s.action == "llm_dispatch")
    assert dispatch_step.tool_input["mode"] == "single_shot"
