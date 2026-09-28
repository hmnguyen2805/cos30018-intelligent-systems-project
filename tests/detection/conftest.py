"""
Shared fixtures/helpers for the tests/detection/ suite. Lifted out of the
old test_detection_subagent.py (now split into test_subagent.py,
llm/test_layer.py, and llm/test_modes.py) because each of those files needs
them — kept here once instead of duplicated per file.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.detection.subagent import DetectionSubagent
from src.shared.schemas import TrafficEvent


@pytest.fixture(autouse=True)
def _default_llm_env(monkeypatch):
    """Tests must not depend on the real .env's DETECTION_LLM_MODEL/_MODE — many
    exercise the real dispatch path (LLMExplanationLayer._call_llm_agent is mocked,
    but resolve_llm_mode()/resolve_llm_model_id() run for real), and mode now
    defaults based on the model id (ollama_chat/* -> single_shot, else agent — see
    llm.layer.resolve_llm_mode). Force a deterministic non-Ollama placeholder here
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
