"""
Tests for the Mitigation Manager's agent mode (src/correlation/agent_loop.py and
the mode switch in manager.py). The LLM is replaced by ScriptedLLM, which returns
pre-written replies in order, so these run without Ollama or any network access.
"""
import copy
import json
import os
from unittest.mock import patch

import numpy as np

from src.correlation import agent_loop
from src.correlation.manager import (ACTION_BY_TECHNIQUE, NO_MATCH_ACTION, NO_THREAT_ACTION,
                                     MitigationManager, mitigation_from_env)
from src.correlation.subagent import CATEGORY_MATCH_CONFIDENCE, CorrelationSubagent
from src.correlation.technique_catalog import CATEGORY_TECHNIQUES, TECHNIQUE_CATALOG
from src.response.llm import LLMConfig, LLMResponse
from src.response.rules import MITIGATION_MIN_CONFIDENCE
from tests.fakes import make_event


class ScriptedLLM:
    """Stands in for the model: returns scripted replies (dicts are JSON-encoded,
    exceptions are raised) and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, schema, schema_name):
        self.calls.append({"messages": copy.deepcopy(messages), "schema_name": schema_name})
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(content=content, prompt_tokens=100, completion_tokens=20, latency_ms=50.0)


class KeywordEncoder:
    """Fake embedding model: a text becomes a vector of keyword counts, so a query
    containing "login" matches the brute-force technique and nothing else."""
    WORDS = ["login", "flood", "scan", "exploit", "message", "stolen"]

    def encode(self, text):
        if isinstance(text, (list, tuple)):
            return [self.encode(t) for t in text]
        lower = text.lower()
        return np.array([lower.count(w) for w in self.WORDS], dtype=float) + 0.001


def detection(category="Infiltration", anomalous=True, confidence=0.9):
    from src.shared.schemas import DetectionResult
    notes = f"[category={category}] [label={category}] Single flow with unusual traffic." if anomalous else "Normal."
    return DetectionResult(
        event=make_event(), is_anomalous=anomalous, confidence=confidence, detector_notes=notes,
        attack_category=category if anomalous else None, attack_label=category if anomalous else None,
        category_confidence=0.85 if anomalous else None,
        traffic_summary="One flow to port 22 with many short packets." if anomalous else None,
    )


def manager(llm, mode="agent", max_steps=6, subagent=None):
    return MitigationManager(subagent=subagent or CorrelationSubagent(encoder=KeywordEncoder()),
                             mode=mode, llm_client=llm, config=LLMConfig(max_steps=max_steps))


def step(action, **extra):
    return {"thought": f"next: {action}", "action": action, **extra}


EVIDENCE = step("get_detection_evidence")


def choose(*ids, reasoning="Best fit."):
    return step("choose_technique", technique_ids=list(ids), reasoning=reasoning)


def actions(result):
    return [s.action for s in result.trace]


def first_message(llm):
    return llm.calls[0]["messages"][1]["content"]


# --- the agent picks among several candidates --------------------------------------

def test_agent_picks_one_of_several_candidates_and_action_comes_from_the_table():
    candidates = CATEGORY_TECHNIQUES["Infiltration"]
    assert len(candidates) == 4
    pick = candidates[2]  # not the first one the rules path would have proposed
    llm = ScriptedLLM(EVIDENCE, step("list_candidate_techniques"), choose(pick))
    result = manager(llm).run(detection())

    assert result.correlation.matched_technique_ids == [pick]
    assert result.proposed_action == ACTION_BY_TECHNIQUE[pick]
    assert result.confidence == CATEGORY_MATCH_CONFIDENCE
    assert actions(result) == ["get_detection_evidence", "list_candidate_techniques", "llm_choose_technique",
                               "decide_action"]
    # the tool result is fed back as an observation, and the schema name is the Mitigation one
    assert any("Observation (list_candidate_techniques)" in m["content"] for m in llm.calls[-1]["messages"])
    assert llm.calls[0]["schema_name"] == "mitigation_step"


def test_llm_cannot_set_the_action_text_or_the_confidence():
    reply = choose("T1190")
    reply["proposed_action"] = "Format the disk."
    reply["confidence"] = 0.99
    result = manager(ScriptedLLM(EVIDENCE, reply)).run(detection(category="Infiltration"))
    assert result.proposed_action == ACTION_BY_TECHNIQUE["T1190"]
    assert result.confidence == CATEGORY_MATCH_CONFIDENCE


def test_usage_is_recorded():
    m = manager(ScriptedLLM(EVIDENCE, choose("T1190")))
    m.run(detection())
    assert m.last_usage["llm_calls"] == 2 and m.last_usage["prompt_tokens"] == 200


# --- invalid replies ------------------------------------------------------------------

def test_unknown_technique_is_rejected_and_the_llm_recovers():
    llm = ScriptedLLM(EVIDENCE, choose("T9999"), choose("T1190"))
    m = manager(llm)
    result = m.run(detection())

    assert result.correlation.matched_technique_ids == ["T1190"]
    assert "invalid_llm_reply" in actions(result)
    # the model was told why its reply was rejected
    rejection = llm.calls[2]["messages"][-1]["content"]
    assert "not valid" in rejection and "T9999" in rejection
    assert m.last_usage["invalid_replies"] == 1 and m.last_usage["recovered_replies"] == 1


def test_two_invalid_replies_in_a_row_fall_back_to_the_rules_path():
    d = detection()
    result = manager(ScriptedLLM("this is not json", step("dance"))).run(d)
    expected = manager(ScriptedLLM(), mode="rules").run(d)

    assert "fallback_to_rules" in actions(result)
    assert result.proposed_action == expected.proposed_action
    assert result.confidence == expected.confidence
    assert result.correlation.matched_technique_ids == expected.correlation.matched_technique_ids


def test_a_valid_reply_between_invalid_ones_resets_the_count():
    llm = ScriptedLLM("oops", EVIDENCE, step("dance"), choose("T1190"))
    result = manager(llm).run(detection())
    assert "fallback_to_rules" not in actions(result)
    assert result.correlation.matched_technique_ids == ["T1190"]


def test_choosing_before_get_detection_evidence_is_rejected():
    llm = ScriptedLLM(choose("T1190"), EVIDENCE, choose("T1190"))
    result = manager(llm).run(detection())

    assert actions(result)[:2] == ["invalid_llm_reply", "get_detection_evidence"]
    assert "get_detection_evidence before" in llm.calls[1]["messages"][-1]["content"]
    assert result.correlation.matched_technique_ids == ["T1190"]


def test_no_match_also_needs_the_evidence_first():
    llm = ScriptedLLM(step("no_match", reasoning="Nothing fits."), step("no_match", reasoning="Nothing fits."))
    result = manager(llm).run(detection())
    assert "fallback_to_rules" in actions(result)  # two rejections in a row


# --- failures fall back to the rules path --------------------------------------------

def test_llm_error_falls_back_to_the_rules_path():
    d = detection()
    result = manager(ScriptedLLM(ConnectionError("ollama is not running"))).run(d)
    expected = manager(ScriptedLLM(), mode="rules").run(d)

    assert "llm_error" in actions(result) and "fallback_to_rules" in actions(result)
    assert result.proposed_action == expected.proposed_action
    fallback = next(s for s in result.trace if s.action == "fallback_to_rules")
    assert fallback.tool_input["reason"] == "llm_error:ConnectionError"


def test_llm_timeout_falls_back_to_the_rules_path():
    m = manager(ScriptedLLM(TimeoutError("timed out")))
    result = m.run(detection())
    assert next(s for s in result.trace if s.action == "fallback_to_rules").tool_input["reason"] == \
        "llm_error:TimeoutError"
    assert m.last_usage["fallback_reason"] == "llm_error:TimeoutError"
    assert result.proposed_action == ACTION_BY_TECHNIQUE[CATEGORY_TECHNIQUES["Infiltration"][0]]


def test_step_limit_falls_back_to_the_rules_path():
    # three different tool calls, never a final answer
    llm = ScriptedLLM(EVIDENCE, step("list_candidate_techniques"), step("get_technique", technique_id="T1190"))
    result = manager(llm, max_steps=3).run(detection())
    fallback = next(s for s in result.trace if s.action == "fallback_to_rules")
    assert fallback.tool_input["reason"] == "max_steps"


# --- repeated tool calls ---------------------------------------------------------------------

def test_repeating_the_same_tool_call_twice_falls_back_early():
    llm = ScriptedLLM(EVIDENCE, EVIDENCE, EVIDENCE)
    result = manager(llm, max_steps=6).run(detection())

    assert len(llm.calls) == 3  # stopped after the second repeat, not after all 6 steps
    fallback = next(s for s in result.trace if s.action == "fallback_to_rules")
    assert fallback.tool_input["reason"] == "invalid_reply:repeated_call"
    assert actions(result).count("get_detection_evidence") == 1  # the repeats were not re-run


def test_one_repeat_is_tolerated_and_the_llm_can_recover():
    llm = ScriptedLLM(EVIDENCE, EVIDENCE, choose("T1190"))
    m = manager(llm)
    result = m.run(detection())

    assert result.correlation.matched_technique_ids == ["T1190"]
    assert "do not repeat" in llm.calls[2]["messages"][-1]["content"].lower()
    assert m.last_usage["invalid_replies"] == 1 and m.last_usage["recovered_replies"] == 1


def test_same_tool_with_different_input_is_not_a_repeat():
    llm = ScriptedLLM(EVIDENCE, step("get_technique", technique_id="T1190"),
                      step("get_technique", technique_id="T1566"),
                      step("search_catalog", query="repeated login attempts"),
                      step("search_catalog", query="flood of requests"), choose("T1190"))
    result = manager(llm, max_steps=6).run(detection())
    assert "invalid_llm_reply" not in actions(result)


def test_repeated_search_is_caught_even_if_the_case_or_spacing_differs():
    llm = ScriptedLLM(EVIDENCE, step("search_catalog", query="Login attempts"),
                      step("search_catalog", query="  login   ATTEMPTS "), choose("T1110"))
    result = manager(llm).run(detection(category="DoS"))
    assert actions(result).count("invalid_llm_reply") == 1


# --- benign traffic ----------------------------------------------------------------------

def test_benign_traffic_never_calls_the_llm():
    llm = ScriptedLLM()  # any call raises AssertionError
    result = manager(llm).run(detection(anomalous=False, confidence=0.97))

    assert llm.calls == []
    assert result.correlation.matched_technique_ids == []
    assert result.proposed_action == NO_THREAT_ACTION
    assert "fallback_to_rules" not in actions(result)


# --- recheck ---------------------------------------------------------------------------------

def test_recheck_reason_is_in_the_llms_first_message():
    llm = ScriptedLLM(EVIDENCE, choose("T1190"))
    result = manager(llm).run(detection(), recheck_reason="Category does not fit the matched technique.")

    assert "Category does not fit the matched technique." in first_message(llm)
    assert "different" in first_message(llm)
    assert "recheck" in actions(result)  # the existing trace step is kept


def test_no_recheck_text_on_a_normal_run():
    llm = ScriptedLLM(EVIDENCE, choose("T1190"))
    manager(llm).run(detection())
    assert "RECHECK" not in first_message(llm)


def test_recheck_tells_the_llm_what_the_first_pass_chose():
    d = detection()
    # two ids, best first: only the first is acted on, so only it is named in the recheck
    llm = ScriptedLLM(EVIDENCE, choose("T1190", "T1566"), EVIDENCE, choose("T1566"))
    m = manager(llm)
    m.run(d)
    first_pass_calls = len(llm.calls)
    m.run(d, recheck_reason="Category does not fit the matched technique.")

    recheck_first_message = llm.calls[first_pass_calls]["messages"][1]["content"]
    assert "Your first pass chose: T1190." in recheck_first_message


def test_recheck_remembers_the_rules_answer_when_the_first_pass_fell_back():
    d = detection()
    llm = ScriptedLLM(ConnectionError("ollama is not running"), EVIDENCE, choose("T1566"))
    m = manager(llm)
    first = m.run(d)  # the LLM fails, so the rules path decides
    rules_choice = first.correlation.matched_technique_ids[0]
    m.run(d, recheck_reason="Category does not fit the matched technique.")

    assert f"Your first pass chose: {rules_choice}." in llm.calls[1]["messages"][1]["content"]


def test_recheck_of_a_different_detection_does_not_reuse_the_first_choice():
    llm = ScriptedLLM(EVIDENCE, choose("T1190"), EVIDENCE, choose("T1566"))
    m = manager(llm)
    m.run(detection())
    m.run(detection(), recheck_reason="Category does not fit the matched technique.")  # a new DetectionResult

    assert "first pass chose" not in llm.calls[2]["messages"][1]["content"]


# --- confidence comes from the evidence -----------------------------------------------------

def test_confidence_is_capped_outside_the_category_without_search_evidence():
    outside = next(t for t in TECHNIQUE_CATALOG if t not in CATEGORY_TECHNIQUES["DoS"])
    result = manager(ScriptedLLM(EVIDENCE, choose(outside))).run(detection(category="DoS"))

    assert result.correlation.matched_technique_ids == [outside]
    assert result.confidence == agent_loop.UNSUPPORTED_CONFIDENCE_CAP
    assert result.confidence < MITIGATION_MIN_CONFIDENCE  # so the Judge escalates
    assert result.proposed_action == ACTION_BY_TECHNIQUE[outside]


def test_confidence_comes_from_the_search_score_when_chosen_from_search_catalog():
    llm = ScriptedLLM(EVIDENCE, step("search_catalog", query="repeated login attempts"), choose("T1110"))
    result = manager(llm).run(detection(category="DoS"))  # T1110 is outside the DoS candidates

    search_step = next(s for s in result.trace if s.action == "search_catalog")
    assert search_step.tool_input == {"query": "repeated login attempts"}
    assert result.confidence > agent_loop.UNSUPPORTED_CONFIDENCE_CAP
    assert result.confidence > 0.9  # the fake encoder gives a near-perfect match
    assert result.correlation.matched_technique_ids == ["T1110"]


def test_confidence_follows_the_first_chosen_technique_only():
    outside = next(t for t in TECHNIQUE_CATALOG if t not in CATEGORY_TECHNIQUES["DoS"])
    inside = CATEGORY_TECHNIQUES["DoS"][0]
    result = manager(ScriptedLLM(EVIDENCE, choose(outside, inside))).run(detection(category="DoS"))
    assert result.confidence == agent_loop.UNSUPPORTED_CONFIDENCE_CAP
    assert result.proposed_action == ACTION_BY_TECHNIQUE[outside]


def test_search_catalog_returns_an_error_observation_without_an_embedding_model():
    sub = CorrelationSubagent()
    sub._encoder_error = "ImportError: sentence-transformers is not installed"  # as if loading failed
    llm = ScriptedLLM(EVIDENCE, step("search_catalog", query="login"), choose("T1190"))
    result = manager(llm, subagent=sub).run(detection())

    assert any("unavailable" in m["content"] for m in llm.calls[-1]["messages"])
    assert result.correlation.matched_technique_ids == ["T1190"]


# --- the other tools and final action ------------------------------------------------------

def test_get_technique_and_get_detection_evidence_return_real_data():
    llm = ScriptedLLM(EVIDENCE, step("get_technique", technique_id="T1190"),
                      step("get_technique", technique_id="T0000"), choose("T1190"))
    result = manager(llm).run(detection())

    evidence = next(s for s in result.trace if s.action == "get_detection_evidence").observation
    assert "traffic_summary" in evidence and "[category=" not in evidence  # tags are stripped
    assert TECHNIQUE_CATALOG["T1190"] in next(s for s in result.trace if s.action == "get_technique").observation


def test_get_technique_without_an_id_is_an_invalid_reply():
    llm = ScriptedLLM(EVIDENCE, step("get_technique"), choose("T1190"))
    result = manager(llm).run(detection())
    assert "invalid_llm_reply" in actions(result)


def test_no_match_gives_the_no_match_action_and_zero_confidence():
    llm = ScriptedLLM(EVIDENCE, step("no_match", reasoning="Nothing fits."))
    result = manager(llm).run(detection())
    assert result.correlation.matched_technique_ids == []
    assert result.proposed_action == NO_MATCH_ACTION and result.confidence == 0.0


class BrokenEncoder:
    """Fake embedding model that crashes, to test a tool failing inside the agent loop."""

    def encode(self, text):
        raise RuntimeError("encoder crashed")


def test_a_tool_crash_becomes_an_error_observation_and_the_run_continues():
    llm = ScriptedLLM(EVIDENCE, step("search_catalog", query="login"), choose("T1190"))
    result = manager(llm, subagent=CorrelationSubagent(encoder=BrokenEncoder())).run(detection())

    assert "fallback_to_rules" not in actions(result)  # the run was not abandoned
    assert result.correlation.matched_technique_ids == ["T1190"]
    assert any("search_catalog failed" in m["content"] for m in llm.calls[-1]["messages"])  # the LLM was told
    crashed = next(s for s in result.trace if s.action == "search_catalog")
    assert "RuntimeError" in crashed.observation


def test_more_than_four_technique_ids_are_cut_to_four():
    ids = list(TECHNIQUE_CATALOG)[:5]
    assert len(ids) == 5
    result = manager(ScriptedLLM(EVIDENCE, choose(*ids))).run(detection())

    assert result.correlation.matched_technique_ids == ids[:4]


# --- modes and configuration ---------------------------------------------------------------

def test_rules_is_the_default_and_never_creates_an_llm_client():
    m = MitigationManager(subagent=CorrelationSubagent(encoder=KeywordEncoder()))
    assert m.mode == "rules" and m._llm_client is None
    result = m.run(detection())
    assert result.proposed_action == ACTION_BY_TECHNIQUE[CATEGORY_TECHNIQUES["Infiltration"][0]]
    assert m._llm_client is None


def test_unknown_mode_is_rejected():
    try:
        MitigationManager(mode="magic")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_mitigation_from_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MITIGATION_MODE", None)
        assert mitigation_from_env().mode == "rules"
    with patch.dict(os.environ, {"MITIGATION_MODE": "agent"}):
        assert mitigation_from_env().mode == "agent"


def test_llm_settings_come_from_mitigation_env_vars():
    env = {"MITIGATION_LLM_MODEL": "ollama_chat/test", "MITIGATION_LLM_TIMEOUT": "12",
           "MITIGATION_LLM_MAX_STEPS": "4"}
    with patch.dict(os.environ, env):
        cfg = agent_loop.mitigation_llm_config()
    assert (cfg.model, cfg.timeout_seconds, cfg.max_steps) == ("ollama_chat/test", 12.0, 4)
    with patch.dict(os.environ, {}):
        for key in env:
            os.environ.pop(key, None)
        cfg = agent_loop.mitigation_llm_config()
    assert (cfg.model, cfg.timeout_seconds, cfg.max_steps) == ("ollama_chat/qwen2.5:3b", 60.0, 6)