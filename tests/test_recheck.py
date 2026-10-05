"""
Tests for the recheck path: the agent-mode Judge sending a case back to a
manager (src/response/agent.py, llm.py), the pipeline running that recheck
(src/pipeline.py), and the Mitigation Manager accepting recheck_reason.

The LLM is the ScriptedLLM from test_judge_llm.py, so no Ollama is needed.
"""
import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from src.correlation.manager import MitigationManager
from src.correlation.subagent import CorrelationSubagent
from src.pipeline import Pipeline
from src.response import rules
from src.response.agent import MODE_AGENT, MODE_RULES, MODE_SINGLE_SHOT
from src.response.llm import REQUEST_RECHECK, LLMOutputError, agent_step_schema, parse_agent_step
from src.shared.schemas import DetectionResult, JudgeInput
from tests.fakes import FakeMitigationManager, make_event, make_mitigation
from tests.test_judge_llm import CHECK, COMPARE, FINALIZE_BLOCK, ScriptedLLM, actions, judge, step


def det(notes, anomalous=True, confidence=0.9):
    return DetectionResult(event=make_event(), is_anomalous=anomalous, confidence=confidence, detector_notes=notes)


def recheck_step(manager="detection", reason="Category DoS does not fit brute force."):
    return step(REQUEST_RECHECK, manager=manager, reason=reason)


ESCALATE = step("escalate", reasoning="Still inconsistent after the recheck.")


class RecordingRecheck:
    """Stands in for the pipeline's callback: returns scripted updated inputs."""

    def __init__(self, *updated_inputs):
        self.updated = list(updated_inputs)
        self.calls = []

    def __call__(self, manager, reason):
        self.calls.append((manager, reason))
        result = self.updated.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def mismatch_case(recheck=None, available=("detection", "mitigation")):
    """Detection says DoS, Mitigation matched brute force: the Week 8 mismatch case."""
    d = det("[category=DoS] [label=DoS Hulk] High packet rate.")
    return JudgeInput(d, make_mitigation(d, ["T1110"], 0.85, "block_source_ip"),
                      recheck=recheck, rechecks_available=list(available))


def consistent_case(available=("mitigation",)):
    d = det("[category=BruteForce] [label=SSH-Patator] Many short SSH connections.")
    return JudgeInput(d, make_mitigation(d, ["T1110"], 0.85, "block_source_ip"),
                      rechecks_available=list(available))


# --- parsing --------------------------------------------------------------------

def test_recheck_is_only_in_the_schema_when_available():
    assert REQUEST_RECHECK not in agent_step_schema(False)["properties"]["action"]["enum"]
    with_recheck = agent_step_schema(True)
    assert REQUEST_RECHECK in with_recheck["properties"]["action"]["enum"]
    assert with_recheck["properties"]["manager"]["enum"] == ["detection", "mitigation"]


def test_parse_recheck_step():
    parsed = parse_agent_step(json.dumps(recheck_step()), ["no_action"], ["detection"])
    assert (parsed.action, parsed.manager) == (REQUEST_RECHECK, "detection")
    assert parsed.recheck_reason == "Category DoS does not fit brute force."


@pytest.mark.parametrize("managers, reply, reason", [
    ((), recheck_step(), "recheck_unavailable"),                        # no recheck offered
    (("mitigation",), recheck_step("detection"), "recheck_unavailable"),  # already used
    (("detection",), recheck_step(reason="  "), "missing_reason"),
])
def test_invalid_recheck_requests_are_rejected(managers, reply, reason):
    with pytest.raises(LLMOutputError) as err:
        parse_agent_step(json.dumps(reply), ["no_action"], managers)
    assert err.value.reason == reason


# --- the agent's recheck ------------------------------------------------------------

def test_agent_rechecks_then_decides_on_the_updated_conclusions():
    # After the Detection recheck, Detection's category fits brute force, so the
    # mismatch is resolved and the confident, consistent case can be acted on.
    callback = RecordingRecheck(consistent_case(available=["mitigation"]))
    llm = ScriptedLLM(COMPARE, CHECK, recheck_step(), CHECK, FINALIZE_BLOCK)

    result = judge(MODE_AGENT, llm).run(mismatch_case(callback))

    assert callback.calls == [("detection", "Category DoS does not fit brute force.")]
    assert actions(result)[:5] == ["compare_conclusions", "check_category_consistency", "request_recheck",
                                   "check_category_consistency", "llm_finalize"]
    assert result.recommended_action == "block_source_ip" and not result.escalated_to_human
    assert "SSH-Patator" in result.detection.detector_notes  # the result carries the updated conclusions
    assert result.llm_usage["rechecks"] == 1
    (record,) = result.rechecks
    assert record.manager == "detection" and record.error is None
    assert record.before and record.after


def test_recheck_observation_includes_the_updated_conclusions():
    llm = ScriptedLLM(COMPARE, recheck_step(), ESCALATE)
    judge(MODE_AGENT, llm).run(mismatch_case(RecordingRecheck(consistent_case())))
    observation = llm.calls[2]["messages"][-1]["content"]
    assert observation.startswith("Observation (request_recheck)")
    assert '"updated_conclusions"' in observation and "SSH-Patator" in observation


def test_recheck_prompt_and_schema_are_offered_only_when_available():
    llm = ScriptedLLM(COMPARE, ESCALATE)
    judge(MODE_AGENT, llm).run(mismatch_case(RecordingRecheck()))
    assert "request_recheck" in llm.calls[0]["messages"][0]["content"]

    llm = ScriptedLLM(COMPARE, ESCALATE)
    judge(MODE_AGENT, llm).run(mismatch_case(recheck=None))
    assert "request_recheck" not in llm.calls[0]["messages"][0]["content"]


def test_after_a_recheck_consistency_must_be_checked_again_before_acting():
    # The old consistency check was on the old evidence, so finalizing straight
    # after the recheck is rejected; the agent checks again, then finalizes.
    llm = ScriptedLLM(COMPARE, CHECK, recheck_step(), FINALIZE_BLOCK, CHECK, FINALIZE_BLOCK)
    result = judge(MODE_AGENT, llm).run(mismatch_case(RecordingRecheck(consistent_case())))
    assert "invalid_llm_reply" in actions(result)
    assert result.recommended_action == "block_source_ip"


def test_each_manager_can_be_rechecked_only_once():
    # The callback (wrongly) still offers detection; the Judge removes it anyway.
    still_offered = consistent_case(available=["detection", "mitigation"])
    callback = RecordingRecheck(still_offered)
    llm = ScriptedLLM(COMPARE, recheck_step(), recheck_step(), ESCALATE)

    result = judge(MODE_AGENT, llm).run(mismatch_case(callback))

    assert len(callback.calls) == 1
    assert "invalid_llm_reply" in actions(result)
    assert result.escalated_to_human


def test_failed_recheck_is_reported_and_earlier_conclusions_stand():
    callback = RecordingRecheck(RuntimeError("model file missing"))
    llm = ScriptedLLM(COMPARE, recheck_step(), ESCALATE)

    result = judge(MODE_AGENT, llm).run(mismatch_case(callback))

    observation = llm.calls[2]["messages"][-1]["content"]
    assert '"recheck_done": false' in observation and "model file missing" in observation
    assert result.rechecks[0].error == "RuntimeError: model file missing"
    assert "DoS Hulk" in result.detection.detector_notes  # unchanged
    assert result.escalated_to_human


def test_guardrails_judge_the_updated_case():
    # Mitigation's recheck comes back unsure: the rules now say escalate, so the
    # guardrail overrides an LLM that still wants to act.
    d = det("[category=BruteForce] [label=SSH-Patator] ...")
    unsure = JudgeInput(d, make_mitigation(d, ["T1110"], 0.4, "block_source_ip"), rechecks_available=[])
    llm = ScriptedLLM(COMPARE, recheck_step("mitigation", "Check the technique again."), CHECK, FINALIZE_BLOCK)

    result = judge(MODE_AGENT, llm).run(mismatch_case(RecordingRecheck(unsure)))

    assert result.case == rules.LOW_MITIGATION_CONFIDENCE
    assert result.escalated_to_human and result.decided_by == "guardrail_override"


@pytest.mark.parametrize("mode", [MODE_RULES, MODE_SINGLE_SHOT])
def test_baselines_never_recheck(mode):
    callback = RecordingRecheck()
    llm = ScriptedLLM({"decision": "escalate", "reasoning": "Inconsistent."})
    result = judge(mode, llm).run(mismatch_case(callback))
    assert callback.calls == [] and result.rechecks == []


# --- pipeline ---------------------------------------------------------------------

def detection_manager(first, recheck_result=None):
    manager = MagicMock()
    manager.run.side_effect = lambda event, recheck_reason=None: recheck_result if recheck_reason else first
    return manager


def test_pipeline_detection_recheck_reruns_detection_then_mitigation():
    first = det("[category=DoS] [label=DoS Hulk] ...")
    second = det("[category=DoS] [label=DoS Hulk] ... Recheck evidence: Top labels: DoS Hulk 0.91.")
    detection = detection_manager(first, second)
    mitigation = FakeMitigationManager("match")
    llm = ScriptedLLM(COMPARE, recheck_step(), ESCALATE)

    run = Pipeline(detection, judge(MODE_AGENT, llm), mitigation).run(make_event())

    assert detection.run.call_args_list[1].kwargs == {"recheck_reason": "Category DoS does not fit brute force."}
    assert mitigation.calls == [first, second]          # mitigation refreshed on the new detection
    assert mitigation.recheck_reasons == [None, None]   # ... as a normal run, not a recheck
    assert run.detection is second and run.response.detection is second
    assert [r.manager for r in run.rechecks] == ["detection"]
    assert "recheck_detection" in run.timings_ms and not run.errors


def test_pipeline_mitigation_recheck_passes_the_reason():
    mitigation = FakeMitigationManager("match")
    llm = ScriptedLLM(COMPARE, recheck_step("mitigation", "Try another technique."), ESCALATE)

    run = Pipeline(detection_manager(det("[category=DoS] ...")), judge(MODE_AGENT, llm), mitigation).run(make_event())

    assert mitigation.recheck_reasons == [None, "Try another technique."]
    assert [r.manager for r in run.rechecks] == ["mitigation"]


def test_pipeline_offers_only_configured_managers():
    llm = ScriptedLLM(COMPARE, ESCALATE)
    Pipeline(detection_manager(det("[category=DoS] ...")), judge(MODE_AGENT, llm), None).run(make_event())
    observation = llm.calls[1]["messages"][-1]["content"]
    assert '"rechecks_available": ["detection"]' in observation


def test_pipeline_records_a_failed_recheck():
    detection = MagicMock()
    detection.run.side_effect = [det("[category=DoS] ..."), RuntimeError("model file missing")]
    llm = ScriptedLLM(COMPARE, recheck_step(), ESCALATE)

    run = Pipeline(detection, judge(MODE_AGENT, llm), FakeMitigationManager("match")).run(make_event())

    assert run.succeeded
    assert run.errors["recheck_detection"] == "RuntimeError: model file missing"
    assert run.rechecks[0].error == "RuntimeError: model file missing"


def test_pipeline_callback_refuses_a_second_recheck():
    pipeline = Pipeline(detection_manager(det("[category=DoS] ...")), judge(MODE_RULES, None), FakeMitigationManager())
    run = pipeline.run(make_event())
    available = ["detection"]
    pipeline._recheck(run, available, "detection", "first")
    with pytest.raises(ValueError):
        pipeline._recheck(run, available, "detection", "second")


# --- Mitigation Manager plumbing ------------------------------------------------------

def test_mitigation_manager_records_the_recheck_reason():
    manager = MitigationManager(CorrelationSubagent(encoder=None))
    d = det("[category=BruteForce] [label=SSH-Patator] ...")
    normal = manager.run(d)
    rechecked = manager.run(d, recheck_reason="Check the technique again.")

    assert rechecked.trace[0].action == "recheck"
    assert rechecked.trace[0].tool_input == {"reason": "Check the technique again."}
    assert normal.trace[0].action == "delegate_to_subagent"
    # Plumbing only for now: the result itself is unchanged.
    assert rechecked.correlation.matched_technique_ids == normal.correlation.matched_technique_ids
    assert rechecked.proposed_action == normal.proposed_action
