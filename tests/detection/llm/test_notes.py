"""
Unit tests for src.detection.llm.notes — the guardrail logic that decides
whether an LLM's proposed explanation is trusted (schema + grounding checks)
or discarded for a deterministic template note.

The LLM never decides the category — that moved to a deterministic tool
(classifier.predict_attack_category, applied by subagent.py._choose_category)
— so its schema is just {explanation, tools_used}, and parse_and_validate has
nothing category-related left to check.

parse_and_validate returns (result, reason): reason is None on success, or
one of the fixed REASON_* strings on failure — subagent.py logs it as a
distinct trace action and evaluate.py reports it as detector_notes'
fallback_reason.
"""
import json

from src.detection.llm import notes as llm_notes

KNOWN_FEATURES = ["Flow Duration", "Packet Count", "Fwd Packet Length Max"]
GROUNDED = ["Flow Duration", "Packet Count"]


def valid_payload(**overrides):
    payload = {
        "explanation": "High Packet Count with short Flow Duration suggests scanning.",
        "tools_used": ["top_features"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_payload_grounded_in_returned_features_is_accepted():
    result, reason = llm_notes.parse_and_validate(valid_payload(), KNOWN_FEATURES, GROUNDED)
    assert reason is None
    assert result == {
        "explanation": "High Packet Count with short Flow Duration suggests scanning.",
        "tools_used": ["top_features"],
    }


def test_invalid_json_is_rejected_with_reason():
    result, reason = llm_notes.parse_and_validate("not json at all", KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_INVALID_JSON


def test_non_object_json_is_rejected_with_reason():
    result, reason = llm_notes.parse_and_validate("[1, 2, 3]", KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_INVALID_JSON


def test_missing_explanation_is_rejected_with_reason():
    raw = json.dumps({"tools_used": []})
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_INVALID_JSON


def test_over_long_explanation_is_rejected_with_reason():
    raw = valid_payload(explanation="x" * (llm_notes.MAX_EXPLANATION_CHARS + 1))
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_INVALID_JSON


def test_non_string_tools_used_is_rejected_with_reason():
    raw = valid_payload(tools_used=[1, 2])
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_INVALID_JSON


def test_explanation_naming_ungrounded_feature_is_rejected_with_reason():
    raw = valid_payload(explanation="Fwd Packet Length Max was unusually high here.")
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert result is None
    assert reason == llm_notes.REASON_UNGROUNDED


def test_explanation_naming_only_grounded_features_is_accepted():
    raw = valid_payload(explanation="Packet Count was unusually high here.")
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert result is not None
    assert reason is None


def test_a_category_field_in_the_payload_is_simply_ignored():
    # The LLM has no schema slot for a category, but if it puts one there anyway (e.g. an
    # older prompt cached somewhere, or a model that doesn't strictly follow schemas), it must
    # not affect validation either way — there's no "bad_category" check left to trip.
    raw = json.dumps({
        "category": "PortScan", "explanation": "Packet Count was high.", "tools_used": [],
    })
    result, reason = llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED)
    assert reason is None
    assert "category" not in result


def test_build_template_note_with_vote_std_does_not_mention_category():
    note = llm_notes.build_template_note(vote_std=0.3)
    assert "category" not in note
    assert "0.300" in note


def test_build_template_note_without_vote_std_does_not_mention_category():
    note = llm_notes.build_template_note(vote_std=None)
    assert "category" not in note
    assert "tree_vote_std" not in note


# --- SYSTEM_PROMPT: must match how ToolCallingAgent actually terminates ------
# ToolCallingAgent requires the run to end via a final_answer *tool call* — a
# prompt that only says "output ONLY a JSON object" gets a small model to just
# type JSON as its message text, which has no tool call in it and fails to
# parse ("model output does not contain any JSON blob").

def test_system_prompt_tells_the_model_to_use_final_answer():
    assert "final_answer" in llm_notes.SYSTEM_PROMPT


def test_system_prompt_tells_the_model_the_category_is_already_decided():
    prompt_lower = llm_notes.SYSTEM_PROMPT.lower()
    assert "already decided" in prompt_lower or "already determined" in prompt_lower


def test_system_prompt_still_forbids_numeric_output():
    prompt_lower = llm_notes.SYSTEM_PROMPT.lower()
    assert "confidence" in prompt_lower or "probability" in prompt_lower
    assert "category" in prompt_lower  # forbids restating it, even though it can't anyway


# --- SINGLE_SHOT_SYSTEM_PROMPT ------------------------------------------------

def test_single_shot_prompt_tells_the_model_the_category_is_already_decided():
    prompt_lower = llm_notes.SINGLE_SHOT_SYSTEM_PROMPT.lower()
    assert "already decided" in prompt_lower or "already determined" in prompt_lower


def test_single_shot_prompt_still_forbids_numeric_output():
    prompt_lower = llm_notes.SINGLE_SHOT_SYSTEM_PROMPT.lower()
    assert "confidence" in prompt_lower or "probability" in prompt_lower


# --- ANSWER_JSON_SCHEMA: no category field ------------------------------------

def test_answer_json_schema_has_no_category_property():
    assert "category" not in llm_notes.ANSWER_JSON_SCHEMA["properties"]
    assert set(llm_notes.ANSWER_JSON_SCHEMA["required"]) == {"explanation", "tools_used"}


# --- extract_json_object: the salvage path's JSON extraction ----------------

def test_extract_json_object_finds_object_surrounded_by_prose():
    text = 'Sure, here is my answer: ' + valid_payload() + ' Let me know if you need more.'
    extracted = llm_notes.extract_json_object(text)
    assert extracted is not None
    assert json.loads(extracted) == json.loads(valid_payload())


def test_extract_json_object_returns_none_with_no_braces_at_all():
    assert llm_notes.extract_json_object("I am not sure how to answer this.") is None


def test_extract_json_object_returns_none_when_extracted_span_is_not_valid_json():
    # Braces present, but not a single well-formed JSON object once sliced out.
    assert llm_notes.extract_json_object("{not json} and then {also not json}") is None


def test_extract_json_object_returns_none_for_non_string_input():
    assert llm_notes.extract_json_object(None) is None


def test_extract_json_object_then_parse_and_validate_recovers_a_valid_answer():
    text = f"Based on my investigation, here's the result: {valid_payload()}"
    extracted = llm_notes.extract_json_object(text)
    result, reason = llm_notes.parse_and_validate(extracted, KNOWN_FEATURES, GROUNDED)
    assert reason is None
    assert result["tools_used"] == ["top_features"]
