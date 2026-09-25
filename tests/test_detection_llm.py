"""
Unit tests for src.detection.llm_notes — the guardrail logic that decides
whether an LLM's proposed category/explanation is trusted (schema + category
+ grounding checks) or discarded for a deterministic template note.
"""
import json

from src.detection import llm_notes

KNOWN_FEATURES = ["Flow Duration", "Packet Count", "Fwd Packet Length Max"]
GROUNDED = ["Flow Duration", "Packet Count"]


def valid_payload(**overrides):
    payload = {
        "category": "PortScan",
        "explanation": "High Packet Count with short Flow Duration suggests scanning.",
        "tools_used": ["top_features"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_payload_grounded_in_returned_features_is_accepted():
    result = llm_notes.parse_and_validate(valid_payload(), KNOWN_FEATURES, GROUNDED)
    assert result == {
        "category": "PortScan",
        "explanation": "High Packet Count with short Flow Duration suggests scanning.",
        "tools_used": ["top_features"],
    }


def test_invalid_json_is_rejected():
    assert llm_notes.parse_and_validate("not json at all", KNOWN_FEATURES, GROUNDED) is None


def test_non_object_json_is_rejected():
    assert llm_notes.parse_and_validate("[1, 2, 3]", KNOWN_FEATURES, GROUNDED) is None


def test_category_outside_fixed_list_is_rejected():
    raw = valid_payload(category="Ransomware")
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is None


def test_missing_explanation_is_rejected():
    raw = json.dumps({"category": "PortScan", "tools_used": []})
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is None


def test_over_long_explanation_is_rejected():
    raw = valid_payload(explanation="x" * (llm_notes.MAX_EXPLANATION_CHARS + 1))
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is None


def test_non_string_tools_used_is_rejected():
    raw = valid_payload(tools_used=[1, 2])
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is None


def test_explanation_naming_ungrounded_feature_is_rejected():
    raw = valid_payload(explanation="Fwd Packet Length Max was unusually high here.")
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is None


def test_explanation_naming_only_grounded_features_is_accepted():
    raw = valid_payload(explanation="Packet Count was unusually high here.")
    assert llm_notes.parse_and_validate(raw, KNOWN_FEATURES, GROUNDED) is not None


def test_build_template_note_with_vote_std_mentions_disagreement_and_unknown():
    note = llm_notes.build_template_note(vote_std=0.3)
    assert "category=Unknown" in note
    assert "0.300" in note


def test_build_template_note_without_vote_std_still_tags_unknown():
    note = llm_notes.build_template_note(vote_std=None)
    assert "category=Unknown" in note
    assert "tree_vote_std" not in note
