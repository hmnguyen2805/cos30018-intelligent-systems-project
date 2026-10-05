"""
Tests for reading Detection's fine labels and categories (src/shared/tags.py,
the new DetectionResult fields, and how the Judge's tools use them), plus the
Heartbleed playbook entry.
"""
from src.correlation.subagent import CorrelationSubagent
from src.response import tools
from src.response.tools import load_playbook
from src.shared.schemas import DetectionResult, JudgeInput, TrafficEvent
from src.shared.tags import parse_category, parse_label, strip_tags
from tests.fakes import make_mitigation

NOTES = "[category=WebAttack] [label=Web Attack - XSS] Single flow to destination port 80 (HTTP)."


def det(notes=NOTES, **fields):
    return DetectionResult(event=TrafficEvent(features={}), is_anomalous=True, confidence=0.9,
                           detector_notes=notes, **fields)


# --- tags -----------------------------------------------------------------------

def test_parse_label_handles_spaces_and_hyphens():
    assert parse_label(NOTES) == "Web Attack - XSS"
    assert parse_label("[category=DoS] [label=DoS Hulk] ...") == "DoS Hulk"


def test_parse_label_without_a_label_tag():
    assert parse_label("[category=DoS] no label here") is None
    assert parse_label(None) is None
    assert parse_label("") is None


def test_parse_category_still_works_next_to_a_label():
    assert parse_category(NOTES) == "WebAttack"


def test_strip_tags_removes_category_and_label():
    assert strip_tags(NOTES) == "Single flow to destination port 80 (HTTP)."
    assert strip_tags("[category=Unknown] [label=Unknown]") == ""
    assert strip_tags(None) == ""


# --- new DetectionResult fields -----------------------------------------------------

def test_new_detection_fields_default_to_none():
    d = det()
    assert (d.attack_category, d.attack_label, d.category_confidence, d.traffic_summary) == (None, None, None, None)


def test_judge_prefers_the_structured_fields_over_the_notes():
    d = det(notes="[category=DoS] [label=DoS Hulk]", attack_category="DDoS", attack_label="DDoS")
    assert tools.detection_category(d) == "DDoS"
    assert tools.detection_label(d) == "DDoS"


def test_judge_falls_back_to_the_notes_tags():
    d = det()
    assert tools.detection_category(d) == "WebAttack"
    assert tools.detection_label(d) == "Web Attack - XSS"


def test_compare_conclusions_shows_category_and_label():
    d = det()
    facts = tools.compare_conclusions(tools.JudgeContext(JudgeInput(d, make_mitigation(d, ["T1190"]))))
    assert facts["detection_category"] == "WebAttack"
    assert facts["detection_label"] == "Web Attack - XSS"
    assert "rechecks_available" not in facts  # only shown to the agent-mode Judge


# --- Heartbleed ---------------------------------------------------------------------

def test_heartbleed_maps_to_exploit_public_facing_application():
    assert load_playbook()["category_techniques"]["Heartbleed"] == ["T1190"]


def test_heartbleed_is_consistent_with_t1190_for_the_judge():
    d = det(notes="[category=Heartbleed] [label=Heartbleed] Single flow to destination port 444.")
    ctx = tools.JudgeContext(JudgeInput(d, make_mitigation(d, ["T1190"])))
    assert tools.check_category_consistency(ctx)["consistent"] is True


def test_heartbleed_category_is_looked_up_by_mitigation():
    d = det(notes="[category=Heartbleed] [label=Heartbleed] Single flow.")
    result = CorrelationSubagent(encoder=None).run(d)
    assert result.matched_technique_ids == ["T1190"]
    assert result.trace[0].action == "lookup_category_techniques"
