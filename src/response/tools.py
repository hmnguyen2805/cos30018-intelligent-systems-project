"""
The Judge's tools: read-only functions the Judge (LLM or code) calls to gather
evidence about a case before deciding.

    compare_conclusions()          both managers' conclusions as plain facts
    lookup_playbook(technique_id)  response guidance for an ATT&CK technique
    check_category_consistency()   does Detection's attack category match the
                                   technique Mitigation matched?

None of them executes generated code or changes anything outside the Judge:
they only read the JudgeInput and the local playbook.json. That is the
sandboxing boundary for the Judge's tool use (assignment criteria 5E).
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.response import rules
from src.shared.schemas import JudgeInput
from src.shared.tags import parse_category

PLAYBOOK_PATH = Path(__file__).with_name("playbook.json")


@lru_cache(maxsize=1)
def load_playbook() -> dict:
    with open(PLAYBOOK_PATH, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class JudgeContext:
    """Everything the tools need about one case. Built once per Judge run."""
    judge_input: JudgeInput
    comparison: rules.Comparison = field(init=False)

    def __post_init__(self):
        self.comparison = rules.compare_conclusions(self.judge_input)

    @property
    def allowed_final_actions(self) -> List[str]:
        """Actions a finalize decision may choose. The Judge never invents an
        action: it can take the Mitigation Manager's proposal or do nothing."""
        actions = []
        proposed = self.comparison.proposed_action
        if self.comparison.mitigation_available and self.comparison.technique_matched and proposed:
            actions.append(proposed)
        actions.append(rules.NO_ACTION)
        return actions


# --- tools -------------------------------------------------------------------

def compare_conclusions(ctx: JudgeContext) -> dict:
    c = ctx.comparison
    detection = ctx.judge_input.detection
    # Mitigation's confidence only matters when it matched a technique; when nothing
    # matched it has nothing to act on, so its flag is "not applicable" (None). The
    # second Ollama run escalated a benign case because of a low, irrelevant confidence.
    nothing_matched = c.mitigation_available and not c.technique_matched
    mitigation_confident = (None if c.mitigation_confidence is None or nothing_matched
                            else c.mitigation_confidence >= rules.MITIGATION_MIN_CONFIDENCE)
    facts = {
        # Spelled out as words and flags as well as raw numbers: small models
        # misread booleans and threshold comparisons (seen in the first Ollama run).
        "detection_verdict": "anomalous" if c.detection_anomalous else "benign",
        "detection_anomalous": c.detection_anomalous,
        "detection_confidence": round(c.detection_confidence, 3),
        "detection_confident": c.detection_confidence >= rules.DETECTION_MIN_CONFIDENCE,
        "confidence_threshold": rules.DETECTION_MIN_CONFIDENCE,
        "detection_notes": detection.detector_notes,
        "mitigation_available": c.mitigation_available,
        "mitigation_error": c.mitigation_error,
        "matched_techniques": list(c.technique_ids),
        # Hidden when nothing matched: a 3B model kept escalating benign traffic
        # because of this irrelevant number, even with a note saying to ignore it.
        "mitigation_confidence": (None if c.mitigation_confidence is None or nothing_matched
                                  else round(c.mitigation_confidence, 3)),
        "mitigation_confident": mitigation_confident,
        "proposed_action": None if nothing_matched else c.proposed_action,
        "managers_agree": c.agents_agree,
        "allowed_final_actions": ctx.allowed_final_actions,
    }
    if nothing_matched:
        facts["mitigation_note"] = "No technique matched, so Mitigation has nothing to act on; its confidence is not relevant."
    return facts


def lookup_playbook(ctx: JudgeContext, technique_id: Optional[str] = None) -> dict:
    note = None
    if not technique_id:
        matched = ctx.comparison.technique_ids
        if not matched:
            return {"error": "No technique was matched, so there is no playbook entry to look up."}
        technique_id = matched[0]
        note = f"technique_id not given; used the matched technique {technique_id}."
    entry = load_playbook()["techniques"].get(technique_id)
    if entry is None:
        known = sorted(load_playbook()["techniques"])
        return {"technique_id": technique_id, "found": False, "known_techniques": known}
    result = {"technique_id": technique_id, "found": True, **entry}
    if note:
        result["note"] = note
    return result


def check_category_consistency(ctx: JudgeContext) -> dict:
    category = parse_category(ctx.judge_input.detection.detector_notes)
    techniques = list(ctx.comparison.technique_ids)
    if not ctx.comparison.detection_anomalous:
        return {"category": category, "matched_techniques": techniques, "consistent": "not_applicable",
                "note": "Detection says the traffic is benign, so there is no attack category to check. "
                        "This is expected for benign traffic."}
    if category is None or category == "Unknown":
        return {"category": category, "matched_techniques": techniques, "consistent": "unknown",
                "note": "Detection gave no usable attack category, so consistency can't be checked."}
    if not techniques:
        return {"category": category, "matched_techniques": [], "consistent": "unknown",
                "note": "Mitigation matched no technique."}
    expected = load_playbook()["category_techniques"].get(category, [])
    consistent = any(t in expected for t in techniques)
    return {
        "category": category,
        "matched_techniques": techniques,
        "techniques_expected_for_category": expected,
        "consistent": consistent,
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., dict]
    params: Dict[str, str] = field(default_factory=dict)


TOOLS: Dict[str, Tool] = {
    t.name: t for t in (
        Tool("compare_conclusions",
             "See both managers' conclusions: detection result and notes, matched techniques, "
             "confidences, proposed action, and the allowed final actions.",
             compare_conclusions),
        Tool("lookup_playbook",
             "Get response guidance for a matched MITRE ATT&CK technique.",
             lookup_playbook, {"technique_id": "e.g. 'T1110'"}),
        Tool("check_category_consistency",
             "Check whether Detection's attack category is consistent with the technique "
             "Mitigation matched.",
             check_category_consistency),
    )
}


def run_tool(name: str, ctx: JudgeContext, **kwargs) -> dict:
    """Execute a Judge tool by name. Unknown names raise KeyError."""
    tool = TOOLS[name]
    accepted = {k: v for k, v in kwargs.items() if k in tool.params}
    return tool.fn(ctx, **accepted)
