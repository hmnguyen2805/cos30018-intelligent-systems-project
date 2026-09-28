"""
Judge decision rules: the deterministic core of the Judge Agent.

Two pure functions, no LLM involved:

    compare_conclusions(JudgeInput) -> Comparison
        Boils both managers' conclusions down to plain facts. In Week 8 this
        becomes the Judge's `compare_conclusions` tool.

    decide(Comparison) -> Decision
        Maps those facts to a final call using the case table below. In Week 8
        this becomes the fallback the Judge uses when the LLM loop fails.

Case table (checked top to bottom, first match wins):

    LOW_DETECTION_CONFIDENCE   detection unsure                     -> escalate
    MITIGATION_MISSING         mitigation failed / unavailable      -> escalate if anomalous,
                                                                       else no action (flagged incomplete)
    BOTH_BENIGN                benign + no technique matched         -> no action
    BENIGN_BUT_ACTION          benign, but a technique matched      -> escalate (conflict)
    ANOMALOUS_NO_MATCH         anomalous, no technique matched      -> escalate (possible unknown attack)
    LOW_MITIGATION_CONFIDENCE  anomalous + matched, mitigation unsure -> escalate
    AGREED_THREAT              anomalous + matched + confident      -> automated response

Note on DETECTION_MIN_CONFIDENCE: the Detection Subagent reports confidence
as max(p, 1 - p), so its borderline band (p between 0.4 and 0.6) comes out as
confidence 0.5 to 0.6. A 0.6 threshold therefore escalates every borderline
detection call.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from src.shared.schemas import JudgeInput

DETECTION_MIN_CONFIDENCE = 0.6
MITIGATION_MIN_CONFIDENCE = 0.6

NO_ACTION = "no_action"
ESCALATE = "escalate_to_human"

# Case labels (also stored on ResponseRecommendation.case for evaluation).
LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
MITIGATION_MISSING = "MITIGATION_MISSING"
BOTH_BENIGN = "BOTH_BENIGN"
BENIGN_BUT_ACTION = "BENIGN_BUT_ACTION"
ANOMALOUS_NO_MATCH = "ANOMALOUS_NO_MATCH"
LOW_MITIGATION_CONFIDENCE = "LOW_MITIGATION_CONFIDENCE"
AGREED_THREAT = "AGREED_THREAT"


@dataclass(frozen=True)
class Comparison:
    """Plain facts about the two conclusions, independent of any decision."""
    detection_anomalous: bool
    detection_confidence: float
    mitigation_available: bool
    mitigation_error: Optional[str] = None
    technique_ids: Tuple[str, ...] = ()
    mitigation_confidence: Optional[float] = None
    proposed_action: Optional[str] = None

    @property
    def technique_matched(self) -> bool:
        return len(self.technique_ids) > 0

    @property
    def agents_agree(self) -> bool:
        """Both managers point the same way: threat + technique, or benign + none.
        Never true when mitigation is missing (there is nothing to agree with)."""
        return self.mitigation_available and self.detection_anomalous == self.technique_matched

    def summary(self) -> str:
        """One-line description for the trace / UI."""
        det = "anomalous" if self.detection_anomalous else "benign"
        parts = [f"detection={det} (conf={self.detection_confidence:.2f})"]
        if not self.mitigation_available:
            parts.append(f"mitigation=unavailable ({self.mitigation_error or 'no reason given'})")
        else:
            techniques = ",".join(self.technique_ids) or "none"
            parts.append(
                f"mitigation techniques={techniques} (conf={self.mitigation_confidence:.2f}), "
                f"proposed_action={self.proposed_action!r}"
            )
        parts.append(f"agree={self.agents_agree}")
        return "; ".join(parts)


@dataclass(frozen=True)
class Decision:
    case: str
    action: str
    escalate: bool
    agents_agree: bool
    reasoning: str


def compare_conclusions(judge_input: JudgeInput) -> Comparison:
    detection = judge_input.detection
    mitigation = judge_input.mitigation

    if mitigation is None:
        return Comparison(
            detection_anomalous=detection.is_anomalous,
            detection_confidence=detection.confidence,
            mitigation_available=False,
            mitigation_error=judge_input.mitigation_error,
        )

    return Comparison(
        detection_anomalous=detection.is_anomalous,
        detection_confidence=detection.confidence,
        mitigation_available=True,
        technique_ids=tuple(mitigation.correlation.matched_technique_ids),
        mitigation_confidence=mitigation.confidence,
        proposed_action=mitigation.proposed_action,
    )


def decide(c: Comparison) -> Decision:
    agree = c.agents_agree

    def escalate(case: str, reasoning: str) -> Decision:
        return Decision(case, ESCALATE, True, agree, reasoning)

    if c.detection_confidence < DETECTION_MIN_CONFIDENCE:
        return escalate(
            LOW_DETECTION_CONFIDENCE,
            f"Detection confidence {c.detection_confidence:.2f} is below "
            f"{DETECTION_MIN_CONFIDENCE}; a human should confirm before acting.",
        )

    if not c.mitigation_available:
        if c.detection_anomalous:
            return escalate(
                MITIGATION_MISSING,
                "Detection flagged a threat but no mitigation analysis is available "
                f"({c.mitigation_error or 'no reason given'}); cannot choose a response automatically.",
            )
        return Decision(
            MITIGATION_MISSING, NO_ACTION, False, agree,
            "Detection is confidently benign. Mitigation analysis was unavailable, "
            "so this result is based on detection only (incomplete).",
        )

    if not c.detection_anomalous and not c.technique_matched:
        return Decision(
            BOTH_BENIGN, NO_ACTION, False, agree,
            "Both managers agree there is no threat.",
        )

    if not c.detection_anomalous and c.technique_matched:
        return escalate(
            BENIGN_BUT_ACTION,
            f"Conflict: detection says benign but mitigation matched {', '.join(c.technique_ids)}.",
        )

    if not c.technique_matched:
        return escalate(
            ANOMALOUS_NO_MATCH,
            "Detection flagged a threat but no known ATT&CK technique matched; "
            "possibly an attack outside the catalogue.",
        )

    if c.mitigation_confidence < MITIGATION_MIN_CONFIDENCE:
        return escalate(
            LOW_MITIGATION_CONFIDENCE,
            f"Mitigation confidence {c.mitigation_confidence:.2f} is below "
            f"{MITIGATION_MIN_CONFIDENCE}; the proposed action is not trusted automatically.",
        )

    return Decision(
        AGREED_THREAT, c.proposed_action, False, agree,
        f"Both managers agree on a threat ({', '.join(c.technique_ids)}) with sufficient "
        f"confidence; applying the proposed action.",
    )
