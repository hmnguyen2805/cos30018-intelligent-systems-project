"""
Test doubles and builders shared across tests.

FakeMitigationManager stands in for Callum's Mitigation Manager until the
real one exists, so the Judge and pipeline can be built and tested now.
Swap in the real class once it lands; the contract is the same:

    run(detection: DetectionResult) -> MitigationRecommendation
"""
from typing import Sequence

from src.shared.schemas import (
    CorrelationResult,
    DetectionResult,
    MitigationRecommendation,
    TrafficEvent,
)


def make_event() -> TrafficEvent:
    return TrafficEvent(features={"Flow Duration": 1.0})


def make_detection(anomalous: bool = True, confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(event=make_event(), is_anomalous=anomalous, confidence=confidence)


def make_mitigation(
    detection: DetectionResult = None,
    technique_ids: Sequence[str] = ("T1110",),
    confidence: float = 0.8,
    action: str = "block_source_ip",
) -> MitigationRecommendation:
    detection = detection or make_detection()
    correlation = CorrelationResult(
        detection=detection,
        matched_technique_ids=list(technique_ids),
        confidence=confidence,
    )
    return MitigationRecommendation(correlation=correlation, proposed_action=action, confidence=confidence)


class FakeMitigationManager:
    """Returns a canned MitigationRecommendation for a named scenario.

    Scenarios:
        "match"           technique matched, confident
        "no_match"        nothing matched
        "low_confidence"  technique matched, but unsure
        "fail"            raises, like a crashed agent or unreachable vector DB
    """
    name = "fake_mitigation_manager"

    def __init__(self, scenario: str = "match"):
        self.scenario = scenario
        self.calls = []

    def run(self, detection: DetectionResult) -> MitigationRecommendation:
        self.calls.append(detection)
        if self.scenario == "match":
            return make_mitigation(detection, ("T1110",), 0.85, "block_source_ip")
        if self.scenario == "no_match":
            return make_mitigation(detection, (), 0.3, "none")
        if self.scenario == "low_confidence":
            return make_mitigation(detection, ("T1190",), 0.4, "isolate_host")
        if self.scenario == "fail":
            raise RuntimeError("vector DB unreachable")
        raise ValueError(f"Unknown scenario: {self.scenario}")
