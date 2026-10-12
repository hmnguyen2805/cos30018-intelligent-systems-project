"""
Demo scenarios for the UI: ready-made cases that run without a trained
detection model, so the system can be shown on any machine.

Only Detection is scripted (ScriptedDetectionManager returns what the real
Detection Manager would report for that kind of traffic, in the same format:
"[category=X] [label=Y] <traffic summary>"). The Mitigation Manager and the
Judge are the real ones, so what the UI shows is the real coordination,
recheck path, guardrails and failure handling.

Two scenarios simulate a crashed agent, to show how failures are reported.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, MitigationRecommendation, TrafficEvent


@dataclass(frozen=True)
class Scenario:
    title: str
    description: str
    features: Dict[str, float]
    anomalous: bool
    confidence: float
    category: Optional[str] = None
    label: Optional[str] = None
    category_confidence: Optional[float] = None
    summary: Optional[str] = None
    recheck_evidence: str = "Recheck evidence: no extra evidence."
    detection_fails: bool = False
    mitigation_fails: bool = False


SCENARIOS: Dict[str, Scenario] = {s.title: s for s in (
    Scenario(
        "SSH brute force (SSH-Patator)",
        "Many short login attempts against SSH. Expected: both managers agree, automated block.",
        {"Destination Port": 22.0, "Flow Duration": 1_200_000.0, "Total Fwd Packets": 3.0},
        anomalous=True, confidence=0.97, category="BruteForce", label="SSH-Patator", category_confidence=0.95,
        summary="Single flow to destination port 22 (SSH): duration 1.20 s (above the training median), "
                "forward packets 3, SYN flags 2, RST flags 1.",
        recheck_evidence="Recheck evidence: Top labels: SSH-Patator 0.95, FTP-Patator 0.03, Benign 0.01. "
                         "Groups: BruteForce 0.97, Benign 0.01. Tree votes: fraction=0.97, std=0.17.",
    ),
    Scenario(
        "DoS Hulk on a web server",
        "A flood of HTTP requests to one server. Expected: automated rate-limit/block.",
        {"Destination Port": 80.0, "Flow Duration": 85_000_000.0, "Total Fwd Packets": 7.0},
        anomalous=True, confidence=0.99, category="DoS", label="DoS Hulk", category_confidence=0.98,
        summary="Single flow to destination port 80 (HTTP): duration 85.00 s (above the training median), "
                "forward packets 7, backward packets 0.",
        recheck_evidence="Recheck evidence: Top labels: DoS Hulk 0.96, DoS GoldenEye 0.02, DDoS 0.01. "
                         "Groups: DoS 0.98, DDoS 0.01.",
    ),
    Scenario(
        "Port scan",
        "One source probing many ports. Expected: automated block of the scanning source.",
        {"Destination Port": 3389.0, "Flow Duration": 40.0, "Total Fwd Packets": 1.0},
        anomalous=True, confidence=0.98, category="PortScan", label="PortScan", category_confidence=0.99,
        summary="Single flow to destination port 3389 (RDP): duration 0.00 s (below the training median), "
                "forward packets 1, SYN flags 1, RST flags 1.",
    ),
    Scenario(
        "Heartbleed against a TLS service",
        "The new Heartbleed category, mapped to T1190 in the playbook. Expected: automated isolate/patch.",
        {"Destination Port": 444.0, "Flow Duration": 119_000_000.0, "Total Fwd Packets": 2_600.0},
        anomalous=True, confidence=0.99, category="Heartbleed", label="Heartbleed", category_confidence=0.99,
        summary="Single flow to destination port 444: duration 119.00 s (above the training median), "
                "forward packets 2600 (above the training median).",
    ),
    Scenario(
        "Normal web browsing (benign)",
        "Ordinary traffic. Expected: both managers agree there is no threat, no action.",
        {"Destination Port": 443.0, "Flow Duration": 300_000.0, "Total Fwd Packets": 12.0},
        anomalous=False, confidence=0.98,
    ),
    Scenario(
        "Borderline detection",
        "Detection is unsure (confidence 0.55). Expected: escalate to a human.",
        {"Destination Port": 8080.0, "Flow Duration": 2_000_000.0, "Total Fwd Packets": 5.0},
        anomalous=True, confidence=0.55, category="Unknown", label="Unknown",
        summary="Single flow to destination port 8080 (HTTP alt): duration 2.00 s (near the training median).",
        recheck_evidence="Recheck evidence: Top labels: Web Attack - Brute Force 0.31, Benign 0.30, DoS Hulk 0.12. "
                         "Tree votes: fraction=0.55, std=0.50.",
    ),
    Scenario(
        "Unknown attack type",
        "Detection is sure it's an attack but can't name the category. Mitigation falls back to "
        "searching the technique catalog. Expected: escalate unless a confident technique is found.",
        {"Destination Port": 8443.0, "Flow Duration": 5_000_000.0, "Total Fwd Packets": 40.0},
        anomalous=True, confidence=0.93, category="Unknown", label="Unknown", category_confidence=0.52,
        summary="Single flow to destination port 8443: duration 5.00 s (above the training median), "
                "forward packets 40 (above the training median).",
        recheck_evidence="Recheck evidence: Top labels: Web Attack - XSS 0.41, Web Attack - Brute Force 0.33, "
                         "DoS Hulk 0.12. Groups: WebAttack 0.74, DoS 0.12.",
    ),
    Scenario(
        "Mitigation Manager crashes",
        "Simulated failure: the Mitigation Manager raises an error. Expected: the run still finishes "
        "and the Judge escalates, because there is no mitigation analysis to rely on.",
        {"Destination Port": 22.0, "Flow Duration": 1_100_000.0, "Total Fwd Packets": 3.0},
        anomalous=True, confidence=0.96, category="BruteForce", label="SSH-Patator", category_confidence=0.94,
        summary="Single flow to destination port 22 (SSH): duration 1.10 s (above the training median).",
        mitigation_fails=True,
    ),
    Scenario(
        "Detection Manager crashes",
        "Simulated failure: the Detection Manager raises an error. Expected: an unsuccessful run, "
        "with the error shown and no final decision.",
        {"Destination Port": 22.0},
        anomalous=False, confidence=0.0, detection_fails=True,
    ),
)}

DEFAULT_SCENARIO = next(iter(SCENARIOS))


def detector_notes(scenario: Scenario, recheck: bool = False) -> Optional[str]:
    """Notes in the real Detection format: tags first, then the summary."""
    if not scenario.anomalous:
        return scenario.recheck_evidence if recheck else None
    parts = [f"[category={scenario.category}] [label={scenario.label}]", scenario.summary]
    if recheck:
        parts.append(scenario.recheck_evidence)
    return " ".join(p for p in parts if p)


class ScriptedDetectionManager(BaseAgent):
    """Stands in for the trained Detection Manager in the demo scenarios.
    Same contract: run(event, recheck_reason=None) -> DetectionResult. Like the
    real one, a recheck adds evidence to the notes but never changes the verdict."""
    name = "detection_manager (demo)"

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario

    def run(self, input_data: TrafficEvent, recheck_reason: Optional[str] = None) -> DetectionResult:
        s = self.scenario
        self._trace = []
        if recheck_reason is not None:
            self.log_step(thought="Judge requested a recheck: gathering extra evidence; the decision itself "
                                  "is computed exactly as in a normal run.",
                          action="recheck", tool_input={"reason": recheck_reason})
        if s.detection_fails:
            raise RuntimeError("Detection model file not found (simulated failure)")
        self.log_step(thought="Run baseline RandomForest classifier on event features.", action="call_classifier",
                      tool_input={"n_features": len(input_data.features)},
                      observation=f"p_anomalous={s.confidence if s.anomalous else 1 - s.confidence:.3f}")
        if s.anomalous:
            self.log_step(thought="Choose the attack label and category.", action="category_decision",
                          observation=f"label={s.label}, category={s.category}")
        self.log_step(thought="Finalize decision.", action="finalize",
                      observation=f"is_anomalous={s.anomalous}, confidence={s.confidence:.3f}")
        return DetectionResult(
            event=input_data, is_anomalous=s.anomalous, confidence=s.confidence,
            detector_notes=detector_notes(s, recheck=recheck_reason is not None), trace=self.get_trace(),
            attack_category=s.category if s.anomalous else None,
            attack_label=s.label if s.anomalous else None,
            category_confidence=s.category_confidence if s.anomalous else None,
            traffic_summary=s.summary if s.anomalous else None,
        )


class FailingMitigationManager(BaseAgent):
    """A Mitigation Manager that always crashes (for the failure scenario)."""
    name = "mitigation_manager (demo failure)"

    def run(self, input_data: DetectionResult, recheck_reason: Optional[str] = None) -> MitigationRecommendation:
        raise RuntimeError("Technique catalog unavailable (simulated failure)")
