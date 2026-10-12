"""
Evaluation tasks: what one test case looks like, how it's loaded from JSON,
and the stand-in agents that let a task script a manager's output.

A task file is a JSON list of objects:

    {
      "id": "D01",
      "difficulty": "difficult",          simple | moderate | difficult | edge
      "split": "dev",                     dev (written while building) | unseen (held out)
      "description": "...",
      "detection": {...} or "fail",       scripted Detection output (see ScriptedDetection)
      "features": {...},                  OR a real CICIDS2017 row for the real Detection Manager
      "mitigation": "real" | "fail" | {"techniques": [...], "confidence": 0.85, "action": "..."},
      "expected": ["escalate"],           acceptable outcomes: act | escalate | no_action | failed
      "true_category": "DoS"              optional: when the system acts, its technique must fit this
    }

Outcomes are judged on what the system did, not on internal details:
    act        an automated response was applied
    escalate   the case went to a human analyst
    no_action  treated as benign
    failed     the run produced no decision (e.g. Detection crashed)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.shared.base import BaseAgent
from src.shared.schemas import (
    CorrelationResult,
    DetectionResult,
    MitigationRecommendation,
    TrafficEvent,
)

OUTCOMES = ("act", "escalate", "no_action", "failed")
DIFFICULTIES = ("simple", "moderate", "difficult", "edge")
SPLITS = ("dev", "unseen")

DEFAULT_FEATURES = {"Destination Port": 0.0, "Flow Duration": 0.0}


@dataclass(frozen=True)
class EvalTask:
    id: str
    difficulty: str
    split: str
    description: str
    expected: List[str]
    detection: Union[Dict[str, Any], str, None] = None   # scripted output, "fail", or None (use the real model)
    features: Dict[str, float] = field(default_factory=dict)
    mitigation: Union[Dict[str, Any], str] = "real"
    true_category: Optional[str] = None

    @property
    def uses_real_detection(self) -> bool:
        return self.detection is None


def task_from_dict(data: Dict[str, Any]) -> EvalTask:
    """Build and check one task. Raises ValueError with the task id on a bad field."""
    task_id = data.get("id", "?")

    def bad(msg: str) -> ValueError:
        return ValueError(f"Task {task_id}: {msg}")

    expected = data.get("expected")
    if isinstance(expected, str):
        expected = [expected]
    if not expected or any(e not in OUTCOMES for e in expected):
        raise bad(f"'expected' must list outcomes from {OUTCOMES}, got {data.get('expected')!r}")
    if data.get("difficulty") not in DIFFICULTIES:
        raise bad(f"'difficulty' must be one of {DIFFICULTIES}")
    if data.get("split", "dev") not in SPLITS:
        raise bad(f"'split' must be one of {SPLITS}")
    detection = data.get("detection")
    features = {str(k): float(v) for k, v in (data.get("features") or {}).items()}
    if detection is None and not features:
        raise bad("needs either 'detection' (scripted) or 'features' (real model)")
    mitigation = data.get("mitigation", "real")
    if not (mitigation in ("real", "fail") or isinstance(mitigation, dict)):
        raise bad("'mitigation' must be 'real', 'fail' or an object")
    return EvalTask(
        id=str(task_id), difficulty=data["difficulty"], split=data.get("split", "dev"),
        description=data.get("description", ""), expected=list(expected), detection=detection,
        features=features, mitigation=mitigation, true_category=data.get("true_category"),
    )


def load_tasks(*paths: Union[str, Path]) -> List[EvalTask]:
    """Load tasks from one or more JSON files. Task ids must be unique across files."""
    tasks: List[EvalTask] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tasks.extend(task_from_dict(d) for d in json.load(f))
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate task ids: {duplicates}")
    return tasks


def event_for(task: EvalTask) -> TrafficEvent:
    return TrafficEvent(features=dict(task.features or DEFAULT_FEATURES))


# --- stand-in agents ------------------------------------------------------------

def _notes(spec: Dict[str, Any], recheck: bool) -> Optional[str]:
    """Notes in Detection's real format: "[category=X] [label=Y] <summary> [recheck evidence]"."""
    parts = []
    if spec.get("anomalous") and spec.get("category"):
        parts.append(f"[category={spec['category']}] [label={spec.get('label') or 'Unknown'}]")
    if spec.get("anomalous") and spec.get("summary"):
        parts.append(spec["summary"])
    if recheck:
        parts.append(spec.get("recheck_evidence") or "Recheck evidence: no extra evidence.")
    return " ".join(parts) or None


class ScriptedDetection(BaseAgent):
    """Returns the Detection result a task specifies. Same contract as the real
    Detection Manager, and like it, a recheck adds evidence to the notes but
    never changes the verdict or confidence.

    Spec keys: anomalous, confidence, category, label, category_confidence,
    summary, recheck_evidence. "fail" instead of a spec makes it crash."""
    name = "detection_manager (scripted)"

    def __init__(self, spec: Union[Dict[str, Any], str]):
        super().__init__()
        self.spec = spec

    def run(self, input_data: TrafficEvent, recheck_reason: Optional[str] = None) -> DetectionResult:
        self._trace = []
        if recheck_reason is not None:
            self.log_step(thought="Judge requested a recheck: add evidence, same verdict.",
                          action="recheck", tool_input={"reason": recheck_reason})
        if self.spec == "fail":
            raise RuntimeError("Detection model unavailable (simulated failure)")
        s = self.spec
        anomalous = bool(s["anomalous"])
        self.log_step(action="finalize", observation=f"is_anomalous={anomalous}, confidence={s['confidence']}")
        return DetectionResult(
            event=input_data, is_anomalous=anomalous, confidence=float(s["confidence"]),
            detector_notes=_notes(s, recheck_reason is not None), trace=self.get_trace(),
            attack_category=s.get("category") if anomalous else None,
            attack_label=s.get("label") if anomalous else None,
            category_confidence=s.get("category_confidence") if anomalous else None,
            traffic_summary=s.get("summary") if anomalous else None,
        )


class ScriptedMitigation(BaseAgent):
    """Returns the Mitigation result a task specifies: {"techniques", "confidence",
    "action"}, or crashes for "fail". Used for cases the real Mitigation
    Manager can't produce on purpose (a confident wrong technique, low confidence)."""
    name = "mitigation_manager (scripted)"

    def __init__(self, spec: Union[Dict[str, Any], str]):
        super().__init__()
        self.spec = spec

    def run(self, input_data: DetectionResult, recheck_reason: Optional[str] = None) -> MitigationRecommendation:
        self._trace = []
        if self.spec == "fail":
            raise RuntimeError("Technique catalog unavailable (simulated failure)")
        techniques = list(self.spec.get("techniques", []))
        confidence = float(self.spec.get("confidence", 0.0))
        action = self.spec.get("action") or ("No confident technique match: escalate for manual review."
                                              if input_data.is_anomalous else
                                              "No action: detection says the traffic is benign.")
        self.log_step(action="scripted_result", observation=f"techniques={techniques}, confidence={confidence}")
        correlation = CorrelationResult(detection=input_data, matched_technique_ids=techniques, confidence=confidence)
        return MitigationRecommendation(correlation=correlation, proposed_action=action, confidence=confidence,
                                        trace=self.get_trace())