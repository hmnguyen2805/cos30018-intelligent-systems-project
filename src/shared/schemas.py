"""
Shared data contracts for the manager-subagent pipeline:

    Detection Manager (+ Detection Subagent)
    Mitigation Manager (+ Correlation Subagent)
    -> Judge

Only this file and base.py are fixed across the team. Everything else about
how an agent works internally is up to its owner.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TraceStep:
    """One step in an agent's reasoning-action loop (for logging/UI observability)."""
    step_number: int
    thought: Optional[str] = None       # what the agent decided / reasoned
    action: Optional[str] = None        # e.g. "call_classifier", "query_technique_db"
    tool_input: Optional[dict] = None
    observation: Optional[str] = None   # what came back from the tool call


@dataclass
class TrafficEvent:
    """A single row / window of network traffic (CICIDS2017, UNSW-NB15, or NSL-KDD)."""
    features: Dict[str, float]
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None


@dataclass
class DetectionResult:
    """Output of the Detection Manager (produced via its Detection Subagent)."""
    event: TrafficEvent
    is_anomalous: bool
    confidence: float                      # 0.0 - 1.0
    detector_notes: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)
    # Structured copies of what Detection also tags in detector_notes
    # ("[category=DoS] [label=DoS Hulk] <summary>"). Detection fills them in for
    # anomalous events; None for benign traffic or when no category model is loaded.
    attack_category: Optional[str] = None      # coarse group, e.g. "DoS", "WebAttack", "Unknown"
    attack_label: Optional[str] = None         # fine CICIDS2017 label, e.g. "DoS Hulk", "Web Attack - XSS"
    category_confidence: Optional[float] = None  # category model's top-group probability
    traffic_summary: Optional[str] = None      # deterministic one-flow description (no LLM)


@dataclass
class CorrelationResult:
    """Output of the Correlation Subagent (used internally by the Mitigation Manager)."""
    detection: DetectionResult
    matched_technique_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    correlation_notes: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)


@dataclass
class MitigationRecommendation:
    """Output of the Mitigation Manager — its conclusion, prior to arbitration by the Judge."""
    correlation: CorrelationResult
    proposed_action: str
    confidence: float = 0.0
    trace: List[TraceStep] = field(default_factory=list)


@dataclass
class RecheckRecord:
    """One time the Judge sent a case back to a manager to look again."""
    manager: str                    # "detection" or "mitigation"
    reason: str                     # the Judge's reason, passed as recheck_reason
    before: str = ""                # one-line summary of the conclusions before the recheck
    after: str = ""                 # ... and after it
    error: Optional[str] = None     # set if the manager failed during the recheck
    duration_ms: float = 0.0


# Signature of the recheck callback the pipeline hands the Judge:
#     recheck(manager, reason) -> JudgeInput with the managers' updated conclusions
RecheckFn = Callable[[str, str], "JudgeInput"]


@dataclass
class JudgeInput:
    """What the Judge receives: both managers' conclusions.

    `mitigation` is None when the Mitigation Manager failed or isn't available;
    `mitigation_error` then says why, so the Judge can handle incomplete input
    explicitly instead of crashing.

    `recheck` is set by the pipeline. It lets the agent-mode Judge send the
    case back to a manager once ("look again, because ..."). None means
    rechecks aren't available (e.g. the Judge is called directly, as in the
    tests and the demo script). `rechecks_available` lists the managers that
    can still be asked; each manager can be rechecked at most once per case.
    """
    detection: DetectionResult
    mitigation: Optional[MitigationRecommendation] = None
    mitigation_error: Optional[str] = None
    recheck: Optional[RecheckFn] = field(default=None, repr=False, compare=False)
    rechecks_available: List[str] = field(default_factory=list)


@dataclass
class ResponseRecommendation:
    """Output of the Judge Agent: the pipeline's final result."""
    detection: DetectionResult
    mitigation: Optional[MitigationRecommendation]
    recommended_action: str
    agents_agree: bool
    escalated_to_human: bool = False
    case: Optional[str] = None           # which decision case applied (for evaluation)
    reasoning: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)
    decided_by: Optional[str] = None     # "rules", "llm", "guardrail_override" or "rules_fallback"
    llm_usage: Dict[str, Any] = field(default_factory=dict)  # LLM calls, tokens, latency, iterations (evaluation)
    rechecks: List[RecheckRecord] = field(default_factory=list)  # cases sent back to a manager


@dataclass
class PipelineRun:
    """Everything that happened for one event: what the UI shows and what the
    report's latency/cost numbers come from.

    `errors` and `timings_ms` are keyed by stage name: "detection",
    "mitigation", "judge", plus "recheck_detection" / "recheck_mitigation"
    when the Judge sent the case back. `detection` and `mitigation` hold the
    latest conclusions (after any recheck); `rechecks` records each one.
    """
    event: TrafficEvent
    detection: Optional[DetectionResult] = None
    mitigation: Optional[MitigationRecommendation] = None
    response: Optional[ResponseRecommendation] = None
    errors: Dict[str, str] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    rechecks: List[RecheckRecord] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.response is not None