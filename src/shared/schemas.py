"""
Shared data contracts for the manager-subagent pipeline:

    Detection Manager (+ Detection Subagent)
    Mitigation Manager (+ Correlation Subagent)
    -> Judge

Only this file and base.py are fixed across the team. Everything else about
how an agent works internally is up to its owner.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
class ResponseRecommendation:
    """Output of the Judge Agent — the pipeline's final result."""
    correlation: CorrelationResult
    recommended_action: str
    agents_agree: bool
    escalated_to_human: bool = False
    reasoning: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)