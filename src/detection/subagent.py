"""
Detection Subagent — decides whether a TrafficEvent looks anomalous.

The baseline RandomForest (classifier.py) is a tool the agent calls, not the
agent itself. On a borderline confidence score, the agent re-examines via
per-tree vote spread (a second, finer-grained tool call) before finalizing —
that borderline-handling loop is what makes this an agent rather than a
single classifier call.

Owned by the Detection Manager (manager.py), which delegates each event here.
"""
from typing import Optional

from src.detection import classifier
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent

BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6
DISAGREEMENT_THRESHOLD = 0.15  # tree-vote std above this = low ensemble consensus


class DetectionSubagent(BaseAgent):
    name = "detection_subagent"

    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self._artifact = classifier.load_artifact(model_path or classifier.DEFAULT_MODEL_PATH)

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        event = input_data
        self._trace = []  # fresh trace per event

        p_anomalous = classifier.predict_proba_anomalous(self._artifact, event.features)
        self.log_step(
            thought="Run baseline RandomForest classifier on event features.",
            action="call_classifier",
            tool_input={"n_features": len(event.features)},
            observation=f"p_anomalous={p_anomalous:.3f}",
        )

        final_p = p_anomalous
        notes = None

        if BORDERLINE_LOW <= p_anomalous <= BORDERLINE_HIGH:
            vote_frac, vote_std = classifier.tree_vote_spread(self._artifact, event.features)
            self.log_step(
                thought=f"Confidence borderline (p={p_anomalous:.3f}). Re-examine via per-tree "
                        "vote spread before deciding.",
                action="inspect_tree_votes",
                tool_input={"n_features": len(event.features)},
                observation=f"tree_vote_frac={vote_frac:.3f}, tree_vote_std={vote_std:.3f}",
            )
            final_p = vote_frac
            if vote_std >= DISAGREEMENT_THRESHOLD:
                notes = (
                    f"Borderline call, high tree disagreement (std={vote_std:.3f}) — "
                    "flagged low-confidence for downstream correlation/response."
                )
            else:
                notes = f"Borderline call, trees agree (std={vote_std:.3f}) — trusting vote fraction."

        is_anomalous = final_p >= 0.5
        confidence = final_p if is_anomalous else 1.0 - final_p

        self.log_step(
            thought="Finalize decision.",
            action="finalize",
            observation=f"is_anomalous={is_anomalous}, confidence={confidence:.3f}",
        )

        return DetectionResult(
            event=event,
            is_anomalous=is_anomalous,
            confidence=confidence,
            detector_notes=notes,
            trace=self.get_trace(),
        )
