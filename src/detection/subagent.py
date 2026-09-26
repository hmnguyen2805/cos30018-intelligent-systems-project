"""
Detection Subagent — decides whether a TrafficEvent looks anomalous.

classifier.py's baseline RandomForest is a tool the agent calls, not the
agent itself: on a borderline score it re-examines via per-tree vote spread
before finalizing. It also deterministically chooses an attack category —
never the LLM, same principle as is_anomalous/confidence. See
docs/architecture.md (category decision) and docs/design-decisions.md for
the algorithm and why it isn't an LLM output.

Optionally (use_llm=True), a bounded LLM loop (llm.layer.LLMExplanationLayer)
writes a grounded explanation of the already-chosen category; its output
schema has no category field, so it cannot override the decision. See
docs/llm-layer.md for the guardrails.

Owned by DetectionManager, which delegates each event here.
"""
from typing import Optional, Tuple

from src.detection import classifier
from src.detection.llm.layer import DEFAULT_CIRCUIT_BREAKER_THRESHOLD, LLMExplanationLayer
from src.detection.llm.notes import build_template_note
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, TrafficEvent

BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6
DISAGREEMENT_THRESHOLD = 0.15  # tree-vote std above this = low ensemble consensus

# Below this, the category model's top class isn't trusted and code reports "Unknown" instead.
# Default chosen by the threshold sweep — see docs/evaluation.md.
DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD = 0.9
CATEGORY_TOP_K = 3


class DetectionSubagent(BaseAgent):
    name = "detection_subagent"

    def __init__(
        self,
        model_path: Optional[str] = None,
        category_model_path: Optional[str] = None,
        use_llm: bool = False,
        llm_timeout_seconds: Optional[float] = None,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        category_confidence_threshold: float = DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self._artifact = classifier.load_artifact(model_path or classifier.DEFAULT_BINARY_MODEL_PATH)
        self._category_artifact = self._load_category_artifact(category_model_path)
        if self._category_artifact is not None:
            classifier.assert_feature_names_match(self._artifact, self._category_artifact)
        self._category_confidence_threshold = category_confidence_threshold

        self.use_llm = use_llm
        self._llm = LLMExplanationLayer(
            self._artifact, log_step=self.log_step,
            llm_timeout_seconds=llm_timeout_seconds, circuit_breaker_threshold=circuit_breaker_threshold,
        )

    @staticmethod
    def _load_category_artifact(category_model_path: Optional[str]) -> Optional[dict]:
        """Load the category model, or None if absent — kept optional for
        backward compatibility with an install that never trained one."""
        try:
            artifact = classifier.load_artifact(category_model_path or classifier.DEFAULT_CATEGORY_MODEL_PATH)
        except FileNotFoundError:
            return None
        if "category_model" not in artifact:
            return None
        return artifact

    def __enter__(self) -> "DetectionSubagent":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        """Tear down the LLM layer's persistent MCP connection, if one is
        open. Safe to call multiple times, or when the LLM layer was never
        used."""
        self._llm.close()

    def warmup(self) -> dict:
        """Open the MCP connection and prime the LLM before a run. No-op
        success when use_llm is False. See LLMExplanationLayer.warmup."""
        if not self.use_llm:
            return {"ok": True, "elapsed_seconds": 0.0, "reason": None}
        return self._llm.warmup()

    def run(self, input_data: TrafficEvent) -> DetectionResult:
        """Classify one event: is_anomalous/confidence come from
        classifier.py only; category and (if enabled) the LLM explanation
        only ever affect detector_notes."""
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
        vote_std = None

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

        # Final: nothing below this line may change is_anomalous/confidence.
        is_anomalous = final_p >= 0.5
        confidence = final_p if is_anomalous else 1.0 - final_p

        chosen_category = None
        category_probability = None
        if is_anomalous:
            chosen_category, category_probability = self._choose_category(event)

        if is_anomalous and self.use_llm:
            notes = self._apply_llm_layer(event, p_anomalous, vote_std, notes, chosen_category, category_probability)
        elif chosen_category is not None:
            notes = self._tag_with_category(chosen_category, notes)

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

    def _choose_category(self, event: TrafficEvent) -> Tuple[str, Optional[float]]:
        """Deterministically choose the attack category (never the LLM).
        Returns (category, top_probability) — top_probability is returned
        even when the result is "Unknown", so callers can compare against
        the raw top-1 class. See docs/architecture.md (category decision)."""
        if self._category_artifact is None:
            self.log_step(
                thought="No category model loaded — reporting category as Unknown.",
                action="category_model_unavailable",
            )
            return "Unknown", None

        ranked = classifier.predict_attack_category(self._category_artifact, event.features, top_k=CATEGORY_TOP_K)
        top_category, top_probability = ranked[0]["category"], ranked[0]["probability"]

        if top_category == classifier.BENIGN_CATEGORY:
            chosen = "Unknown"
            self.log_step(
                thought=f"Category model's top class is {classifier.BENIGN_CATEGORY} "
                        f"(probability={top_probability:.3f}), but the binary model already called "
                        "this event anomalous — binary/category model disagreement. Reporting "
                        "Unknown rather than trusting either side's specific label.",
                action="model_disagreement",
                tool_input={
                    "chosen_category": chosen, "raw_top_category": top_category,
                    "raw_top_probability": top_probability, "threshold": self._category_confidence_threshold,
                },
                observation=f"top3={ranked}",
            )
            return chosen, top_probability

        chosen = top_category if top_probability >= self._category_confidence_threshold else "Unknown"

        self.log_step(
            thought=f"Category model top class: {top_category} (probability={top_probability:.3f}). "
                    f"{'Above' if chosen == top_category else 'Below'} the "
                    f"{self._category_confidence_threshold:.2f} confidence threshold.",
            action="category_decision",
            tool_input={
                "chosen_category": chosen, "raw_top_category": top_category,
                "raw_top_probability": top_probability, "threshold": self._category_confidence_threshold,
            },
            observation=f"top3={ranked}",
        )
        return chosen, top_probability

    @staticmethod
    def _tag_with_category(category: str, base_notes: Optional[str]) -> str:
        tag = f"[category={category}]"
        return f"{tag} {base_notes}" if base_notes else tag

    def _apply_llm_layer(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float],
        deterministic_notes: Optional[str], chosen_category: str, category_probability: Optional[float],
    ) -> str:
        """Return detector_notes after attempting the LLM layer:
        "[category=X] explanation", X always the pre-decided category.
        Never raises — explain() falls back to a template note on failure."""
        llm_result = self._llm.explain(event, p_anomalous, vote_std, chosen_category, category_probability)
        explanation = llm_result["explanation"] if llm_result is not None else build_template_note(vote_std)

        tag = self._tag_with_category(chosen_category, explanation)
        if deterministic_notes:
            return f"{tag} {deterministic_notes}"
        return tag
