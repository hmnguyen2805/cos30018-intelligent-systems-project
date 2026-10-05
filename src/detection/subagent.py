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

# Below this, the category model's top label/group isn't trusted and code reports "Unknown" instead.
# Default chosen by the threshold sweep — see docs/evaluation.md.
DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD = 0.8
CATEGORY_TOP_K = 3
RECHECK_TOP_FEATURES = 10
MAX_RECHECK_REASON_CHARS = 300


def decide_category(top_label, top_label_p, top_category, top_category_p, threshold):
    """The three-way category rule — pure, no LLM. Returns (label, category,
    disagreement). Benign top (fine or coarse) -> Unknown/Unknown + disagreement;
    top fine label >= threshold -> that label and its group; else top coarse
    group >= threshold -> label "Unknown", that group; else Unknown/Unknown."""
    if classifier.BENIGN_CATEGORY in (top_label, top_category):
        return "Unknown", "Unknown", True
    if top_label_p >= threshold:
        return top_label, classifier.coarse_category(top_label), False
    if top_category_p >= threshold:
        return "Unknown", top_category, False
    return "Unknown", "Unknown", False


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

    def run(self, input_data: TrafficEvent, recheck_reason: Optional[str] = None) -> DetectionResult:
        """Classify one event: is_anomalous/confidence come from
        classifier.py only; category, the traffic summary and (if enabled)
        the LLM explanation only ever affect detector_notes.

        recheck_reason (set by the Judge) asks for MORE EVIDENCE, never a
        different decision: is_anomalous/confidence are computed exactly as
        in a normal run. A recheck additionally inspects the tree votes even
        when not borderline, adds the top-3 labels/groups and a top-10
        feature summary to the notes, forces the LLM explanation (reason in
        its prompt) and logs a "recheck" trace step."""
        event = input_data
        recheck_reason = _clean_reason(recheck_reason) if recheck_reason is not None else None
        recheck = recheck_reason is not None
        self._trace = []  # fresh trace per event
        if recheck:
            self.log_step(
                thought="Judge requested a recheck — gathering extra evidence; the decision itself "
                        "is computed exactly as in a normal run.",
                action="recheck", tool_input={"reason": recheck_reason},
            )

        p_anomalous = classifier.predict_proba_anomalous(self._artifact, event.features)
        self.log_step(
            thought="Run baseline RandomForest classifier on event features.",
            action="call_classifier",
            tool_input={"n_features": len(event.features)},
            observation=f"p_anomalous={p_anomalous:.3f}",
        )

        final_p = p_anomalous
        notes = None
        vote_frac = vote_std = None
        borderline = BORDERLINE_LOW <= p_anomalous <= BORDERLINE_HIGH

        if borderline or recheck:
            vote_frac, vote_std = classifier.tree_vote_spread(self._artifact, event.features)
            self.log_step(
                thought=(f"Confidence borderline (p={p_anomalous:.3f}). Re-examine via per-tree "
                         "vote spread before deciding.") if borderline else
                        "Recheck: inspect per-tree vote spread as evidence only (not borderline, "
                        "so it does not affect the decision).",
                action="inspect_tree_votes",
                tool_input={"n_features": len(event.features)},
                observation=f"tree_vote_frac={vote_frac:.3f}, tree_vote_std={vote_std:.3f}",
            )
        if borderline:
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

        chosen_label = chosen_category = category_probability = ranked = summary = None
        if is_anomalous:
            chosen_label, chosen_category, category_probability, ranked = self._choose_category(event)
            summary = classifier.describe_flow(
                self._artifact, event.features, top_k=RECHECK_TOP_FEATURES if recheck else 0,
            )
        evidence = _recheck_evidence(ranked, vote_frac, vote_std) if recheck else None

        if is_anomalous:
            explanation = (
                self._llm_explanation(
                    event, p_anomalous, vote_std, borderline, chosen_category, category_probability,
                    recheck_reason,
                ) if self.use_llm else None
            )
            notes = _compose_notes(chosen_category, chosen_label, [summary, evidence, explanation, notes])
        elif evidence:
            notes = evidence

        self.log_step(
            thought="Finalize decision.",
            action="finalize",
            observation=f"is_anomalous={is_anomalous}, confidence={confidence:.3f}",
        )

        result = DetectionResult(
            event=event,
            is_anomalous=is_anomalous,
            confidence=confidence,
            detector_notes=notes,
            trace=self.get_trace(),
        )
        # shared/schemas.py may not have these fields (yet); the notes carry them either way.
        for field_name, value in (
            ("attack_category", chosen_category), ("attack_label", chosen_label),
            ("category_confidence", category_probability), ("traffic_summary", summary),
        ):
            if hasattr(result, field_name):
                setattr(result, field_name, value)
        return result

    def _choose_category(self, event: TrafficEvent) -> Tuple[str, str, Optional[float], Optional[dict]]:
        """Deterministically choose the fine attack label and its coarse
        category (never the LLM) via decide_category. Returns (label,
        category, top_category_probability, ranked) — the probability is
        returned even when the result is "Unknown", so callers can compare
        against the raw top-1 group; ranked is the model's top-3 labels +
        group probabilities (None without a category model). See docs/architecture.md (category decision)."""
        if self._category_artifact is None:
            self.log_step(
                thought="No category model loaded — reporting label and category as Unknown.",
                action="category_model_unavailable",
            )
            return "Unknown", "Unknown", None, None

        ranked = classifier.predict_attack_category(self._category_artifact, event.features, top_k=CATEGORY_TOP_K)
        top_label, top_label_p = ranked["labels"][0]["label"], ranked["labels"][0]["probability"]
        top_category, top_category_p = ranked["categories"][0]["category"], ranked["categories"][0]["probability"]
        threshold = self._category_confidence_threshold

        label, category, disagreement = decide_category(
            top_label, top_label_p, top_category, top_category_p, threshold,
        )
        tool_input = {
            "chosen_label": label, "chosen_category": category,
            "raw_top_label": top_label, "raw_top_label_probability": top_label_p,
            "raw_top_category": top_category, "raw_top_probability": top_category_p,
            "threshold": threshold,
        }
        if disagreement:
            self.log_step(
                thought=f"Category model's top vote is {classifier.BENIGN_CATEGORY} "
                        f"(label={top_label} p={top_label_p:.3f}, category={top_category} "
                        f"p={top_category_p:.3f}), but the binary model already called this event "
                        "anomalous — binary/category model disagreement. Reporting Unknown rather "
                        "than trusting either side's specific label.",
                action="model_disagreement", tool_input=tool_input, observation=f"top={ranked}",
            )
        else:
            self.log_step(
                thought=f"Top fine label {top_label} (p={top_label_p:.3f}), top group {top_category} "
                        f"(p={top_category_p:.3f}), threshold {threshold:.2f} -> label={label}, "
                        f"category={category}.",
                action="category_decision", tool_input=tool_input, observation=f"top={ranked}",
            )
        return label, category, top_category_p, ranked

    def _llm_explanation(
        self, event: TrafficEvent, p_anomalous: float, vote_std: Optional[float], borderline: bool,
        chosen_category: str, category_probability: Optional[float], recheck_reason: Optional[str],
    ) -> str:
        """The LLM's explanation of the already-decided category, or the
        template note when the layer is skipped/invalid/failed. Never raises
        (explain() falls back)."""
        llm_result = self._llm.explain(
            event, p_anomalous, vote_std, chosen_category, category_probability, recheck_reason=recheck_reason,
        )
        if llm_result is not None:
            return llm_result["explanation"]
        return build_template_note(vote_std if borderline else None)


def _clean_reason(reason: str) -> str:
    """One short bracket-free line, safe to embed in notes and prompts."""
    return " ".join(str(reason).replace("[", "(").replace("]", ")").split())[:MAX_RECHECK_REASON_CHARS]


def _compose_notes(category: str, label: str, parts: list) -> str:
    """"[category=X] [label=Y] <summary> <evidence> <explanation> ..." — the
    tags always come first (src/shared/tags.py parses the category from them)."""
    return " ".join([f"[category={category}] [label={label}]", *(p for p in parts if p)])


def _recheck_evidence(ranked: Optional[dict], vote_frac: Optional[float], vote_std: Optional[float]) -> str:
    """Notes text for a recheck: the extra evidence gathered. The Judge's
    reason is deliberately NOT included — Correlation embeds the notes text,
    and the reason would bias its technique search. It lives only in the
    "recheck" trace step and the LLM prompt."""
    pieces = ["Recheck evidence:"]
    if ranked:
        pieces.append("Top labels: " + ", ".join(f"{e['label']} {e['probability']:.2f}" for e in ranked["labels"]) + ".")
        pieces.append("Groups: " + ", ".join(
            f"{e['category']} {e['probability']:.2f}" for e in ranked["categories"][:CATEGORY_TOP_K]) + ".")
    if vote_frac is not None:
        pieces.append(f"Tree votes: fraction={vote_frac:.2f}, std={vote_std:.2f}.")
    return " ".join(pieces)
