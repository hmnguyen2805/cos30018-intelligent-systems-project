import numpy as np
from sentence_transformers import SentenceTransformer

from src.correlation.technique_catalog import TECHNIQUE_CATALOG
from src.shared.base import BaseAgent
from src.shared.schemas import CorrelationResult, DetectionResult

CONFIDENCE_THRESHOLD = 0.55  # below this, the match is treated as low-confidence


def _cosine(a, b):
    """Cosine similarity: 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class CorrelationSubagent(BaseAgent):
    name = "correlation_subagent"

    def __init__(self):
        super().__init__()
        # Small model, runs on CPU, no API key or cost — downloads once, then cached locally.
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._technique_ids = list(TECHNIQUE_CATALOG.keys())
        self._technique_embeddings = self._model.encode(list(TECHNIQUE_CATALOG.values()))

    def _query_catalog(self, query_text: str):
        """Embed query_text and return the best-matching technique + its score."""
        query_embedding = self._model.encode(query_text)
        scores = [_cosine(query_embedding, te) for te in self._technique_embeddings]
        best_idx = int(np.argmax(scores))
        return self._technique_ids[best_idx], scores[best_idx]

    def run(self, input_data: DetectionResult) -> CorrelationResult:
        self._trace = []
        # detector_notes from the Detection Agent is your best description of what
        # was seen — that's what you search the catalog with.
        query_text = input_data.detector_notes or "anomalous network traffic event"

        best_id, best_score = self._query_catalog(query_text)
        self.log_step(
            thought=f"Query technique catalog with: '{query_text}'",
            action="query_technique_catalog",
            observation=f"best_match={best_id}, score={best_score:.3f}",
        )

        matched = [best_id]
        if best_score < CONFIDENCE_THRESHOLD:
            # Low confidence — this is the "agentic" part: don't just accept a weak
            # match, try again with a broader query before giving up.
            fallback_text = "network intrusion attack technique"
            fallback_id, fallback_score = self._query_catalog(fallback_text)
            self.log_step(
                thought=f"Low confidence ({best_score:.3f}). Retrying with a broader query.",
                action="refine_query",
                observation=f"fallback_match={fallback_id}, score={fallback_score:.3f}",
            )
            if fallback_score > best_score:
                matched, best_score = [fallback_id], fallback_score
            else:
                matched = []  # honestly report "no confident match" rather than force one

        return CorrelationResult(
            detection=input_data,
            matched_technique_ids=matched,
            confidence=best_score,
            correlation_notes=f"Compared against {len(self._technique_ids)} seed techniques.",
            trace=self.get_trace(),
        )