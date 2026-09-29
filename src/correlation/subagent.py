"""
Correlation Subagent: maps a DetectionResult to MITRE ATT&CK techniques.

It uses the best evidence available, in this order:

  1. Benign traffic   -> nothing to correlate, so no match. The Judge then sees
                         both managers agreeing there is no threat.
  2. [category=X] tag -> look X up in the playbook's category_techniques.
                         Detection only names a category when its category model
                         is at least 0.9 sure; below that it says "Unknown".
  3. Unknown / no tag -> embedding search of the catalog with Detection's notes,
                         one retry with a query rephrased from the event itself,
                         then "no confident match" if still below the threshold.

The embedding model (sentence-transformers) is only loaded when step 3 needs it,
so category lookups, the tests and CI run without it installed.
"""
from typing import List, Optional

import numpy as np

from src.correlation.technique_catalog import CATEGORY_TECHNIQUES, TECHNIQUE_CATALOG
from src.response.rules import MITIGATION_MIN_CONFIDENCE
from src.shared.base import BaseAgent
from src.shared.schemas import CorrelationResult, DetectionResult
from src.shared.tags import parse_category, strip_category_tag

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding matches below this are rejected. Same number the Judge uses to trust
# a Mitigation result, so a match this subagent accepts is one the Judge can act on.
CONFIDENCE_THRESHOLD = MITIGATION_MIN_CONFIDENCE

# Confidence reported for a category lookup. Detection only names a category when
# its category model is at least this sure (DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD
# in src/detection/subagent.py), and the category -> technique mapping is fixed.
CATEGORY_MATCH_CONFIDENCE = 0.9

# Lets the retry describe the traffic in words the catalog uses.
SERVICE_BY_PORT = {
    21: "FTP login service", 22: "SSH login service", 23: "Telnet login service",
    25: "SMTP mail service", 53: "DNS service", 80: "HTTP web service",
    139: "SMB file sharing service", 443: "HTTPS web service", 445: "SMB file sharing service",
    3306: "MySQL database service", 3389: "RDP remote login service", 8080: "HTTP web service",
}


def _cosine(a, b):
    """Cosine similarity: 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _load_default_encoder():
    """Small model, runs on CPU, no API key or cost. Downloads once, then cached locally."""
    from sentence_transformers import SentenceTransformer  # optional dependency
    return SentenceTransformer(EMBEDDING_MODEL)


class CorrelationSubagent(BaseAgent):
    name = "correlation_subagent"

    def __init__(self, encoder=None):
        """`encoder` is anything with .encode(text or list of texts). Leave it as
        None to load the sentence-transformers model the first time it's needed;
        tests pass a fake one."""
        super().__init__()
        self._encoder = encoder
        self._encoder_error: Optional[str] = None
        self._technique_ids = list(TECHNIQUE_CATALOG.keys())
        self._technique_embeddings = None

    # --- the three evidence paths ---------------------------------------------

    def run(self, input_data: DetectionResult) -> CorrelationResult:
        self._trace = []

        if not input_data.is_anomalous:
            self.log_step(
                thought="Detection says the traffic is benign, so there is no attack to correlate.",
                action="skip_benign",
            )
            return self._result(input_data, [], 0.0, "Benign traffic: nothing to correlate.")

        category = parse_category(input_data.detector_notes)
        candidates = CATEGORY_TECHNIQUES.get(category, []) if category else []
        if candidates:
            self.log_step(
                thought=f"Detection tagged the attack category '{category}'. Look up the techniques that fit it.",
                action="lookup_category_techniques",
                tool_input={"category": category},
                observation=f"techniques={candidates}",
            )
            return self._result(
                input_data, list(candidates), CATEGORY_MATCH_CONFIDENCE,
                f"Matched from Detection's category '{category}' via the playbook.",
            )

        self.log_step(
            thought=f"No usable category (category={category}). Fall back to searching the catalog.",
            action="fallback_to_embedding_search",
        )
        return self._embedding_search(input_data)

    def _embedding_search(self, detection: DetectionResult) -> CorrelationResult:
        notes_text = strip_category_tag(detection.detector_notes)
        rephrased = self._rephrase(notes_text, detection)
        queries = [q for q in (notes_text, rephrased) if q]
        queries = list(dict.fromkeys(queries))  # drop the retry if it adds nothing new

        if not queries:
            self.log_step(
                thought="Nothing describes this event (no notes text, no destination port), "
                        "so a search would match the same technique every time.",
                action="no_query_text",
            )
            return self._result(detection, [], 0.0, "No description of the event to search with.")

        if self._get_technique_embeddings() is None:
            self.log_step(
                thought="The embedding model isn't available, so the catalog can't be searched.",
                action="embedding_model_unavailable",
                observation=self._encoder_error,
            )
            return self._result(
                detection, [], 0.0,
                "Embedding model unavailable (pip install -r src/correlation/requirements-correlation.txt).",
            )

        best_id, best_score = None, 0.0
        for attempt, query in enumerate(queries):
            technique_id, score = self._query_catalog(query)
            self.log_step(
                thought=(f"Query the technique catalog with: '{query}'" if attempt == 0 else
                         f"Low confidence ({best_score:.3f}). Retry with a query rephrased from the event."),
                action="query_technique_catalog" if attempt == 0 else "refine_query",
                tool_input={"query": query},
                observation=f"best_match={technique_id}, score={score:.3f}",
            )
            if score > best_score:
                best_id, best_score = technique_id, score
            if score >= CONFIDENCE_THRESHOLD:
                return self._result(
                    detection, [technique_id], score,
                    f"Embedding match {technique_id} (score {score:.3f}) "
                    f"out of {len(self._technique_ids)} techniques.",
                )

        # Honestly report "no confident match" rather than force a weak one.
        return self._result(
            detection, [], best_score,
            f"No confident match: best was {best_id} at {best_score:.3f}, "
            f"below the {CONFIDENCE_THRESHOLD} threshold.",
        )

    # --- helpers ---------------------------------------------------------------

    @staticmethod
    def _rephrase(notes_text: str, detection: DetectionResult) -> str:
        """A second query built from the event itself (never a fixed phrase),
        adding the destination port and the service it usually belongs to."""
        port = detection.event.features.get("Destination Port")
        if port is None:
            return notes_text
        port = int(port)
        service = SERVICE_BY_PORT.get(port, "network service")
        return f"{notes_text} Suspicious traffic to destination port {port} ({service}).".strip()

    def _get_technique_embeddings(self):
        if self._technique_embeddings is None:
            if self._encoder is None and self._encoder_error is None:
                try:
                    self._encoder = _load_default_encoder()
                except Exception as exc:  # ImportError, or the model download failed
                    self._encoder_error = f"{type(exc).__name__}: {exc}"
            if self._encoder is None:
                return None
            self._technique_embeddings = self._encoder.encode(list(TECHNIQUE_CATALOG.values()))
        return self._technique_embeddings

    def _query_catalog(self, query_text: str):
        """Embed query_text and return the best-matching technique + its score."""
        query_embedding = self._encoder.encode(query_text)
        scores = [_cosine(query_embedding, te) for te in self._technique_embeddings]
        best_idx = int(np.argmax(scores))
        return self._technique_ids[best_idx], scores[best_idx]

    def _result(self, detection: DetectionResult, matched: List[str], confidence: float,
                notes: str) -> CorrelationResult:
        return CorrelationResult(
            detection=detection,
            matched_technique_ids=matched,
            confidence=confidence,
            correlation_notes=notes,
            trace=self.get_trace(),
        )
