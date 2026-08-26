# Mitigation Manager

**Owner:** Callum Fennessy

Same restructure as Detection: you're now a **Mitigation Manager** with the CTI
Correlation work as your **subagent** underneath it. This README goes into more
implementation detail than usual, since it's a bigger conceptual jump than Detection's —
follow it step by step rather than trying to design it from scratch.

## The big picture

```
DetectionResult (from Detection Manager)
        |
        v
  MITIGATION MANAGER (you)
        |  delegates to
        v
  CORRELATION SUBAGENT (also you)
        |  matches the detection to a known attack technique
        v
  CorrelationResult
        |
        v
  (Manager reasons: "given this technique, what should we do?")
        |
        v
  MitigationRecommendation  →  sent to the Judge (Minh)
```

You're building **two** pieces: the subagent that does the technique-matching, and the
manager that sits above it and decides on an action. Build the subagent first — the
manager is a thin wrapper around it.

## Step 1 — the technique catalog (plain data, no AI needed)

Start with a plain Python dict mapping each seed technique ID to a short description.
This is what your subagent searches against:

```python
# technique_catalog.py
TECHNIQUE_CATALOG = {
    "T1110": "Brute Force: adversary attempts to gain access via repeated login attempts.",
    "T1110.001": "Password Guessing: automated attempts to guess account passwords.",
    "T1566": "Phishing: adversary sends malicious messages to gain initial access.",
    "T1190": "Exploit Public-Facing Application: exploits a weakness in an internet-facing system.",
    "T1041": "Exfiltration Over C2 Channel: data stolen via the existing command-and-control channel.",
    "T1567": "Exfiltration Over Web Service: data stolen using a legitimate external web service.",
}
```

## Step 2 — the Correlation Subagent (this is the "Agentic RAG" part)

"RAG" sounds intimidating but for 6 techniques it's simple: turn text into numeric
vectors (embeddings), and find which technique's description is numerically closest to
your query. `sentence-transformers` does the embedding for you — you don't need to
build or train anything.

```bash
pip install sentence-transformers numpy
```

```python
# subagent.py
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
```

That retry-with-a-broader-query step is what makes this an agent instead of a plain
lookup function — it's deciding to take a second action based on the first result,
which is exactly what the grading criteria asks for.

## Step 3 — the Mitigation Manager

This wraps the subagent and turns "which technique matched" into "what should we do
about it":

```python
# manager.py
from src.correlation.subagent import CorrelationSubagent
from src.shared.base import BaseAgent
from src.shared.schemas import DetectionResult, MitigationRecommendation

ACTION_BY_TECHNIQUE = {
    "T1110": "Lock account after repeated failed logins; alert analyst.",
    "T1110.001": "Lock account after repeated failed logins; alert analyst.",
    "T1566": "Quarantine the message/attachment; warn the user.",
    "T1190": "Patch/isolate the vulnerable service; block source IP.",
    "T1041": "Block outbound C2 traffic; isolate host.",
    "T1567": "Block the destination web service; isolate host.",
}


class MitigationManager(BaseAgent):
    name = "mitigation_manager"

    def __init__(self):
        super().__init__()
        self._subagent = CorrelationSubagent()

    def run(self, input_data: DetectionResult) -> MitigationRecommendation:
        self._trace = []
        self.log_step(thought="Delegate to Correlation Subagent.", action="delegate_to_subagent")

        correlation = self._subagent.run(input_data)

        if correlation.matched_technique_ids:
            technique = correlation.matched_technique_ids[0]
            action = ACTION_BY_TECHNIQUE.get(technique, "Escalate — unrecognised technique.")
        else:
            technique = None
            action = "No confident technique match — escalate for manual review."

        self.log_step(
            thought=f"Reason over correlation result (technique={technique}).",
            action="decide_action",
            observation=action,
        )

        return MitigationRecommendation(
            correlation=correlation,
            proposed_action=action,
            confidence=correlation.confidence,
            trace=self.get_trace(),
        )
```

## Contract summary

```
MitigationManager.run(detection: DetectionResult) -> MitigationRecommendation
```

`MitigationRecommendation` is a new shared type — it's been added to
`src/shared/schemas.py` already, so you can import it directly.

Your Manager's conclusion goes to the Judge (Minh), who compares it against the
Detection Manager's conclusion to produce the final result.

## Suggested order to actually build this in

1. `technique_catalog.py` — five minutes, just typing out the dict above.
2. `subagent.py` — get `_query_catalog()` working first, print results for a few test
   strings before wiring it into `run()`.
3. `manager.py` — once the subagent runs, this part is short.
4. Test locally: `CorrelationSubagent().run(...)` with a fake `DetectionResult` you
   construct by hand, before it's wired into the real pipeline.

## Dependencies

`sentence-transformers`, `numpy` — add these to a `requirements-correlation.txt` in
this folder.
