# Detection Manager

**Owner:** Vinh Nghiem

Per the tutor's manager-subagent recommendation, Detection is two layers: a
**Detection Manager** that owns the top-level contract, and a **Detection
Subagent** underneath it that does the actual classification work — a
baseline RandomForest, with an optional LLM layer that only ever adds an
*explanation*, never a number or a category.

```text
DetectionManager.run(event: TrafficEvent) -> DetectionResult
```

The Manager delegates to the Subagent and owns any manager-level oversight.
`DetectionResult` goes to the Mitigation Manager (Correlation needs to know
what was detected) and to the Judge, who compares it against Mitigation's
conclusion. Full contract details: [docs/architecture.md](docs/architecture.md).

## `detector_notes` format

For every event, `DetectionResult.detector_notes` is one of:

- **`None`** — the event was not anomalous (benign). No note is produced.
- **`"[category=X] <explanation>"`** — the event was anomalous.
  `X` is always present and always code-decided (never an LLM output):
  either a fixed category (`DoS`, `DDoS`, `PortScan`, `BruteForce`,
  `WebAttack`, `Botnet`, `Infiltration`) or `"Unknown"`. `<explanation>` is
  either an LLM-generated explanation (when `use_llm=True` and it succeeds)
  or a fixed template sentence.

**What `Unknown` means:** either the category classifier's top-confidence
class didn't clear `CATEGORY_CONFIDENCE_THRESHOLD`, or the classifier voted
`Benign` on an event the binary model already called anomalous (a
disagreement between the two models — never reported as `Benign`, since
that would contradict `is_anomalous=True`). Downstream agents should treat
`Unknown` as "flagged anomalous, category not resolved," not as a fifth
attack type.

**What `model_disagreement` means:** the specific case above — the category
model's top vote was `Benign` while the binary model called the event
anomalous. Logged as its own trace step and, in `evaluate.py`'s CSV output,
as a `model_disagreement` boolean column. See
[docs/architecture.md](docs/architecture.md#category-decision-a-deterministic-tool-not-the-llm)
for the full decision logic, and
[docs/design-decisions.md](docs/design-decisions.md) for why it exists.

## Quick start

```sh
# install
pip install -r src/detection/requirements-detection.txt

# train both models (binary + category) — writes to models/, gitignored
python -m src.detection.train

# run the tests
pytest tests/ -k detection

# evaluate: sampled RF-vs-RF+LLM comparison
python -m src.detection.evaluation.evaluate --sample-size 200 --sampling random

# evaluate: full-test-split batch metrics + threshold sweep, no LLM
python -m src.detection.evaluation.evaluate --offline

# run the MCP tool server standalone (manual testing)
python -m src.detection.llm.tool_server
```

The optional LLM layer needs a `.env` (see
[docs/llm-layer.md](docs/llm-layer.md) for setup, models, and
troubleshooting) — everything above works with `use_llm=False` (the
default) without one.

## More docs

- [docs/architecture.md](docs/architecture.md) — file map, the
  manager/subagent contract, the detection flow diagram, the MCP server's
  role, and the two LLM modes (`agent` vs `single_shot`).
- [docs/design-decisions.md](docs/design-decisions.md) — chronological
  decision log: problem, evidence, decision, trade-off.
- [docs/evaluation.md](docs/evaluation.md) — how to run both evaluation
  modes, current result tables, and known limitations.
- [docs/llm-layer.md](docs/llm-layer.md) — guardrails, env vars,
  models/modes, fallback reasons, and troubleshooting.
