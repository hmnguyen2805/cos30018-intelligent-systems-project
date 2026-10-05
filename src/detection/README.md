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
- **`"[category=X] [label=Y] <explanation>"`** — the event was anomalous.
  `X` and `Y` are always present and always code-decided (never an LLM
  output). `X` is the coarse category: `DoS`, `DDoS`, `PortScan`,
  `BruteForce`, `WebAttack`, `Botnet`, `Infiltration`, `Heartbleed`, or
  `"Unknown"`. `Y` is the fine CICIDS2017 label (`DoS Hulk`, `FTP-Patator`,
  `Web Attack - XSS`, ...) or `"Unknown"`. `<explanation>` is either an
  LLM-generated explanation (when `use_llm=True` and it succeeds) or a fixed
  template sentence.

**Decision rule** (threshold `CATEGORY_CONFIDENCE_THRESHOLD`, default 0.80;
a group's probability is the sum of its fine labels'):
top fine label >= threshold -> `Y` = that label, `X` = its group; else top
group >= threshold -> `Y` = `Unknown`, `X` = that group; else both `Unknown`.
Benign top -> both `Unknown` (see below).

**What `Unknown` means:** for `Y`, no single fine label cleared the
threshold. For `X` (and `Y`), not even the group did, or the classifier voted
`Benign` on an event the binary model already called anomalous (a
disagreement between the two models — never reported as `Benign`, since
that would contradict `is_anomalous=True`). Downstream agents should treat
`Unknown` as "flagged anomalous, not resolved," not as an attack type.
`X` can be known while `Y` is `Unknown` (e.g. DoS sub-types indistinguishable).

**Known limitation:** `Web Attack - XSS` and `Web Attack - Brute Force`
are largely confused with each other (XSS F1 ~0.16); the coarse `WebAttack`
category is reliable (F1 ~0.985). Prefer `X` unless you need the sub-type —
see [docs/evaluation.md](docs/evaluation.md).

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
