# Detection Manager

**Owner:** Vinh Nghiem

Per the tutor's manager-subagent recommendation, Detection is two layers: a
**Detection Manager** that owns the top-level contract, and a **Detection
Subagent** underneath it that does the actual classification work.

## Structure

```text
src/detection/
├── manager.py                    # DetectionManager — owns the contract, delegates
├── subagent.py                   # DetectionSubagent — classifies, handles borderline cases,
│                                  # optionally runs the LLM explanation layer
├── classifier.py                 # tool layer: load model artifact, predict, per-tree vote
│                                  # spread, top_features (feature-importance ranking)
├── train.py                      # trains the baseline RandomForest on CICIDS2017
├── mcp_server.py                 # MCP tool server exposing classifier.py to the LLM layer
├── llm_notes.py                  # LLM output schema, system prompt, validation/grounding checks
├── evaluate.py                   # RF-only vs RF+LLM ablation
├── requirements-detection.txt
└── README.md
```

## Contract

```text
DetectionManager.run(event: TrafficEvent) -> DetectionResult
```

The Manager delegates to the Subagent internally and owns any manager-level
oversight (e.g. later, deciding whether to trust the subagent's result
as-is, or dispatch to a second detection subagent if one gets added).

The Subagent is the actual agent loop: it calls the baseline RandomForest
(`classifier.py`) for `p_anomalous`, and on a borderline score (0.4-0.6) it
doesn't just trust that number — it re-examines via per-tree vote spread (a
second, finer-grained tool call) before finalizing. High tree disagreement
on a borderline call gets flagged in `detector_notes` for downstream
correlation/response to weigh accordingly.

The Detection Manager's conclusion (`DetectionResult`) goes two places:
across to the Mitigation Manager (Callum), since Correlation needs to know
what was detected, and down to the Judge (Minh), who compares it against the
Mitigation Manager's conclusion to produce the final result.

## Training the baseline model

`classifier.py` loads a trained artifact from `classifier.DEFAULT_MODEL_PATH`
(`models/detection_rf.joblib`, gitignored — train locally, don't commit it).
To produce one:

```sh
pip install -r src/detection/requirements-detection.txt
python -m src.detection.train
```

This downloads CICIDS2017 via `kagglehub` (needs Kaggle API credentials —
see the [kagglehub README](https://github.com/Kagglehub/kagglehub#authenticate)),
trains a binary RandomForest (BENIGN vs anomalous), prints a classification
report + ROC-AUC, and saves the artifact.

## Optional LLM explanation layer

`DetectionSubagent(use_llm=True)` adds a bounded LLM loop on top of the baseline
detector. It only ever affects `detector_notes` — the deterministic RandomForest
(`classifier.py`) always computes `is_anomalous` and `confidence`, before the LLM ever
runs.

```text
TrafficEvent
     |
     v
 RandomForest (classifier.predict_proba_anomalous)  --> is_anomalous, confidence (final)
     |
     v
 p_anomalous >= 0.4 (borderline/anomalous)?
     |                                  \
     no                                  yes, and use_llm=True
     |                                    |
     v                                    v
 detector_notes = deterministic       LLM ToolCallingAgent (smolagents) over MCP
 note (or None)                       (mcp_server.py: predict_proba_anomalous,
                                       tree_vote_spread, top_features)
                                            |
                                            v
                                       JSON {category, explanation, tools_used}
                                            |
                                            v
                                       schema + category + grounding check
                                       (llm_notes.parse_and_validate)
                                        /                  \
                                  passes                 fails / times out / errors
                                     |                          |
                                     v                          v
                          "[category=X] explanation"   template note, "[category=Unknown] ..."
```

`DetectionResult.detector_notes` always carries a `[category=...]` tag when `use_llm=True`
fired — the Correlation Subagent embeds this text to search its technique catalog, so a
plausible category still helps even when the LLM explanation itself was discarded.

The MCP connection (the subprocess in the diagram above — it loads the model artifact on
first use) is opened lazily on the first event that needs it and **reused across events**,
not reopened per event. Call `close()` when done (or use `DetectionManager`/`DetectionSubagent`
as a context manager) to shut it down; if it dies mid-run, that event falls back to a template
note and the next one that needs the LLM opens a fresh connection.

### Guardrails

| Risk | Mitigation |
|---|---|
| LLM invents/edits a number | `is_anomalous`/`confidence` are computed by `classifier.py` calls in `subagent.py` before the LLM runs; the LLM's schema has no numeric fields at all |
| LLM hallucinates a feature/value | Grounding check: every feature name in `explanation` must appear in that event's `top_features` output, or the note is discarded (`llm_notes._is_grounded`) |
| LLM returns malformed/off-schema output | `llm_notes.parse_and_validate` checks JSON validity, exact keys, `category` ∈ the fixed list, and explanation length; anything else falls back to a template note |
| LLM/MCP server hangs | Runs on a daemon thread with a wall-clock timeout (`llm_timeout_seconds`, default 20s); a timeout falls back to a template note and drops the connection so the next event reconnects |
| LLM loops forever calling tools | `ToolCallingAgent(max_steps=3)` — at most 3 tool calls per event |
| MCP connection dies mid-run | The exception is caught in `_run_llm_layer`, logged, and the connection is dropped (`_reset_llm_connection`); that event falls back to a template note, `run()` never raises, and the next event that needs the LLM reconnects |
| Reopening the MCP subprocess (and reloading the ~68 MB model artifact) on every event | Connection is opened once, lazily, and reused across events (`_ensure_llm_connection`); only a lightweight `ToolCallingAgent` is rebuilt per event |
| LLM called on obviously benign traffic (cost/latency) | Only invoked when `p_anomalous >= 0.4` — clear benign events skip it entirely |
| Public contract changes | `DetectionManager.run(event) -> DetectionResult` is unchanged; `use_llm` is an opt-in constructor flag, default `False` |

### Running the MCP server

Mostly for manual testing — `DetectionSubagent` launches this itself as a subprocess when
`use_llm=True`:

```sh
python -m src.detection.mcp_server
```

Exposes `predict_proba_anomalous`, `tree_vote_spread`, and `top_features` over stdio.
Requires a trained model artifact (see above) — it loads (and fails fast on a missing one)
on first tool call.

### Enabling the LLM layer

```python
from src.detection.manager import DetectionManager

with DetectionManager(use_llm=True) as manager:  # closes the MCP connection on exit
    for event in events:
        result = manager.run(event)
```

`manager.close()` also works if you're not using it as a context manager (e.g. it's kept
alive for the app's lifetime and closed on shutdown). The MCP connection opens on the first
event that needs it and is reused after that — don't create a new `DetectionManager` per
event, or you'll reload the model artifact every time.

Configure the model via a `.env` file at the repo root (copy `.env.example`):

```sh
cp .env.example .env
ollama pull qwen2.5:7b   # or whatever model DETECTION_LLM_MODEL names
ollama serve             # if not already running
```

`DETECTION_LLM_MODEL` defaults to `ollama_chat/qwen2.5:7b` (local, no API key). To use
Gemini instead (as in the labs), set `DETECTION_LLM_MODEL=gemini/<model-id>` and
`DETECTION_LLM_API_KEY=<your key>`.

### Running the evaluation

```sh
python -m src.detection.evaluate --sample-size 200
```

Runs `DetectionManager` with `use_llm=False` and `use_llm=True` over the same stratified
held-out sample (weighted to include borderline events), prints an accuracy/F1/latency
comparison table, and writes per-event results to `src/detection/results/*.csv` (gitignored).
Accuracy/F1 should be identical between the two arms — that's the guardrail proof that the
LLM never changes the decision.

## Dependencies

Anything beyond the shared `requirements.txt` is in `requirements-detection.txt`:
`kagglehub` (dataset download), `joblib` (artifact save/load), and, for the optional LLM
layer, `mcp` (MCP server; pinned `<2` — its FastMCP API was renamed in mcp 2.x),
`smolagents[mcp]` (MCPClient + ToolCallingAgent), `litellm` (model backend, via
`LiteLLMModel`), and `pydantic` (LLM output/tool schema validation).
