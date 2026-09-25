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
│                                  # chooses the attack category, optionally delegates to
│                                  # llm_layer.py for an explanation. Reads top-to-bottom as:
│                                  # classify -> tree votes -> category decision -> optional
│                                  # LLM explanation.
├── llm_layer.py                  # LLMExplanationLayer — MCP connection, agent/single_shot
│                                  # dispatch, retries, circuit breaker, validation/salvage.
│                                  # Everything DetectionSubagent needs to ask an LLM to
│                                  # *explain* an already-made decision, and nothing that
│                                  # decides is_anomalous/confidence/category.
├── classifier.py                 # tool layer: load artifacts, predict, per-tree vote spread,
│                                  # top_features (feature-importance ranking), predict_attack_category
├── training/
│   ├── data.py                   # shared loading/cleaning, the ONE train/test split (fixed
│   │                              # random_state), feature stats, map_cicids_label_to_category
│   ├── train_binary.py           # binary RandomForest (BENIGN vs anomalous) -> detection_binary.joblib
│   └── train_category.py         # multiclass RandomForest (+ a downsampled Benign class),
│                                  # anomalous TRAIN rows -> detection_category.joblib
├── train.py                      # thin wrapper: runs train_binary then train_category
├── mcp_server.py                 # MCP tool server exposing classifier.py to the LLM layer
├── llm_notes.py                  # LLM output schema, system prompt, validation/grounding checks
├── evaluate.py                   # RF-only vs RF+LLM ablation, category-accuracy report
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

## Training the models

`classifier.py` loads two artifacts: `classifier.DEFAULT_BINARY_MODEL_PATH`
(`models/detection_binary.joblib`) and `classifier.DEFAULT_CATEGORY_MODEL_PATH`
(`models/detection_category.joblib`), both gitignored — train locally, don't commit them.
To produce them:

```sh
pip install -r src/detection/requirements-detection.txt
python -m src.detection.train
```

This is a thin wrapper around `training/train_binary.py` then `training/train_category.py`.
Both load and clean CICIDS2017 via `training/data.py` (`kagglehub` — needs Kaggle API
credentials, see the
[kagglehub README](https://github.com/Kagglehub/kagglehub#authenticate)) and split it **once**,
with `data.split_train_test` (fixed `random_state=42`, `test_size=0.2`) — the single
authoritative split every training script and `evaluate.py` shares, so the held-out test set
was never seen by either model:

- **`train_binary.py`** trains the binary RandomForest (BENIGN vs anomalous) on the TRAIN split,
  prints a classification report + ROC-AUC, and saves `{model, feature_names, feature_medians,
  feature_mad}` to `detection_binary.joblib`.
- **`train_category.py`** trains a *multiclass* RandomForest on the TRAIN split's anomalous
  rows, with labels mapped via `data.map_cicids_label_to_category` (CICIDS2017's raw multiclass
  `Label` collapsed to a fixed category vocabulary — `DoS`, `DDoS`, `PortScan`, `BruteForce`,
  `WebAttack`, `Botnet`, `Infiltration`, `Unknown`), **plus an explicit `classifier.BENIGN_CATEGORY`
  ("Benign") class** built from BENIGN rows, downsampled to `DEFAULT_BENIGN_TO_ATTACK_RATIO`
  (2.0) times the attack-row count so training stays fast — see "Category decision" below for
  why Benign exists at all. Unrecognized labels are still excluded from training entirely.
  Evaluated on the TEST split at its **real, undownsampled** class balance. Prints per-class
  precision/recall/F1, row counts per class (train and test), and macro F1, then saves
  `{category_model, classes, feature_names}` to `detection_category.joblib`.

Both artifacts store `feature_names`; `classifier.assert_feature_names_match` checks they're
identical (same columns, same order) whenever both are loaded together — `DetectionSubagent`
calls this at construction time and raises immediately on a mismatch (e.g. one model retrained
after a feature-engineering change, the other not), rather than silently misaligning feature
columns later. The category model is optional for backward compatibility: an install with only
`detection_binary.joblib` (or an artifact predating `train_category.py`) still works —
`DetectionSubagent` just reports every category as `"Unknown"`.

**Migrating from the old single-artifact layout:** earlier versions saved one combined
`models/detection_rf.joblib`. That layout is no longer read; if `classifier.load_artifact` can't
find the new path but finds the old one, it raises a `FileNotFoundError` naming both paths and
telling you to run `python -m src.detection.train` — not a generic "file not found."

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
 is_anomalous?
     |                                  \
     no                                  yes
     |                                    |
     v                                    v
 detector_notes = None            _choose_category(event)  [code, deterministic — see below]
                                          |
                                          v
                                   p_anomalous >= 0.4 (borderline/anomalous)?  --  circuit open?
                                          |                                    --> skip LLM, template note
                                          |                                        (category tag still applied)
                                    use_llm=True, circuit closed
                                          |
                                          v
                                   register_event(event_id, features)  [code, not LLM]
                                   LLM ToolCallingAgent (smolagents) over MCP —
                                   task prompt carries event_id, p_anomalous, tree_vote_std,
                                   AND the already-chosen category + its probability — never
                                   raw features. Tools offered: top_features(event_id, k),
                                   tree_vote_spread(event_id) (only when borderline),
                                   predict_attack_category(event_id) (for the LLM's own
                                   investigation only — it does not change the decision),
                                   and final_answer.
                                          |
                                          v
                                   clear_event(event_id)  [code, not LLM]
                                   raw text: JSON via final_answer, or (if the model
                                   never called it) smolagents' own un-tooled fallback
                                   generation — often JSON with stray prose around it
                                          |
                                          v
                                   schema + grounding check (llm_notes.parse_and_validate) —
                                   {explanation, tools_used} only, no category field to
                                   validate; on invalid JSON, retry once on an extracted
                                   {...} substring (llm_notes.extract_json_object — "salvaged")
                                    /                  \
                              passes                 fails / times out / errors
                                 |                          |
                                 v                          v
                      "[category=X] explanation"   template note, "[category=X] ..."
                      (X = code's chosen category   (X = code's chosen category either way;
                      either way)                    + counts toward the circuit breaker)
```

`DetectionResult.detector_notes` always carries a `[category=...]` tag for every truly-anomalous
event, **regardless of `use_llm`** — the category decision is a deterministic classifier call
(see "Category decision" below), not an LLM output, so it happens whether or not the LLM layer
runs at all. The Correlation Subagent embeds this text to search its technique catalog.

The MCP connection (the subprocess in the diagram above — it loads the model artifact on
first use) is opened lazily on the first event that needs it (or by `warmup()`, see below)
and **reused across events**, not reopened per event. `register_event`/`clear_event` are real
MCP tools, but only code calls them directly — they're filtered out of the tool list the LLM
agent sees. `predict_proba_anomalous` is also hidden from the agent: code already computed
`p_anomalous` and put it in the task, so offering the tool just tempts a small/slow model into
a wasted step (it stays on the MCP server for completeness/tests). The agent sees
`top_features` (always), `tree_vote_spread` (only on a borderline event — it's not meaningful
otherwise), and `predict_attack_category` (always — for the LLM's own investigation; code has
already made the final category decision from the same underlying model before the agent ever
runs, so calling it changes nothing), plus smolagents' own built-in `final_answer` tool, all keyed by a short
`event_id` rather than a ~78-feature dict. That's most of what keeps the prompt small on a slow
local model. Call `close()` when done (or use `DetectionManager`/`DetectionSubagent` as a
context manager); if the connection dies mid-run, that event falls back to a template note, and
repeated failures open a circuit breaker rather than reconnecting (and re-paying a slow cold
start) on every following event.

**Why `final_answer` matters:** `ToolCallingAgent` requires every step to be an actual tool
call, and a run only ends when the model calls the built-in `final_answer` tool — it is not
enough for the model to just type its JSON answer as a chat message; that step has no tool
call in it and fails to parse ("model output does not contain any JSON blob"), especially on
smaller local models. `SYSTEM_PROMPT` is written around this explicitly ("call top_features,
then call final_answer with..."), and kept short — small models follow short, concrete
instructions more reliably than a longer rules list.

### Two LLM modes: agent vs single_shot

Real-world finding (verified against `ollama_chat/qwen2.5:3b`): even with the prompt above,
the agent loop is fundamentally unreliable for a small model behind Ollama. The obvious next
fix — force tool use via `tool_choice="required"` — doesn't work, because **litellm's own
`ollama` transformation strips `tool_choice` from every request** (its code comment says why:
`"causes ollama requests to hang"`). So nothing can force `ToolCallingAgent`'s "every step is a
tool call" requirement onto this backend; the model sometimes just answers in prose mid-loop,
hits `AgentParsingError`, and after `max_steps` smolagents falls back to one direct, un-tooled
generation.

What *is* honored: `response_format={"type": "json_schema", ...}`. litellm forwards a
`json_schema` response format to Ollama as its `format` parameter, and Ollama enforces it with
grammar-constrained decoding — verified directly against `qwen2.5:3b`: the `category` field
came back reliably constrained to the exact enum, with no prompting trick needed. That's
`DETECTION_LLM_MODE=single_shot`:

```text
DETECTION_LLM_MODE=agent                        DETECTION_LLM_MODE=single_shot
  ToolCallingAgent decides which tools            code calls top_features (+ tree_vote_spread
  to call, then must call final_answer            if borderline) directly via MCP, then ONE
  — reliable for hosted models with real           litellm.completion call with
  tool_choice support (e.g. Gemini)                response_format=ANSWER_JSON_SCHEMA
                                                   — reliable for small local models
```

Either way, the raw text goes through the exact same `parse_and_validate`/salvage pipeline in
`LLMExplanationLayer._run_llm_layer` — `single_shot` is not a way to skip any guardrail, it only
changes how the raw JSON gets produced. Set via `DETECTION_LLM_MODE=agent|single_shot` in
`.env` to force one explicitly. Left unset (or an unrecognized value): `llm_layer.resolve_llm_mode`
defaults to `single_shot` for any `ollama_chat/*` `DETECTION_LLM_MODEL`, `agent` otherwise —
since `DETECTION_LLM_MODEL` itself defaults to a local Ollama model, the out-of-the-box default
is `single_shot`. `evaluate.py` records which mode ran in the `llm_mode` CSV column and prints
it alongside the model id at startup.

**Recommendation:** for a hosted, capable model (Gemini), use `DETECTION_LLM_MODE=agent`
(or just don't set `DETECTION_LLM_MODEL` to an `ollama_chat/*` id) — it actually gets to use the
multi-step tool-investigation loop as designed. For a local Ollama model (any size, this is a
litellm↔Ollama limitation, not a model-capability one), `single_shot` is now the default and
doesn't need setting explicitly — the agent loop's reliance on `tool_choice` cannot work there.

### Category decision: a deterministic tool, not the LLM

**This used to be an LLM output**, gated by prompt instructions and a fixed-list/synonym check
(`llm_notes._normalize_category`, since removed). A real evaluation run found the LLM-picked
category scored **0% accuracy on n=8** truly-anomalous events with a category — against a 50%
"always guess the most common true category" baseline on that same set. Correct-but-non-committal
("Unknown") beat confidently-wrong, but the LLM was never actually *choosing well*, and n=8 is
far too small a sample to justify tuning the prompt further — that risks overfitting to those 8
events rather than fixing anything general. Real fix: same principle as `is_anomalous`/
`confidence` — stop asking the LLM to decide, and make it a deterministic classifier call.

`DetectionSubagent._choose_category` (called for every truly-anomalous event, whether or not
`use_llm` is set):

1. Calls `classifier.predict_attack_category(category_artifact, event.features, top_k=3)` — a
   multiclass RandomForest (`train_category.py`, trained on TRAIN-split anomalous rows only)
   returns per-category probabilities, ranked descending.
2. If the top class's probability `>= CATEGORY_CONFIDENCE_THRESHOLD` (default `0.6`,
   `DetectionSubagent(category_confidence_threshold=...)`), that class is the chosen category.
   Otherwise the chosen category is `"Unknown"` — reporting a low-confidence guess as if it were
   solid is worse than admitting the model isn't sure.
3. **If the top class is `classifier.BENIGN_CATEGORY` ("Benign")**, the binary model (which
   already called this event anomalous) and the category model disagree — the chosen category
   is `"Unknown"` (never `"Benign"`, which would contradict `is_anomalous=True`), logged as its
   own `action="model_disagreement"` trace step, and this overrides the confidence threshold
   entirely: even a *high-confidence* Benign vote is a disagreement, not a trustworthy answer.
   **Why Benign is a class at all:** an earlier version of the category model trained on attack
   rows only, so it had no way to express "this doesn't look like an attack" — a real evaluation
   run found 5 events the binary model wrongly flagged anomalous all got a confident attack
   category anyway (Botnet 0.97-1.0 ×4, DoS 0.99), because that was the only kind of answer the
   model could give. `evaluate.py`'s `compute_false_positive_categorization` (see below) is the
   metric that would have caught this — `compute_category_report` never scores benign ground
   truth, since it only looks at truly anomalous events.
4. Otherwise: this is logged as its own trace step (`action="category_decision"`, `tool_input`
   carries `chosen_category`, `raw_top_category`, `raw_top_probability`, and `threshold`) so
   `evaluate.py` can report both the thresholded decision and the classifier's raw top-1
   accuracy from the same run.
5. No category model loaded (backward compatibility) → logged as `category_model_unavailable`,
   category is always `"Unknown"`.

The LLM never sees this as something to decide: its task prompt states the chosen category and
its probability as a *given*, and its JSON schema (`llm_notes.ANSWER_JSON_SCHEMA`) has no
`category` property at all — only `{explanation, tools_used}`. A stray `"category"` key in the
LLM's raw JSON (e.g. a stale cached prompt, or a model that doesn't strictly follow schemas) is
simply ignored by `parse_and_validate`, not validated or rejected — there's no "bad category"
outcome left, because the LLM's output can no longer disagree with the decision. In `agent` mode
the LLM may still call `predict_attack_category` itself, purely to inform its own explanation;
whatever it sees there does not change the category code already chose before the agent ran.

### Diagnostics: seeing what the model actually wrote

Two extra trace actions (surfaced as `llm_step_failures` / `llm_forced_final_answer` columns
in `evaluate.py`'s CSV) exist purely to debug the agent-mode failure above without re-running
anything: `_log_agent_memory_diagnostics` inspects `agent.memory.steps` after `agent.run()`
returns and logs the raw text of every step that failed to parse as a tool call
(`llm_step_failed`), plus the forced, un-tooled final answer's raw text if the model never
called `final_answer` at all (`llm_forced_final_answer`). These are diagnostic only — best-
effort and never fatal (wrapped so a mocked/non-standard `agent.memory` in tests can't break
the main flow) — and don't affect validation or the fallback_reason.

### Guardrails

| Risk | Mitigation |
|---|---|
| LLM invents/edits a number | `is_anomalous`/`confidence` are computed by `classifier.py` calls in `subagent.py` before the LLM runs; the LLM's schema has no numeric fields at all |
| LLM invents/edits the attack category | `DetectionSubagent._choose_category` decides it deterministically from `classifier.predict_attack_category`, before the LLM runs; the LLM's schema has no `category` field, and a stray one in its raw JSON is ignored, not validated — see "Category decision" above |
| LLM hallucinates a feature/value | Grounding check: every feature name in `explanation` must appear in that event's `top_features` output, or the note is discarded (`llm_notes._is_grounded`) |
| LLM returns malformed/off-schema output | `llm_notes.parse_and_validate` checks JSON validity, exact keys, and explanation length; anything else falls back to a template note, tagged with a specific `fallback_reason` |
| Large ~78-feature dict inflates the prompt and slows a local CPU model | LLM-facing tools are `event_id`-based (`register_event`/`clear_event` are code-only, hidden from the agent); the task prompt carries only `event_id`, `p_anomalous`, and `tree_vote_std` |
| A tool the LLM doesn't need wastes a step on a small/slow model (e.g. re-deriving `p_anomalous`, which code already computed) | `predict_proba_anomalous` is never offered to the agent (`LLM_HIDDEN_TOOL_NAMES`); `tree_vote_spread` is offered only on a borderline event, since it's not meaningful otherwise |
| Slow local generation runs away | `max_tokens` capped (`MAX_LLM_OUTPUT_TOKENS`, default 300) via `LiteLLMModel` |
| `ToolCallingAgent` requires ending via the `final_answer` tool call — a small model that just types JSON as chat text fails to parse ("model output does not contain any JSON blob") | `SYSTEM_PROMPT` explicitly says "call top_features, then call final_answer with your JSON answer" (not "output ONLY a JSON object"), and is kept short |
| Even so, the model exhausts its steps without ever calling `final_answer` (smolagents then does one direct, un-tooled generation and returns that raw text) | Salvage path: if the raw text fails `parse_and_validate` as pure JSON, `llm_notes.extract_json_object` pulls out a `{...}` substring and the *same* guardrails re-run on that; success is tagged `fallback_reason="salvaged"` (distinct from a clean pass) so it's visible in reporting, not silently the same as a normal success |
| `tool_choice="required"` can't actually force a small local model through the agent loop — litellm's `ollama` transformation drops `tool_choice` entirely ("causes ollama requests to hang") | `DETECTION_LLM_MODE=single_shot`: code calls `top_features`/`tree_vote_spread` directly, one `litellm.completion` call with `response_format`'s `json_schema` (which Ollama *does* enforce, via grammar-constrained decoding) — same `parse_and_validate` pipeline either way; see "Two LLM modes" above |
| LLM/MCP server hangs | Runs on a daemon thread with a wall-clock timeout (`llm_timeout_seconds`, default 20s, configurable — see Config below); a timeout falls back to a template note |
| LLM loops forever calling tools | `ToolCallingAgent(max_steps=3)` — at most 3 tool calls per event |
| Model cold start (subprocess spawn + first inference) eats into the first event's timeout | `warmup()` opens the connection and runs one tiny prompt before the timed run starts, retrying transient errors; `evaluate.py` calls it and reports the time separately |
| Hosted provider (Gemini etc.) rate-limits or is briefly unavailable (429/5xx) | `llm_layer._call_with_retry` retries with exponential backoff (2s/4s/8s, `LLM_RETRY_DELAYS_SECONDS`) inside the per-event timeout, both in `warmup()` and per event; exhausted retries are tagged `rate_limited`/`provider_unavailable`, distinct from a generic `exception`, and don't drop the MCP connection (the failure is provider-side, not the MCP subprocess) |
| smolagents wraps the model-call failure in its own `AgentGenerationError` before we ever see it, hiding the real litellm error's `status_code` | `llm_layer._classify_llm_error` walks `__cause__`/`__context__` (depth-limited, cycle-guarded) instead of only looking at the exception it's handed, and classifies on the first link that resolves |
| smolagents' own built-in retryer (rate-limit-shaped errors only, ~60s+~120s of internal sleeping by default) would blow past our per-event timeout before ever raising — and does nothing for 5xx at all | `LiteLLMModel(..., retry=False)` in `LLMExplanationLayer._get_llm_model`: our `_call_with_retry` (bounded by the per-event timeout) is the single retry authority for both rate limits and provider-unavailable errors, see the comment on `_get_llm_model` |
| Warmup itself fails (bad model id/API key, provider down) and would otherwise crash the eval run | `warmup()` never raises — returns `{"ok", "elapsed_seconds", "reason"}`; `evaluate.py` prints the reason and skips the `+LLM` arm cleanly, still printing/saving the RF-only results |
| MCP connection dies mid-run (not a provider-side error) | The exception is caught in `LLMExplanationLayer._run_llm_layer`, logged, and disconnected immediately (thread already finished); that event falls back to a template note, `run()` never raises |
| A timed-out call's connection can't be touched safely from the main thread (the abandoned thread may still be using it) | It's queued in `LLMExplanationLayer._abandoned_mcp_clients` instead of disconnected on the spot; `close()` disconnects it, and the abandoned thread disconnects it itself (in its own `finally`) once it notices it's no longer the active connection |
| Repeated timeouts cascade into reconnect-then-timeout-again on every following event | Circuit breaker: after `circuit_breaker_threshold` (default 3) consecutive LLM failures, the LLM is skipped entirely (no more connection attempts) for the rest of the run, logged as `llm_circuit_open` |
| An abandoned (timed-out) worker thread's `log_step` calls land in a *later* event's trace | The worker buffers its steps in a local list (`step_buffer`) instead of calling the stored `log_step` callback directly; `LLMExplanationLayer._run_llm_layer` only replays that buffer once it knows the call finished within the timeout — an abandoned call's buffer is simply never read |
| Numpy scalars aren't JSON-serializable over the MCP wire | Features are cast to plain Python `float` before `register_event` (and in `classifier.top_features`'s returned `value`) |
| Reopening the MCP subprocess (and reloading the ~68 MB model artifact) on every event | Connection is opened once, lazily, and reused across events (`LLMExplanationLayer._ensure_llm_connection`); only a lightweight `ToolCallingAgent` is rebuilt per event |
| LLM called on obviously benign traffic (cost/latency) | Only invoked when `p_anomalous >= 0.4` — clear benign events skip it entirely |
| Public contract changes | `DetectionManager.run(event) -> DetectionResult` is unchanged; `use_llm` is an opt-in constructor flag, default `False` |

### Running the MCP server

Mostly for manual testing — `DetectionSubagent` launches this itself as a subprocess when
`use_llm=True`:

```sh
python -m src.detection.mcp_server
```

Exposes `register_event`/`clear_event` (code-only — store/remove an event's features under an
id), `predict_proba_anomalous` (kept for completeness/tests — the LLM agent never sees it, see
Guardrails above), and the `event_id`-based `tree_vote_spread`, `top_features`, and
`predict_attack_category` the agent actually gets, all over stdio. Requires trained model
artifacts (see above) — each is lazily loaded (and fails fast on a missing one) on first use.

### Enabling the LLM layer

```python
from src.detection.manager import DetectionManager

with DetectionManager(use_llm=True) as manager:  # closes the MCP connection on exit
    manager.warmup()  # opens the connection + primes the model, outside any per-event timeout
    for event in events:
        result = manager.run(event)
```

`manager.close()` also works if you're not using it as a context manager (e.g. it's kept
alive for the app's lifetime and closed on shutdown). The MCP connection opens on the first
event that needs it (or on `warmup()`) and is reused after that — don't create a new
`DetectionManager` per event, or you'll reload the model artifact every time.

Configure the model via a `.env` file at the repo root (copy `.env.example`):

```sh
cp .env.example .env
ollama pull qwen2.5:3b   # or whatever model DETECTION_LLM_MODEL names
ollama serve             # if not already running
```

`DETECTION_LLM_MODEL` defaults to `ollama_chat/qwen2.5:3b` (local, no API key). To use
Gemini instead (as in the labs), set `DETECTION_LLM_MODEL=gemini/<model-id>` (check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for a
current model id — these get retired) and `DETECTION_LLM_API_KEY=<your key>`. Gemini's free
tier is rate-limited (requests/day and requests/minute caps) — `evaluate.py` retries a 429
with backoff, but for a full run at default settings also pass `--llm-delay` (see below).

#### Config

| Setting | Source (highest priority first) |
|---|---|
| Per-event LLM timeout | `DetectionManager(llm_timeout_seconds=...)` constructor arg → `DETECTION_LLM_TIMEOUT` env var → 60s default |
| Circuit breaker threshold | `DetectionManager(circuit_breaker_threshold=...)` constructor arg → 3 (default) |
| Model id / API key | `DETECTION_LLM_MODEL` / `DETECTION_LLM_API_KEY` env vars (`.env`) — `evaluate.py` prints the resolved `DETECTION_LLM_MODEL` at startup; defaults to `ollama_chat/qwen2.5:3b` |
| LLM mode (`agent` vs `single_shot`) | `DETECTION_LLM_MODE` env var (`.env`) when set to a recognized value; otherwise defaults to `single_shot` for any `ollama_chat/*` model, `agent` otherwise (`llm_layer.resolve_llm_mode`) — see "Two LLM modes" above |
| Retry backoff schedule (rate limit / 5xx) | `llm_layer.LLM_RETRY_DELAYS_SECONDS` (2s, 4s, 8s — 3 retries, 4 attempts total); not currently exposed via env var |

### Running the evaluation

```sh
python -m src.detection.evaluate --sample-size 200 --sampling random --llm-timeout 20 --llm-delay 0
```

Prints the resolved `DETECTION_LLM_MODEL`, then runs `DetectionManager` with `use_llm=False`
and `use_llm=True` over the same held-out sample. Warms up the LLM connection first (reported
separately from per-event latency) — if warmup fails after retries (bad model id/API key,
provider down), it prints `LLM warmup failed: <reason>. Check DETECTION_LLM_MODEL / API key /
provider status.` and skips the `+LLM` arm entirely, still printing/saving the RF-only results.
Otherwise it prints an accuracy/F1/latency comparison table plus fallback-reason counts, and
writes per-event results to `src/detection/results/rf_only_<sampling>.csv` /
`rf_llm_<sampling>.csv` (gitignored) — including each event's `fallback_reason` (`timeout` /
`exception` / `rate_limited` / `provider_unavailable` / `invalid_json` /
`ungrounded` / `circuit_open` / `salvaged` / empty for success or "never invoked") and its `prompt_tokens`
/ `completion_tokens` (from the LLM provider's own usage reporting; empty when the LLM wasn't
invoked or never returned a response). Accuracy/F1 should be identical between the two arms —
that's the guardrail proof that the LLM never changes the decision. Whether an event was
actually dispatched to the LLM (`llm_invoked` / the `% events -> LLM` line) is counted from an
unbuffered `llm_dispatch` step logged before the timed call, not the buffered
`llm_layer_start` — otherwise a timed-out event would be undercounted as "never invoked" even
though it clearly was.

`--sampling random` (default) draws a plain stratified-by-label sample, representative of the
real class distribution. `--sampling borderline` instead biases the sample toward events near
the decision boundary, so it actually exercises the LLM trigger condition — useful when you
specifically want to stress-test the LLM path, at the cost of the sample no longer reflecting
real-world class balance.

`--llm-delay <seconds>` sleeps after every event that dispatched to the LLM (not after RF-only
events) — use this to stay under a hosted provider's free-tier rate limit over a full run.

#### Category accuracy

Each sampled event's original CICIDS2017 `Label` (multiclass — e.g. "DoS Hulk", "Web Attack �
XSS" — normally discarded by `train_binary.py`'s BENIGN-vs-anomalous binarization) is kept
alongside it and mapped via `data.map_cicids_label_to_category` to the same fixed category
vocabulary `train_category.py`/`_choose_category` use (`Heartbleed` maps to `Unknown`; `BENIGN`
and anything unrecognized map to `None`). It's written to each CSV row as `true_category`.

Since the category decision is now a deterministic classifier call (see "Category decision"
above) — not an LLM output — `compute_category_report` scores it over **every truly anomalous
event in the sample with a known `true_category`**, regardless of `use_llm` or whether the LLM
ran/succeeded on that event (there's nothing LLM-dependent left to gate on). `print_category_report`
(in the console summary) reports:

- **category accuracy** — exact match rate of the code-chosen (thresholded) category against
  `true_category`
- **raw accuracy** — exact match rate of the classifier's raw top-1 class (before the
  `CATEGORY_CONFIDENCE_THRESHOLD` gate), so a low threshold's cost (in punted-to-Unknown events)
  is visible against what the classifier could do unthresholded
- **% predicted Unknown** — how often the thresholded decision punted (either no category model,
  or the top class didn't clear the confidence threshold)
- **confusion table** — true category (rows) vs. code-chosen category (columns)
- **trivial baseline** — accuracy of always guessing whichever true category was most common in
  that same scored set, so a beaten baseline actually means something

**Prior finding that motivated this redesign:** an earlier version had the LLM choose the
category directly (validated against a fixed list/synonym table). A real run scored that at
**0% category accuracy on n=8** truly-anomalous events with a known category, against a 50%
"always guess the most common true category" baseline on that same n=8 — the LLM answered
"Unknown" for every one of them. Non-committal-but-correct beats confidently-wrong, but n=8 is
far too small to justify tuning the prompt further (that risks overfitting to those 8 events),
and the LLM was never actually *choosing well* in the first place. See "Category decision"
above for the fix and its results on the current (larger, RF-driven) evaluation.

#### False-positive categorisation and model disagreement

`compute_category_report` only ever looks at *truly anomalous* events (`true_label == 1`), so it
can't see what happens on the binary model's false positives — genuinely benign traffic it
wrongly called anomalous. `compute_false_positive_categorization` covers exactly that gap:
among events with `true_label == 0` that the binary model still flagged anomalous, what
percentage got a specific attack category rather than `"Unknown"`. This is the metric that
surfaced the original bug (see "Category decision" above): before `train_category.py` had a
Benign class, this was consistently well above 0%; after it, it should be ~0%, since the
category model can now vote Benign on those events (triggering `model_disagreement` — see
below) instead of always guessing an attack. `print_summary` prints it right after the category
report, and `evaluate.py`'s CSV carries each row's `model_disagreement` flag (True when
`_choose_category`'s top vote was Benign for that event) — the console summary also prints the
total `model_disagreement_count` for the run.

## Dependencies

Anything beyond the shared `requirements.txt` is in `requirements-detection.txt`:
`kagglehub` (dataset download), `joblib` (artifact save/load), and, for the optional LLM
layer, `mcp` (MCP server; pinned `<2` — its FastMCP API was renamed in mcp 2.x),
`smolagents[mcp]` (MCPClient + ToolCallingAgent), `litellm` (model backend, via
`LiteLLMModel`), and `pydantic` (LLM output/tool schema validation).
