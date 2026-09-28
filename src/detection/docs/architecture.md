# Architecture

How the Detection module is put together: the file map, the manager/subagent
contract, the detection flow diagram, the MCP server's role, and the two LLM
interaction modes. For *why* things are built this way, see
[design-decisions.md](design-decisions.md). For guardrails, env vars, and
troubleshooting, see [llm-layer.md](llm-layer.md). For current metrics, see
[evaluation.md](evaluation.md).

## File map

| File | Role |
|---|---|
| `manager.py` | `DetectionManager` — owns the contract, delegates |
| `subagent.py` | `DetectionSubagent` — classifies, handles borderline cases, chooses the attack category, optionally delegates to `llm_layer.py` for an explanation. Reads top-to-bottom as: classify -> tree votes -> category decision -> optional LLM explanation |
| `llm_layer.py` | `LLMExplanationLayer` — MCP connection, agent/single_shot dispatch, retries, circuit breaker, validation/salvage. Everything `DetectionSubagent` needs to ask an LLM to *explain* an already-made decision, and nothing that decides `is_anomalous`/`confidence`/category |
| `classifier.py` | Tool layer: load artifacts, predict, per-tree vote spread, `top_features` (feature-importance ranking), `predict_attack_category` |
| `training/data.py` | Shared loading/cleaning, the ONE train/test split (fixed `random_state`), feature stats, `map_cicids_label_to_category` |
| `training/train_binary.py` | Binary RandomForest (BENIGN vs anomalous) -> `detection_binary.joblib` |
| `training/train_category.py` | Multiclass RandomForest (+ a downsampled Benign class), anomalous TRAIN rows -> `detection_category.joblib` |
| `train.py` | Thin wrapper: runs `train_binary` then `train_category` |
| `mcp_server.py` | MCP tool server exposing `classifier.py` to the LLM layer |
| `llm_notes.py` | LLM output schema, system prompt, validation/grounding checks |
| `evaluate.py` | RF-only vs RF+LLM ablation (sampled) + `--offline` CLI entry |
| `offline_eval.py` | Batch (no-LLM, whole-test-split) classifier metrics and the `CATEGORY_CONFIDENCE_THRESHOLD` sweep, used by `--offline` |
| `requirements-detection.txt` | Extra dependencies beyond the shared `requirements.txt` |

## Manager/subagent contract

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

## Detection flow diagram

`DetectionSubagent(use_llm=True)` adds a bounded LLM loop on top of the
baseline detector. It only ever affects `detector_notes` — the deterministic
RandomForest (`classifier.py`) always computes `is_anomalous` and
`confidence`, before the LLM ever runs.

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

`DetectionResult.detector_notes` always carries a `[category=...]` tag for
every truly-anomalous event, **regardless of `use_llm`** — the category
decision is a deterministic classifier call (see below), not an LLM output,
so it happens whether or not the LLM layer runs at all. The Correlation
Subagent embeds this text to search its technique catalog.

## The MCP server's role

`mcp_server.py` exposes `classifier.py` as MCP tools; it never decides
anything itself — every decision (`is_anomalous`, `confidence`, the chosen
category) is made in code in `subagent.py` before or independent of any tool
call. Run it standalone for manual testing:

```sh
python -m src.detection.mcp_server
```

It exposes, over stdio: `register_event`/`clear_event` (code-only — store/
remove an event's features under an id; filtered out of the tool list the
LLM agent sees), `predict_proba_anomalous` (kept for completeness/tests —
code already computed `p_anomalous` and put it in the task, so offering the
tool would just tempt a small/slow model into a wasted step; hidden from the
agent via `LLM_HIDDEN_TOOL_NAMES`), and the `event_id`-based
`tree_vote_spread` (only offered on a borderline event, since it's not
meaningful otherwise), `top_features` (always offered), and
`predict_attack_category` (always offered — for the LLM's own investigation;
code has already made the final category decision from the same underlying
model before the agent ever runs, so calling it changes nothing), plus
smolagents' own built-in `final_answer` tool. Everything is keyed by a short
`event_id` rather than a ~78-feature dict — that's most of what keeps the
prompt small on a slow local model. Requires trained model artifacts (see
the [README](../README.md#quick-start)) — each is lazily loaded
(and fails fast on a missing one) on first use.

`DetectionSubagent` launches this itself as a subprocess when `use_llm=True`
— running it standalone is mostly for manual testing. The connection (it
loads the model artifact on first use) is opened lazily on the first event
that needs it (or by `warmup()`) and reused across events, not reopened per
event. Call `close()` when done (or use `DetectionManager`/`DetectionSubagent`
as a context manager).

**Why `final_answer` matters:** `ToolCallingAgent` requires every step to be
an actual tool call, and a run only ends when the model calls the built-in
`final_answer` tool — it is not enough for the model to just type its JSON
answer as a chat message; that step has no tool call in it and fails to
parse ("model output does not contain any JSON blob"), especially on
smaller local models. `SYSTEM_PROMPT` is written around this explicitly
("call top_features, then call final_answer with..."), and kept short —
small models follow short, concrete instructions more reliably than a
longer rules list.

## Two LLM modes: agent vs single_shot

```text
DETECTION_LLM_MODE=agent                        DETECTION_LLM_MODE=single_shot
  ToolCallingAgent decides which tools            code calls top_features (+ tree_vote_spread
  to call, then must call final_answer            if borderline) directly via MCP, then ONE
  — reliable for hosted models with real           litellm.completion call with
  tool_choice support (e.g. Gemini)                response_format=ANSWER_JSON_SCHEMA
                                                   — reliable for small local models
```

Either way, the raw text goes through the exact same `parse_and_validate`/
salvage pipeline in `LLMExplanationLayer._run_llm_layer` — `single_shot` is
not a way to skip any guardrail, it only changes how the raw JSON gets
produced. Set via `DETECTION_LLM_MODE=agent|single_shot` in `.env` to force
one explicitly. Left unset (or an unrecognized value):
`llm_layer.resolve_llm_mode` defaults to `single_shot` for any
`ollama_chat/*` `DETECTION_LLM_MODEL`, `agent` otherwise — since
`DETECTION_LLM_MODEL` itself defaults to a local Ollama model, the
out-of-the-box default is `single_shot`. `evaluate.py` records which mode
ran in the `llm_mode` CSV column and prints it alongside the model id at
startup.

**Recommendation:** for a hosted, capable model (Gemini), use
`DETECTION_LLM_MODE=agent` (or just don't set `DETECTION_LLM_MODEL` to an
`ollama_chat/*` id) — it actually gets to use the multi-step
tool-investigation loop as designed. For a local Ollama model (any size,
this is a litellm↔Ollama limitation, not a model-capability one),
`single_shot` is now the default and doesn't need setting explicitly — the
agent loop's reliance on `tool_choice` cannot work there. See
[design-decisions.md](design-decisions.md) for why the agent loop is
unreliable on Ollama in the first place.

## Category decision: a deterministic tool, not the LLM

`DetectionSubagent._choose_category` (called for every truly-anomalous
event, whether or not `use_llm` is set):

1. Calls `classifier.predict_attack_category(category_artifact,
   event.features, top_k=3)` — a multiclass RandomForest
   (`train_category.py`, trained on TRAIN-split anomalous rows only)
   returns per-category probabilities, ranked descending.
2. If the top class's probability `>= CATEGORY_CONFIDENCE_THRESHOLD`
   (default `0.6`, `DetectionSubagent(category_confidence_threshold=...)`),
   that class is the chosen category. Otherwise the chosen category is
   `"Unknown"` — reporting a low-confidence guess as if it were solid is
   worse than admitting the model isn't sure.
3. **If the top class is `classifier.BENIGN_CATEGORY` ("Benign")**, the
   binary model (which already called this event anomalous) and the
   category model disagree — the chosen category is `"Unknown"` (never
   `"Benign"`, which would contradict `is_anomalous=True`), logged as its
   own `action="model_disagreement"` trace step, and this overrides the
   confidence threshold entirely: even a *high-confidence* Benign vote is a
   disagreement, not a trustworthy answer. See
   [design-decisions.md](design-decisions.md) for why Benign is a class in
   the category model at all.
4. Otherwise: this is logged as its own trace step
   (`action="category_decision"`, `tool_input` carries `chosen_category`,
   `raw_top_category`, `raw_top_probability`, and `threshold`) so
   `evaluate.py` can report both the thresholded decision and the
   classifier's raw top-1 accuracy from the same run.
5. No category model loaded (backward compatibility) → logged as
   `category_model_unavailable`, category is always `"Unknown"`.

The LLM never sees this as something to decide: its task prompt states the
chosen category and its probability as a *given*, and its JSON schema
(`llm_notes.ANSWER_JSON_SCHEMA`) has no `category` property at all — only
`{explanation, tools_used}`. A stray `"category"` key in the LLM's raw JSON
(e.g. a stale cached prompt, or a model that doesn't strictly follow
schemas) is simply ignored by `parse_and_validate`, not validated or
rejected — there's no "bad category" outcome left, because the LLM's output
can no longer disagree with the decision. In `agent` mode the LLM may still
call `predict_attack_category` itself, purely to inform its own
explanation; whatever it sees there does not change the category code
already chose before the agent ran.

## Dependencies

Anything beyond the shared `requirements.txt` is in
`requirements-detection.txt`: `kagglehub` (dataset download), `joblib`
(artifact save/load), and, for the optional LLM layer, `mcp` (MCP server;
pinned `<2` — its FastMCP API was renamed in mcp 2.x), `smolagents[mcp]`
(MCPClient + ToolCallingAgent), `litellm` (model backend, via
`LiteLLMModel`), and `pydantic` (LLM output/tool schema validation).
