# LLM explanation layer

Guardrails, env vars, models/modes, fallback reasons, and troubleshooting for
the optional LLM layer. For the request flow diagram and the mechanics of
`agent` vs `single_shot`, see [architecture.md](architecture.md). For why the
layer is built this way, see [design-decisions.md](design-decisions.md).

## Enabling it

```python
from src.detection.manager import DetectionManager

with DetectionManager(use_llm=True) as manager:  # closes the MCP connection on exit
    manager.warmup()  # opens the connection + primes the model, outside any per-event timeout
    for event in events:
        result = manager.run(event)
```

`manager.close()` also works if you're not using it as a context manager
(e.g. it's kept alive for the app's lifetime and closed on shutdown). The
MCP connection opens on the first event that needs it (or on `warmup()`) and
is reused after that — don't create a new `DetectionManager` per event, or
you'll reload the model artifact every time.

Configure the model via a `.env` file at the repo root (copy `.env.example`):

```sh
cp .env.example .env
ollama pull qwen2.5:3b   # or whatever model DETECTION_LLM_MODEL names
ollama serve             # if not already running
```

## Models and modes

`DETECTION_LLM_MODEL` defaults to `ollama_chat/qwen2.5:3b` (local, no API
key). To use Gemini instead (as in the labs), set
`DETECTION_LLM_MODEL=gemini/<model-id>` (check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
for a current model id — these get retired) and
`DETECTION_LLM_API_KEY=<your key>`. Gemini's free tier is rate-limited
(requests/day and requests/minute caps) — `evaluate.py` retries a 429 with
backoff, but for a full run at default settings also pass `--llm-delay` (see
[evaluation.md](evaluation.md)).

### Config

| Setting | Source (highest priority first) |
|---|---|
| Per-event LLM timeout | `DetectionManager(llm_timeout_seconds=...)` constructor arg → `DETECTION_LLM_TIMEOUT` env var → 60s default |
| Circuit breaker threshold | `DetectionManager(circuit_breaker_threshold=...)` constructor arg → 3 (default) |
| Model id / API key | `DETECTION_LLM_MODEL` / `DETECTION_LLM_API_KEY` env vars (`.env`) — `evaluate.py` prints the resolved `DETECTION_LLM_MODEL` at startup; defaults to `ollama_chat/qwen2.5:3b` |
| LLM mode (`agent` vs `single_shot`) | `DETECTION_LLM_MODE` env var (`.env`) when set to a recognized value; otherwise defaults to `single_shot` for any `ollama_chat/*` model, `agent` otherwise (`llm_layer.resolve_llm_mode`) — see [architecture.md](architecture.md#two-llm-modes-agent-vs-single_shot) |
| Retry backoff schedule (rate limit / 5xx) | `llm_layer.LLM_RETRY_DELAYS_SECONDS` (2s, 4s, 8s — 3 retries, 4 attempts total); not currently exposed via env var |

## Guardrails

| Risk | Mitigation |
|---|---|
| LLM invents/edits a number | `is_anomalous`/`confidence` are computed by `classifier.py` calls in `subagent.py` before the LLM runs; the LLM's schema has no numeric fields at all |
| LLM invents/edits the attack category | `DetectionSubagent._choose_category` decides it deterministically from `classifier.predict_attack_category`, before the LLM runs; the LLM's schema has no `category` field, and a stray one in its raw JSON is ignored, not validated — see [architecture.md](architecture.md#category-decision-a-deterministic-tool-not-the-llm) |
| LLM hallucinates a feature/value | Grounding check: every feature name in `explanation` must appear in that event's `top_features` output, or the note is discarded (`llm_notes._is_grounded`) |
| LLM returns malformed/off-schema output | `llm_notes.parse_and_validate` checks JSON validity, exact keys, and explanation length; anything else falls back to a template note, tagged with a specific `fallback_reason` |
| Large ~78-feature dict inflates the prompt and slows a local CPU model | LLM-facing tools are `event_id`-based (`register_event`/`clear_event` are code-only, hidden from the agent); the task prompt carries only `event_id`, `p_anomalous`, and `tree_vote_std` |
| A tool the LLM doesn't need wastes a step on a small/slow model (e.g. re-deriving `p_anomalous`, which code already computed) | `predict_proba_anomalous` is never offered to the agent (`LLM_HIDDEN_TOOL_NAMES`); `tree_vote_spread` is offered only on a borderline event, since it's not meaningful otherwise |
| Slow local generation runs away | `max_tokens` capped (`MAX_LLM_OUTPUT_TOKENS`, default 300) via `LiteLLMModel` |
| `ToolCallingAgent` requires ending via the `final_answer` tool call — a small model that just types JSON as chat text fails to parse ("model output does not contain any JSON blob") | `SYSTEM_PROMPT` explicitly says "call top_features, then call final_answer with your JSON answer" (not "output ONLY a JSON object"), and is kept short |
| Even so, the model exhausts its steps without ever calling `final_answer` (smolagents then does one direct, un-tooled generation and returns that raw text) | Salvage path: if the raw text fails `parse_and_validate` as pure JSON, `llm_notes.extract_json_object` pulls out a `{...}` substring and the *same* guardrails re-run on that; success is tagged `fallback_reason="salvaged"` (distinct from a clean pass) so it's visible in reporting, not silently the same as a normal success |
| `tool_choice="required"` can't actually force a small local model through the agent loop — litellm's `ollama` transformation drops `tool_choice` entirely ("causes ollama requests to hang") | `DETECTION_LLM_MODE=single_shot`: code calls `top_features`/`tree_vote_spread` directly, one `litellm.completion` call with `response_format`'s `json_schema` (which Ollama *does* enforce, via grammar-constrained decoding) — same `parse_and_validate` pipeline either way; see [architecture.md](architecture.md#two-llm-modes-agent-vs-single_shot) |
| LLM/MCP server hangs | Runs on a daemon thread with a wall-clock timeout (`llm_timeout_seconds`, default 20s, configurable — see Config above); a timeout falls back to a template note |
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

## Fallback reasons

`fallback_reason` (an `evaluate.py` CSV column, and part of the guardrail
trace) is empty for a clean success or when the LLM was never invoked, or
one of: `timeout`, `exception`, `rate_limited`, `provider_unavailable`,
`invalid_json`, `ungrounded`, `circuit_open`, `salvaged`. See the guardrails
table above for what triggers each one.

## Diagnostics: seeing what the model actually wrote

Two extra trace actions (surfaced as `llm_step_failures` /
`llm_forced_final_answer` columns in `evaluate.py`'s CSV) exist purely to
debug the agent-mode failure described in
[design-decisions.md](design-decisions.md) without re-running anything:
`_log_agent_memory_diagnostics` inspects `agent.memory.steps` after
`agent.run()` returns and logs the raw text of every step that failed to
parse as a tool call (`llm_step_failed`), plus the forced, un-tooled final
answer's raw text if the model never called `final_answer` at all
(`llm_forced_final_answer`). These are diagnostic only — best-effort and
never fatal (wrapped so a mocked/non-standard `agent.memory` in tests can't
break the main flow) — and don't affect validation or the `fallback_reason`.

## Troubleshooting

- **`LLM warmup failed: <reason>`** — bad model id/API key, or the provider
  is down. `evaluate.py` prints the reason and continues with RF-only
  results; check `DETECTION_LLM_MODEL` / `DETECTION_LLM_API_KEY` and, for
  Ollama, that `ollama serve` is running and the model is pulled.
- **Every event falls back to a template note (`fallback_reason=timeout`)**
  — the per-event timeout (`DETECTION_LLM_TIMEOUT`, default 60s) may be too
  short for a cold local model; call `manager.warmup()` before the timed run,
  or raise the timeout.
- **`llm_circuit_open` on every event after a few failures** — the circuit
  breaker (default threshold 3) has tripped; look at the earlier events'
  `fallback_reason` for the root cause instead of the circuit-open ones.
- **Agent mode keeps failing to parse tool calls on a local Ollama model** —
  this is the known litellm/Ollama `tool_choice` limitation (see
  [design-decisions.md](design-decisions.md)); switch to
  `DETECTION_LLM_MODE=single_shot` (the default for `ollama_chat/*` models
  already) rather than tuning the prompt further.
