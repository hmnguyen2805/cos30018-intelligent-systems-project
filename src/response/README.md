# Judge Agent

**Owner:** Minh Nguyen

Arbitrates between the two managers' conclusions (Detection Manager and
Mitigation Manager) and produces the pipeline's final `ResponseRecommendation`.
Also owns `src/pipeline.py`, which runs the whole system.

## Contract

```text
JudgeAgent.run(JudgeInput) -> ResponseRecommendation
Pipeline.run(TrafficEvent) -> PipelineRun
```

`JudgeInput` carries the `DetectionResult` and the `MitigationRecommendation`
(or `None` plus `mitigation_error` when the Mitigation Manager failed or isn't
available). All types are in `src/shared/schemas.py`. `ResponseRecommendation`
also records `decided_by` (`rules`, `llm`, `guardrail_override` or
`rules_fallback`) and `llm_usage` (LLM calls, iterations, tokens, latency) for
the evaluation.

## Structure

```text
src/response/
├── agent.py         # JudgeAgent: three modes (below), every step logged via log_step
├── llm.py           # model config, litellm client, JSON schemas, prompts, reply parsing
├── tools.py         # the Judge's tools: compare_conclusions, lookup_playbook, check_category_consistency
├── playbook.json    # guidance per ATT&CK technique + which techniques fit each Detection category
├── guardrails.py    # the LLM may only be more cautious than the rules
├── rules.py         # decision table (pure functions): baseline, fallback and guardrail
└── README.md
src/pipeline.py              # Detection -> Mitigation -> Judge, failure handling, timings
scripts/judge_llm_demo.py    # run the Judge on example cases against a real model
tests/fakes.py               # FakeMitigationManager until the real one lands
```

## Modes

The same Judge runs in three modes with the same input and output, so they can
be compared directly (the assignment requires a simpler baseline):

| Mode | How it decides | Role in evaluation |
|---|---|---|
| `rules` | decision table only, no LLM | baseline: fixed workflow |
| `single_shot` | code gathers the evidence with the tools, then one LLM call decides | baseline: single LLM call |
| `agent` | iterative loop: each turn the LLM picks the next action (a tool, `finalize` or `escalate`), code runs it and feeds the result back as an observation, until it decides or reaches the step limit | the system |

Why a JSON-per-turn loop instead of smolagents' `ToolCallingAgent`: on local
Ollama models litellm drops `tool_choice`, so native tool calling is
unreliable, while a JSON-schema `response_format` is enforced (documented by
Vinh in `src/detection/docs/design-decisions.md`). Each reply is one JSON
object `{"thought", "action", ...}` constrained to the allowed actions.

What the LLM adds over the rules: it can read text the rules can't. Example:
Detection tags `[category=DoS]` but Mitigation matched T1110 (brute force), both
confidently. The rules see an agreed threat and would act; the agent calls
`check_category_consistency`, sees the mismatch, and escalates.

## Guardrails and fallback

The rule decision is always computed alongside the LLM:

- The LLM may escalate a case the rules would finalize, never the reverse.
- A finalize must use an allowed action (the manager's proposal or `no_action`);
  the Judge can't invent actions.
- Finalizing before looking at the evidence (`compare_conclusions`) is rejected, and so is
  finalizing with a response action before `check_category_consistency`. These preconditions
  are enforced in code because a 3B model skipped the consistency check in testing.
- One invalid reply gets a correction message; two in a row, an LLM error or
  timeout, or hitting the step limit fall back to the rules. The trace records why.

The tools only read the case and `playbook.json`; no generated code is ever
executed. That is the sandboxing answer for criteria 5E.

## Decision cases (rules.py, first match wins)

| Case | When | Result |
|---|---|---|
| LOW_DETECTION_CONFIDENCE | detection confidence < 0.6 | escalate |
| MITIGATION_MISSING | mitigation failed / unavailable | escalate if anomalous, else no action (incomplete) |
| BOTH_BENIGN | benign, no technique | no action |
| BENIGN_BUT_ACTION | benign, technique matched | escalate (conflict) |
| ANOMALOUS_NO_MATCH | anomalous, no technique | escalate (possible unknown attack) |
| LOW_MITIGATION_CONFIDENCE | anomalous + matched, mitigation confidence < 0.6 | escalate |
| AGREED_THREAT | anomalous + matched + confident | apply proposed action |

## Configuration

All optional environment variables (a `.env` file works too):

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_MODE` | `rules` | `rules`, `single_shot` or `agent` (used by `build_default_pipeline()`) |
| `JUDGE_LLM_MODEL` | `ollama_chat/qwen2.5:3b` | any litellm model id, e.g. `gemini/gemini-2.0-flash` |
| `JUDGE_LLM_API_KEY` | none | only for hosted models |
| `JUDGE_LLM_TIMEOUT` | `60` | seconds per LLM call |
| `JUDGE_LLM_MAX_STEPS` | `6` | agent-loop turns before falling back |
| `JUDGE_LLM_MAX_TOKENS` | `400` | output tokens per call |

## Running it

Tests need no model or LLM (the LLM is replaced by a scripted fake):

```sh
python -m pytest tests/ -q
```

Against a real local model:

```sh
ollama pull qwen2.5:3b
python scripts/judge_llm_demo.py --mode agent --trace
python scripts/judge_llm_demo.py --save judge_demo.json     # all modes, results as JSON
```

## Status and next steps

- [x] Week 7: contract, rule-based decision table, pipeline with failure handling and timings, tests.
- [x] Week 8: LLM Judge agent loop, single-shot baseline, tools, playbook, guardrails, fallback, tests, demo script.
- [ ] Week 9: `request_recheck` feedback path to the managers (needs small API additions from Detection and Mitigation).
- [ ] Week 10: swap `FakeMitigationManager` for the real Mitigation Manager in `build_default_pipeline()`.
- [ ] Week 11: evaluation of the three modes on at least 30 test cases.
