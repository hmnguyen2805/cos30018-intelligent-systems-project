# Architecture

How the multi-agent triage system fits together. Each part's own README has the
internal details; this page is the overview. Detection's internals are in
[src/detection/docs/architecture.md](../src/detection/docs/architecture.md).

## Agents and responsibilities

```text
                       ┌───────────────────────────────┐
  TrafficEvent ──────► │ Detection Manager             │  anomalous? confidence?
                       │   └ Detection Subagent        │  attack category + CICIDS2017 label
                       └──────────────┬────────────────┘  traffic summary
                                      │ DetectionResult
                       ┌──────────────▼────────────────┐
                       │ Mitigation Manager            │  ATT&CK technique(s)
                       │   └ Correlation Subagent      │  proposed response action
                       └──────────────┬────────────────┘
                                      │ MitigationRecommendation (or an error)
                       ┌──────────────▼────────────────┐
                       │ Judge (LLM agent)             │  compares both conclusions:
                       │   tools: compare_conclusions, │  act automatically or
                       │   lookup_playbook,            │  escalate to a human
                       │   check_category_consistency, │
                       │   request_recheck ────────────┼──► back to a manager, once each
                       └──────────────┬────────────────┘
                                      │ ResponseRecommendation
                                      ▼
                                 PipelineRun  ──► UI (src/ui/app.py)
```

| Agent | Decides | How |
|---|---|---|
| Detection Manager + Subagent | Is the traffic anomalous; which attack category and fine label | RandomForest binary model (verdict and confidence), category model (label and category), deterministic traffic summary; optional LLM explanation over an MCP tool server |
| Mitigation Manager + Correlation Subagent | Which ATT&CK technique fits; what response to propose | Category lookup in the shared playbook; embedding search of the technique catalog when the category is Unknown |
| Judge | Whether to apply the proposed response automatically or escalate | `agent` mode: LLM reasoning-action loop with tools, rechecks, guardrails and a rule-table fallback. `rules` and `single_shot` are the baselines |

## Coordination

- **Shared contract.** All messages between agents are the dataclasses in
  `src/shared/schemas.py`: `TrafficEvent`, `DetectionResult`,
  `MitigationRecommendation`, `JudgeInput`, `ResponseRecommendation`, `PipelineRun`.
  Every agent extends `BaseAgent` and logs each step (`thought`, `action`,
  `tool_input`, `observation`), which is what the UI shows.
- **Detection tags.** Detection also writes `[category=X] [label=Y] <summary>` at the
  start of `detector_notes` and fills `attack_category`, `attack_label`,
  `category_confidence` and `traffic_summary`. `src/shared/tags.py` parses the tags.
- **Two independent opinions.** Detection reports CICIDS2017-based categories;
  Mitigation maps to ATT&CK techniques. The Judge checks they're consistent using
  `category_techniques` in `src/response/playbook.json`, the same table Mitigation
  reads, so both work from one mapping.
- **Rechecks (two-way).** The pipeline gives the Judge a callback. The agent-mode
  Judge can ask a manager to look again with a reason, once per manager per case:
  - `detection`: re-runs Detection with `recheck_reason` (more evidence, same
    verdict and confidence), then re-runs Mitigation on the new result.
  - `mitigation`: re-runs Mitigation with `recheck_reason`.

  The Judge then continues with the updated conclusions.

## Failure handling

| What fails | What happens |
|---|---|
| Detection | Run stops: nothing to judge. Error recorded, UI shows a failed run |
| Mitigation | Judge still decides with `mitigation=None` and the error; escalates if Detection flagged a threat |
| A manager during a recheck | Error recorded; the Judge keeps the earlier conclusions |
| Judge LLM (error, timeout, invalid replies, step limit) | Rule table decides (`decided_by = "rules_fallback"`), reason in the trace |
| Judge itself | Error recorded, no final response |

Every stage's time is in `PipelineRun.timings_ms`; LLM calls, tokens, latency,
iterations and rechecks are in `ResponseRecommendation.llm_usage`.

## Safety

- The Judge's LLM may only be more cautious than the rule table, never less, and may
  only choose the manager's proposed action or `no_action`.
- Preconditions are enforced in code: look at the evidence before finalizing, check
  category consistency before finalizing with an action (again after a recheck).
- No generated code is executed. Tools only read the case and local JSON files
  (sandboxing, criteria 5E).

## Where to look

| Topic | File |
|---|---|
| Contracts | `src/shared/schemas.py` |
| Pipeline, rechecks, failure handling | `src/pipeline.py` |
| Judge loop, guardrails, rules | `src/response/` ([README](../src/response/README.md)) |
| Mitigation | `src/correlation/` ([README](../src/correlation/README.md)) |
| Detection | `src/detection/` ([README](../src/detection/README.md)) |
| UI | `src/ui/app.py`, `src/ui/scenarios.py` |
| Judge runs on a real model | [docs/judge-llm-runs.md](judge-llm-runs.md) |
