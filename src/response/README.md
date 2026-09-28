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
available). All three types are in `src/shared/schemas.py`.

## Structure

```text
src/response/
├── agent.py    # JudgeAgent: compare -> decide -> finalize/escalate, logged via log_step
├── rules.py    # decision table (pure functions); becomes the LLM fallback in Week 8
└── README.md
src/pipeline.py # Detection -> Mitigation -> Judge, failure handling, timings
tests/fakes.py  # FakeMitigationManager until the real one lands
```

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

## Status and next steps

- [x] Week 7: contract, rule-based decision table, pipeline with failure handling and timings, tests.
- [ ] Week 8: smolagents `ToolCallingAgent` + local Ollama model choosing its own tool calls
  (`compare_conclusions`, `lookup_playbook`, `escalate_to_human`, `finalize`); rules become the fallback.
- [ ] Week 9: `request_recheck` feedback path to the managers (needs small API additions from Detection and Mitigation).
- [ ] Week 10: swap `FakeMitigationManager` for the real Mitigation Manager in `build_default_pipeline()`.

## Running it

`build_default_pipeline()` uses the real Detection Manager (needs a trained
model, see `src/detection/README.md`) and no Mitigation Manager yet, so every
run currently lands in `MITIGATION_MISSING`. Tests don't need a model or an LLM:

```sh
python -m pytest tests/ -q
```
