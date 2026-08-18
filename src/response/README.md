# Response Agent

**Owner:** Minh Nguyen

Takes a `CorrelationResult` and produces the final `ResponseRecommendation` —
including disagreement handling between Detection and Correlation's confidence levels.

## Your contract
run(correlation: CorrelationResult) -> ResponseRecommendation


## Suggested next steps

1. Get the disagreement-threshold logic solid first (already stubbed) — decide
   automated recommendation vs. escalate to human, based on both agents' confidence.
2. Layer in real response reasoning (smolagents + an LLM) on top of the deterministic
   disagreement path once that's tested.
3. Log each reasoning/tool step via `self.log_step(...)` same as the other agents.
4. `pipeline.py` (also yours) wires all three agents together — keep it thin, it should
   just call each `run()` in sequence. Consider adding one feedback path (e.g. sending
   a disagreement case back to Correlation for re-analysis before escalating) — that
   strengthens the "coordination mechanism" requirement beyond a pure one-way chain.