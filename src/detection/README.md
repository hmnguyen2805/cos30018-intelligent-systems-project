# Detection Agent

**Owner:** Vinh Nghiem

Takes a `TrafficEvent` (raw traffic features) and decides whether it's anomalous.

## Your contract
run(event: TrafficEvent) -> DetectionResult


That's the only thing the rest of the pipeline cares about. Structure the internals
however works for you — this folder is yours.

## Important: this needs to be a real agent, not just a classifier call

Per the assignment's grading criteria, a single function call (even a well-built
scikit-learn classifier) does **not** count as an agent — it needs a reasoning loop:
decide what to check, take an action, observe the result, and possibly revise before
answering. So don't just return whatever the classifier says on the first pass.

A workable pattern: treat the classifier as a **tool** your agent calls, not the agent
itself. If the classifier returns a borderline confidence score, have the agent decide
to re-examine (e.g. classify a wider time window, or re-run with adjusted features)
before finalizing — that borderline-handling is what makes it a loop instead of one call.

Call `self.log_step(thought=..., action=..., observation=...)` (inherited from
`BaseAgent`) each time you reason or call the classifier — that's what lets the
pipeline/UI show what your agent actually did, and it's required for the "observable
reasoning-action iterations" part of the rubric.

## Suggested next steps

1. Pick one dataset to start with (CICIDS2017 is the most common starting point).
2. Get a baseline classifier trained and evaluated outside the agent class first
   (a notebook or standalone script is fine) before wiring it in as a tool.
3. Add the reasoning loop around it — even a simple one (check confidence, re-check
   once if borderline, then decide) satisfies the requirement.
4. `tests/test_pipeline_smoke.py` should keep passing throughout — it only checks the
   contract, not your model's accuracy.

## Dependencies

Add anything beyond the shared `requirements.txt` to a `requirements-detection.txt`
in this folder, and note it here.