# Design decisions

Chronological log of decisions that shaped the current design: the problem
that prompted each one, the evidence behind it, the decision made, and its
trade-off. For current metrics referenced as evidence below, see
[evaluation.md](evaluation.md); for the resulting mechanics, see
[architecture.md](architecture.md).

## Manager/Subagent two-layer split

**Problem:** Detection needed a stable top-level contract that other
managers (Mitigation, the Judge) could call, decoupled from how
classification actually happens.

**Decision:** Per the tutor's manager-subagent recommendation, Detection is
two layers: a **Detection Manager** that owns the top-level contract
(`DetectionManager.run(event) -> DetectionResult`), and a **Detection
Subagent** underneath it that does the actual classification work.

**Trade-off:** An extra layer of indirection for a system that currently
only has one subagent — justified because it leaves room for the Manager to
later add oversight (e.g. dispatching to a second detection subagent)
without changing the public contract.

## Borderline re-examination via tree vote spread

**Problem:** A single `p_anomalous` score from the baseline RandomForest can
be too coarse to trust blindly right at the decision boundary.

**Decision:** On a borderline score (0.4-0.6), `DetectionSubagent` doesn't
just trust that number — it re-examines via per-tree vote spread (a second,
finer-grained tool call) before finalizing. High tree disagreement on a
borderline call gets flagged in `detector_notes` for downstream
correlation/response to weigh accordingly.

**Trade-off:** One extra tool call, but only on borderline events, so the
cost is bounded to the cases where it actually adds information.

## Feature-name mismatch guard

**Problem:** The binary and category models are trained and saved
separately (`detection_binary.joblib`, `detection_category.joblib`). If one
is retrained after a feature-engineering change and the other isn't, their
feature columns can silently misalign.

**Decision:** Both artifacts store `feature_names`;
`classifier.assert_feature_names_match` checks they're identical (same
columns, same order) whenever both are loaded together — `DetectionSubagent`
calls this at construction time and raises immediately on a mismatch, rather
than silently misaligning feature columns later.

**Trade-off:** The category model stays optional for backward compatibility
— an install with only `detection_binary.joblib` (or an artifact predating
`train_category.py`) still works, with `DetectionSubagent` just reporting
every category as `"Unknown"`.

## Migration from the single-artifact layout

**Problem:** An earlier version saved one combined `models/detection_rf.joblib`
covering both binary and category prediction.

**Decision:** That layout is no longer read. If `classifier.load_artifact`
can't find the new two-artifact path but finds the old one, it raises a
`FileNotFoundError` naming both paths and telling the caller to run
`python -m src.detection.train` — not a generic "file not found."

**Trade-off:** Anyone with an old artifact must retrain once; the explicit
error message is meant to make that obvious immediately instead of
surfacing as a confusing downstream failure.

## Adding a Benign class to the category model

**Problem:** An earlier version of the category model trained on attack rows
only, so it had no way to express "this doesn't look like an attack."

**Evidence:** A real evaluation run found that every one of the binary
model's false positives (genuinely benign traffic wrongly flagged
anomalous) still got a confident attack category from the category model,
because that was the only kind of answer the model could give.

**Decision:** `train_category.py` now also builds an explicit
`classifier.BENIGN_CATEGORY` ("Benign") class from BENIGN rows, downsampled
to `DEFAULT_BENIGN_TO_ATTACK_RATIO` (2.0) times the attack-row count so
training stays fast. A top vote for this class is never reported as
`"Benign"` (which would contradict `is_anomalous=True`) — it's reported as
`"Unknown"` and logged as `action="model_disagreement"` (see
[architecture.md](architecture.md#category-decision-a-deterministic-tool-not-the-llm)).

**Trade-off:** This reduces, but does not eliminate, false positives getting
a confident specific category instead of `"Unknown"` — see
[evaluation.md](evaluation.md) for the current residual rate and why it
isn't obviously a code bug.

## Category decision: from LLM output to deterministic classifier call

**Problem:** The attack category used to be an LLM output, gated by prompt
instructions and a fixed-list/synonym check (`llm_notes._normalize_category`,
since removed).

**Evidence:** A real evaluation run found the LLM-picked category scored 0%
accuracy on a small sample (n=8) of truly-anomalous events with a category —
against a 50% "always guess the most common true category" baseline on that
same set. Correct-but-non-committal ("Unknown") beat confidently-wrong, but
the LLM was never actually *choosing well*, and n=8 is far too small a
sample to justify tuning the prompt further — that risks overfitting to
those 8 events rather than fixing anything general.

**Decision:** Same principle as `is_anomalous`/`confidence` — stop asking
the LLM to decide, and make category a deterministic classifier call
(`DetectionSubagent._choose_category`, see
[architecture.md](architecture.md#category-decision-a-deterministic-tool-not-the-llm)).
The LLM's JSON schema no longer has a `category` field at all; a stray one
in its raw output is ignored, not validated.

**Trade-off:** The LLM can no longer contribute category judgment even in
principle (in `agent` mode it may still call `predict_attack_category` for
its own investigation, but that never changes the outcome) — accepted
because the classifier's own current accuracy (see
[evaluation.md](evaluation.md)) is far higher than the LLM's ever was.

## Two LLM modes: agent vs single_shot

**Problem:** The agent loop (`ToolCallingAgent`, `DETECTION_LLM_MODE=agent`)
needs every step to be an actual tool call. The obvious way to force that —
`tool_choice="required"` — doesn't work for local models behind Ollama.

**Evidence:** Verified against `ollama_chat/qwen2.5:3b`: litellm's own
`ollama` transformation strips `tool_choice` from every request (its code
comment says why: "causes ollama requests to hang"). So nothing can force
the agent loop's "every step is a tool call" requirement onto this backend —
the model sometimes just answers in prose mid-loop, hits
`AgentParsingError`, and after `max_steps` smolagents falls back to one
direct, un-tooled generation. Separately, `response_format={"type":
"json_schema", ...}` *is* honored: litellm forwards it to Ollama as its
`format` parameter, and Ollama enforces it with grammar-constrained decoding
— verified directly against `qwen2.5:3b`, the `category` field (before it
was removed from the schema — see above) came back reliably constrained to
the exact enum, with no prompting trick needed.

**Decision:** Add `DETECTION_LLM_MODE=single_shot`: code calls the needed
tools directly via MCP, then makes one `litellm.completion` call with
`response_format=ANSWER_JSON_SCHEMA`, instead of running the agent loop. See
[architecture.md](architecture.md#two-llm-modes-agent-vs-single_shot) for
the mechanics of both modes and the current default-selection rule.

**Trade-off:** `single_shot` skips the LLM's own multi-step tool
investigation (it never decides *which* tools to call) in exchange for
actually working reliably on a local Ollama model. For a hosted, capable
model (Gemini) with real `tool_choice` support, `agent` mode is still
recommended since it gets to use the investigation loop as designed.
