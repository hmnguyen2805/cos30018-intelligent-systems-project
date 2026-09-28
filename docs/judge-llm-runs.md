# Judge LLM agent: real-model test runs (Week 8)

Model: `ollama_chat/qwen2.5:3b` (local, Ollama). Command: `python scripts/judge_llm_demo.py --mode agent`.
Five hand-built cases, one run each per round. "Final" is the Judge's output after guardrails;
"LLM alone" is what the model itself decided before the guardrails.

## Results per run

| Case (expected) | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| agreed_brute_force (block) | ✅ | ✅ | ✅ | ✅ |
| category_mismatch (escalate) | ✅ | ✅ | ❌ finalized block | ✅ |
| low_mitigation_confidence (escalate) | ✅ guardrail | ✅ guardrail | ✅ guardrail | ✅ guardrail |
| mitigation_failed (escalate) | ✅ | ✅ | ✅ | ✅ |
| benign (no_action) | ❌ escalated | ❌ escalated | ✅ | ❌ escalated |
| **Final correct** | 4/5 | 4/5 | 4/5 | 4/5 |
| **LLM alone correct** | 3/5 | 3/5 | 3/5 | 3/5 |

Totals: 16/20 final, 12/20 LLM alone. The guardrail overrode the LLM 4/4 times on low mitigation
confidence (0.41 < 0.6). Every final error was either an unnecessary escalation (safe direction) or
a miss the rules baseline also makes; the Judge was never less safe than the rules baseline.

Cost per case: 2 to 5 LLM calls, about 1,000 to 4,000 tokens, 13 to 80 s of LLM time on local
hardware, $0.

## Failures, causes and fixes

| Run | Failure (from the trace) | Root cause | Fix | Regression test |
|---|---|---|---|---|
| 1 | benign escalated; the model's thought said "flagged as anomalous" although `detection_anomalous` was `false` | small model misreads booleans | `compare_conclusions` also returns `detection_verdict` ("benign"/"anomalous") and `*_confident` flags (`src/response/tools.py`) | `test_compare_conclusions_spells_out_verdict_and_confidence` |
| 1 | `lookup_playbook` called without a technique: error, then retry (wasted step) | model omitted the argument | defaults to the matched technique | `test_lookup_playbook_defaults_to_matched_technique` |
| 2 | benign escalated: "managers disagree", mitigation not confident | `mitigation_confident=false` (0.2) shown although nothing matched, and the prompt said to escalate on it; `proposed_action "none"` read as disagreement | when nothing matched: flag is `null`, `proposed_action` is `null`, plus a note; the category check returns `not_applicable` for benign traffic; prompt rules numbered in order | `test_irrelevant_mitigation_confidence_is_not_applicable_when_nothing_matched` |
| 3 | category_mismatch finalized with block: went compare -> finalize, skipping the consistency check | prompt instructions not reliably followed | **precondition enforced in code**: finalizing with an action requires `check_category_consistency` first (`JudgeAgent._check_preconditions` in `src/response/agent.py`) | `test_agent_must_check_consistency_before_acting`, `test_agent_may_finalize_no_action_without_consistency_check` |
| 4 | benign escalated: "Mitigation's confidence is low" | the model still used the raw 0.2 value despite the null flag and note | the value is hidden entirely when nothing matched | assertion added to `test_irrelevant_mitigation_confidence_is_not_applicable_when_nothing_matched` |

## Lessons

- With a 3B model, prompt changes fix one case and shift another, and results vary between runs.
  Checks that matter for safety belong in code (preconditions, guardrails), not only in the prompt.
- Give small models plain facts (words, flags) and hide irrelevant numbers.
- One run of five cases can't measure accuracy: the Week 11 evaluation needs 30+ cases, repeated
  runs, and a comparison against the `rules` and `single_shot` baselines (and possibly a 7B model).
