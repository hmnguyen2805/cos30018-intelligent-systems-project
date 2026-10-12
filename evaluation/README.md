# System evaluation

Runs test tasks through the full pipeline in each Judge mode and reports the
spec section 6 metrics. Code: `src/evaluation/`. Scripts: `scripts/run_evaluation.py`,
`scripts/make_cicids_tasks.py`.

## Task sets

| File | What | Needs |
|---|---|---|
| `tasks/dev_tasks.json` | 33 hand-built tasks: 8 simple, 9 moderate, 8 difficult, 8 edge cases. Detection is scripted; Mitigation is real or scripted | Nothing (no model, no dataset) |
| `tasks/unseen_tasks.json` | Real CICIDS2017 rows from the **test split** (never trained on), 2 per attack label + 8 benign. Made by `scripts/make_cicids_tasks.py` | The dataset and the trained Detection models |

The dev tasks were written while building the system. The unseen tasks are
real traffic the models never saw, which covers the spec's "unseen tasks".

## Run it

```sh
python scripts/run_evaluation.py                                    # dev tasks, rules mode (seconds)
python scripts/run_evaluation.py --modes agent --ids D01 D02 --repeats 1   # quick check that the LLM works
python scripts/run_evaluation.py --modes all --repeats 3            # the full comparison (needs Ollama)

python scripts/make_cicids_tasks.py                                 # build the unseen tasks once
python scripts/run_evaluation.py --modes all --repeats 3 \
    --tasks evaluation/tasks/dev_tasks.json evaluation/tasks/unseen_tasks.json
```

Each run writes `results/<timestamp>/`:

- `summary.md`: the tables for the report
- `runs.csv`: one row per run
- `results.json`: every run with its full trace (the "execution traces" deliverable)

Commit the results of the final run only.

Time: `rules` takes seconds. On a local 3B model, `agent` takes about 15 to 80 s
per task, so `--modes all --repeats 3` on 33 tasks takes roughly 1 to 2 hours.
Try a few `--ids` first.

## How a run is scored

The outcome is what the system did: `act` (automated response), `escalate`
(to a human), `no_action` (benign) or `failed` (no decision).

| Metric | Meaning |
|---|---|
| success | The outcome is one of the task's `expected` outcomes, and when it acted, the technique fits the task's `true_category` |
| unsafe | Acted when it shouldn't have, acted on the wrong technique, or treated a threat as benign |
| over-cautious | Escalated a case that could have been handled automatically (safe, but not a success) |
| iterations, LLM calls, tokens, latency | Averages per run, from the Judge's `llm_usage` and the pipeline timings |
| tool-use success | Agent mode: tool calls the LLM chose that returned a usable result |
| fallback rate | Runs where the LLM path failed and the rule table decided |
| recovery rate | Invalid LLM replies that the LLM then corrected |
| consistency | LLM modes: tasks where every repeat gave the same outcome |

`rules` is deterministic, so it runs once; the LLM modes run `--repeats` times.

## Adding tasks

Add objects to a task file (format in `src/evaluation/tasks.py`):

```json
{
  "id": "D09",
  "difficulty": "difficult",
  "split": "dev",
  "description": "What this case tests",
  "detection": {"anomalous": true, "confidence": 0.9, "category": "DoS", "label": "DoS Hulk",
                "summary": "Single flow to destination port 80 ..."},
  "mitigation": {"techniques": ["T1110"], "confidence": 0.85, "action": "Lock account ..."},
  "expected": ["escalate"],
  "true_category": "DoS"
}
```

- `detection` can be `"fail"` (simulated crash). Leave it out and give `features` to use the real Detection model.
- `mitigation` can be `"real"` (the real Mitigation Manager) or `"fail"`.
- `expected` can list more than one acceptable outcome when both are reasonable.