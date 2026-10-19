# Evaluation and Fresh-Clone Install Check: Results

Vinh Nghiem, 13 Oct 2026. Machine: Windows, Python 3.14.7, local Ollama `qwen2.5:3b` (CPU).

## 1. Unseen tasks

- Built with `python scripts/make_cicids_tasks.py`: 36 tasks (2 rows per attack label x 14 labels + 8 benign) from the shared test split.
- Leakage check: the script loads data through `src/detection/training/data.py` (`load_clean_dataframe`, `split_train_test`), the same split and fixed seed the Detection models were trained with, so no task row was seen in training.
- Reproducible: regenerating on a fresh clone with an empty kagglehub cache produced a byte-for-byte identical `unseen_tasks.json` (same SHA-256).

## 2. Full evaluation run

Command: `python scripts/run_evaluation.py --modes all --repeats 3 --tasks evaluation/tasks/dev_tasks.json evaluation/tasks/unseen_tasks.json`
Results folder: `evaluation/results/20261013-023242` (69 tasks: 33 dev + 36 unseen; LLM modes repeated 3 times).

| Metric | rules | single_shot | agent |
|---|---|---|---|
| Success | 60/69 (87%) | 195/207 (94%) | 189/207 (91%) |
| Unsafe | 7/69 (10%) | 6/207 (3%) | 9/207 (4%) |
| Over-cautious | 2/69 (3%) | 6/207 (3%) | 9/207 (4%) |
| Avg LLM calls | 0 | 1.0 | 4.2 |
| Avg tokens | 0 | 743 | 4,471 |
| Avg total latency | 0.26 s | 8.57 s | 35.95 s |
| Fallback rate | 0% | 0% | 6% (12/207) |
| Rechecks | 0 | 0 | 57 |
| Consistency across repeats | n/a | 69/69 | 69/69 |

| Split | rules | single_shot | agent |
|---|---|---|---|
| Dev (33) | 27/33 (82%) | 96/99 (97%) | 93/99 (94%) |
| Unseen (36) | 33/36 (92%) | 99/108 (92%) | 96/108 (89%) |

Tasks where the modes differ: D01 to D04 (rules 0%, both LLM modes 100%), E04 (single_shot 100%, rules and agent 0%), U-SSH-Patator-2 (agent 0%, others 100%).

Notes:
- An earlier full run (`20261012-235015`) is invalid: `qwen2.5:3b` was not installed, so 99% of LLM runs fell back to the rule table (`NotFoundError`). It should not be used or committed.
- U-SSH-Patator-2 in agent mode is a Judge-side issue: Detection (BruteForce / SSH-Patator) and Mitigation (T1110) were both correct, but the Judge LLM replied "finalize" without an allowed final action, the reply was rejected, and it escalated.

## 3. Detection on the unseen tasks

Detection does not use an LLM in this pipeline, so its output was identical in every mode.

- All 8 benign rows: no action.
- 12 of 14 attack labels: acted, with a technique matching the true category, on both rows (including both Web Attack XSS rows).
- The 3 misses, all on the three rarest classes:

| Task | Detection behaviour | Outcome |
|---|---|---|
| U-WebAttack-SqlInjection-1 | Binary model p_anomalous = 0.17, called benign | Unsafe (in every mode) |
| U-Heartbleed-1 | Anomalous (p = 0.61), but category model top vote Benign (0.70): `model_disagreement`, reported Unknown | Escalated (over-cautious, safe) |
| U-Infiltration-2 | Top label Infiltration 0.49 vs Benign 0.48, below the 0.80 threshold: reported Unknown | Escalated (over-cautious, safe) |

- Recheck API used in practice: in agent mode the Judge rechecked Detection 33 times. On Heartbleed-1 and Infiltration-2 the recheck added evidence but did not change the verdict, as designed.

## 4. Fresh-clone install check

Followed only the root README steps in a new clone (`fresh-check`) with a new virtual environment.

| Step | Result |
|---|---|
| Clone + venv (Python 3.14.7) | OK |
| `pip install` (all three requirements files) | OK |
| `python -m pytest tests/ -q` | 448 tests, all passing |
| `python scripts/run_evaluation.py` (dev, rules) | 27/33, identical to the committed results |
| UI (`python -m src.ui.app`), demo scenario | OK |
| LLM check (`--modes agent --ids D01 --repeats 1`) | LLM reached (6 calls, ~6,800 tokens); this run hit the step limit and fell back, so D01 failed (it passed in 3 calls on the main copy) |
| `python -m src.detection.train` | OK; binary ROC-AUC 0.9998, fine-label macro F1 0.8770 (same as main copy) |
| `python scripts/make_cicids_tasks.py` with an empty kagglehub cache | OK; dataset downloaded with no Kaggle account; output identical to main copy |
| Unseen tasks, rules mode | 33/36, all 36 outcomes identical to the main copy |

Latency note: median task time on the unseen run was 0.04 s, but the slowest was 10 s (82 s on the very first run on this PC). This is the Correlation text-search model loading on first use.

## 5. Suggested fixes

1. README: root `requirements.txt` already includes Detection's packages, so the separate `requirements-detection.txt` step is redundant.
2. `requirements.txt`: `litellm` is listed twice; comments still mention the old `mcp_server.py` name.
3. README: state that the CICIDS2017 download needs no Kaggle account, and give a rough time for download and training.
4. `evaluation/results/`: two old dev-run folders (`20261012-230119`, `20261012-230935`) are committed; remove before final submission.
5. Correlation: load the text-search model before timed runs so latency numbers are not skewed by the first-load delay.
6. Judge: single runs of the 3B model vary (D01 agent passed on one machine, fell back on another); U-SSH-Patator-2 fails in agent mode due to final-action matching.
7. Detection (mine): training output heading still says "anomalous rows only"; it now trains on 15 fine labels including Benign.
