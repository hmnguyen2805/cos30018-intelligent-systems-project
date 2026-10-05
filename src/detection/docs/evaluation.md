# Evaluation

How to run both evaluation modes, the current result tables, and known
limitations. All current metric numbers for the Detection module live here —
other docs reference them without restating the figures. For why the
category decision and the false-positive rate look the way they do, see
[design-decisions.md](design-decisions.md).

## Running the sampled evaluation

```sh
python -m src.detection.evaluate --sample-size 200 --sampling random --llm-timeout 20 --llm-delay 0
```

Prints the resolved `DETECTION_LLM_MODEL`, then runs `DetectionManager` with
`use_llm=False` and `use_llm=True` over the same held-out sample. Warms up
the LLM connection first (reported separately from per-event latency) — if
warmup fails after retries (bad model id/API key, provider down), it prints
`LLM warmup failed: <reason>. Check DETECTION_LLM_MODEL / API key / provider
status.` and skips the `+LLM` arm entirely, still printing/saving the
RF-only results. Otherwise it prints an accuracy/F1/latency comparison table
plus fallback-reason counts, and writes per-event results to
`src/detection/results/rf_only_<sampling>.csv` / `rf_llm_<sampling>.csv`
(gitignored) — including each event's `fallback_reason` (`timeout` /
`exception` / `rate_limited` / `provider_unavailable` / `invalid_json` /
`ungrounded` / `circuit_open` / `salvaged` / empty for success or "never
invoked") and its `prompt_tokens`/`completion_tokens` (from the LLM
provider's own usage reporting; empty when the LLM wasn't invoked or never
returned a response). Accuracy/F1 should be identical between the two arms —
that's the guardrail proof that the LLM never changes the decision. Whether
an event was actually dispatched to the LLM (`llm_invoked` / the `% events
-> LLM` line) is counted from an unbuffered `llm_dispatch` step logged
before the timed call, not the buffered `llm_layer_start` — otherwise a
timed-out event would be undercounted as "never invoked" even though it
clearly was.

`--sampling random` (default) draws a plain stratified-by-label sample,
representative of the real class distribution. `--sampling borderline`
instead biases the sample toward events near the decision boundary, so it
actually exercises the LLM trigger condition — useful when you specifically
want to stress-test the LLM path, at the cost of the sample no longer
reflecting real-world class balance.

`--llm-delay <seconds>` sleeps after every event that dispatched to the LLM
(not after RF-only events) — use this to stay under a hosted provider's
free-tier rate limit over a full run.

### Category accuracy

Each sampled event's original CICIDS2017 `Label` (multiclass — e.g. "DoS
Hulk", "Web Attack – XSS" — normally discarded by `train_binary.py`'s
BENIGN-vs-anomalous binarization) is kept alongside it and mapped via
`data.map_cicids_label_to_category` to the coarse category vocabulary
(`Heartbleed` is its own category; `BENIGN` and anything unrecognized map to
`None`). It's written to each CSV row as `true_category`. This online path
scores the **coarse** category only (parsed from the `[category=X]` tag); the
fine-label tables live in `--offline` below.

Since the category decision is a deterministic classifier call (see
[design-decisions.md](design-decisions.md)) — not an LLM output —
`compute_category_report` scores it over **every truly anomalous event in
the sample with a known `true_category`**, regardless of `use_llm` or
whether the LLM ran/succeeded on that event. `print_category_report` (in the
console summary) reports:

- **category accuracy** — exact match rate of the code-chosen (thresholded)
  category against `true_category`
- **raw accuracy** — exact match rate of the classifier's raw top-1 class
  (before the `CATEGORY_CONFIDENCE_THRESHOLD` gate), so a low threshold's
  cost (in punted-to-Unknown events) is visible against what the classifier
  could do unthresholded
- **% predicted Unknown** — how often the thresholded decision punted
  (either no category model, or the top class didn't clear the confidence
  threshold)
- **confusion table** — true category (rows) vs. code-chosen category
  (columns)
- **trivial baseline** — accuracy of always guessing whichever true category
  was most common in that same scored set, so a beaten baseline actually
  means something

### False-positive categorisation and model disagreement

`compute_category_report` only ever looks at *truly anomalous* events
(`true_label == 1`), so it can't see what happens on the binary model's
false positives — genuinely benign traffic it wrongly called anomalous.
`compute_false_positive_categorization` covers exactly that gap: among
events with `true_label == 0` that the binary model still flagged
anomalous, what percentage got a specific attack category rather than
`"Unknown"`. A real full-test-split run (see `--offline` below) shows
**59.3%** (227/383 false positives) at the default threshold 0.80 (48.6%,
186/383, at 0.90 — a lower threshold trusts more borderline calls, on false
positives too). `print_summary` prints it right after
the category report, and `evaluate.py`'s CSV carries each row's
`model_disagreement` flag (True when `_choose_category`'s top vote was
Benign for that event) — the console summary also prints the total
`model_disagreement_count` for the run. The residual is not obviously a
code bug: a binary false positive is, by definition, a feature vector the
binary model itself found attack-shaped, so it isn't surprising the
(separately-trained, similar-feature) category model often finds it
resembles one specific attack pattern rather than voting Benign — tuning
`benign_to_attack_ratio` or `CATEGORY_CONFIDENCE_THRESHOLD` further (see the
threshold sweep below) is the next lever, not something this evaluation
script can decide on its own.

## Offline evaluation: full-split batch metrics + threshold sweep

```sh
python -m src.detection.evaluation.evaluate --offline [--category-threshold 0.8] [--min-category-accuracy 0.99]
```

A different mode from everything above: no `DetectionManager`/
`DetectionSubagent`, no per-event agent loop, no LLM at all —
`offline_eval.py` batch-predicts both classifiers directly
(`model.predict_proba` over the whole feature matrix at once) over the
**entire TEST split** (hundreds of thousands of rows, not a sample), which
is both simpler and far faster than looping `DetectionManager.run()` per
event. It reports:

- **binary classifier**: precision/recall/F1/ROC-AUC over the whole test
  split.
- **fine-label classifier**: per-fine-label precision/recall/F1/support (the
  15 CICIDS2017 labels + Benign), at `--category-threshold` (default: the
  current `CATEGORY_CONFIDENCE_THRESHOLD`), with fine-level accuracy and
  coverage.
- **coarse-category classifier**: the same table for the coarse category the
  three-way decision picked (`subagent.decide_category`) — including the
  group-only answer when the fine label is `Unknown`.
- **web attack confusion**: true vs predicted fine label for the three Web
  Attack labels (incl. `Unknown`), and how many web-attack flows got a
  *wrong* specific label at confidence >= threshold.
- **false-positive categorisation**: count + rate (see above).
- **model disagreement count**: binary-anomalous events where the category
  model's top vote was Benign.
- **coverage** (at each level): % of true attacks given a non-`Unknown`
  answer. Coarse coverage >= fine coverage, because the group can clear the
  threshold when no single sub-type does.

Any class with support < `offline_eval.MIN_EVAL_SUPPORT` (50) is marked
**"too small to evaluate"** in the tables, and every macro F1 is printed twice:
over all attack classes, and excluding those too-small classes.

### Per-class tables: Benign, Heartbleed, macro F1 scope

1. **Heartbleed is its own class.** It used to map to the string `"Unknown"`
   (the same string a low-confidence punt uses), so it had to be excluded
   from scoring. It is now a fine label and a coarse category of its own
   (`Heartbleed`), and the exclusion (`is_heartbleed_label`) is gone. It has
   only 3 test rows, so it is flagged too small to evaluate.
2. **Per-class PRECISION includes binary false positives.** The table is
   computed over every *flagged* flow — true attacks the binary model also
   caught, **plus its false positives**, whose true class is
   `classifier.BENIGN_CATEGORY` ("Benign") — so a benign flow mislabeled
   "Bot" counts against Bot's precision. `n` (attacks) and `n_table`
   (attacks + false positives) are both reported. `Benign`'s own row always
   reads precision=recall=0 (a Benign top vote is reported as `"Unknown"`,
   never `"Benign"`); it only penalises other classes' precision.
   **Macro F1 averages attack classes only**, never the Benign row.

Real run on the TEST split, threshold **0.80** (n=84,972 true attacks,
n_table=85,355):

Fine labels — accuracy 0.9884, raw top-1 0.9955, coverage 98.9%, macro F1
**0.846** (**0.875** excluding too-small classes):

| fine label | precision | recall | f1 | support |
|---|---|---|---|---|
| Benign | 0.000 | 0.000 | 0.000 | 383 |
| Bot | 0.877 | 0.864 | 0.870 | 330 |
| DDoS | 1.000 | 0.998 | 0.999 | 25,627 |
| DoS GoldenEye | 0.998 | 0.991 | 0.995 | 1,967 |
| DoS Hulk | 1.000 | 0.993 | 0.996 | 34,621 |
| DoS Slowhttptest | 0.991 | 0.992 | 0.991 | 982 |
| DoS slowloris | 0.997 | 0.992 | 0.995 | 1,100 |
| FTP-Patator | 1.000 | 0.999 | 1.000 | 1,129 |
| Heartbleed | 1.000 | 0.667 | 0.800 | 3 (too small) |
| Infiltration | 1.000 | 0.600 | 0.750 | 5 (too small) |
| PortScan | 0.990 | 0.980 | 0.985 | 18,151 |
| SSH-Patator | 1.000 | 0.994 | 0.997 | 631 |
| Web Attack - Brute Force | 0.809 | 0.518 | 0.631 | 311 |
| Web Attack - Sql Injection | 1.000 | 0.500 | 0.667 | 2 (too small) |
| Web Attack - XSS | 0.265 | 0.115 | 0.160 | 113 |

Coarse categories (derived from the fine decision) — accuracy 0.9917, raw
top-1 0.9975, coverage 99.2%, macro F1 **0.923** (**0.972** excluding
too-small classes):

| category | precision | recall | f1 | support |
|---|---|---|---|---|
| Benign | 0.000 | 0.000 | 0.000 | 383 |
| Botnet | 0.877 | 0.864 | 0.870 | 330 |
| BruteForce | 1.000 | 0.997 | 0.999 | 1,760 |
| DDoS | 1.000 | 0.998 | 0.999 | 25,627 |
| DoS | 1.000 | 0.994 | 0.997 | 38,670 |
| Heartbleed | 1.000 | 0.667 | 0.800 | 3 (too small) |
| Infiltration | 1.000 | 0.600 | 0.750 | 5 (too small) |
| PortScan | 0.990 | 0.980 | 0.985 | 18,151 |
| WebAttack | 0.998 | 0.972 | 0.985 | 426 |

### Threshold 0.80 vs 0.90 (TEST split)

| | 0.80 | 0.90 |
|---|---|---|
| coarse macro F1 (all / excl. too-small) | 0.923 / 0.972 | 0.881 / 0.966 |
| fine macro F1 (all / excl. too-small) | 0.846 / 0.875 | 0.813 / 0.861 |
| coarse coverage | 99.2% | 98.7% |
| fine coverage | 98.9% | 98.4% |
| FP categorisation (binary false positives given a specific category) | 227/383 (59.3%) | 186/383 (48.6%) |
| web-attack flows with a wrong specific label | 73/426 | 46/426 |

Before the fine-label change (coarse model, 0.90), coarse macro F1 over the
same 7 attack classes was 0.936; the fine-label model scores 0.935 on those
7 classes at 0.90, i.e. no regression.

### Web attack confusion (known limitation)

True (rows) vs predicted fine label (columns), TEST split, scored attacks:

Threshold 0.80:

| true \ predicted | Unknown | Brute Force | Sql Injection | XSS |
|---|---|---|---|---|
| Web Attack - Brute Force | 114 | 161 | 0 | 36 |
| Web Attack - Sql Injection | 1 | 0 | 1 | 0 |
| Web Attack - XSS | 63 | 37 | 0 | 13 |

**73 of 426** web-attack flows (17%) get a wrong specific fine label with
confidence >= 0.80 (46/426 at 0.90). Brute Force and XSS flows are
statistically near-identical in CICIDS2017's flow features (the attacks run
through the same web-form traffic), so the model cannot separate them; XSS
recall is 0.115. The coarse `WebAttack` category is unaffected (F1 0.985),
which is why downstream consumers of the coarse category see no regression.
Sql Injection has 2 test rows — no conclusion possible.

### Threshold sweep and recommendation rule

`--offline` sweeps `CATEGORY_CONFIDENCE_THRESHOLD` over
`offline_eval.CATEGORY_THRESHOLD_SWEEP` (0.5/0.6/0.7/0.8/0.9) — **on a
VALIDATION split carved from TRAIN** (`data.split_validation`), never on
TEST. Prints the table and saves it to
`src/detection/results/category_threshold_sweep.csv` (gitignored). A real
run (`cat_acc` = coarse category accuracy, which drives the recommendation;
label columns are the fine-label equivalents):

| threshold | n_scored | cat_acc | %cat_unk | label_acc | %lbl_unk | n_fp | fp_rate |
|---|---|---|---|---|---|---|---|
| 0.50 | 68,119 | 0.9980 | 0.2 | 0.9979 | 0.2 | 190 | 50.5 |
| 0.60 | 68,119 | 0.9968 | 0.3 | 0.9967 | 0.3 | 190 | 47.9 |
| 0.70 | 68,119 | 0.9952 | 0.5 | 0.9950 | 0.5 | 190 | 45.8 |
| 0.80 | 68,119 | 0.9931 | 0.7 | 0.9930 | 0.7 | 190 | 45.3 |
| 0.90 | 68,119 | 0.9896 | 1.0 | 0.9894 | 1.1 | 190 | 41.6 |

**Recommendation rule** (`recommend_category_threshold`,
`--min-category-accuracy`, default `0.99`): recommend the **highest** swept
threshold whose validation coarse category accuracy is `>= min_category_accuracy`.
If none clears the bar, fall back to the highest-accuracy row (ties: lower
FP rate, then higher threshold) and say so. `--offline` prints the rule that
fired (`describe_recommendation_rule`), then the recommended threshold's
TEST tables.

**With the fine-label model, validation accuracy at 0.90 is 0.9896 — just
below the 0.99 bar (the old coarse model's was 0.992) — so the rule now
picks 0.80 (0.9931).** `DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD` was set to
0.80 by hand to match; the script itself never changes it. The cost: more
borderline calls are trusted, including on binary false positives
(FP categorisation 48.6% -> 59.3%).

## Known limitations

- **Web Attack - XSS vs Brute Force** are not separable by this model (see
  above); Sql Injection, Heartbleed and Infiltration have too few test rows
  (2, 3, 5) to evaluate.
- **Bot** (support 330, F1 0.870) is the weakest well-supported class.
- The false-positive categorisation rate (59.3% at 0.80) is reduced but not
  eliminated by the Benign class — see
  [design-decisions.md](design-decisions.md) for why this isn't obviously a
  code bug.
- The threshold sweep is run once against a fixed validation split; it is
  not re-run automatically when the training data or feature engineering
  changes.
