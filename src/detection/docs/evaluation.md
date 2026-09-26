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
`data.map_cicids_label_to_category` to the same fixed category vocabulary
`train_category.py`/`_choose_category` use (`Heartbleed` maps to `Unknown`;
`BENIGN` and anything unrecognized map to `None`). It's written to each CSV
row as `true_category`.

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
**79.4%** (304/383 false positives). `print_summary` prints it right after
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
python -m src.detection.evaluate --offline [--category-threshold 0.9] [--min-category-accuracy 0.99]
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
- **category classifier**: per-class precision/recall/F1 (+ row
  counts/support), at `--category-threshold` (default: the current
  `CATEGORY_CONFIDENCE_THRESHOLD`) — see "Per-class table" below for exactly
  what's scored and why.
- **false-positive categorisation**: count + rate (see above).
- **model disagreement count**: binary-anomalous events where the category
  model's top vote was Benign.
- **coverage**: % of true attacks given a specific category (not punted to
  `Unknown`).

### Per-class table: Heartbleed, Benign, and macro F1 scope

Two things the per-class table deliberately does NOT do, both found from a
real `--offline` run:

1. **It never conflates "Unknown" as a true label with "Unknown" as a
   low-confidence prediction.** `data.map_cicids_label_to_category` maps
   CICIDS2017's `Heartbleed` label to the *string* `"Unknown"` — the same
   string `subagent.py._choose_category` uses for a low-confidence punt or a
   binary/category model disagreement. A first version of this table scored
   them as the same class, which silently combined "the model correctly
   recognized this as Heartbleed" with "the model wasn't sure" into one
   meaningless row (precision ~0, recall ~1, dragging macro F1 down for no
   real reason). Fix (`is_heartbleed_label`): **Heartbleed rows are excluded
   from category scoring entirely** and their count reported separately
   (printed as `excluded N Heartbleed event(s)...`) — chosen over giving
   Heartbleed its own category, because the trained category model was
   never taught to tell the two apart (it's trained on
   `data.map_cicids_label_to_category`'s output, which conflates them the
   same way); a real "Heartbleed" class would need retraining, not just a
   reporting fix. Low-confidence predictions are reported as **coverage** (%
   of true attacks given a specific category, an `--offline` top-line
   metric) instead of ever being a class row.
2. **Per-class PRECISION now includes binary false positives.** The table is
   computed over every *flagged* flow — true attacks the binary model also
   caught, **plus its false positives**, whose true class is
   `classifier.BENIGN_CATEGORY` ("Benign") — so a benign flow the category
   model mislabeled "Botnet" now correctly counts against Botnet's
   precision, which it couldn't before false positives were in the scoring
   universe at all. `n` (attack count) and `n_table` (attacks + false
   positives) are both reported. `Benign`'s own row will always read
   precision=recall=0 — "Benign" is never a possible `chosen_category` value
   (a Benign top vote is always reported as `"Unknown"`, never `"Benign"`),
   so it can never itself be "predicted"; its purpose is solely to penalize
   other classes' precision. **Macro F1 is computed over real attack classes
   only** — the Benign row is excluded from the average (it would otherwise
   unfairly drag macro F1 toward 0 for a class that structurally can never
   be "hit").

A real run at the current default (threshold=0.90, excluding 3 Heartbleed
events):

| category | precision | recall | f1 | support |
|---|---|---|---|---|
| Benign | 0.000 | 0.000 | 0.000 | 383 |
| Botnet | 0.946 | 0.791 | 0.861 | 330 |
| BruteForce | 1.000 | 0.993 | 0.997 | 1,760 |
| DDoS | 1.000 | 0.998 | 0.999 | 25,627 |
| DoS | 1.000 | 0.989 | 0.994 | 38,670 |
| Infiltration | 1.000 | 0.600 | 0.750 | 5 |
| PortScan | 0.990 | 0.982 | 0.986 | 18,151 |
| WebAttack | 0.997 | 0.932 | 0.964 | 426 |

n=84,969 true attacks, n_table=85,352 (+383 false positives as Benign).
accuracy=0.9892, raw top-1 accuracy=0.9980, coverage=98.9%, **macro F1
(attack classes only) = 0.936**. Now that Heartbleed's phantom row and its
~0 f1 are gone, macro F1 is markedly higher than the earlier (incorrect)
0.823 figure — the two classes still visibly dragging it down are **Botnet**
(support=330, recall 0.791 — some correct-but-lower-confidence Botnet calls
get punted to `Unknown` at this threshold) and **Infiltration** (support=5,
recall 0.600 — too few rows to draw a real conclusion from). Botnet's and
PortScan's precision (0.946, 0.990) are now visibly below 1.000 — exactly
the false-positive contamination item 2 above exists to surface: some of the
383 binary false positives get mislabeled as those specific categories
instead of punted to `Unknown` (see the false-positive categorisation rate
above).

### Threshold sweep and recommendation rule

`--offline` then sweeps `CATEGORY_CONFIDENCE_THRESHOLD` over
`offline_eval.CATEGORY_THRESHOLD_SWEEP` (0.5/0.6/0.7/0.8/0.9) — **on a
VALIDATION split carved from TRAIN** (`data.split_validation`, a separate
fixed random_state from `data.split_train_test`), never on TEST, so choosing
a threshold never tunes on the data the report above is scored on
(Heartbleed excluded here too, for the same reason). Prints the table and
saves it to `src/detection/results/category_threshold_sweep.csv`
(gitignored). A real run:

| threshold | n_scored | accuracy | %unknown | n_fp | fp_rate |
|---|---|---|---|---|---|
| 0.50 | 68,118 | 0.999 | 0.1 | 190 | 61.1 |
| 0.60 | 68,118 | 0.998 | 0.2 | 190 | 49.5 |
| 0.70 | 68,118 | 0.997 | 0.3 | 190 | 46.8 |
| 0.80 | 68,118 | 0.995 | 0.5 | 190 | 45.8 |
| 0.90 | 68,118 | 0.992 | 0.8 | 190 | 43.7 |

**Recommendation rule** (`recommend_category_threshold`,
`--min-category-accuracy`, default `0.99`): recommend the **highest** swept
threshold whose validation category accuracy is `>= min_category_accuracy` —
among thresholds that all clear the accuracy bar, prefer the strictest one
(more conservative: more borderline guesses get punted to `Unknown` instead
of risking a wrong specific answer). If no threshold clears the bar, falls
back to the single highest-accuracy row (ties broken by the lower
false-positive categorisation rate, then the higher threshold) and says so
explicitly. `--offline` always prints the exact rule that fired
(`describe_recommendation_rule`) alongside the recommendation, then reports
that threshold's full per-class table on TEST for comparison.
`subagent.DEFAULT_CATEGORY_CONFIDENCE_THRESHOLD` is never changed
automatically by this script — only printed as a recommendation.

**On the real sweep above, every one of the 5 swept values (0.999 down to
0.992) already clears the 0.99 bar** — the rule picks the *highest*
threshold among them, **0.90**, directly, with no fallback
(`meets_min_accuracy: True`); it does not matter that 0.90's own accuracy
(0.992) is the lowest of the five, since all five qualify and the rule
prefers the strictest (highest) threshold among qualifiers. **0.90 is the
current default.**

## Known limitations

- **Infiltration** has only 5 support rows in the TEST split — its recall
  (0.600) is not a reliable signal, just too few rows to draw a conclusion
  from.
- **Botnet** recall (0.791) is the other main drag on macro F1: some
  correct-but-lower-confidence Botnet calls get punted to `Unknown` at the
  current threshold.
- The false-positive categorisation rate (79.4%) is reduced but not
  eliminated by the Benign class + threshold tuning — see
  [design-decisions.md](design-decisions.md) for why this isn't obviously a
  code bug, and the threshold/ratio levers still available.
- The threshold sweep is run once against a fixed validation split; it is
  not re-run automatically when the underlying training data or feature
  engineering changes.
