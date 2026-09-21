# Detection Manager

**Owner:** Vinh Nghiem

Per the tutor's manager-subagent recommendation, Detection is two layers: a
**Detection Manager** that owns the top-level contract, and a **Detection
Subagent** underneath it that does the actual classification work.

## Structure

```text
src/detection/
├── manager.py                    # DetectionManager — owns the contract, delegates
├── subagent.py                   # DetectionSubagent — classifies, handles borderline cases
├── classifier.py                 # tool layer: load model artifact, predict, per-tree vote spread
├── train.py                      # trains the baseline RandomForest on CICIDS2017
├── requirements-detection.txt
└── README.md
```

## Contract

```text
DetectionManager.run(event: TrafficEvent) -> DetectionResult
```

The Manager delegates to the Subagent internally and owns any manager-level
oversight (e.g. later, deciding whether to trust the subagent's result
as-is, or dispatch to a second detection subagent if one gets added).

The Subagent is the actual agent loop: it calls the baseline RandomForest
(`classifier.py`) for `p_anomalous`, and on a borderline score (0.4-0.6) it
doesn't just trust that number — it re-examines via per-tree vote spread (a
second, finer-grained tool call) before finalizing. High tree disagreement
on a borderline call gets flagged in `detector_notes` for downstream
correlation/response to weigh accordingly.

The Detection Manager's conclusion (`DetectionResult`) goes two places:
across to the Mitigation Manager (Callum), since Correlation needs to know
what was detected, and down to the Judge (Minh), who compares it against the
Mitigation Manager's conclusion to produce the final result.

## Training the baseline model

`classifier.py` loads a trained artifact from `classifier.DEFAULT_MODEL_PATH`
(`models/detection_rf.joblib`, gitignored — train locally, don't commit it).
To produce one:

```sh
pip install -r src/detection/requirements-detection.txt
python -m src.detection.train
```

This downloads CICIDS2017 via `kagglehub` (needs Kaggle API credentials —
see the [kagglehub README](https://github.com/Kagglehub/kagglehub#authenticate)),
trains a binary RandomForest (BENIGN vs anomalous), prints a classification
report + ROC-AUC, and saves the artifact.

## Dependencies

Anything beyond the shared `requirements.txt` is in `requirements-detection.txt`:
`kagglehub` (dataset download) and `joblib` (artifact save/load).
