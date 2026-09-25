"""
Thin wrapper: trains both models needed at inference time.

Usage:
    python -m src.detection.train

Runs training.train_binary (BENIGN vs anomalous -> models/detection_binary.joblib)
and training.train_category (attack category, anomalous rows only ->
models/detection_category.joblib) — see src/detection/training/data.py for
the shared data loading and the one train/test split both use. Run either
individually via `python -m src.detection.training.train_binary` /
`python -m src.detection.training.train_category` if you only need one.

Migrating from the old single-artifact layout: this project used to save
everything to models/detection_rf.joblib. That path is no longer read by
anything — this script (or classifier.load_artifact's own error message,
if you hit it first) is the fix: retrain to get the new
models/detection_binary.joblib + models/detection_category.joblib.
"""
from src.detection.training import train_binary, train_category


def main():
    print("=== Training binary model (BENIGN vs anomalous) ===")
    train_binary.main()
    print("\n=== Training category model (attack category, anomalous rows only) ===")
    train_category.main()


if __name__ == "__main__":
    main()
