"""
Thin wrapper: trains both models needed at inference time.

Usage:
    python -m src.detection.train

Runs training.train_binary then training.train_category — see
training/data.py for the shared data loading and train/test split both use.
Run either individually (python -m src.detection.training.train_binary /
train_category) if you only need one. See docs/design-decisions.md for the
old single-artifact layout this replaced.
"""
from src.detection.training import train_binary, train_category


def main():
    print("=== Training binary model (BENIGN vs anomalous) ===")
    train_binary.main()
    print("\n=== Training category model (attack category, anomalous rows only) ===")
    train_category.main()


if __name__ == "__main__":
    main()
