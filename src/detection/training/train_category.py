"""
Trains a multiclass RandomForest predicting the attack CATEGORY, plus an
explicit BENIGN_CATEGORY ("Benign") option so it can push back on the binary
model's false positives instead of always guessing an attack — see
docs/design-decisions.md for why. Saves to
classifier.DEFAULT_CATEGORY_MODEL_PATH.

Trained on TRAIN (data.split_train_test): all anomalous rows plus BENIGN
rows downsampled to DEFAULT_BENIGN_TO_ATTACK_RATIO times the attack count
(full real class balance would be slow for no benefit here). TEST is NOT
downsampled, so precision/recall reflect real performance.

Usage:
    python -m src.detection.training.train_category

Or via `python -m src.detection.train`, which runs this and train_binary.py.
"""
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.detection import classifier
from src.detection.training import data

DEFAULT_BENIGN_TO_ATTACK_RATIO = 2.0


def _label_to_category(raw_label):
    """Same as data.map_cicids_label_to_category, but BENIGN maps to
    classifier.BENIGN_CATEGORY instead of None (see module docstring)."""
    category = data.map_cicids_label_to_category(raw_label)
    if category is not None:
        return category
    if isinstance(raw_label, str) and raw_label.strip().lower() == "benign":
        return classifier.BENIGN_CATEGORY
    return None


def build_category_training_set(
    df, feature_names, *, include_benign: bool = False,
    benign_to_attack_ratio: float = None, random_state: int = 42,
):
    """Anomalous rows only, or (include_benign=True) anomalous rows plus
    BENIGN_CATEGORY rows, optionally downsampled to benign_to_attack_ratio
    times the attack count (deterministic, via random_state). Pass no ratio
    to keep every benign row at its real count (as for TEST)."""
    label_fn = _label_to_category if include_benign else data.map_cicids_label_to_category
    categories = df[data.LABEL_COL].map(label_fn)
    mask = categories.notna()
    frame = df.loc[mask]
    y = categories[mask]

    if include_benign and benign_to_attack_ratio is not None:
        is_benign = y == classifier.BENIGN_CATEGORY
        n_attack = int((~is_benign).sum())
        max_benign = int(benign_to_attack_ratio * n_attack)
        benign_index = frame.index[is_benign]
        if len(benign_index) > max_benign:
            rng = np.random.RandomState(random_state)
            keep_benign_index = rng.choice(benign_index, size=max_benign, replace=False)
            keep_index = frame.index[~is_benign].append(pd.Index(keep_benign_index))
            frame = frame.loc[keep_index]
            y = y.loc[keep_index]

    X = frame[feature_names].to_numpy()
    return X, y.to_numpy()


def _class_weights(y) -> dict:
    """Plain class_weight="balanced" would flatten Benign's deliberate
    dominance (see build_category_training_set) down to "one class among
    many", undoing the downsampling ratio. Fix: "balanced" among attack
    classes only, then Benign gets weight 1.0 — keeping its vote weight
    proportional to its actual (still-dominant) row count."""
    is_benign = y == classifier.BENIGN_CATEGORY
    if not is_benign.any():
        return dict(zip(*_balanced(y)))
    attack_classes, attack_weights = _balanced(y[~is_benign])
    weights = dict(zip(attack_classes, attack_weights))
    weights[classifier.BENIGN_CATEGORY] = 1.0
    return weights


def _balanced(y):
    classes = np.unique(y)
    return classes, compute_class_weight("balanced", classes=classes, y=y)


def train_category_model(X, y, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100, class_weight=_class_weights(y), random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return model


def _print_row_counts(label: str, y) -> None:
    counts = dict(sorted(Counter(y).items()))
    print(f"{label}: {len(y)} rows across {len(counts)} categories: {counts}")


def main():
    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    X_train, y_train = build_category_training_set(
        train_df, feature_names, include_benign=True, benign_to_attack_ratio=DEFAULT_BENIGN_TO_ATTACK_RATIO,
    )
    # Test set: real class balance, no downsampling — precision/recall must reflect how the
    # model actually performs on realistic (benign-dominated) traffic.
    X_test, y_test = build_category_training_set(test_df, feature_names, include_benign=True)

    _print_row_counts("Category training set", y_train)
    _print_row_counts("Category test set", y_test)

    model = train_category_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    os.makedirs(os.path.dirname(classifier.DEFAULT_CATEGORY_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"category_model": model, "classes": list(model.classes_), "feature_names": feature_names},
        classifier.DEFAULT_CATEGORY_MODEL_PATH,
    )
    print(f"Saved artifact to {classifier.DEFAULT_CATEGORY_MODEL_PATH}")


if __name__ == "__main__":
    main()
