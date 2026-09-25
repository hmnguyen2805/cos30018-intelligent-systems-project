"""
Trains a multiclass RandomForest that predicts the attack CATEGORY — or
BENIGN_CATEGORY ("Benign"), an explicit "this doesn't look like an attack"
option — and saves it to classifier.DEFAULT_CATEGORY_MODEL_PATH
(models/detection_category.joblib).

Why a Benign class at all: this model is only ever consulted for events the
BINARY model already called anomalous (subagent.py._choose_category), so an
earlier version trained on attack rows only, with no way to express "this
doesn't look like an attack". On a real evaluation run, 5 events the binary
model wrongly flagged anomalous all got a confident attack category (Botnet
0.97-1.0 x4, DoS 0.99) — there was no other option for the model to vote for.
Including Benign lets the category model push back on the binary model's
false positives instead of always guessing an attack; subagent.py treats a
Benign top vote as a binary/category model disagreement (see
_choose_category), not as a specific category.

Trained on the TRAIN half of data.split_train_test (the same split
train_binary.py uses): all anomalous rows (labels mapping to a category via
data.map_cicids_label_to_category) plus BENIGN rows downsampled to
DEFAULT_BENIGN_TO_ATTACK_RATIO times the attack-row count — training on the
full, real class balance (benign vastly outnumbers attacks) would be slow for
no accuracy benefit here; the goal is giving the model an explicit benign
option to vote for, not replicating the real prevalence. The TEST split is
NOT downsampled — evaluated at its real class balance, so precision/recall
per class (Benign included) reflect how the model actually performs.

Usage:
    python -m src.detection.training.train_category

Or via `python -m src.detection.train`, which runs this and train_binary.py
together.
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
    """Same categories as data.map_cicids_label_to_category, but BENIGN maps
    to classifier.BENIGN_CATEGORY instead of None — the category model needs
    an explicit "this looks benign" option to vote for (see module
    docstring). Unrecognized labels still map to None (excluded)."""
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
    """Anomalous rows only (data.map_cicids_label_to_category), or —  when
    include_benign is True — anomalous rows plus classifier.BENIGN_CATEGORY
    rows. Rows with no mapped category (unrecognized labels) are always
    dropped.

    include_benign=True with a benign_to_attack_ratio also downsamples
    BENIGN_CATEGORY rows to at most that many times the attack-row count
    (deterministic, via random_state) — see the module docstring for why.
    Pass include_benign=True with no ratio (as for TEST) to keep every
    benign row at its real count.
    """
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
    """sklearn's class_weight="balanced" gives every class equal TOTAL vote
    weight (n_c * w_c is the same for every class) — exactly what rare attack
    types like Infiltration need against common ones like DoS, but applied to
    Benign too it silently undoes build_category_training_set's deliberate
    benign_to_attack_ratio: Benign is downsampled to *dominate* (e.g. 2x the
    total attack count) specifically so the model has ample benign evidence,
    and "balanced" would flatten that back down to "one class among nine",
    the same as Botnet's 1,564 rows — which is exactly how a real evaluation
    run still handed confident attack categories to benign false positives
    even after Benign was added as a class (see the module docstring).

    Fix: compute "balanced" weights among the ATTACK classes only (so rare
    attacks stay fairly represented against common ones), then give Benign a
    weight of 1.0 — since attack-only "balanced" weights average to 1.0 by
    construction, this keeps Benign's total vote weight proportional to its
    actual (downsampled-but-still-dominant) row count, not artificially
    flattened to match a single attack class."""
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
