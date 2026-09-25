"""
Trains a multiclass RandomForest that predicts the attack CATEGORY, given
that an event is already anomalous, and saves it to
classifier.DEFAULT_CATEGORY_MODEL_PATH (models/detection_category.joblib).

Trained ONLY on the anomalous rows of the train half of data.split_train_test
(the same split train_binary.py uses) — BENIGN rows, and rows whose raw
Label doesn't map to a known category (data.map_cicids_label_to_category
returns None), are excluded: there's no correct target to train on for them.

Usage:
    python -m src.detection.training.train_category

Or via `python -m src.detection.train`, which runs this and train_binary.py
together.
"""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.detection import classifier
from src.detection.training import data


def build_category_training_set(df, feature_names):
    """Anomalous rows only, labeled via data.map_cicids_label_to_category —
    rows with no mapped category (BENIGN or unrecognized) are dropped."""
    categories = df[data.LABEL_COL].map(data.map_cicids_label_to_category)
    mask = categories.notna()
    X = df.loc[mask, feature_names].to_numpy()
    y = categories[mask].to_numpy()
    return X, y


def train_category_model(X, y, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return model


def main():
    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    X_train, y_train = build_category_training_set(train_df, feature_names)
    X_test, y_test = build_category_training_set(test_df, feature_names)
    print(f"Category training set: {len(X_train)} anomalous rows "
          f"({len(set(y_train))} categories); test: {len(X_test)} rows.")

    model = train_category_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(classifier.DEFAULT_CATEGORY_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"category_model": model, "classes": list(model.classes_), "feature_names": feature_names},
        classifier.DEFAULT_CATEGORY_MODEL_PATH,
    )
    print(f"Saved artifact to {classifier.DEFAULT_CATEGORY_MODEL_PATH}")


if __name__ == "__main__":
    main()
