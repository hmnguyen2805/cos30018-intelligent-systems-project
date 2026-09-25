"""
Trains the baseline binary RandomForest (BENIGN vs anomalous) and saves it
to classifier.DEFAULT_BINARY_MODEL_PATH (models/detection_binary.joblib).

Usage:
    python -m src.detection.training.train_binary

Or via `python -m src.detection.train`, which runs this and train_category.py
together. Downloads CICIDS2017 via kagglehub on first run (needs Kaggle API
credentials — see https://github.com/Kagglehub/kagglehub#authenticate),
caching it locally after that.
"""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.detection import classifier
from src.detection.training import data


def train_baseline_model(X, y, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return model


def main():
    df = data.load_clean_dataframe()
    train_df, test_df = data.split_train_test(df)
    feature_names = data.select_feature_names(df, label_col=data.LABEL_COL)

    X_train = train_df[feature_names].to_numpy()
    y_train = data.binarize_labels(train_df[data.LABEL_COL])
    X_test = test_df[feature_names].to_numpy()
    y_test = data.binarize_labels(test_df[data.LABEL_COL])

    model = train_baseline_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    print(classification_report(y_test, y_pred, target_names=["benign", "anomalous"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    feature_medians, feature_mad = data.compute_feature_stats(train_df, feature_names)

    os.makedirs(os.path.dirname(classifier.DEFAULT_BINARY_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names,
         "feature_medians": feature_medians, "feature_mad": feature_mad},
        classifier.DEFAULT_BINARY_MODEL_PATH,
    )
    print(f"Saved artifact to {classifier.DEFAULT_BINARY_MODEL_PATH}")


if __name__ == "__main__":
    main()
