"""
Trains the baseline RandomForest classifier.py loads at inference time, on
CICIDS2017 (binary: BENIGN vs anomalous).

Usage:
    python -m src.detection.train

Downloads the dataset via kagglehub on first run (needs Kaggle API
credentials — see https://github.com/Kagglehub/kagglehub#authenticate),
caching it locally after that. Saves the trained artifact to
classifier.DEFAULT_MODEL_PATH.
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.detection import classifier

KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"
LABEL_COL = "Label"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (a known CICIDS2017 quirk), drop
    rows with inf/NaN feature values, and drop exact-duplicate rows."""
    df = df.rename(columns=lambda c: c.strip())
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    return df


def binarize_labels(labels: pd.Series) -> np.ndarray:
    """BENIGN -> 0, any attack label -> 1."""
    normalized = labels.str.strip().str.upper()
    return (normalized != "BENIGN").astype(int).to_numpy()


def select_feature_names(df: pd.DataFrame, label_col: str = LABEL_COL) -> list:
    """Numeric columns other than the label — the model's input features."""
    numeric_cols = df.select_dtypes(include="number").columns
    return [c for c in numeric_cols if c != label_col]


def train_baseline_model(X, y, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return model


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """Load and concatenate every CSV under dataset_dir (CICIDS2017 ships as
    8 per-day CSVs)."""
    csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def main():
    import kagglehub

    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    df = load_dataset(dataset_dir)
    df = clean_dataframe(df)

    feature_names = select_feature_names(df, label_col=LABEL_COL)
    X = df[feature_names].to_numpy()
    y = binarize_labels(df[LABEL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = train_baseline_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    print(classification_report(y_test, y_pred, target_names=["benign", "anomalous"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    os.makedirs(os.path.dirname(classifier.DEFAULT_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, classifier.DEFAULT_MODEL_PATH)
    print(f"Saved artifact to {classifier.DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
