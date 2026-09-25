"""
Shared data loading, cleaning, and — critically — the ONE train/test split
every training script and evaluate.py must use. A fixed, non-parameterized
random_state: this is the split that decides which rows either model can
ever be trained on, so it can't silently drift between train_binary.py,
train_category.py, and evaluate.py. If it did, evaluate.py could end up
"testing" on a row one of the models was actually trained on.
"""
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"
LABEL_COL = "Label"
SPLIT_RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.2

# The fixed attack-category vocabulary the category model is trained to predict (see
# train_category.py) and the LLM layer explains (see llm_notes.py) — "Unknown" covers both a
# low-confidence classifier decision (subagent.py) and Heartbleed, which has no category of its
# own here (see map_cicids_label_to_category below).
ALLOWED_CATEGORIES = [
    "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet", "Infiltration", "Unknown",
]

# CICIDS2017's raw multiclass Label -> our fixed attack category list above. BENIGN (and
# anything unrecognized) maps to None: "no true attack category to score against". "Web Attack
# � Brute Force" etc. (the dataset ships this mangled-separator encoding for all three Web
# Attack subtypes) are matched by prefix, not exact string.
_CICIDS_LABEL_TO_CATEGORY = {
    "benign": None,
    "bot": "Botnet",
    "ddos": "DDoS",
    "dos goldeneye": "DoS",
    "dos hulk": "DoS",
    "dos slowhttptest": "DoS",
    "dos slowloris": "DoS",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "heartbleed": "Unknown",
    "infiltration": "Infiltration",
    "portscan": "PortScan",
}


def map_cicids_label_to_category(raw_label) -> Optional[str]:
    """Map one CICIDS2017 `Label` value to the fixed category vocabulary
    (ALLOWED_CATEGORIES). Returns None for BENIGN and for any label this
    mapping doesn't recognize — both mean "not a true-positive attack
    category to score/train against", so train_category.py and evaluate.py
    exclude them the same way."""
    if not isinstance(raw_label, str):
        return None
    normalized = raw_label.strip().lower()
    if normalized.startswith("web attack"):
        return "WebAttack"
    return _CICIDS_LABEL_TO_CATEGORY.get(normalized)


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """Load and concatenate every CSV under dataset_dir (CICIDS2017 ships as
    8 per-day CSVs)."""
    csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


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


def select_feature_names(df: pd.DataFrame, label_col: str = LABEL_COL) -> List[str]:
    """Numeric columns other than the label — the models' input features."""
    numeric_cols = df.select_dtypes(include="number").columns
    return [c for c in numeric_cols if c != label_col]


def compute_feature_stats(df: pd.DataFrame, feature_names: List[str]) -> Tuple[dict, dict]:
    """Per-feature median and MAD (median absolute deviation), computed from
    `df` (train_binary.py passes the TRAIN split only — no test-set
    leakage). Saved in the binary artifact alongside the model so
    classifier.top_features can rank a feature by how unusual THIS event's
    value is, and show whether it's above/below/near normal without the LLM
    doing arithmetic itself."""
    medians = {name: float(df[name].median()) for name in feature_names}
    mad = {name: float((df[name] - medians[name]).abs().median()) for name in feature_names}
    return medians, mad


def load_clean_dataframe() -> pd.DataFrame:
    """Download (kagglehub, cached locally after the first run — needs
    Kaggle API credentials, see
    https://github.com/Kagglehub/kagglehub#authenticate) and clean the full
    CICIDS2017 dataset."""
    import kagglehub

    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    return clean_dataframe(load_dataset(dataset_dir))


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """THE train/test split. train_binary.py, train_category.py, and
    evaluate.py all call this — never their own train_test_split on the raw
    data — so a row that lands in the test half here is guaranteed to have
    been in neither model's training data. Stratified on the binary label;
    SPLIT_RANDOM_STATE/SPLIT_TEST_SIZE are module constants, not parameters,
    so this can't be called with a different split by accident."""
    y_binary = binarize_labels(df[LABEL_COL])
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=SPLIT_TEST_SIZE, stratify=y_binary, random_state=SPLIT_RANDOM_STATE,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


VALIDATION_RANDOM_STATE = 123
VALIDATION_SIZE = 0.2


def split_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of the TRAIN split — never the test split —
    for things like selecting CATEGORY_CONFIDENCE_THRESHOLD without tuning on
    test data (evaluate.py --offline's threshold sweep). Uses a separate
    fixed random_state from split_train_test, so this is independent of (and
    reproducible alongside) the main split; call it on the train_df that
    split_train_test already returned, not on the full dataset."""
    y_binary = binarize_labels(train_df[LABEL_COL])
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=VALIDATION_SIZE, stratify=y_binary,
        random_state=VALIDATION_RANDOM_STATE,
    )
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
