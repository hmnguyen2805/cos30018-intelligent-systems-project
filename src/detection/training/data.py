"""
Shared data loading/cleaning, and the ONE train/test split every training
script and evaluate.py must use (fixed, non-parameterized random_state) —
so no script can accidentally test on a row another model trained on.
"""
import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "chethuhn/network-intrusion-dataset"
LABEL_COL = "Label"
SPLIT_RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.2

BENIGN_LABEL = "Benign"

# Fine label (CICIDS2017's own vocabulary, cleaned) -> coarse attack category. The category
# model predicts the fine labels; coarse categories are always derived from this table, never
# predicted separately. BENIGN_LABEL maps to None: "no attack category". "Unknown" is not in
# here: it only ever means a low-confidence decision (subagent.py).
FINE_LABEL_TO_CATEGORY = {
    BENIGN_LABEL: None,
    "Bot": "Botnet",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}

# The fixed coarse vocabulary (plus "Unknown", the low-confidence answer).
ALLOWED_CATEGORIES = sorted({c for c in FINE_LABEL_TO_CATEGORY.values() if c}) + ["Unknown"]

_FINE_BY_LOWER = {label.lower(): label for label in FINE_LABEL_TO_CATEGORY}
# CICIDS2017 ships "Web Attack <mangled separator> Brute Force" etc.; match the prefix and
# rebuild with a plain " - " separator.
_WEB_ATTACK = re.compile(r"^web attack[^a-z]+(.+)$")


def map_cicids_label_to_fine(raw_label) -> Optional[str]:
    """Clean fine label for a raw CICIDS2017 Label (BENIGN -> "Benign", Web
    Attack separator normalised); idempotent on already-clean labels. None
    for anything unrecognized."""
    if not isinstance(raw_label, str):
        return None
    normalized = raw_label.strip().lower()
    web = _WEB_ATTACK.match(normalized)
    if web:
        normalized = f"web attack - {web.group(1)}"
    return _FINE_BY_LOWER.get(normalized)


def map_cicids_label_to_category(raw_label) -> Optional[str]:
    """Fine -> coarse: map one CICIDS2017 Label (raw or already-clean fine
    label) to ALLOWED_CATEGORIES. None for BENIGN and any unrecognized
    label — both excluded from category scoring/training."""
    return FINE_LABEL_TO_CATEGORY.get(map_cicids_label_to_fine(raw_label))


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
    """Per-feature median and MAD, computed from `df` (train_binary.py
    passes TRAIN only, no test-set leakage) — used by classifier.top_features
    to rank how unusual an event's value is."""
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
    """THE train/test split — every training script and evaluate.py must
    call this, never their own train_test_split, so no row is ever tested
    on by a model that trained on it. Stratified on the binary label."""
    y_binary = binarize_labels(df[LABEL_COL])
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=SPLIT_TEST_SIZE, stratify=y_binary, random_state=SPLIT_RANDOM_STATE,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


VALIDATION_RANDOM_STATE = 123
VALIDATION_SIZE = 0.2


def split_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of TRAIN (never TEST), e.g. for the
    threshold sweep in evaluate.py --offline. Call on the train_df
    split_train_test returned, not the full dataset."""
    y_binary = binarize_labels(train_df[LABEL_COL])
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=VALIDATION_SIZE, stratify=y_binary,
        random_state=VALIDATION_RANDOM_STATE,
    )
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
