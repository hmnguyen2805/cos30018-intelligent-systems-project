"""
Baseline classifier — the tools the Detection Subagent calls.

Two separate artifacts, both produced by src/detection/training/:

- Binary (train_binary.py -> DEFAULT_BINARY_MODEL_PATH): {"model":
  RandomForestClassifier, "feature_names", "feature_medians", "feature_mad"}.
  BENIGN vs anomalous.
- Category (train_category.py -> DEFAULT_CATEGORY_MODEL_PATH):
  {"category_model": RandomForestClassifier, "classes", "feature_names"}.
  Multiclass FINE attack label (CICIDS2017's own vocabulary, see
  training/data.FINE_LABEL_TO_CATEGORY) plus an explicit Benign class; coarse
  categories are derived by summing fine-label probabilities (coarse_probabilities).

Both store feature_names in training-time column order, used to turn a
TrafficEvent's feature dict into a matching row (see _feature_vector).
feature_medians/feature_mad and the category artifact are both optional,
for backward compatibility with older artifacts — see
assert_feature_names_match and subagent.py's handling of a missing one.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.detection.training import data

DEFAULT_BINARY_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "detection_binary.joblib")
DEFAULT_CATEGORY_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "detection_category.joblib")

# Old, pre-split single-artifact layout. Not read — only checked so load_artifact can give a
# clear "please retrain" message instead of a bare FileNotFoundError.
_LEGACY_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "detection_rf.joblib")

# Always surfaced by top_features in addition to the top-k, so an LLM always sees flow-shape
# basics even when none are globally important enough to rank on their own. CICIDS2017 columns.
CONTEXT_FEATURE_CANDIDATES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
]

_DEVIATION_SCALE_EPSILON = 1e-6

# The category model's explicit "benign" class — see docs/design-decisions.md for why it
# exists and subagent.py._choose_category for the disagreement handling when it's the top vote.
BENIGN_CATEGORY = data.BENIGN_LABEL


def load_artifact(path: str) -> dict:
    """Load an artifact saved by training/train_binary.py or
    training/train_category.py."""
    if not Path(path).exists():
        if Path(_LEGACY_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"No trained model at {path}, but found an old-format artifact at "
                f"{_LEGACY_MODEL_PATH} — the single-file detection_rf.joblib layout is no "
                "longer used (models are now split into a binary and a category artifact). "
                "Please retrain: `python -m src.detection.train`."
            )
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.detection.train` first."
        )
    return joblib.load(path)


def assert_feature_names_match(binary_artifact: dict, category_artifact: dict) -> None:
    """Raise ValueError if the two artifacts' feature_names differ — a
    mismatch (e.g. one retrained, the other not) would otherwise silently
    misalign feature columns instead of erroring."""
    binary_names = binary_artifact["feature_names"]
    category_names = category_artifact["feature_names"]
    if binary_names != category_names:
        raise ValueError(
            "Binary and category model artifacts were trained on different feature sets — "
            "retrain both together (`python -m src.detection.train`).\n"
            f"binary feature_names ({len(binary_names)}): {binary_names}\n"
            f"category feature_names ({len(category_names)}): {category_names}"
        )


def _feature_vector(artifact: dict, features: Dict[str, float]) -> np.ndarray:
    """Build a model-ready row, aligning `features` to the artifact's
    training-time column order. Missing keys default to 0.0."""
    row = [features.get(name, 0.0) for name in artifact["feature_names"]]
    return np.array([row])


def predict_proba_anomalous(artifact: dict, features: Dict[str, float]) -> float:
    """P(anomalous) for one event, per the baseline RandomForest."""
    model = artifact["model"]
    X = _feature_vector(artifact, features)
    anomalous_idx = list(model.classes_).index(1)
    return float(model.predict_proba(X)[0, anomalous_idx])


def tree_vote_spread(artifact: dict, features: Dict[str, float]) -> Tuple[float, float]:
    """Per-tree vote fraction and std-dev for one event — a finer-grained
    second look the agent uses on borderline calls to gauge ensemble
    consensus rather than trusting the averaged probability alone."""
    model = artifact["model"]
    X = _feature_vector(artifact, features)
    votes = np.array([tree.predict(X)[0] for tree in model.estimators_])
    return float(votes.mean()), float(votes.std())


def coarse_category(fine_label: str) -> str:
    """Coarse group of a fine label (Benign's group is BENIGN_CATEGORY).
    Raises on a label outside the fine vocabulary — e.g. an old coarse-trained
    artifact — rather than silently mis-grouping it."""
    if fine_label not in data.FINE_LABEL_TO_CATEGORY:
        raise ValueError(
            f"Category artifact has label {fine_label!r}, not in the fine-label vocabulary — "
            "retrain: `python -m src.detection.train`."
        )
    return data.FINE_LABEL_TO_CATEGORY[fine_label] or BENIGN_CATEGORY


def coarse_probabilities(fine_probabilities: Dict[str, float]) -> Dict[str, float]:
    """Coarse-category probability = sum of its fine labels' probabilities."""
    totals: Dict[str, float] = {}
    for label, p in fine_probabilities.items():
        group = coarse_category(label)
        totals[group] = totals.get(group, 0.0) + float(p)
    return totals


def predict_attack_category(category_artifact: dict, features: Dict[str, float], top_k: int = 3) -> dict:
    """Category-model output for one event: {"labels": top_k fine labels,
    "categories": every coarse group}, each a descending list of
    {"label"|"category": str, "probability": float}. Coarse probabilities
    are aggregated from the fine ones (coarse_probabilities)."""
    model = category_artifact["category_model"]
    X = _feature_vector(category_artifact, features)
    proba = model.predict_proba(X)[0]
    classes = category_artifact.get("classes") or list(model.classes_)
    fine = dict(zip(classes, proba))
    labels = sorted(fine.items(), key=lambda pair: pair[1], reverse=True)
    categories = sorted(coarse_probabilities(fine).items(), key=lambda pair: pair[1], reverse=True)
    return {
        "labels": [{"label": label, "probability": float(p)} for label, p in labels[:top_k]],
        "categories": [{"category": category, "probability": float(p)} for category, p in categories],
    }


def _direction(value: float, median: Optional[float], scale: Optional[float]) -> Optional[str]:
    """"above" / "below" / "near" `median`, computed by code so the LLM never
    has to do arithmetic on a raw feature value itself. None when there's no
    median to compare against. "near" means the deviation is small relative
    to `scale` (the feature's training-set MAD) — without a scale, any
    nonzero deviation counts as above/below rather than near."""
    if median is None:
        return None
    diff = value - median
    near_threshold = 0.5 * scale if scale else 0.0
    if abs(diff) <= near_threshold:
        return "near"
    return "above" if diff > 0 else "below"


def top_features(artifact: dict, features: Dict[str, float], k: int = 5) -> List[dict]:
    """The k features most relevant to THIS event, plus a small fixed
    context set (CONTEXT_FEATURE_CANDIDATES) so an LLM always sees
    flow-shape basics even when none make the top-k on their own.

    Ranked by importance * |value - median| / mad (how unusual this event's
    value is, not a fixed global ranking) when the artifact has
    feature_medians/feature_mad; falls back to plain feature_importances_
    otherwise — each entry's "ranking" field says which was used, and
    "source" says "top_k" or "context"."""
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    medians = artifact.get("feature_medians")
    scales = artifact.get("feature_mad")
    importances = model.feature_importances_

    if medians is not None and scales is not None:
        ranking_method = "per_event_deviation"
        scores = np.array([
            importances[idx] * abs(float(features.get(name, 0.0)) - medians[name])
            / (scales[name] + _DEVIATION_SCALE_EPSILON)
            for idx, name in enumerate(feature_names)
        ])
    else:
        ranking_method = "global_importance_fallback"
        scores = importances

    ranked_idx = list(np.argsort(scores)[::-1][:k])

    def make_entry(idx: int, source: str) -> dict:
        name = feature_names[idx]
        value = float(features.get(name, 0.0))
        entry = {"name": name, "value": value, "ranking": ranking_method, "source": source}
        if medians is not None:
            median = medians.get(name)
            entry["median"] = median
            entry["direction"] = _direction(value, median, scales.get(name) if scales else None)
        return entry

    result = [make_entry(idx, "top_k") for idx in ranked_idx]

    included_names = {feature_names[idx] for idx in ranked_idx}
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    for context_name in CONTEXT_FEATURE_CANDIDATES:
        if context_name in included_names:
            continue
        idx = name_to_idx.get(context_name)
        if idx is None:
            continue  # not one of this artifact's trained-on columns — skip
        result.append(make_entry(idx, "context"))

    return result
