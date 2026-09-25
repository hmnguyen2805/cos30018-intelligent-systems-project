"""
Baseline classifier — the tool the Detection Subagent calls.

An "artifact" is a dict `{"model": RandomForestClassifier, "feature_names":
[str, ...], "feature_medians": {str: float} | None}`, produced by train.py
and loaded here. `feature_names` fixes the column order the model was
trained on, so a TrafficEvent's feature dict (unordered, possibly partial)
can be turned into a matching row. `feature_medians` is optional — old
artifacts saved before it existed won't have it.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "detection_rf.joblib")


def load_artifact(path: str) -> dict:
    """Load a {"model", "feature_names"} artifact saved by train.py."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.detection.train` "
            "(train.py) first."
        )
    return joblib.load(path)


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


def top_features(artifact: dict, features: Dict[str, float], k: int = 5) -> List[dict]:
    """The k features most relevant to this event's prediction, ranked by the
    model's `feature_importances_` (a fixed, training-time ranking — not
    per-event, but cheap and good enough to point an LLM's explanation at the
    right columns). Each entry carries this event's value for that feature
    and, when the artifact has `feature_medians`, the training-set median so
    the LLM can say "above/below normal" without inventing a number."""
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    medians = artifact.get("feature_medians")

    ranked_idx = np.argsort(model.feature_importances_)[::-1][:k]
    result = []
    for idx in ranked_idx:
        name = feature_names[idx]
        entry = {"name": name, "value": features.get(name, 0.0)}
        if medians is not None:
            entry["median"] = medians.get(name)
        result.append(entry)
    return result
