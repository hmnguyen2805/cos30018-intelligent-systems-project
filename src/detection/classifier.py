"""
Baseline classifier — the tool the Detection Subagent calls.

An "artifact" is a dict `{"model": RandomForestClassifier, "feature_names":
[str, ...]}`, produced by train.py and loaded here. `feature_names` fixes the
column order the model was trained on, so a TrafficEvent's feature dict
(unordered, possibly partial) can be turned into a matching row.
"""
from pathlib import Path
from typing import Dict, Tuple

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
