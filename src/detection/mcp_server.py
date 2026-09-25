"""
MCP tool server exposing the Detection classifier's diagnostic tools to an
LLM agent (see subagent.py's `use_llm` path). The artifact is loaded once,
lazily, on first tool call; every tool wraps a pure function in classifier.py.

Run standalone (stdio transport) for manual testing:
    python -m src.detection.mcp_server

The Detection Subagent instead launches this as a subprocess via smolagents'
MCPClient, which speaks the same stdio protocol.
"""
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from src.detection import classifier

mcp = FastMCP("detection-tools")
_artifact = None  # lazily loaded by _get_artifact() — tests can set this directly


def _get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        _artifact = classifier.load_artifact(classifier.DEFAULT_MODEL_PATH)
    return _artifact


class VoteSpread(BaseModel):
    vote_fraction: float
    vote_std: float


class FeatureEntry(BaseModel):
    """One entry of a top_features result. `median` is null when the loaded
    artifact predates train.py saving feature_medians."""
    name: str
    value: float
    median: Optional[float] = None


@mcp.tool()
def predict_proba_anomalous(features: Dict[str, float]) -> float:
    """Return P(anomalous) in [0, 1] for one traffic event, per the baseline
    RandomForest. `features` maps feature name -> numeric value; missing
    features default to 0.0."""
    return classifier.predict_proba_anomalous(_get_artifact(), features)


@mcp.tool()
def tree_vote_spread(features: Dict[str, float]) -> VoteSpread:
    """Per-tree vote fraction and standard deviation for one event. High
    `vote_std` means the ensemble's trees disagree — a sign the averaged
    probability alone is unreliable for this event."""
    vote_fraction, vote_std = classifier.tree_vote_spread(_get_artifact(), features)
    return VoteSpread(vote_fraction=vote_fraction, vote_std=vote_std)


@mcp.tool()
def top_features(features: Dict[str, float], k: int = 5) -> List[FeatureEntry]:
    """The k features most relevant to this event's prediction. Each item has
    `name`, this event's `value` for that feature, and (when available) the
    training-set `median` for that feature, for comparing this event against
    normal traffic. Ground any feature you name in your explanation in this
    tool's output — do not mention a feature this tool did not return."""
    return [FeatureEntry(**entry) for entry in classifier.top_features(_get_artifact(), features, k=k)]


if __name__ == "__main__":
    _get_artifact()  # fail fast with classifier.py's clear error if no model is trained yet
    mcp.run()
