"""
MCP tool server exposing classifier.py to the LLM agent — see
docs/architecture.md (MCP server's role) for the tool list and why each is
or isn't exposed. Artifacts are loaded once, lazily, on first tool call.

Tools are event_id-based, not features-based: register_event/clear_event
store an event's features server-side under a short id, so the LLM-facing
tools and the task prompt never carry the full feature dict.

Run standalone (stdio transport) for manual testing:
    python -m src.detection.llm.tool_server

DetectionSubagent instead launches this as a subprocess via smolagents'
MCPClient.
"""
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from src.detection import classifier

mcp = FastMCP("detection-tools")
_artifact = None  # lazily loaded; tests can set this directly
_category_artifact = None  # lazily loaded; tests can set this directly
_event_features: Dict[str, Dict[str, float]] = {}


def _get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        _artifact = classifier.load_artifact(classifier.DEFAULT_BINARY_MODEL_PATH)
    return _artifact


def _get_category_artifact() -> dict:
    global _category_artifact
    if _category_artifact is None:
        _category_artifact = classifier.load_artifact(classifier.DEFAULT_CATEGORY_MODEL_PATH)
    return _category_artifact


def _features_for(event_id: str) -> Dict[str, float]:
    if event_id not in _event_features:
        raise KeyError(f"Unknown event_id {event_id!r} — register_event must be called first.")
    return _event_features[event_id]


class VoteSpread(BaseModel):
    vote_fraction: float
    vote_std: float


class FeatureEntry(BaseModel):
    """One entry of a top_features result. median/direction are null when
    the artifact has no feature_medians/feature_mad. ranking says which
    method was used; source is "top_k" or "context" (see
    classifier.CONTEXT_FEATURE_CANDIDATES)."""
    name: str
    value: float
    median: Optional[float] = None
    direction: Optional[str] = None
    ranking: str
    source: str


class CategoryPrediction(BaseModel):
    """One entry of a predict_attack_category result."""
    category: str
    probability: float


@mcp.tool()
def register_event(event_id: str, features: Dict[str, float]) -> bool:
    """Store one event's features under event_id. Code-only, hidden from the
    LLM (see llm.layer.LLM_HIDDEN_TOOL_NAMES); features must be plain floats
    (numpy scalars aren't JSON-serializable over the MCP wire)."""
    _event_features[event_id] = features
    return True


@mcp.tool()
def clear_event(event_id: str) -> bool:
    """Remove a previously registered event's features. Code-only, hidden
    from the LLM."""
    _event_features.pop(event_id, None)
    return True


@mcp.tool()
def predict_proba_anomalous(event_id: str) -> float:
    """P(anomalous) in [0, 1] for the registered event. Hidden from the LLM
    — code already computed this."""
    return classifier.predict_proba_anomalous(_get_artifact(), _features_for(event_id))


@mcp.tool()
def tree_vote_spread(event_id: str) -> VoteSpread:
    """Per-tree vote fraction and standard deviation for the registered
    event. High vote_std means the ensemble's trees disagree."""
    vote_fraction, vote_std = classifier.tree_vote_spread(_get_artifact(), _features_for(event_id))
    return VoteSpread(vote_fraction=vote_fraction, vote_std=vote_std)


@mcp.tool()
def predict_attack_category(event_id: str, top_k: int = 3) -> List[CategoryPrediction]:
    """Top-k category probabilities for the registered event, descending.
    Code has already decided this event's actual category before you were
    called; this is for your own investigation only, it changes nothing."""
    features = _features_for(event_id)
    return [
        CategoryPrediction(**entry)
        for entry in classifier.predict_attack_category(_get_category_artifact(), features, top_k=top_k)
    ]


@mcp.tool()
def top_features(event_id: str, k: int = 5) -> List[FeatureEntry]:
    """The k features most relevant to THIS event, plus a small fixed
    flow-shape context set, each with this event's value and (when known)
    the training-set median and direction. Ground any feature you name in
    your explanation in this tool's output — do not mention one it didn't
    return."""
    features = _features_for(event_id)
    return [FeatureEntry(**entry) for entry in classifier.top_features(_get_artifact(), features, k=k)]


if __name__ == "__main__":
    _get_artifact()  # fail fast with classifier.py's clear error if no model is trained yet
    mcp.run()
