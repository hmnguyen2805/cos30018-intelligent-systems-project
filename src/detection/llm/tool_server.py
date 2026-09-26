"""
MCP tool server exposing the Detection classifier's diagnostic tools to an
LLM agent (see subagent.py's `use_llm` path). Artifacts are loaded once,
lazily, on first tool call; every tool wraps a pure function in classifier.py.

Tools are event_id-based, not features-based: `register_event`/`clear_event`
store an event's features server-side under a short id that code calls
directly (not exposed to the LLM agent — see subagent.py), so the LLM-facing
tools and the task prompt built around them never carry the full
~78-feature dict. That keeps the prompt small, which matters a lot on a slow
local CPU model.

The attack category itself is a deterministic decision code makes (see
subagent.py._choose_category, using classifier.predict_attack_category
directly, not through this server) — predict_attack_category is exposed
here only so an agent-mode LLM can optionally investigate it for its own
explanation-writing; calling it never changes the category subagent.py
actually uses.

Run standalone (stdio transport) for manual testing:
    python -m src.detection.llm.tool_server

The Detection Subagent instead launches this as a subprocess via smolagents'
MCPClient, which speaks the same stdio protocol.
"""
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from src.detection import classifier

mcp = FastMCP("detection-tools")
_artifact = None  # lazily loaded by _get_artifact() — tests can set this directly
_category_artifact = None  # lazily loaded by _get_category_artifact() — tests can set this directly
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
    """One entry of a top_features result. `median`/`direction` are null when
    the loaded artifact predates train.py saving feature_medians/feature_mad.
    `ranking` says how this event's top-k was selected ("per_event_deviation"
    when medians+MAD were available, "global_importance_fallback" otherwise —
    always the same for every entry in one call). `source` is "top_k" or
    "context" (the small fixed set of flow-shape fields — destination port,
    flow duration, packet counts, SYN/FIN/RST flags — always included in
    addition to the top-k, see classifier.CONTEXT_FEATURE_CANDIDATES)."""
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
    """Register one event's features under event_id for later event_id-based
    tool calls. Not an LLM-facing tool — code calls this directly (see
    DetectionSubagent._call_llm_agent) before starting the LLM loop, so the
    LLM never sees a raw feature dict. Values must be plain floats (no numpy
    scalars — those aren't JSON-serializable over the MCP wire)."""
    _event_features[event_id] = features
    return True


@mcp.tool()
def clear_event(event_id: str) -> bool:
    """Remove a previously registered event's features. Not an LLM-facing
    tool — code calls this directly once an event's LLM loop is done."""
    _event_features.pop(event_id, None)
    return True


@mcp.tool()
def predict_proba_anomalous(event_id: str) -> float:
    """Return P(anomalous) in [0, 1] for the event registered under
    event_id (see register_event)."""
    return classifier.predict_proba_anomalous(_get_artifact(), _features_for(event_id))


@mcp.tool()
def tree_vote_spread(event_id: str) -> VoteSpread:
    """Per-tree vote fraction and standard deviation for the event
    registered under event_id. High `vote_std` means the ensemble's trees
    disagree — a sign the averaged probability alone is unreliable."""
    vote_fraction, vote_std = classifier.tree_vote_spread(_get_artifact(), _features_for(event_id))
    return VoteSpread(vote_fraction=vote_fraction, vote_std=vote_std)


@mcp.tool()
def predict_attack_category(event_id: str, top_k: int = 3) -> List[CategoryPrediction]:
    """Per-category probabilities for the event registered under event_id,
    from the multiclass category model (only meaningful for an event
    already believed anomalous — the model is trained on anomalous traffic
    only). Returns the top_k categories sorted by probability, descending.
    Note: code has already decided this event's actual category before you
    were called — this tool is for your own investigation and explanation,
    calling it does not change that decision."""
    features = _features_for(event_id)
    return [
        CategoryPrediction(**entry)
        for entry in classifier.predict_attack_category(_get_category_artifact(), features, top_k=top_k)
    ]


@mcp.tool()
def top_features(event_id: str, k: int = 5) -> List[FeatureEntry]:
    """The k features most relevant to THIS event (ranked by how unusual its
    values are, not a fixed global ranking — different events can return
    different features), plus a small fixed set of flow-shape context
    features (destination port, flow duration, packet counts, SYN/FIN/RST
    flag counts) always included in addition. Each item has `name`, this
    event's `value`, and — when available — the training-set `median` and a
    `direction` ("above"/"below"/"near" median, already computed for you).
    Ground any feature you name in your explanation in this tool's output —
    including the context features — do not mention a feature this tool did
    not return."""
    features = _features_for(event_id)
    return [FeatureEntry(**entry) for entry in classifier.top_features(_get_artifact(), features, k=k)]


if __name__ == "__main__":
    _get_artifact()  # fail fast with classifier.py's clear error if no model is trained yet
    mcp.run()
