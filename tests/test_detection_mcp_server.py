"""
Unit tests for src.detection.mcp_server's tool functions. `@mcp.tool()`
doesn't wrap the underlying function, so these are called directly — no MCP
client, subprocess, or transport involved. The artifact is lazily loaded via
_get_artifact(), so tests set mcp_server._artifact to a small fake artifact
directly, same shape classifier.load_artifact would produce; no real trained
model file or network access needed.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.detection import mcp_server

FEATURE_NAMES = ["duration", "packet_count"]


@pytest.fixture(autouse=True)
def fake_artifact(monkeypatch):
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    artifact = {"model": model, "feature_names": FEATURE_NAMES}
    monkeypatch.setattr(mcp_server, "_artifact", artifact)
    return artifact


def sample_features():
    return {"duration": 10.0, "packet_count": 10.0}


def test_predict_proba_anomalous_returns_float_in_unit_range():
    p = mcp_server.predict_proba_anomalous(sample_features())
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_tree_vote_spread_returns_vote_fraction_and_std():
    result = mcp_server.tree_vote_spread(sample_features())
    assert 0.0 <= result.vote_fraction <= 1.0
    assert result.vote_std >= 0.0


def test_top_features_returns_k_entries_with_name_and_value():
    result = mcp_server.top_features(sample_features(), k=2)
    assert len(result) == 2
    for entry in result:
        assert entry.name in FEATURE_NAMES
        assert entry.value == sample_features()[entry.name]


def test_top_features_includes_median_when_artifact_has_it(fake_artifact):
    fake_artifact["feature_medians"] = {"duration": 1.0, "packet_count": 2.0}
    result = mcp_server.top_features(sample_features(), k=2)
    for entry in result:
        assert entry.median == fake_artifact["feature_medians"][entry.name]


def test_top_features_median_is_none_without_feature_medians():
    result = mcp_server.top_features(sample_features(), k=1)
    assert result[0].median is None


def test_get_artifact_lazily_caches_and_does_not_reload(monkeypatch, fake_artifact):
    import src.detection.classifier as classifier

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_artifact should not be called: artifact already set")

    monkeypatch.setattr(classifier, "load_artifact", fail_if_called)
    assert mcp_server._get_artifact() is fake_artifact
