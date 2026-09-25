"""
Unit tests for src.detection.mcp_server's tool functions. `@mcp.tool()`
doesn't wrap the underlying function, so these are called directly — no MCP
client, subprocess, or transport involved. The artifact is lazily loaded via
_get_artifact(), so tests set mcp_server._artifact to a small fake artifact
directly, same shape classifier.load_artifact would produce; no real trained
model file or network access needed.

Tools are event_id-based: register_event stores an event's features
server-side, and predict_proba_anomalous/tree_vote_spread/top_features read
them back by event_id rather than taking a features dict — that's what keeps
raw features out of the LLM's tool-call arguments and task prompt.
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


@pytest.fixture(autouse=True)
def clear_event_registry():
    """_event_features is process/module-global state — reset it around
    every test so tests can't see each other's registered events."""
    mcp_server._event_features.clear()
    yield
    mcp_server._event_features.clear()


def sample_features():
    return {"duration": 10.0, "packet_count": 10.0}


def test_register_event_then_predict_proba_anomalous_returns_float_in_unit_range():
    mcp_server.register_event("evt-1", sample_features())
    p = mcp_server.predict_proba_anomalous("evt-1")
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_proba_anomalous_unknown_event_id_raises_key_error():
    with pytest.raises(KeyError, match="evt-missing"):
        mcp_server.predict_proba_anomalous("evt-missing")


def test_tree_vote_spread_returns_vote_fraction_and_std():
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.tree_vote_spread("evt-1")
    assert 0.0 <= result.vote_fraction <= 1.0
    assert result.vote_std >= 0.0


def test_top_features_returns_k_entries_with_name_and_value():
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=2)
    assert len(result) == 2
    for entry in result:
        assert entry.name in FEATURE_NAMES
        assert entry.value == sample_features()[entry.name]


def test_top_features_includes_median_when_artifact_has_it(fake_artifact):
    fake_artifact["feature_medians"] = {"duration": 1.0, "packet_count": 2.0}
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=2)
    for entry in result:
        assert entry.median == fake_artifact["feature_medians"][entry.name]


def test_top_features_median_is_none_without_feature_medians():
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=1)
    assert result[0].median is None


def test_top_features_ranking_field_reflects_fallback_without_stats():
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=1)
    assert result[0].ranking == "global_importance_fallback"
    assert result[0].direction is None


def test_top_features_ranking_field_is_per_event_deviation_with_medians_and_mad(fake_artifact):
    fake_artifact["feature_medians"] = {"duration": 1.0, "packet_count": 2.0}
    fake_artifact["feature_mad"] = {"duration": 0.5, "packet_count": 0.5}
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=1)
    assert result[0].ranking == "per_event_deviation"
    assert result[0].direction == "above"  # 10.0 is well above either median


def test_top_features_top_k_entries_are_source_top_k():
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.top_features("evt-1", k=2)
    assert all(entry.source == "top_k" for entry in result)


def test_top_features_includes_context_feature_not_otherwise_in_top_k(fake_artifact):
    # Add a "Destination Port" column that's constant across training rows, so RF gives it
    # ~zero importance and it would never appear in a small top-k on its own.
    from sklearn.ensemble import RandomForestClassifier

    feature_names = FEATURE_NAMES + ["Destination Port"]
    X = np.array([
        [0.0, 0.0, 80.0], [0.1, 0.0, 80.0], [0.0, 0.1, 80.0], [0.1, 0.1, 80.0],
        [10.0, 10.0, 80.0], [10.1, 10.0, 80.0], [10.0, 10.1, 80.0], [10.1, 10.1, 80.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    fake_artifact["model"] = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    fake_artifact["feature_names"] = feature_names

    mcp_server.register_event("evt-1", {"duration": 10.0, "packet_count": 10.0, "Destination Port": 443.0})
    result = mcp_server.top_features("evt-1", k=1)

    port_entry = next(entry for entry in result if entry.name == "Destination Port")
    assert port_entry.source == "context"
    assert port_entry.value == 443.0


def test_clear_event_removes_it_so_later_calls_raise():
    mcp_server.register_event("evt-1", sample_features())
    mcp_server.clear_event("evt-1")
    with pytest.raises(KeyError):
        mcp_server.predict_proba_anomalous("evt-1")


def test_clear_event_on_unknown_id_does_not_raise():
    mcp_server.clear_event("never-registered")  # must not raise


def test_events_are_isolated_by_id():
    mcp_server.register_event("evt-benign", {"duration": 0.0, "packet_count": 0.0})
    mcp_server.register_event("evt-anomalous", {"duration": 10.0, "packet_count": 10.0})

    p_benign = mcp_server.predict_proba_anomalous("evt-benign")
    p_anomalous = mcp_server.predict_proba_anomalous("evt-anomalous")

    assert p_benign < p_anomalous


def test_get_artifact_lazily_caches_and_does_not_reload(monkeypatch, fake_artifact):
    import src.detection.classifier as classifier

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_artifact should not be called: artifact already set")

    monkeypatch.setattr(classifier, "load_artifact", fail_if_called)
    assert mcp_server._get_artifact() is fake_artifact


# --- predict_attack_category ---------------------------------------------

@pytest.fixture
def fake_category_artifact(monkeypatch):
    X = np.array([
        [0.0, 0.0], [0.1, 0.0],       # PortScan-ish
        [10.0, 10.0], [10.1, 10.0],   # DDoS-ish
    ])
    y = np.array(["PortScan", "PortScan", "DDoS", "DDoS"])
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    artifact = {"category_model": model, "classes": list(model.classes_), "feature_names": FEATURE_NAMES}
    monkeypatch.setattr(mcp_server, "_category_artifact", artifact)
    return artifact


def test_predict_attack_category_returns_category_predictions(fake_category_artifact):
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.predict_attack_category("evt-1", top_k=2)
    assert len(result) == 2
    assert all(isinstance(entry.category, str) for entry in result)
    assert result[0].probability >= result[1].probability


def test_predict_attack_category_default_top_k_is_three(fake_category_artifact):
    mcp_server.register_event("evt-1", sample_features())
    result = mcp_server.predict_attack_category("evt-1")
    assert len(result) <= 3  # only 2 classes exist in this fixture, capped by that


def test_predict_attack_category_unknown_event_id_raises_key_error(fake_category_artifact):
    with pytest.raises(KeyError, match="evt-missing"):
        mcp_server.predict_attack_category("evt-missing")


def test_get_category_artifact_lazily_caches_and_does_not_reload(monkeypatch, fake_category_artifact):
    import src.detection.classifier as classifier

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_artifact should not be called: artifact already set")

    monkeypatch.setattr(classifier, "load_artifact", fail_if_called)
    assert mcp_server._get_category_artifact() is fake_category_artifact
