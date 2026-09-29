"""
Tests for the Mitigation Manager and Correlation Subagent (src/correlation/).

A scripted fake encoder replaces sentence-transformers, so no model download
or torch is needed: each catalog technique gets its own axis, and a test
decides how close a query sits to each technique.
"""
import json

import numpy as np
import pytest

from src.correlation import subagent as subagent_module
from src.correlation.manager import (
    ACTION_BY_TECHNIQUE,
    NO_MATCH_ACTION,
    NO_THREAT_ACTION,
    MitigationManager,
)
from src.correlation.subagent import (
    CATEGORY_MATCH_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    CorrelationSubagent,
)
from src.correlation.technique_catalog import CATEGORY_TECHNIQUES, PLAYBOOK_PATH, TECHNIQUE_CATALOG
from src.pipeline import Pipeline
from src.response import rules
from src.response.agent import JudgeAgent
from src.shared.schemas import DetectionResult, TrafficEvent
from src.shared.tags import parse_category, strip_category_tag

TECHNIQUE_IDS = list(TECHNIQUE_CATALOG)


def axis(technique_id: str) -> np.ndarray:
    """One axis per technique, plus a last "unrelated" axis no technique uses."""
    v = np.zeros(len(TECHNIQUE_IDS) + 1)
    v[TECHNIQUE_IDS.index(technique_id)] = 1.0
    return v


def near(technique_id: str, score: float) -> np.ndarray:
    """A query vector with cosine similarity `score` to one technique and 0 to the rest."""
    unrelated = np.zeros(len(TECHNIQUE_IDS) + 1)
    unrelated[-1] = 1.0
    return score * axis(technique_id) + np.sqrt(1 - score ** 2) * unrelated


class FakeEncoder:
    """Catalog texts map to one axis each; queries map to whatever the test scripted."""

    def __init__(self, queries=None):
        self.queries = queries or {}
        self.encoded_queries = []
        self._catalog = {text: axis(tid) for tid, text in TECHNIQUE_CATALOG.items()}

    def encode(self, texts):
        if isinstance(texts, list):
            return np.array([self._catalog[t] for t in texts])
        self.encoded_queries.append(texts)
        return self.queries[texts]


class ExplodingEncoder:
    """Fails the test if anything tries to embed."""

    def encode(self, texts):
        raise AssertionError("the embedding model should not be used here")


def detection(anomalous=True, notes=None, confidence=0.95, features=None):
    return DetectionResult(
        event=TrafficEvent(features=features if features is not None else {"Flow Duration": 1.0}),
        is_anomalous=anomalous,
        confidence=confidence,
        detector_notes=notes,
    )


# --- shared tag parsing ---------------------------------------------------------

def test_tag_helpers():
    assert parse_category("[category=DoS] many requests") == "DoS"
    assert parse_category("no tag") is None
    assert strip_category_tag("[category=DoS] many requests") == "many requests"
    assert strip_category_tag("[category=Unknown]") == ""
    assert strip_category_tag(None) == ""


# --- catalog and playbook stay in sync --------------------------------------------

def test_catalog_matches_playbook_techniques():
    with open(PLAYBOOK_PATH, encoding="utf-8") as f:
        playbook = json.load(f)
    assert set(TECHNIQUE_CATALOG) == set(playbook["techniques"])
    for ids in CATEGORY_TECHNIQUES.values():
        assert set(ids) <= set(TECHNIQUE_CATALOG)


def test_every_technique_has_an_action():
    assert set(ACTION_BY_TECHNIQUE) == set(TECHNIQUE_CATALOG)


def test_dos_ddos_and_portscan_are_covered():
    for category in ("DoS", "DDoS", "PortScan"):
        assert CATEGORY_TECHNIQUES[category]


# --- subagent: benign and category paths ---------------------------------------------

def test_benign_traffic_is_not_correlated():
    result = CorrelationSubagent(encoder=ExplodingEncoder()).run(detection(anomalous=False))
    assert result.matched_technique_ids == []
    assert result.trace[0].action == "skip_benign"


@pytest.mark.parametrize("category", sorted(CATEGORY_TECHNIQUES))
def test_category_tag_maps_to_playbook_techniques(category):
    result = CorrelationSubagent(encoder=ExplodingEncoder()).run(detection(notes=f"[category={category}]"))
    assert result.matched_technique_ids == CATEGORY_TECHNIQUES[category]
    assert result.confidence == CATEGORY_MATCH_CONFIDENCE
    assert result.trace[0].action == "lookup_category_techniques"


# --- subagent: embedding fallback ------------------------------------------------------

def test_unknown_with_nothing_to_search_returns_no_match():
    result = CorrelationSubagent(encoder=ExplodingEncoder()).run(detection(notes="[category=Unknown]"))
    assert result.matched_technique_ids == []
    assert result.trace[-1].action == "no_query_text"


def test_confident_first_query_is_accepted():
    notes = "many failed logins"
    encoder = FakeEncoder({notes: near("T1110", 0.8)})
    result = CorrelationSubagent(encoder=encoder).run(detection(notes=f"[category=Unknown] {notes}"))
    assert result.matched_technique_ids == ["T1110"]
    assert result.confidence == pytest.approx(0.8)
    assert encoder.encoded_queries == [notes]  # no retry needed


def test_retry_uses_the_event_and_can_rescue_a_weak_match():
    notes = "odd traffic"
    features = {"Destination Port": 22.0}
    retry = "odd traffic Suspicious traffic to destination port 22 (SSH login service)."
    encoder = FakeEncoder({notes: near("T1566", 0.4), retry: near("T1110", 0.75)})
    result = CorrelationSubagent(encoder=encoder).run(detection(notes=notes, features=features))
    assert result.matched_technique_ids == ["T1110"]
    assert encoder.encoded_queries == [notes, retry]
    assert [s.action for s in result.trace][-2:] == ["query_technique_catalog", "refine_query"]


def test_weak_matches_are_never_accepted():
    """The old retry accepted any score that beat the first one, even below the threshold."""
    notes = "odd traffic"
    retry = "odd traffic Suspicious traffic to destination port 4444 (network service)."
    encoder = FakeEncoder({notes: near("T1566", 0.3), retry: near("T1567", 0.5)})
    result = CorrelationSubagent(encoder=encoder).run(
        detection(notes=notes, features={"Destination Port": 4444.0}))
    assert result.matched_technique_ids == []
    assert result.confidence == pytest.approx(0.5) and 0.5 < CONFIDENCE_THRESHOLD


def test_missing_embedding_model_degrades_to_no_match(monkeypatch):
    def not_installed():
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(subagent_module, "_load_default_encoder", not_installed)
    result = CorrelationSubagent().run(detection(notes="odd traffic"))
    assert result.matched_technique_ids == []
    assert result.trace[-1].action == "embedding_model_unavailable"
    assert "requirements-correlation.txt" in result.correlation_notes


# --- manager -------------------------------------------------------------------------

def test_manager_proposes_the_action_for_the_first_technique():
    manager = MitigationManager(CorrelationSubagent(encoder=ExplodingEncoder()))
    rec = manager.run(detection(notes="[category=DoS]"))
    assert rec.proposed_action == ACTION_BY_TECHNIQUE[CATEGORY_TECHNIQUES["DoS"][0]]
    assert rec.confidence == CATEGORY_MATCH_CONFIDENCE
    assert [s.action for s in rec.trace] == ["delegate_to_subagent", "decide_action"]


def test_manager_actions_when_nothing_matched():
    manager = MitigationManager(CorrelationSubagent(encoder=ExplodingEncoder()))
    assert manager.run(detection(anomalous=False)).proposed_action == NO_THREAT_ACTION
    assert manager.run(detection(notes="[category=Unknown]")).proposed_action == NO_MATCH_ACTION


# --- with the real Judge (rules mode) ---------------------------------------------------

def run_pipeline(det):
    detection_manager = type("D", (), {"run": lambda self, event: det})()
    mitigation = MitigationManager(CorrelationSubagent(encoder=ExplodingEncoder()))
    return Pipeline(detection_manager, JudgeAgent(), mitigation).run(det.event)


def test_pipeline_ddos_is_handled_automatically():
    run = run_pipeline(detection(notes="[category=DDoS]", confidence=0.95))
    assert not run.errors
    assert run.response.case == rules.AGREED_THREAT
    assert run.response.recommended_action == ACTION_BY_TECHNIQUE["T1498"]


def test_pipeline_benign_is_both_benign():
    run = run_pipeline(detection(anomalous=False, confidence=0.95))
    assert run.response.case == rules.BOTH_BENIGN


def test_pipeline_unknown_category_escalates():
    run = run_pipeline(detection(notes="[category=Unknown]", confidence=0.95))
    assert run.response.case == rules.ANOMALOUS_NO_MATCH
    assert run.response.escalated_to_human
