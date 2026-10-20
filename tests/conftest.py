"""
Shared pytest setup.

Tests must not depend on the developer's shell or .env: a leftover
MITIGATION_MODE=agent (from a real-model run) would switch them onto the real
LLM. This fixture clears it before every test; tests that need agent mode set
it themselves.
"""
import pytest


@pytest.fixture(autouse=True)
def _mitigation_rules_mode_by_default(monkeypatch):
    monkeypatch.delenv("MITIGATION_MODE", raising=False)