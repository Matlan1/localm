"""audit.automatic_write_allowed and the writers migrated onto it."""

from __future__ import annotations

import pytest

from localm import audit, issue_tracker
from localm.memory import gating
from localm.memory.consolidate import run_consolidation
from localm.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_mode_env(monkeypatch):
    monkeypatch.delenv(audit.MODE_ENV_VAR, raising=False)


@pytest.mark.parametrize("mode,expected", [("privacy", False), ("log", True), ("full", True)])
@pytest.mark.parametrize("surface", ["chat", "coder", "server"])
def test_gate_follows_the_mode(monkeypatch, surface, mode, expected):
    monkeypatch.setenv(audit.MODE_ENV_VAR, mode)
    assert audit.automatic_write_allowed(surface) is expected


def test_gate_is_false_when_the_mode_cannot_be_resolved(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unresolvable")
    monkeypatch.setattr(audit, "effective_mode", boom)
    assert audit.automatic_write_allowed("chat") is False


def test_gate_passes_cwd_through(monkeypatch):
    seen = {}

    def fake(surface, cwd=None):
        seen["args"] = (surface, cwd)
        return audit.SessionMode.LOG
    monkeypatch.setattr(audit, "effective_mode", fake)
    assert audit.automatic_write_allowed("coder", cwd="/proj") is True
    assert seen["args"] == ("coder", "/proj")


def test_memory_gate_is_the_audit_gate(monkeypatch):
    monkeypatch.setenv(audit.MODE_ENV_VAR, "privacy")
    assert gating.writes_allowed("chat") is False
    monkeypatch.setenv(audit.MODE_ENV_VAR, "log")
    assert gating.writes_allowed("chat") is True


def test_memory_synthesis_writes_nothing_in_privacy_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(audit.MODE_ENV_VAR, "privacy")
    calls = []
    store = MemoryStore("owner", "chat", root=tmp_path / "mem")
    res = run_consolidation(store, "I like tea", lambda *a, **k: calls.append(1) or "[]")
    assert res["status"] == "skipped"
    assert calls == []
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


@pytest.fixture
def _fresh(monkeypatch):
    monkeypatch.setattr(issue_tracker, "_SESSION_ONLY", [])


def test_issue_record_writes_nothing_in_privacy_mode(monkeypatch, tmp_path, _fresh):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv(audit.MODE_ENV_VAR, "privacy")
    issue_tracker.record_filed(7, "private")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
