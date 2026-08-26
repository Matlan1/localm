# SPDX-License-Identifier: AGPL-3.0-or-later
"""The /api/memory surface: it exposes pending corrections and lets the user
accept or reject them. Drives the real route handlers against a real store."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from localm.memory import MemoryRecord, MemoryStore, PendingCorrection
from localm.plugins.builtin.memory import plug


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    return tmp_path


def _store(home) -> MemoryStore:
    return MemoryStore("owner", "chat", root=home / "memory")


def _seed(home):
    s = _store(home)
    target = s.add(MemoryRecord(text="User lives in Berlin", source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=target.id, action="update", proposed_text="User moved to Munich",
        target_text=target.text, confidence=0.9)])
    return s, target


def test_get_includes_pending_corrections(memhome):
    _s, target = _seed(memhome)
    data = asyncio.run(plug.memory_get(None))
    corrs = data["corrections"]
    assert len(corrs) == 1
    c = corrs[0]
    assert c["action"] == "update"
    assert c["proposed_text"] == "User moved to Munich"
    assert c["target_text"] == "User lives in Berlin"
    assert c["target_id"] == target.id
    assert c["target_updated"] == target.updated       # staleness affordance


def test_accept_route_applies_and_clears(memhome):
    s, _target = _seed(memhome)
    cid = s.corrections()[0].id
    out = asyncio.run(plug.memory_correction_accept(cid, None))
    assert out["status"] == "updated"
    fresh = _store(memhome)
    assert [r.text for r in fresh.all()] == ["User moved to Munich"]
    assert fresh.corrections() == []


def test_reject_route_keeps_fact_and_clears(memhome):
    s, _target = _seed(memhome)
    cid = s.corrections()[0].id
    out = asyncio.run(plug.memory_correction_reject(cid, None))
    assert out["status"] == "rejected"
    fresh = _store(memhome)
    assert [r.text for r in fresh.all()] == ["User lives in Berlin"]
    assert fresh.corrections() == []


def test_unknown_correction_id_404(memhome):
    _seed(memhome)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_correction_accept("nope0000nope0000", None))
    assert ei.value.status_code == 404


def test_privacy_mode_get_hides_corrections(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    _seed(tmp_path)                                    # a real pending correction exists
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)   # switch to privacy
    data = asyncio.run(plug.memory_get(None))
    assert data["writable"] is False
    assert data["corrections"] == []                   # not surfaced, sidecar untouched


@pytest.mark.parametrize("route", ["memory_correction_accept", "memory_correction_reject"])
def test_privacy_mode_blocks_resolution(tmp_path, monkeypatch, route):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    s, _target = _seed(tmp_path)
    cid = s.corrections()[0].id
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)   # flip to privacy
    with pytest.raises(HTTPException) as ei:
        asyncio.run(getattr(plug, route)(cid, None))
    assert ei.value.status_code == 403
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    assert len(_store(tmp_path).corrections()) == 1                # untouched


def test_archive_failure_surfaces_as_500(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    s, _target = _seed(tmp_path)
    cid = s.corrections()[0].id
    monkeypatch.setattr(MemoryStore, "_archive_forgotten", lambda self, recs: False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_correction_accept(cid, None))
    assert ei.value.status_code == 500
    assert len(_store(tmp_path).corrections()) == 1                # not applied


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
