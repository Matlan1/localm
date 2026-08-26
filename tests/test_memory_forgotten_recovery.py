# SPDX-License-Identifier: AGPL-3.0-or-later
"""`_archive_forgotten()` writes evicted and superseded records to a recoverable
`.forgotten.jsonl` sidecar. This covers the read-back half:
`MemoryStore.forgotten()` / `restore_forgotten()` and the
GET/POST /api/memory/forgotten[...] routes built on them.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import HTTPException

from localm.memory import MemoryRecord, MemoryStore, PendingCorrection
from localm.plugins.builtin.memory import plug

NOW = 1_700_000_000.0
DAY = 86400.0


@pytest.fixture(autouse=True)
def _allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


def _evict_via_prune(tmp_path, principal="owner"):
    """Seed a store with enough user facts that prune's size cap evicts one,
    producing a real `.forgotten.jsonl` entry via the normal path (not a
    hand-crafted sidecar)."""
    s = MemoryStore(principal, "chat", root=tmp_path)
    for i in range(20):
        s.add(MemoryRecord(text=f"user fact {i}", source="user", importance=0.8,
                           last_used=NOW - i * DAY), save=False)
    s._save()
    removed = s.prune(now=NOW, n_max=19)
    assert removed == 1
    return s


# --------------------------------------------------------------------- #
#  MemoryStore.forgotten() / restore_forgotten()                         #
# --------------------------------------------------------------------- #

def test_forgotten_lists_archived_records(tmp_path):
    s = _evict_via_prune(tmp_path)
    items = s.forgotten()
    assert len(items) == 1
    assert items[0]["text"] == "user fact 19"     # oldest last_used -> weakest -> evicted
    assert items[0]["source"] == "user"
    assert "forgotten_at" in items[0]


def test_restore_forgotten_reappears_in_live_store(tmp_path):
    s = _evict_via_prune(tmp_path)
    forgotten_id = s.forgotten()[0]["id"]
    assert s.get(forgotten_id) is None            # confirm it is really gone first

    restored = s.restore_forgotten(forgotten_id)
    assert restored is not None
    assert restored.id == forgotten_id
    assert restored.text == "user fact 19"
    assert restored.source == "user"

    # Reload from disk: the restore was actually persisted, not just in-memory.
    fresh = MemoryStore("owner", "chat", root=tmp_path)
    got = fresh.get(forgotten_id)
    assert got is not None and got.text == "user fact 19"
    # Restored entry is consumed from the archive (not left to be re-restored/
    # double-counted) - the archive is a recovery queue, not an audit log.
    assert fresh.forgotten() == []


def test_restore_twice_is_a_noop_second_time(tmp_path):
    s = _evict_via_prune(tmp_path)
    forgotten_id = s.forgotten()[0]["id"]
    assert s.restore_forgotten(forgotten_id) is not None
    # Already live and the archive is now empty for this id, so a second call
    # finds no remaining entry to apply - not a duplicate re-add.
    assert s.restore_forgotten(forgotten_id) is None
    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert len([r for r in fresh.all() if r.id == forgotten_id]) == 1


def test_restore_unknown_id_returns_none(tmp_path):
    s = _evict_via_prune(tmp_path)
    assert s.restore_forgotten("deadbeefdeadbeef") is None


def test_restore_scoped_to_own_namespace(tmp_path):
    """Two different principals share the same root; a record forgotten in
    namespace A must not be visible or restorable from namespace B."""
    a = _evict_via_prune(tmp_path, principal="alice")
    forgotten_id = a.forgotten()[0]["id"]

    b = MemoryStore("bob", "chat", root=tmp_path)
    assert b.forgotten() == []                     # bob's archive is untouched
    assert b.restore_forgotten(forgotten_id) is None   # no matching entry in bob's ns
    assert b.get(forgotten_id) is None              # definitely not leaked into bob's store

    # Alice's own namespace is unaffected by bob's failed lookup.
    assert len(a.forgotten()) == 1


# --------------------------------------------------------------------- #
#  Restoring a SUPERSEDED (id still live) archive entry                  #
# --------------------------------------------------------------------- #

def _accept_update(s, old_text, new_text):
    """Propose+accept an UPDATE correction on the live record whose text is
    *old_text* (the real path resolve_correction takes: archive the pre-change
    snapshot under the SAME id, then mutate the live record in place - the record
    never actually leaves the live store, unlike a prune eviction). Adds the
    record first if it does not exist yet."""
    target = next((r for r in s.all() if r.text == old_text), None)
    if target is None:
        target = s.add(MemoryRecord(text=old_text, source="user"))
    s.propose_corrections([PendingCorrection(
        target_id=target.id, action="update", proposed_text=new_text,
        target_text=target.text, confidence=0.9)])
    cid = s.corrections()[0].id
    out = s.resolve_correction(cid, accept=True)
    assert out["status"] == "updated"
    return target.id


def test_restore_reverts_a_superseded_update_correction_in_place(tmp_path):
    """resolve_correction's accept branch archives the OLD text under the SAME
    id it keeps live under the NEW text, so the id never frees up and
    `self.get(mem_id) is not None` stays true. Restoring must still revert the
    live record rather than treating the collision as "already restored"."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    rid = _accept_update(s, "User lives in Berlin", "User moved to Munich")

    # The live record already has the NEW text; the archive holds the OLD one
    # under the identical id - not a fresh, unclaimed id like a prune eviction.
    assert s.get(rid).text == "User moved to Munich"
    items = s.forgotten()
    assert len(items) == 1 and items[0]["id"] == rid and items[0]["text"] == "User lives in Berlin"

    restored = s.restore_forgotten(rid)
    assert restored is not None
    assert restored.id == rid                       # same id, reverted in place
    assert restored.text == "User lives in Berlin"   # NOT a 404, NOT a no-op

    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert [r.text for r in fresh.all()] == ["User lives in Berlin"]
    assert len(fresh.all()) == 1                     # reverted, not duplicated
    assert fresh.forgotten() == []                   # consumed from the archive


def test_restore_steps_back_through_repeated_supersessions(tmp_path):
    """A record forgotten more than once under the same id (two accepted
    corrections in a row) restores the MOST RECENT snapshot first, then the
    one before it - stepping back through history, not restoring an
    arbitrary/ambiguous entry."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    rid = _accept_update(s, "User lives in Berlin", "User moved to Munich")
    rid2 = _accept_update(s, "User moved to Munich", "User moved to Ghent")
    assert rid == rid2                               # same record, same id throughout
    assert s.get(rid).text == "User moved to Ghent"
    assert [it["text"] for it in s.forgotten()] == ["User moved to Munich",
                                                     "User lives in Berlin"]

    first = s.restore_forgotten(rid)
    assert first.text == "User moved to Munich"       # most recent supersession undone
    assert s.get(rid).text == "User moved to Munich"
    assert [it["text"] for it in s.forgotten()] == ["User lives in Berlin"]

    second = s.restore_forgotten(rid)
    assert second.text == "User lives in Berlin"       # steps back one more
    assert s.get(rid).text == "User lives in Berlin"
    assert s.forgotten() == []

    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert [r.text for r in fresh.all()] == ["User lives in Berlin"]


def test_restore_recomputes_vector_with_embed_fn(tmp_path):
    """Like resolve_correction's own accept path (which re-embeds the new text),
    a restored or reverted record regains a semantic vector rather than keeping
    a stale or absent one."""
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    s = _evict_via_prune(tmp_path)
    forgotten_id = s.forgotten()[0]["id"]
    restored = s.restore_forgotten(forgotten_id, embed_fn=fake_embed)
    assert restored is not None
    assert any("user fact 19" in call for call in calls)
    assert s._vectors.get(forgotten_id) == [1.0, 0.0, 0.0]

    fresh = MemoryStore("owner", "chat", root=tmp_path)
    assert fresh._vectors.get(forgotten_id) == [1.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
#  _load_forgotten() robustness: corrupt-line handling                   #
# --------------------------------------------------------------------- #

def test_corrupt_forgotten_line_skipped_with_warning(tmp_path, caplog):
    s = _evict_via_prune(tmp_path)
    with open(s._forgotten_file(), "a", encoding="utf-8") as fh:
        fh.write("not json at all\n\n")
    with caplog.at_level(logging.WARNING, logger="localm"):
        items = s.forgotten()
    assert len(items) == 1                          # the one good entry survives
    assert "skipped 1 unparseable line" in caplog.text


def test_non_object_forgotten_line_skipped_with_warning(tmp_path, caplog):
    s = _evict_via_prune(tmp_path)
    with open(s._forgotten_file(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps([1, 2, 3]) + "\n")
    with caplog.at_level(logging.WARNING, logger="localm"):
        items = s.forgotten()
    assert len(items) == 1
    assert "skipped 1 unparseable line" in caplog.text


# --------------------------------------------------------------------- #
#  HTTP routes                                                           #
# --------------------------------------------------------------------- #

@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    return tmp_path


def test_route_lists_forgotten(memhome):
    _evict_via_prune(memhome / "memory")
    data = asyncio.run(plug.memory_forgotten(None))
    assert len(data["items"]) == 1
    assert data["items"][0]["text"] == "user fact 19"


def test_route_restores_and_reappears_via_get(memhome):
    s = _evict_via_prune(memhome / "memory")
    forgotten_id = s.forgotten()[0]["id"]
    out = asyncio.run(plug.memory_forgotten_restore(forgotten_id, None))
    assert out["status"] == "restored"
    assert out["item"]["id"] == forgotten_id

    data = asyncio.run(plug.memory_get(None))
    assert forgotten_id in [it["id"] for it in data["items"]]
    forgotten_after = asyncio.run(plug.memory_forgotten(None))
    assert forgotten_after["items"] == []


def test_route_restore_unknown_id_404(memhome):
    _evict_via_prune(memhome / "memory")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_forgotten_restore("deadbeefdeadbeef", None))
    assert ei.value.status_code == 404


def test_route_restore_blocked_in_privacy_mode(memhome, monkeypatch):
    s = _evict_via_prune(memhome / "memory")
    forgotten_id = s.forgotten()[0]["id"]
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_forgotten_restore(forgotten_id, None))
    assert ei.value.status_code == 403
    # Untouched: the archive entry is still there, nothing was restored.
    fresh = MemoryStore("owner", "chat", root=memhome / "memory")
    assert len(fresh.forgotten()) == 1


def test_route_lists_forgotten_in_privacy_mode(memhome, monkeypatch):
    """Listing is read-only (no new trace), so it stays available in privacy
    mode even though restoring (a write) is blocked."""
    _evict_via_prune(memhome / "memory")
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
    data = asyncio.run(plug.memory_forgotten(None))
    assert len(data["items"]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
