# SPDX-License-Identifier: AGPL-3.0-or-later
"""The memory system must never silently, irrecoverably delete data the user
typed.

Covers three data-loss paths:
- prune's size cap must not evict user-typed facts silently or irreversibly;
- the GUI bulk-PUT must not re-mint every record, which would destroy
  provenance and freeze the store against consolidation;
- a corrupt jobs.json must not load as empty and let the next write erase every
  job.
"""

from __future__ import annotations

import json

import pytest

from localm.memory import MemoryRecord, MemoryStore


NOW = 1_700_000_000.0
DAY = 86400.0


# ------------------------------------------------------ prune: no silent loss #

def test_prune_archives_evicted_user_facts(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(20):
        s.add(MemoryRecord(text=f"user fact {i}", source="user",
                           importance=0.8, last_used=NOW - i * DAY), save=False)
    s._save()
    removed = s.prune(now=NOW, n_max=10)
    assert removed == 10
    assert len(s) == 10
    # Evicted user facts are recoverable, not hard-deleted.
    ff = s.path.with_suffix(".forgotten.jsonl")
    assert ff.is_file()
    archived = [json.loads(ln) for ln in ff.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(archived) == 10
    assert all("forgotten_at" in a for a in archived)
    # And the user-eviction is surfaced for the caller.
    assert len(s.last_evicted_user) == 10


def test_prune_no_eviction_leaves_no_archive(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="a user fact", source="user", importance=0.8,
                       last_used=NOW))
    assert s.prune(now=NOW) == 0
    assert s.last_evicted_user == []
    assert not s.path.with_suffix(".forgotten.jsonl").is_file()


def test_consolidation_surfaces_user_eviction(tmp_path, monkeypatch):
    from localm.memory import run_consolidation
    from localm.memory.store import N_MAX
    monkeypatch.setenv("LOCALM_MODE", "log")
    s = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(N_MAX):
        s.add(MemoryRecord(text=f"durable user fact number {i}", source="user",
                           importance=0.8, last_used=NOW - i * DAY), save=False)
    s._save()

    # One new synth fact tips the store over the cap; prune must evict a user
    # record (weakest first) and the result must SAY so.
    def complete(prompt: str) -> str:
        if "durable" in prompt:
            return json.dumps({"facts": [
                {"fact": "brand new distinct synthesized fact", "confidence": 0.9}]})
        return json.dumps({"decision": "ADD", "confidence": 0.9})

    res = run_consolidation(s, "User: a wholly new topic", complete, now=NOW)
    assert res.get("evicted_user", 0) >= 1
    assert "warning" in res and "evicted" in res["warning"]


# ------------------------------------------------- GUI bulk PUT: provenance #

def _put(store_root, text, monkeypatch, home):
    import asyncio

    import localm.plugins.builtin.memory.plug as plug
    monkeypatch.setattr(plug, "_home", lambda: home)
    monkeypatch.setattr(plug, "_memory_root", lambda: store_root)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setenv("LOCALM_MODE", "log")
    return asyncio.run(plug.memory_put(plug.MemoryUpdate(text=text)))


def test_put_preserves_untouched_record_provenance(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    root = home / "memory"
    # Seed a synth record + a user record directly in the chat store.
    s = MemoryStore("owner", "chat", root=root)
    synth = s.add(MemoryRecord(text="User uses Python daily", source="synth",
                               kind="semantic", importance=0.6, uses=5))
    synth_id, synth_created = synth.id, synth.created

    # The user edits the modal: keeps the synth line verbatim, adds one line.
    _put(root, "User uses Python daily\nUser prefers tabs over spaces",
         monkeypatch, home)

    s2 = MemoryStore("owner", "chat", root=root)
    kept = s2.get(synth_id)
    assert kept is not None, "editing another line destroyed the untouched record"
    assert kept.source == "synth"           # provenance preserved, not re-minted
    assert kept.kind == "semantic"
    assert kept.uses == 5
    assert kept.created == synth_created
    # The genuinely new line is a new user record.
    new = [r for r in s2.all() if r.id != synth_id]
    assert len(new) == 1 and new[0].source == "user"


def test_put_rejects_over_cap(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from localm.memory.store import N_MAX
    home = tmp_path / "home"
    home.mkdir()
    root = home / "memory"
    big = "\n".join(f"user fact {i}" for i in range(N_MAX + 5))
    with pytest.raises(HTTPException) as ei:
        _put(root, big, monkeypatch, home)
    assert ei.value.status_code == 413


def test_append_rejects_at_cap(tmp_path, monkeypatch):
    import asyncio

    from fastapi import HTTPException

    import localm.plugins.builtin.memory.plug as plug
    from localm.memory.store import N_MAX
    home = tmp_path / "home"
    home.mkdir()
    root = home / "memory"
    monkeypatch.setattr(plug, "_home", lambda: home)
    monkeypatch.setattr(plug, "_memory_root", lambda: root)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setenv("LOCALM_MODE", "log")
    s = MemoryStore("owner", "chat", root=root)
    for i in range(N_MAX):
        s.add(MemoryRecord(text=f"fact {i}", source="user"), save=False)
    s._save()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_append(plug.MemoryAppend(text="one too many")))
    assert ei.value.status_code == 413


# ------------------------------------------------- jobs.json: corrupt safety #

def test_corrupt_jobs_json_is_quarantined_not_erased(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    from localm.plugins.builtin.jobs.store import JobStore

    st = JobStore()
    # Write a valid job, then corrupt the file on disk.
    from localm.plugins.builtin.jobs.store import Job
    st.add(Job(name="keep-me", task_kind="chat", prompt="hi",
               schedule_kind="interval", schedule=60))
    defs = st._defs_file
    good = defs.read_text(encoding="utf-8")
    defs.write_text(good[: len(good) // 2] + "  <<truncated>>", encoding="utf-8")

    # A fresh store load must NOT silently return empty-and-overwrite; the
    # corrupt file is backed up first.
    st2 = JobStore()
    assert st2.list() == []                    # starts empty (corrupt)
    backups = list(tmp_path.rglob("*.corrupt-*"))
    assert backups, "corrupt jobs file was not backed up before being replaced"
    # The backup holds the original bytes (recoverable).
    assert "keep-me" in backups[0].read_text(encoding="utf-8")


def test_absent_jobs_json_is_simply_empty(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    from localm.plugins.builtin.jobs.store import JobStore
    # No file yet: empty, and NO spurious corrupt backup.
    assert JobStore().list() == []
    assert not list(tmp_path.rglob("*.corrupt-*"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
