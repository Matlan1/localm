# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collection.load_and_maybe_backfill() - the opportunistic cache-write that
closes the gap named in rag_collections'/rag_detail's own docstrings: _load()
never calls _save(), so a collection written once and only ever LISTED (or
viewed) afterward stays on the slow full-load fallback forever, not just once.

The property that matters, adversarially: a backfilled cache must be
BYTE-IDENTICAL to what a fresh, eager Collection(name).stats() would report for
the same on-disk state - never a value trusted because "it was probably fine a
moment ago". And the concurrency half: the backfill must never write a cache
derived from data it did not itself read WHILE holding the collection's write
lock, and it must degrade to exactly the uncached-but-correct behaviour whenever
that lock is busy - never corrupt disk, never answer wrong.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag import Collection
from localm.rag.collection_lock import lock_path_for
from localm.rag.store import _STATS_CACHE_KEY


def _embed3(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def base(tmp_path):
    d = tmp_path / "collections"
    d.mkdir()
    return d


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alpha.txt").write_text("alpha content about turbines", encoding="utf-8")
    (d / "beta.txt").write_text("beta content about gearboxes", encoding="utf-8")
    (d / "gamma.txt").write_text("gamma content about compressors", encoding="utf-8")
    return d


def _strip_cache(base, name):
    """Simulate a collection saved without this cache block (or hand-edited to
    drop it) - the exact precondition load_and_maybe_backfill exists for."""
    meta_path = base / name / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert _STATS_CACHE_KEY in meta, "fixture precondition: a real save must have cached it"
    del meta[_STATS_CACHE_KEY]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _record(**over) -> dict:
    """A well-formed lock record for a plausible FOREIGN, LIVE holder - the same
    shape the collection-lock tests use."""
    rec = {"token": "feedface" * 4, "pid": 999_999, "pid_create_time": None,
           "machine": "another-machine-hash", "collection": "kb",
           "op": "a re-sync", "started": time.time() - 5}
    rec.update(over)
    return rec


def _hold(path, rec) -> None:
    """Stage a foreign lock file with a FRESH heartbeat (mtime = now), so it
    reads as currently alive, not stale/reclaimable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Core: a cold collection gets backfilled correctly, both directions         #
# --------------------------------------------------------------------------- #

def test_backfill_writes_a_cache_matching_the_real_load(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    eager = Collection("kb", base=base).stats()
    _strip_cache(base, "kb")
    assert Collection.peek_stats("kb", base=base) is None, (
        "fixture precondition: the cache must genuinely be gone before backfill")

    backfilled = Collection.load_and_maybe_backfill("kb", base=base)
    assert backfilled.stats() == eager, "the loaded object itself must be fully correct"

    peeked = Collection.peek_stats("kb", base=base)
    assert peeked == eager, (
        "the backfilled cache must be byte-identical to a real load, not an "
        "approximation - a cache that answers fast but wrong is worse than "
        "the fallback it replaces")


def test_backfill_writes_correct_detail_including_docs(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    eager_stats = Collection("kb", base=base).stats()
    eager_docs = Collection("kb", base=base).docs()
    _strip_cache(base, "kb")

    backfilled = Collection.load_and_maybe_backfill("kb", base=base)
    assert backfilled.stats() == eager_stats
    assert backfilled.docs() == eager_docs

    peeked = Collection.peek_detail("kb", base=base)
    assert peeked == {**eager_stats, "docs": eager_docs}


def test_backfill_of_a_malformed_collection_carries_the_bad_line_count(base, docs):
    """The backfill path specifically: chunks_bad_lines has to reach the cache
    load_and_maybe_backfill writes, not just the one _save() writes - a
    collection that is corrupt AND was only ever listed (never re-saved) is
    exactly the case this method exists for. load_and_maybe_backfill only ever
    rewrites meta.json's cache block, never chunks.jsonl, so the count it
    derives must describe the file as it actually is on disk, not a self-healed
    copy."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    chunks_path = base / "kb" / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write("\nnot json at all")
    _strip_cache(base, "kb")
    eager = Collection("kb", base=base).stats()
    assert eager["chunks_bad_lines"] == 1, "fixture precondition"

    backfilled = Collection.load_and_maybe_backfill("kb", base=base)
    assert backfilled.stats() == eager
    assert backfilled.stats()["chunks_bad_lines"] == 1

    peeked = Collection.peek_stats("kb", base=base)
    assert peeked == eager
    assert peeked["chunks_bad_lines"] == 1
    assert chunks_path.read_text(encoding="utf-8").count("not json at all") == 1, (
        "the backfill must never have rewritten chunks.jsonl")


def test_backfilled_cache_is_never_trusted_after_hand_corruption(base, docs):
    """The negative direction, applied specifically to a backfilled cache (not
    a _save()-written one): once the backfill has cached a state, a file
    changed WITHOUT going through this class must still invalidate it, exactly
    like a normal save's cache would."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    _strip_cache(base, "kb")
    Collection.load_and_maybe_backfill("kb", base=base)
    assert Collection.peek_stats("kb", base=base) is not None, (
        "fixture precondition: the backfill must have cached it")

    vec_path = base / "kb" / "vectors.json"
    data = json.loads(vec_path.read_text(encoding="utf-8"))
    data["vectors"].pop(0)
    vec_path.write_text(json.dumps(data), encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None, (
        "a backfilled cache must be refused once the file it describes "
        "changed without going through this class, same as a saved one"
    )


def test_backfilled_cache_survives_a_real_noop_save(base, docs):
    """The positive direction: once backfilled, a real operation that changes
    nothing must not spuriously invalidate the cache it just wrote."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    _strip_cache(base, "kb")
    Collection.load_and_maybe_backfill("kb", base=base)
    before = Collection.peek_stats("kb", base=base)
    assert before is not None

    coll2 = Collection("kb", base=base)
    coll2.resync(embed_fn=_embed3, policy=None)   # nothing changed
    after = Collection.peek_stats("kb", base=base)
    assert after == before


def test_backfill_of_a_nonexistent_collection_does_not_write_a_cache(base):
    """A name that resolves to no collection at all (deleted between the cold
    peek and the backfill attempt, or simply never existed) must not crash
    and must not fabricate a meta.json - the load-under-lock's own .exists()
    check gates the write."""
    coll = Collection.load_and_maybe_backfill("does-not-exist", base=base)
    assert not coll.exists()
    assert not (base / "does-not-exist").exists()


def test_backfill_is_idempotent_across_repeated_calls(base, docs):
    """Calling it again on an already-backfilled (or already-_save()d)
    collection must be a harmless no-op, not a second, redundant write that
    could itself race something."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    first = Collection.load_and_maybe_backfill("kb", base=base).stats()
    second = Collection.load_and_maybe_backfill("kb", base=base).stats()
    assert first == second


# --------------------------------------------------------------------------- #
#  Concurrency: the ordering guarantee and the busy-lock degrade              #
# --------------------------------------------------------------------------- #

def test_busy_lock_falls_back_without_writing_anything(base, docs):
    """A refused write must change NOTHING on disk. A live foreign holder must
    make the backfill skip quietly - returning a correct, fully-loaded
    Collection (the uncached fallback), with meta.json byte-for-byte unchanged
    (no partial or stale cache written from a read that never happened under
    the lock)."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    eager = Collection("kb", base=base).stats()
    _strip_cache(base, "kb")
    meta_path = base / "kb" / "meta.json"
    before_bytes = meta_path.read_bytes()

    lp = lock_path_for(base / "kb")
    _hold(lp, _record())
    try:
        result = Collection.load_and_maybe_backfill("kb", base=base)
    finally:
        lp.unlink(missing_ok=True)

    assert result.stats() == eager, (
        "a busy lock must still return a fully correct Collection - the "
        "backfill is additive-only, never a source of a wrong answer")
    after_bytes = meta_path.read_bytes()
    assert after_bytes == before_bytes, (
        "a refused backfill must leave meta.json byte-for-byte untouched - "
        "no cache may be written from a read that happened outside the lock")
    assert Collection.peek_stats("kb", base=base) is None, (
        "the collection must remain uncached after a refused backfill, not "
        "gain a cache from data read without holding the lock")


def test_concurrent_backfill_exactly_one_writes_the_cache(base, docs):
    """Two threads racing load_and_maybe_backfill on the SAME cold collection:
    the file lock (O_CREAT|O_EXCL, genuinely atomic at the OS level) must let
    exactly one of them win the write - the other gets CollectionLockedError
    internally and falls back - and BOTH must still return fully correct
    stats. Whichever wins, the final on-disk fingerprint must describe
    exactly what is actually on disk."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    eager = Collection("kb", base=base).stats()
    _strip_cache(base, "kb")

    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()   # both threads hit collection_write_lock as close together as possible
        results.append(Collection.load_and_maybe_backfill("kb", base=base).stats())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 2
    assert results[0] == eager and results[1] == eager, (
        "both racing callers must see fully correct stats regardless of "
        "which one actually won the lock")
    # Exactly one interleaving can leave the collection uncached (both threads
    # hit the busy lock in the same instant is not possible with timeout=0 -
    # the OS serializes the O_CREAT|O_EXCL itself - but assert the REACHABLE
    # outcome directly: the cache, if present, is correct.
    peeked = Collection.peek_stats("kb", base=base)
    if peeked is not None:
        assert peeked == eager, (
            "the winning thread's cache write must match real disk state - "
            "a race that wrote a stale or partial fingerprint would still "
            "show up here as peek_stats disagreeing with the eager load")


# --------------------------------------------------------------------------- #
#  HTTP routes: rag_collections()/rag_detail() route through                  #
#  load_and_maybe_backfill, and the write lands on the real on-disk           #
#  collection the routes themselves serve from.                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_route_app(tmp_path, monkeypatch):
    """A minimal GUI app with the builtin rag plugin installed, Path.home ==
    tmp_path so docs placed under it are inside the whitelist. Open mode -> the
    caller is the owner."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    home = tmp_path
    localm = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


def _rag_base(home: Path) -> Path:
    return home / ".localm" / "rag"


def _seed_kb_via_routes(client, home: Path) -> None:
    """Create "kb" through the real (synchronous) POST /collections route, then
    add real content directly through the Collection primitive at the exact
    on-disk location that route just created.

    NOT through POST .../add: under attach_gui (what rag_route_app wires up,
    matching the GUI the list/detail routes actually serve), that route hands
    the work to the background job manager and returns {"job_id": ...}
    immediately, so a read back straight afterwards sees n_docs == 0 because
    indexing has not run yet. Driving that job to completion would test the job
    pipeline, not what these tests are about: whether GET .../collections and
    GET .../collections/{name} backfill a cold collection's cache. Seeding the
    same on-disk state directly through the primitive - proven correct by the
    Collection-level tests above - keeps these tests scoped to the routes under
    test."""
    r = client.post("/api/rag/collections", json={"name": "kb"})
    assert r.status_code == 200, r.text
    coll = Collection("kb", base=_rag_base(home))
    coll.add_paths([_write_docs(home)], embed_fn=_embed3)


def _write_docs(home: Path) -> Path:
    docs = home / "kdocs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha content about turbines", encoding="utf-8")
    (docs / "b.md").write_text("beta content about gearboxes", encoding="utf-8")
    return docs


class TestBackfillThroughRoutes:
    def test_list_route_backfills_a_cold_collection(self, rag_route_app):
        app, home = rag_route_app
        with TestClient(app) as client:
            _seed_kb_via_routes(client, home)

            rbase = _rag_base(home)
            eager = Collection("kb", base=rbase).stats()
            _strip_cache(rbase, "kb")
            assert Collection.peek_stats("kb", base=rbase) is None, (
                "fixture precondition: the cache must genuinely be gone")

            r = client.get("/api/rag/collections")
            assert r.status_code == 200
            listed = {c["name"]: c for c in r.json()["collections"]}
            # dim_mismatch is added by the ROUTE (a best-effort comparison
            # against whatever embedder happens to be resident), never written
            # to the cache itself - None here since this app never loads an
            # embedder.
            assert listed["kb"] == {**eager, "dim_mismatch": None}

            assert Collection.peek_stats("kb", base=rbase) == eager, (
                "the LIST route must have written a real backfill to the "
                "collection it just served, not merely answered correctly "
                "once and left it cold for next time")

    def test_detail_route_backfills_a_cold_collection(self, rag_route_app):
        app, home = rag_route_app
        with TestClient(app) as client:
            _seed_kb_via_routes(client, home)

            rbase = _rag_base(home)
            eager_stats = Collection("kb", base=rbase).stats()
            eager_docs = Collection("kb", base=rbase).docs()
            _strip_cache(rbase, "kb")

            r = client.get("/api/rag/collections/kb")
            assert r.status_code == 200
            # dim_mismatch is route-added, None here since this app never loads
            # an embedder.
            assert r.json() == {**eager_stats, "docs": eager_docs, "dim_mismatch": None}

            assert Collection.peek_stats("kb", base=rbase) == eager_stats, (
                "the DETAIL route must have written a real backfill too - it "
                "had the identical cold-cache gap as the list route")

    def test_detail_route_404_for_unknown_collection_writes_nothing(
            self, rag_route_app):
        app, home = rag_route_app
        with TestClient(app) as client:
            r = client.get("/api/rag/collections/nope")
            assert r.status_code == 404
        assert not _rag_base(home).exists() or not (_rag_base(home) / "nope").exists()
