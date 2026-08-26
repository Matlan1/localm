# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collection.peek_stats() / peek_detail() - the lazy, meta.json-only read
path rag_collections/rag_detail use to avoid materialising chunks.jsonl and
vectors.json just to report counts, which on a real tree takes seconds and
freezes the single-worker event loop for every other in-flight request the
whole time.

The one property that actually matters here, adversarially: peek_stats()
must NEVER report a value that disagrees with what the real, full
``Collection(name).stats()`` would say for the SAME on-disk state. A cache
that answers fast but wrong is worse than the freeze it replaces, because a
freeze is at least visibly slow - a wrong ``has_vectors`` or a silently
stale ``vector_degrade_reason`` looks like a healthy collection. So every
case below builds a collection through the REAL API
(create/add_paths/hand-corrupt-on-disk), then asserts
``peek_stats()``/``peek_detail()`` are BYTE-IDENTICAL to the eager
``stats()``/``{**stats(), "docs": docs()}`` - never a looser "close enough"
comparison.

Also pins the None contract: peek_stats()/peek_detail() must return None
(never a wrong dict) for anything they cannot answer cheaply and correctly -
a collection saved without the cache block, a corrupt meta.json, or a
name that does not exist - so the caller's fallback to the real, full load
is exercised, not skipped.
"""

from __future__ import annotations

import json

import pytest

from localm.rag import Collection
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


def _assert_peek_matches_eager(base, name):
    """The core adversarial assertion: reload the collection FRESH (a brand
    new Collection instance, exactly what a real request does) and compare
    its authoritative stats()/docs() against the lazy peek path, byte for
    byte."""
    eager = Collection(name, base=base)
    eager_stats = eager.stats()
    eager_detail = {**eager_stats, "docs": eager.docs()}

    peeked_stats = Collection.peek_stats(name, base=base)
    peeked_detail = Collection.peek_detail(name, base=base)

    assert peeked_stats == eager_stats, (
        f"peek_stats() disagrees with the real stats():\n"
        f"  eager:  {eager_stats}\n  peeked: {peeked_stats}")
    assert peeked_detail == eager_detail, (
        f"peek_detail() disagrees with the real detail:\n"
        f"  eager:  {eager_detail}\n  peeked: {peeked_detail}")
    return eager_stats


# --------------------------------------------------------------------------- #
#  Byte-identical across every has_vectors / vector_degrade_reason shape      #
# --------------------------------------------------------------------------- #

def test_freshly_created_empty_collection(base):
    Collection("kb", base=base).create()
    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_chunks"] == 0
    assert stats["has_vectors"] is False
    assert stats["corrupt"] is False
    assert stats["chunks_bad_lines"] == 0
    assert stats["vector_degrade_reason"] is None


def test_fully_embedded_collection_has_vectors_true(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_chunks"] == 3
    assert stats["has_vectors"] is True
    assert stats["vector_degrade_reason"] is None


def test_no_embedder_has_vectors_false_no_degrade(base, docs):
    """No embed_fn at all is a legitimate, undegraded state - lexical-only by
    choice, not a fault. has_vectors must be False with NO reason (None), not
    conflated with an actual degrade: distinct causes, distinct signals."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs])
    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_chunks"] == 3
    assert stats["has_vectors"] is False
    assert stats["vector_degrade_reason"] is None


def _flaky_embed(fail_on):
    """A real embed_fn (goes through add_paths' own finiteness/shape checks,
    not a hand-edited file) that raises for any batch containing one of the
    *fail_on* substrings - reproduces genuine partial coverage the way a
    real, spotty embedder would, so the resulting vectors.json is exactly
    what a legitimate _save() writes, not a simulated-corruption fixture.

    IMPORTANT ordering note: a single raised exception sets an internal
    "embed_broken" flag for the REST of that add_paths() call - every file
    processed AFTER the failing one also gets no vector, not just the one
    that raised (an embedder outage is terminal for the batch, not retried
    per file). So *fail_on* must target the LAST file in iteration order
    (alphabetical) to get partial-but-nonzero coverage; targeting an earlier
    file zeroes everything after it too."""
    def embed(texts):
        if any(any(f in t for f in fail_on) for t in texts):
            raise RuntimeError("embedder unavailable for this batch")
        return [[1.0, 0.0, 0.0] for _ in texts]
    return embed


def _vectors_present_count(base, name) -> int:
    vec_path = base / name / "vectors.json"
    if not vec_path.is_file():
        return 0
    data = json.loads(vec_path.read_text(encoding="utf-8"))
    return sum(1 for v in data["vectors"] if v)


def test_partial_coverage_below_threshold_has_vectors_false(base, docs):
    """Below the 80% coverage threshold: has_vectors must read False even
    though SOME vectors exist - this is the exact boundary a naive re-
    derivation could get subtly wrong (e.g. `bool(n_vectors)` instead of the
    real coverage ratio). Reproduced through the REAL embed pipeline (the
    LAST of three files - alpha, beta, gamma, processed in that order - fails
    to embed, exactly like a flaky real embedder), not by hand-editing
    vectors.json after the fact - see the module docstring for why that
    matters: a hand-edit after _save() is indistinguishable from external
    tampering and is covered separately below."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_flaky_embed(["compressors"]))  # gamma (last) fails
    assert _vectors_present_count(base, "kb") == 2, (
        "fixture check: alpha+beta must have embedded, gamma must not have")

    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_chunks"] == 3
    assert stats["has_vectors"] is False   # 2/3 = 67% < 80%


def test_partial_coverage_at_threshold_has_vectors_true(base, tmp_path):
    """>=80% coverage (not "every chunk") is the real has_vectors contract -
    a partially-embedded collection still does hybrid retrieval, so this
    must read True, not under-report a working collection as lexical-only.
    Reproduced through the real embed pipeline (the LAST of five files fails,
    per the ordering note on _flaky_embed)."""
    d = tmp_path / "docs5"
    d.mkdir()
    for i in range(5):
        (d / f"doc{i}.txt").write_text(f"document number {i} about widgets",
                                       encoding="utf-8")
    base2 = tmp_path / "collections2"
    base2.mkdir()
    coll = Collection("kb", base=base2).create()
    coll.add_paths([d], embed_fn=_flaky_embed(["number 4"]))  # doc4 (last) fails
    assert _vectors_present_count(base2, "kb") == 4, (
        "fixture check: doc0-doc3 must have embedded, doc4 must not have")

    stats = _assert_peek_matches_eager(base2, "kb")
    assert stats["n_chunks"] == 5
    assert stats["has_vectors"] is True   # 4/5 = 80%, exactly at threshold


# --------------------------------------------------------------------------- #
#  Files changed WITHOUT going through _save(): the cache must refuse,        #
#  never answer wrong. A real _save() cannot itself produce a chunk or        #
#  vector count mismatch or a non-finite value on disk, so these states       #
#  are reachable only by a hand-edit, an interrupted write, or external       #
#  tampering - what the fingerprint in _save()'s cache block detects.         #
# --------------------------------------------------------------------------- #

def test_hand_corrupted_vectors_after_save_is_never_trusted(base, docs):
    """A structurally-valid vectors.json whose length no longer matches the
    chunk count, written AFTER _save() already cached a healthy state (the
    shape an interrupted/partial embed leaves behind): the fingerprint must no
    longer match, so peek_stats()/peek_detail() return None - NOT the stale
    cached "healthy" answer. The fallback (the real, full stats()) must still
    get it right."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    assert Collection.peek_stats("kb", base=base)["has_vectors"] is True, (
        "precondition: the cache must be trusted before the hand-edit")

    vec_path = base / "kb" / "vectors.json"
    data = json.loads(vec_path.read_text(encoding="utf-8"))
    data["vectors"].pop(0)   # now 2 vectors for 3 chunks: mismatched
    vec_path.write_text(json.dumps(data), encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None, (
        "the cache must be refused once the file it describes changed "
        "without going through _save()")
    assert Collection.peek_detail("kb", base=base) is None
    fallback = Collection("kb", base=base).stats()
    assert fallback["has_vectors"] is False
    assert fallback["vector_degrade_reason"], "a mismatched vectors.json must be explained"


def test_hand_corrupted_non_finite_vectors_after_save_is_never_trusted(base, docs):
    """Same shape as above for NaN/inf: add_paths itself would never write a
    non-finite value (it validates before storing - see the write pipeline
    around _vectors_finite in add_paths), so this can only be reached by a
    hand-edit / external write, and the cache must refuse rather than answer
    from its now-stale "healthy" snapshot."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    assert Collection.peek_stats("kb", base=base)["has_vectors"] is True

    vec_path = base / "kb" / "vectors.json"
    data = json.loads(vec_path.read_text(encoding="utf-8"))
    data["vectors"][0] = [float("nan"), 0.0, 0.0]
    vec_path.write_text(json.dumps(data), encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None
    assert Collection.peek_detail("kb", base=base) is None
    fallback = Collection("kb", base=base).stats()
    assert fallback["has_vectors"] is False
    assert "non-finite" in (fallback["vector_degrade_reason"] or "")


def test_hand_edited_chunks_jsonl_after_save_is_never_trusted(base, docs):
    """The fingerprint covers BOTH files independently - editing chunks.jsonl
    alone (leaving vectors.json untouched) must also invalidate the cache,
    not just a vectors.json edit. Appends a chunk by hand, so n_chunks itself
    would be wrong if the stale cache were trusted."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    assert Collection.peek_stats("kb", base=base)["n_chunks"] == 3

    # dumps_lines() (what _save() writes with) joins records with a newline and
    # adds none at the end, so the append must supply the separator itself or it
    # would concatenate onto the last real record.
    chunks_path = base / "kb" / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps({"source": "hand-added", "pos": 0, "text": "smuggled"}))

    assert Collection.peek_stats("kb", base=base) is None
    assert Collection("kb", base=base).stats()["n_chunks"] == 4


def test_malformed_chunks_jsonl_count_surfaces_in_stats(base, docs):
    """The malformed-line COUNT must reach stats() (not just the boolean
    'corrupt'), so a caller can say "62 malformed chunk line(s)" instead of a
    generic "index damaged". Two bad lines hand-appended: one that is not valid
    JSON at all, one that is valid JSON but the wrong shape (no "text" field) -
    both are the two ways _load() counts a line as bad."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)   # 3 real chunks

    chunks_path = base / "kb" / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write("\nnot json at all")
        f.write("\n" + json.dumps({"source": "x", "pos": 0}))   # missing "text"

    reloaded = Collection("kb", base=base)
    stats = reloaded.stats()
    assert stats["chunks_bad_lines"] == 2
    assert stats["corrupt"] is True
    assert stats["n_chunks"] == 3, "malformed lines are skipped, never counted as chunks"

    # The count also survives into the cache load_and_maybe_backfill writes: it
    # only writes meta.json's cache block and never rewrites chunks.jsonl, so
    # this reads the count from the SAME still-malformed file.
    backfilled = Collection.load_and_maybe_backfill("kb", base=base)
    assert backfilled.stats() == stats
    assert backfilled.stats()["chunks_bad_lines"] == 2
    peeked = Collection.peek_stats("kb", base=base)
    assert peeked == stats, (
        "the backfilled cache must be byte-identical to the eager reload")
    assert peeked["chunks_bad_lines"] == 2

    # A real _save() (unlike the backfill above) rewrites chunks.jsonl from
    # self._chunks, which never held the malformed lines - so it self-heals
    # the file and the count correctly drops to 0.
    reloaded._save()
    assert reloaded.stats()["chunks_bad_lines"] == 0, (
        "_save() rewrites chunks.jsonl from the valid chunks only, so THIS "
        "instance's own corrupt/chunks_bad_lines must clear immediately - "
        "not just on a later fresh reload. Caught live: a repair "
        "(add_paths(force=True) -> this same _save()) left the CACHED "
        "corrupt/chunks_bad_lines stale forever, because peek_stats()'s "
        "fingerprint check still matched what THIS save just wrote, so the "
        "GUI badge and CLI marker never cleared even though the fault was "
        "fixed - see _save()'s own reset and its why-comment")
    assert reloaded.stats()["corrupt"] is False
    assert Collection("kb", base=base).stats()["chunks_bad_lines"] == 0, (
        "a real _save() rewrites chunks.jsonl from the valid chunks only, "
        "so the malformed lines are gone from disk and must not still be "
        "counted on the next load")

    # peek_stats() (what the GUI list and repair button render from) sees the
    # healed state right after this save, with no intervening fresh Collection()
    # construction.
    peeked_after_heal = Collection.peek_stats("kb", base=base)
    assert peeked_after_heal is not None, (
        "fingerprint must still match what _save() just wrote")
    assert peeked_after_heal["chunks_bad_lines"] == 0
    assert peeked_after_heal["corrupt"] is False


def test_fingerprint_survives_a_real_noop_save(base, docs):
    """The other direction: a real _save() that changes NOTHING (e.g. a
    resync that finds no changes) must NOT spuriously invalidate its own
    cache - a fingerprint bug that always mismatches would silently defeat
    the whole fix by falling back to the slow path on every call."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    coll.resync(embed_fn=_embed3, policy=None)   # nothing changed
    _assert_peek_matches_eager(base, "kb")


def test_rejected_sidecar_present_degrade_reason(base, docs):
    """After a rejected vectors.json is set aside, the collection is still
    degraded until a real rebuild - peek must carry that same reason, not
    silently clear it because the (missing) vectors.json alone looks clean."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    vec_path = base / "kb" / "vectors.json"
    data = json.loads(vec_path.read_text(encoding="utf-8"))
    data["vectors"].pop(0)
    vec_path.write_text(json.dumps(data), encoding="utf-8")
    # Load once to trigger _load()'s rejection, then save to quarantine it.
    reloaded = Collection("kb", base=base)
    assert reloaded.vector_degrade_reason
    reloaded._save()
    assert (base / "kb" / "vectors.json.rejected").is_file(), (
        "fixture precondition: the rejected sidecar must exist on disk")

    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["has_vectors"] is False
    assert stats["vector_degrade_reason"], "the set-aside sidecar's degrade must persist"


def test_missing_doc_after_resync_counts_toward_n_missing(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    (docs / "alpha.txt").unlink()
    coll.resync(embed_fn=_embed3, policy=None)

    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_missing"] == 1
    assert stats["n_docs"] == 3, "a missing doc stays counted, per resync semantics"


def test_indexed_folder_root_counts_toward_n_roots(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_roots"] == 1


# --------------------------------------------------------------------------- #
#  The None contract: never a wrong answer, only a clear "cannot answer"      #
# --------------------------------------------------------------------------- #

def test_uncached_collection_returns_none_not_a_guess(base, docs):
    """A collection saved before this cache existed (simulated by stripping
    the key after a real save) must make peek_stats()/peek_detail() return
    None - never fall back to a partial/approximate read of chunks.jsonl or
    vectors.json, which is exactly the "answers wrong instead of not at all"
    shape this whole design avoids."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    meta_path = base / "kb" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert _STATS_CACHE_KEY in meta, "fixture precondition: _save() must have cached it"
    del meta[_STATS_CACHE_KEY]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None
    assert Collection.peek_detail("kb", base=base) is None
    # The real load is still fully correct: returning None makes the CALLER fall
    # back to it, and no data is lost.
    assert Collection("kb", base=base).stats()["n_chunks"] == 3


def test_corrupt_meta_json_returns_none(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    (base / "kb" / "meta.json").write_text("{not valid json", encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None
    assert Collection.peek_detail("kb", base=base) is None


def test_nonexistent_collection_returns_none(base):
    assert Collection.peek_stats("does-not-exist", base=base) is None
    assert Collection.peek_detail("does-not-exist", base=base) is None


def test_invalid_name_returns_none(base):
    assert Collection.peek_stats("../escape", base=base) is None
    assert Collection.peek_detail("bad name with spaces", base=base) is None


def test_hand_edited_stats_cache_wrong_shape_returns_none(base, docs):
    """A hand-edited meta.json with a malformed cache block (not the shape
    _save() writes) must fall back rather than trust garbage."""
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    meta_path = base / "kb" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[_STATS_CACHE_KEY] = "not even a dict"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert Collection.peek_stats("kb", base=base) is None
    assert Collection.peek_detail("kb", base=base) is None


# --------------------------------------------------------------------------- #
#  The cache must track real state across mutations, not just at creation     #
# --------------------------------------------------------------------------- #

def test_cache_updates_after_a_second_write(base, docs, tmp_path):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    first = Collection.peek_stats("kb", base=base)
    assert first["n_chunks"] == 3

    more_docs = tmp_path / "more"
    more_docs.mkdir()
    (more_docs / "delta.txt").write_text("delta content", encoding="utf-8")
    coll.add_paths([more_docs], embed_fn=_embed3)

    second = Collection.peek_stats("kb", base=base)
    assert second["n_chunks"] == 4
    _assert_peek_matches_eager(base, "kb")


def test_cache_updates_after_remove_doc(base, docs):
    coll = Collection("kb", base=base).create()
    coll.add_paths([docs], embed_fn=_embed3)
    doc_key = next(iter(coll.docs()))["path"]
    coll.remove_doc(doc_key)

    stats = _assert_peek_matches_eager(base, "kb")
    assert stats["n_chunks"] == 2
