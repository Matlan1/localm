# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characterization of ``localm.rag.store``: the collection lifecycle, the on-disk
format, the snapshot cache, locking (in-process and cross-process), corrupt or
partial collection files, embedding-dimension switches, indexing confinement,
model relabelling and the provenance report.

``tests/fixtures/rag_store_v1/`` holds collections written by this module and the
answers it gave for them (``expected.json``); the fixture tests assert the current
code opens, searches and rewrites them identically.

Every test reaches the code through ``localm.rag.store``, and the ``TestFacadePatch``
cases replace a name on that module and assert every call site sees the
replacement.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

import localm.rag.store as store
from localm.rag.collection_lock import CollectionLockedError
from localm.rag.store import (Collection, ConfinementError, collection_names,
                              collection_provenance_note,
                              collection_provenance_report, confine_index_path,
                              delete_collection, indexing_policy,
                              relabel_embedding_model)
from tests._process_identity import spawn_on_this_tree

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rag_store_v1"
FIXTURE_COLLECTIONS = ("kb", "lexical", "mixed", "degraded")

_VOCAB = ("turbine", "gearbox", "apple", "banana", "cherry", "salary", "river",
          "notes")


def _embed(texts):
    """Deterministic 8-dimensional bag-of-words vectors over ``_VOCAB``."""
    return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


def _embed4(texts):
    """A different model: 4 dimensions."""
    return [[float(t.lower().count(w)) for w in _VOCAB[:4]] for t in texts]


_LONG_DOC = "\n\n".join(
    f"Section {i}. Inspection log for unit {i}. "
    + "The turbine spins. " * i + "The gearbox hums. " * (7 - i)
    + "River notes describe the cooling loop. " * (i % 3 + 1)
    + "An apple orchard grows nearby. " * ((7 - i) % 4)
    + "Filler text about maintenance windows and shift handover. " * 8
    for i in range(1, 7))

FIXTURE_DOCS = {
    "turbines.txt": _LONG_DOC,
    "fruit.md": "# Fruit\n\nApple and banana and cherry are fruit.\n\nBanana bread notes.",
    "pay.txt": "Salary review notes for the river team.",
    "plain.txt": "Nothing from the vocabulary is mentioned in this sentence at all.",
}

FIXTURE_QUERIES = (
    ("turbine", 4, False, False),
    ("turbine gearbox", 5, True, False),
    ("apple river notes", 6, True, False),
    ("apple river notes", 6, False, True),
    ("banana cherry", 4, True, True),
    ("salary river", 3, False, True),
    ("notes", 8, True, True),
    ("why did that fail", 3, True, True),
    ("", 3, False, False),
)


def _uploads(docs: dict) -> list:
    return [{"filename": k, "data": v.encode("utf-8")} for k, v in docs.items()]


def _observe(c: Collection) -> dict:
    """Everything a caller can read off a loaded collection, JSON-normalised."""
    queries = []
    for text, k, hybrid, relevant in FIXTURE_QUERIES:
        hits = c.query(text, k=k, embed_fn=_embed if hybrid else None,
                       relevant_only=relevant)
        queries.append({"q": [text, k, hybrid, relevant], "hits": hits})
    obs = {
        "exists": c.exists(),
        "stats": c.stats(),
        "docs": c.docs(),
        "roots": c.roots(),
        "documents": sorted(c.documents()),
        "vector_dim": c.vector_dim(),
        "embedding_model": c.embedding_model(),
        "embedding_model_mixed": c.embedding_model_mixed(),
        "is_confined_to_nothing": c.is_confined_to([]),
        "queries": queries,
    }
    return json.loads(json.dumps(obs))


def _strip_scores(obs: dict) -> tuple[dict, list]:
    scores = []
    for q in obs["queries"]:
        for h in q["hits"]:
            scores.append(h.pop("score"))
    return obs, scores


def build_fixture_collections(base: Path) -> None:
    """Write the fixture collections into *base* (used once, with the code under
    characterization, to produce ``tests/fixtures/rag_store_v1``)."""
    kb = Collection("kb", base=base).create()
    kb.add_uploads(_uploads(FIXTURE_DOCS), embed_fn=_embed,
                   model_name="bge-small-en-v1.5")

    Collection("lexical", base=base).create().add_uploads(_uploads(FIXTURE_DOCS))

    mixed = Collection("mixed", base=base).create()
    mixed.add_uploads(_uploads({"turbines.txt": _LONG_DOC}), embed_fn=_embed,
                      model_name="bge-small-en-v1.5")
    mixed.add_uploads(_uploads({"fruit.md": FIXTURE_DOCS["fruit.md"]}),
                      embed_fn=_embed, model_name="nomic-embed-text-v1.5")

    deg = Collection("degraded", base=base).create()
    deg.add_uploads(_uploads({"turbines.txt": _LONG_DOC}), embed_fn=_embed,
                    model_name="bge-small-en-v1.5")
    (deg.dir / "vectors.json").write_text('{"dim": 8, "vectors": ["x"]}',
                                          encoding="utf-8")
    Collection("degraded", base=base, cache=False).add_uploads(
        _uploads({"pay.txt": FIXTURE_DOCS["pay.txt"]}))


@pytest.fixture(autouse=True)
def _fresh_process_state():
    store._COLLECTION_CACHE.clear()
    store._WARNED_DEGRADES.clear()
    yield
    store._COLLECTION_CACHE.clear()
    store._WARNED_DEGRADES.clear()


@pytest.fixture
def base(tmp_path):
    d = tmp_path / "collections"
    d.mkdir()
    return d


@pytest.fixture
def fixture_base(tmp_path):
    dest = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT / "collections", dest)
    return dest


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / "node_modules").mkdir()
    (d / "a.txt").write_text("alpha turbine notes", encoding="utf-8")
    (d / "sub" / "b.md").write_text("# Beta\n\nbanana gearbox notes", encoding="utf-8")
    (d / ".git" / "c.txt").write_text("never indexed", encoding="utf-8")
    (d / "node_modules" / "d.txt").write_text("never indexed", encoding="utf-8")
    (d / "id_rsa").write_text("secret", encoding="utf-8")
    (d / "w.gguf").write_bytes(b"GGUF")
    return d


# --------------------------------------------------------------------------- #
#  On-disk format: collections written before the split                      #
# --------------------------------------------------------------------------- #

class TestOnDiskFormat:
    @pytest.mark.parametrize("numpy_mode", ["as-installed", "pure-python"])
    @pytest.mark.parametrize("name", FIXTURE_COLLECTIONS)
    def test_fixture_collections_answer_as_recorded(self, fixture_base, monkeypatch,
                                                    name, numpy_mode):
        if numpy_mode == "pure-python":
            monkeypatch.setattr(store, "_numpy", None)
        expected = json.loads((FIXTURE_ROOT / "expected.json").read_text("utf-8"))
        got, got_scores = _strip_scores(_observe(Collection(name, base=fixture_base)))
        want, want_scores = _strip_scores(expected[name])
        assert got == want
        assert got_scores == pytest.approx(want_scores, abs=2e-4)

    @pytest.mark.parametrize("name", FIXTURE_COLLECTIONS)
    def test_a_rewrite_reproduces_the_files(self, fixture_base, name):
        src = FIXTURE_ROOT / "collections" / name
        c = Collection(name, base=fixture_base)
        c.add_uploads([])
        coll = fixture_base / name
        assert sorted(p.name for p in coll.iterdir()) == sorted(
            p.name for p in src.iterdir())
        for fname in ("chunks.jsonl", "vectors.json", "vectors.json.rejected"):
            if (src / fname).exists():
                assert (coll / fname).read_bytes() == (src / fname).read_bytes(), fname
        got = json.loads((coll / "meta.json").read_text("utf-8"))
        want = json.loads((src / "meta.json").read_text("utf-8"))
        got[store._STATS_CACHE_KEY].pop("fingerprint")
        want[store._STATS_CACHE_KEY].pop("fingerprint")
        assert got == want

    def test_a_stale_stats_cache_is_refused_then_backfilled(self, fixture_base):
        os.utime(fixture_base / "kb" / "chunks.jsonl", ns=(1, 1))
        assert Collection.peek_stats("kb", fixture_base) is None
        assert Collection.peek_detail("kb", fixture_base) is None
        coll = Collection.load_and_maybe_backfill("kb", fixture_base)
        peeked = Collection.peek_stats("kb", fixture_base)
        assert peeked == coll.stats()
        detail = Collection.peek_detail("kb", fixture_base)
        assert detail == {**coll.stats(), "docs": coll.docs()}

    def test_fixture_meta_layout(self):
        meta = json.loads((FIXTURE_ROOT / "collections" / "kb" / "meta.json")
                          .read_text("utf-8"))
        assert sorted(meta) == ["_stats_cache", "created", "docs",
                                "embedding_model", "name"]
        assert sorted(meta["_stats_cache"]) == [
            "chunks_bad_lines", "corrupt", "fingerprint", "has_vectors", "n_chunks",
            "vector_degrade_reason", "vector_dim"]
        entry = meta["docs"]["upload:fruit.md"]
        assert sorted(entry) == ["chunks", "hash", "size", "uploaded"]
        vec = json.loads((FIXTURE_ROOT / "collections" / "kb" / "vectors.json")
                         .read_text("utf-8"))
        assert sorted(vec) == ["dim", "vectors"] and vec["dim"] == 8


# --------------------------------------------------------------------------- #
#  Lifecycle over host paths                                                  #
# --------------------------------------------------------------------------- #

class TestLifecycle:
    def test_create_writes_an_empty_collection(self, base):
        c = Collection("kb", base=base)
        assert not c.exists() and c.stats()["n_docs"] == 0
        assert c.create() is c and c.exists()
        meta = json.loads((base / "kb" / "meta.json").read_text("utf-8"))
        assert meta["name"] == "kb" and meta["docs"] == {}
        assert isinstance(meta["created"], float)
        assert meta[store._STATS_CACHE_KEY]["n_chunks"] == 0
        assert (base / "kb" / "chunks.jsonl").read_text("utf-8") == ""
        assert not (base / "kb" / "vectors.json").exists()
        assert collection_names(base) == ["kb"]
        created = meta["created"]
        Collection("kb", base=base).create()
        meta = json.loads((base / "kb" / "meta.json").read_text("utf-8"))
        assert meta["created"] == created

    def test_add_a_folder(self, base, folder):
        said = []
        c = Collection("kb", base=base).create()
        res = c.add_paths([folder], on_progress=said.append)
        assert res == {"added": 2, "updated": 0, "skipped": 0, "failed": [],
                       "chunks": 2}
        rel = sorted(Path(d["path"]).relative_to(folder.resolve()).as_posix()
                     for d in c.docs())
        assert rel == ["a.txt", "sub/b.md"]
        assert c.roots() == [str(folder.resolve())]
        assert said == ["[1/2] reading a.txt...", "indexed a.txt (1 chunks)",
                        "[2/2] reading b.md...", "indexed b.md (1 chunks)"]
        entry = c.docs()[0]
        assert sorted(entry) == ["chunks", "hash", "mtime", "path", "size"]
        chunk = c.query("banana", k=1)[0]
        assert chunk["format"] == "markdown" and chunk["pos"] == 1
        assert chunk["score"] == 1.0

    def test_unchanged_changed_and_forced(self, base, folder):
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        assert c.add_paths([folder])["skipped"] == 2
        a = folder / "a.txt"
        st = a.stat()
        a.write_text("ALPHA turbine notes", encoding="utf-8")
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns))
        res = c.add_paths([folder])
        assert (res["updated"], res["skipped"]) == (1, 1)
        assert c.add_paths([folder], force=True)["updated"] == 2

    def test_an_explicit_binary_pick_is_a_per_file_failure(self, base, folder):
        said = []
        c = Collection("kb", base=base).create()
        res = c.add_paths([folder / "w.gguf", folder / "a.txt"],
                          on_progress=said.append)
        assert res["added"] == 1
        assert res["failed"] == [{
            "path": str((folder / "w.gguf").resolve()),
            "error": "w.gguf: no extractable text (binary, media, or model weights)"}]
        assert said[0] == ("skip w.gguf: no extractable text (binary, media, or "
                           "model weights)")

    def test_a_folder_with_nothing_indexable_still_records_its_root(self, base, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        c = Collection("kb", base=base).create()
        assert c.add_paths([empty]) == {"added": 0, "updated": 0, "skipped": 0,
                                        "failed": [], "chunks": 0}
        assert Collection("kb", base=base).roots() == [str(empty.resolve())]

    def test_query_hybrid_and_lexical(self, base, folder):
        c = Collection("kb", base=base).create()
        c.add_paths([folder], embed_fn=_embed, model_name="bge-small-en-v1.5")
        lex = c.query("gearbox", k=4)
        assert [Path(h["source"]).name for h in lex] == ["b.md"]
        hyb = c.query("gearbox", k=4, embed_fn=_embed)
        assert [Path(h["source"]).name for h in hyb] == ["b.md"]
        assert hyb[0]["score"] == 1.0
        assert c.query("   ", k=4) == []
        assert c.stats()["has_vectors"] is True and c.vector_dim() == 8
        assert c.embedding_model() == "bge-small-en-v1.5"

    def test_remove_doc_and_delete(self, base, folder):
        c = Collection("kb", base=base).create()
        c.add_paths([folder], embed_fn=_embed)
        key = str((folder / "a.txt").resolve())
        assert c.remove_doc(key) is True
        assert c.remove_doc(key) is False
        reloaded = Collection("kb", base=base)
        assert [Path(d["path"]).name for d in reloaded.docs()] == ["b.md"]
        assert reloaded.stats()["n_chunks"] == 1
        assert len(json.loads((base / "kb" / "vectors.json").read_text("utf-8"))
                   ["vectors"]) == 1
        assert delete_collection("kb", base) is True
        assert delete_collection("kb", base) is False
        assert collection_names(base) == []

    def test_names(self, base):
        for bad in ("", "a.b", "a/b", "con", "LPT1", " x", "x" * 65):
            with pytest.raises(ValueError):
                Collection(bad, base=base)
        assert store.check_collection_name("Docs 2") == "Docs 2"
        assert collection_names(base / "missing") == []


# --------------------------------------------------------------------------- #
#  Resync                                                                     #
# --------------------------------------------------------------------------- #

class TestResync:
    def test_new_changed_missing_restored_pruned(self, base, folder):
        said = []
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        (folder / "new.txt").write_text("cherry notes", encoding="utf-8")
        (folder / "a.txt").write_text("alpha turbine notes, edited", encoding="utf-8")
        (folder / "sub" / "b.md").unlink()
        res = c.resync(on_progress=said.append)
        assert sorted(res) == sorted([
            "added", "updated", "skipped", "failed", "chunks", "roots",
            "unavailable_roots", "blocked_roots", "missing", "restored", "pruned",
            "missing_total", "vector_degrade_reason"])
        b_key = str((folder / "sub" / "b.md").resolve())
        assert (res["added"], res["updated"]) == (1, 1)
        assert res["missing"] == [b_key] and res["missing_total"] == 1
        assert f"missing: {b_key} (kept in the index, flagged)" in said
        entry = {d["path"]: d for d in Collection("kb", base=base).docs()}[b_key]
        assert entry["missing"] is True and isinstance(entry["missing_since"], float)
        assert Collection("kb", base=base).stats()["n_missing"] == 1
        assert c.query("banana", k=2)

        (folder / "sub" / "b.md").write_text("# Beta\n\nbanana gearbox notes",
                                             encoding="utf-8")
        res = c.resync()
        assert res["restored"] == [b_key] and res["missing_total"] == 0

        (folder / "sub" / "b.md").unlink()
        res = c.resync(prune_missing=True)
        assert res["pruned"] == [b_key]
        assert b_key not in Collection("kb", base=base).documents()

    def test_an_unavailable_root_is_left_untouched(self, base, folder):
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        shutil.rmtree(folder)
        res = c.resync(prune_missing=True)
        assert res["unavailable_roots"] == [{
            "root": str(folder.resolve()),
            "reason": "the indexed folder is not available (deleted, unmounted, "
                      "or unreadable)"}]
        assert res["pruned"] == [] and res["missing"] == []
        assert len(Collection("kb", base=base).documents()) == 2

    def test_a_root_replaced_by_a_file(self, base, folder):
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        shutil.rmtree(folder)
        folder.write_text("x", encoding="utf-8")
        res = c.resync()
        assert res["unavailable_roots"][0]["reason"] == (
            "the indexed folder is now a file, not a directory")

    def test_a_root_the_policy_now_refuses_is_blocked(self, base, folder, tmp_path):
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        policy = {"mode": "blacklist", "allowed": [], "denied": [folder]}
        res = c.resync(policy=policy, prune_missing=True)
        assert [r["root"] for r in res["blocked_roots"]] == [str(folder.resolve())]
        assert "denied list" in res["blocked_roots"][0]["reason"]
        assert res["added"] == res["updated"] == 0 and res["pruned"] == []


# --------------------------------------------------------------------------- #
#  Uploads                                                                    #
# --------------------------------------------------------------------------- #

class TestUploads:
    def test_counters_keys_and_progress(self, base):
        calls = []

        def progress(text, **kw):
            calls.append((text, kw))

        c = Collection("kb", base=base).create()
        res = c.add_uploads(_uploads({"a.txt": "apple notes", "b.txt": "banana"})
                            + [{"filename": "  ", "data": b"cherry"},
                               {"filename": "bad.pdf", "data": b"not a pdf"}],
                            on_progress=progress)
        assert (res["added"], res["updated"], res["skipped"]) == (3, 0, 0)
        assert [f["path"] for f in res["failed"]] == ["upload:bad.pdf"]
        assert sorted(c.documents()) == ["upload:a.txt", "upload:b.txt", "upload:upload"]
        structured = [kw for _t, kw in calls if kw]
        assert [k["done"] for k in structured] == [1, 2, 3, 4]
        assert {(k["phase"], k["total"], k["unit"]) for k in structured} == {
            ("indexing uploads", 4, "files")}
        assert calls[0] == ("[1/4] reading a.txt...", {})
        assert calls[1] == ("indexed a.txt (1 chunks)",
                            {"phase": "indexing uploads", "done": 1, "total": 4,
                             "unit": "files"})
        again = c.add_uploads(_uploads({"a.txt": "apple notes"}), on_progress=progress)
        assert (again["skipped"], calls[-1][0]) == (1, "skip a.txt (unchanged)")
        assert c.add_uploads(_uploads({"a.txt": "apple notes"}),
                             force=True)["updated"] == 1


# --------------------------------------------------------------------------- #
#  Snapshot cache                                                             #
# --------------------------------------------------------------------------- #

class TestSnapshotCache:
    def test_a_second_instance_is_served_from_the_cache(self, base):
        Collection("kb", base=base).create().add_uploads(
            _uploads({"a.txt": "apple notes"}), embed_fn=_embed)
        first = Collection("kb", base=base)
        snap = store._get_cached_collection_data(first.dir)
        assert snap is not None and len(store._COLLECTION_CACHE) == 1
        second = Collection("kb", base=base)
        assert second._chunks[0] is snap.chunks[0]
        with pytest.raises(TypeError):
            second._chunks[0]["text"] = "x"
        second._meta["docs"]["upload:a.txt"]["size"] = -1
        assert Collection("kb", base=base)._meta["docs"]["upload:a.txt"]["size"] == 11
        assert first._lexical_index() is second._lexical_index()

    def test_cache_false_never_stores(self, base):
        Collection("kb", base=base).create()
        store._COLLECTION_CACHE.clear()
        Collection("kb", base=base, cache=False)
        assert len(store._COLLECTION_CACHE) == 0

    def test_every_write_drops_the_entry(self, base):
        c = Collection("kb", base=base).create()
        for write in (lambda: c.add_uploads(_uploads({"a.txt": "apple"})),
                      lambda: c.reembed(embed_fn=_embed),
                      lambda: c.remove_doc("upload:a.txt")):
            Collection("kb", base=base)
            assert store._get_cached_collection_data(c.dir) is not None
            write()
            assert store._get_cached_collection_data(c.dir) is None

    def test_an_outside_rewrite_is_seen(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}))
        Collection("kb", base=base)
        line = json.dumps({"text": "banana", "source": "upload:a.txt", "pos": 1})
        (c.dir / "chunks.jsonl").write_text(line + "\n" + line + "\n", encoding="utf-8")
        assert Collection("kb", base=base).stats()["n_chunks"] == 2

    def test_no_store_while_files_are_changing(self, base):
        c = Collection("kb", base=base).create()
        store._COLLECTION_CACHE.clear()
        with store._files_changing(c.dir):
            Collection("kb", base=base)
            assert len(store._COLLECTION_CACHE) == 0
        Collection("kb", base=base)
        assert len(store._COLLECTION_CACHE) == 1

    def test_the_entry_count_is_bounded(self, base):
        for i in range(store._MAX_CACHED_COLLECTIONS + 2):
            Collection(f"c{i}", base=base).create()
            Collection(f"c{i}", base=base)
        assert len(store._COLLECTION_CACHE) == store._MAX_CACHED_COLLECTIONS
        assert store._get_cached_collection_data(base / "c0") is None

    @pytest.mark.parametrize("budget", ["below-text", "below-snapshot"])
    def test_the_byte_budget_is_read_off_the_store_module(self, base, monkeypatch,
                                                          budget):
        Collection("kb", base=base).create().add_uploads(
            _uploads({"a.txt": "apple " * 50}))
        store._COLLECTION_CACHE.clear()
        c = Collection("kb", base=base)
        snap = store._get_cached_collection_data(c.dir)
        text_bytes = c._min_cached_nbytes()
        assert snap is not None and text_bytes < snap.nbytes
        limit = 10 if budget == "below-text" else (text_bytes + snap.nbytes) // 2
        store._COLLECTION_CACHE.clear()
        monkeypatch.setattr(store, "_COLLECTION_CACHE_MAX_BYTES", limit)
        Collection("kb", base=base)
        assert len(store._COLLECTION_CACHE) == 0
        assert store._COLLECTION_CACHE.total_bytes() == 0


# --------------------------------------------------------------------------- #
#  Locking                                                                    #
# --------------------------------------------------------------------------- #

_HOLD = '''
    import sys
    from pathlib import Path
    from localm.rag.collection_lock import collection_write_lock
    with collection_write_lock(Path(sys.argv[1]), collection="kb",
                               op="a long index", timeout=30):
        print("HELD", flush=True)
        sys.stdin.read()
'''

_WRITE = '''
    import sys
    from pathlib import Path
    from localm.rag.store import Collection
    Collection("kb", base=Path(sys.argv[1])).add_uploads(
        [{"filename": "other.txt", "data": b"cherry from another process"}])
    print("DONE", flush=True)
'''


class TestLocking:
    def test_the_in_process_lock_folds_case(self):
        assert store._collection_lock("Docs") is store._collection_lock("docs")

    def test_concurrent_writers_lose_nothing(self, base):
        Collection("kb", base=base).create()
        errors = []

        def add(i):
            try:
                Collection("kb", base=base).add_uploads(
                    _uploads({f"d{i}.txt": f"document number {i}"}))
            except Exception as e:      # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        assert len(Collection("kb", base=base).documents()) == 6

    def test_a_delete_waits_out_an_in_process_writer_then_refuses(self, base, monkeypatch):
        Collection("kb", base=base).create()
        monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "0.2")
        lock = store._collection_lock("kb")
        lock.acquire()
        try:
            got = []
            t = threading.Thread(target=lambda: got.append(
                _raises(lambda: delete_collection("kb", base))))
            t.start()
            t.join(30)
        finally:
            lock.release()
        assert isinstance(got[0], CollectionLockedError)
        assert Collection("kb", base=base).exists()

    def test_another_process_holding_the_lock(self, base, tmp_path, monkeypatch,
                                              heavy_slot):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}), embed_fn=_embed, model_name="m1")
        before = (c.dir / "meta.json").read_bytes()
        monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "0.3")
        home = tmp_path / "holder-home"
        home.mkdir()
        p = spawn_on_this_tree(_HOLD, home, store.lock_path_for(c.dir),
                               stdin=subprocess.PIPE)
        try:
            assert p.stdout.readline().split()[:1] == ["HELD"]
            with pytest.raises(CollectionLockedError):
                delete_collection("kb", base)
            with pytest.raises(CollectionLockedError):
                c.add_uploads(_uploads({"b.txt": "banana"}))
            assert relabel_embedding_model("m1", "m2", base) == ([], ["kb"])
            os.utime(c.dir / "chunks.jsonl", ns=(1, 1))
            backfilled = Collection.load_and_maybe_backfill("kb", base)
            assert backfilled.documents() == ["upload:a.txt"]
            assert (c.dir / "meta.json").read_bytes() == before
        finally:
            p.communicate(timeout=60)
        assert relabel_embedding_model("m1", "m2", base) == (["kb"], [])
        assert delete_collection("kb", base) is True

    def test_a_write_from_another_process_is_seen(self, base, tmp_path, heavy_slot):
        Collection("kb", base=base).create().add_uploads(_uploads({"a.txt": "apple"}))
        assert Collection("kb", base=base).documents() == ["upload:a.txt"]
        home = tmp_path / "writer-home"
        home.mkdir()
        p = spawn_on_this_tree(_WRITE, home, base)
        out, err = p.communicate(timeout=120)
        assert "DONE" in out, err
        assert sorted(Collection("kb", base=base).documents()) == [
            "upload:a.txt", "upload:other.txt"]


def _raises(fn):
    try:
        fn()
    except Exception as e:          # noqa: BLE001
        return e
    return None


# --------------------------------------------------------------------------- #
#  Corrupt or partial collection files                                        #
# --------------------------------------------------------------------------- #

def _write_chunks(coll_dir: Path, *sources: str) -> None:
    coll_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"text": f"apple text {i}", "source": s, "pos": 1})
             for i, s in enumerate(sources)]
    (coll_dir / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCorruptFiles:
    def test_absent_meta(self, base):
        _write_chunks(base / "kb", "upload:a.txt")
        c = Collection("kb", base=base)
        assert not c.exists() and c.corrupt is False and c._chunks == []
        assert collection_names(base) == []
        assert Collection.peek_stats("kb", base) is None
        assert Collection.confined_to("kb", ["x"], base) is None
        assert delete_collection("kb", base) is False

    @pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]", b"\xff\xfe\x00bad"],
                             ids=["malformed", "not-an-object", "undecodable"])
    def test_unusable_meta_is_flagged_and_rebuilt_from_chunks(self, base, raw, tmp_path):
        _write_chunks(base / "kb", "upload:a.txt", "upload:a.txt", "/x/b.txt")
        (base / "kb" / "meta.json").write_bytes(raw)
        c = Collection("kb", base=base)
        assert c.exists() and c.corrupt is True and c._meta_unreadable is True
        assert c.stats()["corrupt"] is True and c.stats()["chunks_bad_lines"] == 0
        assert c.docs() == [{"path": "/x/b.txt", "chunks": 1},
                            {"path": "upload:a.txt", "chunks": 2, "uploaded": True}]
        assert c._meta["name"] == "kb" and "created" not in c._meta
        if raw.startswith(b"\xff"):
            with pytest.raises(UnicodeDecodeError):
                Collection.peek_stats("kb", base)
            with pytest.raises(UnicodeDecodeError):
                Collection.confined_to("kb", [str(tmp_path)], base)
        else:
            assert Collection.peek_stats("kb", base) is None
            assert Collection.confined_to("kb", [str(tmp_path)], base) is None
        assert c.is_confined_to([str(tmp_path)]) is False
        assert c.is_confined_to([]) is True
        c.add_uploads([])
        fixed = Collection("kb", base=base)
        assert fixed.corrupt is False and fixed._meta_unreadable is False
        assert Collection.peek_stats("kb", base)["n_docs"] == 2

    def test_bad_chunk_lines_keep_the_real_docs_map(self, base, caplog):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}))
        with (c.dir / "chunks.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('not json\n[1]\n{"no_text": 1}\n')
        with caplog.at_level(logging.WARNING, logger="localm"):
            again = Collection("kb", base=base)
            Collection("kb", base=base, cache=False)
        assert again.corrupt is True and again.chunks_bad_lines == 3
        assert again._meta_unreadable is False
        assert again.documents() == ["upload:a.txt"]
        warned = [r.getMessage() for r in caplog.records if "malformed" in r.getMessage()]
        assert warned == ["RAG collection 'kb': skipped 3 malformed line(s) in "
                          "chunks.jsonl; run 'localm rag repair'"]

    @pytest.mark.parametrize("payload,reason", [
        ("{nope", "vectors.json is unreadable (JSONDecodeError); using BM25 "
                  "lexical retrieval only"),
        ('{"dim": 8, "vectors": ["x"]}', "vectors.json is malformed (entries are "
                                         "not vectors); using BM25 lexical "
                                         "retrieval only"),
        ('{"dim": 1, "vectors": [[NaN]]}', "vectors.json has non-finite (NaN/inf) "
                                           "or non-numeric values; using BM25 "
                                           "lexical retrieval only"),
        ('{"dim": 1, "vectors": [[1.0], [2.0]]}',
         "vectors.json has 2 vectors for 1 chunks (orphaned entries from a prior, "
         "larger index); using BM25 lexical retrieval only"),
    ], ids=["unreadable", "malformed", "non-finite", "orphaned"])
    def test_an_unusable_vectors_file_is_named_and_set_aside(self, base, payload,
                                                              reason):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}), embed_fn=_embed)
        (c.dir / "vectors.json").write_text(payload, encoding="utf-8")
        bad = Collection("kb", base=base)
        assert bad.vector_degrade_reason == reason
        assert bad.vector_dim() is None and bad.stats()["has_vectors"] is False
        bad.add_uploads(_uploads({"b.txt": "banana"}))
        assert (c.dir / "vectors.json.rejected").read_text("utf-8") == payload
        assert not (c.dir / "vectors.json").exists()
        after = Collection("kb", base=base)
        assert after.vector_degrade_reason.startswith(
            "an earlier vector index was unusable and was set aside as "
            "vectors.json.rejected")
        after.reembed(embed_fn=_embed, model_name="m")
        assert not (c.dir / "vectors.json.rejected").exists()
        assert Collection("kb", base=base).vector_degrade_reason is None

    def test_a_partial_embed(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple", "b.txt": "banana"}))
        (c.dir / "vectors.json").write_text('{"dim": 8, "vectors": [[1,0,0,0,0,0,0,0]]}',
                                            encoding="utf-8")
        assert Collection("kb", base=base).vector_degrade_reason == (
            "vectors.json has 1 vectors for 2 chunks (a partial embed); using BM25 "
            "lexical retrieval only")

    def test_set_aside_names_are_numbered_and_pruned(self, base):
        c = Collection("kb", base=base).create()
        for i in range(store._MAX_REJECTED_KEPT + 2):
            c.add_uploads(_uploads({f"d{i}.txt": "apple"}), embed_fn=_embed)
            (c.dir / "vectors.json").write_text("{bad", encoding="utf-8")
            Collection("kb", base=base, cache=False).add_uploads([])
            time.sleep(0.02)
        names = sorted(p.name for p in c.dir.glob("vectors.json.rejected*"))
        assert len(names) == store._MAX_REJECTED_KEPT
        assert "vectors.json.rejected" not in names[:1] or len(names) == 3

    def test_roots_that_are_not_an_object(self, base, folder, caplog):
        c = Collection("kb", base=base).create()
        meta = json.loads((c.dir / "meta.json").read_text("utf-8"))
        meta["roots"] = ["not", "a", "map"]
        (c.dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="localm"):
            Collection("kb", base=base).add_paths([folder])
        assert any("meta.json 'roots' was not an object (list)" in r.getMessage()
                   for r in caplog.records)
        assert Collection("kb", base=base).roots() == [str(folder.resolve())]


# --------------------------------------------------------------------------- #
#  Embedding model and dimension                                              #
# --------------------------------------------------------------------------- #

class TestEmbeddingDimension:
    def test_a_different_dimension_is_refused_after_saving_finished_files(
            self, base, tmp_path):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}), embed_fn=_embed,
                      model_name="bge-small-en-v1.5")
        d = tmp_path / "more"
        d.mkdir()
        (d / "1.txt").write_text("banana", encoding="utf-8")
        (d / "2.txt").write_text("cherry", encoding="utf-8")
        calls = []

        def switching(texts):
            calls.append(texts)
            return _embed(texts) if len(calls) == 1 else _embed4(texts)

        with pytest.raises(ValueError) as exc:
            c.add_paths([d], embed_fn=switching)
        assert str(exc.value) == (
            "Embedding dimension changed (8 -> 4): collection 'kb' was built with "
            "bge-small-en-v1.5 and its stored vectors cannot be mixed with a "
            "different model's. Re-embed it in place from the text already stored "
            "(no source files needed, nothing deleted):\n"
            "    localm rag reembed kb\n"
            "or use 'Re-embed' on the Knowledge page. To keep the existing index "
            "instead, switch the embedding model back with bge-small-en-v1.5.")
        names = sorted(Path(p).name for p in Collection("kb", base=base).documents())
        assert names == ["1.txt", "upload:a.txt"]
        with pytest.raises(ValueError, match=r"changed \(8 -> 4\)"):
            c.add_uploads(_uploads({"z.txt": "river"}), embed_fn=_embed4)

    def test_a_query_with_another_dimension_degrades_and_warns_once(self, base, caplog):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple notes", "b.txt": "banana notes"}),
                      embed_fn=_embed)
        lexical = c.query("apple", k=2)
        with caplog.at_level(logging.WARNING, logger="localm"):
            first = c.query("apple", k=2, embed_fn=_embed4)
            Collection("kb", base=base).query("apple", k=2, embed_fn=_embed4)
        assert first == lexical
        reason = ("embedding model changed (query dim 4 != stored 8); using BM25 "
                  "lexical retrieval only - rebuild the collection to restore "
                  "semantic search")
        assert c.vector_degrade_reason == reason
        assert [r.getMessage() for r in caplog.records] == [
            f"RAG collection 'kb': {reason}"]
        c.query("apple", k=2, embed_fn=_embed)
        assert c.vector_degrade_reason is None

    def test_a_failing_query_embedder(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}), embed_fn=_embed)

        def boom(texts):
            raise OSError("backend down")

        assert c.query("apple", k=1, embed_fn=boom)[0]["score"] == 1.0
        assert c.vector_degrade_reason == (
            "query embedding failed (OSError); using BM25 lexical retrieval only")

    def test_reembed(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple", "b.txt": "banana"}), embed_fn=_embed,
                      model_name="old")
        c.add_uploads(_uploads({"c.txt": "cherry"}), embed_fn=_embed, model_name="other")
        assert c.embedding_model_mixed() is True
        seen = []
        res = c.reembed(embed_fn=_embed4, model_name="new",
                        on_progress=lambda t, **kw: seen.append((t, kw)), batch=2)
        assert res == {"chunks": 3, "dim": 4, "model": "new"}
        assert seen == [("re-embedding 2/3", {"phase": "re-embedding", "done": 2,
                                              "total": 3, "unit": "chunks"}),
                        ("re-embedding 3/3", {"phase": "re-embedding", "done": 3,
                                              "total": 3, "unit": "chunks"})]
        meta = json.loads((c.dir / "meta.json").read_text("utf-8"))
        assert (meta["embedding_model"], meta["embedding_dim"]) == ("new", 4)
        assert "embedding_model_mixed" not in meta
        assert Collection("kb", base=base).vector_dim() == 4

    def test_reembed_refusals_leave_the_index(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple", "b.txt": "banana"}), embed_fn=_embed)
        before = (c.dir / "vectors.json").read_bytes()
        with pytest.raises(RuntimeError, match="returned 1 vectors for 2 chunks"):
            c.reembed(embed_fn=lambda t: _embed(t)[:1], batch=8)
        with pytest.raises(RuntimeError, match="inconsistent vector sizes"):
            c.reembed(embed_fn=lambda t: [[1.0], [1.0, 2.0]][:len(t)], batch=8)
        with pytest.raises(RuntimeError, match="returned nothing for chunks"):
            c.reembed(embed_fn=lambda t: None)
        assert (c.dir / "vectors.json").read_bytes() == before
        empty = Collection("e", base=base).create()
        assert empty.reembed(embed_fn=_embed, model_name="m") == {
            "chunks": 0, "dim": None, "model": "m",
            "note": "collection has no chunks; nothing to re-embed"}

    def test_relevance_floors(self, base):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple banana notes", "b.txt": "river notes"}),
                      embed_fn=_embed, model_name="bge-small-en-v1.5")
        assert [h["source"] for h in c.query("apple banana", k=2, embed_fn=_embed,
                                            relevant_only=True)] == ["upload:a.txt"]
        assert c.query("why did that fail", k=2, embed_fn=_embed,
                       relevant_only=True) == []
        assert store.refers_to_conversation("say that again") is True
        assert store.refers_to_conversation("apple banana") is False


# --------------------------------------------------------------------------- #
#  Confinement                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def jail(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    for d in (home, cwd, allowed, outside):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd, allowed, outside


def _reason(p, policy=None):
    try:
        confine_index_path(p, policy)
    except ConfinementError as e:
        assert isinstance(e, ValueError) and e.path is not None
        return e.reason
    return None


class TestConfinement:
    def test_policy_none_is_the_hard_floor_only(self, jail, tmp_path):
        _home, _cwd, _allowed, outside = jail
        assert confine_index_path(outside) == outside.resolve()
        assert _reason(outside / "proj" / ".ssh" / "k.txt") == "credential"
        assert _reason(outside / "x" / ".AWS") == "credential"
        assert _reason("\\\\server\\share\\doc.txt") == "unc_or_device"
        assert _reason("//server/share/doc.txt") == "unc_or_device"
        assert _reason(outside / "id_rsa") is None

    def test_whitelist(self, jail):
        home, cwd, allowed, outside = jail
        policy = indexing_policy({"rag_allowed_roots": [str(allowed)]})
        assert policy == {"mode": "whitelist", "allowed": [allowed.resolve()],
                          "denied": [], "allow_network_drives": True}
        for ok in (home / "a.txt", cwd / "b", allowed / "c"):
            assert _reason(ok, policy) is None
        assert _reason(outside / "d", policy) == "outside_allowed"
        (allowed / "id_rsa").write_text("k", encoding="utf-8")
        assert _reason(allowed / "id_rsa", policy) == "secret_file"
        assert _reason(allowed / "cert.PEM", policy) == "secret_file"
        (allowed / "id_rsa_dir.pem").mkdir()
        assert _reason(allowed / "id_rsa_dir.pem", policy) is None

    def test_blacklist(self, jail):
        _home, _cwd, allowed, outside = jail
        policy = indexing_policy({"rag_indexing_mode": "blacklist",
                                  "rag_denied_roots": [str(outside)]})
        assert policy["mode"] == "blacklist"
        assert _reason(allowed / "x", policy) is None
        assert _reason(outside / "x", policy) == "denied"
        assert _reason(allowed / ".gnupg" / "x", policy) == "credential"

    def test_key_scoped(self, jail):
        home, _cwd, allowed, _outside = jail
        policy = indexing_policy({"allow_network_drives": False},
                                 key_roots=[str(allowed)])
        assert policy == {"mode": "whitelist", "allowed": [allowed.resolve()],
                          "denied": [], "key_scoped": True,
                          "allow_network_drives": False}
        assert _reason(home / "a.txt", policy) == "outside_allowed"
        assert _reason(allowed / "a.txt", policy) is None

    def test_an_unknown_mode_falls_back_to_whitelist(self):
        assert indexing_policy({"rag_indexing_mode": "open"})["mode"] == "whitelist"

    def test_add_paths_with_a_policy(self, jail, base):
        _home, _cwd, allowed, outside = jail
        (allowed / "a.txt").write_text("apple", encoding="utf-8")
        (allowed / ".aws").mkdir()
        (allowed / ".aws" / "creds.txt").write_text("secret", encoding="utf-8")
        (outside / "o.txt").write_text("outside", encoding="utf-8")
        policy = indexing_policy({"rag_allowed_roots": [str(allowed)]})
        c = Collection("kb", base=base).create()
        with pytest.raises(ConfinementError) as exc:
            c.add_paths([outside], policy=policy)
        assert exc.value.reason == "outside_allowed"
        c.add_paths([allowed], policy=policy)
        assert [Path(p).name for p in c.documents()] == ["a.txt"]

    def test_confined_to_reads_meta_only(self, base, folder, tmp_path):
        c = Collection("kb", base=base).create()
        c.add_paths([folder])
        c.add_uploads(_uploads({"u.txt": "upload"}))
        assert Collection.confined_to("kb", [str(folder)], base) is True
        assert Collection.confined_to("kb", [str(folder / "sub")], base) is False
        assert Collection.confined_to("kb", [], base) is True
        assert c.is_confined_to([str(folder)]) is True
        assert c.is_confined_to([str(tmp_path / "elsewhere")]) is False


# --------------------------------------------------------------------------- #
#  Relabel and provenance                                                     #
# --------------------------------------------------------------------------- #

class TestRelabelAndProvenance:
    def test_relabel(self, base):
        for name, model in (("a", "old"), ("b", "old"), ("c", "keep")):
            Collection(name, base=base).create().add_uploads(
                _uploads({"x.txt": "apple"}), embed_fn=_embed, model_name=model)
        Collection("d", base=base).create()
        (base / "d" / "meta.json").write_text("{bad", encoding="utf-8")
        assert relabel_embedding_model("old", "new", base) == (["a", "b"], [])
        assert [Collection(n, base=base).embedding_model() for n in "abc"] == [
            "new", "new", "keep"]
        assert relabel_embedding_model("old", "new", base) == ([], [])

    def test_provenance_report(self):
        base = store.rag_dir()
        Collection("emb", base=base).create().add_uploads(
            _uploads({"x.txt": "apple"}), embed_fn=_embed, model_name="m1")
        Collection("lex", base=base).create().add_uploads(_uploads({"x.txt": "apple"}))
        mixed = Collection("mix", base=base).create()
        mixed.add_uploads(_uploads({"x.txt": "apple"}), embed_fn=_embed, model_name="m1")
        mixed.add_uploads(_uploads({"y.txt": "pear"}), embed_fn=_embed, model_name="m2")
        Collection("cold", base=base).create().add_uploads(
            _uploads({"x.txt": "apple"}), embed_fn=_embed)
        os.utime(base / "cold" / "chunks.jsonl", ns=(1, 1))
        assert collection_provenance_report() == [
            {"name": "cold", "built_with": None, "n_chunks": 1},
            {"name": "emb", "built_with": "m1", "n_chunks": 1},
            {"name": "mix", "built_with": "m1", "n_chunks": 2}]
        assert collection_provenance_report("m1") == [
            {"name": "cold", "built_with": None, "n_chunks": 1},
            {"name": "mix", "built_with": "m1", "n_chunks": 2}]

    def test_provenance_names_a_collection_it_could_not_read(self, monkeypatch, caplog):
        base = store.rag_dir()
        Collection("emb", base=base).create().add_uploads(
            _uploads({"x.txt": "apple"}), embed_fn=_embed, model_name="m1")
        os.utime(base / "emb" / "chunks.jsonl", ns=(1, 1))

        def boom(self):
            raise OSError("disk gone")

        monkeypatch.setattr(Collection, "stats", boom)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert collection_provenance_report() == [
                {"name": "emb", "built_with": None, "n_chunks": None,
                 "reason": "could not be read"}]
        assert any("could not be read for the embedding-switch impact preview "
                   "(OSError: disk gone)" in r.getMessage() for r in caplog.records)

    def test_provenance_note(self):
        assert collection_provenance_note("m", [{"name": "a"}]) == (
            "Switching to 'm' may invalidate the semantic search of 1 existing "
            "collection(s) until they are re-embedded. The exact impact cannot be "
            "confirmed until the new model is loaded and tested - re-embed after "
            "switching if any of them drop to BM25/lexical-only.")
        assert collection_provenance_note("m", [], unchanged=True) == (
            "'m' is already the active embedding model, so there is nothing to "
            "invalidate.")
        assert collection_provenance_note("m", []) == (
            "Switching to 'm' has nothing to invalidate: no existing collection's "
            "semantic search would change.")


# --------------------------------------------------------------------------- #
#  Names replaced on localm.rag.store reach every call site                   #
# --------------------------------------------------------------------------- #

def _counting(monkeypatch, name):
    real = getattr(store, name)
    calls = []

    def wrapper(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(store, name, wrapper)
    return calls


class TestFacadePatch:
    def test_numpy(self, base, monkeypatch):
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple notes", "b.txt": "banana notes"}),
                      embed_fn=_embed)
        with_numpy = c.query("apple", k=2, embed_fn=_embed)
        store._COLLECTION_CACHE.clear()
        monkeypatch.setattr(store, "_numpy", None)
        monkeypatch.setattr(store, "_NUMPY_DEGRADE_LOGGED", set())
        pure = Collection("kb", base=base)
        assert pure._norm_matrix is None
        assert pure.query("apple", k=2, embed_fn=_embed) == with_numpy
        assert pure._norm_matrix is None
        assert store._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert store._NUMPY_DEGRADE_LOGGED == {True}

    def test_a_numpy_stub(self, monkeypatch, caplog):
        monkeypatch.setattr(store, "_numpy", object())
        monkeypatch.setattr(store, "_NUMPY_IS_STUB", True)
        logged = set()
        monkeypatch.setattr(store, "_NUMPY_DEGRADE_LOGGED", logged)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert store._vectors_finite([[1.0, float("nan")]]) is False
            assert store._vectors_finite([[1.0]]) is True
        assert logged == {True}
        assert [r.getMessage() for r in caplog.records][0].startswith(
            "numpy imported as an EMPTY NAMESPACE PACKAGE")

    def test_reading_and_writing_files(self, base, monkeypatch):
        split = _counting(monkeypatch, "split_jsonl")
        writes = _counting(monkeypatch, "_storekit_atomic_write")
        fps = _counting(monkeypatch, "_collection_cache_fingerprint")
        c = Collection("kb", base=base).create()
        c.add_uploads(_uploads({"a.txt": "apple"}), embed_fn=_embed, model_name="m")
        assert writes and split
        n = len(writes)
        relabel_embedding_model("m", "m2", base)
        assert len(writes) == n + 1
        assert fps == []
        Collection("kb", base=base)
        assert len(fps) == 2
        Collection("kb", base=base)
        assert len(fps) == 3

    def test_indexing_entry_points(self, base, folder, monkeypatch):
        confines = _counting(monkeypatch, "confine_index_path")
        extracts = _counting(monkeypatch, "extract_bytes")
        policy = {"mode": "blacklist", "allowed": [], "denied": []}
        c = Collection("kb", base=base).create()
        c.add_paths([folder], policy=policy)
        n = len(confines)
        assert n >= 3
        c.resync(policy=policy)
        assert len(confines) >= n + 3
        c.add_uploads(_uploads({"u.txt": "upload"}))
        assert len(extracts) == 1

    def test_the_data_dir(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt-rag"
        monkeypatch.setattr(store, "rag_dir", lambda: alt)
        Collection("kb").create().add_uploads(_uploads({"a.txt": "apple"}),
                                              embed_fn=_embed, model_name="m")
        assert (alt / "kb" / "meta.json").is_file()
        assert collection_names() == ["kb"]
        assert Collection.peek_stats("kb")["n_docs"] == 1
        assert Collection.load_and_maybe_backfill("kb").documents() == ["upload:a.txt"]
        assert relabel_embedding_model("m", "m2") == (["kb"], [])
        assert collection_provenance_report() == [
            {"name": "kb", "built_with": "m2", "n_chunks": 1}]
        assert delete_collection("kb") is True

    def test_collection_names_and_collection(self, monkeypatch):
        seen = []

        class Fake:
            def __init__(self, name, base=None, *, cache=True):
                seen.append((name, cache))

            @classmethod
            def peek_stats(cls, name, base=None):
                return None

            def stats(self):
                return {"has_vectors": True, "n_chunks": 7}

            def embedding_model(self):
                return "fake"

            def embedding_model_mixed(self):
                return False

        monkeypatch.setattr(store, "collection_names", lambda base=None: ["zz"])
        monkeypatch.setattr(store, "Collection", Fake)
        assert collection_provenance_report() == [
            {"name": "zz", "built_with": "fake", "n_chunks": 7}]
        assert seen == [("zz", False)]
