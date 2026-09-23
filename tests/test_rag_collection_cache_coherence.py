# SPDX-License-Identifier: AGPL-3.0-or-later
"""The in-memory RAG collection cache serves only coherent, immutable snapshots.

Every test drives the real Collection against a throwaway directory. Where a
test races a reader against a writer, the writer is a real add/remove that
commits to disk while the reader is between its reads and the point where it
would store what it read.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import localm.rag.store as store
from localm.rag.store import Collection


@pytest.fixture(autouse=True)
def _empty_cache():
    store._COLLECTION_CACHE.clear()
    yield
    store._COLLECTION_CACHE.clear()


def _disk_docs(coll_dir: Path) -> set:
    meta = json.loads((coll_dir / "meta.json").read_text(encoding="utf-8"))
    return set(meta["docs"])


def _disk_chunk_sources(coll_dir: Path) -> set:
    text = (coll_dir / "chunks.jsonl").read_text(encoding="utf-8")
    return {json.loads(line)["source"] for line in text.split("\n") if line.strip()}


def _upload(name: str, text: str) -> dict:
    return {"filename": name, "data": text.encode("utf-8")}


def _commit_during_next_cache_fill(monkeypatch, commit) -> dict:
    """Run *commit* inside the first cache fingerprint taken after a reader has
    read chunks.jsonl, i.e. after the reader's file reads and before it could
    store them. Returns the state dict; ``state["fired"]`` says it ran."""
    state = {"read": False, "fired": False}
    real_split = store.split_jsonl
    real_fingerprint = store._collection_cache_fingerprint

    def split(text):
        state["read"] = True
        return real_split(text)

    def fingerprint(coll_dir):
        if state["read"] and not state["fired"]:
            state["fired"] = True
            commit()
        return real_fingerprint(coll_dir)

    monkeypatch.setattr(store, "split_jsonl", split)
    monkeypatch.setattr(store, "_collection_cache_fingerprint", fingerprint)
    return state


def _constant_fingerprint(monkeypatch):
    """Make every fingerprint identical, as if no write were ever visible in the
    file stats."""
    monkeypatch.setattr(store, "_collection_cache_fingerprint",
                        lambda coll_dir: {"meta": None, "chunks": None,
                                          "vectors": None})


def _upload_from_another_process(base: Path, name: str, text: str) -> None:
    """Index an upload into collection 'kb' from a separate Python process, so
    none of this process's cache bookkeeping sees the write."""
    repo_root = Path(store.__file__).resolve().parents[2]
    script = ("import sys; from pathlib import Path; "
              "from localm.rag.store import Collection; "
              "Collection('kb', base=Path(sys.argv[1])).add_uploads("
              "[{'filename': sys.argv[2], 'data': sys.argv[3].encode()}])")
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    done = subprocess.run([sys.executable, "-c", script, str(base), name, text],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def _embed(texts):
    vocab = ("apples", "bananas", "cherries", "notes", "salary")
    return [[1.0 if w in t.lower() else 0.0 for w in vocab] + [0.01] for t in texts]


class TestReaderRaceWithCommittingWriter:
    def test_upload_committed_during_a_readers_fill_is_never_erased(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])

        state = _commit_during_next_cache_fill(
            monkeypatch,
            lambda: Collection("kb", base=base).add_uploads(
                [_upload("b.txt", "bananas are yellow fruit")]))
        Collection("kb", base=base)
        assert state["fired"], "the concurrent upload never ran"

        Collection("kb", base=base).add_uploads(
            [_upload("c.txt", "cherries are small fruit")])

        want = {"upload:a.txt", "upload:b.txt", "upload:c.txt"}
        assert _disk_docs(base / "kb") == want, "a committed upload was erased"
        assert _disk_chunk_sources(base / "kb") == want
        hits = Collection("kb", base=base).query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"]

    def test_in_process_write_blocks_the_store_even_when_stats_cannot_show_it(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        _constant_fingerprint(monkeypatch)
        state = _commit_during_next_cache_fill(
            monkeypatch,
            lambda: Collection("kb", base=base).add_uploads(
                [_upload("b.txt", "bananas are yellow fruit")]))
        Collection("kb", base=base)
        assert state["fired"], "the concurrent upload never ran"

        hits = Collection("kb", base=base).query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"], (
            "a reader cached the state from before a committed upload")

    def test_other_process_commit_during_a_readers_fill_is_not_cached(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        state = _commit_during_next_cache_fill(
            monkeypatch,
            lambda: _upload_from_another_process(base, "b.txt",
                                                 "bananas are yellow fruit"))
        Collection("kb", base=base)
        assert state["fired"], "the other process's upload never ran"
        assert "upload:b.txt" in _disk_docs(base / "kb")

        hits = Collection("kb", base=base).query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"], (
            "a reader cached files another process rewrote while it read them")

    def test_write_drops_the_cached_entry_even_when_stats_cannot_show_it(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        _constant_fingerprint(monkeypatch)
        Collection("kb", base=base)
        Collection("kb", base=base).add_uploads(
            [_upload("b.txt", "bananas are yellow fruit")])

        hits = Collection("kb", base=base).query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"], (
            "an entry cached before a write was served after it")

    def test_writer_reads_disk_even_when_the_cache_is_wrong(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        _constant_fingerprint(monkeypatch)
        Collection("kb", base=base)

        _upload_from_another_process(base, "b.txt", "bananas are yellow fruit")
        assert "upload:b.txt" in _disk_docs(base / "kb"), "the other process did not write"

        Collection("kb", base=base).add_uploads(
            [_upload("c.txt", "cherries are small fruit")])

        want = {"upload:a.txt", "upload:b.txt", "upload:c.txt"}
        assert _disk_docs(base / "kb") == want, (
            "a writer trusted a stale cache entry and erased another process's upload")
        assert _disk_chunk_sources(base / "kb") == want


class TestFailedSave:
    def test_failed_save_leaves_no_phantom_document_behind(
            self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        src = tmp_path / "src"
        src.mkdir()
        a = src / "a.txt"
        a.write_text("apples are red fruit", encoding="utf-8")
        b = src / "b.txt"
        b.write_text("bananas are yellow fruit", encoding="utf-8")
        Collection("kb", base=base).create().add_paths([a])
        Collection("kb", base=base)

        real_write = store._storekit_atomic_write
        failures = []

        def no_space_for_meta(path, data):
            if Path(path).name == "meta.json" and not failures:
                failures.append(path)
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write(path, data)

        monkeypatch.setattr(store, "_storekit_atomic_write", no_space_for_meta)
        with pytest.raises(OSError):
            Collection("kb", base=base).add_paths([b])
        assert failures, "the injected write failure never fired"
        monkeypatch.setattr(store, "_storekit_atomic_write", real_write)

        assert str(b.resolve()) not in Collection("kb", base=base).documents(), (
            "a failed save left the document listed")
        result = Collection("kb", base=base).add_paths([b])
        assert result["added"] == 1, result
        assert str(b.resolve()) in _disk_docs(base / "kb")
        hits = Collection("kb", base=base).query("bananas")
        assert [h["source"] for h in hits] == [str(b.resolve())]


class TestSnapshotIsolation:
    def test_mutating_a_served_instance_never_reaches_the_cache(self, tmp_path):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit"),
             _upload("b.txt", "bananas are yellow fruit")], embed_fn=_embed)
        Collection("kb", base=base)
        served = Collection("kb", base=base)
        assert served._vectors is not None

        served._meta["docs"]["upload:zzz.txt"] = {"chunks": 1}
        served._meta.setdefault("roots", {})["/nowhere"] = {"added": 0}
        served._chunks.append({"text": "phantom", "source": "upload:zzz.txt"})
        served._vectors.append([0.0] * 6)
        try:
            served._chunks[0]["text"] = "rewritten"
        except TypeError:
            pass
        try:
            served._vectors[0][0] = 99.0
        except TypeError:
            pass

        fresh = Collection("kb", base=base)
        assert "upload:zzz.txt" not in fresh._meta["docs"]
        assert "/nowhere" not in fresh.roots()
        assert [c["text"] for c in fresh._chunks] == [
            "apples are red fruit", "bananas are yellow fruit"]
        assert len(fresh._vectors) == 2
        assert list(fresh._vectors[0]) == _embed(["apples are red fruit"])[0]


class TestLexicalIndexBelongsToItsSnapshot:
    def test_index_built_for_an_older_state_is_never_attached_to_a_newer_one(
            self, tmp_path):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit"),
             _upload("b.txt", "bananas are yellow fruit")])
        r1 = Collection("kb", base=base)
        assert len(r1._chunks) == 2
        Collection("kb", base=base).remove_doc("upload:a.txt")
        Collection("kb", base=base)
        r1.query("apples")

        r3 = Collection("kb", base=base)
        hits = r3.query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"]
        assert len(r3._bm25._lengths) == len(r3._chunks)

    def test_a_still_referenced_old_snapshot_is_not_attached_either(self, tmp_path):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit"),
             _upload("b.txt", "bananas are yellow fruit")])
        r1 = Collection("kb", base=base)
        old = store._get_cached_collection_data(base / "kb")
        assert old is not None
        Collection("kb", base=base).remove_doc("upload:a.txt")
        Collection("kb", base=base)
        r1.query("apples")

        r3 = Collection("kb", base=base)
        hits = r3.query("bananas")
        assert [h["source"] for h in hits] == ["upload:b.txt"]
        assert len(r3._bm25._lengths) == len(r3._chunks)
        del old


class TestVectorMatrixFollowsItsVectors:
    def test_reassigned_vectors_of_the_same_length_are_scored_as_themselves(
            self, tmp_path):
        base = tmp_path / "rag"
        c = Collection("kb", base=base).create()
        c._chunks = [{"text": "first", "source": "s", "pos": 0},
                     {"text": "second", "source": "s", "pos": 1}]
        c._vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        c._vec_dim = 3
        c._save()

        r = Collection("kb", base=base)

        def query_vec(texts):
            return [[1.0, 0.0, 0.0]]

        before = r._vector_scores("q", query_vec)
        assert before[0] > before[1]
        r._vectors = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        after = r._vector_scores("q", query_vec)
        assert after[1] > after[0], "scored against the vectors it was built from before"


class TestUnusualFiles:
    def test_collection_with_a_malformed_chunk_line_is_cached_under_its_path(
            self, tmp_path):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        chunks = base / "kb" / "chunks.jsonl"
        chunks.write_text(chunks.read_text(encoding="utf-8") + "\n{not json",
                          encoding="utf-8")

        loaded = Collection("kb", base=base)
        assert loaded.chunks_bad_lines == 1
        assert store._get_cached_collection_data(base / "kb") is not None, (
            "the snapshot was not stored under the collection's own key")

        Collection("kb", base=base).add_uploads(
            [_upload("b.txt", "bananas are yellow fruit")])
        assert len(store._COLLECTION_CACHE) == 0, "a write left an entry behind"

    def test_deeply_nested_json_loads_through_the_cache(self, tmp_path):
        base = tmp_path / "rag"
        Collection("kb", base=base).create().add_uploads(
            [_upload("a.txt", "apples are red fruit")])
        deep = "[" * 1500 + "]" * 1500
        meta_path = base / "kb" / "meta.json"
        meta_text = meta_path.read_text(encoding="utf-8").rstrip()
        meta_path.write_text(meta_text[:-1] + ', "deep": ' + deep + "}",
                             encoding="utf-8")
        (base / "kb" / "chunks.jsonl").write_text(
            '{"text": "bananas are yellow fruit", "source": "upload:a.txt", '
            '"pos": 0, "deep": ' + deep + "}", encoding="utf-8")

        Collection("kb", base=base)
        served = Collection("kb", base=base)

        assert store._get_cached_collection_data(base / "kb") is not None
        assert [h["source"] for h in served.query("bananas")] == ["upload:a.txt"]


def _scoped_app(tmp_path, monkeypatch, *, rag_roots):
    from fastapi import FastAPI
    from localm import auth
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    import localm.config as cfg
    home = tmp_path
    data = home / ".localm"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-xyz")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(cfg, "HOME_DIR", data)
    monkeypatch.setattr(cfg, "MODELS_DIR", data / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", data / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", data / "registry.json")
    scoped = auth.create_key("dev", ["rag"], rag_roots=rag_roots)["key"]
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, {"Authorization": f"Bearer {scoped}"}


def _two_root_collection(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    notes = allowed / "notes.txt"
    notes.write_text("meeting notes about the roadmap", encoding="utf-8")
    salary = outside / "salary.txt"
    salary.write_text("SALARYSECRET payroll figures", encoding="utf-8")
    Collection("kb").create().add_paths([notes, salary])
    return allowed, notes.resolve(), salary.resolve()


class TestConfinedQueryServesOnlyWhatItChecked:
    def test_removed_out_of_root_document_is_not_served_after_a_race(
            self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        allowed = tmp_path / "allowed"
        app, hdr = _scoped_app(tmp_path, monkeypatch, rag_roots=[str(allowed)])
        _, _, salary = _two_root_collection(tmp_path)

        state = _commit_during_next_cache_fill(
            monkeypatch, lambda: Collection("kb").remove_doc(str(salary)))
        Collection("kb")
        assert state["fired"], "the concurrent removal never ran"
        assert str(salary) not in _disk_docs(store.rag_dir() / "kb")

        with TestClient(app) as client:
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "payroll figures", "k": 5},
                            headers=hdr)
        assert "SALARYSECRET" not in r.text, "a removed out-of-root document was served"
        assert r.status_code == 200, r.text

    def test_confinement_is_decided_on_the_chunks_that_would_be_served(
            self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        allowed = tmp_path / "allowed"
        app, hdr = _scoped_app(tmp_path, monkeypatch, rag_roots=[str(allowed)])
        _, _, salary = _two_root_collection(tmp_path)

        meta_path = store.rag_dir() / "kb" / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["docs"][str(salary)]
        store._storekit_atomic_write(meta_path, json.dumps(meta, indent=2))
        assert str(salary) in _disk_chunk_sources(store.rag_dir() / "kb")

        with TestClient(app) as client:
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "payroll figures", "k": 5},
                            headers=hdr)
        assert "SALARYSECRET" not in r.text, (
            "chunks from outside the key's roots were served because only "
            "meta.json was checked")
        assert r.status_code == 403, r.text

    def test_owner_query_is_not_confined(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        allowed = tmp_path / "allowed"
        app, _hdr = _scoped_app(tmp_path, monkeypatch, rag_roots=[str(allowed)])
        _two_root_collection(tmp_path)
        with TestClient(app) as client:
            r = client.post("/api/rag/collections/kb/query",
                            json={"query": "payroll figures", "k": 5},
                            headers={"Authorization": "Bearer owner-key-xyz"})
        assert r.status_code == 200, r.text
        assert "SALARYSECRET" in r.text

