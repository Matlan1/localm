# SPDX-License-Identifier: AGPL-3.0-or-later
"""The in-memory RAG collection cache is bounded by memory, and bulk or report
paths that load every collection never populate it."""

from __future__ import annotations

import gc
import json
import random
import threading
import tracemalloc
from pathlib import Path

import pytest

import localm.rag.store as store
from localm.rag.store import Collection


@pytest.fixture(autouse=True)
def _empty_cache():
    store._COLLECTION_CACHE.clear()
    yield
    store._COLLECTION_CACHE.clear()


def _build(base: Path, name: str, *, n_chunks: int = 500, dim: int = 32,
           seed: int = 0, model: "str | None" = None) -> None:
    rnd = random.Random(seed)
    words = [f"w{seed}x{i}" for i in range(400)]
    c = Collection(name, base=base).create()
    sources = [f"upload:{name}-{i}.txt" for i in range(n_chunks)]
    c._chunks = [{"text": " ".join(rnd.choice(words) for _ in range(30)),
                  "source": src, "pos": 0} for src in sources]
    c._vectors = [[rnd.random() for _ in range(dim)] for _ in range(n_chunks)]
    c._vec_dim = dim
    c._meta["docs"] = {src: {"chunks": 1, "uploaded": True} for src in sources}
    if model:
        c._meta["embedding_model"] = model
    c._save()


def _query_vec(dim: int):
    return lambda texts: [[0.5] * dim for _ in texts]


def _retained_by_cache(base: Path, names: list, dim: int) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for name in names:
            coll = Collection(name, base=base)
            coll.query(f"w0x1 w1x2 w{names.index(name)}x3", embed_fn=_query_vec(dim))
            del coll
        gc.collect()
        return tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()


class TestByteBudget:
    def test_cache_retains_no_more_memory_than_its_budget(self, tmp_path, monkeypatch):
        budget = 6 * 2**20
        monkeypatch.setattr(store, "_COLLECTION_CACHE_MAX_BYTES", budget, raising=False)
        base = tmp_path / "rag"
        names = [f"kb{i}" for i in range(6)]
        for i, name in enumerate(names):
            _build(base, name, seed=i)
        store._COLLECTION_CACHE.clear()

        retained = _retained_by_cache(base, names, 32)

        assert len(store._COLLECTION_CACHE) >= 1, "nothing was cached at all"
        assert retained <= budget, (
            f"the cache retained {retained / 2**20:.1f} MiB against a "
            f"{budget / 2**20:.1f} MiB budget")
        assert store._COLLECTION_CACHE.total_bytes() <= budget

    def test_estimate_tracks_real_retained_memory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "_COLLECTION_CACHE_MAX_BYTES", 2**30, raising=False)
        base = tmp_path / "rag"
        _build(base, "kb", n_chunks=800, dim=64)
        store._COLLECTION_CACHE.clear()

        retained = _retained_by_cache(base, ["kb"], 64)

        estimate = store._COLLECTION_CACHE.total_bytes()
        assert len(store._COLLECTION_CACHE) == 1
        assert retained * 0.9 <= estimate <= retained * 2.5, (
            f"estimate {estimate} bytes for {retained} bytes actually retained")

    def test_least_recently_used_entry_is_evicted_first(self, tmp_path, monkeypatch):
        base = tmp_path / "rag"
        for i, name in enumerate(("a", "b", "c")):
            _build(base, name, n_chunks=200, seed=i)
        store._COLLECTION_CACHE.clear()
        Collection("a", base=base)
        one = store._COLLECTION_CACHE.total_bytes()
        assert one > 0
        store._COLLECTION_CACHE.clear()
        monkeypatch.setattr(store, "_COLLECTION_CACHE_MAX_BYTES", int(one * 2.5))

        Collection("a", base=base)
        Collection("b", base=base)
        Collection("a", base=base)
        Collection("c", base=base)

        assert store._get_cached_collection_data(base / "b") is None
        assert store._get_cached_collection_data(base / "a") is not None
        assert store._get_cached_collection_data(base / "c") is not None

    def test_entry_larger_than_the_budget_is_served_but_not_cached(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "_COLLECTION_CACHE_MAX_BYTES", 1, raising=False)
        base = tmp_path / "rag"
        _build(base, "kb", n_chunks=50)
        store._COLLECTION_CACHE.clear()

        coll = Collection("kb", base=base)
        hits = coll.query("w0x1 w0x2")

        assert hits, "an uncacheable collection must still answer"
        assert len(store._COLLECTION_CACHE) == 0

    def test_entry_count_ceiling_still_applies(self, tmp_path):
        base = tmp_path / "rag"
        names = [f"kb{i}" for i in range(12)]
        for i, name in enumerate(names):
            _build(base, name, n_chunks=5, seed=i)
        store._COLLECTION_CACHE.clear()
        for name in names:
            Collection(name, base=base)
        assert 1 <= len(store._COLLECTION_CACHE) <= store._MAX_CACHED_COLLECTIONS


class TestLexicalIndexMemory:
    def test_lexical_index_holds_little_beyond_its_postings(self):
        from localm.rag.bm25 import BM25
        rnd = random.Random(3)
        words = [f"t{i}" for i in range(3000)]
        texts = [" ".join(rnd.choice(words) for _ in range(80)) for _ in range(600)]
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            index = BM25(texts)
            gc.collect()
            held = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()

        postings = sum(len(p) for p in index._postings.values())
        assert index.scores("t1 t2")
        assert held <= postings * 100, (
            f"the index holds {held / postings:.0f} bytes per posting")


def _isolated_home(tmp_path, monkeypatch) -> Path:
    import localm.config as cfg
    data = tmp_path / "home"
    data.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.setattr(cfg, "HOME_DIR", data)
    monkeypatch.setattr(cfg, "CONFIG_FILE", data / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", data / "registry.json")
    return data / "rag"


class TestBulkPathsDoNotPopulateTheCache:
    def test_embedding_switch_dimension_report(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        for i in range(4):
            _build(base, f"kb{i}", n_chunks=20, seed=i)
        store._COLLECTION_CACHE.clear()
        from localm.plugins.builtin.rag.plug import _collection_dim_report

        report = _collection_dim_report(64)

        assert sorted(d["name"] for d in report["degrades"]) == [
            "kb0", "kb1", "kb2", "kb3"]
        assert len(store._COLLECTION_CACHE) == 0

    def test_model_rename_label_migration(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        for i in range(3):
            _build(base, f"kb{i}", n_chunks=20, seed=i, model="old-model")
        _build(base, "other", n_chunks=20, seed=9, model="unrelated-model")
        store._COLLECTION_CACHE.clear()
        from localm.model_manager.registry import _migrate_model_references

        notes = _migrate_model_references("old-model", "new-model")

        labels = {n: json.loads((base / n / "meta.json").read_text(encoding="utf-8"))
                  .get("embedding_model") for n in ("kb0", "kb1", "kb2", "other")}
        assert labels == {"kb0": "new-model", "kb1": "new-model",
                          "kb2": "new-model", "other": "unrelated-model"}, notes
        assert len(store._COLLECTION_CACHE) == 0

    def test_model_rename_leaves_a_collection_another_writer_holds(
            self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        _build(base, "kb0", n_chunks=5, model="old-model")
        monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "0.2")
        held, release = threading.Event(), threading.Event()

        def hold():
            with store._collection_lock("kb0"):
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(5), "the holder never took the lock"
        from localm.model_manager.registry import _migrate_model_references
        try:
            notes = _migrate_model_references("old-model", "new-model")
        finally:
            release.set()
            holder.join(5)

        meta = json.loads((base / "kb0" / "meta.json").read_text(encoding="utf-8"))
        assert meta["embedding_model"] == "old-model", (
            "the rename rewrote meta.json while another writer held the collection")
        assert any("kb0" in n and "old-model" in n for n in notes), notes

    def test_cli_rag_list(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        for i in range(3):
            _build(base, f"kb{i}", n_chunks=20, seed=i)
        store._COLLECTION_CACHE.clear()
        from click.testing import CliRunner
        from localm.cli.rag import rag_group

        result = CliRunner().invoke(rag_group, ["list"])

        assert result.exit_code == 0, result.output
        assert "kb0" in result.output and "kb2" in result.output
        assert len(store._COLLECTION_CACHE) == 0

    def test_coder_rag_list_tool(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        for i in range(3):
            _build(base, f"kb{i}", n_chunks=20, seed=i)
        store._COLLECTION_CACHE.clear()
        from localm.plugins.coder.tools.rag import tool_rag_list_collections

        result = tool_rag_list_collections(tmp_path)

        assert "kb1" in result.output
        assert len(store._COLLECTION_CACHE) == 0

    def test_listing_cold_path_while_the_collection_is_busy(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        _build(base, "kb0", n_chunks=20)
        store._COLLECTION_CACHE.clear()
        from localm.rag.collection_lock import collection_write_lock, lock_path_for

        with collection_write_lock(lock_path_for(base / "kb0"), collection="kb0",
                                   op="a test hold"):
            stats = Collection.load_and_maybe_backfill("kb0").stats()

        assert stats["n_chunks"] == 20
        assert len(store._COLLECTION_CACHE) == 0

    def test_listing_cold_path_backfill_reads_disk_not_the_cache(
            self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        _build(base, "kb", n_chunks=20)
        other = tmp_path / "other"
        _build(other, "kb", n_chunks=25)
        monkeypatch.setattr(store, "_collection_cache_fingerprint",
                            lambda coll_dir: {"meta": None, "chunks": None,
                                              "vectors": None})
        store._COLLECTION_CACHE.clear()
        Collection("kb")
        for name in ("chunks.jsonl", "vectors.json", "meta.json"):
            (base / "kb" / name).write_bytes((other / "kb" / name).read_bytes())

        stats = Collection.load_and_maybe_backfill("kb").stats()

        meta = json.loads((base / "kb" / "meta.json").read_text(encoding="utf-8"))
        assert meta["_stats_cache"]["n_chunks"] == 25, (
            "the backfill persisted stats for data the cache held, not the files")
        assert stats["n_chunks"] == 25

    def test_listing_cold_path_backfill(self, tmp_path, monkeypatch):
        base = _isolated_home(tmp_path, monkeypatch)
        for i in range(3):
            _build(base, f"kb{i}", n_chunks=20, seed=i)
        store._COLLECTION_CACHE.clear()

        stats = [Collection.load_and_maybe_backfill(f"kb{i}").stats()
                 for i in range(3)]

        assert [s["n_chunks"] for s in stats] == [20, 20, 20]
        assert len(store._COLLECTION_CACHE) == 0
