# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for RAG vector matrix dot product acceleration, BM25 inverted index,
and in-memory collection cache."""

import math
import pytest

from localm.rag import Collection, delete_collection
from localm.rag.bm25 import BM25, _B, _K1, tokenize
from localm.rag.store import (
    _get_cached_collection_data,
)


def _fake_embed_3d(texts):
    results = []
    for t in texts:
        h = hash(t)
        v = [float((h >> 0) & 0xFF), float((h >> 8) & 0xFF), float((h >> 16) & 0xFF)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        results.append([x / norm for x in v])
    return results


class TestBM25InvertedIndex:
    def test_postings_scores_match_full_lexical(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "turbines and renewable wind energy generation",
            "gearboxes in wind turbines require periodic maintenance",
            "unrelated content about cooking sourdough bread and flour",
            "solar panels and photovoltaic renewable energy systems",
        ]
        bm = BM25(docs)
        assert hasattr(bm, "_postings")
        queries = [
            "wind turbines",
            "sourdough bread",
            "energy systems",
            "completely missing terms",
            "",
        ]
        for q in queries:
            postings_scores = bm.scores(q)
            assert len(postings_scores) == len(docs)
            q_terms = [t for t in tokenize(q) if t in bm._idf]
            expected = [0.0] * len(docs)
            for i, d in enumerate(docs):
                d_tokens = tokenize(d)
                d_len = len(d_tokens)
                for term in q_terms:
                    tf = d_tokens.count(term)
                    if tf > 0:
                        num = tf * (_K1 + 1)
                        den = tf + _K1 * (1 - _B + _B * d_len / bm._avg_len)
                        expected[i] += bm._idf[term] * (num / den)
            for p_score, exp_score in zip(postings_scores, expected):
                assert math.isclose(p_score, exp_score, rel_tol=1e-5, abs_tol=1e-5)


class TestVectorMatrixAcceleration:
    def test_norm_matrix_matches_scalar_cosine(self, tmp_path):
        base = tmp_path / "rag"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        for i in range(10):
            (docs_dir / f"doc_{i}.txt").write_text(
                f"Document {i} discussing topic {i % 3} with details.",
                encoding="utf-8",
            )

        coll = Collection("accel_test", base=base).create()
        coll.add_paths([docs_dir], embed_fn=_fake_embed_3d)

        # Re-load to trigger _load and cached norm matrix computation
        loaded = Collection("accel_test", base=base)
        import localm.rag.store as rag_store
        if rag_store._numpy is not None:
            assert loaded._norm_matrix is not None
        else:
            assert loaded._norm_matrix is None
        assert loaded._vectors is not None
        assert len(loaded._vectors) == len(loaded._chunks)

        query_text = "topic 1 details"
        query_vec = _fake_embed_3d([query_text])[0]

        vector_scores = loaded._vector_scores(query_text, embed_fn=_fake_embed_3d)
        assert vector_scores is not None
        assert len(vector_scores) == len(loaded._chunks)

        qv_norm = math.sqrt(sum(x * x for x in query_vec))
        expected_raw = []
        for v in loaded._vectors:
            if not v:
                expected_raw.append(0.0)
                continue
            v_norm = math.sqrt(sum(x * x for x in v))
            denom = qv_norm * v_norm
            if denom == 0.0:
                expected_raw.append(0.0)
            else:
                dot = sum(a * b for a, b in zip(query_vec, v))
                expected_raw.append(max(0.0, dot / denom))

        top = max(expected_raw) if expected_raw else 0.0
        expected = [s / top for s in expected_raw] if top > 0 else expected_raw

        for actual, exp in zip(vector_scores, expected):
            assert math.isclose(actual, exp, rel_tol=1e-5, abs_tol=1e-5)

    def test_vector_scores_with_empty_and_zero_vectors(self, tmp_path):
        base = tmp_path / "rag"
        coll = Collection("empty_vec_test", base=base).create()
        coll._chunks = [{"text": "chunk1"}, {"text": "chunk2"}, {"text": "chunk3"}]
        coll._vectors = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
        coll._vec_dim = 3

        scores = coll._vector_scores("query", embed_fn=lambda _: [[1.0, 0.0, 0.0]])
        assert scores is not None
        assert len(scores) == 3
        assert math.isclose(scores[0], 1.0, rel_tol=1e-5)
        assert scores[1] > 0.0
        assert scores[2] == 0.0

        zero_scores = coll._vector_scores("query", embed_fn=lambda _: [[0.0, 0.0, 0.0]])
        assert zero_scores == [0.0, 0.0, 0.0]


class TestCollectionCache:
    def test_cache_hit_and_invalidation(self, tmp_path):
        base = tmp_path / "rag"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "file.txt").write_text("initial content", encoding="utf-8")

        c1 = Collection("cache_test", base=base).create()
        c1.add_paths([docs_dir], embed_fn=_fake_embed_3d)

        # First load after creation/save populates the cache
        c2 = Collection("cache_test", base=base)
        cached_entry = _get_cached_collection_data(base / "cache_test")
        assert cached_entry is not None
        assert len(cached_entry.chunks) == len(c2._chunks)

        # Subsequent load reads from cache
        c3 = Collection("cache_test", base=base)
        assert len(c3._chunks) == len(c2._chunks)

        # Adding new document invalidates cache
        (docs_dir / "file2.txt").write_text("second file content", encoding="utf-8")
        c2.add_paths([docs_dir], embed_fn=_fake_embed_3d)

        c4 = Collection("cache_test", base=base)
        assert len(c4._chunks) > len(c3._chunks)

        # Deleting collection purges cache entry
        delete_collection("cache_test", base=base)
        assert _get_cached_collection_data(base / "cache_test") is None


class TestRagPluginRoutes:
    @pytest.mark.anyio
    async def test_rag_delete_does_not_load_collection(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "home_dir", lambda: tmp_path)
        base = tmp_path / "rag"
        Collection("to_delete", base=base).create()
        (base / "to_delete" / "chunks.jsonl").write_text("dummy", encoding="utf-8")

        load_called = []
        orig_load = Collection._load

        def tracked_load(self):
            load_called.append(self.name)
            return orig_load(self)

        monkeypatch.setattr(Collection, "_load", tracked_load)

        from localm.plugins.builtin.rag.plug import rag_delete
        from starlette.applications import Starlette
        from starlette.requests import Request
        app = Starlette()
        scope = {"type": "http", "headers": [], "app": app}
        req = Request(scope)

        res = await rag_delete("to_delete", req)
        assert res["status"] == "deleted"
        assert not (base / "to_delete" / "meta.json").exists()
        assert len(load_called) == 0

    @pytest.mark.anyio
    async def test_rag_query_executes_in_executor(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "home_dir", lambda: tmp_path)
        base = tmp_path / "rag"
        coll = Collection("to_query", base=base).create()
        coll._chunks = [{"text": "renewable energy wind", "source": "f.txt", "pos": 0}]
        coll._save()

        from localm.plugins.builtin.rag.plug import RagQueryRequest, rag_query
        from starlette.applications import Starlette
        from starlette.requests import Request
        app = Starlette()
        scope = {"type": "http", "headers": [], "app": app}
        req = Request(scope)

        body = RagQueryRequest(query="wind", k=2)
        res = await rag_query("to_query", body, req)
        assert res["collection"] == "to_query"
        assert len(res["hits"]) > 0
