# SPDX-License-Identifier: AGPL-3.0-or-later
"""``Collection.query(rerank_fn=...)`` and the rerank evaluation harness."""

import math

import pytest

from localm.rag import Collection
from tests import rag_rerank_eval as ev

_NOISE = "filler words about unrelated gardening topics " * 12


@pytest.fixture
def coll(tmp_path):
    """Five documents that all mention 'harbor crane'; BM25 ranks them by how
    often the phrase appears, so the blended order is known."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for n in range(1, 6):
        body = ("harbor crane " * (6 - n)) + _NOISE
        (docs / f"doc{n}.txt").write_text(body, encoding="utf-8")
    c = Collection("kb", base=tmp_path / "rag").create()
    assert c.add_paths([str(docs)])["failed"] == []
    return c


def _names(hits):
    return [h["source"].replace("\\", "/").rsplit("/", 1)[-1] for h in hits]


def _by_text(coll, preferred):
    """rerank_fn scoring highest the chunk whose source is *preferred*."""
    by_text = {c["text"]: c["source"] for c in coll._chunks}
    calls = []

    def fn(query, texts):
        calls.append((query, list(texts)))
        return [1.0 if by_text[t].replace("\\", "/").endswith(preferred) else 0.0
                for t in texts]

    return fn, calls


class TestRerankStage:
    def test_without_rerank_fn_order_and_fields_are_unchanged(self, coll):
        hits = coll.query("harbor crane", k=3)
        assert _names(hits) == ["doc1.txt", "doc2.txt", "doc3.txt"]
        assert all("rerank_score" not in h for h in hits)
        assert coll.rerank_degrade_reason is None

    def test_rerank_promotes_a_candidate_outside_the_top_k(self, coll):
        fn, calls = _by_text(coll, "doc5.txt")
        hits = coll.query("harbor crane", k=2, rerank_fn=fn, rerank_candidates=5)
        assert _names(hits)[0] == "doc5.txt"
        assert len(hits) == 2
        assert hits[0]["rerank_score"] == 1.0
        assert calls[0][0] == "harbor crane"
        assert len(calls[0][1]) == 5

    def test_candidate_count_bounds_what_the_reranker_sees(self, coll):
        fn, calls = _by_text(coll, "doc5.txt")
        hits = coll.query("harbor crane", k=2, rerank_fn=fn, rerank_candidates=3)
        assert len(calls[0][1]) == 3
        assert "doc5.txt" not in _names(hits)

    def test_pool_is_never_smaller_than_k(self, coll):
        fn, calls = _by_text(coll, "doc4.txt")
        hits = coll.query("harbor crane", k=4, rerank_fn=fn, rerank_candidates=1)
        assert len(calls[0][1]) == 4
        assert _names(hits)[0] == "doc4.txt"

    def test_reranker_receives_chunks_in_blended_order(self, coll):
        fn, calls = _by_text(coll, "doc1.txt")
        coll.query("harbor crane", k=3, rerank_fn=fn, rerank_candidates=5)
        by_text = {c["text"]: c["source"] for c in coll._chunks}
        order = [by_text[t].replace("\\", "/").rsplit("/", 1)[-1]
                 for t in calls[0][1]]
        assert order == ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]

    def test_single_candidate_is_not_sent_to_the_reranker(self, coll):
        fn, calls = _by_text(coll, "doc1.txt")
        hits = coll.query("harbor crane", k=1, rerank_fn=fn, rerank_candidates=1)
        assert calls == []
        assert len(hits) == 1 and "rerank_score" not in hits[0]

    @pytest.mark.parametrize("bad, fragment", [
        (lambda q, t: (_ for _ in ()).throw(RuntimeError("model gone")),
         "reranking failed (RuntimeError)"),
        (lambda q, t: [0.5], "returned 1 scores for 5 chunks"),
        (lambda q, t: [math.nan] + [0.0] * (len(t) - 1), "non-finite"),
        (lambda q, t: [math.inf] * len(t), "non-finite"),
    ])
    def test_a_failing_reranker_keeps_the_blended_order_and_says_why(
            self, coll, bad, fragment):
        hits = coll.query("harbor crane", k=3, rerank_fn=bad, rerank_candidates=5)
        assert _names(hits) == ["doc1.txt", "doc2.txt", "doc3.txt"]
        assert all("rerank_score" not in h for h in hits)
        assert fragment in coll.rerank_degrade_reason

    def test_degrade_reason_clears_on_the_next_good_query(self, coll):
        coll.query("harbor crane", k=3, rerank_fn=lambda q, t: [], rerank_candidates=5)
        assert coll.rerank_degrade_reason
        coll.query("harbor crane", k=3)
        assert coll.rerank_degrade_reason is None

    def test_degrade_reason_never_carries_the_exception_text(self, coll):
        def boom(q, t):
            raise RuntimeError("secret chunk text leaked here")
        coll.query("harbor crane", k=3, rerank_fn=boom, rerank_candidates=5)
        assert "secret" not in coll.rerank_degrade_reason

    def test_the_relevance_floor_applies_before_reranking(self, tmp_path):
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "full.txt").write_text("alpha beta gamma delta " + _NOISE,
                                       encoding="utf-8")
        (docs / "partial.txt").write_text("alpha " + _NOISE, encoding="utf-8")
        c = Collection("floor", base=tmp_path / "rag").create()
        c.add_paths([str(docs)])
        fn, calls = _by_text(c, "partial.txt")
        unfloored = c.query("alpha beta gamma delta", k=2, rerank_fn=fn,
                            rerank_candidates=5)
        assert _names(unfloored)[0] == "partial.txt"
        hits = c.query("alpha beta gamma delta", k=2, relevant_only=True,
                       rerank_fn=fn, rerank_candidates=5)
        assert _names(hits) == ["full.txt"]

    def test_equal_rerank_scores_keep_the_blended_order(self, coll):
        hits = coll.query("harbor crane", k=3, rerank_candidates=5,
                          rerank_fn=lambda q, t: [0.25] * len(t))
        assert _names(hits) == ["doc1.txt", "doc2.txt", "doc3.txt"]
        assert all(h["rerank_score"] == 0.25 for h in hits)

    def test_a_blank_query_never_reaches_the_reranker(self, coll):
        fn, calls = _by_text(coll, "doc1.txt")
        assert coll.query("   ", rerank_fn=fn) == []
        assert calls == []


class TestHarnessMetrics:
    def test_hit_and_reciprocal_rank(self):
        assert ev.hit_at([0, 0, 1], 2) == 0.0
        assert ev.hit_at([0, 0, 1], 3) == 1.0
        assert ev.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
        assert ev.reciprocal_rank([0, 0, 0]) == 0.0

    def test_ndcg_is_one_for_an_ideal_ranking_and_less_when_demoted(self):
        assert ev.ndcg_at([1, 1, 0], 3, 2) == pytest.approx(1.0)
        assert ev.ndcg_at([0, 1, 1], 3, 2) < 1.0
        assert ev.ndcg_at([0, 0, 0], 3, 2) == 0.0
        assert ev.ndcg_at([1], 3, 0) == 0.0

    def test_relevance_needs_the_right_document_and_a_gold_phrase(self):
        q = {"doc": "a.md", "gold": ["two words"]}
        assert ev.is_relevant({"source": "x/a.md", "text": "has two\n words here"}, q)
        assert not ev.is_relevant({"source": "x/b.md", "text": "two words"}, q)
        assert not ev.is_relevant({"source": "x/a.md", "text": "two other words"}, q)

    def test_evaluate_reports_a_perfect_and_a_worst_reranker(self, coll):
        q = [{"id": "q", "doc": "doc5.txt", "query": "harbor crane",
              "gold": ["harbor crane"]}]
        base = ev.evaluate(coll, q, k=1)
        assert base["hit@1"] == 0.0
        fn, _ = _by_text(coll, "doc5.txt")
        better = ev.evaluate(coll, q, k=1, rerank_fn=fn, candidates=5)
        assert better["hit@1"] == 1.0 and better["mrr@10"] == 1.0
        assert better["rerank_degraded"] == 0
        failing = ev.evaluate(coll, q, k=1, candidates=5,
                              rerank_fn=lambda a, b: [])
        assert failing["rerank_degraded"] == 1
        assert failing["hit@1"] == base["hit@1"]


class TestFixtureLabels:
    def test_every_query_has_a_few_relevant_chunks_in_the_corpus(self, tmp_path):
        queries = ev.load_queries()
        coll = ev.build_collection(tmp_path)
        counts = ev.relevant_chunk_counts(coll, queries)
        assert len(queries) >= 40
        assert len({q["id"] for q in queries}) == len(queries)
        bad = {i: n for i, n in counts.items() if not 1 <= n <= 3}
        assert bad == {}

    def test_gold_phrases_stay_inside_the_named_document(self):
        for q in ev.load_queries():
            assert (ev.CORPUS_DIR / q["doc"]).is_file(), q["id"]
