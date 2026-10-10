# SPDX-License-Identifier: AGPL-3.0-or-later
"""``Collection.query(rerank_fn=...)`` and the rerank evaluation harness."""

import math
from pathlib import Path

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


class TestScoreGate:
    @pytest.fixture
    def kb(self, tmp_path):
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "full.txt").write_text("alpha beta gamma delta " + _NOISE,
                                       encoding="utf-8")
        (docs / "partial.txt").write_text("alpha " + _NOISE, encoding="utf-8")
        c = Collection("gate", base=tmp_path / "rag").create()
        c.add_paths([str(docs)])
        return c

    QUERY = "alpha beta gamma delta"

    def _scores(self, coll, **by_source):
        by_text = {c["text"]: Path(c["source"]).name for c in coll._chunks}
        calls = []

        def fn(query, texts):
            calls.append(list(texts))
            return [by_source.get(by_text[t], 0.0) for t in texts]

        return fn, calls

    def test_a_hit_the_floor_drops_is_kept_when_the_reranker_scores_it_high(self, kb):
        fn, _ = self._scores(kb, **{"partial.txt": 0.9, "full.txt": 0.1})
        floored = kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                           rerank_candidates=5)
        assert _names(floored) == ["full.txt"]
        gated = kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                         rerank_candidates=5, rerank_min_score=0.5)
        assert _names(gated) == ["partial.txt"]
        assert gated[0]["rerank_score"] == 0.9

    def test_a_hit_the_floor_keeps_is_dropped_below_the_minimum_score(self, kb):
        fn, _ = self._scores(kb, **{"full.txt": 0.2, "partial.txt": 0.1})
        assert _names(kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                               rerank_candidates=5)) == ["full.txt"]
        assert kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                        rerank_candidates=5, rerank_min_score=0.5) == []

    def test_the_whole_pool_reaches_the_reranker_not_just_the_floor_survivors(self, kb):
        fn, calls = self._scores(kb, **{"partial.txt": 0.9})
        kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                 rerank_candidates=5, rerank_min_score=0.5)
        assert len(calls) == 1 and len(calls[0]) == 2

    def test_a_single_candidate_is_still_scored(self, kb):
        fn, calls = self._scores(kb, **{"partial.txt": 0.9})
        hits = kb.query("alpha", k=1, relevant_only=True, rerank_fn=fn,
                        rerank_candidates=1, rerank_min_score=0.5)
        assert len(calls) == 1 and len(calls[0]) == 1
        assert len(hits) == 1

    def test_without_relevant_only_the_minimum_score_is_not_applied(self, kb):
        fn, _ = self._scores(kb, **{"full.txt": 0.2, "partial.txt": 0.1})
        hits = kb.query(self.QUERY, k=2, rerank_fn=fn, rerank_candidates=5,
                        rerank_min_score=0.5)
        assert len(hits) == 2

    def test_without_a_minimum_score_the_floor_stays_the_gate(self, kb):
        fn, _ = self._scores(kb, **{"partial.txt": 0.9})
        hits = kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                        rerank_candidates=5, rerank_min_score=None)
        assert _names(hits) == ["full.txt"]

    def test_a_reference_to_the_conversation_never_passes_on_a_rerank_score(self, kb):
        fn, _ = self._scores(kb, **{"full.txt": 0.99, "partial.txt": 0.99})
        hits = kb.query("alpha beta gamma delta, why did that fail", k=2,
                        relevant_only=True, rerank_fn=fn, rerank_candidates=5,
                        rerank_min_score=0.5)
        assert hits == []

    def test_a_failing_reranker_falls_back_to_the_floor(self, kb):
        def boom(q, t):
            raise RuntimeError("down")
        hits = kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=boom,
                        rerank_candidates=5, rerank_min_score=0.5)
        assert _names(hits) == ["full.txt"]
        assert all("rerank_score" not in h for h in hits)
        assert "reranking failed (RuntimeError)" in kb.rerank_degrade_reason

    def test_the_minimum_score_is_inclusive(self, kb):
        fn, _ = self._scores(kb, **{"partial.txt": 0.5, "full.txt": 0.4999})
        hits = kb.query(self.QUERY, k=2, relevant_only=True, rerank_fn=fn,
                        rerank_candidates=5, rerank_min_score=0.5)
        assert _names(hits) == ["partial.txt"]


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
        assert better["hit@1"] == 1.0 and better["mrr@1"] == 1.0
        assert better["rerank_degraded"] == 0
        failing = ev.evaluate(coll, q, k=1, candidates=5,
                              rerank_fn=lambda a, b: [])
        assert failing["rerank_degraded"] == 1
        assert failing["hit@1"] == base["hit@1"]


class TestGateHarness:
    def test_evaluate_gate_counts_recall_and_off_topic_precision(self, coll):
        on = [{"id": "q", "doc": "doc5.txt", "query": "harbor crane",
               "gold": ["harbor crane"]}]
        off = [{"id": "o1", "category": "near", "query": "harbor crane"},
               {"id": "o2", "category": "unrelated", "query": "zebra xylophone"}]
        fn, _ = _by_text(coll, "doc5.txt")
        floor = ev.evaluate_gate(coll, on, off, k=1)
        assert floor["recall"] == 0.0 and floor["answered"] == 1.0
        assert floor["precision"] == 0.5 and floor["leaks"] == ["o1"]
        assert floor["categories"] == {"near": 0.0, "unrelated": 1.0}
        gated = ev.evaluate_gate(coll, on, off, k=1, rerank_fn=fn,
                                 min_score=0.5, candidates=5)
        assert gated["recall"] == 1.0 and gated["precision"] == 0.5
        strict = ev.evaluate_gate(coll, on, off, k=1, rerank_fn=fn,
                                  min_score=2.0, candidates=5)
        assert strict["recall"] == 0.0 and strict["precision"] == 1.0
        assert strict["leaks"] == []

    def test_the_gate_table_has_one_row_per_gate_and_a_column_per_category(self):
        res = {"categories": {"near": 0.5, "unrelated": 1.0}, "recall": 0.25,
               "answered": 0.5, "precision": 0.75, "leaks": []}
        table = ev.format_gate_table({"floor only": res, "other": res})
        lines = table.splitlines()
        assert len(lines) == 3
        assert "near" in lines[0] and "unrelated" in lines[0]
        assert lines[1].startswith("floor only") and "0.250" in lines[1]


class TestOfftopicLabels:
    def test_the_off_topic_questions_are_labelled_and_unique(self):
        queries = ev.load_offtopic()
        assert len(queries) >= 40
        assert len({q["id"] for q in queries}) == len(queries)
        assert {q["category"] for q in queries} == {
            "unrelated", "near", "conversation", "chitchat"}
        assert all(q["query"].strip() for q in queries)

    def test_no_off_topic_question_is_also_an_on_topic_one(self):
        on = {q["query"].lower() for q in ev.load_queries()}
        assert not on & {q["query"].lower() for q in ev.load_offtopic()}


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
