# SPDX-License-Identifier: AGPL-3.0-or-later
"""``Collection.query`` with the reranker-score gate: a hit that holds every
counted word of the question and reaches the embedding model's weak cosine
floor is kept even when the reranker scores it below the minimum, so a short
question with no context ("How do I connect a phone?") still finds the
document about it. Anything less keeps the reranker's verdict.

Vectors are 5-dimensional as in ``test_rag_relevance_floor``: each chunk embeds
to its own axis and a query to ``[cA, cB, cC, 0, rest]``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from localm.rag.store import RELEVANCE_FLOORS, Collection

GPU = "Install the ROCm runtime and set the GPU layers for the gfx1030 card."
WEB = "The web search reads the top result pages and quotes them as evidence."
SYNC = "Knowledge collections are rebuilt with resync when files change on disk."
EXTRA = "Release notes are published with every tagged version of the package."
AXES = {GPU: 0, WEB: 1, SYNC: 2, EXTRA: 3}

BGE = "bge-small-en-v1.5"
BGE_STRONG, BGE_WEAK, _ = RELEVANCE_FLOORS[BGE]

ASK = "How do I set the GPU layers?"
MISSING_WORD = "How do I set the GPU fans?"
FILLER_ONLY = "How do I?"
CONVERSATION = "How do I set the GPU layers you just mentioned?"


def _vec(cos_gpu=0.0, cos_web=0.0):
    rest = 1.0 - cos_gpu ** 2 - cos_web ** 2
    return [cos_gpu, cos_web, 0.0, 0.0, math.sqrt(rest)]


def _embedder(queries: dict):
    def embed(texts):
        out = []
        for t in texts:
            if t in AXES:
                v = [0.0] * 5
                v[AXES[t]] = 1.0
                out.append(v)
            elif t in queries:
                out.append(queries[t])
            else:
                raise RuntimeError("embedder down")
        return out
    return embed


def _collection(tmp_path: Path, model: str = BGE) -> Collection:
    coll = Collection("kb", base=tmp_path / "rag").create()
    coll.add_uploads(
        [{"filename": f"{n}.md", "data": text.encode()}
         for n, text in (("gpu", GPU), ("web", WEB), ("sync", SYNC), ("extra", EXTRA))],
        embed_fn=_embedder({}), model_name=model)
    coll = Collection("kb", base=tmp_path / "rag")
    assert len(coll._chunks) == 4 and coll._vectors and all(coll._vectors)
    return coll


def _name(source: str) -> str:
    return Path(source).stem.rsplit(":", 1)[-1]


def _names(hits):
    return [_name(h["source"]) for h in hits]


def _scores(coll, **by_name):
    by_text = {c["text"]: _name(c["source"]) for c in coll._chunks}
    calls = []

    def fn(query, texts):
        calls.append(list(texts))
        return [by_name.get(by_text[t], 0.01) for t in texts]

    return fn, calls


def _gated(coll, text, queries, fn):
    return coll.query(text, k=4, embed_fn=_embedder(queries), relevant_only=True,
                      rerank_fn=fn, rerank_candidates=10, rerank_min_score=0.5)


@pytest.fixture
def coll(tmp_path):
    return _collection(tmp_path)


class TestEveryWordAndTheCosineFloor:
    def test_a_hit_with_every_word_and_the_weak_cosine_is_kept_below_the_minimum(
            self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        hits = _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK + 0.01)}, fn)
        assert _names(hits) == ["gpu"]
        assert hits[0]["rerank_score"] == 0.06

    def test_the_weak_cosine_floor_is_inclusive(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        hits = _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK)}, fn)
        assert _names(hits) == ["gpu"]

    def test_one_missing_word_leaves_the_reranker_in_charge(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, MISSING_WORD,
                      {MISSING_WORD: _vec(cos_gpu=BGE_STRONG + 0.2)}, fn) == []

    def test_a_cosine_below_the_weak_floor_leaves_the_reranker_in_charge(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK - 0.01)}, fn) == []

    def test_hits_over_the_minimum_score_rank_first(self, coll):
        fn, _ = _scores(coll, gpu=0.06, web=0.9)
        hits = _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK + 0.01,
                                            cos_web=0.3)}, fn)
        assert _names(hits) == ["web", "gpu"]

    def test_the_reranker_still_scores_every_candidate_once(self, coll):
        fn, calls = _scores(coll, gpu=0.06)
        _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK + 0.01)}, fn)
        assert len(calls) == 1


class TestWhereTheRuleDoesNotApply:
    def test_an_unmeasured_embedding_model_has_no_cosine_floor_to_agree_with(
            self, tmp_path):
        coll = _collection(tmp_path, model="some-other-embedder")
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, ASK, {ASK: _vec(cos_gpu=0.99)}, fn) == []

    def test_a_failed_query_embedding_leaves_the_reranker_in_charge(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, ASK, {}, fn) == []
        assert coll.vector_degrade_reason

    def test_a_question_of_only_question_words_is_never_kept_on_agreement(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, FILLER_ONLY,
                      {FILLER_ONLY: _vec(cos_gpu=0.99)}, fn) == []

    def test_a_reference_to_the_conversation_keeps_the_strong_floor(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, CONVERSATION,
                      {CONVERSATION: _vec(cos_gpu=BGE_WEAK + 0.01)}, fn) == []

    def test_without_the_minimum_score_nothing_changes(self, coll):
        fn, _ = _scores(coll, gpu=0.06)
        floored = coll.query(ASK, k=4, embed_fn=_embedder(
                                 {ASK: _vec(cos_gpu=BGE_WEAK + 0.01)}),
                             relevant_only=True, rerank_fn=fn, rerank_candidates=10)
        assert _names(floored) == ["gpu"]

    def test_a_mixed_embedding_collection_has_no_cosine_floor(self, coll):
        coll._meta["embedding_model_mixed"] = True
        fn, _ = _scores(coll, gpu=0.06)
        assert _gated(coll, ASK, {ASK: _vec(cos_gpu=BGE_WEAK + 0.01)}, fn) == []
