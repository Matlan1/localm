# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reranker-score relevance gate with the real embedding model and the two
measured rerankers (the Q8_0 files named in ``CALIBRATED_MIN_SCORES``).

Skipped unless the embedding model and a calibrated reranker are installed; the
recorded form of the same comparison, which runs everywhere, is
``tests/test_rag_rerank_replay.py``. Re-record it with
``python -m tests.rag_rerank_eval --embed --reranker A --reranker B --record PATH``.
"""

from __future__ import annotations

import pytest

from tests import rag_rerank_eval as ev

pytestmark = pytest.mark.integration


def _calibrated_rerankers() -> dict:
    from localm.config import load_registry
    from localm.rag.rerank import CALIBRATED_MIN_SCORES, make_rerank_fn
    out = {}
    for name, entry in load_registry().items():
        sha = str((entry or {}).get("sha256") or "").lower()
        if sha in CALIBRATED_MIN_SCORES and sha not in {s for _, s in out.values()}:
            out[name] = (make_rerank_fn(name)[1], sha)
    return out


@pytest.fixture(scope="module")
def real_suite(tmp_path_factory):
    from localm.config import load_config
    from localm.inference.embedder import embed_texts
    from localm.rag.rerank import CALIBRATED_MIN_SCORES
    model = str(load_config().get("embedding_model") or "")
    rerankers = _calibrated_rerankers()
    if model != "bge-small-en-v1.5" or not rerankers:
        pytest.skip("needs bge-small-en-v1.5 and a calibrated reranker installed")
    if embed_texts(["probe"]) is None:
        pytest.skip("the embedding model does not load here")

    def embed_fn(texts):
        out = embed_texts(texts)
        if out is None:
            raise RuntimeError("the embedder is unavailable")
        return out

    coll = ev.build_collection(tmp_path_factory.mktemp("real"), embed_fn, model)
    suite = ev.gate_suite(
        coll, ev.load_queries(), ev.load_offtopic(), ev.load_followups(),
        embed_fn=embed_fn,
        rerankers={name: (fn, CALIBRATED_MIN_SCORES[sha])
                   for name, (fn, sha) in rerankers.items()})
    return suite, {name: CALIBRATED_MIN_SCORES[sha]
                   for name, (_, sha) in rerankers.items()}


def test_the_score_gate_beats_floor_then_rerank_with_real_models(real_suite):
    suite, minimums = real_suite
    for name, minimum in minimums.items():
        gate = suite["gates"][f"{name}: score >= {minimum:g}"]
        before = suite["gates"][f"{name}: floor, then rerank"]
        assert gate["recall"] >= before["recall"] + 0.15, (name, gate, before)
        assert gate["precision"] >= before["precision"] - 0.02, (name, gate, before)
        for category in ("unrelated", "chitchat", "conversation"):
            assert gate["categories"][category] == 1.0, (name, category, gate)


def test_follow_ups_find_their_topic_with_the_previous_message_with_real_models(
        real_suite):
    suite, minimums = real_suite
    for name, minimum in minimums.items():
        without = suite["followups"][f"{name}: score >= {minimum:g}, no context"]
        with_context = suite["followups"][f"{name}: score >= {minimum:g}, context"]
        assert with_context["recall"] >= without["recall"] + 0.3, (name, with_context)
        assert with_context["precision"] >= 0.85, (name, with_context)


def test_simultaneous_reranked_queries_succeed_with_a_real_reranker(tmp_path_factory):
    from concurrent.futures import ThreadPoolExecutor
    from localm.config import load_config
    from localm.inference.embedder import embed_texts
    from localm.rag.rerank import CALIBRATED_MIN_SCORES
    if str(load_config().get("embedding_model") or "") != "bge-small-en-v1.5":
        pytest.skip("needs bge-small-en-v1.5")
    rerankers = _calibrated_rerankers()
    if not rerankers or embed_texts(["probe"]) is None:
        pytest.skip("needs a calibrated reranker and a loadable embedder")
    name, (fn, sha) = next(iter(rerankers.items()))

    def embed_fn(texts):
        return embed_texts(texts)

    coll = ev.build_collection(tmp_path_factory.mktemp("conc"), embed_fn,
                               "bge-small-en-v1.5")
    queries = [q["query"] for q in ev.load_queries()[:12]]

    from localm.rag import Collection

    def ask_own(text):
        own = Collection(coll.name, base=coll.dir.parent)
        hits = own.query(text, k=4, embed_fn=embed_fn, relevant_only=True,
                         rerank_fn=fn, rerank_min_score=CALIBRATED_MIN_SCORES[sha])
        return hits, own.rerank_degrade_reason

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(ask_own, queries * 3))
    assert len(results) == 36, name
    assert [reason for _, reason in results if reason] == [], name
    assert all(all("rerank_score" in h for h in hits) for hits, _ in results), name
