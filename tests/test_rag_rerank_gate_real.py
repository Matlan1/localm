# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reranker-score relevance gate with the real embedding model and the two
measured rerankers (the Q8_0 files named in ``CALIBRATED_MIN_SCORES``).

Needs a localm home holding bge-small-en-v1.5 and at least one calibrated
reranker; point ``LOCALM_TEST_REAL_HOME`` at it (every test otherwise runs in a
throwaway home), or the tests skip. The recorded form of the same comparison,
which runs everywhere, is ``tests/test_rag_rerank_replay.py``. Re-record it with
``python -m tests.rag_rerank_eval --embed --reranker A --reranker B --record PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests import rag_rerank_eval as ev

pytestmark = pytest.mark.integration

EMBEDDING_MODEL = "bge-small-en-v1.5"


def _real_home() -> Path:
    raw = os.environ.get("LOCALM_TEST_REAL_HOME")
    if not raw or not Path(raw).is_dir():
        pytest.skip("LOCALM_TEST_REAL_HOME names no localm home with the models")
    return Path(raw)


def _point_at(mp: pytest.MonkeyPatch, home: Path) -> None:
    from localm import config as cfg
    mp.setenv("LOCALM_HOME", str(home))
    mp.setattr(cfg, "HOME_DIR", home)
    mp.setattr(cfg, "MODELS_DIR", home / "models")
    mp.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    mp.setattr(cfg, "CONFIG_FILE", home / "config.json")


def _calibrated_rerankers() -> dict:
    from localm.config import load_registry
    from localm.rag.rerank import CALIBRATED_MIN_SCORES, make_rerank_fn
    out: dict = {}
    seen: set = set()
    for name, entry in load_registry().items():
        sha = str((entry or {}).get("sha256") or "").lower()
        if sha in CALIBRATED_MIN_SCORES and sha not in seen:
            seen.add(sha)
            out[name] = (make_rerank_fn(name)[1], sha)
    return out


def _embedder():
    from localm.config import load_config
    from localm.inference.embedder import embed_texts
    if str(load_config().get("embedding_model") or "") != EMBEDDING_MODEL:
        pytest.skip(f"the home's embedding model is not {EMBEDDING_MODEL}")
    if embed_texts(["probe"]) is None:
        pytest.skip("the embedding model does not load here")

    def embed_fn(texts):
        out = embed_texts(texts)
        if out is None:
            raise RuntimeError("the embedder is unavailable")
        return out

    return embed_fn


@pytest.fixture(scope="module")
def real_suite(tmp_path_factory):
    from localm.rag.rerank import CALIBRATED_MIN_SCORES
    home = _real_home()
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, home)
        rerankers = _calibrated_rerankers()
        if not rerankers:
            pytest.skip("no calibrated reranker is installed in that home")
        embed_fn = _embedder()
        coll = ev.build_collection(tmp_path_factory.mktemp("real"), embed_fn,
                                   EMBEDDING_MODEL)
        minimums = {name: CALIBRATED_MIN_SCORES[sha]
                    for name, (_, sha) in rerankers.items()}
        suite = ev.gate_suite(
            coll, ev.load_queries(), ev.load_offtopic(), ev.load_followups(),
            embed_fn=embed_fn,
            rerankers={name: (fn, minimums[name])
                       for name, (fn, _) in rerankers.items()})
    return suite, minimums


def test_the_score_gate_beats_floor_then_rerank_with_real_models(real_suite):
    suite, minimums = real_suite
    for name, minimum in minimums.items():
        gate = suite["gates"][f"{name}: score >= {minimum:g}"]
        before = suite["gates"][f"{name}: floor, then rerank"]
        assert gate["recall"] >= before["recall"] + 0.05, (name, gate, before)
        assert gate["precision"] >= before["precision"] - 0.02, (name, gate, before)
        for category in ("unrelated", "chitchat", "conversation"):
            assert gate["categories"][category] == 1.0, (name, category, gate)


def test_follow_ups_find_their_topic_with_the_previous_message_with_real_models(
        real_suite):
    suite, minimums = real_suite
    for name, minimum in minimums.items():
        without = suite["followups"][f"{name}: score >= {minimum:g}, no context"]
        with_context = suite["followups"][f"{name}: score >= {minimum:g}, context"]
        assert with_context["recall"] >= without["recall"] + 0.2, (name, with_context)
        assert with_context["precision"] >= 0.85, (name, with_context)


def test_simultaneous_reranked_queries_succeed_with_a_real_reranker(
        tmp_path_factory):
    from concurrent.futures import ThreadPoolExecutor
    from localm.rag import Collection
    from localm.rag.rerank import CALIBRATED_MIN_SCORES
    home = _real_home()
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, home)
        rerankers = _calibrated_rerankers()
        if not rerankers:
            pytest.skip("no calibrated reranker is installed in that home")
        embed_fn = _embedder()
        name, (fn, sha) = next(iter(rerankers.items()))
        coll = ev.build_collection(tmp_path_factory.mktemp("conc"), embed_fn,
                                   EMBEDDING_MODEL)
        queries = [q["query"] for q in ev.load_queries()[:12]]

        def ask(text):
            own = Collection(coll.name, base=coll.dir.parent)
            hits = own.query(text, k=4, embed_fn=embed_fn, relevant_only=True,
                             rerank_fn=fn,
                             rerank_min_score=CALIBRATED_MIN_SCORES[sha])
            return hits, own.rerank_degrade_reason

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(ask, queries * 3))
    assert len(results) == 36, name
    assert [reason for _, reason in results if reason] == [], name
    assert all(all("rerank_score" in h for h in hits) for hits, _ in results), name
