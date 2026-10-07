# SPDX-License-Identifier: AGPL-3.0-or-later
"""``Collection.query(relevant_only=True)``: hits below the absolute relevance
floor are dropped, so a question unrelated to the collection returns nothing,
while the default ranking still returns its top-k however weak they are.

Vectors are 5-dimensional: each chunk embeds to its own axis, and a query embeds
to ``[cA, cB, cC, 0, rest]`` so its raw cosine to every chunk is chosen exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from localm.rag.bm25 import BM25, stem
from localm.rag.store import (LEXICAL_RELEVANCE_COVERAGE, RELEVANCE_FLOORS,
                              Collection, refers_to_conversation)

GPU = "Install the ROCm runtime and set the GPU layers for the gfx1030 card."
WEB = "The web search reads the top result pages and quotes them as evidence."
SYNC = "Knowledge collections are rebuilt with resync when files change on disk."
EXTRA = "Release notes are published with every tagged version of the package."
AXES = {GPU: 0, WEB: 1, SYNC: 2, EXTRA: 3}

BGE = "bge-small-en-v1.5"
BGE_STRONG, BGE_WEAK, BGE_COVERAGE = RELEVANCE_FLOORS[BGE]


def _query_vec(cos_gpu=0.0, cos_web=0.0, cos_sync=0.0):
    rest = 1.0 - cos_gpu ** 2 - cos_web ** 2 - cos_sync ** 2
    assert rest >= 0
    return [cos_gpu, cos_web, cos_sync, 0.0, math.sqrt(rest)]


def _embedder(queries: dict):
    def embed(texts):
        out = []
        for t in texts:
            if t in AXES:
                v = [0.0] * 5
                v[AXES[t]] = 1.0
                out.append(v)
            else:
                out.append(queries[t])
        return out
    return embed


def _collection(tmp_path, model=BGE, queries=None):
    coll = Collection("kb", base=tmp_path / "rag").create()
    coll.add_uploads(
        [{"filename": f"{n}.md", "data": text.encode()}
         for n, text in (("gpu", GPU), ("web", WEB), ("sync", SYNC), ("extra", EXTRA))],
        embed_fn=_embedder(queries or {}), model_name=model)
    coll = Collection("kb", base=tmp_path / "rag")
    assert len(coll._chunks) == 4 and coll._vectors and all(coll._vectors)
    assert coll.embedding_model() == model
    return coll


def _texts(hits):
    return [h["text"] for h in hits]


class TestCosineFloor:
    def test_off_topic_question_returns_nothing(self, tmp_path):
        q = "tell me a joke about cats"
        embed = _embedder({q: _query_vec(cos_gpu=BGE_WEAK - 0.08)})
        coll = _collection(tmp_path)
        assert coll.query(q, k=4, embed_fn=embed, relevant_only=True) == []
        assert _texts(coll.query(q, k=4, embed_fn=embed)) == [GPU], \
            "the default ranking still returns the least-bad chunk"

    def test_question_about_the_conversation_is_dropped_despite_shared_words(
            self, tmp_path):
        q = "why did that search fail?"
        embed = _embedder({q: _query_vec(cos_web=BGE_WEAK - 0.03)})
        coll = _collection(tmp_path)
        assert _texts(coll.query(q, k=4, embed_fn=embed)) == [WEB]
        assert coll.query(q, k=4, embed_fn=embed, relevant_only=True) == []

    def test_strong_cosine_is_kept_without_any_shared_word(self, tmp_path):
        q = "how do I make my graphics hardware work"
        embed = _embedder({q: _query_vec(cos_gpu=BGE_STRONG + 0.01)})
        coll = _collection(tmp_path)
        assert BM25([GPU]).scores(q) == [0.0]
        assert _texts(coll.query(q, k=4, embed_fn=embed, relevant_only=True)) == [GPU]

    def test_weak_cosine_needs_the_query_words(self, tmp_path):
        covered = "rocm runtime gfx1030"
        uncovered = "rocm sourdough bread recipes dinner"
        embed = _embedder({covered: _query_vec(cos_gpu=BGE_WEAK + 0.02),
                           uncovered: _query_vec(cos_gpu=BGE_WEAK + 0.02)})
        coll = _collection(tmp_path)
        gpu_i = [c["text"] for c in coll._chunks].index(GPU)
        cov_covered = coll._lexical_index().coverage(covered, [0, 1, 2, 3])
        cov_uncovered = coll._lexical_index().coverage(uncovered, [0, 1, 2, 3])
        assert cov_covered[gpu_i] >= BGE_COVERAGE > cov_uncovered[gpu_i] > 0
        assert _texts(coll.query(covered, k=4, embed_fn=embed,
                                 relevant_only=True)) == [GPU]
        assert _texts(coll.query(uncovered, k=4, embed_fn=embed)) == [GPU]
        assert coll.query(uncovered, k=4, embed_fn=embed, relevant_only=True) == []

    def test_only_hits_over_the_floor_survive_and_keep_their_order(self, tmp_path):
        q = "what does resync do with the gpu"
        embed = _embedder({q: _query_vec(cos_gpu=BGE_WEAK - 0.1,
                                          cos_sync=BGE_STRONG + 0.05)})
        coll = _collection(tmp_path)
        everything = _texts(coll.query(q, k=4, embed_fn=embed))
        assert set(everything) == {GPU, SYNC}
        assert _texts(coll.query(q, k=4, embed_fn=embed, relevant_only=True)) == [SYNC]


class TestKeywordCoverageBasis:
    def test_lexical_only_query_uses_keyword_coverage(self, tmp_path):
        coll = _collection(tmp_path)
        q = "why did that search fail?"
        assert _texts(coll.query(q, k=4)) == [WEB]
        assert coll.query(q, k=4, relevant_only=True) == []
        assert _texts(coll.query("resync files on disk", k=4,
                                 relevant_only=True)) == [SYNC]

    def test_uncalibrated_model_ignores_cosine(self, tmp_path):
        paraphrase = "how do I make my graphics hardware work"
        keywords = "rocm runtime gfx1030"
        embed = _embedder({paraphrase: _query_vec(cos_gpu=0.99),
                           keywords: _query_vec()})
        coll = _collection(tmp_path, model="my-own-embedder", queries={})
        assert "my-own-embedder" not in RELEVANCE_FLOORS
        assert _texts(coll.query(paraphrase, k=4, embed_fn=embed)) == [GPU]
        assert coll.query(paraphrase, k=4, embed_fn=embed, relevant_only=True) == []
        assert _texts(coll.query(keywords, k=4, embed_fn=embed,
                                 relevant_only=True)) == [GPU]

    def test_mixed_embedding_models_ignore_cosine(self, tmp_path):
        paraphrase = "how do I make my graphics hardware work"
        embed = _embedder({paraphrase: _query_vec(cos_gpu=0.99)})
        coll = _collection(tmp_path)
        assert _texts(coll.query(paraphrase, k=4, embed_fn=embed,
                                 relevant_only=True)) == [GPU]
        coll.add_uploads([{"filename": "more.md", "data": b"unrelated cooking notes"}],
                         embed_fn=lambda texts: [[0.0, 0.0, 0.0, 0.0, 1.0] for _ in texts],
                         model_name="nomic-embed-text-v1.5")
        coll = Collection("kb", base=tmp_path / "rag")
        assert coll.embedding_model_mixed()
        assert coll.query(paraphrase, k=4, embed_fn=embed, relevant_only=True) == []


class TestSentenceQuestionsAndVectorlessChunks:
    def test_sentence_style_question_is_kept_on_keyword_basis(self, tmp_path):
        coll = _collection(tmp_path)
        q = "explain how knowledge collections are resynced when files change"
        assert _texts(coll.query(q, k=4, relevant_only=True)) == [SYNC]
        q = "tell me about how the ROCm runtime is installed"
        assert _texts(coll.query(q, k=4, relevant_only=True)) == [GPU]

    def test_chunk_without_a_vector_is_judged_on_keywords(self, tmp_path):
        q = "zeppelin airship hydrogen envelope"
        embed = _embedder({q: _query_vec()})
        coll = _collection(tmp_path)
        coll.add_uploads([{"filename": "zeppelin.md",
                           "data": b"The zeppelin airship kept hydrogen in its envelope."}],
                         model_name=BGE)
        coll = Collection("kb", base=tmp_path / "rag")
        assert sum(1 for v in coll._vectors if v) == 4 and len(coll._chunks) == 5
        assert not coll.embedding_model_mixed()
        assert coll._raw_cosines(q, embed) is not None, "cosines are in use"
        hits = coll.query(q, k=4, embed_fn=embed, relevant_only=True)
        assert [h["source"] for h in hits] == ["upload:zeppelin.md"]


class TestConversationReference:
    @pytest.mark.parametrize("text", [
        "why did that search fail?", "what did you mean by that?",
        "what does that resync do?", "can you rewrite your previous answer",
        "ok and what about the second one?", "summarize what we just talked about",
        "why did my last message not get through", "can you say that again",
        "what time is it",
    ])
    def test_detected(self, text):
        assert refers_to_conversation(text)

    @pytest.mark.parametrize("text", [
        "how does the BM25 index handle stopwords",
        "what happens to a document that was deleted from disk",
        "can the model browse the web by itself",
        "how do I add a folder to a knowledge collection",
        "explain how knowledge collections are resynced when files change",
        "",
    ])
    def test_not_detected(self, text):
        assert not refers_to_conversation(text)

    def test_long_adversarial_input_is_handled(self):
        assert not refers_to_conversation("it " * 50_000 + "x")
        assert not refers_to_conversation("why did " * 50_000 + "x")
        assert refers_to_conversation("x" + " " * 100_000 + "it")

    def test_keyword_basis_injects_nothing_for_a_conversation_reference(self, tmp_path):
        coll = _collection(tmp_path)
        q = "what does that resync do?"
        sync_i = [c["text"] for c in coll._chunks].index(SYNC)
        assert coll._lexical_index().coverage(q, [sync_i]) == [1.0]
        assert _texts(coll.query(q, k=4)) == [SYNC]
        assert coll.query(q, k=4, relevant_only=True) == []

    def test_cosine_basis_needs_the_strong_floor(self, tmp_path):
        q = "what does that resync do?"
        mid = _embedder({q: _query_vec(cos_sync=(BGE_WEAK + BGE_STRONG) / 2)})
        strong = _embedder({q: _query_vec(cos_sync=BGE_STRONG + 0.01)})
        coll = _collection(tmp_path)
        assert coll.query(q, k=4, embed_fn=mid, relevant_only=True) == []
        assert _texts(coll.query(q, k=4, embed_fn=strong, relevant_only=True)) == [SYNC]


class TestFloorsAndCoverage:
    def test_stem_joins_plural_and_tense_forms(self):
        assert {stem(w) for w in ("change", "changes", "changed", "changing")} == {"chang"}
        assert stem("resynced") == stem("resync")
        assert stem("files") == stem("file")
        assert stem("is") == "is" and stem("use") == "use"

    def test_coverage_does_not_count_filler_words(self):
        idx = BM25(["resync rebuilds collections", "unrelated words here"])
        assert idx.coverage("please explain resync", [0]) == [1.0]
        assert 0 < idx.coverage("please rebuild zeppelins", [0])[0] < 0.5
        assert idx.coverage("please explain", [0]) == [0.0]

    def test_coverage_matches_suffix_variants(self):
        idx = BM25(["the collection was resynced yesterday", "other text entirely"])
        assert idx.coverage("resync collections", [0]) == [1.0]

    def test_floor_values_are_ordered(self):
        for model, (strong, weak, coverage) in RELEVANCE_FLOORS.items():
            assert 0 < weak < strong < 1, model
            assert 0 < coverage <= 1, model
        assert 0 < LEXICAL_RELEVANCE_COVERAGE <= 1

    def test_coverage_is_an_idf_weighted_fraction(self):
        idx = BM25(["alpha beta", "alpha gamma", "alpha delta"])
        assert idx.coverage("", [0, 1]) == [0.0, 0.0]
        assert idx.coverage("alpha beta", [0]) == [1.0]
        rare, common = idx.coverage("beta alpha", [0, 2])
        assert rare == 1.0
        assert 0 < common < 0.5, "alpha is in every text, so it carries little weight"
        unseen, = idx.coverage("beta zeta", [0])
        assert 0 < unseen < 0.5, "a term absent from the index weighs as the rarest"

    def test_coverage_on_an_empty_index(self):
        assert BM25([]).coverage("anything", []) == []


@pytest.fixture
def rag_app(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from localm.plugins.engine import PluginManager
    home = tmp_path / "userhome"
    home.mkdir()
    localm_home = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm_home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm_home)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm_home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm_home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm_home / "registry.json")
    from localm.plugins.gui.jobs import JobManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    app.state.jobs = JobManager()
    return app


class TestQueryRoute:
    def test_relevant_only_drops_an_unrelated_question(self, rag_app):
        from fastapi.testclient import TestClient
        with TestClient(rag_app) as c:
            assert c.post("/api/rag/collections", json={"name": "kb"}).status_code == 200
            Collection("kb").add_uploads(
                [{"filename": f"{n}.md", "data": t.encode()}
                 for n, t in (("gpu", GPU), ("web", WEB), ("sync", SYNC))])
            q = {"query": "why did that search fail?", "k": 4}
            plain = c.post("/api/rag/collections/kb/query", json=q)
            gated = c.post("/api/rag/collections/kb/query",
                           json={**q, "relevant_only": True})
            on_topic = c.post("/api/rag/collections/kb/query",
                              json={"query": "resync files on disk", "k": 4,
                                    "relevant_only": True})
        assert [h["text"] for h in plain.json()["hits"]] == [WEB]
        assert plain.json()["relevant_only"] is False
        assert gated.json()["hits"] == []
        assert gated.json()["relevant_only"] is True
        assert [h["text"] for h in on_topic.json()["hits"]] == [SYNC]
        assert (plain.status_code, gated.status_code, on_topic.status_code) == (200, 200, 200)


class TestCliRelevantOnly:
    def test_flag_gates_and_default_does_not(self, cli_runner):
        from localm.cli import main
        coll = Collection("kb").create()
        coll.add_uploads([{"filename": f"{n}.md", "data": t.encode()}
                          for n, t in (("gpu", GPU), ("web", WEB), ("sync", SYNC))])
        plain = cli_runner.invoke(main, ["rag", "query", "kb", "why did that search fail?"])
        gated = cli_runner.invoke(main, ["rag", "query", "kb", "why did that search fail?",
                                         "--relevant-only"])
        assert "web.md" in plain.output
        assert "(no relevant matches)" in gated.output
        assert "web.md" not in gated.output
        assert (plain.exit_code, gated.exit_code) == (0, 0), plain.output + gated.output
