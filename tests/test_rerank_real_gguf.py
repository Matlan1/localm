# SPDX-License-Identifier: AGPL-3.0-or-later
"""A real reranker GGUF through the real isolated worker.

Set LOCALM_TEST_RERANK_MODEL to a reranker GGUF (for example
bge-reranker-v2-m3-Q8_0.gguf, jina-reranker-v1-tiny-en.Q8_0.gguf or
Qwen3-Reranker-0.6B-Q8_0.gguf) to run it; it is skipped otherwise and is never
downloaded by the suite. The worker runs on the CPU.
"""

import math
import os

import pytest

from localm.inference import embedder as emb
from localm.inference import reranker as rr
from localm.model_manager.gguf import gguf_reranker_signal

MODEL = os.environ.get("LOCALM_TEST_RERANK_MODEL")

QUERY = "what is the capital of France?"
DOCS = [
    "Paris is the capital and largest city of France.",
    "Bananas are rich in potassium and grow in tropical climates.",
    "Berlin is the capital of Germany and its largest city.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
]


@pytest.fixture
def worker(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(var, "-1")
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
    monkeypatch.setattr(emb, "_choose_embedder_gpu_layers",
                        lambda path, cfg: (0, "the test runs on the CPU"))
    rr.reset_reranker()
    yield
    rr.reset_reranker()


pytestmark = [
    pytest.mark.integration,
    pytest.mark.real_gguf,
    pytest.mark.skipif(not MODEL, reason="set LOCALM_TEST_RERANK_MODEL to a reranker GGUF"),
]


def order(rows):
    return [r["index"] for r in rows]


def test_the_file_is_recognised_as_a_reranker():
    assert gguf_reranker_signal(MODEL) is True


def test_the_relevant_document_ranks_first_and_the_unrelated_one_last(worker):
    outcome = rr.rerank(MODEL, QUERY, DOCS)
    assert len(outcome.scored) == len(DOCS)
    for item in outcome.scored:
        assert item["scores"] and all(math.isfinite(s) for s in item["scores"])
        assert item["tokens"] > 0 and item["truncated"] is False
    ranked = order(rr.rank_results(outcome.scored, labels=outcome.labels))
    assert ranked[0] in (0, 3)
    assert ranked[-1] == 1


def test_the_ranking_does_not_depend_on_which_documents_share_a_batch(worker):
    together = order(rr.rank_results(rr.rerank(MODEL, QUERY, DOCS).scored))
    alone = [rr.rerank(MODEL, QUERY, [d]).scored[0] for d in DOCS]
    assert order(rr.rank_results(alone)) == together


def test_a_document_longer_than_the_window_is_cut_and_still_scored(worker):
    long_doc = "Paris is the capital of France. " * 4000
    (item,) = rr.rerank(MODEL, QUERY, [long_doc]).scored
    assert item["truncated"] is True and math.isfinite(item["scores"][0])


def test_the_same_pair_scores_the_same_twice(worker):
    first = rr.rerank(MODEL, QUERY, DOCS[:2]).scored
    second = rr.rerank(MODEL, QUERY, DOCS[:2]).scored
    for a, b in zip(first, second, strict=True):
        assert a["scores"] == pytest.approx(b["scores"], abs=1e-3)
