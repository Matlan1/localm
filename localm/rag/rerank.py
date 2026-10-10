# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reranker behind ``Collection.query(rerank_fn=...)``: scores retrieved
chunks against the query with a registered reranker GGUF, as configured by
``rag_rerank``, ``rag_rerank_model`` and ``rag_rerank_candidates``."""

from __future__ import annotations

from typing import NamedTuple, Optional

from ._store.search import DEFAULT_RERANK_CANDIDATES
from ._store.types import RerankFn

MIN_CANDIDATES = 5
MAX_CANDIDATES = 100


class RerankPlan(NamedTuple):
    """What a Knowledge query does about reranking.

    *fn* is the rerank function, or None when the query is not reranked;
    *candidates* is how many hits the reranker sees; *model* is the reranker's
    registered name when *fn* is set; *note* says why a wanted rerank cannot run
    (None when reranking is off or simply not installed); *min_score* is the
    reranker score a hit needs to count as relevant under
    ``relevant_only``, or None when the reranker is not calibrated and the
    relevance floor stays the gate."""
    fn: Optional[RerankFn]
    candidates: int
    model: Optional[str]
    note: Optional[str]
    min_score: Optional[float] = None


#: Minimum relevant score by the sha256 of the measured reranker file:
#: bge-reranker-v2-m3-Q8_0.gguf from gpustack/bge-reranker-v2-m3-GGUF (an
#: unbounded logit) and qwen3-reranker-0.6b-q8_0.gguf from
#: ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF (the probability of "yes"). Scores
#: are only comparable within one model.
CALIBRATED_MIN_SCORES: dict[str, float] = {
    "a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3": -1.5,
    "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48": 0.5,
}


def calibrated_min_score(name: str) -> Optional[float]:
    """The minimum relevant score for the registered reranker *name*, or None
    when its registered file is not one of the measured files in
    ``CALIBRATED_MIN_SCORES`` (matched by sha256, whatever it is named)."""
    from localm.config import load_registry
    entry = load_registry().get(name)
    if not isinstance(entry, dict):
        return None
    return CALIBRATED_MIN_SCORES.get(str(entry.get("sha256") or "").lower())


def make_rerank_fn(model: Optional[str] = None) -> tuple[str, RerankFn]:
    """``(registered name, rerank function)`` for the reranker *model* (the only
    registered reranker when *model* is empty).

    Raises ``localm.inference.reranker.RerankerModelError`` when no usable
    reranker is registered. The returned function scores a query against each
    text (higher is more relevant) and raises when the model cannot load or
    score, which ``Collection.query`` reports as a degrade."""
    from localm.inference import reranker
    name, path = reranker.resolve_reranker(model)

    def rerank_fn(query: str, texts: list[str]) -> list[float]:
        outcome = reranker.rerank(path, query, texts)
        return [float(item["scores"][0]) for item in outcome.scored]

    return name, rerank_fn


def configured_candidates(cfg: dict) -> int:
    """``rag_rerank_candidates`` from *cfg*, clamped to the supported range."""
    try:
        n = int(cfg.get("rag_rerank_candidates", DEFAULT_RERANK_CANDIDATES))
    except (TypeError, ValueError):
        n = DEFAULT_RERANK_CANDIDATES
    return max(MIN_CANDIDATES, min(MAX_CANDIDATES, n))


def rerank_plan(cfg: Optional[dict] = None, *,
                enabled: Optional[bool] = None) -> RerankPlan:
    """The reranking a Knowledge query should do under *cfg* (default: the live
    config). *enabled* overrides ``rag_rerank`` for one query.

    Never raises: a configured reranker that cannot be resolved yields a plan
    with no function and the reason in ``note``; no reranker installed yields
    a plan with no function and no note."""
    if cfg is None:
        from localm.config import load_config
        cfg = load_config()
    candidates = configured_candidates(cfg)
    wanted = bool(cfg.get("rag_rerank", True)) if enabled is None else enabled
    if not wanted:
        return RerankPlan(None, candidates, None, None)
    from localm.inference import reranker
    model = str(cfg.get("rag_rerank_model") or "").strip()
    if not model:
        installed = reranker.registered_rerankers()
        if not installed:
            return RerankPlan(None, candidates, None, None)
        if len(installed) > 1:
            return RerankPlan(None, candidates, None,
                              f"several rerankers are installed ({', '.join(installed)}); "
                              "choose one in Settings > Knowledge > Reranker model")
    try:
        name, fn = make_rerank_fn(model)
    except reranker.RerankerModelError as e:
        return RerankPlan(None, candidates, None, str(e))
    return RerankPlan(fn, candidates, name, None, calibrated_min_score(name))
