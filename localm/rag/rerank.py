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
    (None when reranking is off or simply not installed)."""
    fn: Optional[RerankFn]
    candidates: int
    model: Optional[str]
    note: Optional[str]


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
    if not model and not reranker.registered_rerankers():
        return RerankPlan(None, candidates, None, None)
    try:
        name, fn = make_rerank_fn(model)
    except reranker.RerankerModelError as e:
        return RerankPlan(None, candidates, None, str(e))
    return RerankPlan(fn, candidates, name, None)
