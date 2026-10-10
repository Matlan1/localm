# SPDX-License-Identifier: AGPL-3.0-or-later
"""Searching a collection: BM25, cosine similarity and the relevance floor."""

from __future__ import annotations

import math
import operator
import re
from pathlib import Path
from typing import Optional

from localm.debuglog import logger as _log
from localm.rag import store as _st

from ..bm25 import BM25, ENGLISH_STOP_WORDS
from .cache import _COLLECTION_CACHE
from .types import EmbedFn, RerankFn
from .vectors import _cosine, _maxnorm, _vectors_finite


#: Chunks handed to the reranker when ``Collection.query`` is given a
#: ``rerank_fn`` and no explicit ``rerank_candidates``.
DEFAULT_RERANK_CANDIDATES = 20

_WARNED_RERANK_DEGRADES: set = set()


#: Absolute relevance floors applied by ``Collection.query(relevant_only=True)``
#: when the collection's vectors were built with one of these embedding models:
#: ``(strong, weak, coverage)``. A hit is relevant when its raw query cosine
#: reaches *strong*, or reaches *weak* while the hit holds at least *coverage*
#: of the query's IDF-weighted terms (``BM25.coverage``).
RELEVANCE_FLOORS: dict[str, tuple[float, float, float]] = {
    "bge-small-en-v1.5": (0.67, 0.60, 0.65),
    "nomic-embed-text-v1.5": (0.72, 0.56, 0.55),
}


#: Phrasings that point back at the conversation rather than at a topic
#: ("why did that search fail", "your previous answer", "the second one").
#: Every alternative is a fixed sequence of words, so matching stays linear.
_CONVERSATION_REF_RE = re.compile(
    r"\b(?:that|this|those|these|it)\W*\Z"
    r"|\b(?:why|how|what|when|where)\s+(?:did|does|do|is|was|were|has|had)\s+"
    r"(?:that|this|those|these|it)\b"
    r"|\b(?:your|my|our)\s+(?:last|previous|earlier|first|second)\b"
    r"|\b(?:previous|last|earlier)\s+(?:answer|reply|message|response|question|search)\b"
    r"|\bwe\s+(?:just\s+)?(?:talked|discussed|said|did)\b"
    r"|\b(?:first|second|third|other|last)\s+one\b"
    r"|\b(?:say|explain|do|try)\s+(?:that|it)\s+again\b",
    re.IGNORECASE)


def refers_to_conversation(text: str) -> bool:
    """True when *text* is phrased as a reference back to the conversation
    (see ``_CONVERSATION_REF_RE``)."""
    return _CONVERSATION_REF_RE.search(text) is not None


#: Coverage a hit needs under ``relevant_only`` when there is no calibrated
#: cosine: lexical-only scoring, vectors from a model not in RELEVANCE_FLOORS,
#: or a chunk stored without a vector.
LEXICAL_RELEVANCE_COVERAGE = 0.6


class _CollectionSearch:
    """Mixin of ``Collection``: querying a collection."""

    name: str
    dir: Path
    _chunks: list[dict]

    def query(self, text: str, k: int = 4,
              embed_fn: Optional[EmbedFn] = None, *,
              relevant_only: bool = False,
              rerank_fn: Optional[RerankFn] = None,
              rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
              rerank_min_score: Optional[float] = None) -> list[dict]:
        """Top-*k* chunks for *text*: max-normalised BM25, blended 50/50 with
        max-normalised cosine similarity when vectors cover the corpus and the
        query can be embedded. ``score`` is that blend, relative to the best
        chunk for this query, so the top hit scores near 1.0 however weak it is.

        With *relevant_only*, each of the top-*k* hits is also checked against
        an absolute floor and dropped when it falls below it: the raw cosine
        floors in ``RELEVANCE_FLOORS`` for the collection's embedding model, or
        ``LEXICAL_RELEVANCE_COVERAGE`` of the query's words (``BM25.coverage``)
        for a chunk with no calibrated cosine. A query that ``refers_to_conversation``
        keeps only hits over the strong cosine floor, and none without one. A
        query unrelated to the collection then returns [].

        With *rerank_fn*, the best *rerank_candidates* chunks by the blend above
        (never fewer than *k*, after the *relevant_only* floor) are re-scored by
        ``rerank_fn(query, chunk_texts)``, which returns one finite float per
        text, and the top *k* by that score are returned, each carrying
        ``rerank_score`` next to the blended ``score``. A *rerank_fn* that raises
        or returns the wrong number of scores, or a non-finite one, leaves the
        blended order in place and records why in ``rerank_degrade_reason``
        (None after a query that reranked or was not asked to).

        With *relevant_only*, *rerank_fn* and *rerank_min_score* together, the
        reranker's score replaces the floor above as the relevance gate: the
        whole candidate pool is reranked and only hits scoring at least
        *rerank_min_score* are returned, so a paraphrased question the floor
        would drop is kept when the reranker ranks its answer highly. The scale
        of *rerank_min_score* is the reranker's own. A query that
        ``refers_to_conversation`` keeps the floor above, and so does a query
        whose reranking degraded."""
        self.rerank_degrade_reason = None
        if not text.strip() or not self._chunks:
            return []
        index = self._lexical_index()
        scores = index.scores(text)
        top = max(scores) if scores else 0.0
        if top > 0:
            scores = [s / top for s in scores]

        cosines = self._raw_cosines(text, embed_fn)
        if cosines is not None:
            vec_scores = _maxnorm(cosines)
            scores = [0.5 * lex + 0.5 * vec
                      for lex, vec in zip(scores, vec_scores, strict=False)]

        k = max(1, k)
        pool = k if rerank_fn is None else max(k, rerank_candidates)
        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)[:pool]
        order = [i for i in order if scores[i] > 0]
        score_gate = (relevant_only and rerank_fn is not None
                      and rerank_min_score is not None
                      and not refers_to_conversation(text))
        if relevant_only and not score_gate:
            order = self._relevant(text, order, index, cosines)
        reranked: dict[int, float] = {}
        if rerank_fn is not None and (len(order) > 1 or (score_gate and order)):
            reranked = self._rerank_scores(text, order, rerank_fn)
            if reranked:
                order = sorted(order, key=lambda i: reranked[i], reverse=True)
                if score_gate:
                    order = [i for i in order if reranked[i] >= rerank_min_score]
            elif score_gate:
                order = self._relevant(text, order, index, cosines)
        order = order[:k]
        return [
            {**self._chunks[i], "score": round(scores[i], 4),
             **({"rerank_score": round(reranked[i], 4)} if reranked else {})}
            for i in order
        ]

    def _rerank_scores(self, text: str, order: list[int],
                       rerank_fn: RerankFn) -> dict[int, float]:
        """``rerank_fn``'s score for each chunk index in *order*; {} when it
        raised or returned something unusable, with the reason recorded in
        ``rerank_degrade_reason`` and logged once per distinct reason."""
        try:
            raw = rerank_fn(text, [self._chunks[i]["text"] for i in order])
            values = [float(v) for v in raw]
        except Exception as e:
            return self._note_rerank_degrade(
                f"reranking failed ({type(e).__name__}); "
                f"using the unreranked order")
        if len(values) != len(order):
            return self._note_rerank_degrade(
                f"the reranker returned {len(values)} scores for "
                f"{len(order)} chunks; using the unreranked order")
        if not all(math.isfinite(v) for v in values):
            return self._note_rerank_degrade(
                "the reranker returned a non-finite score; using the "
                "unreranked order")
        return dict(zip(order, values, strict=True))

    def _note_rerank_degrade(self, reason: str) -> dict[int, float]:
        """Record *reason* as why this query was not reranked and return {}."""
        self.rerank_degrade_reason = reason
        key = (str(self.dir), reason)
        if key not in _WARNED_RERANK_DEGRADES:
            _WARNED_RERANK_DEGRADES.add(key)
            _log.warning("RAG collection %r: %s", self.name, reason)
        return {}

    def _relevant(self, text: str, order: list[int], index: BM25,
                  cosines: Optional[list[float]]) -> list[int]:
        """The members of *order* that clear the absolute relevance floor
        documented on ``query``, in the same order."""
        if not order:
            return []
        floors = None
        if cosines is not None and not self.embedding_model_mixed():
            floors = RELEVANCE_FLOORS.get(self.embedding_model() or "")
        vectors = self._vectors or []
        conversational = refers_to_conversation(text)
        coverage = index.coverage(text, order)

        def passes(i: int, cov: float) -> bool:
            if floors is not None and i < len(vectors) and vectors[i]:
                strong, weak, need = floors
                if conversational:
                    return cosines[i] >= strong
                return cosines[i] >= strong or (cosines[i] >= weak and cov >= need)
            return not conversational and cov >= LEXICAL_RELEVANCE_COVERAGE

        keep = [i for i, cov in zip(order, coverage, strict=True) if passes(i, cov)]
        if len(keep) < len(order):
            _log.debug("RAG collection %r: %d of %d hit(s) below the relevance "
                       "floor (%s%s)", self.name, len(order) - len(keep), len(order),
                       "cosine" if floors is not None else "keyword coverage",
                       ", conversation reference" if conversational else "")
        return keep

    def _lexical_index(self) -> BM25:
        """The BM25 index over this instance's current chunks, built once per
        chunks list. An instance whose chunks are exactly its cached snapshot's
        shares that snapshot's index."""
        chunks = self._chunks
        if (self._bm25 is not None and self._bm25_source is chunks
                and self._bm25_size == len(chunks)):
            return self._bm25
        snap = self._snapshot_ref() if self._snapshot_ref is not None else None
        if (snap is not None and len(chunks) == len(snap.chunks)
                and all(map(operator.is_, chunks, snap.chunks))):
            index = _COLLECTION_CACHE.lexical_index(snap)
        else:
            # Filter English stopwords from the lexical index, so a query and a
            # chunk that overlap ONLY on a stopword cannot win the BM25 half.
            index = BM25([c["text"] for c in chunks], stop_words=ENGLISH_STOP_WORDS)
        self._bm25, self._bm25_source, self._bm25_size = index, chunks, len(chunks)
        return index

    def _vector_scores(self, text: str,
                       embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        """``_raw_cosines`` divided by its maximum, so the best chunk scores
        1.0; None when vector scoring is unavailable."""
        cosines = self._raw_cosines(text, embed_fn)
        return None if cosines is None else _maxnorm(cosines)

    def _raw_cosines(self, text: str,
                     embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        """Cosine similarity of the embedded *text* to every chunk, in chunk
        order; None when vector scoring is unavailable (no embedder, no or
        partial vectors, mixed or mismatched dimensions, a failed or non-finite
        query embedding), with the reason recorded by _note_vector_degrade."""
        if embed_fn is None or self._vectors is None:
            return None
        present = [v for v in self._vectors if v]
        if not self._chunks or len(present) / len(self._chunks) < 0.8:
            # Partial coverage while a collection is still embedding: recorded
            # (visible in stats) but not warned about.
            self._note_vector_degrade(
                "vector coverage below 80% (index not fully embedded); "
                "using BM25 lexical retrieval only", warn=False)
            return None
        # Stored vectors must share one dimensionality. A legacy collection with
        # mixed-dim vectors is ambiguous, so vector scoring is skipped and the
        # answer is lexical.
        dims = {len(v) for v in present}
        if len(dims) > 1:
            self._note_vector_degrade(
                "stored vectors have mixed dimensionality (legacy index); "
                "using BM25 lexical retrieval only", warn=True)
            return None
        stored_dim = next(iter(dims))
        try:
            qvec = embed_fn([text])[0]
        except Exception as e:
            # The embedder raised: a real failure (backend down, model unloaded),
            # not an expected transient like partial coverage. The note is
            # idempotent, so this warns once per distinct error, not per query.
            self._note_vector_degrade(
                f"query embedding failed ({type(e).__name__}); "
                f"using BM25 lexical retrieval only", warn=True)
            return None
        # A query vector of a different dimensionality than the stored ones (a
        # switched embedding model) falls back to lexical-only.
        if len(qvec) != stored_dim:
            self._note_vector_degrade(
                f"embedding model changed (query dim {len(qvec)} != stored "
                f"{stored_dim}); using BM25 lexical retrieval only - rebuild the "
                f"collection to restore semantic search", warn=True)
            return None
        if not _vectors_finite([qvec]):
            # A non-finite query embedding would make every cosine nan and empty
            # the result set.
            self._note_vector_degrade(
                "query embedding has non-finite (NaN/inf) values; using BM25 "
                "lexical retrieval only", warn=True)
            return None
        # Vectors are usable: clear any stale query-time degrade note.
        self.vector_degrade_reason = None
        if _st._numpy is not None:
            try:
                np = _st._numpy
                if (self._norm_matrix is None or self._norm_source is not self._vectors
                        or len(self._norm_matrix) != len(self._vectors)):
                    mat = np.zeros((len(self._vectors), stored_dim), dtype="float32")
                    for idx, vec in enumerate(self._vectors):
                        if vec and len(vec) == stored_dim:
                            mat[idx] = vec
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    normalized = mat / norms
                    self._norm_matrix = np.where(np.isfinite(normalized), normalized, 0.0)
                    self._norm_source = self._vectors

                qv = np.asarray(qvec, dtype="float32")
                qnorm = float(np.linalg.norm(qv))
                if qnorm > 0:
                    qv = qv / qnorm
                    sims = np.dot(self._norm_matrix, qv)
                    sims = np.where(np.isfinite(sims), sims, 0.0)
                    return [float(s) for s in sims]
                else:
                    return [0.0] * len(self._vectors)
            except Exception:
                pass
        return [_cosine(qvec, v) if v else 0.0 for v in self._vectors]
