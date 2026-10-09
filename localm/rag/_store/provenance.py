# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which collections an embedding-model switch would affect."""

from __future__ import annotations

from typing import Optional

from localm.debuglog import logger as _log
from localm.rag import store as _st


def _provenance_excludable(built_with: Optional[str], mixed: bool,
                            candidate_model: Optional[str]) -> bool:
    """True when a collection's provenance is a single known model equal to
    *candidate_model*, so switching to it will not invalidate this
    collection: false for a mixed-provenance or unlabelled collection, which
    always stays in the report."""
    return bool(candidate_model) and not mixed and built_with == candidate_model


def collection_provenance_report(candidate_model: Optional[str] = None) -> list:
    """Every collection that currently has vectors, with its recorded 'built
    with' model (``Collection.embedding_model()``, None if never recorded)
    and chunk count - the pre-switch, new-model-dimension-free report used by
    every writer of the ``embedding_model`` config key (the RAG picker's
    ``POST /api/rag/embedding``, ``PATCH /v1/config``, and
    ``localm setup-embeddings``) to warn what an embedding-model switch is
    about to invalidate, before it happens.

    If *candidate_model* is provided, collections already built with that
    exact model are excluded: switching to the model they were built with
    will not invalidate their semantic search. A collection an add/upload/
    repair has since embedded under a SECOND model at the same dimension
    (``Collection.embedding_model_mixed()``) is never excluded this way, even
    when its recorded label happens to equal *candidate_model*.

    Does NOT assert whether a given collection's dimension will actually
    change: that would need the CANDIDATE model's own dimension, which means
    resolving and loading it. This reports only what can be read from disk:
    which collections have semantic search today, and what they were built
    with.

    Best-effort per collection: one that fails to construct is still named,
    with the failure NOTED (not silently dropped from the count). The
    exception's own text is logged server-side only, never placed in the field
    this function returns."""
    out: list = []
    for name in _st.collection_names():
        try:
            peeked = _st.Collection.peek_stats(name)
            if peeked is not None:
                if not peeked.get("has_vectors"):
                    continue
                found = _st.Collection._peek_meta(name)
                raw_model = found[2].get("embedding_model") if found else None
                built_with = str(raw_model) if raw_model else None
                mixed = bool(found[2].get("embedding_model_mixed")) if found else False
                if _provenance_excludable(built_with, mixed, candidate_model):
                    continue
                out.append({"name": name, "built_with": built_with,
                            "n_chunks": peeked.get("n_chunks")})
                continue
            coll = _st.Collection(name, cache=False)
            stats = coll.stats()
        except Exception as e:
            _log.warning("rag: %r could not be read for the embedding-switch "
                        "impact preview (%s: %s)", name, type(e).__name__, e)
            out.append({"name": name, "built_with": None, "n_chunks": None,
                        "reason": "could not be read"})
            continue
        if not stats.get("has_vectors"):
            continue
        built_with = coll.embedding_model()
        if _provenance_excludable(built_with, coll.embedding_model_mixed(),
                                   candidate_model):
            continue
        out.append({"name": name, "built_with": built_with,
                    "n_chunks": stats["n_chunks"]})
    return out


def collection_provenance_note(model: str, affected: list, *,
                                unchanged: bool = False) -> str:
    """The human-readable note accompanying a ``collection_provenance_report()``
    result, shared by every writer of ``embedding_model`` (the RAG picker,
    ``PATCH /v1/config``, ``localm setup-embeddings``) so the wording a user
    sees does not drift between which surface they switched from.

    *unchanged* is True when the caller already knows *model* is the
    currently active embedding_model, so it never ran the report at all (its
    own short-circuit: switching to what is already active cannot invalidate
    anything). Passing an empty *affected* here would otherwise be
    indistinguishable from "nothing has embeddings" - which is not what is
    known in this case, only that nothing is CHANGING."""
    if affected:
        return (
            f"Switching to '{model}' may invalidate the semantic search of "
            f"{len(affected)} existing collection(s) until they are "
            "re-embedded. The exact impact cannot be confirmed until the "
            "new model is loaded and tested - re-embed after switching if "
            "any of them drop to BM25/lexical-only.")
    if unchanged:
        return (f"'{model}' is already the active embedding model, so there "
                "is nothing to invalidate.")
    # See TestEmbeddingSetConfirmGate
    # .test_unconfirmed_with_candidate_matching_collection_provenance_reports_nothing_to_invalidate
    # and ..._with_same_active_model_reports_nothing_to_invalidate.
    return (f"Switching to '{model}' has nothing to invalidate: no existing "
            f"collection's semantic search would change.")
