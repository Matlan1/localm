# SPDX-License-Identifier: AGPL-3.0-or-later
"""A collection's embedding model: its recorded name and re-embedding."""

from __future__ import annotations

from typing import Optional

from .cache import _files_changing
from .types import EmbedFn, ProgressFn


class _CollectionEmbedding:
    """Mixin of ``Collection``: the recorded embedding model and re-embedding from
    the stored text."""

    def embedding_model(self) -> Optional[str]:
        """The model NAME this collection's vectors were built with, if recorded."""
        v = self._meta.get("embedding_model")
        return str(v) if v else None

    def embedding_model_mixed(self) -> bool:
        """True when an add/upload/repair embedded into this collection under a
        model different from ``embedding_model()``, so that label no longer
        describes every stored vector. Cleared by ``reembed()``."""
        return bool(self._meta.get("embedding_model_mixed"))

    def _reembed_locked(self, *, embed_fn: EmbedFn,
                        model_name: Optional[str] = None,
                        on_progress: Optional[ProgressFn] = None,
                        batch: int = 32) -> dict:
        """The reembed body. MUST run under _collection_lock after _load()."""
        if not self._chunks:
            return {"chunks": 0, "dim": None, "model": model_name,
                    "note": "collection has no chunks; nothing to re-embed"}

        texts = [c.get("text") or "" for c in self._chunks]
        total = len(texts)
        fresh: list = []
        for i in range(0, total, max(1, batch)):
            part = embed_fn(texts[i:i + max(1, batch)])
            if part is None:
                raise RuntimeError(
                    "the embedding function returned nothing for chunks "
                    f"{i}-{i + len(texts[i:i + batch])} of {total}")
            fresh.extend(part)
            if on_progress:
                done = min(i + batch, total)
                on_progress(f"re-embedding {done}/{total}", phase="re-embedding",
                            done=done, total=total, unit="chunks")

        # Validate BEFORE touching the live index: a short or ragged result is
        # refused here, leaving the previous index untouched.
        if len(fresh) != total:
            raise RuntimeError(
                f"embedder returned {len(fresh)} vectors for {total} chunks; "
                "the previous index has been left untouched")
        dims = {len(v) for v in fresh if v is not None}
        if len(dims) != 1 or not dims or next(iter(dims)) <= 0:
            raise RuntimeError(
                f"embedder returned inconsistent vector sizes {sorted(dims)}; "
                "the previous index has been left untouched")

        self._vectors = fresh
        self._vec_dim = next(iter(dims))
        # Record the model NAME as well as its dimension, so a later mismatch
        # can say which model built this index.
        if model_name:
            self._meta["embedding_model"] = str(model_name)
        # Every vector just came from this one embed_fn call, so the
        # collection is no longer mixed regardless of the label above.
        self._meta.pop("embedding_model_mixed", None)
        self._meta["embedding_dim"] = self._vec_dim
        # A rebuilt full-coverage index clears the degrade state and makes any
        # set-aside sidecar moot.
        self._vectors_file_rejected = False
        self.vector_degrade_reason = None
        self.corrupt = False
        self._save()
        with _files_changing(self.dir):
            self._discard_rejected_vectors(
                f"re-embedded to {self._vec_dim} dimensions"
                + (f" with {model_name}" if model_name else ""))
        return {"chunks": total, "dim": self._vec_dim, "model": model_name}
