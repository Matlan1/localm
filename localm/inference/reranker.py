# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-device reranking with a reranker GGUF.

A reranker (a BERT / XLM-R cross-encoder such as bge-reranker-v2-m3, or a decoder
reranker such as Qwen3-Reranker) scores a query against each of several documents.
It runs in the same isolated worker process as the embedder
(``embedder.IsolatedEmbedder`` / ``_embedder_runner.py``), loaded with RANK
pooling, so a native driver failure cannot take this process down.

One reranker is resident at a time, loaded on first use and replaced when a
request names a different one. Loading serialises on the engine's process-global
load lock, so it never races a chat-model load onto the GPU.

``resolve_reranker`` maps a registered model name to its GGUF path and refuses
anything that is not a registered reranker; a request can never name a filesystem
path. ``rerank`` scores and ``rank_results`` shapes the response.
"""

from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path
from typing import NamedTuple, Optional

from localm import pathscrub
from localm.debuglog import logger
from localm.inference.backends.base import RerankerHeadMissingError

# LOCK ORDER: engine._LOAD_LOCK (outer) -> _LOCK (inner), never the reverse, the
# same order embedder._LOCK documents. _LOCK is held for the whole of a worker
# spawn and native model load, so every reader below is cheap in work and
# unbounded in waiting: none of them may be called from an `async def` handler.
_LOCK = threading.RLock()
_RERANKER = None
# _file_key of the file _RERANKER was loaded from, taken when it was loaded.
_RERANKER_KEY: Optional[tuple[str, int, int]] = None
# (resolved path, mtime_ns, size) -> (why that file failed to load, when). A file
# that failed is refused again without respawning a worker for _LOAD_RETRY_AFTER_S
# seconds, until it changes, or until reset_reranker() runs.
_LOAD_FAILED: dict[tuple[str, int, int], tuple[str, float]] = {}
_LOAD_RETRY_AFTER_S = 60.0


class RerankerModelError(Exception):
    """A request names a model that cannot be used for reranking. *status* is
    the HTTP status the route reports."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class RerankerUnavailableError(RuntimeError):
    """The reranker model is registered but could not be loaded."""


def _file_key(path: str) -> tuple[str, int, int]:
    try:
        st = Path(path).stat()
        return (str(Path(path).resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), 0, 0)


def _latched_failure(key: tuple[str, int, int]) -> Optional[str]:
    """Why a recent load of the file with this key failed, or None once the retry
    window has passed. Call with _LOCK held."""
    failed = _LOAD_FAILED.get(key)
    if failed is None:
        return None
    reason, at = failed
    if time.monotonic() - at < _LOAD_RETRY_AFTER_S:
        return reason
    del _LOAD_FAILED[key]
    return None


def _registered_gguf_path(name: str) -> Optional[str]:
    """The GGUF file registered under *name*, or None when it names no single
    existing local file."""
    from localm.model_manager.registry import get_model_info
    from localm.pathsafe import is_unc_or_device_path
    info = get_model_info(name)
    if not info:
        return None
    path = info[0] if isinstance(info, tuple) else info
    if not path or is_unc_or_device_path(str(path)):
        return None
    return str(path) if Path(path).is_file() else None


_KIND_WORDS = {
    "llm": "a chat model",
    "mmproj": "a vision projector",
    "diffusion-unet": "an image or video checkpoint",
    "text-encoder": "a text encoder",
    "vae": "a VAE",
    "lora": "a LoRA adapter",
    "tts": "a text-to-speech model",
    "unknown": "a model of unknown type",
}


def registered_rerankers() -> list[str]:
    """Names of the registered models that are rerankers, sorted."""
    from localm.config import load_registry
    from localm.model_manager.registry import entry_is_reranker
    reg = load_registry()
    return sorted(name for name, entry in reg.items()
                  if isinstance(entry, dict) and entry_is_reranker(name, entry))


def resolve_reranker(model: Optional[str]) -> tuple[str, str]:
    """``(registry name, GGUF path)`` for the reranker a request names.

    An omitted or ``localm`` model resolves to the only registered reranker.
    Raises :class:`RerankerModelError` (with the HTTP status to report) when the
    model is not registered (404), names no reranker or several (400), or is
    registered as something that cannot rerank (422)."""
    from localm.config import load_registry
    from localm.model_manager.registry import entry_is_reranker
    reg = load_registry()
    name = (model or "").strip()
    if not name or name == "localm":
        names = registered_rerankers()
        if not names:
            raise RerankerModelError(
                "No reranker model is registered. Add a reranker GGUF (for "
                "example bge-reranker-v2-m3 or Qwen3-Reranker) and name it in "
                "the request's model field.", 404)
        if len(names) > 1:
            raise RerankerModelError(
                "Several rerankers are registered (" + ", ".join(names) +
                "); name one in the request's model field.", 400)
        name = names[0]
    entry = reg.get(name)
    if not isinstance(entry, dict):
        raise RerankerModelError(f"Model {name!r} is not registered.", 404)
    kind = entry.get("model_type") or "llm"
    if not isinstance(kind, str):
        kind = "unknown"
    if kind != "embedding":
        what = _KIND_WORDS.get(kind, f"a {kind} model")
        raise RerankerModelError(
            f"Model {name!r} is {what}, not a reranker; /v1/rerank needs a "
            "reranker GGUF.", 422)
    if not entry_is_reranker(name, entry):
        raise RerankerModelError(
            f"Model {name!r} is an embedding model, not a reranker: it has no "
            "classifier head to score a query against a document. Use "
            "/v1/embeddings for it.", 422)
    path = _registered_gguf_path(name)
    if path is None:
        raise RerankerModelError(
            f"Model {name!r} is registered but its GGUF file is missing or is "
            "not a single local file.", 422)
    return name, path


def _check_head(path: str) -> None:
    """Refuse a model whose complete tensor list holds no classifier head."""
    from localm.model_manager.gguf import gguf_classifier_head_tensors
    heads = gguf_classifier_head_tensors(Path(path))
    if heads is not None and not heads:
        raise RerankerHeadMissingError(
            f"{Path(path).name} has no classifier head (no 'cls.weight' or "
            "'cls.output.weight' tensor), so it would return an arbitrary "
            "pooled value instead of a relevance score. Use a reranker "
            "conversion that includes the head.")


def get_reranker(path: str):
    """The resident reranker for the GGUF at *path*, loading it first (and
    releasing a different resident reranker). Raises
    :class:`RerankerHeadMissingError` for a model without a classifier head and
    :class:`RerankerUnavailableError` when the load fails (a failed file is not
    retried for a minute unless it changes) or a different reranker is still
    busy."""
    global _RERANKER, _RERANKER_KEY
    from localm.config import load_config
    from localm.inference import embedder as emb
    key = _file_key(path)
    with _LOCK:
        if _RERANKER is not None and _RERANKER_KEY == key:
            return _RERANKER
        failed = _latched_failure(key)
        if failed is not None:
            raise RerankerUnavailableError(failed)
    _check_head(path)
    cfg = load_config()
    explicit = emb._explicit_embedder_gpu_layers(cfg)
    emb._maybe_swap_for_embedder(path, explicit if explicit is not None else 99)
    ngl, placement_reason = emb._choose_embedder_gpu_layers(path, cfg)
    if placement_reason is not None:
        logger.warning("reranker placement: %s", placement_reason)
    from localm.inference.engine import _LOAD_LOCK
    with _LOAD_LOCK:
        with _LOCK:
            if _RERANKER is not None and _RERANKER_KEY == key:
                return _RERANKER
            failed = _latched_failure(key)
            if failed is not None:
                raise RerankerUnavailableError(failed)
            current = _RERANKER
            if current is not None and current.active_requests > 0:
                raise RerankerUnavailableError(
                    "another reranker is still scoring a request; retry shortly")
            if current is not None:
                _RERANKER = None
                _RERANKER_KEY = None
                current.close()
            try:
                _RERANKER = emb.IsolatedEmbedder(
                    path, n_gpu_layers=ngl, pooling_type=emb._POOLING_RANK,
                    gpu_fallback_reason=placement_reason)
            except Exception as e:
                reason = pathscrub.scrub_paths(str(e))
                _LOAD_FAILED[key] = (reason, time.monotonic())
                logger.warning("could not load reranker %s (%s)",
                               Path(path).name, e)
                raise RerankerUnavailableError(reason) from e
            _RERANKER_KEY = key
            logger.info("reranker ready: %s (labels=%s)", Path(path).name,
                        _RERANKER.cls_labels or "none declared")
            return _RERANKER


class RerankOutcome(NamedTuple):
    """What :func:`rerank` returns: one entry per document, in request order,
    each ``{"scores", "tokens", "truncated"}``, and the classifier head's label
    names (empty when the model declares none)."""
    scored: list[dict]
    labels: list[str]


def rerank(path: str, query: str, documents: list[str]) -> RerankOutcome:
    """Score *query* against each of *documents* with the reranker at *path*."""
    emb = get_reranker(path)
    scored = emb.rerank([(query, d) for d in documents])
    return RerankOutcome(scored, list(emb.cls_labels))


def reranker_info() -> Optional[dict]:
    """``{"path", "labels"}`` of the resident reranker, or None. Does not load."""
    with _LOCK:
        if _RERANKER is None:
            return None
        return {"path": _RERANKER.model_path, "labels": list(_RERANKER.cls_labels)}


def is_loaded() -> bool:
    """True while a reranker is resident. Does not load."""
    with _LOCK:
        return _RERANKER is not None


def is_resident() -> bool:
    """True while a reranker is resident. Takes no lock, so an exit path can ask
    while a load is running; the answer is a snapshot."""
    return _RERANKER is not None


def active_requests() -> int:
    """In-flight rerank calls on the resident reranker, or 0."""
    with _LOCK:
        return _RERANKER.active_requests if _RERANKER is not None else 0


def reset_reranker(*, force: bool = True) -> bool:
    """Release the resident reranker and clear the failed-load memory. Returns
    True when a reranker was released; with ``force=False`` a reranker with a
    request in flight is left alone (and nothing is cleared), checked and
    released in one locked step."""
    global _RERANKER, _RERANKER_KEY
    with _LOCK:
        if not force and _RERANKER is not None and _RERANKER.active_requests > 0:
            return False
        current = _RERANKER
        if current is not None:
            current.close()
        _RERANKER = None
        _RERANKER_KEY = None
        _LOAD_FAILED.clear()
        return current is not None


def release_for_exit() -> bool:
    """Release the reranker's worker for a caller about to ``os._exit()`` /
    ``os.execv()``, which bypass atexit and would orphan it with its model in
    VRAM. Takes no lock (a load holds ``_LOCK`` for its whole duration): a busy
    worker is terminated without waiting, an idle one is closed politely. The
    module state is left for reset_reranker() to clear."""
    emb = _RERANKER
    if emb is None:
        return False
    runner = getattr(emb, "_runner", None)
    if runner is None:
        return False
    runner.shutdown(grace=0 if emb.active_requests > 0 else 5.0)
    return True


def rank_results(scored: list[dict], top_n: Optional[int] = None,
                 labels: Optional[list[str]] = None) -> list[dict]:
    """Rerank results from per-document *scored* entries (as returned by
    :func:`rerank`, aligned with the request's documents): ``{"index",
    "relevance_score"}`` for each, best first (ties keep request order), cut to
    *top_n*. A truncated document carries ``"truncated": true``; a head with
    several outputs adds ``"label_scores"`` keyed by label."""
    rows = []
    for index, item in enumerate(scored):
        row = {"index": index, "relevance_score": item["scores"][0]}
        if item.get("truncated"):
            row["truncated"] = True
        if len(item["scores"]) > 1:
            names = labels or []
            row["label_scores"] = {
                (names[i] if i < len(names) and names[i] else str(i)): s
                for i, s in enumerate(item["scores"])}
        rows.append(row)
    rows.sort(key=lambda r: (-r["relevance_score"], r["index"]))
    return rows if top_n is None else rows[:max(0, top_n)]


atexit.register(reset_reranker)
