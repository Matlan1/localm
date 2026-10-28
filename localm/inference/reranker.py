# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-device reranking with a reranker GGUF or a Hugging Face cross-encoder folder.

A reranker (a BERT / XLM-R cross-encoder such as bge-reranker-v2-m3, or a decoder
reranker such as Qwen3-Reranker) scores a query against each of several documents.
A GGUF runs in the same isolated worker process as the embedder
(``embedder.IsolatedEmbedder`` / ``_embedder_runner.py``), loaded with RANK
pooling, so a native driver failure cannot take this process down. A Hugging Face
folder whose architecture is a single-score ``*ForSequenceClassification`` runs
in an isolated HF worker (``backends.hf.HFBackend`` with ``task="rerank"``) and
returns the head's raw logit.

The same resident slot also serves a registered embedding GGUF that a request
names when it is not the configured ``embedding_model`` (``embedder.get_embedder``
keeps that one): its vectors come from that model, and the configured embedder is
left as it is.

One model is resident in the slot at a time, a reranker or such an embedding
model, loaded on first use and replaced when a request names a different one.
Loading serialises on the engine's process-global load lock, so it never races a
chat-model load onto the GPU.

``resolve_reranker`` maps a registered model name to its GGUF file or model folder
and refuses anything that is not a registered reranker; a request can never name a
filesystem path. ``rerank`` scores and ``rank_results`` shapes the response.
``named_embedding_path`` and ``embed`` do the same for an embedding model.
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
# same order embedder._LOCK documents. An HFReranker restart nests its own
# _call_lock between them: _LOAD_LOCK -> _call_lock -> _LOCK. _LOCK is held for the whole of a worker
# spawn and native model load, so every reader below is cheap in work and
# unbounded in waiting: none of them may be called from an `async def` handler.
_LOCK = threading.RLock()
_RERANKER = None
# (_file_key of the file _RERANKER was loaded from, taken when it was loaded,
# _RERANK or _EMBED).
_RERANKER_KEY: Optional[tuple[tuple[str, int, int], str]] = None
_RERANK = "rerank"
_EMBED = "embed"
# Requests counted on _RERANKER by _pin and not yet released by _unpin. Read and
# written with _LOCK held; zeroed whenever _RERANKER changes.
_RESIDENT_REQUESTS = 0
# Notified, with _LOCK held, when _RESIDENT_REQUESTS drops to zero or _RERANKER
# changes.
_IDLE = threading.Condition(_LOCK)
# How long a request for a model other than the resident one waits for the
# resident one's in-flight requests to finish before it is refused.
_REPLACE_WAIT_S = 120.0
# (resolved path, mtime_ns, size) -> (why that file failed to load, when). A file
# that failed is refused again without respawning a worker for _LOAD_RETRY_AFTER_S
# seconds, until it changes, or until reset_reranker() runs.
_LOAD_FAILED: dict[tuple[str, int, int], tuple[str, float]] = {}
_LOAD_RETRY_AFTER_S = 60.0
# HFReranker.stop(): how long it waits for the request holding the worker to
# leave, and how often it kills the worker again meanwhile.
_STOP_TIMEOUT = 30.0
_STOP_POLL = 0.5


class RerankerModelError(Exception):
    """A request names a model that cannot be used for reranking. *status* is
    the HTTP status the route reports."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class RerankerUnavailableError(RuntimeError):
    """The reranker model is registered but could not be loaded."""


def _file_key(path: str) -> tuple[str, int, int]:
    """Identity of the model at *path*: its resolved path with the mtime and size
    of the file, or for a model folder the newest mtime and the total size of the
    files directly in it."""
    try:
        p = Path(path)
        if p.is_dir():
            stats = [c.stat() for c in p.iterdir() if c.is_file()]
            return (str(p.resolve()), max((s.st_mtime_ns for s in stats), default=0),
                    sum(s.st_size for s in stats))
        st = p.stat()
        return (str(p.resolve()), st.st_mtime_ns, st.st_size)
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


def _registered_model_path(name: str) -> Optional[str]:
    """The GGUF file or Hugging Face model folder registered under *name*, or None
    when it names no existing local one."""
    from localm.model_manager.registry import get_model_info
    from localm.pathsafe import is_unc_or_device_path
    info = get_model_info(name)
    if not info:
        return None
    path = info[0] if isinstance(info, tuple) else info
    if not path or is_unc_or_device_path(str(path)):
        return None
    p = Path(path)
    if p.is_file():
        return str(path)
    return str(path) if (p.is_dir() and (p / "config.json").is_file()) else None


_KIND_WORDS = {
    "llm": "a chat model",
    "mmproj": "a vision projector",
    "diffusion-unet": "an image or video checkpoint",
    "text-encoder": "a text encoder",
    "vae": "a VAE",
    "lora": "a LoRA adapter",
    "upscaler": "an upscaler",
    "controlnet": "a ControlNet",
    "textual-inversion": "a textual-inversion embedding",
    "segmentation": "a segmentation model",
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
    path = _registered_model_path(name)
    if path is None:
        raise RerankerModelError(
            f"Model {name!r} is registered but its GGUF file or model folder is "
            "missing or is not a single local one.", 422)
    return name, path


def named_embedding_path(name: str, entry: dict) -> Optional[str]:
    """The GGUF file of the registered embedding model *name* (registry *entry*)
    to embed with in this module's resident slot, or None when that file is the
    one the configured ``embedding_model`` resolves to, which
    ``embedder.embed_texts`` serves.

    Raises :class:`RerankerModelError` (422) when the entry has no path, a
    network or device path, or no file on disk. Blocking: call it off the event
    loop."""
    from localm.inference import embedder as emb
    from localm.model_manager import _entry_path
    from localm.pathsafe import is_unc_or_device_path
    raw = _entry_path(entry)
    if not raw:
        raise RerankerModelError(
            f"Embedding model {name!r} is registered with no file path; remove it "
            f"(localm rm {name}) and add it again.", 422)
    if is_unc_or_device_path(str(raw)):
        raise RerankerModelError(
            f"Embedding model {name!r} is registered at a network or device path, "
            "which is not loaded; register a local GGUF file instead.", 422)
    path = Path(raw)
    if not path.is_file():
        raise RerankerModelError(
            f"Embedding model {name!r} is registered but its GGUF file is missing; "
            f"pull it again or remove the entry (localm rm {name}).", 422)
    configured = emb.configured_model_path()
    if configured is not None:
        try:
            if path.samefile(configured):
                return None
        except OSError:
            pass
    return str(path.resolve())


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


def _check_hf_head(path: str) -> None:
    """Refuse a Hugging Face folder whose complete weight listing holds no
    classification head tensor."""
    from localm.inference.backends._hf_fp8 import weight_files
    from localm.inference.hf_head_check import classifier_head_present
    present = classifier_head_present(weight_files(path))
    if present is False:
        raise RerankerHeadMissingError(
            f"{Path(path).name} has no classification head (no 'classifier' or 'score' "
            "tensor in its weights), so it would return an arbitrary value instead of "
            "a relevance score. Use a cross-encoder checkpoint that includes the head.")


class HFReranker:
    """A resident Hugging Face cross-encoder: the isolated worker behind an
    ``HFBackend(task="rerank")``, with the surface of ``IsolatedEmbedder`` that
    this module uses (``rerank``, ``cls_labels``, ``active_requests``, ``close``,
    ``model_path`` and ``_runner``)."""

    # Set by stop(): no request starts or restarts the worker afterwards.
    _stopped: bool = False

    def __init__(self, path: str, device: Optional[str] = None) -> None:
        from localm.inference.backends.hf import HFBackend
        self.model_path = path
        self.active_requests = 0
        self._call_lock = threading.Lock()
        self._backend = HFBackend(path, device=device, task="rerank")
        self._backend.load()
        self.cls_labels = list(self._backend.rerank_labels)
        self._runner = self._backend._runner

    def rerank(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Score *pairs*, starting the worker again first when it has died.

        The restart holds the engine's ``_LOAD_LOCK`` (outer) and ``_call_lock``
        (inner); ``_call_lock`` is never held while ``_LOAD_LOCK`` is awaited. A
        restart that fails releases this reranker and latches its folder like a
        failed first load, and raises :class:`RerankerUnavailableError`."""
        self.active_requests += 1
        try:
            with self._call_lock:
                self._refuse_if_stopped()
                alive = self._backend.loaded
                if alive:
                    return self._rerank_locked(pairs)
            self._restart()
            with self._call_lock:
                self._refuse_if_stopped()
                return self._rerank_locked(pairs)
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    def _rerank_locked(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Score *pairs* on the worker; a failure caused by :meth:`stop` raises
        :class:`RerankerUnavailableError`. Caller must hold ``_call_lock``."""
        try:
            return self._backend.rerank(pairs)
        except RuntimeError as e:
            if self._stopped:
                raise RerankerUnavailableError(self._stopped_message()) from e
            raise

    def _refuse_if_stopped(self) -> None:
        if self._stopped:
            raise RerankerUnavailableError(self._stopped_message())

    def _stopped_message(self) -> str:
        return (f"{Path(self.model_path).name} was unloaded while this request was "
                "running; retry the request to load it again")

    def stop(self, timeout: float = _STOP_TIMEOUT) -> bool:
        """Release the worker even while a request is in flight: the worker
        process is killed, the request running on it and every request queued
        behind it raise :class:`RerankerUnavailableError`, and once the running
        request has left, the worker is released. Waits up to *timeout*
        seconds for that request to leave. Returns whether the worker was
        released; on False its process is killed but its handles stay open."""
        self._stopped = True
        deadline = time.monotonic() + timeout
        while True:
            runner = self._backend._runner
            if runner is not None:
                runner.kill()
            if self._call_lock.acquire(timeout=_STOP_POLL):
                try:
                    self.close()
                finally:
                    self._call_lock.release()
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    "reranker %s: a request still held the worker %.0fs after it "
                    "was killed; the worker's handles were not released",
                    Path(self.model_path).name, timeout)
                return False

    def _restart(self) -> None:
        """Load the worker again under the engine's load lock when it is not
        running. Raises :class:`RerankerUnavailableError` without loading when
        this reranker is no longer the resident one."""
        from localm.inference.engine import _LOAD_LOCK
        with _LOAD_LOCK:
            if self._backend.loaded:
                return
            with self._call_lock:
                if self._backend.loaded:
                    return
                if not self._is_resident():
                    raise RerankerUnavailableError(
                        f"{Path(self.model_path).name} was released while this "
                        "request waited for it; retry the request")
                logger.warning("reranker worker is not running; reloading %s",
                               self.model_path)
                try:
                    self._backend.unload()
                    self._backend.load()
                except Exception as e:
                    reason = pathscrub.scrub_paths(str(e))
                    logger.warning("could not reload reranker %s (%s)",
                                   Path(self.model_path).name, e)
                    self.close()
                    _release_failed(self, reason)
                    raise RerankerUnavailableError(reason) from e
                self._runner = self._backend._runner

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Refuse: a cross-encoder scores query and document pairs and cannot
        produce embeddings."""
        raise RuntimeError(
            f"{Path(self.model_path).name} is a reranker: it scores query and "
            "document pairs and cannot produce embeddings")

    def _is_resident(self) -> bool:
        """Whether this reranker is still the module's resident one."""
        with _LOCK:
            return _RERANKER is self

    def close(self) -> None:
        self._backend.unload()
        self._runner = None


def _busy(model) -> bool:
    """Whether *model*, the resident one, has a request in flight. Call with
    _LOCK held."""
    return model.active_requests > 0 or _RESIDENT_REQUESTS > 0


def _release_failed(failed, reason: str) -> None:
    """Drop *failed* as the resident reranker when it still is, and latch its
    file as a failed load with *reason*."""
    global _RERANKER, _RERANKER_KEY, _RESIDENT_REQUESTS
    with _LOCK:
        if _RERANKER is failed:
            _RERANKER = None
            _RERANKER_KEY = None
            _RESIDENT_REQUESTS = 0
            _IDLE.notify_all()
        _LOAD_FAILED[_file_key(failed.model_path)] = (reason, time.monotonic())


def get_reranker(path: str):
    """The resident reranker for the GGUF file or Hugging Face model folder at
    *path*, loading it first (and releasing a different resident model).
    Raises :class:`RerankerHeadMissingError` for a model without a classifier head
    and :class:`RerankerUnavailableError` when the load fails (a failed file is not
    retried for a minute unless it changes) or a different resident model is
    still busy."""
    return _get_resident(path, _RERANK)


def get_embedding_model(path: str):
    """The resident embedding model for the GGUF file at *path*, loading it first
    (and releasing a different resident model). Pooling is the one the model
    declares (``embedder.POOLING_AUTO``); the ``embedding_pooling`` setting
    applies to the configured embedder only. Raises
    :class:`RerankerUnavailableError` when the load fails (a failed file is not
    retried for a minute unless it changes), the model loads as a reranker, or a
    different resident model is still busy after ``_REPLACE_WAIT_S``."""
    return _get_resident(path, _EMBED)


def _get_resident(path: str, purpose: str):
    """The resident model for *path* loaded for *purpose* (``_RERANK`` or
    ``_EMBED``), loading it first. A different resident model with requests in
    flight is waited for, up to ``_REPLACE_WAIT_S`` and holding no lock, before
    it is replaced. See :func:`get_reranker` and :func:`get_embedding_model`."""
    global _RERANKER, _RERANKER_KEY, _RESIDENT_REQUESTS
    from localm.config import load_config
    from localm.inference import embedder as emb
    hf_folder = purpose == _RERANK and Path(path).is_dir()
    file_key = _file_key(path)
    key = (file_key, purpose)
    with _LOCK:
        if _RERANKER is not None and _RERANKER_KEY == key:
            return _RERANKER
        failed = _latched_failure(file_key)
        if failed is not None:
            raise RerankerUnavailableError(failed)
        deadline = time.monotonic() + _REPLACE_WAIT_S
        while (_RERANKER is not None and _RERANKER_KEY != key and _RESIDENT_REQUESTS > 0
               and time.monotonic() < deadline):
            _IDLE.wait(timeout=max(0.0, deadline - time.monotonic()))
        if _RERANKER is not None and _RERANKER_KEY == key:
            return _RERANKER
    hf_device: Optional[str] = None
    pooling: object = emb.POOLING_AUTO
    if purpose == _EMBED:
        cfg = load_config()
        explicit = emb._explicit_embedder_gpu_layers(cfg)
        emb._maybe_swap_for_embedder(path, explicit if explicit is not None else 99)
        ngl, placement_reason = emb._choose_embedder_gpu_layers(path, cfg)
        if placement_reason is not None:
            logger.warning("embedding model placement: %s", placement_reason)
    elif hf_folder:
        _check_hf_head(path)
        ngl, placement_reason = 0, None
        if emb._explicit_embedder_gpu_layers(load_config()) == 0:
            hf_device = "cpu"
    else:
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
            failed = _latched_failure(file_key)
            if failed is not None:
                raise RerankerUnavailableError(failed)
            current = _RERANKER
            if current is not None and _busy(current):
                raise RerankerUnavailableError(
                    "another reranker or embedding model is still scoring or "
                    "embedding a request; retry shortly")
            if current is not None:
                _RERANKER = None
                _RERANKER_KEY = None
                _RESIDENT_REQUESTS = 0
                _IDLE.notify_all()
                current.close()
            try:
                if hf_folder:
                    loaded = HFReranker(path, hf_device)
                elif purpose == _EMBED:
                    loaded = emb.IsolatedEmbedder(
                        path, n_gpu_layers=ngl, pooling_type=pooling,
                        gpu_fallback_reason=placement_reason)
                    if getattr(loaded, "effective_pooling", None) == emb._POOLING_RANK:
                        loaded.close()
                        raise RuntimeError(
                            f"{Path(path).name} is a reranker (rank pooling), not "
                            "an embedding model: it scores query and document "
                            "pairs (POST /v1/rerank)")
                else:
                    loaded = emb.IsolatedEmbedder(
                        path, n_gpu_layers=ngl, pooling_type=emb._POOLING_RANK,
                        gpu_fallback_reason=placement_reason)
            except Exception as e:
                reason = pathscrub.scrub_paths(str(e))
                _LOAD_FAILED[file_key] = (reason, time.monotonic())
                logger.warning("could not load %s %s (%s)",
                               "reranker" if purpose == _RERANK else "embedding model",
                               Path(path).name, e)
                raise RerankerUnavailableError(reason) from e
            _RERANKER = loaded
            _RERANKER_KEY = key
            if purpose == _EMBED:
                logger.info("embedding model ready: %s (dim=%d)", Path(path).name,
                            getattr(loaded, "dim", 0))
            else:
                logger.info("reranker ready: %s (labels=%s)", Path(path).name,
                            _RERANKER.cls_labels or "none declared")
            return _RERANKER


class RerankOutcome(NamedTuple):
    """What :func:`rerank` returns: one entry per document, in request order,
    each ``{"scores", "tokens", "truncated"}``, and the classifier head's label
    names (empty when the model declares none)."""
    scored: list[dict]
    labels: list[str]


def _pin(path: str, purpose: str):
    """Load or reuse the resident model for *path* and *purpose* and count one
    request on it, in the same ``_LOCK`` hold that confirms it is still the
    resident one, so ``reset_reranker(force=False)`` leaves it loaded until
    :func:`_unpin`. Raises :class:`RerankerUnavailableError` when another
    request replaces it twice in a row."""
    global _RESIDENT_REQUESTS
    for _attempt in range(2):
        model = _get_resident(path, purpose)
        with _LOCK:
            if _RERANKER is model:
                _RESIDENT_REQUESTS += 1
                return model
    raise RerankerUnavailableError(
        f"{Path(path).name} was replaced by another model while this request "
        "waited for it; retry the request")


def _unpin(model) -> None:
    """Release the request counted by :func:`_pin` on *model*; nothing when
    *model* is no longer the resident one."""
    global _RESIDENT_REQUESTS
    with _LOCK:
        if _RERANKER is model:
            _RESIDENT_REQUESTS = max(0, _RESIDENT_REQUESTS - 1)
            if _RESIDENT_REQUESTS == 0:
                _IDLE.notify_all()


def rerank(path: str, query: str, documents: list[str]) -> RerankOutcome:
    """Score *query* against each of *documents* with the reranker at *path*."""
    emb = _pin(path, _RERANK)
    try:
        scored = emb.rerank([(query, d) for d in documents])
    finally:
        _unpin(emb)
    return RerankOutcome(scored, list(emb.cls_labels))


def embed(path: str, texts: list[str]) -> list[list[float]]:
    """Embed *texts* with the embedding model at *path* (see
    :func:`get_embedding_model`). Raises :class:`RerankerUnavailableError` as
    that does, and ``RuntimeError`` when the worker fails mid-call."""
    model = _pin(path, _EMBED)
    try:
        return model.embed(list(texts))
    finally:
        _unpin(model)


def reranker_info() -> Optional[dict]:
    """``{"path", "labels", "kind"}`` of the resident model, or None. *kind* is
    ``"reranker model"`` or ``"embedding model"``. Does not load."""
    with _LOCK:
        if _RERANKER is None:
            return None
        kind = ("embedding model" if _RERANKER_KEY is not None and _RERANKER_KEY[1] == _EMBED
                else "reranker model")
        return {"path": _RERANKER.model_path, "labels": list(_RERANKER.cls_labels),
                "kind": kind}


def is_loaded() -> bool:
    """True while a reranker is resident. Does not load."""
    with _LOCK:
        return _RERANKER is not None


def is_resident() -> bool:
    """True while a reranker is resident. Takes no lock, so an exit path can ask
    while a load is running; the answer is a snapshot."""
    return _RERANKER is not None


def active_requests() -> int:
    """In-flight requests on the resident model, or 0."""
    with _LOCK:
        if _RERANKER is None:
            return 0
        return max(_RERANKER.active_requests, _RESIDENT_REQUESTS)


def reset_reranker(*, force: bool = True) -> bool:
    """Release the resident reranker and clear the failed-load memory. Returns
    True when a reranker was released; with ``force=False`` a reranker with a
    request in flight is left alone (and nothing is cleared), checked and
    released in one locked step. With ``force=True`` a reranker with a request
    in flight is detached under ``_LOCK`` and stopped outside it (its
    ``stop()``): its worker is killed and every request in flight on it
    raises."""
    global _RERANKER, _RERANKER_KEY, _RESIDENT_REQUESTS
    with _LOCK:
        busy = _RERANKER is not None and _busy(_RERANKER)
        if not force and busy:
            return False
        current = _RERANKER
        if current is not None and not busy:
            current.close()
        _RERANKER = None
        _RERANKER_KEY = None
        _RESIDENT_REQUESTS = 0
        _IDLE.notify_all()
        _LOAD_FAILED.clear()
    if current is not None and busy:
        _stop_busy(current)
    return current is not None


def _stop_busy(model) -> None:
    """Stop *model*, detached while requests were in flight on it:
    ``model.stop()`` when it has one, else ``model.close()``."""
    stop = getattr(model, "stop", None)
    if callable(stop):
        stop()
    else:
        model.close()


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
    runner.shutdown(grace=0 if (emb.active_requests > 0 or _RESIDENT_REQUESTS > 0) else 5.0)
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
