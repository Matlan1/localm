# SPDX-License-Identifier: AGPL-3.0-or-later
"""
On-device text embeddings via a DEDICATED embedding GGUF, on the bundled runtime.

This module loads a small, dedicated embedding model (bge / nomic, ~25-90 MB) into
its OWN llama.cpp model + context in EMBEDDINGS mode, independent of whatever chat
model is loaded, so semantic retrieval (RAG hybrid search, agent memory) works on
the default GGUF runtime with consistent quality and without disturbing the chat
model. The bundled GGUF chat backend cannot embed at all (``backends/gguf.py``,
``can_embed=False``).

It is a process-wide, lazily-loaded singleton (``get_embedder`` / ``embed_texts``):
one small model is loaded once and shared by every caller (chat memory, coder
memory, /v1/embeddings). Loading serialises on the engine's process-global load lock
so it never races a chat-model load onto the GPU.

Provisioning (``resolve_embedding_model_path``): the ``embedding_model`` config key
is either a filesystem path, a registered model name, or a known key (default
``bge-small-en-v1.5``). A known model missing from ``<home>/models/embeddings/`` is
downloaded on demand, gated by the network policy (never behind ``net_mode=off``
unless ``net_allow_model_downloads`` exempts it; auto only under ``net_mode=allow``
- otherwise the user runs ``localm setup-embeddings`` or the GUI's download action).
When no embedding model can be resolved, callers degrade to lexical-only retrieval;
the reason is recorded for ``last_error()`` (what the GUI's RAG-embedding status
reads) on every resolve attempt, not only when a load is actually attempted. A
registered model can name something real and still not be usable HERE, because
this embedder loads a single GGUF file only; a HuggingFace-format pull (a
directory of shards, not a GGUF) is served by ``inference/backends/hf.py``'s
``HFBackend`` when such a checkpoint is loaded as the primary model instead.

The native load (and every ``embed()`` call - it hits ``llama_decode``, the
same abort-prone native call class) runs inside an ISOLATED CHILD PROCESS
(``_embedder_runner.py``), not in this process, so a native driver failure that
``abort()``s in C cannot take this process down. ``GGUFEmbedder`` below is the
raw, unguarded native loader (constructed only inside that child);
``IsolatedEmbedder`` is the parent-side handle ``get_embedder()`` returns.
"""

from __future__ import annotations

import atexit
import ctypes
import math
import re
import threading
from pathlib import Path
from typing import Callable, List, Optional

from localm import pathscrub
from localm.debuglog import dedup_native_stderr, logger
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

# Known small embedding models, keyed by friendly name -> (hf_repo, filename).
# embedding_model may also name any GGUF path or a registered model.
KNOWN_EMBEDDING_MODELS = {
    "bge-small-en-v1.5": (
        "CompendiumLabs/bge-small-en-v1.5-gguf", "bge-small-en-v1.5-q4_k_m.gguf"),
    "nomic-embed-text-v1.5": (
        "nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
}
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"

# HF endpoint for embedding-model downloads. The ambient HF_ENDPOINT /
# HF_HUB_ENDPOINT env vars are not read.
_HF_ENDPOINT = "https://huggingface.co"

# llama.cpp LLAMA_POOLING_TYPE_* values. UNSPECIFIED (-1) is llama.cpp's own
# default and means the model's declared pooling decides.
_POOLING_UNSPECIFIED = -1
_POOLING_NONE = 0
_POOLING_MEAN = 1
_POOLING_CLS = 2
_POOLING_LAST = 3

# Pooling settings a user may choose (config embedding_pooling). auto is not a
# llama.cpp value: it is resolved per-model against what the GGUF declares.
POOLING_AUTO = "auto"
_POOLING_BY_NAME = {"none": _POOLING_NONE, "mean": _POOLING_MEAN,
                    "cls": _POOLING_CLS, "last": _POOLING_LAST}
_POOLING_NAMES = {v: k for k, v in _POOLING_BY_NAME.items()}
POOLING_CHOICES = [POOLING_AUTO, *_POOLING_BY_NAME]

# Pooling used when embedding_pooling was never configured: MEAN for every
# declaration (CLS, MEAN, unspecified) except LAST, which resolves to LAST. An
# explicit choice, including an explicit mean, always wins over this.
_POOLING_DEFAULT = _POOLING_MEAN
# Marks embedding_pooling as never configured, distinct from both an explicit
# numeric override and POOLING_AUTO (which follows whatever the model declares).
# Internal only, and never offered in POOLING_CHOICES.
_POOLING_UNSET = "unset"
# The isolated worker sizes the embedding window to the loaded model's own
# native training context (llama_model_n_ctx_train, read once the model handle
# exists - see GGUFEmbedder.__init__). _EMBED_CTX_CEILING caps that auto-sized
# window; an explicit n_ctx argument wins outright over the auto-sized value.
_EMBED_CTX_CEILING = 2048
# Used only when a model's native window cannot be read at all: the API call
# fails, or it returns an implausible value <= 0.
_EMBED_CTX_FALLBACK = 512


def _resolve_embed_ctx(native_ctx_train: int) -> int:
    """Cap the model's declared training window *native_ctx_train* at
    _EMBED_CTX_CEILING, or return _EMBED_CTX_FALLBACK when the model does not
    usefully declare one (<= 0)."""
    return min(native_ctx_train, _EMBED_CTX_CEILING) if native_ctx_train > 0 else _EMBED_CTX_FALLBACK


# The largest batch (in TEXTS, not tokens) this embedder will try to pack into
# one native multi-sequence llama_decode call. See _choose_n_seq_max.
_EMBED_BATCH_TARGET = 32


def configure_embed_context(cp, n_ctx: int, n_seq_max: int, pooling_type: int):
    """Fill a llama_context_params for EMBEDDING and return it.

    kv_unified: ONE shared KV cache across the sequences of a batch, instead of
    llama.cpp's default of carving n_ctx into n_seq_max private slices.
    _pack_groups bounds a group by its SUMMED token count against n_ctx, which
    is the budget a shared cache requires.
    """
    cp.n_ctx = n_ctx
    cp.n_batch = n_ctx
    cp.n_ubatch = n_ctx      # non-causal encode needs ubatch >= seq len
    cp.n_seq_max = n_seq_max
    cp.kv_unified = True
    cp.embeddings = True
    cp.pooling_type = pooling_type
    return cp


def _choose_n_seq_max(n_ubatch: int, target_max: int = _EMBED_BATCH_TARGET) -> int:
    """How many sequences a multi-sequence embed batch may use, for a context
    whose n_ubatch is *n_ubatch*.

    An ``n_seq_max`` that does not evenly divide its context's ``n_ubatch``
    makes llama.cpp hit an uncatchable native
    ``GGML_ASSERT(ggml_can_mul_mat(a, b))`` abort() *during context creation*,
    before any batch is submitted, so the value returned here must always
    divide *n_ubatch* exactly. This is a property of the (n_seq_max, n_ubatch)
    pair, not of the model.

    Searches powers of two descending from *target_max*, returning the first
    that evenly divides n_ubatch, and falls back to 1, which always divides."""
    n = target_max
    while n > 1:
        if n_ubatch % n == 0:
            return n
        n //= 2
    return 1


def resolve_pooling_setting(spec: object) -> object:
    """Map the ``embedding_pooling`` config value to a llama.cpp pooling int,
    POOLING_AUTO for per-model resolution, or _POOLING_UNSET when nothing was
    configured (see _effective_pooling for what that resolves to). An
    unrecognised value warns and returns _POOLING_UNSET; it never fails the
    load."""
    if spec is None:
        return _POOLING_UNSET
    text = str(spec).strip().lower()
    if not text:
        return _POOLING_UNSET
    if text == POOLING_AUTO:
        return POOLING_AUTO
    if text in _POOLING_BY_NAME:
        return _POOLING_BY_NAME[text]
    logger.warning(
        "embedding_pooling=%r is not one of %s; using the default",
        spec, ", ".join(POOLING_CHOICES))
    return _POOLING_UNSET


def declared_pooling_type(model, api) -> Optional[int]:
    """The pooling type the GGUF itself DECLARES (``<arch>.pooling_type``), or
    None when it declares none. Read from model metadata, so it needs no context.

    Best-effort and never raises: a model that declares nothing, or metadata
    that cannot be read, returns None and is debug-logged, and the caller then
    keeps its configured pooling."""
    try:
        if not api.has_model_meta_api():
            logger.debug("llama.dll exports no metadata reader; cannot read the "
                         "model's declared pooling type")
            return None
        arch = api.llama_model_meta_val_str(model, "general.architecture")
        if not arch:
            return None
        raw = api.llama_model_meta_val_str(model, f"{arch}.pooling_type")
        if raw is None:
            return None
        value = int(str(raw).strip())
        # A model that explicitly declares UNSPECIFIED is reported the same as
        # an absent key, so auto falls back to MEAN.
        return None if value == _POOLING_UNSPECIFIED else value
    except Exception as e:
        logger.debug("could not read the declared pooling type (%s: %s)",
                     type(e).__name__, e)
        return None


def _effective_pooling(requested: object, declared: Optional[int]) -> int:
    """Resolve the pooling actually used:
    - UNSET (nothing configured) resolves to MEAN, except when the model
      declares LAST-token pooling, which resolves to LAST;
    - an explicit choice (a real pooling int, including an explicit "mean") is
      returned as-is, never overridden;
    - AUTO honours what the model declares, falling back to MEAN when it
      declares nothing usable."""
    if requested == _POOLING_UNSET:
        return _POOLING_LAST if declared == _POOLING_LAST else _POOLING_DEFAULT
    if requested != POOLING_AUTO:
        return int(requested)
    if declared in (_POOLING_MEAN, _POOLING_CLS, _POOLING_LAST):
        return declared
    return _POOLING_DEFAULT


def pooling_name(value: Optional[int]) -> str:
    if value is None:
        return "not declared"
    return _POOLING_NAMES.get(value, f"type {value}")


def _embeddings_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "models" / "embeddings"


_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def _current_spec() -> str:
    """The ``embedding_model`` config value, defaulted and stripped - the
    identity ``resolve_embedding_model_path`` resolves from, and the same
    identity ``_LOAD_FAILED_SPEC``/``_set_resolve_outcome``/``get_embedder``
    compare against to tell "still the same broken spec" from "the config
    changed"."""
    from localm.config import load_config
    return str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()


def _nonlocal_spec_reason(spec: str) -> Optional[str]:
    """Why *spec* is not something this module may hand to the filesystem, or
    None if it is fine. Purely LEXICAL - no syscall - so it can run first.

    ``embedding_model`` accepts exactly three shapes: a KNOWN_EMBEDDING_MODELS
    key, a registered model name, or a LOCAL GGUF path. A UNC/device path and a
    URL are none of those and are refused here.

    The UNC/device half delegates to pathsafe's shared predicate, which also
    covers the mixed separator spellings (``\\/host\\share``) Windows resolves
    as UNC. The URL-scheme regex requires TWO or more scheme characters, so a
    Windows drive letter can never match: ``C://models/x.gguf`` stays a path
    while ``http://``, ``file://`` and ``smb://`` are caught."""
    from localm.pathsafe import is_unc_or_device_path
    if is_unc_or_device_path(spec):
        return ("it is a UNC or device path; only a local filesystem path, a "
                "known key, or a registered model name is allowed")
    if _URL_SCHEME_RE.match(spec):
        return "it is a URL, not a local file path"
    return None


def _set_resolve_outcome(spec: str, reason: Optional[str]) -> None:
    """The ONE place any RESOLVE-side code may touch ``_LAST_ERROR`` - never
    assigned directly anywhere else in this module below this point. Every
    resolve-side writer (``_record_resolve_failure``, ``_record_resolve_success``,
    and ``_download_known``'s two policy-decline branches) routes through this
    single choke point.

    *spec* is the ``embedding_model`` config value the CALLER is currently
    resolving - required, not optional: it is what distinguishes a stale
    failure latched for an abandoned spec from a live one for the spec
    actually configured now.

    Suppression applies ONLY when *spec* matches the latched
    ``_LOAD_FAILED_SPEC`` exactly, and then covers both outcomes: a successful
    resolve and a resolve FAILURE for that spec are both discarded. A resolve
    outcome for ANY OTHER spec is never suppressed, regardless of what is
    latched. What is latched otherwise changes only on an ACTUAL new load
    attempt for a MATCHING spec (``get_embedder()``'s own except/success
    branches), a load attempt for a DIFFERENT spec (which clears the stale
    latch itself - see ``get_embedder()``), or an explicit
    ``reset_embedder()``."""
    global _LAST_ERROR
    if _LOAD_FAILED_SPEC is not None and _LOAD_FAILED_SPEC == spec:
        return
    _LAST_ERROR = reason


def _record_resolve_failure(spec: str, reason: str) -> None:
    """Record *reason* as the current resolve failure for *spec* (subject to
    ``_set_resolve_outcome``'s guard), and log it - WARNING the first time
    this exact reason is seen, DEBUG on repeats of the same one.

    ``resolve_embedding_model_path`` can run on every single ``embed_texts()``
    call while no embedder is loaded, so an unchanged misconfiguration re-hits
    this function on every embed. ``last_error()`` carries the reason on every
    call regardless of the log level."""
    global _LAST_RESOLVE_WARNED
    _set_resolve_outcome(spec, reason)
    if _LAST_RESOLVE_WARNED != reason:
        logger.warning(reason)
        _LAST_RESOLVE_WARNED = reason
    else:
        logger.debug(reason)


def _record_resolve_success(spec: str) -> None:
    """Clear a prior RESOLVE failure once *spec* resolves (subject to
    ``_set_resolve_outcome``'s guard), so ``last_error()`` stops reporting it,
    and clear the warn-dedup entry so a later, DIFFERENT misconfiguration
    warns again."""
    global _LAST_RESOLVE_WARNED
    _set_resolve_outcome(spec, None)
    _LAST_RESOLVE_WARNED = None


def resolve_embedding_model_path(*, allow_download: Optional[bool] = None) -> Optional[str]:
    """Resolve the configured embedding model to a GGUF path, or None.

    Order: an explicit filesystem path -> a registered model name -> a known key
    (downloaded into <home>/models/embeddings if missing and the net policy allows).
    ``allow_download`` overrides the policy (used by ``localm setup-embeddings`` to
    force the fetch); default follows net_mode (auto only under 'allow').

    Every failure to resolve is recorded via ``last_error()`` (see
    ``_record_resolve_failure``/``_record_resolve_success``) - including a
    registered model that names something real but is not a single GGUF file
    (almost always a HuggingFace-format pull: a directory of safetensors shards,
    not a file)."""
    spec = _current_spec()
    if not spec:
        return None

    # 0. Refuse a non-local spec BEFORE any filesystem call: a single is_file()
    #    on a UNC path reaches the Windows SMB redirector. Returns None,
    #    recording and logging what was refused.
    bad = _nonlocal_spec_reason(spec)
    if bad:
        _record_resolve_failure(
            spec,
            # Not {spec!r}: repr() doubles the separators in a Windows path,
            # which pathscrub does not recognise.
            f"ignoring embedding_model '{spec}': {bad}. Use a known key "
            f"{tuple(KNOWN_EMBEDDING_MODELS)}, a registered model name, or a "
            "local GGUF path.")
        return None

    # 1. An explicit path to a GGUF.
    p = Path(spec).expanduser()
    if p.is_file():
        _record_resolve_success(spec)
        return str(p)

    # 2. A registered model name. A hit that does not resolve to a single FILE
    #    is reported as a distinct outcome from not being registered at all; a
    #    HuggingFace-format model lands as a DIRECTORY, and get_model_info's
    #    contract is exists (file or directory), not is-a-loadable-GGUF.
    registered_not_gguf = None   # None: no registry hit; else bool, is it a directory
    try:
        from localm.model_manager.registry import get_model_info
        info = get_model_info(spec)
        if info:
            path = info[0] if isinstance(info, tuple) else info
            # Same ordering rule as step 0: refuse a UNC/device path BEFORE any
            # stat. A UNC hit falls through exactly as an absent entry does.
            from localm.pathsafe import is_unc_or_device_path
            if path and not is_unc_or_device_path(str(path)):
                p2 = Path(path)
                if p2.is_file():
                    _record_resolve_success(spec)
                    return str(path)
                try:
                    registered_not_gguf = p2.is_dir()
                except OSError:
                    # A stat that fails differently from the is_file() above (a
                    # permission error, a race with a delete) is still a real
                    # registry hit, of unknown shape: reported generically
                    # rather than letting the exception escape.
                    registered_not_gguf = False
    except Exception:
        pass

    # 3. A known embedding-model key.
    known = KNOWN_EMBEDDING_MODELS.get(spec)
    if not known:
        # Neither branch uses {spec!r} - see the step-0 refusal above.
        if registered_not_gguf is not None:
            kind = ("a directory (a HuggingFace-format model)"
                    if registered_not_gguf else "not a single file")
            reason = (
                f"embedding_model '{spec}' is registered but resolves to {kind}, "
                "not a GGUF file. This dedicated embedder only loads a GGUF "
                f"embedding model: pick a known key {tuple(KNOWN_EMBEDDING_MODELS)}, "
                "a registered GGUF model, or a local .gguf path. A HuggingFace "
                "embedding checkpoint is used directly when loaded as the primary "
                "model instead, not via embedding_model.")
        else:
            reason = (
                f"embedding_model '{spec}' is not a path, a registered model, or "
                f"a known key {tuple(KNOWN_EMBEDDING_MODELS)}.")
        _record_resolve_failure(spec, reason)
        return None
    repo, filename = known
    dest = _embeddings_dir() / filename
    if dest.is_file():
        _record_resolve_success(spec)
        return str(dest)
    result = _download_known(spec, repo, filename, dest, allow_download)
    if result:
        _record_resolve_success(spec)
    return result


def _download_known(name: str, repo: str, filename: str, dest: Path,
                    allow_download: Optional[bool]) -> Optional[str]:
    """Fetch a known embedding GGUF, gated by the network policy.

    Every failure path also records into ``last_error()``; both policy
    branches below route through the SAME ``_set_resolve_outcome`` choke
    point, so its spec-aware guard applies here too (*name* is that spec,
    since this is only ever called with the ``spec`` a caller in
    ``resolve_embedding_model_path`` was already resolving). The two
    policy-gated branches log at INFO regardless of whether the write is
    skipped; only the download-failure branch warns, deduped the same way as
    ``resolve_embedding_model_path``'s own WARNING-worthy failures."""
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if allow_download is None:
        allow_download = network_mode() == "allow"
    if not allow_download:
        reason = (
            f"embedding model {name!r} not present and not auto-downloading "
            f"(net_mode={network_mode()}); use the one-time download action, "
            "or set net_mode=allow, to enable semantic search (memory/RAG use "
            "lexical BM25 until then)")
        _set_resolve_outcome(name, reason)
        logger.info(reason)
        return None
    if network_mode() == "off" and not downloads_allowed_when_off():
        reason = f"embedding model {name!r} missing and network is off; lexical-only"
        _set_resolve_outcome(name, reason)
        logger.info(reason)
        return None
    try:
        from huggingface_hub import hf_hub_download
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading embedding model %s/%s (one-time)...", repo, filename)
        got = hf_hub_download(repo, filename, local_dir=str(dest.parent), endpoint=_HF_ENDPOINT)
        # hf may nest under the repo dir; normalise to dest.
        got_p = Path(got)
        if got_p.resolve() != dest.resolve() and got_p.is_file():
            return str(got_p)
        return str(dest) if dest.is_file() else (str(got_p) if got_p.is_file() else None)
    except Exception as e:
        _record_resolve_failure(
            name, f"embedding model {name!r} download failed ({e}); lexical-only")
        return None


def _warn_if_context_config_drifted(api, ctx, requested_n_ctx: int) -> None:
    """Best-effort, log-only comparison of what this embedder REQUESTED for its
    context (n_ctx, and kv_unified=True - see configure_embed_context) against
    what the loaded native runtime actually reports, via llama_n_ctx /
    llama_n_ctx_seq - the closest observable proxy for kv_unified, since
    llama.cpp exposes no direct getter for that flag once a context exists.

    With kv_unified honoured, n_ctx_seq is close to n_ctx; a runtime that
    silently falls back to slicing n_ctx across n_seq_max private KV slots
    instead reports a much smaller n_ctx_seq, which is what a large embedding
    batch failing native decode looks like. This can only OBSERVE and log the
    drift, not explain it.

    Never raises: a probe failure here must not take down an otherwise-working
    embedder, and this never changes what was actually configured."""
    try:
        actual_n_ctx = int(api.llama_n_ctx(ctx))
        actual_n_ctx_seq = api.llama_n_ctx_seq(ctx)
    except Exception as e:
        logger.debug("embedder context drift check failed: %s", e)
        return
    if actual_n_ctx != requested_n_ctx:
        logger.warning(
            "embedder: requested n_ctx=%d but the loaded context reports "
            "n_ctx=%d - the runtime may not honour the requested context size",
            requested_n_ctx, actual_n_ctx)
    if not isinstance(actual_n_ctx_seq, int):
        return                    # no accessor (older build), or a non-int
                                   # probe result - nothing usable to compare
    # A generous margin, not exact equality: kv_unified honoured can still
    # leave n_ctx_seq a hair under n_ctx on some builds. Below half is only
    # reachable via the sliced-KV fallback (n_ctx / n_seq_max, with n_seq_max
    # always >= 2 whenever this matters - see _choose_n_seq_max), so it is the
    # threshold for "kv_unified was silently not honoured".
    if actual_n_ctx_seq < requested_n_ctx // 2:
        logger.warning(
            "embedder: requested kv_unified=True (n_ctx=%d) but the loaded "
            "context reports n_ctx_seq=%d, far smaller than requested - "
            "kv_unified was likely not honoured by this runtime build, and a "
            "large embedding batch may fail to decode",
            requested_n_ctx, actual_n_ctx_seq)


class GGUFEmbedder:
    """A dedicated embedding GGUF loaded in embeddings mode via the native llama.dll."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: Optional[int] = None,
                 pooling_type: object = _POOLING_DEFAULT,
                 gpu_split_ratios: Optional[list] = None) -> None:
        from localm.inference.backends.llamacpp import _api as api
        from localm.inference.backends.llamacpp._structs import (
            llama_pos, llama_seq_id, llama_token)
        self._api = api
        self._llama_token = llama_token
        self._llama_pos = llama_pos
        self._llama_seq_id = llama_seq_id
        self.model_path = model_path
        self._lock = threading.RLock()
        # Resolved once the model handle exists, below. None means not yet
        # sized; nothing that reads self.n_ctx runs before __init__ finishes.
        self.n_ctx = n_ctx
        self._model = None
        self._ctx = None
        self._vocab = None
        self._mem = None
        self.dim = 0
        # What the GGUF declares versus what is actually pooled with. Reported
        # up to IsolatedEmbedder through the runner's load meta.
        self.declared_pooling: Optional[int] = None
        self.pooling_type: int = _POOLING_DEFAULT

        if not api.has_embeddings_api():
            raise RuntimeError(
                "this llama.dll build does not expose the embeddings API")
        api.llama_backend_init()
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        if n_gpu_layers >= 99:
            # See llamacpp/llama.py: newer builds have no use_mmap field, and
            # assigning it would silently write into check_tensors instead.
            from localm.inference.backends.llamacpp._structs import set_use_mmap
            set_use_mmap(mp, False)
        # Multi-GPU: honour the configured main_gpu_index / gpu_split_indices.
        # The returned buffer must stay alive through
        # llama_load_model_from_file below. VRAM preflight lives in the PARENT
        # (IsolatedEmbedder, below), not here.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        # gpu_split_ratios: the PARENT's already-resolved effective ratios. This
        # isolated child must not probe for them itself
        # (discover.resolve_auto_split_ratios).
        _tensor_split_keepalive = apply_gpu_split(
            mp, ratios_override=gpu_split_ratios)
        # One contiguous scope over both native calls below, entered once per
        # load. It groups and repeat-collapses llama_load_model_from_file's
        # per-tensor lines and llama_init_from_model's reserve lines, which
        # otherwise go straight to this child's inherited fd 2. It writes to a
        # plain utf-8 file object with errors backslashreplace and touches
        # neither stdout nor stdin, so it cannot corrupt this child's
        # multiprocessing-Queue IPC.
        with dedup_native_stderr():
            self._model = api.llama_load_model_from_file(model_path, mp)
            if not self._model:
                raise RuntimeError(f"failed to load embedding model: {model_path}")
            self._vocab = api.llama_model_get_vocab(self._model)
            self.dim = int(api.llama_model_n_embd(self._model))
            # Read what the model declares BEFORE creating the context; a
            # metadata read needs no context.
            self.declared_pooling = declared_pooling_type(self._model, api)
            self.pooling_type = _effective_pooling(pooling_type, self.declared_pooling)
            logger.debug("embedder %s: declared pooling %s, using %s",
                         Path(model_path).name, pooling_name(self.declared_pooling),
                         pooling_name(self.pooling_type))
            if self.n_ctx is None:
                native_ctx = int(api.llama_model_n_ctx_train(self._model))
                self.n_ctx = _resolve_embed_ctx(native_ctx)
                logger.debug(
                    "embedder %s: native training window %d token(s), using %d",
                    Path(model_path).name, native_ctx, self.n_ctx)
            cp = api.llama_context_default_params()
            # How many texts embed() may pack into one native decode call. See
            # _choose_n_seq_max: the wrong (n_seq_max, n_ubatch) pairing is a
            # hard native crash, not a graceful error.
            self._n_seq_max = _choose_n_seq_max(self.n_ctx)
            configure_embed_context(cp, self.n_ctx, self._n_seq_max,
                                    self.pooling_type)
            self._ctx = api.llama_init_from_model(self._model, cp)
            if not self._ctx:
                api.llama_free_model(self._model)
                self._model = None
                raise RuntimeError("failed to create embedding context")
        self._mem = api.llama_get_memory(self._ctx) if api.has_memory_api() else None
        _warn_if_context_config_drifted(api, self._ctx, self.n_ctx)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize *text*, truncated to fit n_ctx. Returns ``[]`` only for
        the degenerate retokenize-failure case below - a real success always
        yields at least the BERT CLS/SEP special tokens, so an empty list is
        an unambiguous "could not tokenize this at all" signal to callers,
        never a legitimate zero-token text."""
        api = self._api
        raw = (text or " ").encode("utf-8")
        buf = (self._llama_token * self.n_ctx)()
        n = api.llama_tokenize(self._vocab, raw, len(raw), buf, self.n_ctx,
                               True, True)   # add_special (BERT CLS/SEP), parse_special
        if n < 0:
            # Over-long input: llama_tokenize returns -(tokens needed) and
            # writes NOTHING into the short buffer, so the zero-filled buffer
            # must not be read as tokens. Retokenize into a right-sized buffer
            # and truncate explicitly to the context window.
            needed = -n
            full = (self._llama_token * needed)()
            n2 = api.llama_tokenize(self._vocab, raw, len(raw), full, needed,
                                    True, True)
            if n2 <= 0:                      # should not happen; fail visibly
                logger.warning(
                    "embedder: retokenize of an over-long text failed (%d)", n2)
                return []
            # Keep the first n_ctx tokens but preserve the FINAL token of the
            # full sequence: with add_special=True on the BERT-family models
            # this embedder serves, that is the [SEP] the pooled encoding
            # expects.
            for i in range(self.n_ctx):
                buf[i] = full[i]
            buf[self.n_ctx - 1] = full[n2 - 1]
            n = self.n_ctx
            logger.debug(
                "embedder: input of %d tokens truncated to the %d-token window",
                n2, self.n_ctx)
        if n <= 0:
            return []
        return list(buf[:n])

    def _decode_single(self, tokens: List[int]) -> List[float]:
        """One sequence via ``llama_batch_get_one`` - the path a lone text
        takes. A single text never goes through the multi-sequence batching
        below."""
        api = self._api
        if self._mem is not None:
            api.llama_memory_clear(self._mem, True)
        if not tokens:
            return [0.0] * self.dim
        arr = (self._llama_token * len(tokens))(*tokens)
        batch = api.llama_batch_get_one(arr, len(tokens))
        ret = api.llama_decode(self._ctx, batch)
        if ret != 0:
            raise RuntimeError(f"embedding decode failed (code {ret})")
        ptr = api.llama_get_embeddings_seq(self._ctx, 0)
        if not ptr:
            raise RuntimeError("null embedding (pooling produced no output)")
        v = [float(ptr[i]) for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def _decode_batch(self, token_lists: List[List[int]]) -> List[List[float]]:
        """Decode 2+ texts in ONE native ``llama_decode`` call, each as its
        own sequence (`self._n_seq_max` computed at load time - see
        ``_choose_n_seq_max``).

        FAILURE GRANULARITY: a nonzero decode return, OR a null/non-finite
        readout for ANY ONE sequence in the group, raises and discards the
        WHOLE group's result - never a partial list, and never a fabricated
        placeholder vector for the sequence that failed. embed()'s external
        contract is all-or-nothing either way; the failure UNIT here is one
        group rather than one text."""
        api = self._api
        if self._mem is not None:
            api.llama_memory_clear(self._mem, True)
        n_seq = len(token_lists)
        total_tokens = sum(len(t) for t in token_lists)
        batch = api.llama_batch_init(total_tokens, 0, n_seq)
        try:
            token_arr = ctypes.cast(batch.token, ctypes.POINTER(self._llama_token))
            pos_arr = ctypes.cast(batch.pos, ctypes.POINTER(self._llama_pos))
            n_seq_id_arr = ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
            seq_id_arr = ctypes.cast(
                batch.seq_id, ctypes.POINTER(ctypes.POINTER(self._llama_seq_id)))
            logits_arr = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
            idx = 0
            for seq, toks in enumerate(token_lists):
                for pos, tok in enumerate(toks):
                    token_arr[idx] = tok
                    pos_arr[idx] = pos
                    n_seq_id_arr[idx] = 1
                    seq_id_arr[idx][0] = seq
                    # Pooled, non-causal embedding needs output for every token
                    # in the sequence, not just the last one.
                    logits_arr[idx] = 1
                    idx += 1
            batch.n_tokens = total_tokens

            ret = api.llama_decode(self._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"batched embedding decode failed (code {ret})")

            out = []
            for seq in range(n_seq):
                ptr = api.llama_get_embeddings_seq(self._ctx, seq)
                if not ptr:
                    raise RuntimeError(
                        f"null embedding for sequence {seq} of {n_seq} in a "
                        "batched decode (pooling produced no output)")
                v = [float(ptr[i]) for i in range(self.dim)]
                norm = math.sqrt(sum(x * x for x in v))
                if not math.isfinite(norm):
                    raise RuntimeError(
                        f"non-finite embedding for sequence {seq} of {n_seq} "
                        "in a batched decode")
                out.append([x / norm for x in v] if norm else v)
            return out
        finally:
            api.llama_batch_free(batch)

    def _pack_groups(self, token_lists: List[List[int]]) -> List[List[int]]:
        """Group token-list INDICES for one embed() call into batches of up
        to ``self._n_seq_max`` texts, never letting a group's SUMMED token
        count exceed ``self.n_ctx`` (== n_ubatch here) - llama.cpp's own hard
        constraint for this non-causal 'encoder' architecture ("encoder
        requires n_ubatch >= n_tokens"). A single text is already truncated to
        fit n_ctx by _tokenize, so it always fits alone; a group of multiple
        texts is shrunk automatically when their combined length would
        overflow."""
        groups: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        for i, toks in enumerate(token_lists):
            n = len(toks)
            if current and (len(current) >= self._n_seq_max
                            or current_tokens + n > self.n_ctx):
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(i)
            current_tokens += n
        if current:
            groups.append(current)
        return groups

    def embed(self, texts: List[str]) -> List[List[float]]:
        """L2-normalised embedding per text (aligned 1:1 with *texts*).

        Multiple texts sharing one call are packed into groups of up to
        ``self._n_seq_max`` and decoded with ONE native ``llama_decode`` per
        group. A lone text uses the single-sequence path.

        Every ``llama_decode`` call here, single or batched, writes native
        lines like ``decode: cannot decode batches with this context
        (calling encode() instead)`` to raw stderr, and is NOT wrapped in
        ``dedup_native_stderr()`` at this level. The collapsing scope spans
        the isolated child's whole run of "embed" commands instead - see
        ``_embedder_runner.py``'s dispatch loop."""
        with self._lock:
            if self._ctx is None:
                raise RuntimeError("embedder is closed")
            if not texts:
                return []
            token_lists = [self._tokenize(t) for t in texts]
            out: List[Optional[List[float]]] = [None] * len(texts)
            # A text that failed to tokenize needs no native call: it is
            # short-circuited to a zero vector and kept out of the packing
            # below.
            pending_idx: List[int] = []
            pending_toks: List[List[int]] = []
            for i, toks in enumerate(token_lists):
                if toks:
                    pending_idx.append(i)
                    pending_toks.append(toks)
                else:
                    out[i] = [0.0] * self.dim
            groups = self._pack_groups(pending_toks) if pending_toks else []
            for group in groups:
                group_idx = [pending_idx[j] for j in group]
                group_toks = [pending_toks[j] for j in group]
                vecs = ([self._decode_single(group_toks[0])] if len(group_toks) == 1
                        else self._decode_batch(group_toks))
                for gi, v in zip(group_idx, vecs):
                    out[gi] = v
            return out

    def close(self) -> None:
        with self._lock:
            api = self._api
            if self._ctx is not None:
                try:
                    api.llama_free(self._ctx)
                finally:
                    self._ctx = None
            if self._model is not None:
                try:
                    api.llama_free_model(self._model)
                finally:
                    self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class IsolatedEmbedder(VramSizingMixin):
    """Parent-side handle to a ``GGUFEmbedder`` running inside an isolated
    child process (see ``_embedder_runner.py``), containing native aborts on
    the embedder's GGUF/llama.cpp load path.

    Preflight VRAM sizing (inherited from VramSizingMixin, shared with
    GgufBackend) runs HERE, before a child is even spawned, so a load that can
    never fit fails fast without paying a process-spawn cost."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: Optional[int] = None,
                 pooling_type: object = _POOLING_DEFAULT,
                 gpu_fallback_reason: Optional[str] = None) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.effective_gpu_layers = None    # no auto-sizing for the embedder
        # None means auto-size to the model's own native training window (see
        # GGUFEmbedder.__init__ / _EMBED_CTX_CEILING), which only the child can
        # determine. self.n_ctx starts at the ceiling, which the child's
        # auto-sized value can never exceed, so the parent's VRAM preflight has
        # a concrete number before any child spawns; _reload overwrites it with
        # the figure the child reports at load. The ORIGINAL request (None for
        # auto, or an explicit override) is kept separately so _reload forwards
        # it rather than this placeholder.
        self._requested_n_ctx = n_ctx
        self.n_ctx = n_ctx if n_ctx is not None else _EMBED_CTX_CEILING
        self._pooling_type = pooling_type
        self.dim = 0
        # Reported by the child at load (see _reload): what the GGUF declares and
        # what is actually pooled with.
        self.declared_pooling: Optional[int] = None
        self.effective_pooling: Optional[int] = None
        # Set once a GPU-offloaded embed() crashes the worker and this embedder
        # falls back to CPU (see embed()'s crash-recovery branch), or seeded at
        # construction when automatic placement already chose CPU
        # (get_embedder -> _choose_embedder_gpu_layers). None while on GPU.
        # When set, _reload() passes cpu_only to the child from the first spawn.
        self.gpu_fallback_reason: Optional[str] = gpu_fallback_reason
        self._runner = None
        # Serializes embed() below. The worker protocol has NO request-id
        # correlation (one req_q/resp_q pair, one command at a time), so two
        # overlapping RPCs are two threads blocked in the same resp_q.get(),
        # each taking whichever response arrives first, which can hand a caller
        # vectors belonging to a DIFFERENT text.
        self._rpc_lock = threading.RLock()
        # In-flight embed() calls, mirroring Engine.active_requests. Checked by
        # http_server.py's unload/shutdown/restart paths (via active_requests()
        # below) before releasing this embedder. Plain int, incremented and
        # decremented lock-free in embed(): best-effort precision.
        self.active_requests = 0
        self._reload()

    def _preflight_vram(self) -> None:
        """Refuse a load that cannot fit BEFORE spawning a child.

        A multi-GPU split gets its own per-device check:
        VramSizingMixin._check_vram() budgets the split's COMBINED capacity
        (see _split_free_total_bytes), so one device's proportional share can
        be individually short while the aggregate passes. The single-GPU case
        uses _check_vram()."""
        from localm.config import load_config
        from localm.discover import applied_split_device_count, gpu_split_shortfall
        cfg = load_config()
        # applied_split_device_count (loader truth), NOT split_device_count: on
        # the vulkan build list_gpus() is blind to the real split devices, so
        # the DETECTED count collapses a live 2-way split to < 2. On vulkan the
        # gpu_split_shortfall() call below self-skips the per-device check and
        # logs the skip at INFO.
        if applied_split_device_count(cfg) >= 2:
            try:
                file_size = Path(self.model_path).stat().st_size
            except OSError:
                file_size = 0
            if file_size:
                shortfall = gpu_split_shortfall(int(file_size * 1.2), cfg)
                if shortfall:
                    detail = "; ".join(
                        f"GPU {d['index']} needs ~{d['needed'] // 1024 ** 2} MB, "
                        f"{d['free'] // 1024 ** 2} MB free" for d in shortfall)
                    raise RuntimeError(
                        f"Not enough VRAM on the configured split device(s) "
                        f"to load the embedding model ({detail}).")
        else:
            # The single-GPU case: the same preflight GgufBackend.load() runs
            # before every chat-model load.
            self._check_vram()

    def _reload(self) -> None:
        """(Re)run preflight and spawn a fresh child that loads the model.
        Used both by __init__ and by embed()'s auto-respawn after a crash;
        preflight re-runs each time because VRAM may have changed.

        cpu_only is passed on EVERY reload once gpu_fallback_reason is set:
        n_gpu_layers=0 alone does not guarantee no GPU backend involvement
        (see _embedder_runner.py's cpu_only handling)."""
        self._preflight_vram()
        from ._embedder_runner import EmbedderRunner
        # Same parent-pins-worker-consumes contract as GgufBackend._load_native:
        # the effective split distribution (auto free-VRAM-proportional when
        # gpu_split_ratios is unset) is resolved HERE and carried into the
        # child, which must not probe for it. Skipped when this embedder is
        # CPU-bound anyway (cpu_only, or zero GPU layers).
        from localm.discover import resolve_auto_split_ratios
        cpu_only = self.gpu_fallback_reason is not None
        auto_ratios = None
        if not cpu_only and self.n_gpu_layers != 0:
            # wait_for_inflight: loads run off the event loop, so a
            # heartbeat-probe collision joins instead of declining auto into
            # the equal fallback.
            auto_ratios = resolve_auto_split_ratios(wait_for_inflight=True)
        params = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                      n_ctx=self._requested_n_ctx, pooling_type=self._pooling_type,
                      cpu_only=cpu_only, gpu_split_ratios=auto_ratios)
        self._runner = EmbedderRunner()
        meta = self._runner.spawn_and_load(params)
        self.dim = meta["dim"]
        self.declared_pooling = meta.get("declared_pooling")
        self.effective_pooling = meta.get("effective_pooling")
        # Overwrites the preflight-time ceiling placeholder set in __init__:
        # only the child, which loads the model, can resolve auto against the
        # model's real native window.
        self.n_ctx = meta["n_ctx"]
        self._warn_if_mispooled()

    def _warn_if_mispooled(self) -> None:
        """Warn when this model is being pooled against its own training.

        A model declaring LAST-token pooling is a DECODER-based embedder whose
        embeddings are read off the final token; pooling it any other way
        degrades them while still returning healthy, normalised, plausible
        vectors.

        Fires only when embedding_pooling has been set EXPLICITLY to something
        other than last (mean/cls/none) for such a model - the untouched
        default already resolves LAST for it (see _effective_pooling). No other
        declaration warns.
        """
        declared, effective = self.declared_pooling, self.effective_pooling
        if declared != _POOLING_LAST or effective == _POOLING_LAST:
            return
        logger.warning(
            "embedding model %s declares %s-token pooling (it is a decoder-based "
            "embedder) but embedding_pooling is explicitly set to %s, which "
            "degrades its embeddings. Unset embedding_pooling (or set it to "
            "'last'/'auto') to use the model's own pooling; existing RAG "
            "collections and memory vectors were built with %s and need "
            "re-indexing after the change.",
            Path(self.model_path).name, pooling_name(declared),
            pooling_name(effective), pooling_name(effective))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via the isolated worker, transparently respawning it
        first if a PRIOR call's crash left it dead. A crash DURING this call is
        raised to the caller, never swallowed; only the NEXT call recovers
        automatically, EXCEPT for the GPU-crash-once case below, which falls
        back to CPU and retries inline.

        Pinned via ``active_requests`` for the whole call, including a respawn,
        so a ``reset_embedder()`` arriving mid-respawn cannot free the runner
        this call is about to use.

        Serialized on ``_rpc_lock`` (see __init__): the respawn check and the
        RPC together, so concurrent callers can neither swap responses on the
        correlation-free queue pair nor both respawn and orphan a worker. The
        pin is taken BEFORE the lock, so a caller merely queued behind another
        still counts as in-flight."""
        self.active_requests += 1
        try:
            with self._rpc_lock:
                if self._runner is None or not self._runner.is_alive():
                    logger.warning("embedder worker is not running; reloading %s",
                                   self.model_path)
                    self._reload()
                try:
                    return self._runner.embed(list(texts))
                except RuntimeError:
                    # Discard the worker ONLY if it is actually gone. The child's
                    # dispatch loop answers an ordinary embed failure with a
                    # clean (error, msg) envelope and keeps serving, and the
                    # parent sees the SAME RuntimeError for that as for a crash.
                    # Dropping the reference unconditionally would orphan a LIVE
                    # child that no close(), reset_embedder() or
                    # release_for_exit() can reach.
                    if self._runner is not None and self._runner.is_alive():
                        logger.exception(
                            "embedding failed; the worker is healthy and will "
                            "serve the next call")
                        raise
                    logger.exception(
                        "embedding worker fault; it will reload on the next call")
                    runner, self._runner = self._runner, None
                    if runner is not None:
                        # Release its queues/handles. Safe and idempotent when
                        # the child is already dead, and _wait's own timeout path
                        # has already done it.
                        runner.shutdown(grace=0)
                    # A crash while GPU-offloaded falls back to CPU ONCE per
                    # embedder instance, retries THIS call inline, and warns.
                    # In-memory only: the user's n_gpu_layers setting still
                    # governs the chat model.
                    if self.n_gpu_layers > 0 and self.gpu_fallback_reason is None:
                        self.gpu_fallback_reason = (
                            "GPU-offloaded embedding crashed natively (worker "
                            "exit); retrying on CPU. Chat-model GPU settings "
                            "are unaffected. Run 'localm setup-llama --force' "
                            "to reprovision the GPU runtime and restore "
                            "GPU-accelerated embedding.")
                        logger.warning("embedder %s: %s",
                                       Path(self.model_path).name,
                                       self.gpu_fallback_reason)
                        self.n_gpu_layers = 0
                        self._reload()
                        return self._runner.embed(list(texts))
                    raise
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
            self._runner = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Process-wide singleton                                                      #
# --------------------------------------------------------------------------- #

# LOCK ORDER: engine._LOAD_LOCK (outer) -> _LOCK (inner), never the reverse.
# The engine's load path holds _LOAD_LOCK for a whole model load and calls
# loaded_path() here from its ctx sizing, so acquiring _LOAD_LOCK - directly or
# by starting a load - while holding _LOCK is an ABBA deadlock against every
# concurrent model load. get_embedder is the one site that needs both locks, and
# takes them in the legal order.
#
# HOLD TIME: microseconds normally, up to 300s during a load. get_embedder holds
# it across an entire IsolatedEmbedder construction (process spawn plus native
# model load, _embedder_runner.LOAD_TIMEOUT_DEFAULT). So the readers below -
# loaded_dim, loaded_path, last_error, gpu_fallback_reason - are cheap in WORK
# and unbounded in WAITING. NONE OF THEM MAY BE CALLED FROM AN `async def` ROUTE
# HANDLER: on the event loop that wait blocks every client and every route.
_LOCK = threading.RLock()

# Two budgets for offloading one of those readers, e.g.
# run_in_threadpool_bounded(loaded_dim, timeout=PEEK_TIMEOUT_S). _LOCK is either
# free (the read costs microseconds) or held by a load (it costs minutes), with
# nothing in between, so a budget expiring reports that a load is running.
#
#   PEEK   for a caller that can act on that fact. POST /api/embedding/warmup
#          reads loaded_dim only to skip work it is about to do anyway, so a
#          timeout means not-already-warm.
#   READ   for a caller whose ANSWER is the value, where waiting is correct.
#          Sized past a legitimate cold load
#          (_embedder_runner.LOAD_TIMEOUT_DEFAULT, 300s) plus the queue behind
#          engine._LOAD_LOCK it may sit in first, so it fires only for a
#          genuinely wedged lock.
PEEK_TIMEOUT_S = 2.0
READ_TIMEOUT_S = 660.0
_EMBEDDER: Optional[IsolatedEmbedder] = None
# A model that is present on disk but fails to LOAD (corrupt or OOM) is cached as
# failed, so the expensive load is not retried on every call. A MISSING model is
# NOT cached: `localm setup-embeddings` can install one into a running server, and
# get_embedder re-checks the filesystem each call so it is picked up without a
# restart.
#
# Holds the embedding_model config value (stripped) that was current when a LOAD
# last failed, or None when nothing is latched. A changed spec is told apart
# from the same broken one: see get_embedder for how a changed spec clears the
# stale latch, and _set_resolve_outcome for the matching resolve-side guard.
_LOAD_FAILED_SPEC: Optional[str] = None
_TRIED_DOWNLOAD = False          # one-time auto-download attempt (only net_mode=allow)
# Why the last LOAD or RESOLVE failed, for the GUI picker. See last_error() and
# _record_resolve_failure/_record_resolve_success above.
_LAST_ERROR: Optional[str] = None
# Dedup key for _record_resolve_failure's WARNING-once-then-DEBUG: the failure
# text last logged at WARNING, so an unchanged misconfiguration does not re-warn
# on every embed_texts() call.
_LAST_RESOLVE_WARNED: Optional[str] = None


def _explicit_embedder_gpu_layers(cfg: dict) -> Optional[int]:
    """The user's EXPLICIT GPU-layer choice for the embedder, or None when the
    automatic placement below should decide.

    ``embedding_gpu_layers`` (the dedicated key) wins outright. Failing that,
    a global ``n_gpu_layers`` moved off its "everything" default (99) is
    inherited. A bool sneaking in via a hand-edited config is not an int
    choice; malformed values fall through to auto rather than raising."""
    raw = cfg.get("embedding_gpu_layers")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    try:
        chat = int(cfg.get("n_gpu_layers", 99))
    except (TypeError, ValueError):
        return None
    if chat != 99:
        return chat
    return None


def _choose_embedder_gpu_layers(path: str, cfg: dict, *,
                                read_free=None) -> "tuple[int, Optional[str]]":
    """Pick the embedder's ``n_gpu_layers``: ``(layers, reason)`` where
    *reason* is a user-facing explanation only when automatic placement chose
    CPU over a GPU that cannot hold the model.

    An explicit user choice (see _explicit_embedder_gpu_layers) is honored
    verbatim, tight VRAM or not. Unmeasurable free VRAM, or an unreadable model
    file, keeps the full-offload attempt.

    read_free()'s default checks FRESHNESS, not scope: a GPU_PROBE_TIMEOUT/BUSY
    reading is a frozen last-known-good value rather than this call's own
    measurement, and is treated as unmeasurable. It does not gate on
    free_scope - a PROCESS-scoped reading over-states free, so it can only ever
    make the insufficient-VRAM branch below under-fire, never fire wrongly."""
    explicit = _explicit_embedder_gpu_layers(cfg)
    if explicit is not None:
        return explicit, None
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    if not size:
        return 99, None
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import GPU_PROBE_OK, vram_capacity
            info, status = vram_capacity(return_status=True)
            return info.get("free") if status == GPU_PROBE_OK else None
    try:
        free = read_free()
    except Exception:
        free = None
    if free is None:
        return 99, None
    # Same loaded-footprint approximation the swap decision and the split
    # preflight already use (weights + ~20% KV/compute slop).
    estimate = int(size * 1.2)
    if free >= estimate:
        return 99, None
    reason = (f"insufficient free VRAM for the embedding model "
              f"(~{estimate // 1024 ** 2} MB needed, "
              f"{int(free) // 1024 ** 2} MB free); the embedder runs on CPU "
              f"so the loaded chat model is not slowed by VRAM "
              f"oversubscription - set embedding_gpu_layers to override")
    return 0, reason


def _maybe_swap_for_embedder(path: str, n_gpu_layers: int) -> None:
    """Before the embedder's native load, evict a resident chat model when it
    would not otherwise fit - the SAME VRAM-aware swap the image/music/video
    plugins run before their own model load (see ``localm.vram.decide_media_swap``),
    generalized here via ``decide_embedder_swap``/``evict_chat_for_embedder``.

    Distinct from IsolatedEmbedder's own ``_preflight_vram()`` (below), which
    only REFUSES a load that will not fit and never makes room. This function
    is what frees VRAM, by evicting the resident chat model first.

    A CPU-only embedder load (``n_gpu_layers <= 0``) never contends for VRAM,
    so it is skipped. Best-effort: any failure to read the file size or decide
    the swap returns without blocking the load."""
    if n_gpu_layers <= 0:
        return
    try:
        file_size = Path(path).stat().st_size
    except OSError:
        return
    if not file_size:
        return
    from localm.config import load_config
    from localm.vram import decide_embedder_swap, evict_chat_for_embedder, resolve_swap_policy
    cfg = load_config()
    policy = resolve_swap_policy({}, cfg)
    # Same loaded-footprint approximation IsolatedEmbedder._preflight_vram()
    # already uses (weights plus about 20% KV-cache and compute slop).
    estimate = int(file_size * 1.2)
    if not decide_embedder_swap(estimate, policy=policy):
        return
    evict_chat_for_embedder()


def _emit_stage(on_progress: Optional[Callable[[str], None]], msg: str) -> None:
    """Best-effort coarse stage announcement for get_embedder()'s caller. A
    raising sink is swallowed and never aborts a load. Parent-side only: this
    never touches the isolated embedder child's own load/embed IPC protocol."""
    if on_progress is None:
        return
    try:
        on_progress(msg)
    except Exception:
        logger.debug("embedder progress sink raised (ignored)", exc_info=True)


def get_embedder(*, on_progress: Optional[Callable[[str], None]] = None
                  ) -> Optional[IsolatedEmbedder]:
    """The shared embedder, loading the configured model on first use. Returns None
    when no embedding model can be resolved - callers then fall back to lexical
    retrieval. A missing model is re-checked on every call (so a mid-session
    ``localm setup-embeddings`` is picked up without a restart); only a genuine
    load FAILURE is cached. Loading holds the engine's process-global load lock so
    it cannot race a chat-model load onto the GPU.

    *on_progress*, if given, receives coarse human-readable stage announcements
    (resolving/downloading/evicting/loading/ready). A cold first load can span
    two 300s timeout windows (VRAM eviction wait, then child spawn and native
    load).

    The pre-load swap check (``_maybe_swap_for_embedder``) MUST run OUTSIDE
    ``_LOCK`` (double-checked locking below). It can evict a resident chat
    model via ``vram.evict_chat_for_embedder``, which submits
    ``http_server.unload_all_models()`` onto the SERVER'S event-loop thread and
    blocks THIS thread waiting for it, and that coroutine calls
    ``loaded_dim()``, which needs ``_LOCK``. ``_LOCK`` is a plain
    ``threading.RLock`` and is not reentrant across threads, so holding it
    across that call is a cross-thread deadlock."""
    global _EMBEDDER, _LOAD_FAILED_SPEC, _TRIED_DOWNLOAD, _LAST_ERROR
    cur_spec = None   # computed lazily, only once _EMBEDDER is confirmed unset;
                       # the "already loaded" path does no I/O.
    with _LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _LOAD_FAILED_SPEC is not None:
            # The identity a latched load failure is compared against. Reused
            # below, not re-derived, for the second latch test.
            cur_spec = _current_spec()
            if _LOAD_FAILED_SPEC == cur_spec:
                return None
            # The config has moved on since this was latched, so it no longer
            # blocks the load.
            _LOAD_FAILED_SPEC = None
            _LAST_ERROR = None
        _emit_stage(on_progress, "Resolving the embedding model...")
        # Cheap filesystem-only re-check every call (NO download): finds a model a
        # user just installed into this running server.
        path = resolve_embedding_model_path(allow_download=False)
        if not path and not _TRIED_DOWNLOAD:
            # First miss: one auto-download attempt, gated by net policy inside
            # (it only fetches under net_mode=allow). Latched so a batch of embed
            # calls does not re-attempt the download on every chunk.
            _TRIED_DOWNLOAD = True
            _emit_stage(on_progress, "Not found locally - attempting a download...")
            path = resolve_embedding_model_path()
        if not path:
            _emit_stage(on_progress, "No embedding model is configured or available.")
            return None
        from localm.config import load_config
        _cfg = load_config()
        pooling = resolve_pooling_setting(_cfg.get("embedding_pooling"))
        # GPU INTENT for the pre-load eviction check below: an explicit user
        # choice, else full offload. The FINAL placement is chosen after the
        # swap runs, against post-eviction free VRAM (see
        # _choose_embedder_gpu_layers).
        explicit = _explicit_embedder_gpu_layers(_cfg)
        intent_ngl = explicit if explicit is not None else 99

    # OUTSIDE _LOCK: this blocks on the cross-thread eviction round trip.
    # Another thread may reach the same window and run its own swap check;
    # evict_chat_for_embedder() and decide_embedder_swap() are idempotent, and
    # the load below re-checks the singleton state.
    _emit_stage(on_progress, "Checking VRAM (may evict a resident chat model)...")
    _maybe_swap_for_embedder(path, intent_ngl)
    ngl, placement_reason = _choose_embedder_gpu_layers(path, _cfg)
    if placement_reason is not None:
        logger.warning("embedder placement: %s", placement_reason)

    # LOCK ORDER: the engine's process-global load lock is acquired FIRST,
    # strictly OUTSIDE _LOCK. Engine.load() holds _LOAD_LOCK across an entire
    # model load, and its ctx sizing calls loaded_path(), which takes _LOCK, via
    # _sizing.embedder_ctx_reservation_bytes. The only permitted nesting is
    # _LOAD_LOCK (outer) -> _LOCK (inner); see both lock definitions. Waiting
    # here for a running chat load therefore does not hold _LOCK.
    from localm.inference.engine import _LOAD_LOCK
    with _LOAD_LOCK:
        with _LOCK:
            # Re-check: another thread may have completed or failed the load
            # while this thread was outside the lock running the swap check,
            # including latching a NEW failure for whatever spec is current now,
            # which is re-derived here rather than read from the local above.
            if _EMBEDDER is not None:
                return _EMBEDDER
            if _LOAD_FAILED_SPEC is not None:
                if cur_spec is None:
                    cur_spec = _current_spec()
                if _LOAD_FAILED_SPEC == cur_spec:
                    return None
                _LOAD_FAILED_SPEC = None
                _LAST_ERROR = None
            try:
                # The stated bound is the deadline this stage runs under:
                # _embedder_runner.LOAD_TIMEOUT_DEFAULT (300s), since _reload's
                # spawn_and_load() passes no override.
                _emit_stage(on_progress,
                           "Loading into memory (this can take up to five minutes)...")
                _EMBEDDER = IsolatedEmbedder(
                    path, n_gpu_layers=ngl, pooling_type=pooling,
                    gpu_fallback_reason=placement_reason)
                # getattr, not attribute access: this line runs inside the try,
                # so anything it raised would be caught as a LOAD failure.
                logger.info("embedding model ready: %s (dim=%d, pooling=%s)", path,
                            _EMBEDDER.dim,
                            pooling_name(getattr(_EMBEDDER, "effective_pooling", None)))
                _LAST_ERROR = None
                _emit_stage(on_progress, f"Ready ({_EMBEDDER.dim}-dim).")
                return _EMBEDDER
            except Exception as e:
                # Latch by SPEC: what just failed to load is whatever
                # embedding_model names right now, and a later call compares
                # against that identity.
                _LOAD_FAILED_SPEC = cur_spec if cur_spec is not None else _current_spec()
                _LAST_ERROR = str(e)
                logger.warning("could not load embedding model %s (%s); lexical-only",
                               path, e)
                _emit_stage(on_progress, f"Load failed: {e}")
                return None


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed *texts* with the shared embedder, or None when unavailable."""
    emb = get_embedder()
    if emb is None:
        return None
    return emb.embed(list(texts))


def loaded_dim() -> Optional[int]:
    """Dimension of the currently-loaded embedder, or None if none is loaded.
    Does NOT trigger a load - safe for a cheap status probe (GUI picker)."""
    with _LOCK:
        return _EMBEDDER.dim if _EMBEDDER is not None else None


def gpu_fallback_reason() -> Optional[str]:
    """Why the currently-loaded embedder dropped from GPU to CPU after a native
    crash (see IsolatedEmbedder.embed), or None when it hasn't (still on GPU,
    or configured for CPU from the start). Does NOT trigger a load.

    Path-scrubbed on the way out, like ``last_error`` below."""
    with _LOCK:
        raw = _EMBEDDER.gpu_fallback_reason if _EMBEDDER is not None else None
    return pathscrub.scrub_paths(raw) if raw else raw


def loaded_path() -> Optional[str]:
    """Filesystem path of the currently-loaded embedder, or None if none is
    loaded. Does NOT trigger a load. Lets a caller outside this module (the
    Models page, the targeted unload route) recognise a registered model entry
    as the resident embedder by comparing against the entry's own resolved
    path; the embedder never appears in ``http_server._engines``. Pairs with
    ``loaded_dim()``."""
    with _LOCK:
        return _EMBEDDER.model_path if _EMBEDDER is not None else None


def active_requests() -> int:
    """In-flight embed() calls on the currently-loaded embedder, or 0 if none is
    loaded. Mirrors ``Engine.active_requests`` - checked by http_server.py's
    unload-all/targeted-unload/shutdown/restart paths before releasing this
    embedder, the same way a pinned chat Engine is skipped by
    unload_all_models/unload_one_model."""
    with _LOCK:
        return _EMBEDDER.active_requests if _EMBEDDER is not None else 0


def last_error() -> Optional[str]:
    """Why the last embedding-model LOAD or RESOLVE failed (e.g. the model is not
    an embedding model, or the configured spec resolves to a HuggingFace-format
    directory rather than a GGUF file), or None. resolve_embedding_model_path
    records a failure here on every call, not only when a load is actually
    attempted (see _record_resolve_failure), so GET /api/rag/embedding can
    explain a broken embedding_model without anyone triggering an embed first.

    PATH-SCRUBBED on the way out. The stored message is the raw exception text,
    which for a load failure carries an absolute path, and every caller of this
    is an HTTP response (``GET /api/rag/embedding``, the 422 in routes/chat.py,
    the rag job log) readable by a rag-scoped, non-owner client.

    Scrubbed on READ, not at the assignment, so the WARNING logged at the
    failure site keeps the full path. Only the directory prefix is replaced;
    the CAUSE is preserved verbatim."""
    with _LOCK:
        raw = _LAST_ERROR
    return pathscrub.scrub_paths(raw) if raw else raw


def reset_embedder(*, force: bool = True) -> bool:
    """Drop the cached embedder and its negative caches (tests / a model
    change). Returns True if an embedder was actually cleared, False if none
    was loaded or (``force=False``) one was loaded but pinned.

    ``force=False`` checks ``active_requests() == 0`` and clears the embedder
    in the SAME ``_LOCK`` acquisition, atomically, with no ``await`` between
    the read and the close. That check must not be made separately by the
    caller: ``IsolatedEmbedder.embed()`` pins ``active_requests`` without
    taking ``_LOCK``, so a concurrent embed() could start between an unlocked
    check and the ``close()`` below and have its worker torn out from under it.

    ``force=True`` (the default) clears unconditionally, for callers that must
    tear down regardless of a pin (e.g. an explicit model-selection change).
    ``release_for_exit()`` does NOT go through this function at all - see its
    own docstring.

    A pinned (``force=False``, busy) embedder is a full no-op, including the
    negative caches. Otherwise ``_LOAD_FAILED_SPEC``/``_TRIED_DOWNLOAD``/
    ``_LAST_ERROR``/``_LAST_RESOLVE_WARNED`` are always cleared alongside
    ``_EMBEDDER``, even when no embedder was loaded at all and only a cached
    load FAILURE was latched, so the next ``get_embedder()`` retries fresh."""
    global _EMBEDDER, _LOAD_FAILED_SPEC, _TRIED_DOWNLOAD, _LAST_ERROR, _LAST_RESOLVE_WARNED
    with _LOCK:
        if not force and _EMBEDDER is not None and _EMBEDDER.active_requests > 0:
            return False
        cleared = _EMBEDDER is not None
        if cleared:
            _EMBEDDER.close()
        _EMBEDDER = None
        _LOAD_FAILED_SPEC = None
        _TRIED_DOWNLOAD = False
        _LAST_ERROR = None
        _LAST_RESOLVE_WARNED = None
        return cleared


def release_for_exit() -> bool:
    """Release the isolated embedder worker for a caller that is about to
    ``os._exit()`` / ``os.execv()``. Returns True if a worker was released.

    Both of those bypass ``atexit``, and multiprocessing's daemon-child
    reclamation IS an atexit hook (``multiprocessing.util._exit_function``), so
    a worker left running does NOT die with the parent: it survives as an
    orphan still holding its model in VRAM.

    Nothing here takes ``_LOCK``, and nothing here may: ``get_embedder()``
    holds ``_LOCK`` for the full duration of an embedding-model load, so a stop
    or restart issued during a load would block on the guard itself. The
    singleton is snapshotted directly, a best-effort read.

    A BUSY worker (a request is mid-``embed()``) is terminated WITHOUT waiting;
    the polite "shutdown" command would queue behind the in-flight embed and
    stall the exit for the grace period. An IDLE worker gets the polite command
    first, so the child frees the model cleanly before it goes.

    The module singleton is NOT cleared - that is reset_embedder()'s job, and
    it needs ``_LOCK``."""
    emb = _EMBEDDER
    if emb is None:
        return False
    runner = getattr(emb, "_runner", None)
    if runner is None:
        return False
    busy = getattr(emb, "active_requests", 0) > 0
    runner.shutdown(grace=0 if busy else 5.0)
    return True


# Torn down explicitly at interpreter exit so the worker's queue feeder threads
# never delay shutdown. The embedder outlives any single request and is not torn
# down by an owning caller on every unload.
atexit.register(reset_embedder)
