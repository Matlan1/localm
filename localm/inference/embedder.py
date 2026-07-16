# SPDX-License-Identifier: AGPL-3.0-or-later
"""
On-device text embeddings via a DEDICATED embedding GGUF, on the bundled runtime.

This module loads a small, dedicated embedding model (bge / nomic, ~25-90 MB) into
its OWN llama.cpp model + context in EMBEDDINGS mode, independent of whatever chat
model is loaded, so semantic retrieval (RAG hybrid search, agent memory) works on
the default GGUF runtime with consistent quality and without disturbing the chat
model.

Why a dedicated model rather than the chat model, in order of how binding each
reason is:

1. CAPABILITY: the bundled GGUF chat backend cannot embed at all - the ctypes
   binding exposes no create_embedding (``backends/gguf.py``, ``can_embed=False``).
2. COST: loading a multi-GB chat model purely to embed would be wasteful, and
   under VRAM pressure would evict the resident one.
3. QUALITY: a chat model is a decoder LLM trained to predict the next token, not
   to place related texts near each other, so its pooled hidden states make poor
   embeddings. Measured 2026-07-15 through this very module, CPU-only, against
   memory's REL_COS_MIN=0.55 gate: bge-small separated related from unrelated
   pairs by +0.29 (min related 0.6498 vs max unrelated 0.3585, 4/4 gate), while
   Qwen2.5-0.5B's max UNRELATED cosine (0.7523) EXCEEDED its min RELATED cosine
   (0.7518) - NO threshold splits them. That is decoder anisotropy (no contrastive
   objective), not a pooling artifact: LAST-token pooling was measured too and
   scored worse (-0.17). The failure is silent - the vectors are non-zero,
   normalised and plausible - which is precisely why this path is not optional.

It is a process-wide, lazily-loaded singleton (``get_embedder`` / ``embed_texts``):
one small model is loaded once and shared by every caller (chat memory, coder
memory, /v1/embeddings). Loading serialises on the engine's process-global load lock
so it never races a chat-model load onto the GPU.

Provisioning (``resolve_embedding_model_path``): the ``embedding_model`` config key
is either a filesystem path, a registered model name, or a known key (default
``bge-small-en-v1.5``). A known model missing from ``<home>/models/embeddings/`` is
downloaded on demand, gated by the network policy (never behind ``net_mode=off``;
auto only under ``net_mode=allow`` - otherwise the user runs ``localm setup-embeddings``).
When no embedding model can be resolved, callers degrade to lexical-only retrieval
(surfaced via a debug log, never a silent success).

The native load (and every ``embed()`` call - it hits ``llama_decode``, the
same abort-prone native call class) runs inside an ISOLATED CHILD PROCESS
(``_embedder_runner.py``), not in this process - mirroring
``backends/gguf.py``'s containment for the identical native-abort risk (PR
#606: a native CUDA/HIP driver failure can hard-``abort()`` the whole process
in C, uncatchable from Python). ``GGUFEmbedder`` below is the raw, unguarded
native loader (constructed only inside that child); ``IsolatedEmbedder`` is
the parent-side handle ``get_embedder()`` actually returns.
"""

from __future__ import annotations

import atexit
import math
import threading
from pathlib import Path
from typing import List, Optional

from localm.debuglog import logger
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

# Known small embedding models, keyed by a friendly name -> (hf_repo, filename).
# Chosen for size + quality: bge-small is 24 MB Q4 and scores well; nomic is the
# common alternative. A user can also point embedding_model at any GGUF path or a
# registered model name.
KNOWN_EMBEDDING_MODELS = {
    "bge-small-en-v1.5": (
        "CompendiumLabs/bge-small-en-v1.5-gguf", "bge-small-en-v1.5-q4_k_m.gguf"),
    "nomic-embed-text-v1.5": (
        "nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
}
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"

# llama.cpp LLAMA_POOLING_TYPE_* values. UNSPECIFIED (-1) is llama.cpp's own
# default and means "the model's declared pooling decides" (see llamacpp/_abi.py,
# which treats that -1 as an ABI keystone).
_POOLING_UNSPECIFIED = -1
_POOLING_NONE = 0
_POOLING_MEAN = 1
_POOLING_CLS = 2
_POOLING_LAST = 3

# Pooling settings a user may choose (config ``embedding_pooling``). "auto" is not
# a llama.cpp value: it is resolved per-model against what the GGUF declares.
POOLING_AUTO = "auto"
_POOLING_BY_NAME = {"none": _POOLING_NONE, "mean": _POOLING_MEAN,
                    "cls": _POOLING_CLS, "last": _POOLING_LAST}
_POOLING_NAMES = {v: k for k, v in _POOLING_BY_NAME.items()}
POOLING_CHOICES = [POOLING_AUTO, *_POOLING_BY_NAME]

# Why MEAN is the DEFAULT rather than "whatever the model declares" (measured
# 2026-07-15 against the real GGUFs, via the bundled llama.dll):
#
#   model                 declares                 forced MEAN today
#   bge-small (default)   bert.pooling_type=2 CLS  works: +0.29 related/unrelated margin
#   nomic (known key)     nomic-bert...=1 MEAN     already exactly right
#   gte-Qwen2-1.5B        (no pooling key)         MEAN is the RESCUE (-1 -> NONE -> no output)
#   Qwen3-Embedding-0.6B  qwen3.pooling_type=3 LAST  MIS-POOLED - the real defect
#
# So MEAN is right for three of the four and is what every existing index was
# built with. Switching the default to the declared type would silently flip
# bge from MEAN to CLS at the SAME 384 dims - no dim guard fires (rag/store.py's
# mixed-dim check, memory's dim check), so every already-embedded collection
# would quietly stop matching new queries. That is the exact silent degradation
# this module must not cause. The fix for a mis-pooled model is therefore an
# explicit opt-in (``embedding_pooling``) plus a LOUD warning when a model
# declares LAST and is being pooled otherwise - never a default change under
# users' feet.
_POOLING_DEFAULT = _POOLING_MEAN
_EMBED_CTX = 512          # embedding models cap at 512 tokens; short texts anyway


def resolve_pooling_setting(spec: object) -> object:
    """Map the ``embedding_pooling`` config value to a llama.cpp pooling int, or
    POOLING_AUTO for per-model resolution. An unrecognised value falls back to the
    MEAN default rather than failing the load, but says so (never silently)."""
    if spec is None:
        return _POOLING_DEFAULT
    text = str(spec).strip().lower()
    if not text:
        return _POOLING_DEFAULT
    if text == POOLING_AUTO:
        return POOLING_AUTO
    if text in _POOLING_BY_NAME:
        return _POOLING_BY_NAME[text]
    logger.warning(
        "embedding_pooling=%r is not one of %s; using 'mean'",
        spec, ", ".join(POOLING_CHOICES))
    return _POOLING_DEFAULT


def declared_pooling_type(model, api) -> Optional[int]:
    """The pooling type the GGUF itself DECLARES (``<arch>.pooling_type``), or
    None when it declares none. Read from model metadata, so it needs no context.

    Verified 2026-07-15: bge-small -> 2 (CLS), nomic -> 1 (MEAN),
    Qwen3-Embedding-0.6B -> 3 (LAST), gte-Qwen2 / Qwen2.5-chat -> not declared.
    Best-effort by design: a model that declares nothing is the NORMAL case, and
    an unreadable value must not fail an otherwise fine load - the caller then
    just keeps its configured pooling (and this is debug-logged, not silent)."""
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
        # A model that explicitly declares UNSPECIFIED has declared nothing
        # usable; report it the same as an absent key so "auto" falls back to
        # MEAN rather than asking llama.cpp to pool by an unspecified rule.
        return None if value == _POOLING_UNSPECIFIED else value
    except Exception as e:
        logger.debug("could not read the declared pooling type (%s: %s)",
                     type(e).__name__, e)
        return None


def _effective_pooling(requested: object, declared: Optional[int]) -> int:
    """Resolve the pooling actually used: an explicit choice wins as-is; AUTO
    honours what the model declares, falling back to MEAN when it declares
    nothing usable (a decoder declaring NONE would otherwise produce no pooled
    output at all - see _embed_one's null-embedding guard)."""
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


def resolve_embedding_model_path(*, allow_download: Optional[bool] = None) -> Optional[str]:
    """Resolve the configured embedding model to a GGUF path, or None.

    Order: an explicit filesystem path -> a registered model name -> a known key
    (downloaded into <home>/models/embeddings if missing and the net policy allows).
    ``allow_download`` overrides the policy (used by ``localm setup-embeddings`` to
    force the fetch); default follows net_mode (auto only under 'allow')."""
    from localm.config import load_config
    spec = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
    if not spec:
        return None

    # 1. An explicit path to a GGUF.
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p)

    # 2. A registered model name.
    try:
        from localm.model_manager.registry import get_model_info
        info = get_model_info(spec)
        if info:
            path = info[0] if isinstance(info, tuple) else info
            if path and Path(path).is_file():
                return str(path)
    except Exception:
        pass

    # 3. A known embedding-model key.
    known = KNOWN_EMBEDDING_MODELS.get(spec)
    if not known:
        logger.debug("embedding_model %r is not a path, a registered model, or a "
                     "known key %s", spec, tuple(KNOWN_EMBEDDING_MODELS))
        return None
    repo, filename = known
    dest = _embeddings_dir() / filename
    if dest.is_file():
        return str(dest)
    return _download_known(spec, repo, filename, dest, allow_download)


def _download_known(name: str, repo: str, filename: str, dest: Path,
                    allow_download: Optional[bool]) -> Optional[str]:
    """Fetch a known embedding GGUF, gated by the network policy."""
    from localm.netpolicy import network_mode
    if allow_download is None:
        allow_download = network_mode() == "allow"
    if not allow_download:
        logger.info(
            "embedding model %r not present and not auto-downloading (net_mode=%s); "
            "run 'localm setup-embeddings' or set net_mode=allow to enable semantic "
            "search (memory/RAG use lexical BM25 until then)", name, network_mode())
        return None
    if network_mode() == "off":
        logger.info("embedding model %r missing and network is off; lexical-only", name)
        return None
    try:
        from huggingface_hub import hf_hub_download
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading embedding model %s/%s (one-time)...", repo, filename)
        got = hf_hub_download(repo, filename, local_dir=str(dest.parent))
        # hf may nest under the repo dir; normalise to dest.
        got_p = Path(got)
        if got_p.resolve() != dest.resolve() and got_p.is_file():
            return str(got_p)
        return str(dest) if dest.is_file() else (str(got_p) if got_p.is_file() else None)
    except Exception as e:
        logger.warning("embedding model download failed (%s); lexical-only", e)
        return None


class GGUFEmbedder:
    """A dedicated embedding GGUF loaded in embeddings mode via the native llama.dll."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: int = _EMBED_CTX,
                 pooling_type: object = _POOLING_DEFAULT) -> None:
        from localm.inference.backends.llamacpp import _api as api
        from localm.inference.backends.llamacpp._structs import llama_token
        self._api = api
        self._llama_token = llama_token
        self.model_path = model_path
        self._lock = threading.RLock()
        self._n_ctx = n_ctx
        self._model = None
        self._ctx = None
        self._vocab = None
        self._mem = None
        self.dim = 0
        # What the GGUF declares vs what we actually pool with. Reported up to
        # IsolatedEmbedder (via the runner's load meta) so the PARENT can warn a
        # user whose model is being mis-pooled - this class runs in an isolated
        # child process, so it reports facts rather than owning that decision.
        self.declared_pooling: Optional[int] = None
        self.pooling_type: int = _POOLING_DEFAULT

        if not api.has_embeddings_api():
            raise RuntimeError(
                "this llama.dll build does not expose the embeddings API")
        api.llama_backend_init()
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        if n_gpu_layers >= 99:
            mp.use_mmap = False
        # Multi-GPU: honour the configured main_gpu_index / gpu_split_indices,
        # same as the chat backend (see discover.apply_main_gpu/apply_gpu_split
        # and llamacpp/llama.py). The returned buffer must stay alive through
        # llama_load_model_from_file below - it is read once at load time, not
        # held as a live pointer. VRAM preflight (both the single-GPU
        # VramSizingMixin._check_vram()-style check and the multi-GPU
        # gpu_split_shortfall() check) now lives in the PARENT
        # (IsolatedEmbedder, below) instead of here, mirroring how
        # GgufWorker.load() carries no preflight of its own (only
        # GgufBackend.load() does) - this class is the raw native loader from
        # here on, isolated inside a child process by _embedder_runner.py.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        _tensor_split_keepalive = apply_gpu_split(mp)
        self._model = api.llama_load_model_from_file(model_path, mp)
        if not self._model:
            raise RuntimeError(f"failed to load embedding model: {model_path}")
        self._vocab = api.llama_model_get_vocab(self._model)
        self.dim = int(api.llama_model_n_embd(self._model))
        # Read what the model declares BEFORE creating the context (a metadata
        # read needs no context), so "auto" can honour it and so a mis-pool is
        # reportable either way.
        self.declared_pooling = declared_pooling_type(self._model, api)
        self.pooling_type = _effective_pooling(pooling_type, self.declared_pooling)
        logger.debug("embedder %s: declared pooling %s, using %s",
                     Path(model_path).name, pooling_name(self.declared_pooling),
                     pooling_name(self.pooling_type))
        cp = api.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = n_ctx
        cp.n_ubatch = n_ctx           # non-causal encode needs ubatch >= seq len
        cp.embeddings = True
        cp.pooling_type = self.pooling_type
        self._ctx = api.llama_init_from_model(self._model, cp)
        if not self._ctx:
            api.llama_free_model(self._model)
            self._model = None
            raise RuntimeError("failed to create embedding context")
        self._mem = api.llama_get_memory(self._ctx) if api.has_memory_api() else None

    def _embed_one(self, text: str) -> List[float]:
        api = self._api
        if self._mem is not None:
            api.llama_memory_clear(self._mem, True)
        raw = (text or " ").encode("utf-8")
        buf = (self._llama_token * self._n_ctx)()
        n = api.llama_tokenize(self._vocab, raw, len(raw), buf, self._n_ctx,
                               True, True)   # add_special (BERT CLS/SEP), parse_special
        if n < 0:
            # Over-long input: llama_tokenize returns -(tokens needed) and, on
            # this DLL, writes NOTHING into the short buffer (probe-verified),
            # so treating the zero-filled buffer as tokens embedded EVERY
            # over-long text to one identical garbage vector (all token 0;
            # memory-audit 2026-07-02, high). Retokenize into a right-sized
            # buffer and truncate explicitly to the context window.
            needed = -n
            full = (self._llama_token * needed)()
            n2 = api.llama_tokenize(self._vocab, raw, len(raw), full, needed,
                                    True, True)
            if n2 <= 0:                      # should not happen; fail visibly
                logger.warning(
                    "embedder: retokenize of an over-long text failed (%d)", n2)
                return [0.0] * self.dim
            # Keep the first n_ctx tokens but preserve the FINAL token of the
            # full sequence: with add_special=True on the BERT-family models
            # this embedder serves (bge/nomic), that is the [SEP] the pooled
            # encoding expects; dropping it degrades the embedding.
            for i in range(self._n_ctx):
                buf[i] = full[i]
            buf[self._n_ctx - 1] = full[n2 - 1]
            n = self._n_ctx
            logger.debug(
                "embedder: input of %d tokens truncated to the %d-token window",
                n2, self._n_ctx)
        if n <= 0:
            return [0.0] * self.dim
        arr = (self._llama_token * n)(*buf[:n])
        batch = api.llama_batch_get_one(arr, n)
        ret = api.llama_decode(self._ctx, batch)
        if ret != 0:
            raise RuntimeError(f"embedding decode failed (code {ret})")
        ptr = api.llama_get_embeddings_seq(self._ctx, 0)
        if not ptr:
            raise RuntimeError("null embedding (pooling produced no output)")
        v = [float(ptr[i]) for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed(self, texts: List[str]) -> List[List[float]]:
        """L2-normalised embedding per text (aligned 1:1 with *texts*)."""
        with self._lock:
            if self._ctx is None:
                raise RuntimeError("embedder is closed")
            return [self._embed_one(t) for t in texts]

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
    child process (see ``_embedder_runner.py``) - the same native-abort
    containment PR #606 gave the chat backend, applied to the embedder's
    separate, previously-unisolated GGUF/llama.cpp load path (honesty-audit
    follow-up, 2026-07-14).

    Preflight VRAM sizing (inherited from VramSizingMixin, shared with
    GgufBackend) runs HERE, before a child is even spawned - a load that can
    never fit still fails fast without paying a process-spawn cost, exactly
    mirroring GgufBackend/GgufWorker's split of responsibilities."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: int = _EMBED_CTX,
                 pooling_type: object = _POOLING_DEFAULT) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.effective_gpu_layers = None    # no auto-sizing for the embedder
        self.n_ctx = n_ctx
        self._pooling_type = pooling_type
        self.dim = 0
        # Reported by the child at load (see _reload): what the GGUF declares and
        # what is actually pooled with.
        self.declared_pooling: Optional[int] = None
        self.effective_pooling: Optional[int] = None
        self._runner = None
        # Serializes embed() below. The worker protocol has NO request-id
        # correlation (one req_q/resp_q pair, see _embedder_runner.py's module
        # docstring: "one command processed at a time"), so two overlapping
        # RPCs are two threads blocked in the same resp_q.get() and the queue
        # hands each whichever response arrives first - a caller can silently
        # receive vectors belonging to a DIFFERENT text. This restores the
        # serialization the pre-#643 in-process GGUFEmbedder.embed() had via
        # its own RLock (still above), which the move to an isolated worker
        # dropped. Costs no throughput: the single child drains req_q FIFO, so
        # concurrent embeds were never actually parallel.
        self._rpc_lock = threading.RLock()
        # In-flight embed() calls, mirroring Engine.active_requests - checked by
        # http_server.py's unload/shutdown/restart paths (via active_requests()
        # below) before releasing this embedder, exactly like a pinned chat
        # Engine is skipped by unload_all_models/unload_one_model (AUDIT-CRIT-1).
        # Plain int, incremented/decremented lock-free in embed() below: the
        # same best-effort precision Engine's own _pin/_unpin already accept.
        self.active_requests = 0
        self._reload()

    def _preflight_vram(self) -> None:
        """Refuse a load that cannot fit BEFORE spawning a child. The
        multi-GPU split case needs its own per-device check distinct from
        VramSizingMixin._check_vram() (which only reasons about the single
        main GPU device) - see gpu_split_shortfall's docstring (discover.py)
        for the full rationale. The single-GPU case (the common one) used to
        have NO real check at all; _check_vram() closes that gap."""
        from localm.config import load_config
        from localm.discover import gpu_split_shortfall, split_device_count
        cfg = load_config()
        if split_device_count(cfg) >= 2:
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
            # The common single-GPU case: the same preflight GgufBackend.load()
            # runs before every chat-model load.
            self._check_vram()

    def _reload(self) -> None:
        """(Re)run preflight and spawn a fresh child that loads the model.
        Used both by __init__ and by embed()'s auto-respawn after a crash -
        VRAM may have changed since the last load, so preflight re-runs too."""
        self._preflight_vram()
        from ._embedder_runner import EmbedderRunner
        params = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                      n_ctx=self.n_ctx, pooling_type=self._pooling_type)
        self._runner = EmbedderRunner()
        meta = self._runner.spawn_and_load(params)
        self.dim = meta["dim"]
        self.declared_pooling = meta.get("declared_pooling")
        self.effective_pooling = meta.get("effective_pooling")
        self._warn_if_mispooled()

    def _warn_if_mispooled(self) -> None:
        """Say so when this model is being pooled against its own training.

        A model declaring LAST-token pooling is a DECODER-based embedder
        (Qwen3-Embedding, verified 2026-07-15: qwen3.pooling_type=3). Its
        embeddings are trained to be read off the final token, so pooling it any
        other way degrades them - silently, because it still returns healthy,
        normalised, plausible vectors. That silence is the defect; the pooling
        choice itself stays the user's (embedding_pooling), we just stop hiding
        the consequence (AGENTS.md rule 5).

        Only LAST is worth a warning. bge declares CLS and is pooled MEAN, which
        measures fine (+0.29 related/unrelated margin) and is what every existing
        index was built with - warning about that on the DEFAULT setup would be
        noise, not signal, so it stays at debug level in GGUFEmbedder.
        """
        declared, effective = self.declared_pooling, self.effective_pooling
        if declared != _POOLING_LAST or effective == _POOLING_LAST:
            return
        logger.warning(
            "embedding model %s declares %s-token pooling (it is a decoder-based "
            "embedder) but is being pooled with %s, which degrades its "
            "embeddings. Set embedding_pooling=last (or 'auto') to use the "
            "model's own pooling; existing RAG collections and memory vectors "
            "were built with %s and need re-indexing after the change.",
            Path(self.model_path).name, pooling_name(declared),
            pooling_name(effective), pooling_name(effective))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via the isolated worker, transparently respawning it
        first if a PRIOR call's crash left it dead - so one transient native
        fault does not permanently disable embeddings for the rest of the
        process's life (mirrors Engine.chat_stream's auto-reload after a
        chat-backend crash). A crash DURING this call is still raised to the
        caller (rule 5: never silently swallowed) - only the NEXT call
        recovers automatically.

        Pinned via ``active_requests`` for the whole call (including a
        respawn), not just the RPC itself - a ``reset_embedder()`` arriving
        mid-respawn would free the very runner this call is about to use.

        Serialized on ``_rpc_lock`` (see __init__): the respawn check and the
        RPC together, so concurrent callers can neither swap responses on the
        correlation-free queue pair nor both respawn and orphan a worker. The
        pin is taken BEFORE the lock, so a caller merely queued behind another
        still counts as in-flight and keeps this embedder from being freed."""
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
                    # clean ("error", msg) envelope and keeps serving - embedding
                    # is stateless per call, so one bad request is not supposed to
                    # take down a worker that can still work (see
                    # _embedder_runner's module docstring, and
                    # test_embed_dispatch_catches_ordinary_exceptions_without_
                    # crashing, which pins that the child stays alive). The parent
                    # sees the SAME RuntimeError for that as for a crash, so
                    # dropping the reference unconditionally orphaned a LIVE child:
                    # EmbedderRunner has no __del__ and GC never terminates an
                    # mp.Process, so it sat blocked on req_q.get() holding the
                    # model in VRAM while the next call spawned a SECOND worker
                    # beside it - one leak per clean embed error, invisible to
                    # close()/reset_embedder()/release_for_exit(), which all only
                    # reach the CURRENT runner.
                    if self._runner is not None and self._runner.is_alive():
                        logger.exception(
                            "embedding failed; the worker is healthy and will "
                            "serve the next call")
                        raise
                    logger.exception(
                        "embedding worker fault; it will reload on the next call")
                    runner, self._runner = self._runner, None
                    if runner is not None:
                        # Release its queues/handles. Safe and idempotent when the
                        # child is already dead (ModelRunner.shutdown's contract),
                        # and _wait's own timeout path has already done it.
                        runner.shutdown(grace=0)
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

_LOCK = threading.RLock()
_EMBEDDER: Optional[IsolatedEmbedder] = None
# A model that is present on disk but fails to LOAD (corrupt / OOM) is cached as
# failed so the expensive load is not retried on every call. A *missing* model is
# deliberately NOT cached: `localm setup-embeddings` can install one into a RUNNING
# server, and get_embedder re-checks the filesystem each call so it is picked up
# without a restart. (Regression fixed: the old single `_TRIED` latch cached the
# "no model" result for the whole process lifetime, so embeddings stayed dead -
# 422 - until a restart even right after `setup-embeddings`.)
_LOAD_FAILED = False
_TRIED_DOWNLOAD = False          # one-time auto-download attempt (only net_mode=allow)
_LAST_ERROR: Optional[str] = None   # why the last load failed (for the GUI picker)


def _maybe_swap_for_embedder(path: str, n_gpu_layers: int) -> None:
    """Before the embedder's native load, evict a resident chat model when it
    would not otherwise fit - the SAME VRAM-aware swap the image/music/video
    plugins run before their own model load (see ``localm.vram.decide_media_swap``),
    generalized here via ``decide_embedder_swap``/``evict_chat_for_embedder``.

    This is complementary to, not redundant with, IsolatedEmbedder's own
    ``_preflight_vram()`` (below): that check only REFUSES a load that will
    not fit (fail fast, no child spawned). It never makes room. This function
    is what actually frees VRAM by evicting the resident chat model first -
    the same swap-before-load the image/music/video plugins already do -
    so the preflight then has a real chance of succeeding instead of just
    failing faster.

    A CPU-only embedder load (``n_gpu_layers <= 0``) never contends for VRAM,
    so it is skipped. Best-effort: any failure to read the file size or decide
    the swap leaves the load path exactly as before this check existed (never
    blocks a legitimate load - the swap is an optimization, not a
    correctness requirement of the load itself)."""
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
    # already uses for its own gpu_split_shortfall check (weights + ~20%
    # KV-cache/compute slop).
    estimate = int(file_size * 1.2)
    if not decide_embedder_swap(estimate, policy=policy):
        return
    evict_chat_for_embedder()


def get_embedder() -> Optional[IsolatedEmbedder]:
    """The shared embedder, loading the configured model on first use. Returns None
    when no embedding model can be resolved - callers then fall back to lexical
    retrieval. A missing model is re-checked on every call (so a mid-session
    ``localm setup-embeddings`` is picked up without a restart); only a genuine
    load FAILURE is cached. Loading holds the engine's process-global load lock so
    it cannot race a chat-model load onto the GPU.

    The pre-load swap check (``_maybe_swap_for_embedder``) deliberately runs
    OUTSIDE ``_LOCK`` (double-checked locking below), not inside a single held
    lock the way the rest of this function is structured. It can evict a
    resident chat model via ``vram.evict_chat_for_embedder``, which submits
    ``http_server.unload_all_models()`` onto the SERVER'S event-loop thread and
    blocks THIS thread waiting for it - and that coroutine calls
    ``loaded_dim()``, which itself needs ``_LOCK``. Since ``_LOCK`` is a plain
    ``threading.RLock`` (thread-owned, not reentrant across threads), holding
    it here while blocking on that coroutine is a genuine cross-thread
    deadlock: this thread waits on the coroutine, the coroutine (on the event
    loop) waits on this thread's lock. Confirmed via live reproduction
    (2026-07-14 review) - the event loop stayed blocked for the full
    eviction timeout before either side could proceed."""
    global _EMBEDDER, _LOAD_FAILED, _TRIED_DOWNLOAD, _LAST_ERROR
    with _LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _LOAD_FAILED:
            return None
        # Cheap filesystem-only re-check every call (NO download): finds a model a
        # user just installed into this running server.
        path = resolve_embedding_model_path(allow_download=False)
        if not path and not _TRIED_DOWNLOAD:
            # First miss: one auto-download attempt, gated by net policy inside
            # (only actually fetches under net_mode=allow). Latched so a batch of
            # embed calls does not re-attempt the download on every chunk.
            _TRIED_DOWNLOAD = True
            path = resolve_embedding_model_path()
        if not path:
            return None
        from localm.config import load_config
        _cfg = load_config()
        ngl = int(_cfg.get("n_gpu_layers", 99))
        pooling = resolve_pooling_setting(_cfg.get("embedding_pooling"))

    # OUTSIDE _LOCK (see docstring): safe to block here on the cross-thread
    # eviction round trip. Another thread may concurrently reach this same
    # window and run its own swap check too - harmless, since
    # evict_chat_for_embedder()/decide_embedder_swap() are idempotent (a
    # second call simply finds nothing left to evict, or enough VRAM already
    # free) and the actual load below re-checks the singleton state.
    _maybe_swap_for_embedder(path, ngl)

    with _LOCK:
        # Re-check: another thread may have completed (or failed) the load
        # while this thread was outside the lock running the swap check.
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _LOAD_FAILED:
            return None
        try:
            from localm.inference.engine import _LOAD_LOCK
            with _LOAD_LOCK:
                _EMBEDDER = IsolatedEmbedder(path, n_gpu_layers=ngl,
                                             pooling_type=pooling)
            # getattr, not attribute access: this status line sits INSIDE the
            # try below, so anything it raises would be caught as a LOAD failure
            # and silently drop embeddings to lexical-only for the rest of the
            # process. A line that merely describes the load must never be able
            # to fail it.
            logger.info("embedding model ready: %s (dim=%d, pooling=%s)", path,
                        _EMBEDDER.dim,
                        pooling_name(getattr(_EMBEDDER, "effective_pooling", None)))
            _LAST_ERROR = None
            return _EMBEDDER
        except Exception as e:
            _LOAD_FAILED = True
            _LAST_ERROR = str(e)
            logger.warning("could not load embedding model %s (%s); lexical-only",
                           path, e)
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


def loaded_path() -> Optional[str]:
    """Filesystem path of the currently-loaded embedder, or None if none is
    loaded. Does NOT trigger a load. Lets a caller outside this module (the
    Models page, the targeted unload route) recognise a registered model
    entry as "this is the resident embedder" - it never appears in
    ``http_server._engines``, so that is otherwise invisible - by comparing
    against the entry's own resolved path. Pairs with ``loaded_dim()``."""
    with _LOCK:
        return _EMBEDDER.model_path if _EMBEDDER is not None else None


def active_requests() -> int:
    """In-flight embed() calls on the currently-loaded embedder, or 0 if none is
    loaded. Mirrors ``Engine.active_requests`` - checked by http_server.py's
    unload-all/targeted-unload/shutdown/restart paths before releasing this
    embedder, exactly like a pinned chat Engine is skipped by
    unload_all_models/unload_one_model (AUDIT-CRIT-1): without this, any of
    those paths could free the embedder - and the isolated worker process a
    request is waiting on - out from under an in-flight embed() call."""
    with _LOCK:
        return _EMBEDDER.active_requests if _EMBEDDER is not None else 0


def last_error() -> Optional[str]:
    """Why the last embedding-model LOAD failed (e.g. the model is not an embedding
    model), or None. For the GUI picker to tell the user what went wrong."""
    with _LOCK:
        return _LAST_ERROR


def reset_embedder() -> None:
    """Drop the cached embedder and its negative caches (tests / a model change)."""
    global _EMBEDDER, _LOAD_FAILED, _TRIED_DOWNLOAD, _LAST_ERROR
    with _LOCK:
        if _EMBEDDER is not None:
            _EMBEDDER.close()
        _EMBEDDER = None
        _LOAD_FAILED = False
        _TRIED_DOWNLOAD = False
        _LAST_ERROR = None


def release_for_exit() -> bool:
    """Release the isolated embedder worker for a caller that is about to
    ``os._exit()`` / ``os.execv()``. Returns True if a worker was released.

    Both of those bypass ``atexit`` - and multiprocessing's daemon-child
    reclamation IS an atexit hook (``multiprocessing.util._exit_function``) - so
    a worker left running does NOT die with the parent: it survives as an orphan
    still holding its model in VRAM until killed by hand, and the restarted
    server spawns a second one next to it. Verified live (2026-07-15): a daemon
    child is reclaimed on a normal interpreter exit, but survives BOTH os._exit
    and os.execv.

    This is the WHOLE exit-path decision on purpose, because every other way to
    make it takes ``_LOCK``: ``active_requests()`` does, and ``reset_embedder()``
    does. ``get_embedder()`` holds ``_LOCK`` for the FULL duration of an
    embedding-model load (up to its load timeout), so a stop or restart issued
    during a load would block on the guard itself and hang the very action the
    user asked for - never reaching the teardown, and leaving exactly the orphan
    this function exists to prevent. Nothing here touches ``_LOCK``: the
    singleton is snapshotted directly, which is a best-effort read the exiting
    caller can afford (a hard exit racing a load that has not yet published its
    embedder is inherent to any hard exit).

    A BUSY worker (a request is mid-``embed()``) is terminated WITHOUT waiting:
    the pinned request cannot be answered either way once the process exits, so
    there is nothing left to protect, and the polite "shutdown" command would
    only queue behind the in-flight embed and stall the exit for the grace
    period. An IDLE worker gets the polite command first, so the child frees the
    model cleanly before it goes.

    The module singleton is deliberately NOT cleared (that is reset_embedder()'s
    job, and it needs ``_LOCK``): the process is going away regardless."""
    emb = _EMBEDDER
    if emb is None:
        return False
    runner = getattr(emb, "_runner", None)
    if runner is None:
        return False
    busy = getattr(emb, "active_requests", 0) > 0
    runner.shutdown(grace=0 if busy else 5.0)
    return True


# A daemon worker already dies with the parent, but tear it down explicitly at
# interpreter exit so its queue feeder threads never delay shutdown - mirrors
# voice.py's atexit registration for the identical process-wide-singleton shape
# (the embedder, unlike GgufBackend/ModelRunner, is not torn down by an owning
# caller on every unload - it outlives any single request).
atexit.register(reset_embedder)
