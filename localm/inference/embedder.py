# SPDX-License-Identifier: AGPL-3.0-or-later
"""
On-device text embeddings via a DEDICATED embedding GGUF, on the bundled runtime.

localm's chat models are decoder LLMs; a chat model's pooled hidden states make
poor embeddings. This module loads a small, dedicated embedding model (bge / nomic,
~25-90 MB) into its OWN llama.cpp model + context in EMBEDDINGS mode, independent of
whatever chat model is loaded, so semantic retrieval (RAG hybrid search, agent
memory) works on the default GGUF runtime with consistent quality and without
disturbing the chat model.

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
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import List, Optional

from localm.debuglog import logger

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

# Pooling: MEAN (1) always yields a pooled vector for any model and scored well for
# bge in verification; CLS/model-default can leave a NONE-pooled decoder with no
# output. Universal, robust default.
_POOLING_MEAN = 1
_EMBED_CTX = 512          # embedding models cap at 512 tokens; short texts anyway


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
                 n_ctx: int = _EMBED_CTX, pooling_type: int = _POOLING_MEAN) -> None:
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
        # held as a live pointer.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        _tensor_split_keepalive = apply_gpu_split(mp)
        self._model = api.llama_load_model_from_file(model_path, mp)
        if not self._model:
            raise RuntimeError(f"failed to load embedding model: {model_path}")
        self._vocab = api.llama_model_get_vocab(self._model)
        self.dim = int(api.llama_model_n_embd(self._model))
        cp = api.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = n_ctx
        cp.n_ubatch = n_ctx           # non-causal encode needs ubatch >= seq len
        cp.embeddings = True
        cp.pooling_type = pooling_type
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


# --------------------------------------------------------------------------- #
#  Process-wide singleton                                                      #
# --------------------------------------------------------------------------- #

_LOCK = threading.RLock()
_EMBEDDER: Optional[GGUFEmbedder] = None
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


def get_embedder() -> Optional[GGUFEmbedder]:
    """The shared embedder, loading the configured model on first use. Returns None
    when no embedding model can be resolved - callers then fall back to lexical
    retrieval. A missing model is re-checked on every call (so a mid-session
    ``localm setup-embeddings`` is picked up without a restart); only a genuine
    load FAILURE is cached. Loading holds the engine's process-global load lock so
    it cannot race a chat-model load onto the GPU."""
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
        try:
            from localm.config import load_config
            from localm.inference.engine import _LOAD_LOCK
            ngl = int(load_config().get("n_gpu_layers", 99))
            with _LOAD_LOCK:
                _EMBEDDER = GGUFEmbedder(path, n_gpu_layers=ngl)
            logger.info("embedding model ready: %s (dim=%d)", path, _EMBEDDER.dim)
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
