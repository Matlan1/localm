# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model factory - picks the right backend and exposes a unified inference API."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Iterator, List, Optional

from localm.config import load_config
from localm.console import console
from localm.debuglog import logger
from localm.inference.backends.base import BaseBackend
from localm.textnorm import scrub_stream


# Process-global model-load lock. Loading a model onto the GPU is the dangerous,
# memory-spiking step; running two loads at once (e.g. a chat request and a
# background job, each with its own Engine) thrashes VRAM, garbles the interleaved
# console output, and can freeze the machine. Serialise every load process-wide so
# only one model is ever loading at a time. Inference itself is NOT held here, only
# the load. RLock (not Lock) so a re-entrant load on the same thread cannot
# deadlock.
#
# LOCK ORDER with localm.inference.embedder._LOCK: _LOAD_LOCK is the OUTER
# lock of the pair. The load path under this lock calls into embedder status
# reads that take embedder._LOCK (backend ctx sizing -> loaded_path, #767),
# so code holding embedder._LOCK must NEVER acquire _LOAD_LOCK or start a
# load - that inversion deadlocked the whole server on 2026-08-18
# (embedder.get_embedder held _LOCK and waited here while a chat preload
# held this lock and waited on _LOCK; nothing had a timeout).
_LOAD_LOCK = threading.RLock()


# A bare config.json is not enough to call a directory a model: localm's own
# data directory keeps its settings in a config.json too. A real HF model also
# carries weights or a tokenizer next to the config, so require one of those.
_HF_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
_HF_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "sentencepiece.bpe.model",
)


def _has_hf_model_artifacts(p: Path) -> bool:
    """True when an HF directory holds real model files, not just a config.json."""
    if (p / "adapter_config.json").exists():   # LoRA / adapter directory
        return True
    if any(next(p.glob(pat), None) is not None for pat in _HF_WEIGHT_GLOBS):
        return True
    return any((p / t).exists() for t in _HF_TOKENIZER_FILES)


def _is_hf_dir(path: str) -> bool:
    """True when path is a directory that looks like a HuggingFace model."""
    p = Path(path)
    return (
        p.is_dir()
        and (p / "config.json").exists()
        and _has_hf_model_artifacts(p)
    )


def _is_gguf(path: str) -> bool:
    p = Path(path)
    # Standard GGUF extension OR Ollama blob (sha256-<digest>, no extension)
    return p.suffix.lower() == ".gguf" or (p.is_file() and p.name.startswith("sha256-"))


def _resolve_vram_overhead_bytes(cfg: dict) -> int:
    """``vram_overhead_mb`` (MB) from config, in bytes, or the built-in default on a missing/unparseable value."""
    from localm.vram import VRAM_OVERHEAD_BYTES
    raw = cfg.get("vram_overhead_mb")
    if raw is None:
        return VRAM_OVERHEAD_BYTES
    try:
        return int(raw) * 1024 ** 2
    except (TypeError, ValueError):
        from localm.debuglog import logger as _dbg
        _dbg.warning("vram_overhead_mb is set but not a valid number (%r); "
                     "using the default %.1f GB", raw, VRAM_OVERHEAD_BYTES / 1024 ** 3)
        return VRAM_OVERHEAD_BYTES


def create_backend(
    model_path: str,
    *,
    mmproj_path: Optional[str] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    device: Optional[str] = None,
) -> BaseBackend:
    """Return the appropriate backend for the given model path, without loading it."""
    cfg = load_config()

    if _is_hf_dir(model_path):
        from localm.inference.backends.hf import HFBackend
        return HFBackend(model_path, device=device)

    if _is_gguf(model_path):
        from localm.inference.backends.gguf import GgufBackend
        return GgufBackend(
            model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx or cfg["n_ctx"],
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else cfg["n_gpu_layers"],
            n_ctx_max=cfg.get("n_ctx_max", 16384),
            n_ctx_grow=cfg.get("n_ctx_grow", 4096),
            ctx_auto=bool(cfg.get("ctx_auto", False)),
            n_gpu_layers_auto=bool(cfg.get("n_gpu_layers_auto", True)),
            n_cpu_moe=int(cfg.get("n_cpu_moe", 0) or 0),
            mtp_enabled=bool(cfg.get("mtp_enabled", True)),
            vram_overhead_bytes=_resolve_vram_overhead_bytes(cfg),
        )

    raise ValueError(
        f"Cannot determine backend for: {model_path}\n"
        "Expected a HuggingFace model directory (contains config.json) "
        "or a .gguf file."
    )


# A model's self-declared name is echoed to API/GUI callers, so it is treated as
# untrusted text, not as a label we own. Real values look like "meta-llama/
# Llama-3-8B" or "gemma-3-4b-it": a short run of word characters with . _ - / in
# between. Anything else (control characters, newlines, markup, a whole absolute
# path, a paragraph) is a value we will not repeat back to a caller - we fall back
# to the directory name, which we DO control. Kept as an allowlist, so a shape
# nobody anticipated is refused rather than passed through.
_SANE_DISPLAY_NAME = re.compile(r"^[\w.-]+(?:/[\w.-]+)*$")
_MAX_DISPLAY_NAME = 96


def _sane_display_name(value) -> Optional[str]:
    """*value* if it is a plausible model name, else None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > _MAX_DISPLAY_NAME:
        return None
    return v if _SANE_DISPLAY_NAME.match(v) else None


def model_display_name(model_path: str) -> str:
    """Human-readable model name from path."""
    p = Path(model_path)
    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                declared = _sane_display_name(cfg.get("_name_or_path"))
                if declared:
                    return declared
                if cfg.get("_name_or_path"):
                    logger.debug(
                        "model_display_name: ignoring implausible _name_or_path in "
                        "%s; using the directory name instead", cfg_file)
            except Exception as exc:
                # config.json is optional for the display name only; load() validates
                # it later. Falling back to the dir name is fine, but surface the
                # corrupt config here so it is discoverable under --debug.
                logger.debug("model_display_name: unreadable config.json at %s: %s", cfg_file, exc)
        return p.name
    return p.stem


class Engine:
    """High-level wrapper: loads a backend and streams chat completions."""

    def __init__(
        self,
        model_path: str,
        *,
        mmproj_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        device: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.display_name = display_name or model_display_name(model_path)
        # Every Engine shares the one process-global load lock (see _LOAD_LOCK), so
        # loads serialise across the server, jobs, and embeds, not just within a
        # single Engine instance.
        self._load_lock = _LOAD_LOCK
        self._backend = create_backend(
            model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            device=device,
        )
        self.active_requests = 0
        # Set True by an unload/eviction path for the duration of the native
        # free, so get_engine()/switch_engine()'s fast paths refuse to hand this
        # engine back (and thus refuse to let a request pin it) while it is being
        # torn down. Closes the pin-arrives-during-the-unload-await window that
        # active_requests alone cannot (a request pins lock-free, AFTER the
        # active_requests==0 check has already passed). See http_server.py.
        self.unloading = False

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

    def set_load_cancel(self, event) -> None:
        """Install a cancel event on the backend so an in-flight load() can be aborted mid-flight when a newer model selection supersedes it (preemptive switching)."""
        self._backend.set_load_cancel(event)

    def load(self) -> None:
        if not hasattr(self, "_load_lock"):
            self._load_lock = _LOAD_LOCK
        with self._load_lock:
            if self._backend.loaded:
                return
            backend_type = type(self._backend).__name__.replace("Backend", "")
            console.print(
                f"Loading [bold cyan]{self.display_name}[/bold cyan] "
                f"[dim](backend: {backend_type})[/dim]"
            )
            self._backend.load()

    def unload(self) -> None:
        self._backend.unload()

    @property
    def effective_ctx_max(self):
        """Resolved context ceiling of the last load (VRAM-derived when ctx_auto is on), or None when unknown / not loaded yet."""
        return getattr(self._backend, "effective_ctx_max", None)

    @property
    def gpu_placement(self) -> Optional[dict]:
        """Where the last load's transformer layers actually ended up: GPU vs CPU. ``{'gpu_layers_offloaded': N, 'gpu_layers_total': M, 'degraded': bool}`` when the backend can report it (GgufBackend, once a load has completed and the model's true layer count is known), else None - e.g. before any load, or for..."""
        offloaded = getattr(self._backend, "gpu_layers_offloaded", None)
        total = getattr(self._backend, "gpu_layers_total", None)
        if offloaded is None or not total:
            return None
        return {
            "gpu_layers_offloaded": offloaded,
            "gpu_layers_total": total,
            "degraded": offloaded < total,
        }

    @property
    def last_finish_reason(self) -> str:
        """Why the most recent generation ended: 'stop' (model finished) or 'length' (the max_tokens budget ran out)."""
        return getattr(self._backend, "last_finish_reason", "stop")

    @property
    def supports_images(self) -> bool:
        """True when the active backend can actually see image input."""
        return getattr(self._backend, "supports_images", False)

    @property
    def can_be_multimodal(self) -> bool:
        """True when the backend class could support images, so it is worth loading the model to find out."""
        return getattr(self._backend, "can_be_multimodal", False)

    @property
    def supports_mtp(self) -> bool:
        """True when the loaded model has active Multi-Token Prediction (MTP) heads."""
        return getattr(self._backend, "supports_mtp", False)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* using the loaded backend's tokenizer."""
        return self._backend.count_tokens(text)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return the number of tokens in a list of structured messages, including chat template formatting."""
        return self._backend.count_messages_tokens(messages)

    def context_capacity(self) -> Optional[int]:
        """Maximum token capacity of the loaded model's context window."""
        b = self._backend
        eff = getattr(b, "effective_ctx_max", None)
        if isinstance(eff, int) and eff > 0:
            return eff
        for attr in ("n_ctx_max", "_n_ctx_max", "n_ctx", "_n_ctx"):
            v = getattr(b, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        llm = getattr(b, "_llm", None)
        try:
            n = llm.n_ctx() if llm is not None else None
            return int(n) if n else None
        except Exception:
            return None

    def _embed_via_dedicated(self, texts: List[str]) -> List[List[float]]:
        """Embed with the small DEDICATED on-device embedding model (:mod:`localm.inference.embedder`), or raise with the one command that fixes it."""
        from localm.inference.embedder import embed_texts
        vecs = embed_texts(list(texts))
        if vecs is not None:
            return vecs
        raise NotImplementedError(
            "No embedding model available. Run 'localm setup-embeddings' (or "
            "set net_mode=allow) to enable semantic search; memory and RAG use "
            "lexical BM25 until then.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts."""
        if getattr(self._backend, "can_embed", True) is False:
            return self._embed_via_dedicated(texts)
        if not self._backend.loaded:
            with _LOAD_LOCK:
                if not self._backend.loaded:
                    self._backend.load()
        if getattr(self._backend, "can_embed", True) is False:
            # Only knowable now (HF): the loaded checkpoint is a chat decoder.
            # Substituting the dedicated embedder is the CORRECT outcome, not a
            # degradation, but it must not be invisible either (rule 5).
            logger.debug(
                "%s is not an embedding model; embedding via the dedicated "
                "on-device embedder instead", self.display_name)
            return self._embed_via_dedicated(texts)
        return self._backend.embed(texts)

    @property
    def supports_grammar(self) -> bool:
        """True when the active backend can actually constrain generation to a grammar."""
        return getattr(self._backend, "supports_grammar", False)

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Up-front grammar validation, delegated to the backend."""
        self._backend.validate_grammar(grammar, lazy=lazy)

    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # Auto-reload if the model was unloaded (e.g. to free VRAM for image gen).
        # Hold the process-global load lock so a reload cannot race another load
        # (chat vs job vs embed) onto the GPU, and double-check inside the lock so
        # we do not reload a model another thread just brought back.
        if not self._backend.loaded:
            with _LOAD_LOCK:
                if not self._backend.loaded:
                    console.print(
                        f"[dim]Reloading [bold]{self.display_name}[/bold]…[/dim]"
                    )
                    self._backend.load()

        cfg = load_config()
        # Normalise model-internal control markers (harmony/Gemma channel tags,
        # etc.) once here so every backend inherits it - the GGUF backend also
        # scrubs internally, which is fine because scrub_stream is idempotent,
        # while the HF backend relies on this pass alone.
        return scrub_stream(self._backend.chat_stream(
            messages,
            max_tokens=max_tokens if max_tokens is not None else cfg["max_tokens"],
            temperature=temperature if temperature is not None else cfg["temperature"],
            top_p=top_p if top_p is not None else cfg["top_p"],
            top_k=top_k if top_k is not None else cfg["top_k"],
            repeat_penalty=repeat_penalty if repeat_penalty is not None else cfg["repeat_penalty"],
            grammar=grammar,
            grammar_lazy=grammar_lazy,
            grammar_triggers=grammar_triggers,
            seed=seed,
        ))

    def __enter__(self) -> "Engine":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
