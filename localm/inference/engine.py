# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model factory - picks the right backend and exposes a unified inference API."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from localm.config import load_config
from localm.debuglog import logger
from localm.inference.backends.base import BaseBackend
from localm.inference.textnorm import scrub_stream

console = Console()


# Process-global model-load lock. Loading a model onto the GPU is the dangerous,
# memory-spiking step; running two loads at once (e.g. a chat request and a
# background job, each with its own Engine) thrashes VRAM, garbles the interleaved
# console output, and can freeze the machine. Serialise every load process-wide so
# only one model is ever loading at a time. Inference itself is NOT held here, only
# the load. RLock (not Lock) so a re-entrant load on the same thread cannot
# deadlock.
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


def create_backend(
    model_path: str,
    *,
    mmproj_path: Optional[str] = None,
    n_ctx: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    device: Optional[str] = None,
) -> BaseBackend:
    """
    Return the appropriate backend for the given model path, without loading it.

    model_path:   HF model directory  →  HFBackend
                  *.gguf file         →  GgufBackend
    """
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
        )

    raise ValueError(
        f"Cannot determine backend for: {model_path}\n"
        "Expected a HuggingFace model directory (contains config.json) "
        "or a .gguf file."
    )


def model_display_name(model_path: str) -> str:
    """Human-readable model name from path."""
    p = Path(model_path)
    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                if "_name_or_path" in cfg and cfg["_name_or_path"]:
                    return cfg["_name_or_path"]
            except Exception as exc:
                # config.json is optional for the display name only; load() validates
                # it later. Falling back to the dir name is fine, but surface the
                # corrupt config here so it is discoverable under --debug.
                logger.debug("model_display_name: unreadable config.json at %s: %s", cfg_file, exc)
        return p.name
    return p.stem


class Engine:
    """
    High-level wrapper: loads a backend and streams chat completions.

    Usage:
        engine = Engine(model_path)
        engine.load()
        for tok in engine.chat_stream(messages):
            print(tok, end="", flush=True)
        engine.unload()
    """

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

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

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
        """Resolved context ceiling of the last load (VRAM-derived when
        ctx_auto is on), or None when unknown / not loaded yet."""
        return getattr(self._backend, "effective_ctx_max", None)

    @property
    def last_finish_reason(self) -> str:
        """Why the most recent generation ended: "stop" (model finished) or
        "length" (the max_tokens budget ran out). Backends that cannot tell
        report "stop"."""
        return getattr(self._backend, "last_finish_reason", "stop")

    @property
    def supports_images(self) -> bool:
        """True when the active backend can actually see image input. For HF
        this is only accurate once the model is loaded (see can_be_multimodal)."""
        return getattr(self._backend, "supports_images", False)

    @property
    def can_be_multimodal(self) -> bool:
        """True when the backend class could support images, so it is worth
        loading the model to find out. False for text-only backends (GGUF)."""
        return getattr(self._backend, "can_be_multimodal", False)

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens in *text* using the loaded backend's
        tokenizer.  Falls back to a chars-÷-4 heuristic when the model is
        not yet loaded.
        """
        return self._backend.count_tokens(text)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embedding vectors for a list of texts.

        Delegates to the backend.  Raises ``NotImplementedError`` when the
        loaded model does not support embeddings.
        """
        if not self._backend.loaded:
            with _LOAD_LOCK:
                if not self._backend.loaded:
                    self._backend.load()
        return self._backend.embed(texts)

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
            seed=seed,
        ))

    def __enter__(self) -> "Engine":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
