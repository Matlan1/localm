"""Model factory — picks the right backend and exposes a unified inference API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from localm.config import load_config
from localm.inference.backends.base import BaseBackend

console = Console()


def _is_hf_dir(path: str) -> bool:
    """True when path is a directory that looks like a HuggingFace model."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()


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
            except Exception:
                pass
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
    ) -> Iterator[str]:
        # Auto-reload if the model was unloaded (e.g. to free VRAM for image gen)
        if not self._backend.loaded:
            console.print(
                f"[dim]Reloading [bold]{self.display_name}[/bold]…[/dim]"
            )
            self._backend.load()

        cfg = load_config()
        return self._backend.chat_stream(
            messages,
            max_tokens=max_tokens or cfg["max_tokens"],
            temperature=temperature if temperature is not None else cfg["temperature"],
            top_p=top_p or cfg["top_p"],
            top_k=top_k or cfg["top_k"],
            repeat_penalty=repeat_penalty or cfg["repeat_penalty"],
            grammar=grammar,
        )

    def __enter__(self) -> "Engine":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
