"""GGUF backend — uses our native ctypes wrapper around llama.dll.

The native wrapper (localm.inference.backends.llamacpp) handles GPU DLL
loading automatically.  If llama.dll cannot be found, falls back to running
llama-cli.exe as a subprocess (model reloads each call — slow but portable).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from .base import BaseBackend

console = Console()


class GgufBackend(BaseBackend):
    """
    Inference backend for GGUF model files.

    Prefers llama-cpp-python for in-process inference (fast, keeps model in
    memory between calls).  Falls back to llama-cli.exe subprocess if the
    Python bindings are not installed.
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._loaded = False
        self._use_subprocess = False   # set to True if llama-cpp-python unavailable

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        try:
            self._load_native()
        except Exception as exc:
            console.print(
                f"[yellow]Native llama backend failed ({exc}) — "
                "falling back to llama-cli.exe (model reloads each request).[/yellow]"
            )
            self._use_subprocess = True
            self._loaded = True

    def _load_native(self) -> None:
        """Load via our own ctypes wrapper (no llama-cpp-python required)."""
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()   # ensure DLLs are loaded before importing the class

        from localm.inference.backends.llamacpp import LlamaCpp
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task(
                f"Loading model  (ctx={self.n_ctx}, gpu_layers={self.n_gpu_layers})",
                total=None,
            )
            self._llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

        self._loaded = True

        # VRAM usage after load (torch / ROCm / CUDA only — skip if torch absent)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved  = torch.cuda.memory_reserved(i)  / 1e9
                    console.print(
                        f"[dim]  vram     : {allocated:.2f} GB allocated / "
                        f"{reserved:.2f} GB reserved (device {i})[/dim]"
                    )
        except Exception:
            pass

        console.print("[green]✓[/green] Model loaded")

    def unload(self) -> None:
        self._llm = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded model's vocabulary."""
        if self._llm is not None:
            return len(self._llm.tokenize(text, add_bos=False))
        # Subprocess fallback or not loaded yet — fall back to heuristic
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        grammar: Optional[str] = None,
    ) -> Iterator[str]:
        if self._use_subprocess:
            yield from self._subprocess_stream(messages, max_tokens, temperature)
            return

        # native ctypes path (LlamaCpp)
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            grammar=grammar,
            stream=True,
        ):
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            if token:
                yield token

    def _subprocess_stream(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """One-shot subprocess call to llama-cli.exe — slow but always works."""
        from localm.config import find_binary_dir

        binary_dir = find_binary_dir()
        if binary_dir is None:
            yield "[error: llama-cli.exe not found and llama-cpp-python not installed]"
            return

        cli = binary_dir / "llama-cli.exe"

        # Format messages as a simple prompt
        prompt = _format_messages_for_llama_cli(messages)

        env = os.environ.copy()
        env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")

        cmd = [
            str(cli),
            "-m", self.model_path,
            "-p", prompt,
            "--no-display-prompt",
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-ngl", str(self.n_gpu_layers),
            "--log-disable",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=str(binary_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for char in iter(lambda: proc.stdout.read(1), ""):
            yield char
        proc.wait()


def _format_messages_for_llama_cli(messages: List[dict]) -> str:
    """Minimal ChatML formatting for llama-cli prompt mode."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
