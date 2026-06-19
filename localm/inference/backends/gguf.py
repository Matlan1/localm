# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF backend - uses our native ctypes wrapper around llama.dll.

The native wrapper (localm.inference.backends.llamacpp) handles GPU DLL
loading automatically. There is no subprocess fallback: GGUF runs through this
in-process binding or it fails loudly, pointing you at `localm setup-llama`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from .base import BaseBackend

console = Console()


class GgufBackend(BaseBackend):
    """
    Inference backend for GGUF model files.

    Runs in-process via our own ctypes binding to llama.dll (no
    llama-cpp-python, no subprocess), keeping the model in memory between
    calls. If the native runtime cannot be loaded, load() raises rather than
    degrading to a slower, lower-fidelity path.
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        n_ctx_max: Optional[int] = None,
        n_ctx_grow: int = 4096,
        ctx_auto: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx_max = n_ctx_max       # ceiling for dynamic growth (0/None = unlimited)
        self.n_ctx_grow = n_ctx_grow
        self.ctx_auto = ctx_auto         # derive n_ctx_max from free VRAM at load
        self.effective_ctx_max: Optional[int] = None   # resolved ceiling of the last load
        self._llm = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    # Rough VRAM headroom for KV cache + compute buffers beyond model weights
    _VRAM_OVERHEAD_BYTES = int(1.5e9)

    @staticmethod
    def _free_vram_bytes() -> Optional[int]:
        """Free VRAM on device 0 in bytes, or None when not measurable."""
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info(0)
                return int(free)
        except Exception:
            pass
        return None

    @staticmethod
    def _vram_levels() -> list:
        """(free, total) bytes per device, [] when not measurable.

        Driver-level numbers (mem_get_info), NOT torch allocator counters:
        llama.dll allocates through HIP/CUDA directly, so
        torch.cuda.memory_allocated() reads zero for GGUF loads no matter
        how much VRAM the model actually occupies."""
        try:
            import torch
            if torch.cuda.is_available():
                return [tuple(torch.cuda.mem_get_info(i))
                        for i in range(torch.cuda.device_count())]
        except Exception:
            pass
        return []

    def _model_bytes(self) -> int:
        """Total size of the model on disk (all parts of a split GGUF)."""
        from localm.model_manager import split_gguf_parts
        p = Path(self.model_path)
        parts = split_gguf_parts(p.name)
        if parts:
            return sum(
                (p.parent / part).stat().st_size
                for part in parts if (p.parent / part).is_file()
            )
        return p.stat().st_size if p.is_file() else 0

    def _check_vram(self) -> None:
        """
        Warn - loudly and with options - when the model is unlikely to fit
        in the currently free VRAM. Never blocks: partial offload and system
        RAM spill can still work, and the estimate is approximate.
        """
        if self.n_gpu_layers == 0:
            return  # CPU-only run, VRAM is irrelevant
        free = self._free_vram_bytes()
        if free is None:
            return  # can't measure (no torch / no GPU) - nothing useful to say
        need = self._model_bytes() + self._VRAM_OVERHEAD_BYTES
        if free >= need:
            return
        console.print(
            f"[yellow]⚠ Low VRAM:[/yellow] this model needs roughly "
            f"[bold]{need / 1024**3:.1f} GB[/bold] (weights + buffers) but only "
            f"[bold]{free / 1024**3:.1f} GB[/bold] is free.\n"
            f"  [dim]Likely cause: another GPU app is holding memory "
            f"(ComfyUI, a browser, another model).[/dim]\n"
            f"  Options:\n"
            f"    • Free VRAM first (close the other app, or POST "
            f"/v1/models/unload on its server)\n"
            f"    • Offload fewer layers:  [bold]-g 24[/bold]  "
            f"(or [bold]-g 0[/bold] for CPU-only)\n"
            f"  Continuing anyway - load may be slow or fail."
        )

    # Bounds for VRAM-derived context ceilings
    _AUTO_CTX_MIN = 4096
    _AUTO_CTX_MAX = 65536
    _AUTO_CTX_FALLBACK = 16384   # no GPU visibility - match common practice

    def _auto_ctx_max(self) -> int:
        """
        Derive a context ceiling from available resources.

        Budget = free VRAM - model weights - fixed overhead. The KV cost per
        token is estimated from the model's size class (larger models have
        more layers and wider KV heads; sliding-window models need less, so
        the estimate is deliberately conservative). The result is clamped to
        a sane range and rounded to whole KiB of tokens.
        """
        free = self._free_vram_bytes()
        if free is None:
            return self._AUTO_CTX_FALLBACK
        model = self._model_bytes()
        budget = free - model - self._VRAM_OVERHEAD_BYTES
        if budget <= 0:
            return max(self.n_ctx, self._AUTO_CTX_MIN)
        # Heuristic: ~1 byte of KV per token per 100KB of model weights,
        # clamped so tiny and huge models stay in a plausible band.
        bytes_per_token = min(max(model // 100_000, 16_000), 512_000)
        auto = budget // bytes_per_token
        auto = (auto // 1024) * 1024
        return int(max(self._AUTO_CTX_MIN, min(self._AUTO_CTX_MAX, auto)))

    def _effective_ctx_max(self) -> Optional[int]:
        """The context ceiling to use for this load (auto or configured)."""
        if self.ctx_auto:
            auto = self._auto_ctx_max()
            console.print(
                f"[dim]  ctx auto : window may grow to {auto:,} tokens "
                f"(from free VRAM)[/dim]"
            )
            return auto
        return self.n_ctx_max

    def load(self) -> None:
        # Split GGUF pre-flight: all sibling parts must be present, otherwise
        # llama.cpp fails with a cryptic native error mid-load.
        from localm.model_manager import missing_split_parts
        missing = missing_split_parts(Path(self.model_path))
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Split GGUF is incomplete - missing part(s): {names}. "
                f"Re-run 'localm pull' to download all parts."
            )
        self._check_vram()
        try:
            self._load_native()
        except Exception as exc:
            free = self._free_vram_bytes()
            vram_hint = ""
            if free is not None and free < self._model_bytes() + self._VRAM_OVERHEAD_BYTES:
                vram_hint = (
                    " The GPU is low on memory - free VRAM or retry with "
                    "fewer GPU layers (-g 24, or -g 0 for CPU)."
                )
            # No subprocess fallback by design: GGUF runs through our in-process
            # ctypes binding to llama.dll or not at all. Fail loud and actionable
            # instead of silently degrading to a slower, lower-fidelity path.
            raise RuntimeError(
                f"Native llama runtime failed to load: {exc}.{vram_hint}\n"
                "Provision or repair it with  localm setup-llama  "
                "(or set LLAMA_CPP_LIB to a working llama.dll)."
            ) from exc

    def _load_native(self) -> None:
        """Load via our own ctypes wrapper (no llama-cpp-python required)."""
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()   # ensure DLLs are loaded before importing the class

        from localm.inference.backends.llamacpp import LlamaCpp
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        vram_before = self._vram_levels()

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            ctx_max = self._effective_ctx_max()
            self.effective_ctx_max = ctx_max
            cap_label = f"→{ctx_max}" if ctx_max else "→∞"
            progress.add_task(
                f"Loading model  (ctx={self.n_ctx}{cap_label}, "
                f"gpu_layers={self.n_gpu_layers})",
                total=None,
            )
            self._llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx_max=ctx_max,
                n_ctx_grow=self.n_ctx_grow,
                verbose=False,
            )

        self._loaded = True

        # VRAM usage after load - device-level driver numbers, because
        # torch's allocator counters (memory_allocated/reserved) can only
        # see torch's own allocations and always read 0.00 for llama.dll.
        # "in use" therefore includes every process on the GPU; the delta
        # is what this load itself consumed.
        for i, (free, total) in enumerate(self._vram_levels()):
            used = (total - free) / 1024**3
            line = (f"  vram     : {used:.2f} GB in use / "
                    f"{total / 1024**3:.2f} GB total (device {i}")
            if i < len(vram_before):
                delta = (vram_before[i][0] - free) / 1024**3
                line += f", {delta:+.2f} GB this load"
            console.print(f"[dim]{line})[/dim]")

        console.print("[green]✓[/green] Model loaded")

    def unload(self) -> None:
        # Close explicitly so native teardown (and its stderr suppression)
        # happens now, not whenever the garbage collector gets around to it
        if self._llm is not None and hasattr(self._llm, "close"):
            try:
                self._llm.close()
            except Exception:
                pass
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
        # Not loaded yet - fall back to a chars/4 heuristic
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    # The native ctypes binding (localm.inference.backends.llamacpp) does not
    # expose create_embedding, so this backend cannot produce embeddings. This
    # flag lets callers (the MCP tools/list and /v1/embeddings) decide NOT to
    # advertise an embed capability that would always raise (FAC-6), without
    # having to load a model first.
    can_embed: bool = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._llm:
            raise RuntimeError("Model not loaded - call load() first")
        if not hasattr(self._llm, "create_embedding"):
            raise NotImplementedError(
                "Embeddings are not supported by the built-in GGUF binding yet. "
                "Use a HuggingFace-format model for /v1/embeddings."
            )
        result = self._llm.create_embedding(texts)
        # create_embedding returns {"data": [{"embedding": [...], "index": N}, ...]}
        data = result.get("data", result) if isinstance(result, dict) else result
        return [item["embedding"] for item in data]

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
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # GGUF is text-only here: llama.py keeps only text parts and the
        # mmproj/image path is unimplemented. Refuse an image outright rather
        # than drop it and answer about a picture the model never received.
        from .base import IMAGE_UNSUPPORTED_MESSAGE, UnsupportedInputError, messages_contain_image
        if messages_contain_image(messages):
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)

        # Some native llama.dll builds ship a grammar sampler that faults at
        # sample time (a C++ exception across the C ABI) without harming the
        # loaded model. Once we have seen that on this build, skip grammar
        # up-front and generate unconstrained - the same soft-degrade contract
        # the HF backend offers - so a grammar request never breaks chat.
        if grammar and getattr(self, "_grammar_unsupported", False):
            console.print(
                "[yellow]grammar is not supported by this native llama build; "
                "generating without constraint.[/yellow]"
            )
            grammar = None

        def _make_kwargs(g: Optional[str]) -> dict:
            kw: dict = dict(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                grammar=g,
                stream=True,
            )
            if seed is not None:
                kw["seed"] = seed
            return kw

        def _stream(g: Optional[str]) -> Iterator[str]:
            for chunk in self._llm.create_chat_completion(**_make_kwargs(g)):
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    # "length" = the max_tokens budget ran out mid-reply
                    self.last_finish_reason = choice["finish_reason"]
                token = choice.get("delta", {}).get("content", "")
                if token:
                    yield token

        # native ctypes path (LlamaCpp)
        self.last_finish_reason = "stop"
        yielded = False
        try:
            for token in _stream(grammar):
                yielded = True
                yield token
        except OSError as e:
            # A grammar sampler fault on a build that does not implement it: the
            # model decode context is unharmed (only the sampler threw), so if
            # nothing was emitted yet, remember it and retry once WITHOUT the
            # grammar instead of failing the request.
            if grammar and not yielded:
                self._grammar_unsupported = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("native grammar sampler faulted (%s); "
                             "degrading to unconstrained generation", e)
                console.print(
                    "[yellow]grammar is not supported by this native llama "
                    "build; generating without constraint.[/yellow]"
                )
                self.last_finish_reason = "stop"
                yield from _stream(None)
                return
            # Any other native fault (access violation etc.) leaves the loaded
            # model in an unknown state. Without this, every later request
            # returns an instant empty stream - a zombie server. Drop the broken
            # instance so the next request triggers a clean reload.
            # Mark the reason so the response doesn't report a clean "stop".
            self.last_finish_reason = "error"
            from localm.debuglog import logger as _dbg
            _dbg.exception("native inference fault - dropping model instance")
            try:
                self.unload()
            except Exception:
                self._llm = None
                self._loaded = False
            raise RuntimeError(
                f"Native inference fault ({e}). The model has been unloaded "
                "and will reload on the next request. See the debug log for "
                "the native stack trace."
            ) from e
