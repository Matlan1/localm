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

from localm.vram import VRAM_OVERHEAD_BYTES
from .base import BaseBackend, ModelLoadCancelled

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
        n_gpu_layers_auto: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx_max = n_ctx_max       # ceiling for dynamic growth (0/None = unlimited)
        self.n_ctx_grow = n_ctx_grow
        self.ctx_auto = ctx_auto         # derive n_ctx_max from free VRAM at load
        # Auto-size how many layers go on the GPU from free VRAM at load, but only
        # when n_gpu_layers is left at its "everything" default (see
        # _effective_gpu_layers - an explicit -g is never overridden).
        self.n_gpu_layers_auto = n_gpu_layers_auto
        self.effective_ctx_max: Optional[int] = None   # resolved ceiling of the last load
        self.effective_gpu_layers: Optional[int] = None  # resolved gpu layers of the last load
        self._llm = None
        self._loaded = False
        self._load_cancel = None         # threading.Event to abort a load mid-flight
        # One-time guard for the RAM-offload notice in _check_context_fit: a
        # card-filling model with the default grow step overflows free VRAM on
        # EVERY grow, so without this the "kept in system RAM" warning would repeat
        # on each grow of one conversation. load() clears it, so the hint fires once
        # per loaded-model session even when the SAME backend instance is reloaded
        # (Engine.chat_stream's auto-reload reuses the instance, unlike a fresh
        # switch_engine load).
        self._ram_kv_hint_shown = False

    def set_load_cancel(self, event) -> None:
        """Install (or clear with None) the cancel event honoured by load() via
        llama.cpp's native load-progress callback, so a superseded switch aborts
        the load instead of running it to completion."""
        self._load_cancel = event

    @property
    def can_be_multimodal(self) -> bool:
        """A vision GGUF needs an mmproj; only then is it worth loading the model
        to discover whether vision actually works (the HTTP route uses this to load
        before deciding to reject an image)."""
        return bool(self.mmproj_path)

    @property
    def supports_images(self) -> bool:
        """True once loaded with a working mmproj (mtmd vision, C1)."""
        return bool(self._loaded and self._llm is not None
                    and self._llm.supports_images)

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    # Rough VRAM headroom for KV cache + compute buffers beyond model weights.
    # Single-sourced from localm.vram so the loader, the GUI estimate, and the fit
    # badge all reason about "does it fit" with the same number; kept as a class
    # attribute (not read from the module directly) so tests can monkeypatch it.
    _VRAM_OVERHEAD_BYTES = VRAM_OVERHEAD_BYTES

    @staticmethod
    def _free_total_vram_bytes() -> "tuple[Optional[int], Optional[int]]":
        """(free, total) bytes on the configured main GPU device (device 0 when
        unset - see main_gpu_index / discover.resolve_main_gpu_index), or
        (None, None) when not measurable. Shared by _free_vram_bytes() and
        _total_vram_bytes() so both read the same device in one call."""
        try:
            import torch
            if torch.cuda.is_available():
                from localm.config import load_config
                from localm.discover import resolve_main_gpu_index
                idx = resolve_main_gpu_index(load_config().get("main_gpu_index"))
                free, total = torch.cuda.mem_get_info(idx)
                return int(free), int(total)
        except Exception:
            pass
        return None, None

    @staticmethod
    def _free_vram_bytes() -> Optional[int]:
        """Free VRAM in bytes on the GPU the model runs on, or None when not
        measurable.

        Prefers the ACTIVE ggml backend's own view (loader.gpu_memory): the same
        runtime that allocates the model, so a KV-offload decision budgets against
        the numbers the backend will actually enforce. It also needs no torch, so
        it works on the Vulkan/Metal builds that ship without a CUDA/ROCm torch -
        where torch.cuda.mem_get_info answered nothing and the offload decision was
        blind. Falls back to torch.cuda on the configured main GPU when the native
        backend is not loaded yet (a preflight before the first load), the build
        lacks the query, or a multi-GPU split is configured (torch honours
        main_gpu_index). See loader.gpu_memory for the WDDM committed-budget caveat."""
        try:
            from localm.inference.backends.llamacpp import _loader
            mem = _loader.gpu_memory()
            if mem is not None:
                return mem[0]
        except Exception:
            pass  # backend query unavailable - fall back to the torch reading
        return GgufBackend._free_total_vram_bytes()[0]

    @staticmethod
    def _total_vram_bytes() -> Optional[int]:
        """Total VRAM in bytes on the configured main GPU device, or None when
        not measurable. The hard physical ceiling: unlike free VRAM, nothing
        can be freed to raise it, so a load that needs more than this can
        never fit on this device - see _check_vram()."""
        return GgufBackend._free_total_vram_bytes()[1]

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

    def _vram_holder_hint(self) -> str:
        """Best-effort: name a concrete live sibling localm instance holding
        VRAM on this same GPU device (port, model, how long ago it last
        updated), via the cross-install GPU-coordination registry
        (``localm.gpu_registry``) - instead of the generic "another GPU app"
        guess. Falls back to the generic text when the registry is empty/
        unavailable or the lookup itself fails; this is purely a nicer
        diagnostic (never load-blocking either way).

        Best-effort self-reference note: this does not exclude THIS process's
        own registry entry (the backend layer has no reliable handle on "my
        own instance_id" without importing the HTTP server module, which
        would be a backwards layering dependency). In the rare case this
        model load is itself running inside an advertised server mid model-
        switch, the entry named here could be this same instance's own
        previous state - cosmetically odd but still an accurate snapshot of
        what was last recorded, and never a safety issue since this text is
        advisory-only."""
        try:
            from localm.config import load_config
            from localm.discover import resolve_main_gpu_index
            from localm import gpu_registry
            idx = resolve_main_gpu_index(load_config().get("main_gpu_index"))
            peers = gpu_registry.list_gpu_peers()
            holder = next(
                (p for p in peers
                 if p.get("model") and int(p.get("gpu_index", 0) or 0) == idx),
                None,
            )
            if holder is not None:
                age = gpu_registry.age_seconds(holder.get("updated_at"))
                age_txt = f"{int(age)}s ago" if age is not None else "recently"
                return (
                    f"another localm instance (port {holder.get('port')}) is "
                    f"running '{holder.get('model')}' (active {age_txt}) - "
                    f"POST /v1/models/unload on port {holder.get('port')} to free it."
                )
        except Exception:
            pass  # advisory only - fall through to the generic hint
        return "another GPU app is holding memory (ComfyUI, a browser, another model)."

    @staticmethod
    def _bytes_per_token(model_bytes: int) -> int:
        """KV bytes per token, estimated from the model's size class (larger
        models have more layers and wider KV heads; sliding-window models need
        less, so this stays deliberately conservative). Shared by
        _check_vram()'s preflight KV-cache estimate and _auto_ctx_max()'s
        VRAM-derived ceiling so both reason about a load's KV cost the same
        way."""
        return min(max(model_bytes // 100_000, 16_000), 512_000)

    def _check_vram(self) -> None:
        """
        Warn - loudly and with options - when the model is unlikely to fit in
        currently free VRAM, and refuse outright when even a clean, otherwise-
        empty card could not hold weights plus the requested context's KV
        cache (a "can never fit" case, distinct from "something else is using
        the GPU" - freeing VRAM elsewhere would not help).

        ``need`` includes the KV cache for ``self.n_ctx`` - the base context
        size _load_native() actually passes to context creation, regardless of
        ctx_auto (which only governs the growth ceiling, not this initial
        size). A weights-only estimate stayed silent for a large -c/n_ctx
        request, so a load could pass this check with room to spare yet still
        ask the driver to reserve a KV cache many times bigger than VRAM. On
        ROCm that reservation has been observed to either silently spill into
        slow system memory or crash the GPU driver outright ("unspecified
        launch failure") with nothing surfaced to the user - see
        dev-notes/ for the real-hardware repro (CHK-KVCACHE-OVERFLOW).
        """
        # Use the resolved offload count when load() already picked it (auto), else
        # the configured value (also covers a direct _check_vram() call in tests).
        gpu_layers = (self.effective_gpu_layers
                      if self.effective_gpu_layers is not None
                      else self.n_gpu_layers)
        if gpu_layers == 0:
            return  # CPU-only run, VRAM is irrelevant
        free = self._free_vram_bytes()
        if free is None:
            return  # can't measure (no torch / no GPU) - nothing useful to say
        model_bytes = self._model_bytes()
        kv_cache = self.n_ctx * self._bytes_per_token(model_bytes)
        # Charge only the offloaded fraction of the weights: a partial load
        # (0 < g < 99, whether auto-sized or a user's explicit -g) puts only some
        # layers on the GPU, so it needs far less VRAM than the whole model. A
        # full/"all" load (>= 99) charges the entire weight, as before. This is
        # what lets an auto-sized partial load pass the check instead of being
        # refused "cannot fit regardless".
        if gpu_layers >= self._DEFAULT_GPU_LAYERS:
            weights = model_bytes
        else:
            layers = self._cached_layer_count() or self._ASSUMED_LAYERS
            weights = int(model_bytes * min(1.0, gpu_layers / layers))
        need = weights + kv_cache + self._VRAM_OVERHEAD_BYTES
        ctx_hint = f"weights + a {self.n_ctx:,}-token KV cache + buffers"
        total = self._total_vram_bytes()
        if total is not None and need > total:
            raise RuntimeError(
                f"Context too large for available VRAM: this load needs "
                f"roughly {need / 1024**3:.1f} GB ({ctx_hint}) but this GPU "
                f"only has {total / 1024**3:.1f} GB total - freeing other "
                f"VRAM will not help, it cannot fit regardless.\n"
                f"  Options:\n"
                f"    - Lower the context:  -c 32768  (or smaller)\n"
                f"    - Offload fewer layers:  -g 24  (or -g 0 for CPU-only)\n"
                f"    - Let localm auto-size GPU offload:  "
                f"localm config n_gpu_layers_auto true\n"
                f"    - Let localm auto-size the context:  "
                f"localm config ctx_auto true"
            )
        if free >= need:
            return
        console.print(
            f"[yellow]⚠ Low VRAM:[/yellow] this model needs roughly "
            f"[bold]{need / 1024**3:.1f} GB[/bold] ({ctx_hint}) but only "
            f"[bold]{free / 1024**3:.1f} GB[/bold] is free.\n"
            f"  [dim]Likely cause: {self._vram_holder_hint()}[/dim]\n"
            f"  Options:\n"
            f"    • Free VRAM first (close the other app, or POST "
            f"/v1/models/unload on its server)\n"
            f"    • Lower the context:  [bold]-c 32768[/bold]  (or smaller)\n"
            f"    • Offload fewer layers:  [bold]-g 24[/bold]  "
            f"(or [bold]-g 0[/bold] for CPU-only)\n"
            f"  Continuing anyway - load may be slow or fail."
        )

    def _check_context_fit(self, n_ctx: int, current_ctx: int = 0) -> Optional[bool]:
        """Decide WHERE the KV cache for a context of *n_ctx* tokens must live -
        wired as LlamaCpp's ``vram_check`` hook, consulted by
        ``_prefill_fresh_context()`` before it (re)creates a bigger context.

        Returns True to keep the KV cache in VRAM (``offload_kqv`` - full speed),
        False to place it in SYSTEM RAM (slower, but keeps the FULL context window),
        or None when VRAM is unmeasurable / irrelevant (caller keeps the default,
        VRAM). It NEVER shrinks the window and NEVER raises: when the KV cache does
        not fit free VRAM we move it to RAM and generation runs slower - a degrade,
        not an abort. A model that CAN run always runs; a genuine can't-fit-even-in-
        RAM case is still surfaced by ``_prefill_fresh_context``'s NULL-pointer check
        on the native context (so a real problem is not hidden).

        The charge depends on WHERE the currently-resident KV lives (see the inline
        comment). When it is in VRAM, only the NET growth is charged: recreation frees
        that KV back to VRAM, and ``free`` was measured with it still resident, so
        charging the delta double-counts nothing. When a prior grow already moved the
        KV to system RAM, the GPU holds none, so the FULL target must be charged - else
        the delta undercount would flip it back to VRAM and overflow. Weights and the
        compute buffers are already resident and do not change with the context length
        (n_batch is unchanged), so neither is charged. Historic context: the old hard-refusing version
        false-aborted the first-prompt grow on a model that fills the card yet
        generates fine once the old KV is reclaimed; then a cap-to-fit version still
        shrank the window and errored when a big prompt did not fit. Moving the KV to
        RAM keeps the window and keeps generating (see dev-notes/vram-grow-fail-rootcause.md).
        """
        # Gate on the RESOLVED offload count (auto may have chosen it), not the raw
        # configured n_gpu_layers - a CPU-only auto load (effective 0) still carries
        # n_gpu_layers==99 and would false-act here (twin of _check_vram's resolution).
        gpu_layers = (self.effective_gpu_layers
                      if self.effective_gpu_layers is not None
                      else self.n_gpu_layers)
        if gpu_layers == 0:
            return None  # CPU-only run: KV already lives in RAM, nothing to decide
        free = self._free_vram_bytes()
        if free is None:
            return None  # can't measure (no torch / no GPU) - keep the default (VRAM)
        # KV bytes per token on the GPU: only the offloaded layers keep their KV in
        # VRAM (offload_kqv); CPU layers' KV lives in system RAM. Mirror _check_vram's
        # weight fraction so a partial auto load is charged only its GPU share.
        # Prefer the architecture-accurate per-token KV size computed at load
        # (LlamaCpp.kv_bytes_per_token) over the size-class heuristic, which
        # under-counts wide-KV models badly (~2.6x low on a 12B) and would judge a
        # KV cache that actually overflows VRAM to fit. Fall back to the estimate
        # only when the accurate value is unavailable (model not loaded yet in a
        # direct/test call, or a stripped DLL without the head accessors).
        per_token = (getattr(self._llm, "kv_bytes_per_token", 0)
                     or self._bytes_per_token(self._model_bytes()))
        if gpu_layers < self._DEFAULT_GPU_LAYERS:
            layers = self._cached_layer_count() or self._ASSUMED_LAYERS
            per_token = int(per_token * min(1.0, gpu_layers / layers))
        if per_token <= 0:
            return None
        # How much NEW KV must land in VRAM to grow to n_ctx depends on WHERE the
        # currently-resident KV lives (the recreate frees it first):
        #  - current KV in VRAM (normal): freeing it returns that KV to the VRAM pool,
        #    so only the NET growth over it is a new VRAM charge. `free` was measured
        #    with the old KV still resident, so charging the delta correctly answers
        #    "does the FULL target fit VRAM" (delta <= free  <=>  full target <= budget).
        #  - current KV already in SYSTEM RAM (a prior grow chose offload_kqv=False):
        #    the GPU holds NO KV, so `free` reads the whole KV budget and freeing the
        #    old (RAM) context reclaims RAM, not VRAM. Charging only the delta would
        #    wrongly call it a VRAM fit, flip offload_kqv back to True, and overflow
        #    VRAM with the FULL target -> a NULL context in _prefill_fresh_context ->
        #    an abort of a reply that was generating fine (this defeats #554). So charge
        #    the FULL target: it returns to VRAM only when the whole KV genuinely fits,
        #    otherwise stays in RAM. Never aborts a model that can run.
        # current_ctx==0 means "not told" (a direct/test call) -> the base n_ctx is the
        # safe stand-in (the first grow's real current size).
        kv_in_ram = getattr(self._llm, "_offload_kqv", True) is False
        current = max(int(current_ctx), self.n_ctx)
        charge = (n_ctx * per_token if kv_in_ram
                  else (n_ctx - current) * per_token)   # NET when the old VRAM KV is reclaimed
        if charge <= free:
            return True                                # KV cache fits VRAM - keep it there
        # Does not fit VRAM: keep the FULL window, put the KV cache in system RAM and
        # let generation run slower. A one-time hint (per loaded-model session) so the
        # slowdown is explained without spamming the log on every subsequent grow -
        # each grow of a card-filling model overflows the same way, so repeating it
        # adds noise, not information.
        if not self._ram_kv_hint_shown:
            self._ram_kv_hint_shown = True
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "large context (%s tokens): the KV cache does not fit free VRAM "
                "(need %.2f GB > %.2f GB free), so it is kept in system RAM and generation "
                "will be slower. Free VRAM or lower n_ctx_max for full-speed GPU KV cache.",
                f"{n_ctx:,}", charge / 1024**3, free / 1024**3)
        return False

    # Bounds for VRAM-derived context ceilings
    _AUTO_CTX_MIN = 4096
    _AUTO_CTX_MAX = 65536
    _AUTO_CTX_FALLBACK = 16384   # no GPU visibility - match common practice

    def _auto_ctx_max(self, capped: bool = True) -> int:
        """
        Derive a context ceiling from available resources.

        Budget = free VRAM - model weights - fixed overhead. The KV cost per
        token is estimated from the model's size class (larger models have
        more layers and wider KV heads; sliding-window models need less, so
        the estimate is deliberately conservative). The result is clamped to
        a sane range and rounded to whole KiB of tokens.

        ``capped`` applies the _AUTO_CTX_MAX safety clamp on top of the crude
        KV estimate. It is lifted only when the user explicitly asked for an
        unlimited ceiling (n_ctx_max=0): then "auto" means the full VRAM-derived
        budget, honoring their "grow until VRAM" choice. The _AUTO_CTX_MIN floor
        always applies.
        """
        free = self._free_vram_bytes()
        if free is None:
            return self._AUTO_CTX_FALLBACK
        model = self._model_bytes()
        budget = free - model - self._VRAM_OVERHEAD_BYTES
        if budget <= 0:
            return max(self.n_ctx, self._AUTO_CTX_MIN)
        auto = budget // self._bytes_per_token(model)
        auto = (auto // 1024) * 1024
        hi = auto if not capped else min(self._AUTO_CTX_MAX, auto)
        return int(max(self._AUTO_CTX_MIN, hi))

    def _effective_ctx_max(self) -> Optional[int]:
        """The context ceiling to use for this load (auto or configured).

        ctx_auto sizes the ceiling from free VRAM. n_ctx_max==0 means the user
        asked for NO fixed ceiling ("grow until VRAM"); combined with ctx_auto we
        honor that by lifting the conservative _AUTO_CTX_MAX safety clamp so the
        window can use the full VRAM-derived budget (never silently override an
        explicit user choice). When ctx_auto is off, n_ctx_max is used verbatim
        (0/None already mean unlimited downstream)."""
        if self.ctx_auto:
            unlimited = (self.n_ctx_max == 0)
            auto = self._auto_ctx_max(capped=not unlimited)
            extra = "; no max (n_ctx_max=0)" if unlimited else ""
            console.print(
                f"[dim]  ctx auto : window may grow to {auto:,} tokens "
                f"(from free VRAM{extra})[/dim]"
            )
            return auto
        return self.n_ctx_max

    # The "offload everything" sentinel: n_gpu_layers left at this value is the
    # signal that the user did NOT pin a specific layer count, so auto may size it.
    _DEFAULT_GPU_LAYERS = 99
    # Layer count assumed for a model we have never loaded (true count not cached
    # yet). Deliberately conservative: llama.cpp offloads min(n, real_count), so
    # overshooting is harmless (the extra is clamped - REC-GPULAYERS-CLAMP in
    # llama.py) and undershooting only underuses VRAM; the true count is cached
    # after the first load (model_meta) and used from then on.
    _ASSUMED_LAYERS = 32

    def _cached_layer_count(self) -> Optional[int]:
        """The model's true transformer layer count if a prior load cached it,
        else None (never loaded yet - the caller falls back to _ASSUMED_LAYERS)."""
        from localm.model_meta import cached_n_layers
        return cached_n_layers(self.model_path)

    def _auto_gpu_layers(self) -> Optional[int]:
        """Pick how many layers to offload to the GPU from free VRAM, or None when
        VRAM is not measurable (no torch.cuda: Vulkan/Metal/CPU builds - the caller
        then falls back honestly to the configured value instead of guessing a
        precise offload it could not compute).

        Returns 99 ("all") when the whole model plus its KV cache and overhead fit
        in free VRAM; otherwise the largest layer count whose weight share fits the
        GPU budget left after reserving the KV cache + overhead (conservative: the
        KV cache is charged wholly to the GPU). 0 means even that budget is gone -
        run entirely on CPU, still a working (slow) load, the extreme end of the
        promised RAM offload."""
        free = self._free_vram_bytes()
        if free is None:
            return None                       # unmeasurable - honest fallback (A0)
        model = self._model_bytes()
        if model <= 0:
            return self._DEFAULT_GPU_LAYERS    # can't size - attempt full offload
        kv = self.n_ctx * self._bytes_per_token(model)
        overhead = self._VRAM_OVERHEAD_BYTES
        if free >= model + kv + overhead:
            return self._DEFAULT_GPU_LAYERS    # full offload fits
        weight_budget = free - kv - overhead
        if weight_budget <= 0:
            return 0                           # no room even for one layer's share
        fraction = min(max(weight_budget / model, 0.0), 1.0)
        layers = self._cached_layer_count() or self._ASSUMED_LAYERS
        n = int(fraction * layers)
        return max(0, min(self._DEFAULT_GPU_LAYERS, n))

    def _effective_gpu_layers(self) -> int:
        """The n_gpu_layers this load will actually use.

        Auto only acts when it is ON and the user left n_gpu_layers at the
        "everything" default (99): an explicit value (e.g. -g 24) is honoured
        verbatim so a deliberate choice is never silently overridden (hard-won
        rule: never override a user's explicit selection). When auto sizes a
        partial offload it prints a mandatory one-line notice, since the model
        will run slower with layers on CPU. When VRAM is unmeasurable it says so
        and attempts the configured value rather than faking a precise number."""
        if not self.n_gpu_layers_auto:
            return self.n_gpu_layers
        if self.n_gpu_layers != self._DEFAULT_GPU_LAYERS:
            return self.n_gpu_layers          # explicit choice - respect it as-is
        auto = self._auto_gpu_layers()
        if auto is None:
            # Unmeasurable VRAM (no torch.cuda: Vulkan/Metal/CPU builds) is the
            # NORMAL case on those backends, where offload via the display driver
            # is the right default and the model usually fits fine. Keep it a
            # discoverable debug line, not a per-load console notice that would
            # fire on every load and read oddly on a CPU-only build - if a full
            # offload then does not fit, the native load fails loudly with a VRAM
            # hint (load()'s except handler), so the failure is never hidden.
            from localm.debuglog import logger as _dbg
            _dbg.debug("gpu layers auto: VRAM not measurable; using configured "
                       "n_gpu_layers=%s", self.n_gpu_layers)
            return self.n_gpu_layers
        if auto >= self._DEFAULT_GPU_LAYERS:
            return auto                        # full offload fits - no scary notice
        count = self._cached_layer_count()
        of = f"{count}" if count else f"~{self._ASSUMED_LAYERS} (estimated)"
        console.print(
            f"[yellow]  gpu layers auto:[/yellow] offloading {auto}/{of} layers to "
            f"the GPU, the rest on CPU (model too big for full GPU offload - "
            f"slower). Set n_gpu_layers to override, or n_gpu_layers_auto false."
        )
        return auto

    def load(self) -> None:
        # Fresh loaded-model session: let the RAM-offload notice fire once again
        # (the same instance can be reloaded via Engine.chat_stream's auto-reload).
        self._ram_kv_hint_shown = False
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
        # Resolve the effective GPU-layer count ONCE (it may probe free VRAM and
        # print a notice), then let _check_vram and _load_native both read it - so
        # the preflight and the actual load agree on how many layers go on the GPU.
        self.effective_gpu_layers = self._effective_gpu_layers()
        self._check_vram()
        try:
            self._load_native()
        except ModelLoadCancelled:
            # Deliberate abort (a newer model selection superseded this load).
            # Not a failure - propagate as-is so the caller reports "superseded",
            # never the "native runtime failed to load" error below.
            raise
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
            # load() resolves this before _check_vram; fall back for a direct
            # _load_native() call (e.g. in tests) so the value is never None here.
            gpu_layers = self.effective_gpu_layers
            if gpu_layers is None:
                gpu_layers = self._effective_gpu_layers()
                self.effective_gpu_layers = gpu_layers
            cap_label = f"→{ctx_max}" if ctx_max else "→∞"
            progress.add_task(
                f"Loading model  (ctx={self.n_ctx}{cap_label}, "
                f"gpu_layers={gpu_layers})",
                total=None,
            )
            self._llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=gpu_layers,
                n_ctx_max=ctx_max,
                n_ctx_grow=self.n_ctx_grow,
                mmproj_path=self.mmproj_path,   # C1: in-process vision via mtmd
                cancel_event=self._load_cancel,  # abort mid-load if superseded
                vram_check=self._check_context_fit,  # guard context GROWTH too
                verbose=False,
            )

        self._loaded = True

        # Remember the model's true transformer layer count (read once during the
        # native load - the only place it is knowable) so the next load and the
        # GUI VRAM estimate can size a partial GPU offload precisely instead of
        # from _ASSUMED_LAYERS. Static model metadata, not chat/session content -
        # written regardless of privacy mode (see localm.model_meta).
        n_layers = getattr(self._llm, "n_layers", None)
        if isinstance(n_layers, int) and n_layers > 0:
            from localm.model_meta import store_n_layers
            store_n_layers(self.model_path, n_layers)

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
            except Exception as e:
                # Surface the failed native close: leftover VRAM/context here can
                # make a later load fail mysteriously, so log a correlatable line
                # rather than swallowing it. Teardown is best-effort, so we still
                # drop the instance below instead of escalating to a hard failure.
                from localm.debuglog import logger as _dbg
                _dbg.debug("llama close() failed (%s); context may not be fully freed",
                           type(e).__name__)
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

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with the
        model's embedded chat template."""
        if self._llm is not None:
            try:
                from .llamacpp.llama import _apply_model_template
                # Convert the message content list to text parts if any
                text_messages = []
                for m in messages:
                    content = m.get("content")
                    if isinstance(content, list):
                        text = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    else:
                        text = content or ""
                    text_messages.append({"role": m.get("role", "user"), "content": text})
                prompt = _apply_model_template(self._llm._model_ptr, text_messages)
                bos_markers = ("<bos>", "<s>", "﻿")
                add_bos = not any(prompt.startswith(m) for m in bos_markers)
                return len(self._llm.tokenize(prompt.encode("utf-8"), add_bos=add_bos))
            except Exception:
                pass
        return super().count_messages_tokens(messages)

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
        grammar_lazy: bool = False,
        grammar_triggers: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # Image input: when an mmproj is loaded (mtmd vision, C1) it flows through
        # to create_chat_completion's image path. Otherwise the model is text-only,
        # so refuse the image rather than drop it and answer about a picture the
        # model never received.
        from .base import IMAGE_UNSUPPORTED_MESSAGE, UnsupportedInputError, messages_contain_image
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)

        # Safety net for a native build whose grammar sampler genuinely faults
        # at sample time (a C++ exception across the C ABI) without harming
        # the loaded model: skip grammar up-front once seen and generate
        # unconstrained - the same soft-degrade contract the HF backend
        # offers - so a grammar request never breaks chat. NOTE: the fault
        # this path was originally written for turned out to be OUR bug (a
        # redundant llama_sampler_accept after llama_sampler_sample, fixed in
        # llama.py's _generate) - grammar works on the bundled build now; this
        # stays as a fallback for truly grammar-less builds.
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
                grammar_lazy=grammar_lazy,
                grammar_triggers=grammar_triggers,
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
