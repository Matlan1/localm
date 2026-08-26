# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF backend - drives our native ctypes wrapper around llama.dll through an
isolated worker PROCESS (see llamacpp/_runner.py and llamacpp/_worker.py).

The model's whole lifecycle (load, generate, tokenize, grammar-check, unload)
runs in a disposable child process, not here: llama_load_model_from_file (and
every later context-grow, which hits the same native call class) can
hard-abort the WHOLE PROCESS on a native CUDA/HIP driver failure - no Python
try/except can catch that. Isolating just the load call is not enough (a
model must go on to serve many later requests, and context growth is just as
abort-prone as the initial load), and isolating just a native handle back to
this process is not possible (a ctypes.c_void_p model/context pointer is
meaningless outside the process that created it) - so the isolation boundary
wraps the model's entire lifecycle. A crash in the child kills only the
child; this process reports it as a clean, catchable error and the backend
reloads fresh on the next request, exactly like today's in-process contract.

This class itself (GgufBackend) stays a thin, parent-side proxy: preflight
VRAM sizing (VramSizingMixin, shared with the child) still runs here, before
a child is even spawned, so a load that can never fit still fails fast
without paying a process-spawn cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from localm.console import console

from .base import BaseBackend, ModelLoadCancelled
from .llamacpp._runner import RunnerBusy
from .llamacpp._sizing import VramSizingMixin

# Module-level, not per-backend-instance: a per-instance latch would reset on
# every load(), so a genuinely permanent count_messages_tokens RPC bug would
# re-warn on every model (re)load for the life of a long-running server. One
# WARNING for the whole process is enough to make a stuck chars/4 fallback
# visible - see count_messages_tokens below.
_count_messages_tokens_rpc_warned = False


class GgufBackend(VramSizingMixin, BaseBackend):
    """
    Inference backend for GGUF model files.

    Drives our own ctypes binding to llama.dll inside an isolated worker
    process (see the module docstring) - this class never imports LlamaCpp or
    touches a native pointer itself. If the native runtime cannot be loaded,
    load() raises rather than degrading to a slower, lower-fidelity path.
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
        vram_overhead_bytes: Optional[int] = None,
        n_cpu_moe: int = 0,
        mtp_enabled: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        # Opt-in MoE expert placement: keep the expert weights of the first N
        # layers in system RAM (llama.cpp's --n-cpu-moe). 0 = off, the default.
        self.n_cpu_moe = n_cpu_moe
        self.mtp_enabled = mtp_enabled
        self.n_ctx_max = n_ctx_max       # ceiling for dynamic growth (0/None = unlimited)
        self.n_ctx_grow = n_ctx_grow
        self.ctx_auto = ctx_auto         # derive n_ctx_max from free VRAM at load
        # Auto-size how many layers go on the GPU from free VRAM at load, but only
        # when n_gpu_layers is left at its "everything" default (see
        # _effective_gpu_layers - an explicit -g is never overridden).
        self.n_gpu_layers_auto = n_gpu_layers_auto
        # Per-instance override of VramSizingMixin's class-level default (config's
        # vram_overhead_mb - see config.py's docstring on that key for what this
        # actually funds and the risk of lowering it). None keeps the inherited
        # class attribute untouched, preserving every existing test's
        # monkeypatch-the-class-attribute pattern for callers that never pass this.
        if vram_overhead_bytes is not None:
            self._VRAM_OVERHEAD_BYTES = vram_overhead_bytes
        self.effective_ctx_max: Optional[int] = None   # resolved ceiling of the last load
        self.effective_gpu_layers: Optional[int] = None  # resolved gpu layers of the last load
        # The multi-GPU split distribution the last load actually applied
        # ({"source": "auto"|"pinned"|"equal", "devices": [{"index", "share"},
        # ...]}), or None when no split applied - recorded in _load_native for
        # the GUI's loaded-model status (GET /api/models -> active_gpu_split).
        self.applied_gpu_split: Optional[dict] = None
        # How many of the model's transformer layers actually ended up on the
        # GPU vs its true total, resolved in _load_native from the same
        # ground truth as effective_gpu_layers (see there): None until a load
        # has completed, or when the true layer count is unknowable this load
        # (unmeasurable VRAM with no prior cache - see model_meta). Read by
        # Engine.gpu_placement so a caller can tell a full GPU load from a
        # partial/zero (CPU-fallback) one instead of a bare "loaded" (AGENTS.md
        # rule 5 - a silent downgrade must not report unqualified success).
        self.gpu_layers_offloaded: Optional[int] = None
        self.gpu_layers_total: Optional[int] = None
        # The isolated worker process holding the real model - see
        # llamacpp/_runner.py. None until load() succeeds.
        self._runner = None
        # True once loaded, from the child's load response - supports_images
        # used to read self._llm.supports_images directly; the real LlamaCpp
        # instance now lives in the child, so this is cached instead.
        self._supports_images = False
        self._supports_mtp = False
        # Kept only for the VramSizingMixin test-monkeypatch surface (e.g.
        # test_kv_bytes_offload.py assigns a stub here to drive
        # _check_context_fit directly) - never set to a real object in
        # production; the real LlamaCpp instance lives in GgufWorker, inside
        # the child process, not here.
        self._llm = None
        self._loaded = False
        self._grammar_unsupported = False
        # Latches True the first time this model needed the ChatML fallback
        # (RAG-VISION-1), mirroring _grammar_unsupported immediately above -
        # including that neither is reset on a later load() of a different
        # model on this same backend instance; this instance's OWN persistent
        # policy state is what must survive across many calls (see
        # _grammar_unsupported's own comment further down). See the
        # chatml_fallback_reason handling in chat_stream() below for why this
        # needs surfacing outside --debug.
        self._chatml_fallback = False
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
        """True once loaded with a working mmproj (mtmd vision, C1).

        Cached from the child's load response (self._supports_images) rather
        than read live off a real LlamaCpp instance - that instance now lives
        in the isolated worker process, not here."""
        return bool(self.loaded and self._supports_images)   # property: a dead worker has no vision

    @property
    def supports_mtp(self) -> bool:
        """True once loaded with active Multi-Token Prediction heads (MTP).

        Cached from the child's load response (self._supports_mtp)."""
        return bool(self.loaded and self._supports_mtp)

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #
    # VRAM measurement, preflight checks, and GPU-layer/context auto-sizing
    # (_check_vram, _check_context_fit, _effective_gpu_layers, _effective_ctx_max,
    # etc.) are inherited from VramSizingMixin - see llamacpp/_sizing.py. None of
    # them touch the abort-prone native call, so they stay usable from both this
    # class and GgufWorker (the isolated child) without duplication.

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
            # Same combined-when-split budget as the preflight itself, so this
            # advisory hint never claims "low on memory" against one card of a
            # split whose combined free was fine (helper never raises).
            free, _split_total, _split_devices = self._split_free_total_bytes()
            if free is None:
                free = self._free_vram_bytes()
            vram_hint = ""
            if free is not None and free < self._model_bytes() + self._VRAM_OVERHEAD_BYTES:
                vram_hint = (
                    " The GPU is low on memory - free VRAM or retry with "
                    "fewer GPU layers (-g 24, or -g 0 for CPU)."
                )
            # The isolated worker itself already failed (or crashed) loading
            # the model - fail loud and actionable here too, rather than
            # silently degrading to a slower, lower-fidelity path.
            raise RuntimeError(
                f"Native llama runtime failed to load: {exc}.{vram_hint}\n"
                "Provision or repair it with  localm setup-llama  "
                "(or set LLAMA_CPP_LIB to a working llama.dll)."
            ) from exc

    def _load_native(self) -> None:
        """Load by spawning an isolated worker process and handing it the
        already-resolved parameters (see the module docstring for why this
        runs out-of-process). Preflight sizing (ctx_max/gpu_layers) stays
        here, exactly as when the native call was made in-process - none of
        it touches the abort-prone call, so it can safely run before a child
        even exists."""
        from localm.inference.backends.llamacpp._runner import ModelRunner
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        # load() resolves this before _check_vram; fall back for a direct
        # _load_native() call (e.g. in tests) so the value is never None here.
        # Resolved HERE, before the VRAM-display read below, so that read can
        # be skipped for a CPU-only load (see there for why that matters).
        gpu_layers = self.effective_gpu_layers
        if gpu_layers is None:
            gpu_layers = self._effective_gpu_layers()
            self.effective_gpu_layers = gpu_layers

        # discover.list_gpus (not the raw torch _vram_levels() this line used to
        # call), so the before/after console line gets the SAME cross-process
        # correction and trust gate as every other VRAM display surface (see the
        # end of this method and sysstats._vram_reading_trusted) - see #960: a
        # raw torch read is blind to other processes on Windows + AMD ROCm/HIP,
        # and every GGUF load runs in its OWN isolated worker process, so the
        # raw reading could never see the very model this line is about.
        # wait_for_inflight=True is safe here for the same reason it already is a
        # few lines below for resolve_auto_split_ratios: this always runs off the
        # event loop (an executor/CLI thread), never the server's single loop.
        #
        # SKIPPED for a CPU-only load (gpu_layers == 0), mirroring _check_vram's
        # own "CPU-only run, VRAM is irrelevant" a few lines up the call chain
        # (load() calls _check_vram() before _load_native()). A GPU load always
        # reaches here with torch already resident in this process - _check_vram
        # -> _free_vram_bytes() already paid the GPU driver's one-time init cost,
        # measured live at 0.000s for list_gpus() once that has run. A CPU-only
        # load never touches VRAM anywhere else in the preflight, so without this
        # guard it would be the FIRST VRAM touch of the process and pay the
        # isolated-probe cold cost TWICE (before AND after) for a console line
        # with nothing to report (this load never places anything on a GPU) -
        # measured live at ~2.5-3.4s PER CALL when torch is not yet resident.
        from localm.discover import list_gpus
        vram_before, vram_before_status = [], None
        if gpu_layers != 0:
            try:
                vram_before, vram_before_status = list_gpus(
                    return_status=True, wait_for_inflight=True)
            except Exception:
                vram_before, vram_before_status = [], None

        ctx_max = self._effective_ctx_max()
        self.effective_ctx_max = ctx_max

        # Resolve the effective split distribution HERE, parent-side, and pin
        # it into the worker: with gpu_split_ratios unset this is the auto
        # free-VRAM-proportional split (None when no split is configured, the
        # ratios are pinned by the user, or per-device free is unmeasurable -
        # the worker then keeps the config-driven equal/pinned behavior). The
        # worker cannot resolve this itself: probing there is the Windows+AMD
        # torch DLL conflict (#754/#771), and only the parent has the
        # #697/#700 corrected readings. By-symbol function-scoped import so
        # tests can patch localm.discover.resolve_auto_split_ratios.
        # wait_for_inflight: this runs off the event loop (executor/CLI
        # thread), and a collision with the GUI's stats-heartbeat probe
        # must JOIN rather than decline auto into the equal fallback.
        from localm.config import load_config
        from localm.discover import resolve_auto_split_ratios, resolve_gpu_split
        auto_ratios = resolve_auto_split_ratios(wait_for_inflight=True)

        # Record what this load will actually apply, for the GUI's
        # loaded-model status (GET /api/models -> active_gpu_split): the
        # parent-side mirror of the worker's own apply_gpu_split resolution
        # (auto override when computed, else the config ratios, else equal -
        # through the same resolve_gpu_split validation), normalized to
        # shares. Display data only, never fed back into the load. An INVALID
        # pinned config (length mismatch) shows its actual fallback (equal)
        # shares under the "pinned" label - the shares never lie, and
        # resolve_gpu_split's WARNING names the misconfig.
        cfg = load_config()
        _display_ratios = auto_ratios if auto_ratios else cfg.get("gpu_split_ratios")
        _pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), _display_ratios)
        if len(_pairs) >= 2:
            _total = sum(r for _, r in _pairs) or 1.0
            self.applied_gpu_split = {
                "source": ("auto" if auto_ratios
                           else "pinned" if cfg.get("gpu_split_ratios")
                           else "equal"),
                "devices": [{"index": i, "share": r / _total}
                            for i, r in _pairs],
            }
        else:
            self.applied_gpu_split = None

        params = dict(
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,       # C1: vision via mtmd, in the child
            n_ctx=self.n_ctx,
            n_gpu_layers=gpu_layers,
            n_ctx_max=ctx_max,
            n_ctx_grow=self.n_ctx_grow,
            # So the worker's own _check_context_fit (mid-session context grow,
            # the ONLY VramSizingMixin method it calls - see the module docstring)
            # uses the SAME reserved overhead this parent already resolved for the
            # initial load, not the class-level default: an instance-level
            # override here would otherwise silently not reach the child process.
            vram_overhead_bytes=self._VRAM_OVERHEAD_BYTES,
            gpu_split_ratios=auto_ratios,
            n_cpu_moe=self.n_cpu_moe,
            mtp_enabled=self.mtp_enabled,
        )
        timeout = self._load_timeout_seconds()

        cap_label = f"→{ctx_max}" if ctx_max else "→∞"
        self._runner = ModelRunner()
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task(
                f"Loading model  (ctx={self.n_ctx}{cap_label}, "
                f"gpu_layers={gpu_layers})",
                total=None,
            )
            # cancel_event abort mid-load if superseded (relayed to the child
            # over its control queue - see ModelRunner.spawn_and_load).
            meta = self._runner.spawn_and_load(
                params, cancel_event=self._load_cancel, timeout=timeout)

        self._loaded = True
        self._supports_images = bool(meta.get("supports_images"))
        self._supports_mtp = bool(meta.get("supports_mtp"))

        # Remember the model's true transformer layer count (reported once by
        # the child, the only place it is knowable) so the next load and the
        # GUI VRAM estimate can size a partial GPU offload precisely instead of
        # from _ASSUMED_LAYERS. Static model metadata, not chat/session content -
        # written regardless of privacy mode (see localm.model_meta).
        n_layers = meta.get("n_layers")
        if isinstance(n_layers, int) and n_layers > 0:
            from localm.model_meta import store_n_layers
            store_n_layers(self.model_path, n_layers)

        # Ground truth for how many of those layers actually ended up on the
        # GPU this load, vs the model's true total, so a caller (HTTP
        # /v1/models/load, GUI, MCP) can tell a full GPU load from a silent
        # CPU fallback (AGENTS.md rule 5) instead of a bare "loaded" that
        # hides it. `gpu_layers` is the exact n_gpu_layers this load handed
        # llama.cpp (resolved above, before the native call); llama.cpp
        # offloads min(requested, real layer count), so once the load has
        # succeeded that arithmetic against the now-known true count is the
        # ground truth, not a re-guess.
        total_layers = n_layers if isinstance(n_layers, int) and n_layers > 0 else self._cached_layer_count()
        self.gpu_layers_total = total_layers
        if total_layers:
            self.gpu_layers_offloaded = min(gpu_layers, total_layers)
        elif gpu_layers < self._DEFAULT_GPU_LAYERS:
            self.gpu_layers_offloaded = gpu_layers   # already an absolute count
        else:
            self.gpu_layers_offloaded = None   # "everything" sentinel, true count unknown

        # VRAM usage after load - device-level driver numbers, corrected for
        # cross-process blindness via discover.list_gpus (see the top of this
        # method for why, and #960). used/free are shown only when
        # sysstats._vram_reading_trusted says the reading is BOTH fresh (a probe
        # actually completed, not a served last-known-good value) and
        # device-global (counts every process's VRAM, not just this one) - the
        # exact gate the GUI/API status bar already uses, reused here rather
        # than inventing a second one. An untrusted reading shows total-only:
        # printing a used/free number we cannot stand behind is the AGENTS.md
        # rule 5 failure #960 was filed over (0.14 GB "in use" shown next to a
        # comment block that already knew it was wrong).
        #
        # torch's allocator counters (memory_allocated/reserved) are a different
        # thing again: they see only torch's own allocations and always read 0.00 for
        # llama.dll, which is why they are not used here.
        #
        # SKIPPED for a CPU-only load - see the vram_before guard above for the
        # cost reasoning; nothing was placed on the GPU this load, so there is
        # nothing this line could report anyway.
        if gpu_layers != 0:
            from localm.sysstats import _vram_reading_trusted
            try:
                vram_after, vram_after_status = list_gpus(
                    return_status=True, wait_for_inflight=True)
            except Exception:
                vram_after, vram_after_status = [], None
            for i, gpu in enumerate(vram_after):
                total = gpu.get("total")
                if not total:
                    continue
                if _vram_reading_trusted(gpu, vram_after_status):
                    free = gpu["free"]
                    used = (total - free) / 1024**3
                    line = (f"  vram     : {used:.2f} GB in use / "
                            f"{total / 1024**3:.2f} GB total (device {i}")
                    before = vram_before[i] if i < len(vram_before) else None
                    if before is not None and _vram_reading_trusted(before, vram_before_status):
                        delta = (before["free"] - free) / 1024**3
                        line += f", {delta:+.2f} GB this load"
                    line += ")"
                else:
                    line = (f"  vram     : {total / 1024**3:.2f} GB total (device {i}"
                            f", used/free reading not trusted on this platform)")
                console.print(f"[dim]{line}[/dim]")

        # MoE expert placement (opt-in, n_cpu_moe): llama.cpp's own load_tensors
        # report of where each backend's share of the weights landed - the only
        # ground truth for whether n_cpu_moe actually moved anything (see
        # GgufWorker.load's docstring; there is no llama.h API for this, only
        # this captured printf). Gated on n_cpu_moe>0 so an ordinary load stays
        # as quiet as before. An empty report is stated honestly rather than
        # silently showing nothing: a user who set n_cpu_moe and sees no
        # confirmation cannot otherwise tell "it worked but there is nothing to
        # show" from "it silently did nothing" (AGENTS.md rule 5). When
        # moe_skip_reason is set the override never ran at all, so the
        # placement numbers below describe an ordinary, unrelated load - they
        # are only printed in the branch where the override actually applied
        # (skip_reason is None); printing them alongside a skip message would
        # read as proof the setting moved something when it did nothing here.
        if self.n_cpu_moe > 0:
            # Why the override did NOT apply (no experts on this model, or the
            # CPU buffer type could not be resolved) - rendered HERE, in the
            # parent, from the fact _apply_cpu_moe recorded and carried out of
            # the isolated child via this same metadata dict. That function
            # runs inside the child and must never console.print directly (its
            # stdout is not this process's own console - see
            # isolated-child-must-not-console-print); this is the one place
            # allowed to.
            skip_reason = meta.get("moe_skip_reason")
            if skip_reason is not None:
                from .llamacpp.llama import MOE_SKIP_MESSAGES
                console.print(MOE_SKIP_MESSAGES.get(
                    skip_reason,
                    f"[yellow]  n_cpu_moe:[/yellow] did not apply ({skip_reason})."))
            else:
                placement = meta.get("weight_placement") or []
                if placement:
                    ram_mib = sum(b["mib"] for b in placement if b["is_ram"])
                    vram_mib = sum(b["mib"] for b in placement if not b["is_ram"])
                    console.print(
                        f"[dim]  moe placement: {ram_mib:.2f} MiB system RAM / "
                        f"{vram_mib:.2f} MiB VRAM across {len(placement)} backend "
                        f"buffer(s) (n_cpu_moe={self.n_cpu_moe})[/dim]")
                else:
                    console.print(
                        "[dim]  moe placement: not reported by this llama.cpp "
                        f"build (n_cpu_moe={self.n_cpu_moe} was still "
                        "requested)[/dim]")

        console.print("[green]✓[/green] Model loaded")

    @staticmethod
    def _load_timeout_seconds() -> float:
        """Model-load timeout, from config (``gguf_load_timeout_s``) or the
        generous built-in default. Unlike the VRAM-probe daemon's short
        bounded wait (which has a safe "unmeasurable, skip" fallback), a
        stalled model load has no safe default - see ModelRunner.spawn_and_load
        for why this always raises rather than silently reporting not-loaded.
        Configurable because a multi-GB model on a slow disk can legitimately
        take minutes, and that varies far more by install than a fixed
        constant could ever cover."""
        from localm.inference.backends.llamacpp._runner import LOAD_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_load_timeout_s")
        try:
            return float(raw or LOAD_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value (e.g. "abc", a list) is a real
            # misconfiguration, distinct from the benign missing/empty case
            # (None/0 -> the default above with no exception). Surface it under
            # --debug rather than silently masking a typo'd config as if the
            # user had simply not set it (rule 5).
            from localm.debuglog import logger as _dbg
            _dbg.warning("gguf_load_timeout_s is set but not a valid number "
                         "(%r); using the default %.0fs", raw, LOAD_TIMEOUT_DEFAULT)
            return LOAD_TIMEOUT_DEFAULT

    @staticmethod
    def _first_token_timeout_seconds() -> float:
        """How long to wait for a reply's FIRST token, from config
        (``gguf_first_token_timeout_s``) or the generous built-in default.
        Configurable for the same reason as the load timeout above: it covers
        prompt PREFILL, whose duration varies enormously by install (CPU vs GPU,
        partial offload, prompt length) - far more than a fixed constant could
        cover. See FIRST_TOKEN_TIMEOUT_DEFAULT for why this is not the per-token
        ceiling."""
        from localm.inference.backends.llamacpp._runner import FIRST_TOKEN_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_first_token_timeout_s")
        try:
            return float(raw or FIRST_TOKEN_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # Same rule-5 reasoning as _load_timeout_seconds above: a typo'd
            # value is a real misconfiguration, not a silent fall-through.
            from localm.debuglog import logger as _dbg
            _dbg.warning("gguf_first_token_timeout_s is set but not a valid "
                         "number (%r); using the default %.0fs",
                         raw, FIRST_TOKEN_TIMEOUT_DEFAULT)
            return FIRST_TOKEN_TIMEOUT_DEFAULT

    def unload(self) -> None:
        # Ask the isolated worker to close cleanly (native teardown + its
        # stderr suppression happen there), killing it if it does not exit
        # promptly. Safe to call when the worker already crashed or was never
        # spawned - ModelRunner.shutdown() is a no-op in both cases.
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                # Leftover VRAM/context here can make a later load fail
                # mysteriously, so log a correlatable line rather than
                # swallowing it. Teardown is best-effort, so we still drop the
                # reference below instead of escalating to a hard failure.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf worker shutdown failed (%s); its process may "
                           "not be fully torn down", type(e).__name__)
        self._runner = None
        self._llm = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        # A backend whose isolated worker is GONE is not loaded, whatever
        # _loaded says. The worker can be killed out from under this object on
        # paths that never run unload(): _cancel_stream_and_drain kills it when
        # a mid-stream cancel is not confirmed within its drain timeout, and
        # _simple_request kills it on its own timeout - both call
        # ModelRunner.shutdown(), which nulls the runner's queues and process.
        # Neither raises RuntimeError into chat_stream's handler below
        # (GeneratorExit is not a RuntimeError), so _loaded stayed True next to
        # a dead runner. Engine.chat_stream then skipped its auto-reload and
        # called straight into the dead runner, whose first act is
        # self._req_q.put(...) on None -> AttributeError, which is not caught or
        # unloaded either, so EVERY later request to this model repeated it
        # until the model was manually evicted or the server restarted.
        # Reporting the truth here makes the next request reload cleanly instead
        # (load() builds a fresh ModelRunner) - REG-606.
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    # llama.cpp applies a GBNF grammar natively in the sampler, with no optional
    # dependency behind it, so this backend can always honour one. Declared as a
    # plain class attribute (shadowing BaseBackend's deny-by-default property)
    # because the answer is fixed for every GGUF model and every install.
    supports_grammar: bool = True

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Raise :class:`InvalidGrammarError` for a malformed GBNF string, up front,
        so a bad grammar is a clean 400 rather than a native fault that would latch
        _grammar_unsupported and silently strip grammar from later requests. No-op
        when not loaded (no vocab to parse against) or when *grammar* is empty.

        *lazy* is ACCEPTED AND IGNORED, and both halves are deliberate. Accepted
        because the chat routes now pass it by keyword on every grammar request:
        without it in this signature, overriding the base method would turn every
        GGUF grammar request into a TypeError. Ignored because this backend has no
        honest answer to give from here. ``_api.has_lazy_grammar()`` is the only
        probe available and it cannot be called in the server parent (it raises
        RuntimeError when no runtime is provisioned, and loads llama.dll into this
        process when one is - see ``BaseBackend.validate_grammar`` for the
        measurement). Answering "supported" here to look complete would be the
        worse of the two, because a capability claim is something callers act on.

        What silence here costs is now only EARLINESS. A GGUF build lacking the
        native lazy export used to DROP the grammar at generation time behind a
        DEBUG line and answer with unconstrained text; ``_build_sampler`` in
        ``llamacpp/llama.py`` now RAISES :class:`GrammarUnsupportedError` there
        instead, and ``_runner.py`` carries that type across the worker IPC as a
        tagged envelope, so the caller gets the same clean 400 it would have got
        from an up-front check - one request later, and never a reply that
        silently does not match the grammar. Evidence:
        ``dev-notes/lazy-grammar-silent-unconstrained-2026-08-12.md``."""
        if grammar and self.loaded and self._runner is not None:   # property, see count_tokens
            try:
                self._runner.check_grammar(grammar)
            except RunnerBusy:
                # A generation is streaming on this model right now and holds the
                # worker's response queue. validate_grammar is called
                # synchronously on the server's async event loop (before the
                # per-model inference semaphore), so blocking here would freeze
                # the whole loop for the length of that generation (HON-02
                # review finding). Defer the check: when generation builds the
                # sampler it rejects a malformed grammar with the SAME clean
                # InvalidGrammarError (llama.py raises it on the native NULL
                # return, before any token) - so deferring only moves a bad
                # grammar from an up-front 400 to a generation-time error, never
                # a native fault or a latched degrade.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf validate_grammar: worker busy with a live "
                           "stream; deferring grammar validation to generation time")

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded model's vocabulary (an
        RPC to the isolated worker), or the chars/4 heuristic when the worker
        is busy streaming or the model is not loaded yet."""
        # `self.loaded`, NOT the raw `self._loaded`: the property is what knows the
        # worker is gone. _simple_request kills it on its own timeout (and the
        # cancel-drain does the same), nulling the queues while _loaded stays True -
        # so gating on the attribute here called straight into a dead runner and
        # `self._req_q.put(...)` raised AttributeError on None, which is not
        # RunnerBusy and so escaped uncaught, on every later count (REG-606).
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                # A generation is streaming on this model right now and holds
                # the worker's response queue (HON-02). Rather than block this
                # request's token accounting behind a whole generation - or risk
                # its RPC reply racing the live stream's envelopes - fall back to
                # the documented chars/4 estimate immediately and say so under
                # --debug.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_tokens: worker busy with a live stream; "
                           "using the chars/4 estimate")
        # Not loaded yet, or the worker is mid-stream - chars/4 heuristic. A
        # genuine RPC failure (worker crash/timeout) is NOT swallowed here: it
        # propagates, preserving this method's existing contract.
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with
        the model's embedded chat template (an RPC to the isolated worker,
        which alone holds the native model pointer the template needs)."""
        if self.loaded and self._runner is not None:   # the property - see count_tokens
            try:
                return self._runner.count_messages_tokens(messages)
            except RunnerBusy:
                # Worker busy with a live stream (HON-02) - use the heuristic
                # rather than queue behind the generation. See count_tokens.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens: worker busy with a live "
                           "stream; using the heuristic estimate")
            except Exception as e:
                # An unexpected RPC failure (worker crash/timeout, an encode
                # error): the super() return below calls self.count_tokens(text)
                # (BaseBackend.count_messages_tokens dispatches polymorphically
                # onto THIS class, not BaseBackend's own count_tokens) - so the
                # degrade is actually GgufBackend.count_tokens's own fallback
                # chain: a real, untemplated tokenizer count when the worker can
                # still answer plain count_tokens, or the chars/4 heuristic only
                # if that ALSO fails. Either way it is missing the chat
                # template's own tokens, so context-budgeting downstream is
                # trusting an approximation. Log str(e), not just the exception
                # TYPE - logging only type(e).__name__ is what let a permanent
                # bytes/str bug in this exact RPC (#956) run silently on every
                # request, forever, with nothing but the useless word
                # "RuntimeError" in the log to go on.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens RPC failed (%s: %s); "
                           "falling back to an untemplated estimate",
                           type(e).__name__, e)
                global _count_messages_tokens_rpc_warned
                if not _count_messages_tokens_rpc_warned:
                    # A permanently failing RPC (a code bug, not a transient
                    # worker hiccup) would otherwise degrade every prompt-token
                    # count for the rest of the process without ever saying so
                    # above --debug (rule 5). One WARNING per process is enough
                    # to surface it without flooding the log on a chat
                    # session's per-request cadence - every occurrence is still
                    # logged at --debug above.
                    _count_messages_tokens_rpc_warned = True
                    _dbg.warning(
                        "gguf count_messages_tokens RPC failed (%s: %s); token "
                        "counts are silently falling back to an estimate that "
                        "ignores the chat template, instead of the real "
                        "templated count (this notice prints once per "
                        "process; see --debug for every occurrence)",
                        type(e).__name__, e)
                    # The comment above already says the intent is to surface
                    # this "above --debug" - but debuglog.logger's file handler
                    # only exists under --debug, so a real user never saw it
                    # (same bug class as the ChatML fallback fix above).
                    console.print(
                        "[yellow]token counts are falling back to an estimate "
                        "that ignores the chat template (a worker request "
                        "failed); context budgeting may be less accurate than "
                        "usual for the rest of this session[/yellow]"
                    )
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
        if not self._loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # The isolated worker's GgufWorker never exposes create_embedding (the
        # ctypes binding does not implement it) - this call always raises, so
        # there is no RPC to make; see can_embed above and the GGUFEmbedder
        # class (localm/inference/embedder.py) for the real embedding path.
        raise NotImplementedError(
            "Embeddings are not supported by the built-in GGUF binding yet. "
            "Use a HuggingFace-format model for /v1/embeddings."
        )

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

        kwargs: dict = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            grammar=grammar,
            grammar_lazy=grammar_lazy,
            grammar_triggers=grammar_triggers,
        )
        if seed is not None:
            kwargs["seed"] = seed

        # The grammar-fault retry-without-grammar logic (a genuinely
        # recoverable native OSError, not a process abort) now runs INSIDE the
        # isolated worker (GgufWorker.chat_stream) - it already has the real
        # model, so retrying there is a plain in-process call, no round trip.
        # This method just relays the resulting stream and, on a normal
        # finish, re-applies the worker's report of that decision to this
        # instance's OWN persistent policy state (_grammar_unsupported must
        # survive across many calls on the same backend, so it stays here).
        #
        # yield from (not a manual for-loop) is required so that closing THIS
        # generator (http_server.py's mid-stream cancel, unchanged) correctly
        # forwards GeneratorExit into the runner's generator too - that is
        # what triggers ModelRunner.chat_stream's own cancel-and-drain cleanup
        # instead of abandoning the runner's generator mid-flight.
        self.last_finish_reason = "stop"
        try:
            yield from self._runner.chat_stream(
                first_chunk_timeout=self._first_token_timeout_seconds(), **kwargs)
        except RuntimeError:
            # The isolated worker crashed or stalled (a real native abort, or
            # an unrecoverable fault GgufWorker.chat_stream deliberately left
            # uncaught) - the model is gone; without this, every later
            # request would hit a dead process. Drop it so the next request
            # triggers a clean reload, matching the in-process contract this
            # replaces exactly (same user-facing message and effect).
            self.last_finish_reason = "error"
            from localm.debuglog import logger as _dbg
            # Deliberately class-NEUTRAL, and it used to say "native inference
            # fault". This except catches EVERY RuntimeError the runner raises -
            # a genuine native fault, an uncaught Python exception in the worker
            # (exit 1), a generation stall, and an unload racing the stream - so
            # naming one of them was wrong for the other three. It is also the
            # line a reader greps first in a field log (it is the ERROR line in
            # issues 1222/1223), which is exactly why a false label here is
            # expensive: it pre-classifies the crash before anyone has looked.
            # exception() carries the real message and traceback, and the runner's
            # own message now states the class only when the exit code or a
            # captured trace establishes it (see _mp_spawn.death_was_a_native_fault).
            _dbg.exception("worker failure during generation - dropping model instance")
            try:
                self.unload()
            except Exception:
                self._runner = None
                self._llm = None
                self._loaded = False
            raise
        else:
            done = getattr(self._runner, "last_done", None) or {}
            self.last_finish_reason = done.get("finish_reason", "stop")
            if done.get("grammar_unsupported") and not self._grammar_unsupported:
                # First time this model has shown it can't do grammar: latch
                # it (future calls skip sending grammar at all, see the
                # up-front check above) and surface the same degrade notice
                # the in-process path always printed - a degrade must stay
                # visible even after crossing a process boundary.
                self._grammar_unsupported = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("native grammar sampler faulted; degrading to "
                             "unconstrained generation")
                console.print(
                    "[yellow]grammar is not supported by this native llama "
                    "build; generating without constraint.[/yellow]"
                )
            if done.get("chatml_fallback_reason") and not self._chatml_fallback:
                # First time this model has shown it needs the ChatML
                # fallback (RAG-VISION-1): latch it and surface the same
                # degrade notice llama.py's _apply_model_template always
                # logged - a degrade must stay visible even after crossing a
                # process boundary, exactly like the grammar_unsupported case
                # immediately above. Previously this only reached
                # debuglog.logger, which is invisible without --debug
                # (AGENTS.md rule 5: the whole point was to surface it).
                self._chatml_fallback = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("chat template not recognized by llama.cpp's "
                             "built-in matcher (%s); falling back to a "
                             "generic ChatML prompt",
                             done["chatml_fallback_reason"])
                console.print(
                    "[yellow]this model's chat template is not recognized; "
                    "falling back to a generic prompt format it may not "
                    "understand - chat and vision output quality may be "
                    "degraded[/yellow]"
                )
