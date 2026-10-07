# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF backend - drives our native ctypes wrapper around llama.dll through an
isolated worker PROCESS (see llamacpp/_runner.py and llamacpp/_worker.py).

The model's whole lifecycle (load, generate, tokenize, grammar-check, unload)
runs in a disposable child process, not here: llama_load_model_from_file (and
every later context-grow, which hits the same native call class) can
hard-abort the WHOLE PROCESS on a native CUDA/HIP driver failure, and no Python
try/except can catch that. The isolation boundary wraps the model's ENTIRE
lifecycle, not just the load call: a model goes on to serve many later
requests, context growth is as abort-prone as the initial load, and a
ctypes.c_void_p model/context pointer is meaningless outside the process that
created it. A crash in the child kills only the child; this process reports it
as a clean, catchable error and the backend reloads fresh on the next request.

This class itself (GgufBackend) stays a thin, parent-side proxy: preflight
VRAM sizing (VramSizingMixin, shared with the child) still runs here, before
a child is even spawned, so a load that can never fit still fails fast
without paying a process-spawn cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Optional

from localm.console import console

from .base import (BaseBackend, ModelLoadCancelled, PretokenizerUnsafeInputError,
                   PretokenizerUnusableModelError)
from .llamacpp._runner import RunnerBusy
from .llamacpp._sizing import VramSizingMixin

# Process-wide latch: the count_messages_tokens RPC warning prints once per process.
_count_messages_tokens_rpc_warned = False


# Statuses the child reports once speculation has stopped for the rest of a
# model's life. A per-call status outside this set says nothing about whether
# the model can still speculate: an image turn does not speculate, and that does
# not mean the next text turn will not.
_MTP_STOPPED = frozenset({
    "rewind-unsupported", "draft-context-full", "hidden-state-refused",
    "context-refused", "no-ctx-type-field", "draft-prefill-failed",
    "draft-prefill-error", "draft-trim-error", "no-mtp-graph",
})


def _mtp_status_kind(status) -> str:
    """The part of an MTP status before its ``:`` detail suffix."""
    return str(status or "").split(":", 1)[0]


def _count(value) -> int:
    """A non-negative int from an envelope field, 0 when absent or not a number."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0

# The worker's message when llama_init_from_model returned NULL: the runtime
# loaded the weights, then could not create the context.
_CONTEXT_FAILED_MSG = "Failed to create llama context"


def _load_failure_message(exc: BaseException, hint: str = "") -> str:
    """The user-facing text for a failed native load, from the worker's error
    *exc*, plus *hint* (extra sentences, may be empty).

    Advice to provision or repair the runtime is added only when nothing shows
    the runtime works: a crash the runner placed in context creation, or a
    context the runtime created as NULL after loading the weights, is about the
    model's context size, split or memory instead. A crash report from the
    runner already carries its own advice, so none is appended to it."""
    from .llamacpp._loader import lib_filename
    from .llamacpp._runner import NativeLoadCrashError
    reason = str(exc).strip()
    if reason and reason[-1] not in ".!?":
        reason += "."
    if isinstance(exc, NativeLoadCrashError):
        prefix = ("The model failed to load" if exc.phase == "context"
                  else "Native llama runtime failed to load")
        return f"{prefix}: {reason}{hint}"
    if reason.startswith(_CONTEXT_FAILED_MSG):
        return (f"The model failed to load: {reason} The weights loaded, so the "
                f"runtime itself works; the context did not fit - try a smaller "
                f"context (n_ctx), fewer GPUs (gpu_split_indices) or fewer GPU "
                f"layers (n_gpu_layers).{hint}")
    return (f"Native llama runtime failed to load: {reason}{hint}\n"
            f"Provision or repair it with  localm setup-llama  "
            f"(or set LLAMA_CPP_LIB to a working {lib_filename()}).")


class GgufBackend(VramSizingMixin, BaseBackend):
    """
    Inference backend for GGUF model files.

    Drives our own ctypes binding to llama.dll inside an isolated worker
    process (see the module docstring) - this class never imports LlamaCpp or
    touches a native pointer itself. If the native runtime cannot be loaded,
    load() raises; there is no degraded fallback path.
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
        mtp_draft_tokens: Optional[int] = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        # Opt-in MoE expert placement: keep the expert weights of the first N
        # layers in system RAM (llama.cpp's --n-cpu-moe). 0 = off, the default.
        self.n_cpu_moe = n_cpu_moe
        self.mtp_enabled = mtp_enabled
        self.mtp_draft_tokens = mtp_draft_tokens   # None = the native default
        self.n_ctx_max = n_ctx_max       # ceiling for dynamic growth (0/None = unlimited)
        self.n_ctx_grow = n_ctx_grow
        self.ctx_auto = ctx_auto         # derive n_ctx_max from free VRAM at load
        # Auto-size how many layers go on the GPU from free VRAM at load, only
        # when n_gpu_layers is left at its everything default. An explicit -g is
        # never overridden.
        self.n_gpu_layers_auto = n_gpu_layers_auto
        # Per-instance override of VramSizingMixin's class-level VRAM overhead.
        # None leaves the inherited class attribute untouched.
        if vram_overhead_bytes is not None:
            self._VRAM_OVERHEAD_BYTES = vram_overhead_bytes
        self.effective_ctx_max: Optional[int] = None   # resolved ceiling of the last load
        self.effective_gpu_layers: Optional[int] = None  # resolved gpu layers of the last load
        # The multi-GPU split distribution the last load applied
        # ({"source": "auto"|"pinned"|"equal", "devices": [{"index", "share"}, ...]}),
        # or None when no split applied. Set in _load_native.
        self.applied_gpu_split: Optional[dict] = None
        # The GPU (list_gpus numbering) the last load ran on alone, or None when
        # it did not run on one GPU chosen by a 1-entry gpu_split_indices or the
        # implicit split fit. Set in _load_native.
        self.load_gpu_index: Optional[int] = None
        # How many of the model's transformer layers ended up on the GPU, and the
        # model's true total. None until a load has completed, or when the true
        # layer count is unknowable for this load.
        self.gpu_layers_offloaded: Optional[int] = None
        self.gpu_layers_total: Optional[int] = None
        # The isolated worker process holding the real model. None until load()
        # succeeds.
        self._runner = None
        # True once loaded, from the child's load response.
        self._supports_images = False
        self._supports_mtp = False
        self.last_mtp_status = None    # why speculation is or is not running
        self.last_mtp_active = False   # whether the last call actually speculated
        self.last_mtp_call_status = ""  # why the last call stopped speculating partway
        self.last_mtp_drafted = 0      # draft tokens the last call sent to verification
        self.last_mtp_accepted = 0     # how many of those the target accepted
        self.last_mtp_steps = 0        # verification batches the last call decoded
        self.last_mtp_paused_steps = 0  # steps it ran plain because drafting was slower
        self.last_mtp_skipped = ""     # why the last call could not draft at all
        self._mtp_stopped_this_call = False  # the last call turned MTP off for the model
        # Always None in production; the real LlamaCpp instance lives in the
        # child process.
        self._llm = None
        self._loaded = False
        self._grammar_unsupported = False
        # Latches True the first time this model needs the ChatML fallback, and is
        # not reset by a later load() on this same instance.
        self._chatml_fallback = False
        self._load_cancel = None         # threading.Event to abort a load mid-flight
        # One-time guard for the RAM-offload notice in _check_context_fit.
        # load() clears it, so the hint fires once per loaded-model session.
        self._ram_kv_hint_shown = False
        # The implicit split's unresolved per-device shortfall for the current
        # load, appended to a load-failure message. Set in _load_native.
        self._split_fit_note = ""

    def _report_split_fit(self, plan) -> None:
        """Tell the user what the implicit split's per-device fit decided:
        which devices were left out (and, for a single kept device, that the
        load runs on it alone), or which device is short of memory when no
        plan fits. Devices are named by their torch index
        (``_fit_source_index``), the number ``localm gpus`` and the GUI show.
        Sets ``_split_fit_note`` for the unresolved case."""
        from localm.debuglog import logger as _dbg
        gb = 1024 ** 3
        short = [c for c in plan.default if not c.fits]
        source = getattr(self, "_fit_source_index", None) or {}

        def gpu(i) -> int:
            return source.get(i, i)

        def _what(c) -> str:
            parts = []
            if c.layers:
                parts.append(f"{c.layers} layer{'s' if c.layers != 1 else ''}")
            if c.holds_output:
                parts.append(f"the output layer and a {c.logits / gb:.1f} GB "
                             f"logits buffer")
            return " plus ".join(parts) or "no layers"

        _dbg.info(
            "implicit GPU split fit: %s",
            "; ".join(f"device {gpu(c.index)}: {c.free / gb:.1f} GB free, needs "
                      f"{c.need / gb:.1f} GB ({c.layers} layers "
                      f"{c.weights / gb:.1f}, output {c.output / gb:.2f}, kv "
                      f"{c.kv / gb:.2f}, logits {c.logits / gb:.2f}, reserve "
                      f"{c.reserve / gb:.1f})" for c in plan.default))
        if plan.tensor_split:
            left_out = ", ".join(str(gpu(i)) for i in plan.excluded)
            kept = ", ".join(str(gpu(i)) for i in plan.tensor_split)
            short_ids = {c.index for c in short}
            parts = [f"device {gpu(c.index)} has {c.free / gb:.1f} GB free but would "
                     f"hold about {c.need / gb:.1f} GB ({_what(c)})"
                     for c in short if c.index in plan.excluded]
            parts += [f"device {gpu(i)} would then be short of memory as well"
                      for i in plan.excluded if i not in short_ids]
            detail = "; ".join(parts)
            if len(plan.tensor_split) == 1:
                _dbg.info("implicit GPU split fit: leaving out device(s) %s, "
                          "loading on device %s only", left_out, kept)
                console.print(
                    f"[yellow]  gpu split:[/yellow] loading on device {kept} "
                    f"only - {detail}. Set gpu_split_indices to choose the "
                    f"devices yourself.")
                return
            _dbg.info("implicit GPU split fit: leaving out device(s) %s, "
                      "splitting over %s", left_out, kept)
            console.print(
                f"[yellow]  gpu split:[/yellow] leaving device {left_out} out "
                f"of the split - {detail}. Splitting over devices {kept} "
                f"instead. Set gpu_split_indices to choose the devices yourself.")
            return
        detail = "; ".join(
            f"device {gpu(c.index)} has {c.free / gb:.1f} GB free but the split "
            f"would place about {c.need / gb:.1f} GB on it ({_what(c)})"
            for c in short)
        self._split_fit_note = (
            f"{detail[0].upper()}{detail[1:]}. Set gpu_split_indices to leave "
            f"it out, or lower n_ctx or n_gpu_layers.")
        _dbg.warning("implicit GPU split fit: no split fits every device: %s",
                     detail)
        console.print(
            f"[yellow]⚠ gpu split:[/yellow] {detail}; creating the context may "
            f"fail. Set gpu_split_indices to leave it out, or lower n_ctx or "
            f"n_gpu_layers.")

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
        return bool(self.loaded and self._supports_images)   # a dead worker has no vision

    def _reset_mtp_call(self) -> None:
        """Clear the per-call MTP figures before a call, so a call that ends
        without a done envelope reports none of the previous call's."""
        self.last_mtp_active = False
        self.last_mtp_call_status = ""
        self.last_mtp_drafted = 0
        self.last_mtp_accepted = 0
        self.last_mtp_steps = 0
        self.last_mtp_paused_steps = 0
        self.last_mtp_skipped = ""
        self._mtp_stopped_this_call = False

    def _record_mtp(self, done: dict) -> None:
        """Take the speculation state from one call's done envelope.

        The child can turn speculation off after the load that reported it, so
        the load response goes stale on its own. This is the same latch shape
        grammar_unsupported uses: the child reports this call's outcome, the
        parent owns the state that has to survive across calls.

        A status whose kind (the part before ':') is outside _MTP_STOPPED never re-enables anything - a model that
        stopped speculating does not start again on the next reply.
        """
        if "mtp_status" not in done:
            return          # an older child, or a path that does not report it
        self.last_mtp_status = done.get("mtp_status")
        self.last_mtp_active = bool(done.get("mtp_active"))
        self.last_mtp_call_status = str(done.get("mtp_call_status") or "")
        self.last_mtp_drafted = _count(done.get("mtp_drafted"))
        self.last_mtp_accepted = _count(done.get("mtp_accepted"))
        self.last_mtp_steps = _count(done.get("mtp_steps"))
        self.last_mtp_paused_steps = _count(done.get("mtp_paused_steps"))
        self.last_mtp_skipped = str(done.get("mtp_skipped") or "")
        if _mtp_status_kind(self.last_mtp_status) in _MTP_STOPPED:
            self._mtp_stopped_this_call = self._supports_mtp
            self._supports_mtp = False

    @property
    def last_mtp_usage(self) -> Optional[dict]:
        """MTP figures for the reply that just finished, or None when MTP is
        not enabled for this model.

        ``state`` is "stopped" when the reply stopped speculating partway (the
        reason is the per-call status, or the model status when this reply
        turned speculation off for the model), "paused" when drafting was measured
        slower than one-token decoding for at least as many steps as it ran,
        "on" when it speculated, "unavailable" when the model cannot speculate
        (the reason is the model status), "off" when this reply could not
        draft at all (reason "grammar" for grammar-constrained output, "image"
        for a turn with an image), and "idle" otherwise. ``drafted`` and
        ``accepted`` count draft tokens sent to verification and kept;
        ``paused_steps`` counts the steps run without drafting because it was
        slower.
        """
        if not self.mtp_enabled or not self.loaded:
            return None
        paused = self.last_mtp_paused_steps
        if self.last_mtp_call_status:
            state, reason = "stopped", self.last_mtp_call_status
        elif self._mtp_stopped_this_call:
            state, reason = "stopped", self.last_mtp_status
        elif paused and paused >= self.last_mtp_steps:
            state, reason = "paused", "slower-than-plain"
        elif self.last_mtp_active:
            state, reason = "on", None
        elif not self._supports_mtp:
            state, reason = "unavailable", self.last_mtp_status or None
        elif self.last_mtp_skipped:
            state, reason = "off", self.last_mtp_skipped
        else:
            state, reason = "idle", None
        return {"state": state, "drafted": self.last_mtp_drafted,
                "accepted": self.last_mtp_accepted, "paused_steps": paused,
                "reason": reason}

    @property
    def supports_mtp(self) -> bool:
        """True once loaded with active Multi-Token Prediction heads (MTP).

        A MODEL capability, not a statement about any one request: a turn
        carrying an image never speculates, because the multimodal path has no
        draft context and upstream's own driver skips vision batches for the
        same reason, yet this still reads True there, because the same model
        speculates normally on its text turns. ``last_mtp_active`` is the
        per-call answer.

        It starts from the child's load response and is corrected by every call
        after that, so a session whose speculation has stopped reads False from
        the next reply onward. ``last_mtp_status`` carries the reason, and
        ``last_mtp_active`` says whether the call that just finished actually
        speculated - False on a turn carrying an image even when this is True."""
        return bool(self.loaded and self._supports_mtp)

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #
    # VRAM measurement, preflight checks, and GPU-layer/context auto-sizing
    # (_check_vram, _check_context_fit, _effective_gpu_layers, _effective_ctx_max)
    # are inherited from VramSizingMixin.

    def load(self) -> None:
        # Fresh loaded-model session: re-arm the RAM-offload notice.
        self._ram_kv_hint_shown = False
        # Split GGUF pre-flight: all sibling parts must be present.
        from localm.model_manager import missing_split_parts
        missing = missing_split_parts(Path(self.model_path))
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Split GGUF is incomplete - missing part(s): {names}. "
                f"Re-run 'localm pull' to download all parts."
            )
        # A model whose declared pre-tokenizer cannot hold a conversation is
        # refused here, before any VRAM probe or worker spawn. The worker's own
        # metadata read refuses it again when the header read reports None.
        from localm.inference import pretokenizer_guard
        from localm.model_manager import gguf_pretokenizer
        refusal = pretokenizer_guard.load_refusal(
            gguf_pretokenizer(Path(self.model_path)))
        if refusal is not None:
            raise PretokenizerUnusableModelError(refusal)
        # Resolve the effective GPU-layer count once, so _check_vram and
        # _load_native both read the same value.
        self.effective_gpu_layers = self._effective_gpu_layers()
        self._check_vram()
        from localm.discover import GpuSplitConfigError
        try:
            self._load_native()
        except (ModelLoadCancelled, PretokenizerUnusableModelError,
                GpuSplitConfigError):
            # Propagate as-is, bypassing the load-failure handling below.
            raise
        except Exception as exc:
            vram_hint = ""
            if self.effective_gpu_layers:   # 0 = CPU-only, no VRAM probe needed
                # Combined-when-split free VRAM budget; the helper never raises.
                free, _split_total, _split_devices = self._split_free_total_bytes()
                if free is None:
                    free = self._free_vram_bytes()
                if free is not None and free < self._model_bytes() + self._VRAM_OVERHEAD_BYTES:
                    vram_hint = (
                        " The GPU is low on memory - free VRAM or retry with "
                        "fewer GPU layers (-g 24, or -g 0 for CPU)."
                    )
            split_note = getattr(self, "_split_fit_note", "")
            if split_note:
                vram_hint += f" {split_note}"
            raise RuntimeError(_load_failure_message(exc, vram_hint)) from exc

    def _load_native(self) -> None:
        """Load by spawning an isolated worker process and handing it the
        already-resolved parameters (see the module docstring). Preflight sizing
        (ctx_max/gpu_layers) stays here: none of it touches the abort-prone
        native call, so it can safely run before a child even exists."""
        from localm.inference.backends.llamacpp._runner import ModelRunner
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        # load() resolves this before _check_vram; recompute it for a direct
        # _load_native() call so the value is never None here. Resolved before the
        # VRAM-display read below, which is skipped for a CPU-only load.
        gpu_layers = self.effective_gpu_layers
        if gpu_layers is None:
            gpu_layers = self._effective_gpu_layers()
            self.effective_gpu_layers = gpu_layers

        # VRAM before load, read through discover.list_gpus so it gets the same
        # cross-process correction and trust gate as every other VRAM display.
        # wait_for_inflight=True requires running off the event loop; this method
        # always runs on an executor or CLI thread.
        #
        # Skipped for a CPU-only load (gpu_layers == 0): nothing is placed on a
        # GPU, and this would otherwise be the first VRAM touch of the process.
        from localm.discover import list_gpus
        vram_before, vram_before_status = [], None
        if gpu_layers != 0:
            try:
                vram_before, vram_before_status = list_gpus(
                    return_status=True, wait_for_inflight=True)
            except Exception:
                vram_before, vram_before_status = [], None

        # Resolve the effective split distribution parent-side and pin it into the
        # worker: with gpu_split_ratios unset this is the auto
        # free-VRAM-proportional split, and None when no split is configured, the
        # ratios are pinned, or per-device free is unmeasurable (the worker then
        # keeps the config-driven equal/pinned behavior). A configured split on
        # a box whose integrated GPUs llama.cpp leaves out is renumbered into
        # llama.cpp's devices (configured_split_placement); a 1-entry
        # gpu_split_indices becomes a {llama.cpp device: 1.0} mapping; otherwise
        # the implicit split fit's plan applies. main_gpu_index travels as a
        # llama.cpp device whenever the parent renumbered the placement, and is
        # left to the worker's config read otherwise. By-symbol,
        # function-scoped imports so the resolvers stay patchable.
        # wait_for_inflight=True requires running off the event loop.
        from localm.config import load_config
        from localm.discover import (configured_split_placement,
                                     default_split_main_gpu, fit_main_gpu,
                                     resolve_auto_split_ratios, resolve_gpu_split,
                                     single_gpu_placement)
        auto_ratios = resolve_auto_split_ratios(wait_for_inflight=True)
        worker_split = auto_ratios
        main_gpu = None
        self._split_fit_note = ""
        self._fit_source_index = {}
        self.load_gpu_index = None
        placement = None
        plan = None
        if gpu_layers != 0:
            placement = configured_split_placement(ratios=auto_ratios,
                                                   gpus=vram_before,
                                                   wait_for_inflight=True)
            if placement is None and not auto_ratios:
                placement = single_gpu_placement(wait_for_inflight=True)
        if placement is not None:
            worker_split, main_gpu = placement.mapping, placement.main_gpu
            if len(placement.mapping) == 1:
                from localm.discover import single_gpu_index
                self.load_gpu_index = single_gpu_index(
                    load_config().get("gpu_split_indices"), gpus=[], quiet=True)
        elif not auto_ratios:
            plan = self._implicit_split_fit(gpu_layers)
            if plan is not None and plan.tensor_split:
                worker_split = plan.tensor_split
                self._report_split_fit(plan)
                main_gpu = fit_main_gpu(source_index=self._fit_source_index,
                                        used=list(plan.tensor_split))
                if len(plan.tensor_split) == 1:
                    (slot,) = plan.tensor_split
                    self.load_gpu_index = self._fit_source_index.get(slot, slot)
            elif plan is not None and not plan.default_fits:
                self._report_split_fit(plan)
        if gpu_layers != 0 and placement is None and not isinstance(worker_split, dict):
            main_gpu = default_split_main_gpu(gpus=vram_before, wait_for_inflight=True)

        # The context ceiling is sized from the devices this load uses: a fit
        # that leaves devices out bounds it by the devices it keeps.
        ctx_budget = None
        if plan is not None and plan.tensor_split:
            used = [c for c in plan.default if c.index in plan.tensor_split]
            ctx_budget = (sum(c.free for c in used), len(used))
        ctx_max = self._effective_ctx_max(split_budget=ctx_budget)
        self.effective_ctx_max = ctx_max

        # Record what this load applies, for the GUI's loaded-model status: the
        # auto override when computed, else the config ratios, else equal, through
        # the same resolve_gpu_split validation, normalized to shares. Display data
        # only, never fed back into the load. An invalid pinned config shows its
        # actual fallback (equal) shares under the pinned label.
        cfg = load_config()
        _display_ratios = auto_ratios if auto_ratios else cfg.get("gpu_split_ratios")
        _pairs = resolve_gpu_split(cfg.get("gpu_split_indices"), _display_ratios)
        if placement is not None and not placement.mapping:
            self.applied_gpu_split = None
        elif isinstance(worker_split, dict) and placement is None:
            self.applied_gpu_split = {
                "source": "auto",
                "devices": [{"index": i, "share": s}
                            for i, s in worker_split.items()],
            } if len(worker_split) >= 2 else None
        elif len(_pairs) >= 2:
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
            mmproj_path=self.mmproj_path,       # vision via mtmd, in the child
            n_ctx=self.n_ctx,
            n_gpu_layers=gpu_layers,
            n_ctx_max=ctx_max,
            n_ctx_grow=self.n_ctx_grow,
            # Give the worker's own _check_context_fit the same reserved overhead
            # this parent resolved, rather than the class-level default.
            vram_overhead_bytes=self._VRAM_OVERHEAD_BYTES,
            gpu_split_ratios=worker_split,
            n_cpu_moe=self.n_cpu_moe,
            mtp_enabled=self.mtp_enabled,
        )
        if self.mtp_draft_tokens is not None:
            params["mtp_draft_tokens"] = int(self.mtp_draft_tokens)
        if main_gpu is not None:
            params["main_gpu"] = main_gpu
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
            # cancel_event aborts the load mid-flight if superseded, relayed to the
            # child over its control queue.
            meta = self._runner.spawn_and_load(
                params, cancel_event=self._load_cancel, timeout=timeout)

        self._loaded = True
        self._supports_images = bool(meta.get("supports_images"))
        self._supports_mtp = bool(meta.get("supports_mtp"))

        # Record the model's true transformer layer count, reported once by the
        # child, so the next load and the GUI VRAM estimate can size a partial GPU
        # offload. Static model metadata, written regardless of privacy mode.
        n_layers = meta.get("n_layers")
        if isinstance(n_layers, int) and n_layers > 0:
            from localm.model_meta import store_n_layers
            store_n_layers(self.model_path, n_layers)

        # How many of those layers ended up on the GPU, against the model's true
        # total. gpu_layers is the exact n_gpu_layers this load handed llama.cpp,
        # which offloads min(requested, real layer count).
        total_layers = n_layers if isinstance(n_layers, int) and n_layers > 0 else self._cached_layer_count()
        self.gpu_layers_total = total_layers
        if total_layers:
            self.gpu_layers_offloaded = min(gpu_layers, total_layers)
        elif gpu_layers < self._DEFAULT_GPU_LAYERS:
            self.gpu_layers_offloaded = gpu_layers   # already an absolute count
        else:
            self.gpu_layers_offloaded = None   # "everything" sentinel, true count unknown

        # VRAM usage after load - device-level driver numbers, corrected for
        # cross-process blindness via discover.list_gpus. used/free are shown only
        # when sysstats._vram_reading_trusted says the reading is both fresh and
        # device-global; an untrusted reading shows total-only.
        #
        # Skipped for a CPU-only load: nothing was placed on the GPU.
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
        # report of where each backend's share of the weights landed. Gated on
        # n_cpu_moe > 0, and an empty report is stated rather than shown as
        # nothing. The placement numbers print only in the branch where the
        # override actually applied (skip_reason is None).
        if self.n_cpu_moe > 0:
            # Why the override did not apply, rendered here in the parent from the
            # metadata the child returned. The child must never console.print.
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
        stalled model load has no safe default - see ModelRunner.spawn_and_load,
        which always raises rather than silently reporting not-loaded.
        Configurable because a multi-GB model on a slow disk can legitimately
        take minutes, and that varies far more by install than a fixed constant
        could cover."""
        from localm.inference.backends.llamacpp._runner import LOAD_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_load_timeout_s")
        try:
            return float(raw or LOAD_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value is a misconfiguration, distinct from
            # the benign missing/empty case (None or 0 uses the default above).
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
        partial offload, prompt length). This is not the per-token ceiling - see
        FIRST_TOKEN_TIMEOUT_DEFAULT."""
        from localm.inference.backends.llamacpp._runner import FIRST_TOKEN_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_first_token_timeout_s")
        try:
            return float(raw or FIRST_TOKEN_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value is a misconfiguration, not a silent
            # fall-through.
            from localm.debuglog import logger as _dbg
            _dbg.warning("gguf_first_token_timeout_s is set but not a valid "
                         "number (%r); using the default %.0fs",
                         raw, FIRST_TOKEN_TIMEOUT_DEFAULT)
            return FIRST_TOKEN_TIMEOUT_DEFAULT

    def unload(self) -> None:
        # Ask the isolated worker to close cleanly, killing it if it does not exit
        # promptly. A no-op when the worker already crashed or was never spawned.
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                # Teardown is best-effort: log a correlatable line and drop the
                # reference below rather than escalating to a hard failure.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf worker shutdown failed (%s); its process may "
                           "not be fully torn down", type(e).__name__)
        self._runner = None
        self._llm = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        # A backend whose isolated worker is gone is not loaded, whatever _loaded
        # says: the worker can be killed out from under this object on paths that
        # never run unload(). Reporting the truth makes the next request reload.
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    # llama.cpp applies a GBNF grammar natively in the sampler, so this backend can
    # always honour one. A plain class attribute, shadowing BaseBackend's
    # deny-by-default property.
    supports_grammar: bool = True

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Raise :class:`InvalidGrammarError` for a malformed GBNF string, up front,
        so a bad grammar is a clean 400 rather than a native fault that would latch
        _grammar_unsupported and silently strip grammar from later requests. No-op
        when not loaded (no vocab to parse against) or when *grammar* is empty.

        *lazy* is ACCEPTED AND IGNORED. Accepted because the chat routes pass it
        by keyword on every grammar request, and without it in this signature
        overriding the base method would turn every GGUF grammar request into a
        TypeError. Ignored because this backend has no honest answer to give from
        here: ``_api.has_lazy_grammar()`` is the only probe available and it
        cannot be called in the server parent (it raises RuntimeError when no
        runtime is provisioned, and loads llama.dll into this process when one
        is - see ``BaseBackend.validate_grammar``). A capability claim is
        something callers act on, so answering "supported" here is not an option.

        What silence here costs is EARLINESS only. On a GGUF build lacking the
        native lazy export, ``_build_sampler`` in ``llamacpp/llama.py`` RAISES
        :class:`GrammarUnsupportedError` at generation time instead of dropping
        the grammar and answering with unconstrained text, and ``_runner.py``
        carries that type across the worker IPC as a tagged envelope, so the
        caller gets the same clean 400 an up-front check would have given - one
        request later, and never a reply that silently does not match the
        grammar."""
        if grammar and self.loaded and self._runner is not None:   # the loaded property, not the raw flag
            try:
                self._runner.check_grammar(grammar)
            except RunnerBusy:
                # A generation is streaming on this model and holds the worker's
                # response queue. validate_grammar runs synchronously on the
                # server's event loop and must not block, so the check is deferred:
                # generation raises the same InvalidGrammarError when it builds the
                # sampler.
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
        # self.loaded, not the raw self._loaded: the property is what knows the
        # worker is gone and its queues are nulled.
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                # A generation is streaming and holds the worker's response queue.
                # Fall back to the chars/4 estimate rather than queue behind it.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_tokens: worker busy with a live stream; "
                           "using the chars/4 estimate")
        # Not loaded yet, or the worker is mid-stream - chars/4 heuristic. A genuine
        # RPC failure (worker crash or timeout) propagates instead.
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with
        the model's embedded chat template (an RPC to the isolated worker,
        which alone holds the native model pointer the template needs)."""
        if self.loaded and self._runner is not None:   # the loaded property, not the raw flag
            try:
                return self._runner.count_messages_tokens(messages)
            except RunnerBusy:
                # Worker busy with a live stream - use the heuristic rather than
                # queue behind the generation.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens: worker busy with a live "
                           "stream; using the heuristic estimate")
            except PretokenizerUnsafeInputError:
                # The worker REFUSED this text; it did not fail. Falling back to
                # an estimate would answer a request that must be rejected, and
                # the once-per-process notice below would report a permanent
                # degradation caused by one request.
                raise
            except Exception as e:
                # An unexpected RPC failure (worker crash, timeout, encode error).
                # The super() return below dispatches back onto this class's
                # count_tokens, so the degrade is a real untemplated tokenizer
                # count, or chars/4 only if that also fails. Either way the chat
                # template's own tokens are missing. Logs str(e), not only the
                # exception type.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens RPC failed (%s: %s); "
                           "falling back to an untemplated estimate",
                           type(e).__name__, e)
                global _count_messages_tokens_rpc_warned
                if not _count_messages_tokens_rpc_warned:
                    # One WARNING per process, so a permanently failing RPC is
                    # visible above --debug without flooding the log. Every
                    # occurrence is still logged at --debug above.
                    _count_messages_tokens_rpc_warned = True
                    _dbg.warning(
                        "gguf count_messages_tokens RPC failed (%s: %s); token "
                        "counts are silently falling back to an estimate that "
                        "ignores the chat template, instead of the real "
                        "templated count (this notice prints once per "
                        "process; see --debug for every occurrence)",
                        type(e).__name__, e)
                    # debuglog's file handler only exists under --debug, so the
                    # notice also goes to the console.
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

    # The native ctypes binding does not expose create_embedding, so this backend
    # cannot produce embeddings. Callers read this flag to avoid advertising an
    # embed capability that would always raise, without loading a model first.
    can_embed: bool = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # The worker never exposes create_embedding, so there is no RPC to make.
        # GGUFEmbedder is the real embedding path.
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
        on_status: Optional[Callable[[str], None]] = None,
        thinking: Optional[bool] = None,
    ) -> Iterator[str]:
        # Image input: with an mmproj loaded it flows through to
        # create_chat_completion's image path. A text-only model refuses the image
        # rather than dropping it.
        from .base import IMAGE_UNSUPPORTED_MESSAGE, UnsupportedInputError, messages_contain_image
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)

        # Once a native grammar fault has been seen, skip grammar up-front and
        # generate unconstrained, so a grammar request never breaks chat.
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
        if thinking is not None:
            kwargs["thinking"] = thinking

        # The grammar-fault retry-without-grammar logic runs inside the isolated
        # worker (GgufWorker.chat_stream). This method relays the resulting stream
        # and, on a normal finish, re-applies the worker's report of that decision
        # to this instance's persistent policy state.
        #
        # yield from (not a manual for-loop) is required so that closing THIS
        # generator forwards GeneratorExit into the runner's generator, which is
        # what triggers ModelRunner.chat_stream's cancel-and-drain cleanup.
        self.last_finish_reason = "stop"
        self._reset_mtp_call()
        try:
            yield from self._runner.chat_stream(
                first_chunk_timeout=self._first_token_timeout_seconds(),
                on_status=on_status,
                **kwargs)
        except RuntimeError:
            # The isolated worker crashed or stalled and the model is gone. Drop it
            # so the next request triggers a clean reload.
            self.last_finish_reason = "error"
            from localm.debuglog import logger as _dbg
            # Class-neutral: this except catches every RuntimeError the runner
            # raises - a native fault, an uncaught Python exception in the worker, a
            # generation stall, and an unload racing the stream. exception() carries
            # the real message and traceback.
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
            self._record_mtp(done)
            if done.get("grammar_unsupported") and not self._grammar_unsupported:
                # First time this model has shown it cannot do grammar: latch it, so
                # later calls skip sending grammar at all, and surface the degrade.
                self._grammar_unsupported = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("native grammar sampler faulted; degrading to "
                             "unconstrained generation")
                console.print(
                    "[yellow]grammar is not supported by this native llama "
                    "build; generating without constraint.[/yellow]"
                )
            if done.get("chatml_fallback_reason") and not self._chatml_fallback:
                # First time this model has needed the ChatML fallback: latch it and
                # surface the degrade on the console, not only in the debug log.
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
