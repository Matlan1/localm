# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM measurement and load-sizing logic shared by ``GgufBackend`` (the
parent-facing proxy) and ``GgufWorker`` (the child process that owns the real
native model).

- ``GgufBackend`` uses the whole mixin: preflight VRAM checks and
  GPU-layer/context-size sizing all happen BEFORE a model process is spawned.
- ``GgufWorker`` uses only ``_check_context_fit`` (and its transitive
  dependencies), as the ``vram_check`` callback
  ``LlamaCpp._prefill_fresh_context`` calls during a mid-generation context
  grow. It runs wherever the loaded model itself lives.

None of these methods call ``llama_load_model_from_file``/
``llama_init_from_model``/``llama_decode``; they only call
``torch.cuda.mem_get_info`` or the subprocess-isolated
``loader.gpu_memory_isolated()`` (see ``_loader.py``).

Every entry into torch from here is both latch-guarded and deadline-bounded;
see ``_free_total_vram_bytes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

from localm.console import console
from localm.vram import VRAM_OVERHEAD_BYTES


def embedder_ctx_reservation_bytes() -> int:
    """VRAM to hold back from the auto-sized context budget for the CONFIGURED
    embedding model, so the chat model's window does not claim the room the
    embedder will need when it loads later (first memory/RAG use).

    Returns the embedder's expected footprint (file size + 20% KV/compute
    slop), or 0 when the embedder is ALREADY loaded (the free reading already
    pays for it), no embedding model resolves, or anything fails. Never
    raises; a failure is surfaced at debug.
    """
    try:
        from localm.inference import embedder as _emb
        if _emb.loaded_path() is not None:
            return 0
        path = _emb.resolve_embedding_model_path(allow_download=False)
        if not path:
            return 0
        return int(Path(path).stat().st_size * 1.2)
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("embedder ctx reservation unavailable (%s); reserving "
                   "nothing", type(e).__name__)
        return 0


# Free system RAM an auto full-offload load must have beyond its host-resident
# weights for them to be read into memory instead of memory-mapped.
HOST_RAM_HEADROOM_BYTES = 2 * 1024 ** 3


class MmapDecision(NamedTuple):
    """How a load memory-maps the model file (:func:`decide_use_mmap`).

    ``use_mmap`` is True to force mmap, False to force it off, None to keep the
    runtime's own default (mmap on every device that supports it). ``reason``
    is one of ``"user_on"``, ``"user_off"``, ``"partial_offload"``,
    ``"no_gpu"``, ``"fits_ram"``, ``"exceeds_ram"``, ``"ram_unknown"`` or
    ``"host_unknown"``. ``host_bytes`` is the weight bytes the load keeps in
    system RAM and ``ram_total``/``ram_available`` the RAM reading, each None
    when not known or not needed."""
    use_mmap: Optional[bool]
    reason: str
    host_bytes: Optional[int] = None
    ram_total: Optional[int] = None
    ram_available: Optional[int] = None


def decide_use_mmap(setting, full_offload: bool, host_bytes: Optional[int],
                    ram_total: Optional[int], ram_available: Optional[int]) -> MmapDecision:
    """The mmap mode for one load.

    *setting* is the configured ``use_mmap``: ``"on"`` or ``"off"`` (any case)
    forces that mode; anything else is ``auto``. Under auto:

    - a load that does not put every layer on the GPU (*full_offload* False)
      keeps the runtime default;
    - a full-offload load reads its host-resident weights into memory (mmap
      off) when *host_bytes* plus :data:`HOST_RAM_HEADROOM_BYTES` fits
      *ram_available*, and keeps the runtime default when it does not, when
      *ram_available* is None, or when *host_bytes* is None.
    """
    mode = setting.strip().lower() if isinstance(setting, str) else ""
    if mode == "on":
        return MmapDecision(True, "user_on")
    if mode == "off":
        return MmapDecision(False, "user_off")
    if not full_offload:
        return MmapDecision(None, "partial_offload")
    if host_bytes is None:
        return MmapDecision(None, "host_unknown", None, ram_total, ram_available)
    if ram_available is None:
        return MmapDecision(None, "ram_unknown", host_bytes, ram_total, None)
    if host_bytes + HOST_RAM_HEADROOM_BYTES <= ram_available:
        return MmapDecision(False, "fits_ram", host_bytes, ram_total, ram_available)
    return MmapDecision(None, "exceeds_ram", host_bytes, ram_total, ram_available)


class _AutoLayerBudget(NamedTuple):
    """The sizing inputs behind one ``_auto_gpu_layers()`` decision - the same
    numbers ``_effective_gpu_layers()`` needs to explain WHY a partial offload
    happened, kept so the VRAM budget is computed once and read twice rather
    than probed again for the notice."""
    layers: int
    free: int
    total: Optional[int]
    model: int
    kv: int
    overhead: int
    split_devices: int
    # Layers whose routed-expert weights the decision keeps in system RAM
    # (an automatic n_cpu_moe), 0 when it keeps none.
    moe_cpu_layers: int = 0


class VramSizingMixin:
    """VRAM measurement, preflight checks, and GPU-layer/context auto-sizing.

    Mixed into both ``GgufBackend`` (parent) and ``GgufWorker`` (child); the
    module docstring lists which methods each side calls.

    Expects the host class to provide: ``model_path`` (str), ``n_ctx`` (int),
    ``n_gpu_layers`` (int, the configured/raw value), ``effective_gpu_layers``
    (Optional[int], the resolved value once known), ``n_cpu_moe`` (int, the
    configured value) and ``effective_n_cpu_moe`` (Optional[int], the resolved
    value once known; both read via getattr), ``ctx_auto`` (bool),
    ``n_ctx_max`` (Optional[int]), ``mtp_enabled`` (bool, read defensively via
    getattr so a test double may omit it), ``_llm`` (the loaded native model,
    or None - only read by ``_check_context_fit`` for
    ``kv_bytes_per_token``/``_offload_kqv``), and ``_ram_kv_hint_shown`` (bool,
    mutable one-time-hint guard).
    """

    model_path: str
    n_ctx: int
    n_gpu_layers: int

    # Rough VRAM headroom for KV cache + compute buffers beyond model weights,
    # single-sourced from localm.vram.
    _VRAM_OVERHEAD_BYTES = VRAM_OVERHEAD_BYTES

    # Latched once a torch import has hit the DLL entry-point conflict in this
    # process, so it is never retried. A class attribute, shared across every
    # instance in this process.
    _torch_rocm_init_broken: bool = False

    # Latched once a torch VRAM read has blown its deadline in this process.
    # Separate from _torch_rocm_init_broken above, which names an import that
    # RAISES; this one is a wait that never returns. Both stop torch being asked.
    _torch_vram_read_wedged: bool = False

    @staticmethod
    def _torch_vram_read_deadline() -> float:
        """How long a torch VRAM read may block its caller, in seconds.

        Derived from ``discover._GPU_PROBE_DEADLINE``, re-read on every call
        rather than captured at import time."""
        from localm import discover
        return float(discover._GPU_PROBE_DEADLINE)

    @staticmethod
    def _free_total_vram_bytes() -> "tuple[Optional[int], Optional[int]]":
        """(free, total) bytes on the device a load's single-device readings
        come from (``discover.resolve_load_gpu_index``: the one device a 1-entry
        gpu_split_indices names, else main_gpu_index, device 0 when unset), or
        (None, None) when not measurable. Shared by _free_vram_bytes() and
        _total_vram_bytes() so both read the same device in one call.

        NEVER BLOCKS ITS CALLER WITHOUT A BOUND. Two guards sit in front of the
        read:

        - **The discover latch.** When ``discover.isolated_torch_unavailable()``
          reports that the out-of-process probe has already proven torch cannot
          finish enumerating on this box, no attempt is made at all. The latch
          is only consulted when torch is NOT already resident: once it is in
          ``sys.modules`` the reads below are ordinary calls on an imported
          module.
        - **No cold import on Windows.** Unless torch is already fully imported
          in this process, the answer is (None, None) at once, so the caller
          reads through the isolated native probe.
        - **A deadline.** Everything else runs on a helper thread with
          :meth:`_torch_vram_read_deadline`, and on overrun the caller is
          released with (None, None) - this method's "unmeasurable" answer,
          which every caller already handles: ``_free_vram_bytes`` falls through
          to the crash-isolated native probe, and if that cannot answer either,
          sizing degrades to the configured n_gpu_layers.

        A thread abandoned on overrun is never stopped and its result is
        discarded. That costs one thread once, because the overrun latches.

        Skips the attempt entirely once ``import torch`` has been confirmed
        broken in this process: with llama.cpp's bundled HIP/ROCm runtime
        already loaded here (via ``_loader.load_lib()``), a later ``import
        torch`` makes torch's ``rocm_sdk`` package ``ctypes.CDLL()`` its own
        ROCm library, which resolves to an incompatible DLL already in this
        process's address space (``OSError: [WinError 127]``). Python evicts a
        module that faults during import from ``sys.modules``, so an uncached
        failure re-attempts and re-faults on every subsequent VRAM check for
        the life of the process."""
        import sys
        import threading

        from localm.debuglog import logger as _dbg
        if VramSizingMixin._torch_rocm_init_broken:
            return None, None
        if VramSizingMixin._torch_vram_read_wedged:
            return None, None
        # A torch import hits the DLL-identity conflict whenever llama.cpp's own
        # native runtime is already loaded here; the caller falls back to
        # gpu_memory_isolated(), which answers without touching torch.
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            return None, None
        # Only when torch is not already resident: an imported torch makes the
        # reads below ordinary calls.
        if sys.platform == "win32":
            from localm.gpu_usage import torch_fully_imported
            if not torch_fully_imported():
                # A cold torch import takes the OS loader lock and blocks thread
                # creation process-wide, which no deadline here can bound.
                _dbg.debug(
                    "free-vram: torch is not imported in this process; reading "
                    "through the isolated native probe instead of importing it")
                return None, None
        if "torch" not in sys.modules:
            from localm import discover
            if discover.isolated_torch_unavailable():
                _dbg.debug(
                    "free-vram: skipping the in-process torch read - the "
                    "isolated probe already proved torch cannot answer on this "
                    "box; using the isolated native probe instead")
                return None, None
        # Bounded: neither the import nor a torch.cuda call has a timeout of its
        # own. The thread is abandoned on overrun.
        result: dict = {}
        done = threading.Event()

        def _read() -> None:
            try:
                result["value"] = VramSizingMixin._torch_free_total_uncapped()
            except BaseException as e:      # noqa: BLE001 - re-reported below
                result["error"] = e
            finally:
                done.set()

        try:
            threading.Thread(target=_read, name="localm-torch-vram-read",
                             daemon=True).start()
        except Exception as e:
            # Could not spawn a thread: degrade to unmeasurable rather than run
            # the read unbounded on the caller's own thread.
            _dbg.warning("free-vram: could not start the bounded torch read "
                         "(%s); treating VRAM as unmeasurable for this call",
                         type(e).__name__)
            return None, None
        deadline = VramSizingMixin._torch_vram_read_deadline()
        if not done.wait(deadline):
            VramSizingMixin._torch_vram_read_wedged = True
            # Once per process - the latch above guarantees that.
            _dbg.warning(
                "free-vram: torch did not answer within %.1fs; it is being "
                "skipped for the rest of this process and VRAM will be read "
                "via the isolated native probe. The GPU driver may be busy or "
                "wedged.", deadline)
            return None, None
        if "error" in result:
            raise result["error"]
        return result["value"]

    @staticmethod
    def _torch_free_total_uncapped() -> "tuple[Optional[int], Optional[int]]":
        """The actual torch read, with NO bound - call
        :meth:`_free_total_vram_bytes`, not this."""
        try:
            import torch
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("torch import failed (%s); VRAM reads will use the "
                       "isolated native probe fallback for the rest of this "
                       "process", type(e).__name__)
            VramSizingMixin._torch_rocm_init_broken = True
            return None, None
        try:
            if torch.cuda.is_available():
                from localm.discover import resolve_load_gpu_index
                idx = resolve_load_gpu_index()
                free, total = torch.cuda.mem_get_info(idx)
                return int(free), int(total)
        except Exception as e:
            # Degrades to (None, None), which sends _free_vram_bytes to the
            # isolated-probe fallback.
            from localm.debuglog import logger as _dbg
            _dbg.debug("torch.cuda.mem_get_info read failed (%s); falling back to "
                       "the isolated native VRAM probe", type(e).__name__)
        return None, None

    @classmethod
    def _free_vram_bytes(cls) -> Optional[int]:
        """Free VRAM in bytes on the GPU the model runs on, or None when not
        measurable.

        Prefers torch.cuda on the configured main GPU (honours main_gpu_index
        for a multi-GPU split) when torch is available, bounded by
        _free_total_vram_bytes. Falls back to loader.gpu_memory_isolated() when
        torch cannot answer, which includes "did not answer in time". Never
        calls loader.gpu_memory() directly in this process: that call's
        ggml_backend_dev_memory -> hipMemGetInfo path can abort the whole
        process from C on a transient driver condition, which no Python
        try/except can catch. The isolated probe runs the identical query
        against a long-lived daemon subprocess.

        A classmethod, so a subclass or test that monkeypatches
        ``_free_total_vram_bytes`` on itself is honoured here too: the internal
        cross-reference dispatches through ``cls``.

        Both reads above report ``total - THIS process's own allocations`` on
        Windows plus an AMD ROCm/HIP build and are blind to every other process,
        so the free reading is corrected to a device-global figure where the raw
        one is known blind (see _device_global_free_bytes); on every other
        platform it is returned unchanged."""
        # One debug line per read, stating raw/source/corrected.
        from localm.debuglog import logger as _dbg
        free_raw, total = cls._free_total_vram_bytes()
        src = "torch"
        if free_raw is None:
            from localm.inference.backends.llamacpp import _loader
            mem = _loader.gpu_memory_isolated()
            src = "isolated-probe"
            if mem is not None:
                free_raw, total = int(mem[0]), int(mem[1])
        if free_raw is None:
            _dbg.debug("free-vram read: unmeasurable (neither torch nor the "
                       "isolated probe answered)")
            return None
        corrected = cls._device_global_free_bytes(total)
        _dbg.debug("free-vram read: raw=%d total=%s source=%s "
                   "device-global-corrected=%s", free_raw, total, src, corrected)
        return corrected if corrected is not None else free_raw

    @staticmethod
    def _device_global_free_bytes(total: Optional[int]) -> Optional[int]:
        """``total`` minus ALL-process VRAM usage on the configured main GPU, or
        None when no device-global correction applies - the raw reading is not
        known-blind on this platform, or the correction source cannot map/answer.

        Never raises: a correction that cannot be made degrades to None, so the
        caller uses the uncorrected reading. Only acts where
        ``gpu_usage.raw_reading_is_process_scoped()`` is True - Windows plus a
        ROCm/HIP torch build, and the torch-less processes whose readings come
        from the resident bundled HIP runtime (the GGUF worker deciding a
        context grow, answered via ``discover.native_hip_runtime_resident()``).
        NVIDIA / Linux / Vulkan reads are left unchanged.

        The device entry carries the PCI bus id from the last completed GPU
        probe (``discover.last_known_gpus``, no new probe), which is what pairs
        the card with its ADL adapter exactly on a box with more than one
        adapter; without a completed probe the single-adapter rule applies."""
        if total is None:
            return None
        try:
            from localm.gpu_usage import (device_global_used_bytes,
                                          raw_reading_is_process_scoped)
            if not raw_reading_is_process_scoped():
                return None
            from localm.discover import last_known_gpus, resolve_load_gpu_index
            idx = resolve_load_gpu_index()
            entry = {"index": idx, "total": total}
            known = next((g for g in last_known_gpus() if g.get("index") == idx), None)
            if known and known.get("pci_bus_id") is not None:
                entry["pci_bus_id"] = known["pci_bus_id"]
            used = device_global_used_bytes([entry])
            u = used.get(idx)
            if u is None:
                return None
            return max(0, total - int(u))
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("cross-process VRAM correction unavailable (%s); using the "
                       "uncorrected free reading for sizing", type(e).__name__)
            return None

    @staticmethod
    def _free_reading_may_be_blind() -> bool:
        """Whether the free-VRAM figure a caller is about to PRINT could still be
        the raw, cross-process-blind reading rather than _device_global_free_bytes's
        correction of it.

        _free_vram_bytes() tries that correction and silently falls back to the
        raw value when it fails or declines, with no signal a caller can check
        to know which value it got back. This re-checks the same platform
        heuristic _device_global_free_bytes gates its own correction attempt on,
        without another probe. True here can mean the correction actually
        succeeded, so this errs toward an occasional unneeded caveat rather than
        ever omitting a needed one."""
        from localm.gpu_usage import raw_reading_is_process_scoped
        return raw_reading_is_process_scoped()

    @classmethod
    def _total_vram_bytes(cls) -> Optional[int]:
        """Total VRAM in bytes on the configured main GPU device, or None when
        not measurable by ANY path - same torch-then-isolated-probe fallback
        as _free_vram_bytes, so a caller that already has a real free reading
        via the isolated probe (torch unavailable/broken/wedged) is not left
        holding a None total purely because torch specifically could not
        answer. The hard physical ceiling: nothing can be freed to raise it,
        so a load that needs more than this can never fit on this device."""
        total = cls._free_total_vram_bytes()[1]
        if total is not None:
            return total
        from localm.inference.backends.llamacpp import _loader
        mem = _loader.gpu_memory_isolated()
        return int(mem[1]) if mem is not None else None

    @classmethod
    def _split_free_total_bytes(cls) -> "tuple[Optional[int], Optional[int], int]":
        """``(free, total, devices)`` summed across the 2+ devices this load
        will actually spread over, or ``(None, None, 0)`` when no combined
        budget applies and the caller must fall back to the single-device
        readings above. With no ``gpu_split_indices``, ``devices`` is 1 when
        ``discover.implicit_split_capacity`` answers for the one discrete GPU
        left beside integrated ones.

        BOTH SPLITS COUNT. A CONFIGURED ``gpu_split_indices`` writes an
        explicit ``tensor_split`` (``discover.apply_gpu_split``). An UNSET one
        does NOT produce a single-GPU load: it leaves llama.cpp's own defaults,
        ``LLAMA_SPLIT_MODE_LAYER`` with ``tensor_split = NULL``, which is an
        IMPLICIT layer split across every registered GPU weighted by each
        device's free memory. ``discover.implicit_split_capacity`` owns that
        second case.

        For a split load, weights and KV both draw on the split's combined
        capacity: ``discover.apply_gpu_split`` tensor-splits the weights across
        the configured devices, and llama.cpp places each layer's KV cache on
        the device that holds the layer.

        Answers ``(None, None, 0)`` - "no combined budget, use the
        single-device reading" - in these cases:

        - The native llama/ggml runtime is loaded IN THIS PROCESS (the
          GgufWorker or an isolated child, where ``_check_context_fit`` runs
          mid-generation): no probe is attempted at all, because
          ``discover.list_gpus``'s probe does ``import torch``, the
          DLL-identity conflict ``_free_total_vram_bytes`` guards against. The
          single-device fallback stays honest there: the isolated native probe
          declines to answer on a 2+-GPU-device box
          (``_loader._resolve_gpu_memory``).
        - Fewer than 2 GPU devices are detected, whether or not a split is
          configured. With no ``gpu_split_indices`` this is answered by
          ``discover.implicit_split_capacity``, which short-circuits from
          config alone (no hardware probe) only when a split IS configured; on
          a genuine single-GPU box it costs one ``list_gpus`` call.
        - The probe did not complete fresh this call (non-``GPU_PROBE_OK``),
          so the served figure may be a frozen last-known-good value.
          ``wait_for_inflight=True`` is passed with the default
          cold-init-tolerant deadline; this path always runs off the event loop
          (model loads run in an executor or CLI thread).
        - Fewer than 2 split devices are detected (``vram_capacity``'s
          ``combined_only`` contract returns ``{}``).
        - A test double patched ``vram_capacity`` without the opt-in kwargs
          (TypeError) or with a plain dict lacking the ``"devices"`` key.

        Never raises: a failure to fetch a combined reading is surfaced at
        debug and answered as "no combined reading"."""
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            return None, None, 0
        try:
            from localm.config import load_config
            cfg = load_config()
            if not cfg.get("gpu_split_indices"):
                from localm.discover import implicit_split_capacity
                info = implicit_split_capacity(cfg, wait_for_inflight=True)
                free, total = info.get("free"), info.get("total")
                devices = info.get("devices") or 0
                if devices < 1 or free is None or total is None:
                    return None, None, 0
                return int(free), int(total), int(devices)
            from localm.discover import GPU_PROBE_OK, vram_capacity
            try:
                result = vram_capacity(cfg, return_status=True,
                                       wait_for_inflight=True,
                                       combined_only=True)
            except TypeError:
                # A test double without the opt-in kwargs cannot answer
                # "combined or nothing" - so there is no combined reading.
                return None, None, 0
            if isinstance(result, tuple) and len(result) == 2:
                info, status = result
            else:
                # A plain-dict double (no return_status support) is treated as a
                # completed probe.
                info, status = result, GPU_PROBE_OK
            if status != GPU_PROBE_OK or not isinstance(info, dict):
                return None, None, 0
            devices = info.get("devices") or 0
            free, total = info.get("free"), info.get("total")
            if devices < 2 or free is None or total is None:
                return None, None, 0
            return int(free), int(total), int(devices)
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("combined split VRAM reading unavailable (%s); sizing "
                       "against the single main GPU instead", type(e).__name__)
            return None, None, 0

    def _split_overhead_bytes(self, devices: int) -> int:
        """``_VRAM_OVERHEAD_BYTES`` scaled by how many devices the load spreads
        over - the flat constant when it spreads over one, or when the count is
        not known.

        The constant covers "KV cache + compute buffers beyond model weights",
        and COMPUTE BUFFERS ARE PER DEVICE: llama.cpp reserves a compute buffer
        on each device that holds layers, so an N-device split reserves N of
        them. It also absorbs the two residuals a free-proportional split leaves
        behind: layers are integral, so a device can receive at most one layer
        more than its exact share, and the free reading is a snapshot another
        process can invalidate between the probe and the load."""
        return self._VRAM_OVERHEAD_BYTES * max(1, int(devices or 1))

    # Largest n_batch/n_ubatch llama.py gives a context: n_batch = min(n_ctx,
    # this), n_ubatch = n_batch.
    _MAX_BATCH = 2048

    def _implicit_split_fit(self, gpu_layers: int, n_cpu_moe: Optional[int] = None):
        """Per-device fit of llama.cpp's IMPLICIT layer split for this load, as
        a :class:`~localm.inference.backends.llamacpp._split_fit.SplitFitPlan`,
        or ``None`` when it does not apply or cannot be measured.

        Applies only to a GPU load (``gpu_layers != 0``) with no configured
        ``gpu_split_indices``, on 2+ devices whose
        readings :func:`localm.discover.implicit_split_devices` returns, for a
        model whose GGUF header :func:`localm.model_manager.gguf.gguf_split_layout`
        reads. Each device that receives a layer or the output layer is charged
        its layers' weights (less the routed experts of the blocks below
        *n_cpu_moe*, default this load's :meth:`_load_n_cpu_moe`, which stay in
        system RAM) and KV cache and ``_VRAM_OVERHEAD_BYTES``; the device
        that receives the output layer is
        also charged the output weights and the logits buffer (twice when an
        MTP draft context will be created). The weights of MTP / nextn layers
        are charged only when MTP is enabled, since llama.cpp skips loading
        them otherwise; with MTP enabled the MTP draft context
        (:meth:`_mtp_draft_context_vram_bytes`) is charged as the KV cache of
        those layers. The recurrent state (:meth:`_recurrent_state_vram_bytes`)
        is charged in equal parts to the layers that keep one. A draft model on
        the GPU (:meth:`_draft_model_vram_bytes`) is split over the same devices
        as the target, so it is charged to them in proportion to their shares.
        A plan that writes a split or reports a shortfall is
        returned only when :func:`localm.discover.runtime_split_devices_match`
        confirms the device numbering; when the runtime instead keeps the
        integrated GPUs (:func:`localm.discover.runtime_identity_split_devices`),
        the plan is made over every GPU in torch's numbering. ``_fit_source_index`` maps each
        planned device to its torch index. Must run off the event loop (it
        probes). Never raises."""
        if gpu_layers == 0:
            return None
        inputs = self._implicit_split_inputs(gpu_layers)
        if inputs is None:
            return None
        return self._implicit_split_plan(
            inputs, self._load_n_cpu_moe() if n_cpu_moe is None else int(n_cpu_moe))

    def _implicit_split_inputs(self, gpu_layers: int) -> Optional[dict]:
        """What :meth:`_implicit_split_plan` needs that does not depend on
        n_cpu_moe, read from the GGUF header and probed once: the device
        readings, every tensor's size, the layer counts and the per-layer KV,
        output and logits charges for *gpu_layers*. None when the fit does not
        apply (see :meth:`_implicit_split_fit`). Must run off the event loop.
        Never raises."""
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            return None
        try:
            from localm.config import load_config
            from localm.discover import implicit_split_devices
            from localm.inference.backends.llamacpp._split_fit import (
                logits_buffer_bytes)
            from localm.model_manager.gguf import (
                gguf_nextn_predict_layers, gguf_split_layout)
            cfg = load_config()
            if cfg.get("gpu_split_indices"):
                return None
            path = Path(self.model_path)
            layout = gguf_split_layout(path)
            if layout is None:
                return None
            devices = implicit_split_devices(cfg, wait_for_inflight=True,
                                             check_runtime=False,
                                             with_source_index=True)
            if not devices:
                return None
            n_layer_all = int(layout["block_count"])
            sizes = layout["tensor_bytes"]
            arch, nextn = gguf_nextn_predict_layers(path)
            nextn = max(0, min(int(nextn), n_layer_all))
            mtp_on = bool(getattr(self, "mtp_enabled", False)) and nextn > 0
            if mtp_on:
                from localm.inference.backends.llamacpp._api import (
                    MTP_GRAPH_ARCHITECTURES)
                mtp_on = arch in MTP_GRAPH_ARCHITECTURES
            output_bytes = (sizes.get("output.weight")
                            or sizes.get("token_embd.weight") or 0)
            output_bytes += sizes.get("output_norm.weight", 0)
            repeating = max(1, n_layer_all - nextn)
            kv_per_layer = (self.n_ctx * self._kv_bytes_per_token()) // repeating
            draft_per_layer = (-(-self._mtp_draft_context_vram_bytes() // nextn)
                               if mtp_on else 0)
            from localm.model_manager.gguf import _RECURRENT_LAYER_TENSOR_RE
            recurrent = {int(m.group(1)) for name in sizes
                         if (m := _RECURRENT_LAYER_TENSOR_RE.match(name))}
            state_per_layer = (-(-self._recurrent_state_vram_bytes() // len(recurrent))
                               if recurrent else 0)
            layer_kv = [(kv_per_layer if il < n_layer_all - nextn else draft_per_layer)
                        + (state_per_layer if il in recurrent else 0)
                        for il in range(n_layer_all)]
            logits = logits_buffer_bytes(layout["n_vocab"], self.n_ctx,
                                         max_batch=self._MAX_BATCH,
                                         contexts=2 if mtp_on else 1)
            return {
                "devices": devices, "sizes": sizes, "n_layer_all": n_layer_all,
                "nextn": nextn, "mtp_on": mtp_on, "runtime_match": None,
                "identity_devices": None,
                "fit_kw": dict(output_bytes=int(output_bytes),
                               layer_kv_bytes=layer_kv, n_gpu_layers=int(gpu_layers),
                               logits_bytes=logits,
                               reserve_bytes=int(self._VRAM_OVERHEAD_BYTES),
                               spread_bytes=int(self._draft_model_vram_bytes())),
            }
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("implicit split fit unavailable (%s: %s); keeping "
                       "llama.cpp's default split", type(e).__name__, e)
            return None

    def _implicit_split_plan(self, inputs: dict, n_cpu_moe: int):
        """The :class:`~localm.inference.backends.llamacpp._split_fit.SplitFitPlan`
        for *inputs* (:meth:`_implicit_split_inputs`) with the routed experts of
        the blocks below *n_cpu_moe* charged to no device, and
        ``_fit_source_index`` set for its devices. The device-numbering checks
        of :meth:`_implicit_split_fit` run once per *inputs* and are kept in it.
        None when the runtime's numbering cannot be matched. Never raises."""
        try:
            from localm.discover import (runtime_identity_split_devices,
                                         runtime_split_devices_match)
            from localm.inference.backends.llamacpp._split_fit import plan_split
            from localm.model_manager.gguf import _MOE_EXPERT_TENSOR_RE
            n_layer_all = inputs["n_layer_all"]
            layer_bytes = [0] * n_layer_all
            for name, size in inputs["sizes"].items():
                # An encoder-decoder model's enc.blk.<i> and dec.blk.<i> both
                # belong to layer i.
                if name.startswith(("enc.blk.", "dec.blk.")):
                    name = name[4:]
                if not name.startswith("blk."):
                    continue
                head, _, _rest = name[4:].partition(".")
                if not head.isdigit() or int(head) >= n_layer_all:
                    continue
                expert = _MOE_EXPERT_TENSOR_RE.search(name)
                if expert is not None and int(expert.group(1)) < n_cpu_moe:
                    continue
                layer_bytes[int(head)] += int(size)
            if not inputs["mtp_on"]:
                for il in range(n_layer_all - inputs["nextn"], n_layer_all):
                    layer_bytes[il] = 0
            devices = inputs["devices"]
            plan = plan_split(devices, layer_bytes=layer_bytes, **inputs["fit_kw"])
            if plan.tensor_split or not plan.default_fits:
                if inputs["runtime_match"] is None:
                    inputs["runtime_match"] = bool(runtime_split_devices_match(devices))
                if not inputs["runtime_match"]:
                    if inputs["identity_devices"] is None:
                        inputs["identity_devices"] = runtime_identity_split_devices() or []
                    devices = inputs["identity_devices"]
                    if not devices:
                        return None
                    plan = plan_split(devices, layer_bytes=layer_bytes, **inputs["fit_kw"])
            self._fit_source_index = {d["index"]: d.get("source_index", d["index"])
                                      for d in devices}
            return plan
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("implicit split fit unavailable (%s: %s); keeping "
                       "llama.cpp's default split", type(e).__name__, e)
            return None

    def _split_fitting_n_cpu_moe(self, free: int, kv: int,
                                 overhead: int) -> "tuple[bool, Optional[int]]":
        """For a load llama.cpp's implicit split spreads over 2+ GPUs: ``(True,
        n)`` with the smallest n_cpu_moe, 0 included, whose per-device fit
        (:meth:`_implicit_split_plan`, every layer on a GPU) fits every device
        or leaves out a device that does not, and whose combined need
        (``_vram_model_bytes(n) + kv + overhead``, the one ``_check_vram``
        charges) fits the combined *free*; ``(True, None)`` when none does, and
        ``(False, None)`` when no per-device fit can be made (a configured
        ``gpu_split_indices``, an unreadable layout, no device readings, or a
        device numbering the runtime cannot be matched to).
        Must run off the event loop. Never raises."""
        inputs = self._implicit_split_inputs(self._DEFAULT_GPU_LAYERS)
        if inputs is None:
            return False, None
        for n in [0] + [layer + 1 for layer in sorted(self._moe_expert_bytes_by_layer())]:
            plan = self._implicit_split_plan(inputs, n)
            if plan is None:
                return False, None
            if ((plan.default_fits or plan.tensor_split)
                    and self._vram_model_bytes(n) + kv + overhead <= free):
                return True, n
        return True, None

    @staticmethod
    def _gpu_split_configured() -> bool:
        """Whether ``gpu_split_indices`` is set in the config. False when the
        config cannot be read. Never raises."""
        try:
            from localm.config import load_config
            return bool(load_config().get("gpu_split_indices"))
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("config unreadable for the GPU split check (%s)", type(e).__name__)
            return False

    def _mtp_draft_context_vram_bytes(self) -> int:
        """Extra VRAM llama.py's MTP draft context (cp_mtp) will need beyond
        the shared model weights, for THIS load - 0 when ``mtp_enabled`` is
        off or this GGUF is not eligible for one.

        Two charges: the draft context's own KV cache, sized to the main
        ``self.n_ctx`` (the draft context is created at the main context's size)
        and its own nextn/draft layer count rather than the whole stack; and a flat
        compute-buffer charge of ``_VRAM_OVERHEAD_BYTES``, the same constant
        the main context's own KV-cache-plus-compute-buffer overhead already
        uses - the draft context's n_batch/n_ubatch are set to its own n_ctx
        regardless of layer count, so its buffers are not assumed to shrink
        proportionally with it.

        Eligibility is the SAME two-part gate llama_model_mtp_support checks
        on an already-loaded model (declared nextn metadata AND an
        architecture whose llama.cpp class actually builds an MTP graph -
        MTP_GRAPH_ARCHITECTURES), read from the GGUF header directly so this
        can answer before any load exists. Never raises: a probe failure
        charges 0, the same as "not eligible". Memoised per instance - the
        file read happens once per load."""
        if not getattr(self, "mtp_enabled", False):
            return 0
        cached = getattr(self, "_mtp_draft_vram_bytes_cached", None)
        if cached is not None:
            return cached
        charge = 0
        try:
            from localm.inference.backends.llamacpp._api import (
                MTP_GRAPH_ARCHITECTURES)
            from localm.model_manager.gguf import (
                gguf_mtp_draft_kv_bytes_per_token, gguf_nextn_predict_layers)
            path = Path(self.model_path)
            arch, nextn_layers = gguf_nextn_predict_layers(path)
            if arch in MTP_GRAPH_ARCHITECTURES and nextn_layers > 0:
                kv_per_token = gguf_mtp_draft_kv_bytes_per_token(
                    path, nextn_layers)
                self._mtp_draft_kv_per_token_cached = int(kv_per_token)
                charge = self.n_ctx * kv_per_token + self._VRAM_OVERHEAD_BYTES
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("mtp draft-context VRAM probe failed (%s); charging "
                       "nothing extra for it", type(exc).__name__)
            charge = 0
        self._mtp_draft_vram_bytes_cached = charge
        return charge

    def _recurrent_state_vram_bytes(self) -> int:
        """VRAM the main context's recurrent state takes for THIS load - 0 for a
        model with no recurrent layers.

        A hybrid (linear-attention / state-space) stack keeps a fixed-size state
        per recurrent layer that does not grow with the context.
        ``gguf_recurrent_state_bytes`` is one copy of it; the context allocates
        ``1 + n_rs_seq`` copies. ``n_rs_seq`` is 0 without speculation; with
        MTP enabled it is ``llama.mtp_rs_seq`` of the draft-token count, and
        with the ngram or draft source ``_ngram.ngram_rs_seq`` of that source's
        draft cap for a recurrent model (the same calls LlamaCpp makes when it
        creates the context). Never raises: a probe failure charges 0.
        Memoised per instance."""
        cached = getattr(self, "_recurrent_state_vram_bytes_cached", None)
        if cached is not None:
            return cached
        charge = 0
        try:
            from localm.model_manager.gguf import gguf_recurrent_state_bytes
            per_copy = gguf_recurrent_state_bytes(
                Path(self.model_path), _parsed=self._gguf_parsed_tensor_entries())
            if per_copy:
                n_rs_seq = 0
                source = getattr(self, "spec_source", None)
                if source in ("ngram", "draft"):
                    from localm.inference.backends.llamacpp._draftmodel import (
                        DRAFT_MODEL_DRAFT_TOKENS_DEFAULT)
                    from localm.inference.backends.llamacpp._ngram import (
                        NGRAM_DRAFT_TOKENS_DEFAULT, ngram_draft_cap, ngram_rs_seq)
                    n_rs_seq = ngram_rs_seq(0, ngram_draft_cap(
                        getattr(self, "spec_draft_tokens", None), True,
                        default=(NGRAM_DRAFT_TOKENS_DEFAULT if source == "ngram"
                                 else DRAFT_MODEL_DRAFT_TOKENS_DEFAULT)))
                elif getattr(self, "mtp_enabled", False):
                    from localm.inference.backends.llamacpp.llama import (
                        MTP_DRAFT_TOKENS_DEFAULT, mtp_rs_seq)
                    draft = getattr(self, "mtp_draft_tokens", None)
                    n_rs_seq = mtp_rs_seq(
                        0, MTP_DRAFT_TOKENS_DEFAULT if draft is None else draft)
                charge = per_copy * (1 + n_rs_seq)
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("recurrent-state VRAM probe failed (%s); charging "
                       "nothing extra for it", type(exc).__name__)
            charge = 0
        self._recurrent_state_vram_bytes_cached = charge
        return charge

    def _draft_model_charge_bytes(self) -> int:
        """VRAM the draft source's second model needs on the GPU for THIS load,
        0 unless ``spec_source`` is "draft", this model is neither an
        encoder-decoder nor a diffusion model (neither drafts), and
        ``spec_draft_model`` is a readable GGUF that
        ``_draft_model_rejected_by_metadata`` does not reject.

        The draft model's file size (its weights), its KV cache for
        ``self.n_ctx`` tokens, the logits buffer of a context whose batch is
        ``DRAFT_CONTEXT_BATCH``, and ``DRAFT_COMPUTE_MARGIN_BYTES``; the draft
        context is created at the main context's size. Never raises: a probe
        failure charges 0. Memoised per instance."""
        if getattr(self, "spec_source", None) != "draft" or self._keeps_no_kv_cache():
            return 0
        cached = getattr(self, "_draft_model_charge_bytes_cached", None)
        if cached is not None:
            return cached
        charge = 0
        try:
            from localm.inference.backends.llamacpp._draftmodel import (
                DRAFT_COMPUTE_MARGIN_BYTES, DRAFT_CONTEXT_BATCH)
            from localm.inference.backends.llamacpp._split_fit import logits_buffer_bytes
            from localm.model_manager.gguf import (
                _GGUF_ENCODER_DECODER_ARCHITECTURES, _gguf_split_layout_meta, gguf_architecture,
                gguf_file_bytes, gguf_kv_bytes_per_token)
            raw = getattr(self, "spec_draft_model", None)
            path = Path(raw) if raw else None
            encoder_decoder = (gguf_architecture(Path(self.model_path))
                               in _GGUF_ENCODER_DECODER_ARCHITECTURES)
            if (path is not None and path.is_file() and not encoder_decoder
                    and not self._draft_model_rejected_by_metadata(path)):
                kv_per_token = int(gguf_kv_bytes_per_token(path))
                meta = _gguf_split_layout_meta(path)
                n_vocab = meta[1] if meta else 0
                self._draft_kv_per_token_cached = kv_per_token
                charge = (gguf_file_bytes(path) + self.n_ctx * kv_per_token
                          + logits_buffer_bytes(n_vocab, self.n_ctx,
                                                max_batch=DRAFT_CONTEXT_BATCH)
                          + DRAFT_COMPUTE_MARGIN_BYTES)
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("draft-model VRAM probe failed (%s); charging nothing "
                       "extra for it", type(exc).__name__)
            charge = 0
        self._draft_model_charge_bytes_cached = charge
        return charge

    def _draft_model_rejected_by_metadata(self, path: Path) -> bool:
        """Whether *path*'s GGUF metadata already shows the load would reject it
        as this model's draft model: not a causal chat model
        (``draft_role_refusal``), recurrent-state layers, or a vocabulary
        ``draft_vocab_mismatch`` refuses against this model's. False when the
        metadata cannot be read. Never raises."""
        try:
            from localm.inference.backends.llamacpp._draftmodel import (
                draft_role_refusal, draft_vocab_mismatch, gguf_vocab_view)
            from localm.model_manager.gguf import (
                gguf_recurrent_state_bytes, gguf_vocab_signature)
            if draft_role_refusal(path) is not None:
                return True
            if gguf_recurrent_state_bytes(path) > 0:
                return True
            target = gguf_vocab_signature(Path(self.model_path))
            draft = gguf_vocab_signature(path)
            if target is None or draft is None:
                return False
            return draft_vocab_mismatch(gguf_vocab_view(target),
                                        gguf_vocab_view(draft)) is not None
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("draft-model metadata check failed (%s); charging it",
                       type(exc).__name__)
            return False

    def _draft_model_vram_bytes(self) -> int:
        """VRAM the draft model takes for THIS load: ``_draft_model_charge_bytes``
        while ``draft_model_on_gpu`` is True (the default until
        ``_decide_draft_placement`` runs), else 0."""
        if not getattr(self, "draft_model_on_gpu", True):
            return 0
        return self._draft_model_charge_bytes()

    def _decide_draft_placement(self) -> bool:
        """Whether the draft model goes on the GPU, recorded in
        ``draft_model_on_gpu`` and returned.

        True when the draft source is not configured, its charge is 0, or free
        VRAM cannot be read. False when the target runs on the CPU
        (``n_gpu_layers`` 0). Under llama.cpp's implicit split over 2+ devices
        (:meth:`_implicit_split_fit` applies), True exactly when the per-device
        plan with the draft split over the target's devices keeps the
        target's split and fits every device. Otherwise True exactly when free
        VRAM covers the target's GPU weights (every layer, or ``n_gpu_layers``
        of them when fewer are configured), its KV cache for ``self.n_ctx``
        tokens, the compute overhead, any MTP draft context, the recurrent
        state and the draft charge. A draft model that does not fit runs on the
        CPU, so it never takes GPU layers or devices from the target. Reads
        free VRAM, so it must not run on an event loop thread."""
        on_gpu = True
        charge = self._draft_model_charge_bytes()
        if getattr(self, "spec_source", None) == "draft" and charge > 0:
            self.draft_model_on_gpu = True
            with_draft = (self._implicit_split_fit(self.n_gpu_layers)
                          if self.n_gpu_layers > 0 else None)
            if self.n_gpu_layers <= 0:
                on_gpu = False
            elif with_draft is not None:
                self.draft_model_on_gpu = False
                without = self._implicit_split_fit(self.n_gpu_layers)
                charges = with_draft.chosen or with_draft.default
                on_gpu = (without is not None
                          and with_draft.tensor_split == without.tensor_split
                          and all(c.fits for c in charges))
            else:
                free, _total, split_devices = self._split_free_total_bytes()
                if free is None:
                    free, split_devices = self._free_vram_bytes(), 1
                if free is not None:
                    model = self._vram_model_bytes(int(getattr(self, "n_cpu_moe", 0) or 0))
                    if self.n_gpu_layers < self._DEFAULT_GPU_LAYERS:
                        layers = self._cached_layer_count() or self._ASSUMED_LAYERS
                        model = model * min(self.n_gpu_layers, layers) // layers
                    need = (model + self.n_ctx * self._kv_bytes_per_token()
                            + self._split_overhead_bytes(split_devices or 1)
                            + self._mtp_draft_context_vram_bytes()
                            + self._recurrent_state_vram_bytes() + charge)
                    on_gpu = free >= need
        self.draft_model_on_gpu = on_gpu
        return on_gpu

    def _spec_extra_vram_bytes(self) -> int:
        """VRAM the configured draft source needs beyond the main model and
        context: the MTP draft context or the draft model."""
        return self._mtp_draft_context_vram_bytes() + self._draft_model_vram_bytes()

    def _spec_kv_per_token(self) -> int:
        """KV bytes per token of context a draft source adds on the GPU: the MTP
        draft context's or the draft model's while it is on the GPU, both of
        which grow with the main one."""
        draft = 0
        if self._draft_model_vram_bytes():
            draft = int(getattr(self, "_draft_kv_per_token_cached", 0) or 0)
        return self._mtp_draft_kv_per_token() + draft

    def _mtp_draft_kv_per_token(self) -> int:
        """Draft-context KV bytes per token of context, 0 when this load has no
        MTP draft context. The draft context grows with the main one, so a
        context-growth decision charges this on top of the main KV per token."""
        self._mtp_draft_context_vram_bytes()
        return int(getattr(self, "_mtp_draft_kv_per_token_cached", 0) or 0)

    @staticmethod
    def _vram_levels() -> list:
        """(free, total) bytes per device, [] when not measurable.

        Driver-level numbers (mem_get_info), NOT torch allocator counters:
        llama.dll allocates through HIP/CUDA directly, so
        torch.cuda.memory_allocated() reads zero for GGUF loads no matter
        how much VRAM the model actually occupies.

        Skips the torch attempt entirely once ``_loader.native_lib_loaded()``
        is True - the same precondition ``_free_total_vram_bytes`` guards for
        the identical DLL-identity conflict. With llama.cpp's own native
        runtime loaded in this process, a later ``import torch`` on a Windows
        plus AMD ROCm build hits STATUS_ENTRYPOINT_NOT_FOUND, Python evicts the
        faulted module, and an unguarded caller re-triggers it on every call.

        Nothing in ``localm/`` calls this method; only tests do. This method's
        ``import torch`` is UNBOUNDED, unlike ``_free_total_vram_bytes``'s, so
        anything that gives it a caller again needs the same deadline."""
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            from localm.debuglog import logger as _dbg
            _dbg.debug(
                "_vram_levels: skipping the torch VRAM read - llama.cpp's "
                "native runtime is already loaded in this process, so `import "
                "torch` here is the known-doomed DLL-identity conflict (see "
                "VramSizingMixin._free_total_vram_bytes's docstring); "
                "returning [] (display-only, no load decision reads this)")
            return []
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
        from localm.model_manager.gguf import gguf_file_bytes
        return gguf_file_bytes(Path(self.model_path))

    def _gguf_parsed_tensor_entries(self):
        """This load's own ``_gguf_tensor_offset_entries(model_path)`` result
        (or its ``None`` failure), read at most once per instance and shared
        by every excluded-tensor-byte probe in
        ``_effective_model_bytes_for_vram``. Degrades to ``None`` - the same
        outcome as a parse the function's own contract already reports as a
        failure - on any exception, so a violation of that contract cannot
        crash the caller."""
        if not hasattr(self, "_gguf_parsed_entries_cache"):
            from localm.model_manager.gguf import _gguf_tensor_offset_entries
            try:
                parsed = _gguf_tensor_offset_entries(Path(self.model_path))
            except Exception as exc:  # contracted not to raise - surface if it does
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf tensor-offset parse failed (%s); VRAM sizing "
                           "will charge the affected tensors' bytes",
                           type(exc).__name__)
                parsed = None
            self._gguf_parsed_entries_cache = parsed
        return self._gguf_parsed_entries_cache

    def _gguf_excluded_bytes(self, attr: str, probe, desc: str) -> int:
        """Probe/memoize/degrade one VRAM-excluded-tensor-byte subtraction:
        call *probe* (no arguments) at most once per instance, cache the
        result under *attr*, and degrade to 0 - charging those bytes as
        still VRAM-resident - on any exception from *probe*, logged at debug
        with *desc*. Shared by the input-layer and MoE-pinned-expert
        subtractions in ``_effective_model_bytes_for_vram``, which differ
        only in *probe* and *attr*."""
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        try:
            value = probe()
        except Exception as exc:  # contracted not to raise - surface if it does
            from localm.debuglog import logger as _dbg
            _dbg.debug("gguf %s probe failed (%s); charging those bytes for "
                       "VRAM sizing", desc, type(exc).__name__)
            value = None
        result = value if value is not None else 0
        setattr(self, attr, result)
        return result

    def _load_n_cpu_moe(self) -> int:
        """The n_cpu_moe THIS load uses: ``effective_n_cpu_moe`` once
        ``_effective_gpu_layers()`` resolved it (an automatic choice or the
        configured value), else the configured ``n_cpu_moe``."""
        effective = getattr(self, "effective_n_cpu_moe", None)
        if effective is not None:
            return int(effective)
        return int(getattr(self, "n_cpu_moe", 0) or 0)

    def _block_bytes(self) -> "dict[int, tuple[int, int]]":
        """``gguf_block_bytes`` for this model (``{block: (all bytes, routed
        expert bytes)}``), read at most once per instance. ``{}`` on a probe
        failure, which charges every expert byte to VRAM."""
        cached = getattr(self, "_block_bytes_cache", None)
        if cached is not None:
            return cached
        from localm.model_manager.gguf import gguf_block_bytes
        try:
            blocks = gguf_block_bytes(
                Path(self.model_path), _parsed=self._gguf_parsed_tensor_entries())
        except Exception as exc:  # contracted not to raise - surface if it does
            from localm.debuglog import logger as _dbg
            _dbg.debug("gguf block-byte probe failed (%s); charging every "
                       "expert byte for VRAM sizing", type(exc).__name__)
            blocks = None
        if blocks is None:
            from localm.debuglog import logger as _dbg
            _dbg.debug("gguf block-byte probe could not read %s; charging "
                       "every expert byte for VRAM sizing",
                       Path(self.model_path).name)
            blocks = {}
        self._block_bytes_cache = blocks
        return blocks

    def _moe_expert_bytes_by_layer(self) -> "dict[int, int]":
        """Routed-expert bytes of each block that has any (``_block_bytes``).
        ``{}`` for a dense model and on a probe failure."""
        return {layer: experts for layer, (_total, experts) in self._block_bytes().items()
                if experts > 0}

    def _moe_pinned_bytes(self, n_cpu_moe: int) -> int:
        """Bytes of routed-expert weights the first *n_cpu_moe* layers keep in
        system RAM (0 for a dense model or *n_cpu_moe* <= 0)."""
        if n_cpu_moe <= 0:
            return 0
        return sum(size for layer, size in self._moe_expert_bytes_by_layer().items()
                   if layer < n_cpu_moe)

    def _vram_model_bytes(self, n_cpu_moe: int) -> int:
        """VRAM-resident weight bytes for a load that keeps the routed experts
        of the first *n_cpu_moe* layers in system RAM: ``_model_bytes()`` minus
        the input-layer tensors (every load) and those experts. See
        :meth:`_effective_model_bytes_for_vram`."""
        model_bytes = self._model_bytes()
        parsed = self._gguf_parsed_tensor_entries()

        from localm.model_manager.gguf import gguf_input_layer_bytes
        input_bytes = self._gguf_excluded_bytes(
            "_gguf_input_layer_bytes",
            lambda: gguf_input_layer_bytes(Path(self.model_path), _parsed=parsed),
            "input-layer byte")
        model_bytes = max(0, model_bytes - input_bytes)
        return max(0, model_bytes - self._moe_pinned_bytes(n_cpu_moe))

    def _effective_model_bytes_for_vram(self) -> int:
        """VRAM-resident weight bytes for THIS load: ``_model_bytes()``, minus
        every tensor llama.cpp never places in VRAM for it.

        Two independent, always-additive subtractions:

        - The INPUT-LAYER tensors (``token_embd`` and its siblings - see
          ``gguf_input_layer_bytes``/``_INPUT_LAYER_TENSOR_NAMES``). Applies
          to every load, dense or MoE, n_cpu_moe set or not.
        - Whatever this load's n_cpu_moe (:meth:`_load_n_cpu_moe`, configured
          or chosen automatically) pins to SYSTEM RAM (see llama.py's
          ``_apply_cpu_moe`` - the routed-expert tensors of the first
          n_cpu_moe layers never touch VRAM at all either).

        Both are computed from each excluded tensor's EXACT size via its
        file's tensor-info offsets (never a per-quantization-type size
        table), and both degrade to charging the tensor's bytes anyway on a
        probe failure or an unparseable header. The input-layer result and the
        per-layer expert bytes are memoised per instance."""
        return self._vram_model_bytes(self._load_n_cpu_moe())

    def _vram_holder_hint(self) -> str:
        """Best-effort: name a concrete live sibling localm instance holding
        VRAM on this same GPU device (port, model), found by asking the running
        instances directly (``localm.gpu_registry``) - instead of the generic
        "another GPU app" text. Falls back to the generic text when no peer is
        found or the lookup itself fails; purely a diagnostic, never
        load-blocking.

        ``gpu_registry.list_gpu_peers()`` always excludes THIS process
        (matched by pid), so ``holder`` below is always a genuinely different
        instance. When no external holder is found,
        :func:`gpu_registry.own_status` reports whether THIS process's own live
        status explains it (e.g. this server has another model resident while
        loading a second one)."""
        try:
            from localm.config import load_config
            from localm.discover import last_gpu_reading, resolve_load_gpu_index
            from localm import gpu_registry
            idx = resolve_load_gpu_index(load_config(), gpus=last_gpu_reading() or [],
                                         quiet=True)
            peers = gpu_registry.list_gpu_peers()
            holder = next(
                (p for p in peers
                 if p.get("model") and int(p.get("gpu_index", 0) or 0) == idx),
                None,
            )
            if holder is not None:
                return (
                    f"another localm instance (port {holder.get('port')}) is "
                    f"running '{holder.get('model')}' - "
                    f"POST /v1/models/unload on port {holder.get('port')} to free it."
                )
            self_entry = gpu_registry.own_status()
            if (self_entry is not None and self_entry.get("model")
                    and int(self_entry.get("gpu_index", 0) or 0) == idx):
                return (
                    f"this server's own currently-loaded model "
                    f"'{self_entry.get('model')}' is holding it."
                )
        except Exception:
            pass  # advisory only - fall through to the generic hint
        return "another GPU app is holding memory (ComfyUI, a browser, another model)."

    @staticmethod
    def _bytes_per_token(model_bytes: int) -> int:
        """KV bytes per token, estimated from the model's size class (larger
        models have more layers and wider KV heads; sliding-window models need
        less, so the estimate stays conservative). Shared by _check_vram()'s
        preflight KV-cache estimate and _auto_ctx_max()'s VRAM-derived
        ceiling."""
        return min(max(model_bytes // 100_000, 16_000), 512_000)

    def _kv_bytes_per_token(self) -> int:
        """Per-token KV cost for a sizing decision, best source first.

        1. the LOADED model's own attention accessors
           (``LlamaCpp.kv_bytes_per_token``) - exact, but only exists AFTER a
           load;
        2. this file's own GGUF header (``gguf_kv_bytes_per_token``) - equally
           exact and available BEFORE the load;
        3. ``_bytes_per_token(file size)`` - the size-class heuristic, used
           only for a file whose header cannot be read.

        Step 2 is memoised per instance: it reads a bounded prefix of the file.
        Returns 0 for a model that keeps no KV cache at all (a diffusion
        language model, see :meth:`_keeps_no_kv_cache`); never 0 otherwise -
        step 3's floor is 16 KB."""
        if self._keeps_no_kv_cache():
            return 0
        accurate = getattr(getattr(self, "_llm", None), "kv_bytes_per_token", 0)
        if accurate:
            return int(accurate)
        cached = getattr(self, "_gguf_kv_bpt", None)
        if cached is None:
            from localm.model_manager.gguf import gguf_kv_bytes_per_token
            try:
                cached = int(gguf_kv_bytes_per_token(Path(self.model_path)))
            except Exception as exc:  # contracted not to raise - surface if it does
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf KV-shape probe failed (%s); falling back to the "
                           "size-class estimate", type(exc).__name__)
                cached = 0
            self._gguf_kv_bpt = cached
        if cached:
            return cached
        return self._bytes_per_token(self._model_bytes())

    def _keeps_no_kv_cache(self) -> bool:
        """True for a model llama.cpp creates no KV cache for: a diffusion
        language model, which re-reads its whole canvas every step. True once a
        load reported one (``_diffusion_loaded``); otherwise read from the
        loaded model when there is one, else from the file's
        ``general.architecture`` (memoised per instance)."""
        if getattr(self, "_diffusion_loaded", False) is True:
            return True
        loaded = getattr(getattr(self, "_llm", None), "is_diffusion", None)
        if isinstance(loaded, bool):
            return loaded
        cached = getattr(self, "_no_kv_cache", None)
        if cached is None:
            from localm.model_manager.gguf import (gguf_architecture,
                                                   gguf_is_diffusion_architecture)
            path = getattr(self, "model_path", None)
            cached = bool(path) and gguf_is_diffusion_architecture(
                gguf_architecture(Path(path)))
            self._no_kv_cache = cached
        return cached

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
        size).

        On a box with an applied multi-GPU split, ``free``/``total`` here are
        the split's COMBINED figures (see _split_free_total_bytes) and the
        refusal wording names the split, not "this GPU".
        """
        # The resolved offload count when load() already picked it, else the
        # configured value.
        gpu_layers = (self.effective_gpu_layers
                      if self.effective_gpu_layers is not None
                      else self.n_gpu_layers)
        if gpu_layers == 0:
            return  # CPU-only run, VRAM is irrelevant
        # Budget against the COMBINED capacity of an applied multi-GPU split:
        # weights and per-layer KV both spread across the split devices.
        free, total, split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
            if free is None:
                return  # can't measure (no torch / no GPU) - nothing useful to say
            total = self._total_vram_bytes()
            split_devices = 1   # single-device reading - the flat overhead
        # An n_cpu_moe load pins its routed-expert weights to system RAM, where
        # they never draw on this budget at all.
        model_bytes = self._effective_model_bytes_for_vram()
        kv_cache = self.n_ctx * self._kv_bytes_per_token()
        # Charge only the offloaded fraction of the weights; a full or "all"
        # load (>= 99) charges the entire weight.
        if gpu_layers >= self._DEFAULT_GPU_LAYERS:
            weights = model_bytes
        else:
            layers = self._cached_layer_count() or self._ASSUMED_LAYERS
            weights = int(model_bytes * min(1.0, gpu_layers / layers))
        overhead = (self._split_overhead_bytes(split_devices)
                    + self._spec_extra_vram_bytes()
                    + self._recurrent_state_vram_bytes())
        need = weights + kv_cache + overhead
        ctx_hint = f"weights + a {self.n_ctx:,}-token KV cache + buffers"
        if total is not None and need > total:
            # On a split box the ceiling exceeded is the split's combined one.
            ceiling = (
                f"the {split_devices} GPUs in the configured split only have "
                f"{total / 1024**3:.1f} GB combined - freeing other VRAM will "
                f"not help, it cannot fit across this split"
                if split_devices >= 2 else
                f"this GPU only has {total / 1024**3:.1f} GB total - freeing "
                f"other VRAM will not help, it cannot fit regardless"
            )
            options = "".join(
                f"    - {label}:  {command}\n"
                for label, command in self._vram_fit_options(
                    total, need, kv_cache, overhead, gpu_layers))
            raise RuntimeError(
                f"Context too large for available VRAM: this load needs "
                f"roughly {need / 1024**3:.1f} GB ({ctx_hint}) but "
                f"{ceiling}.\n"
                f"  Options:\n"
                f"{options}"
                f"    - Let localm auto-size GPU offload:  "
                f"localm config n_gpu_layers_auto true\n"
                f"    - Let localm auto-size the context:  "
                f"localm config ctx_auto true"
            )
        if free >= need:
            return
        where = (f" across the {split_devices} GPUs in the configured split"
                 if split_devices >= 2 else "")
        # The quoted GB figure can still be the raw, cross-process-blind reading
        # when the device-global correction declined.
        blind_note = ("  [yellow](this reading may not see other processes' "
                      "VRAM use)[/yellow]" if self._free_reading_may_be_blind() else "")
        options = "".join(
            f"    • {label}:  [bold]{command}[/bold]\n"
            for label, command in self._vram_fit_options(
                free, need, kv_cache, overhead, gpu_layers))
        console.print(
            f"[yellow]⚠ Low VRAM:[/yellow] this model needs roughly "
            f"[bold]{need / 1024**3:.1f} GB[/bold] ({ctx_hint}) but only "
            f"[bold]{free / 1024**3:.1f} GB[/bold] is free{where}.{blind_note}\n"
            f"  [dim]Likely cause: {self._vram_holder_hint()}[/dim]\n"
            f"  Options:\n"
            f"    • Free VRAM first (close the other app, or POST "
            f"/v1/models/unload on its server)\n"
            f"{options}"
            f"  Continuing anyway - load may be slow or fail."
        )

    # Smallest context a "lower the context" suggestion names.
    _FIT_HINT_MIN_CTX = 1024

    def _vram_fit_options(self, budget: int, need: int, kv_cache: int,
                          overhead: int, gpu_layers: int) -> "list[tuple[str, str]]":
        """``(label, command)`` suggestions for a load needing *need* bytes of
        a *budget* it exceeds, given its *kv_cache* and *overhead* charges:

        - a context that fits *budget*, rounded down to whole KiB of tokens
          and below the current n_ctx, when one of at least
          ``_FIT_HINT_MIN_CTX`` tokens fits;
        - for a Mixture-of-Experts model loaded with every layer on the GPU,
          the smallest n_cpu_moe that fits *budget*, when one does and it is
          above this load's;
        - fewer GPU layers."""
        options = []
        per_token = self._kv_bytes_per_token() + self._spec_kv_per_token()
        fixed = need - kv_cache - self.n_ctx * self._spec_kv_per_token()
        if per_token > 0 and budget > fixed:
            ctx = ((budget - fixed) // per_token // 1024) * 1024
            ctx = min(ctx, ((self.n_ctx - 1) // 1024) * 1024)
            if ctx >= self._FIT_HINT_MIN_CTX:
                options.append(("Lower the context", f"-c {ctx}"))
        if gpu_layers >= self._DEFAULT_GPU_LAYERS:
            n = self._smallest_fitting_n_cpu_moe(budget, kv_cache, overhead)
            if n is not None and n > self._load_n_cpu_moe():
                options.append(("Keep MoE experts in system RAM",
                                f"localm config n_cpu_moe {n}"))
        options.append(("Offload fewer layers", "-g 24  (or -g 0 for CPU-only)"))
        return options

    def _smallest_fitting_n_cpu_moe(self, budget: int, kv: int,
                                    overhead: int) -> Optional[int]:
        """The smallest n_cpu_moe for which every layer on the GPU fits
        *budget* alongside *kv* and *overhead*, or None when no n_cpu_moe makes
        it fit (or the model has no routed experts). Pinning layer i's experts
        frees exactly their bytes (``_moe_expert_bytes_by_layer``)."""
        by_layer = self._moe_expert_bytes_by_layer()
        if not by_layer:
            return None
        weights = self._vram_model_bytes(0)
        for n in range(1, max(by_layer) + 2):
            weights -= by_layer.get(n - 1, 0)
            if weights + kv + overhead <= budget:
                return n
        return None

    def _check_context_fit(self, n_ctx: int, current_ctx: int = 0) -> Optional[bool]:
        """Decide WHERE the KV cache for a context of *n_ctx* tokens must live -
        wired as LlamaCpp's ``vram_check`` hook, consulted by
        ``_prefill_fresh_context()`` before it (re)creates a bigger context.

        Returns True to keep the KV cache in VRAM (``offload_kqv`` - full speed),
        False to place it in SYSTEM RAM (slower, but keeps the FULL context window),
        or None when VRAM is unmeasurable / irrelevant (caller keeps the default,
        VRAM). It NEVER shrinks the window and NEVER raises: when the KV cache does
        not fit free VRAM it moves to RAM and generation runs slower. A genuine
        can't-fit-even-in-RAM case is still surfaced by
        ``_prefill_fresh_context``'s NULL-pointer check on the native context.

        The charge depends on WHERE the currently-resident KV lives. When it is
        in VRAM, only the NET growth is charged: recreation frees that KV back to
        VRAM, and ``free`` was measured with it still resident. When a prior grow
        already moved the KV to system RAM, the GPU holds none, so the FULL target
        is charged. Weights and the compute buffers are already resident and do
        not change with the context length (n_batch is unchanged), so neither is
        charged.
        """
        # Gate on the RESOLVED offload count, not the raw configured
        # n_gpu_layers: a CPU-only auto load (effective 0) still carries
        # n_gpu_layers==99.
        gpu_layers = (self.effective_gpu_layers
                      if self.effective_gpu_layers is not None
                      else self.n_gpu_layers)
        if gpu_layers == 0:
            return None  # CPU-only run: KV already lives in RAM, nothing to decide
        # Combined split budget first (per-layer KV spreads across the split
        # devices with the weights - see _check_vram). Inside the worker the
        # helper answers (None, None, 0) without probing.
        free, _split_total, _split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
        if free is None:
            return None  # can't measure (no torch / no GPU) - keep the default (VRAM)
        # KV bytes per token on the GPU: only the offloaded layers keep their KV
        # in VRAM (offload_kqv); CPU layers' KV lives in system RAM, so a partial
        # load is charged only its GPU share.
        per_token = self._kv_bytes_per_token()
        if gpu_layers < self._DEFAULT_GPU_LAYERS:
            layers = self._cached_layer_count() or self._ASSUMED_LAYERS
            per_token = int(per_token * min(1.0, gpu_layers / layers))
        per_token += self._spec_kv_per_token()
        if per_token <= 0:
            return None
        # How much NEW KV must land in VRAM to grow to n_ctx depends on WHERE the
        # currently-resident KV lives (the recreate frees it first):
        #  - current KV in VRAM: freeing it returns that KV to the VRAM pool and
        #    `free` was measured with it still resident, so the NET growth is
        #    the charge (delta <= free  <=>  full target <= budget).
        #  - current KV already in SYSTEM RAM (offload_kqv=False): the GPU holds
        #    NO KV, `free` already reads the whole KV budget, and the FULL
        #    target is charged.
        # current_ctx==0 means "not told" (a direct or test call), so the base
        # n_ctx stands in.
        kv_in_ram = getattr(self._llm, "_offload_kqv", True) is False
        current = max(int(current_ctx), self.n_ctx)
        charge = (n_ctx * per_token if kv_in_ram
                  else (n_ctx - current) * per_token)   # NET when the old VRAM KV is reclaimed
        # State the decision - nothing else in the worker surfaces it.
        from localm.debuglog import logger as _dbg
        _dbg.debug("ctx-grow fit: target=%d free=%d per_token=%d charge=%d "
                   "kv_in_ram=%s -> KV in %s", n_ctx, free, per_token, charge,
                   kv_in_ram, "VRAM" if charge <= free else "system RAM")
        if charge <= free:
            return True                                # KV cache fits VRAM - keep it there
        # Does not fit VRAM: keep the FULL window and put the KV cache in system
        # RAM. The hint is emitted once per loaded-model session.
        if not self._ram_kv_hint_shown:
            self._ram_kv_hint_shown = True
            from localm.debuglog import logger as _dbg
            # The quoted GB figure can still be the raw, cross-process-blind
            # reading when the device-global correction declined.
            blind_note = (" (this reading may not see other processes' VRAM use)"
                          if self._free_reading_may_be_blind() else "")
            _dbg.warning(
                "large context (%s tokens): the KV cache does not fit free VRAM "
                "(need %.2f GB > %.2f GB free)%s, so it is kept in system RAM and "
                "generation will be slower. Free VRAM or lower n_ctx_max for "
                "full-speed GPU KV cache.",
                f"{n_ctx:,}", charge / 1024**3, free / 1024**3, blind_note)
        return False

    # Bounds for VRAM-derived context ceilings
    _AUTO_CTX_MIN = 4096
    _AUTO_CTX_MAX = 65536
    _AUTO_CTX_FALLBACK = 16384   # no GPU visibility - match common practice

    def _auto_ctx_max(self, capped: bool = True,
                      split_budget: "Optional[tuple[int, int]]" = None) -> int:
        """
        Derive a context ceiling from available resources.

        Budget = free VRAM - model weights - fixed overhead - the configured
        embedder's expected footprint (see embedder_ctx_reservation_bytes).
        The KV cost per token is estimated from the model's size class
        (larger models have more layers and wider KV heads; sliding-window
        models need less, so the estimate stays conservative). The result is
        clamped to a sane range and rounded to whole KiB of tokens.

        The reservation applies HERE only, not in _auto_gpu_layers: chat
        weights keep VRAM priority and the context window is the flexible
        resource.

        ``capped`` applies the _AUTO_CTX_MAX safety clamp on top of the crude
        KV estimate. It is lifted only when the user explicitly asked for an
        unlimited ceiling (n_ctx_max=0): then "auto" means the full VRAM-derived
        budget. The _AUTO_CTX_MIN floor always applies.

        On an applied multi-GPU split the budget starts from the split's
        COMBINED free VRAM (see _split_free_total_bytes), and the embedder
        reservation is deducted from that combined budget - a GPU-placed
        embedder is itself tensor-split across the same devices, so its
        footprint draws on the combined pool.

        ``split_budget`` = ``(free, devices)``, when given, replaces that
        reading: the free VRAM summed over the devices the load uses and how
        many there are (the implicit split fit's kept devices).
        """
        if split_budget is not None:
            free, split_devices = split_budget
        else:
            free, _split_total, split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
            split_devices = 1   # single-device reading - the flat overhead
        if free is None:
            return self._AUTO_CTX_FALLBACK
        # An n_cpu_moe load's pinned expert weights never draw on this budget.
        model = self._effective_model_bytes_for_vram()
        budget = (free - model - self._split_overhead_bytes(split_devices)
                  - embedder_ctx_reservation_bytes()
                  - self._spec_extra_vram_bytes()
                  - self._recurrent_state_vram_bytes())
        per_token = self._kv_bytes_per_token()
        if budget <= 0 or per_token <= 0:
            return max(self.n_ctx, self._AUTO_CTX_MIN)
        auto = budget // per_token
        auto = (auto // 1024) * 1024
        hi = auto if not capped else min(self._AUTO_CTX_MAX, auto)
        return int(max(self._AUTO_CTX_MIN, hi))

    def _effective_ctx_max(self, split_budget: "Optional[tuple[int, int]]" = None
                           ) -> Optional[int]:
        """The context ceiling to use for this load (auto or configured).

        ctx_auto sizes the ceiling from free VRAM. n_ctx_max==0 means the user
        asked for NO fixed ceiling ("grow until VRAM"); combined with ctx_auto
        that lifts the conservative _AUTO_CTX_MAX safety clamp so the window can
        use the full VRAM-derived budget. When ctx_auto is off, n_ctx_max is used
        verbatim (0/None already mean unlimited downstream). ``split_budget`` is
        passed to :meth:`_auto_ctx_max`. A model that keeps no KV cache (a
        diffusion model) gets None: its fixed window is reported by the worker
        at load."""
        if self._keeps_no_kv_cache():
            return None
        if self.ctx_auto:
            unlimited = (self.n_ctx_max == 0)
            auto = self._auto_ctx_max(capped=not unlimited, split_budget=split_budget)
            extra = "; no max (n_ctx_max=0)" if unlimited else ""
            console.print(
                f"[dim]  ctx auto : window may grow to {auto:,} tokens "
                f"(from free VRAM{extra})[/dim]"
            )
            return auto
        return self.n_ctx_max

    # The "offload everything" sentinel: n_gpu_layers left at this value means
    # the user did NOT pin a specific layer count, so auto may size it.
    _DEFAULT_GPU_LAYERS = 99
    # Layer count assumed for a model that has never been loaded (true count not
    # cached yet). The true count is cached after the first load (model_meta)
    # and used from then on.
    _ASSUMED_LAYERS = 32

    def _cached_layer_count(self) -> Optional[int]:
        """The model's true transformer layer count if a prior load cached it,
        else None (never loaded yet - the caller falls back to _ASSUMED_LAYERS)."""
        from localm.model_meta import cached_n_layers
        return cached_n_layers(self.model_path)

    def _full_offload_parts(self, split_devices: int) -> "tuple[int, int, int]":
        """``(model, kv, overhead)`` bytes a full GPU offload of this load
        charges: the VRAM-resident weights with the CONFIGURED n_cpu_moe
        (``_vram_model_bytes``; never an automatic choice of an earlier
        load), the KV cache for ``self.n_ctx`` tokens, and the compute overhead
        for *split_devices* devices plus the draft source (MTP draft context or
        draft model) and the recurrent state."""
        model = self._vram_model_bytes(int(getattr(self, "n_cpu_moe", 0) or 0))
        kv = self.n_ctx * self._kv_bytes_per_token()
        overhead = (self._split_overhead_bytes(split_devices)
                    + self._spec_extra_vram_bytes()
                    + self._recurrent_state_vram_bytes())
        return model, kv, overhead

    def full_offload_vram_bytes(self) -> Optional[int]:
        """Free VRAM this load needs for every layer to go on the GPU: the
        ``model + kv + overhead`` that ``_auto_gpu_layers_budget`` compares
        free VRAM with, for the devices the load spreads over
        (``_split_free_total_bytes``, one when no combined reading applies).

        None when ``n_gpu_layers`` asks for fewer than all layers or the model
        file's size cannot be read. Reads the GGUF header and may take a GPU
        reading, so it must not run on an event loop thread."""
        if self.n_gpu_layers < self._DEFAULT_GPU_LAYERS or self._model_bytes() <= 0:
            return None
        _free, _total, split_devices = self._split_free_total_bytes()
        model, kv, overhead = self._full_offload_parts(split_devices or 1)
        return model + kv + overhead

    def _auto_gpu_layers_budget(self) -> Optional[_AutoLayerBudget]:
        """The full computation behind ``_auto_gpu_layers()``: the same
        layer-count decision, plus the free/total/model/kv/overhead breakdown
        a partial-offload notice needs to name the actual cause. None under
        the exact same "VRAM unmeasurable" condition ``_auto_gpu_layers()``
        returns None for.

        "Free VRAM" is the COMBINED free across every device the load will
        actually spread over when that is measurable (see
        _split_free_total_bytes) - a CONFIGURED split, and equally the IMPLICIT
        one llama.cpp performs by default on any multi-GPU box.

        For a Mixture-of-Experts model with no configured ``n_cpu_moe``, the
        routed experts move to system RAM before any layer does:
        ``moe_cpu_layers`` is the smallest n_cpu_moe that fits every layer on
        the GPU (``layers`` 99).

        - On one GPU it is fitted against the free reading, and only when the
          whole model does not fit.
        - On 2+ GPUs under llama.cpp's implicit split it is fitted against every
          device's own charge and the combined reading
          (:meth:`_split_fitting_n_cpu_moe`), also when the combined reading
          fits the whole model. When no n_cpu_moe fits every
          device and the combined reading fits, nothing is pinned. When that
          per-device fit cannot be made, the experts of every layer are pinned
          if that fits the combined reading.
        - On a configured ``gpu_split_indices`` split nothing is pinned
          automatically.

        Otherwise, when the model does not fit, the experts of every layer stay
        in system RAM and ``layers`` is sized over the remaining weights
        (``moe_cpu_layers`` 0 when ``layers`` is 0). ``model`` is always the
        VRAM-resident weight bytes WITHOUT the automatic choice."""
        free, total, split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
            if free is None:
                return None                   # unmeasurable - honest fallback (A0)
            total = self._total_vram_bytes()
            split_devices = 1                 # single-device reading - the flat overhead
        if self._model_bytes() <= 0:
            # Can't size - attempt full offload. model/kv/overhead are left 0:
            # nothing that reaches this needs them (auto >= 99 skips the notice).
            return _AutoLayerBudget(self._DEFAULT_GPU_LAYERS, free, total, 0, 0, 0,
                                     split_devices)
        # Only the EXISTENCE check above needs the raw file size; an n_cpu_moe
        # load's pinned expert weights never draw on this budget.
        auto_moe = int(getattr(self, "n_cpu_moe", 0) or 0) <= 0
        model, kv, overhead = self._full_offload_parts(split_devices)
        split_checked = False
        if auto_moe and split_devices >= 2 and self._moe_expert_bytes_by_layer():
            split_checked, fitting = self._split_fitting_n_cpu_moe(free, kv, overhead)
            if fitting is not None:
                return _AutoLayerBudget(self._DEFAULT_GPU_LAYERS, free, total,
                                        model, kv, overhead, split_devices,
                                        moe_cpu_layers=fitting)
            if not split_checked and self._gpu_split_configured():
                auto_moe = False
        if model <= 0 or free >= model + kv + overhead:
            return _AutoLayerBudget(self._DEFAULT_GPU_LAYERS, free, total, model,
                                    kv, overhead, split_devices)
        moe_cpu_layers = 0
        weights = model
        by_layer = self._moe_expert_bytes_by_layer() if auto_moe else {}
        if by_layer:
            all_layers = max(by_layer) + 1
            if split_devices < 2:
                fitting = self._smallest_fitting_n_cpu_moe(free, kv, overhead)
            elif (not split_checked
                  and free >= self._vram_model_bytes(all_layers) + kv + overhead):
                fitting = all_layers
            else:
                fitting = None
            if fitting is not None:
                return _AutoLayerBudget(self._DEFAULT_GPU_LAYERS, free, total,
                                        model, kv, overhead, split_devices,
                                        moe_cpu_layers=fitting)
            moe_cpu_layers = all_layers
            weights = self._vram_model_bytes(moe_cpu_layers)
        weight_budget = free - kv - overhead
        if weight_budget <= 0 or weights <= 0:
            layers = 0                     # no room even for one layer's share
        else:
            fraction = min(max(weight_budget / weights, 0.0), 1.0)
            layer_count = self._cached_layer_count() or self._ASSUMED_LAYERS
            layers = max(0, min(self._DEFAULT_GPU_LAYERS, int(fraction * layer_count)))
        if layers == 0:
            moe_cpu_layers = 0
        return _AutoLayerBudget(layers, free, total, model, kv, overhead, split_devices,
                                moe_cpu_layers=moe_cpu_layers)

    def _auto_gpu_layers(self) -> Optional[int]:
        """Pick how many layers to offload to the GPU from free VRAM, or None when
        VRAM is not measurable by ANY path - neither torch.cuda nor the isolated
        native probe (_free_vram_bytes' loader.gpu_memory_isolated() fallback,
        which answers independently of torch.cuda) can answer, e.g. no GPU is
        present at all or its backend/daemon is unreachable. The caller then
        falls back to the configured value.

        Returns 99 ("all") when the whole model plus its KV cache and overhead fit
        in free VRAM; otherwise the largest layer count whose weight share fits the
        GPU budget left after reserving the KV cache + overhead (conservative: the
        KV cache is charged wholly to the GPU). 0 means even that budget is gone -
        run entirely on CPU."""
        budget = self._auto_gpu_layers_budget()
        return budget.layers if budget is not None else None

    def _auto_gpu_layers_cause(self, budget: _AutoLayerBudget) -> "tuple[str, bool]":
        """One-line explanation for why ``_effective_gpu_layers()`` sized a
        PARTIAL offload (``0 <= budget.layers < 99``), and whether the
        free-VRAM-blind caveat applies to it. Checked in priority order:

        (a) the model's weights alone already exceed this GPU's TOTAL capacity
            - no amount of freed VRAM or a smaller context would help;
        (b) weights + this context's KV cache WOULD fit a clean card (total),
            but not the VRAM actually free right now - something else is
            holding it;
        (c) on 2+ GPUs whose combined free reading holds the whole model: one
            GPU cannot hold its own share of it;
        (d) none of the above: the KV cache for the configured context is
            what tips an otherwise-fitting model over the free budget.

        Mirrors the same total-vs-free distinction ``_check_vram()`` already
        draws between "can never fit" and "something else is using the GPU",
        extended with the KV-specific case _check_vram has no need to separate
        (it already has a fixed n_ctx to charge in full). TOTAL is not subject
        to the cross-process blindness that only affects a FREE reading, so
        the caveat is reported False for case (a)."""
        total = budget.total
        if total is not None and budget.model + budget.overhead > total:
            where = (
                f"the {budget.split_devices} GPUs in the configured split only "
                f"have {total / 1024**3:.1f} GB combined"
                if budget.split_devices >= 2 else
                f"this GPU only has {total / 1024**3:.1f} GB total"
            )
            return (f"{where} - the model needs {budget.model / 1024**3:.1f} GB "
                    f"regardless of context size"), False
        if (total is not None and budget.free < total
                and budget.model + budget.kv > budget.free):
            hint = self._vram_holder_hint().rstrip(".")
            return (f"only {budget.free / 1024**3:.1f} of {total / 1024**3:.1f} "
                    f"GB free - {hint}"), True
        if (budget.split_devices >= 2
                and budget.free >= budget.model + budget.kv + budget.overhead):
            return (f"the {budget.split_devices} GPUs have {budget.free / 1024**3:.1f} "
                    f"GB free combined, but one of them cannot hold its own "
                    f"share of the model"), True
        return (f"the KV cache for a {self.n_ctx:,}-token context takes "
                f"{budget.kv / 1024**3:.1f} GB - lower n_ctx to fit more of "
                f"the model on the GPU"), True

    def _record_gpu_sizing(self, mode: str, layers: int,
                           budget: Optional[_AutoLayerBudget] = None, *,
                           cause: Optional[str] = None) -> None:
        """Store how ``_effective_gpu_layers()`` chose *layers* as
        ``self.last_gpu_sizing``, a dict with ``mode`` ("configured": auto off
        or an explicit n_gpu_layers; "unmeasurable": auto on, no VRAM reading;
        "auto": sized from free VRAM), ``layers``, ``n_ctx`` and ``n_cpu_moe``
        (the load's resolved value) with ``n_cpu_moe_auto`` (True when the
        automatic sizing chose it). An "auto" record also carries
        ``free_bytes``, ``total_bytes``, ``model_bytes``, ``kv_bytes`` and
        ``overhead_bytes`` from *budget*, and ``cause`` for a placement that
        left weights off the GPU."""
        record = {"mode": mode, "layers": layers, "n_ctx": self.n_ctx,
                  "n_cpu_moe": self._load_n_cpu_moe(),
                  "n_cpu_moe_auto": bool(budget is not None
                                         and budget.moe_cpu_layers > 0)}
        if budget is not None:
            record.update(free_bytes=budget.free, total_bytes=budget.total,
                          model_bytes=budget.model, kv_bytes=budget.kv,
                          overhead_bytes=budget.overhead)
        if cause is not None:
            record["cause"] = cause
        self.last_gpu_sizing = record

    def _moe_ram_bytes_per_token(self, n_cpu_moe: int) -> int:
        """Expert-weight bytes one generated token reads from system RAM when
        the experts of the first *n_cpu_moe* layers live there: their bytes
        times the share of experts the router selects per token
        (``expert_used_count / expert_count``). 0 for a dense model, for
        *n_cpu_moe* <= 0, or when the header does not declare both counts."""
        pinned = self._moe_pinned_bytes(n_cpu_moe)
        if pinned <= 0:
            return 0
        from localm.model_manager.gguf import gguf_expert_counts
        n_expert, n_used = gguf_expert_counts(Path(self.model_path))
        if n_expert <= 0 or n_used <= 0:
            return 0
        return pinned * min(n_used, n_expert) // n_expert

    def _host_resident_bytes(self) -> Optional[int]:
        """Weight bytes a load with every layer on the GPU keeps in system RAM:
        the model file's bytes minus :meth:`_vram_model_bytes` for this load's
        n_cpu_moe, so the input layer and any pinned routed experts. None when
        the file size cannot be read, or when n_cpu_moe is
        above 0 and the per-block byte probe failed. An input-layer probe
        failure counts the input layer as 0 bytes here."""
        try:
            model_bytes = self._model_bytes()
        except OSError:
            return None
        if model_bytes <= 0:
            return None
        n_cpu_moe = self._load_n_cpu_moe()
        if n_cpu_moe > 0 and not self._block_bytes():
            return None
        return max(0, model_bytes - self._vram_model_bytes(n_cpu_moe))

    @staticmethod
    def _layers_reach_a_gpu(gpu_readings) -> bool:
        """Whether a load's GPU layers can land on a GPU: False when the native
        runtime ships no GPU backend library
        (``discover.native_backend_has_gpu``); else, on a build whose GPU index
        space list_gpus cannot see (Vulkan, SYCL), whether the native device
        registry lists a device; else whether *gpu_readings* (this load's
        list_gpus reading) is non-empty."""
        from localm import discover
        if discover.native_backend_has_gpu() is False:
            return False
        if discover._native_gpu_index_space_is_opaque():
            return bool(discover.native_gpu_devices())
        return bool(gpu_readings)

    def _resolve_use_mmap(self, gpu_layers: int, gpu_readings) -> MmapDecision:
        """This load's :class:`MmapDecision` for *gpu_layers* and the configured
        ``use_mmap`` (read via getattr, ``auto`` when absent), stored as
        ``last_mmap_decision`` and logged at debug.

        An auto load with *gpu_layers* at 99 whose layers cannot reach a GPU
        (:meth:`_layers_reach_a_gpu` on *gpu_readings*, this load's list_gpus
        reading) keeps the runtime default with reason ``no_gpu``. System RAM
        is read only for an auto load with every layer on a GPU. Never raises:
        a failing GPU check yields ``no_gpu`` and a failing host-resident byte
        count ``host_unknown``."""
        setting = getattr(self, "use_mmap", "auto")
        full = gpu_layers >= self._DEFAULT_GPU_LAYERS
        mode = setting.strip().lower() if isinstance(setting, str) else ""
        from localm.debuglog import logger as _dbg
        if mode in ("on", "off") or not full:
            decision = decide_use_mmap(setting, full, None, None, None)
        else:
            try:
                on_gpu = self._layers_reach_a_gpu(gpu_readings)
            except Exception as exc:
                _dbg.debug("mmap: GPU presence check failed (%s)", type(exc).__name__)
                on_gpu = False
            if not on_gpu:
                decision = MmapDecision(None, "no_gpu")
            else:
                try:
                    host = self._host_resident_bytes()
                except Exception as exc:
                    _dbg.debug("mmap: host-resident byte count failed (%s)",
                               type(exc).__name__)
                    host = None
                total = available = None
                if host is not None:
                    from localm.sysstats import system_ram
                    total, available = system_ram()
                decision = decide_use_mmap(setting, True, host, total, available)
        self.last_mmap_decision = decision
        _dbg.debug("mmap: use_mmap=%s -> %s (%s; host %s bytes, RAM available %s "
                   "of %s bytes)", setting, decision.use_mmap, decision.reason,
                   decision.host_bytes, decision.ram_available, decision.ram_total)
        return decision

    def _effective_gpu_layers(self) -> int:
        """The n_gpu_layers this load will actually use, and the n_cpu_moe it
        uses (``effective_n_cpu_moe``).

        Auto only acts when it is ON and the user left n_gpu_layers at the
        "everything" default (99): an explicit value (e.g. -g 24) is honoured
        verbatim, and so is a configured n_cpu_moe above 0. For a
        Mixture-of-Experts model that does not fit, auto keeps routed experts
        in system RAM before it moves whole layers to the CPU (see
        :meth:`_auto_gpu_layers_budget`). Whenever auto leaves weights off the
        GPU it prints a one-line notice naming the actual cause and logs it at
        WARNING. When VRAM is unmeasurable it says so and attempts the
        configured value.

        Records the decision in ``last_gpu_sizing`` (see
        :meth:`_record_gpu_sizing`)."""
        self.effective_n_cpu_moe = int(getattr(self, "n_cpu_moe", 0) or 0)
        if not self.n_gpu_layers_auto or self.n_gpu_layers != self._DEFAULT_GPU_LAYERS:
            # Auto off, or an explicit choice: respected as-is.
            self._record_gpu_sizing("configured", self.n_gpu_layers)
            return self.n_gpu_layers
        budget = self._auto_gpu_layers_budget()
        if budget is None:
            # Unmeasurable VRAM still needs a working default: attempt the
            # configured value. A debug line rather than a per-load console
            # notice; a full offload that then does not fit fails loudly in
            # load()'s except handler.
            from localm.debuglog import logger as _dbg
            _dbg.debug("gpu layers auto: VRAM not measurable; using configured "
                       "n_gpu_layers=%s", self.n_gpu_layers)
            self._record_gpu_sizing("unmeasurable", self.n_gpu_layers)
            return self.n_gpu_layers
        if budget.moe_cpu_layers > 0:
            self.effective_n_cpu_moe = budget.moe_cpu_layers
        auto = budget.layers
        if auto >= self._DEFAULT_GPU_LAYERS and budget.moe_cpu_layers == 0:
            self._record_gpu_sizing("auto", auto, budget)
            return auto                        # full offload fits - no scary notice
        count = self._cached_layer_count()
        of = f"{count}" if count else f"~{self._ASSUMED_LAYERS} (estimated)"
        cause, maybe_blind = self._auto_gpu_layers_cause(budget)
        self._record_gpu_sizing("auto", auto, budget, cause=cause)
        blind = maybe_blind and self._free_reading_may_be_blind()
        blind_note = ("  [yellow](this reading may not see other processes' "
                      "VRAM use)[/yellow]" if blind else "")
        by_layer = self._moe_expert_bytes_by_layer()
        moe_layers = len(by_layer)
        pinned_layers = sum(1 for layer in by_layer if layer < budget.moe_cpu_layers)
        if auto >= self._DEFAULT_GPU_LAYERS:
            what = (f"every layer on the GPU, with the routed experts of "
                    f"{pinned_layers}/{moe_layers} layers in system RAM "
                    f"(Mixture-of-Experts)")
            override = "Set n_cpu_moe or n_gpu_layers to override"
        elif budget.moe_cpu_layers > 0:
            what = (f"offloading {auto}/{of} layers to the GPU, the rest on CPU "
                    f"(slower), with every layer's routed experts in system RAM "
                    f"(Mixture-of-Experts)")
            override = "Set n_cpu_moe or n_gpu_layers to override"
        else:
            what = (f"offloading {auto}/{of} layers to the GPU, the rest on CPU "
                    f"(slower)")
            override = "Set n_gpu_layers to override"
        console.print(
            f"[yellow]  gpu layers auto:[/yellow] {what} - {cause}.{blind_note} "
            f"{override}, or n_gpu_layers_auto false."
        )
        from localm.debuglog import logger as _dbg
        _dbg.warning("gpu layers auto: %s: %s - %s%s", Path(self.model_path).name,
                     what, cause, " (the free reading may not see other processes' "
                     "VRAM use)" if blind else "")
        return auto
