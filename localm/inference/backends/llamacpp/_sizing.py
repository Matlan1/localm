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
from typing import Optional

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


class VramSizingMixin:
    """VRAM measurement, preflight checks, and GPU-layer/context auto-sizing.

    Mixed into both ``GgufBackend`` (parent) and ``GgufWorker`` (child); the
    module docstring lists which methods each side calls.

    Expects the host class to provide: ``model_path`` (str), ``n_ctx`` (int),
    ``n_gpu_layers`` (int, the configured/raw value), ``effective_gpu_layers``
    (Optional[int], the resolved value once known), ``ctx_auto`` (bool),
    ``n_ctx_max`` (Optional[int]), ``mtp_enabled`` (bool, read defensively via
    getattr so a test double may omit it), ``_llm`` (the loaded native model,
    or None - only read by ``_check_context_fit`` for
    ``kv_bytes_per_token``/``_offload_kqv``), and ``_ram_kv_hint_shown`` (bool,
    mutable one-time-hint guard).
    """

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
        """(free, total) bytes on the configured main GPU device (device 0 when
        unset - see main_gpu_index / discover.resolve_main_gpu_index), or
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
                from localm.config import load_config
                from localm.discover import resolve_main_gpu_index
                idx = resolve_main_gpu_index(load_config().get("main_gpu_index"))
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
        NVIDIA / Linux / Vulkan reads are left unchanged."""
        if total is None:
            return None
        try:
            from localm.gpu_usage import (device_global_used_bytes,
                                          raw_reading_is_process_scoped)
            if not raw_reading_is_process_scoped():
                return None
            from localm.config import load_config
            from localm.discover import resolve_main_gpu_index
            idx = resolve_main_gpu_index(load_config().get("main_gpu_index"))
            used = device_global_used_bytes([{"index": idx, "total": total}])
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
        not measurable. The hard physical ceiling: nothing can be freed to
        raise it, so a load that needs more than this can never fit on this
        device."""
        return cls._free_total_vram_bytes()[1]

    @classmethod
    def _split_free_total_bytes(cls) -> "tuple[Optional[int], Optional[int], int]":
        """``(free, total, devices)`` summed across the 2+ devices this load
        will actually spread over, or ``(None, None, 0)`` when no combined
        budget applies and the caller must fall back to the single-device
        readings above.

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
                if devices < 2 or free is None or total is None:
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

    # The MTP draft context is never created larger than this many tokens
    # regardless of the main n_ctx - matches llama.py's own
    # cp_mtp.n_ctx = min(n_ctx, 2048).
    _MTP_DRAFT_CTX_CAP = 2048

    def _mtp_draft_context_vram_bytes(self) -> int:
        """Extra VRAM llama.py's MTP draft context (cp_mtp) will need beyond
        the shared model weights, for THIS load - 0 when ``mtp_enabled`` is
        off or this GGUF is not eligible for one.

        Two charges: the draft context's own KV cache, sized to its capped
        n_ctx (``min(self.n_ctx, _MTP_DRAFT_CTX_CAP)``) and its own
        nextn/draft layer count rather than the whole stack; and a flat
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
                draft_ctx = min(self.n_ctx, self._MTP_DRAFT_CTX_CAP)
                kv_per_token = gguf_mtp_draft_kv_bytes_per_token(
                    path, nextn_layers)
                charge = draft_ctx * kv_per_token + self._VRAM_OVERHEAD_BYTES
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("mtp draft-context VRAM probe failed (%s); charging "
                       "nothing extra for it", type(exc).__name__)
            charge = 0
        self._mtp_draft_vram_bytes_cached = charge
        return charge

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
        from localm.model_manager import split_gguf_parts
        p = Path(self.model_path)
        parts = split_gguf_parts(p.name)
        if parts:
            return sum(
                (p.parent / part).stat().st_size
                for part in parts if (p.parent / part).is_file()
            )
        return p.stat().st_size if p.is_file() else 0

    def _effective_model_bytes_for_vram(self) -> int:
        """VRAM-resident weight bytes for THIS load: ``_model_bytes()``, minus
        whatever ``n_cpu_moe`` pins to SYSTEM RAM instead (see llama.py's
        ``_apply_cpu_moe`` - the routed-expert tensors of the first
        ``n_cpu_moe`` layers never touch VRAM at all).

        Computed via ``gguf_moe_pinned_expert_bytes``, which reads each pinned
        tensor's EXACT size from the file's own tensor-info offsets.

        Falls back to the unadjusted ``_model_bytes()`` when ``n_cpu_moe`` is
        unset or 0, the header cannot be parsed, or nothing in the pinned range
        matched (e.g. a dense model, where ``n_cpu_moe`` has no effect per
        ``_apply_cpu_moe``'s own ``gguf_expert_count() == 0`` guard).
        Memoised per instance: the file read happens once per load."""
        model_bytes = self._model_bytes()
        n_cpu_moe = getattr(self, "n_cpu_moe", 0) or 0
        if n_cpu_moe <= 0:
            return model_bytes
        pinned = getattr(self, "_gguf_moe_pinned_bytes", None)
        if pinned is None:
            from localm.model_manager.gguf import gguf_moe_pinned_expert_bytes
            try:
                pinned = gguf_moe_pinned_expert_bytes(
                    Path(self.model_path), n_cpu_moe)
            except Exception as exc:  # contracted not to raise - surface if it does
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf MoE expert-byte probe failed (%s); charging "
                           "the whole file for VRAM sizing", type(exc).__name__)
                pinned = None
            self._gguf_moe_pinned_bytes = pinned if pinned is not None else 0
        return max(0, model_bytes - self._gguf_moe_pinned_bytes)

    def _vram_holder_hint(self) -> str:
        """Best-effort: name a concrete live sibling localm instance holding
        VRAM on this same GPU device (port, model, how long ago it last
        updated), via the cross-install GPU-coordination registry
        (``localm.gpu_registry``) - instead of the generic "another GPU app"
        text. Falls back to the generic text when the registry is empty or
        unavailable or the lookup itself fails; purely a diagnostic, never
        load-blocking.

        ``gpu_registry.list_gpu_peers()`` always excludes THIS process
        (matched by pid), so ``holder`` below is always a genuinely different
        instance. When no external holder is found,
        :func:`gpu_registry.own_entry` reports whether THIS process's own
        registry entry explains it (e.g. this server has another model
        resident while loading a second one)."""
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
            self_entry = gpu_registry.own_entry()
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
        Never returns 0 - step 3's floor is 16 KB - so callers can divide by
        it."""
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
        need = (weights + kv_cache + self._split_overhead_bytes(split_devices)
                + self._mtp_draft_context_vram_bytes())
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
            raise RuntimeError(
                f"Context too large for available VRAM: this load needs "
                f"roughly {need / 1024**3:.1f} GB ({ctx_hint}) but "
                f"{ceiling}.\n"
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
        where = (f" across the {split_devices} GPUs in the configured split"
                 if split_devices >= 2 else "")
        # The quoted GB figure can still be the raw, cross-process-blind reading
        # when the device-global correction declined.
        blind_note = ("  [yellow](this reading may not see other processes' "
                      "VRAM use)[/yellow]" if self._free_reading_may_be_blind() else "")
        console.print(
            f"[yellow]⚠ Low VRAM:[/yellow] this model needs roughly "
            f"[bold]{need / 1024**3:.1f} GB[/bold] ({ctx_hint}) but only "
            f"[bold]{free / 1024**3:.1f} GB[/bold] is free{where}.{blind_note}\n"
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

    def _auto_ctx_max(self, capped: bool = True) -> int:
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
        """
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
                  - self._mtp_draft_context_vram_bytes())
        if budget <= 0:
            return max(self.n_ctx, self._AUTO_CTX_MIN)
        auto = budget // self._kv_bytes_per_token()
        auto = (auto // 1024) * 1024
        hi = auto if not capped else min(self._AUTO_CTX_MAX, auto)
        return int(max(self._AUTO_CTX_MIN, hi))

    def _effective_ctx_max(self) -> Optional[int]:
        """The context ceiling to use for this load (auto or configured).

        ctx_auto sizes the ceiling from free VRAM. n_ctx_max==0 means the user
        asked for NO fixed ceiling ("grow until VRAM"); combined with ctx_auto
        that lifts the conservative _AUTO_CTX_MAX safety clamp so the window can
        use the full VRAM-derived budget. When ctx_auto is off, n_ctx_max is used
        verbatim (0/None already mean unlimited downstream)."""
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
        run entirely on CPU.

        "Free VRAM" is the COMBINED free across every device the load will
        actually spread over when that is measurable (see
        _split_free_total_bytes) - a CONFIGURED split, and equally the IMPLICIT
        one llama.cpp performs by default on any multi-GPU box."""
        free, _split_total, split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
            split_devices = 1   # single-device reading - the flat overhead
        if free is None:
            return None                       # unmeasurable - honest fallback (A0)
        if self._model_bytes() <= 0:
            return self._DEFAULT_GPU_LAYERS    # can't size - attempt full offload
        # Only the EXISTENCE check above needs the raw file size; an n_cpu_moe
        # load's pinned expert weights never draw on this budget.
        model = self._effective_model_bytes_for_vram()
        kv = self.n_ctx * self._kv_bytes_per_token()
        overhead = (self._split_overhead_bytes(split_devices)
                    + self._mtp_draft_context_vram_bytes())
        if model <= 0 or free >= model + kv + overhead:
            return self._DEFAULT_GPU_LAYERS    # full offload fits (or nothing left to size)
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
        verbatim. When auto sizes a partial offload it prints a one-line
        notice. When VRAM is unmeasurable it says so and attempts the
        configured value."""
        if not self.n_gpu_layers_auto:
            return self.n_gpu_layers
        if self.n_gpu_layers != self._DEFAULT_GPU_LAYERS:
            return self.n_gpu_layers          # explicit choice - respect it as-is
        auto = self._auto_gpu_layers()
        if auto is None:
            # Unmeasurable VRAM still needs a working default: attempt the
            # configured value. A debug line rather than a per-load console
            # notice; a full offload that then does not fit fails loudly in
            # load()'s except handler.
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
