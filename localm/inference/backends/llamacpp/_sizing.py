# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM measurement and load-sizing logic shared by ``GgufBackend`` (the
parent-facing proxy) and ``GgufWorker`` (the child process that owns the real
native model).

Extracted verbatim from ``gguf.py`` (no behavior change) so both sides of the
subprocess split can use it without duplicating logic or cross-process calls:

- ``GgufBackend`` uses the whole mixin exactly as before - preflight VRAM
  checks and GPU-layer/context-size sizing all happen BEFORE a model process
  is even spawned, so a load that can never fit still fails fast without
  paying a process-spawn cost.
- ``GgufWorker`` uses only ``_check_context_fit`` (and its transitive
  dependencies) as the ``vram_check`` callback ``LlamaCpp._prefill_fresh_context``
  calls deep inside a native call stack during a mid-generation context grow -
  that callback cannot round-trip to the parent process without adding
  multi-second IPC latency to the decode hot path, so it must run wherever the
  loaded model itself lives.

None of these methods call the abort-prone ``llama_load_model_from_file``/
``llama_init_from_model``/``llama_decode`` - they only call
``torch.cuda.mem_get_info`` (exception-safe) or the already subprocess-isolated
``loader.gpu_memory_isolated()`` (see ``_loader.py``), so none of them need to
run inside the isolated worker for safety - only ``_check_context_fit`` needs
to be there, for the latency reason above.
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

    Without this, ``ctx_auto`` sized the window from ALL free VRAM at
    chat-load time, the embedder then found the card full, force-loaded
    anyway, and Windows/WDDM paged the overcommit to system RAM - measured
    live 2026-07-18 on a 16 GB card: generation collapsed from ~34 to 5.0
    tok/s with a 4.3 GB embedder actively embedding per turn, and recovered
    to 31.9 tok/s once the context was small enough for both to fit.

    Returns the embedder's expected footprint (file size + the 20% KV/compute
    slop every other embedder estimate in this codebase already uses), or 0
    when reserving would be wrong: the embedder is ALREADY loaded (the free
    reading is already paying for it - reserving again would double-count),
    no embedding model resolves (nothing will ever load), or anything fails
    (a reservation is an optimization and must never break a chat load -
    surfaced at debug, not swallowed silently).
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

    Mixed into both ``GgufBackend`` (parent) and ``GgufWorker`` (child) - see
    the module docstring for which methods each side actually calls.

    Expects the host class to provide: ``model_path`` (str), ``n_ctx`` (int),
    ``n_gpu_layers`` (int, the configured/raw value), ``effective_gpu_layers``
    (Optional[int], the resolved value once known), ``ctx_auto`` (bool),
    ``n_ctx_max`` (Optional[int]), ``_llm`` (the loaded native model, or None -
    only read by ``_check_context_fit`` for ``kv_bytes_per_token``/
    ``_offload_kqv``), and ``_ram_kv_hint_shown`` (bool, mutable one-time-hint
    guard).
    """

    # Rough VRAM headroom for KV cache + compute buffers beyond model weights.
    # Single-sourced from localm.vram so the loader, the GUI estimate, and the fit
    # badge all reason about "does it fit" with the same number; kept as a class
    # attribute (not read from the module directly) so tests can monkeypatch it.
    _VRAM_OVERHEAD_BYTES = VRAM_OVERHEAD_BYTES

    # BACKSTOP, not the primary defense: _free_total_vram_bytes() now skips the
    # torch attempt entirely whenever _loader.native_lib_loaded() is True (see
    # its docstring), which is the actual precondition for this conflict and
    # catches it before it happens on every ordinary worker. This flag remains
    # for any path that reaches `import torch` some other way (a direct call
    # in a test, or a future caller of this mixin that does not route through
    # _free_total_vram_bytes) - it still stops a genuinely-uncaught fault from
    # retrying and re-faulting on every subsequent call. A class attribute
    # (shared across every GgufBackend/GgufWorker instance in this process,
    # which is exactly the right scope: the conflict is process-wide, not
    # per-instance) so tests can reset it like other per-process caches in
    # this codebase.
    _torch_rocm_init_broken: bool = False

    @staticmethod
    def _free_total_vram_bytes() -> "tuple[Optional[int], Optional[int]]":
        """(free, total) bytes on the configured main GPU device (device 0 when
        unset - see main_gpu_index / discover.resolve_main_gpu_index), or
        (None, None) when not measurable. Shared by _free_vram_bytes() and
        _total_vram_bytes() so both read the same device in one call.

        Skips the attempt entirely once `import torch` has been CONFIRMED
        broken in this process (see below) - not merely "torch unavailable"
        (that case is a normal, cheap ImportError every call, harmless to
        retry), but a specific, root-caused DLL conflict that otherwise
        retries and re-faults on EVERY call for the rest of the process's
        life.

        ROOT CAUSE (confirmed live, reproduced on demand, not assumed):
        once llama.cpp's bundled HIP/ROCm runtime is loaded into this process
        (via `_loader.load_lib()` - both `GgufWorker` in the isolated worker
        process, which owns the real loaded model, and any other code path
        that reaches `load_lib()`, e.g. the embedder), a LATER `import torch`
        triggers torch's separate `rocm_sdk` package to `ctypes.CDLL()` its
        own ROCm library, which resolves to an incompatible DLL already
        present in this process's address space (`OSError: [WinError 127]
        The specified procedure could not be found` - a classic Windows DLL
        entry-point mismatch, not a driver race). ctypes wraps native DLL
        loads in Windows SEH specifically so a bad DLL raises a catchable
        exception instead of crashing the interpreter, which is why this is
        caught here rather than taking the whole process down - but Python's
        import machinery evicts a module that faults during import from
        `sys.modules`, so an uncached failure re-attempts (and re-faults) on
        every single subsequent VRAM check for the rest of the process's
        life, each one printing a scary "Windows fatal exception" diagnostic
        to stderr even though it is fully handled. Two entirely separate
        native ROCm/HIP runtimes (llama.cpp's bundled one and torch's own)
        cannot safely coexist in one process on this box - caching the
        failure stops the pointless, noisy retry loop without hiding the
        underlying incompatibility."""
        if VramSizingMixin._torch_rocm_init_broken:
            return None, None
        # ROOT-CAUSE FIX, not a catch: once llama.cpp's own native runtime is
        # already loaded IN THIS PROCESS (see _loader.native_lib_loaded -
        # true inside GgufWorker/the embedder's isolated worker, which loaded
        # it themselves to run their model), a torch import is KNOWN to hit
        # the DLL-identity conflict below on this platform/build combination -
        # not sometimes, every time, because each worker is a fresh process so
        # the _torch_rocm_init_broken cache above never carries over. Skip the
        # doomed attempt instead of triggering it and catching the aftermath:
        # gpu_memory_isolated() (this method's own caller falls back to it)
        # answers exactly as well without ever touching torch.
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            return None, None
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
            # Degrades honestly to (None, None) -> the isolated-probe fallback in
            # _free_vram_bytes. Surfaced at debug (rule 5) rather than a bare pass so
            # a driver hiccup on this read is diagnosable under --debug.
            from localm.debuglog import logger as _dbg
            _dbg.debug("torch.cuda.mem_get_info read failed (%s); falling back to "
                       "the isolated native VRAM probe", type(e).__name__)
        return None, None

    @classmethod
    def _free_vram_bytes(cls) -> Optional[int]:
        """Free VRAM in bytes on the GPU the model runs on, or None when not
        measurable.

        Prefers torch.cuda on the configured main GPU (honours main_gpu_index for
        a multi-GPU split) when torch is available - cheap, in-process, and
        exception-safe. Falls back to loader.gpu_memory_isolated() when torch
        cannot answer (Vulkan/Metal builds ship with no CUDA/ROCm torch at all;
        also covers a torch-less CPU build, or an NVIDIA box without a separately
        installed CUDA torch - see the note on this project's torch packaging
        below). Never calls loader.gpu_memory() directly in this process.

        WHY the fallback is the ISOLATED probe, not the direct in-process call:
        confirmed live, three times, on this exact code path (_check_vram ->
        here) - loader.gpu_memory()'s mem_fn() ctypes call
        (ggml_backend_dev_memory -> hipMemGetInfo) can hard-abort the WHOLE
        PROCESS ("Fatal Python error: Aborted") on a transient driver condition.
        llama.cpp's own CUDA/HIP error macro treats ANY such failure as
        unrecoverable and calls abort() in C - no Python try/except, including
        gpu_memory()'s own, can catch it - and this is a documented, recurring
        class of issue on ROCm/HIP generally (ollama/ollama#3840;
        ROCm/ROCm#5378 shows even torch.cuda.mem_get_info can hit the identical
        underlying "HIP error: invalid argument" under concurrent/containerized
        GPU access - the difference that matters here is torch surfaces it as a
        catchable exception, while the raw ggml call does not). Every model load
        (plus every context-grow check via _check_context_fit) goes through this
        method, so a probe that can crash the server on a routine, recoverable
        condition is unacceptable (RULE 5) - hence loader.gpu_memory_isolated(),
        which runs the identical query against a long-lived daemon subprocess
        (the same principle ollama - the most comparable llama.cpp-wrapping
        project - uses for all of its model-runner GPU work, applied here just
        to this one narrow call rather than the whole load; a DAEMON rather than
        a fresh subprocess per call because the cold-start cost - importing this
        binding and loading the ggml/HIP DLL - measured live at 1.9-7.9s, varying
        with OS/driver warm-up state - which a fresh-process-per-call design
        would pay on EVERY query).

        This project's own torch dependency (pyproject.toml) is an AMD-ROCm,
        Windows-only wheel - there is no NVIDIA CUDA or Linux/macOS torch wired in
        at all, so "torch unavailable" is the COMMON case here, not a rare edge:
        NVIDIA, Linux, macOS, and Vulkan/Metal users all reach the isolated
        fallback by default. See loader.gpu_memory()'s own docstring for the full
        account of why the direct call is retired from every automatic path.

        A classmethod (not staticmethod), so a subclass/test that monkeypatches
        ``_free_total_vram_bytes`` on itself is honoured here too - the internal
        cross-reference dispatches through ``cls``, not a hardcoded class name.

        CROSS-PROCESS CORRECTION: both reads above (torch and the isolated ggml
        probe) report ``total - THIS process's own allocations`` on Windows + an AMD
        ROCm/HIP build and are blind to every other process (measured; see
        dev-notes/vram-cross-process-blindness.md). A LOAD DECISION made on that blind
        number OVERSTATES free VRAM and overcommits exactly when another process (a
        game, ComfyUI, another model in its isolated worker) holds the card - the
        TDR / WDDM-spill case auto-offload exists to avoid. So the free reading is
        corrected to a device-global figure where the raw one is KNOWN blind; on
        every other platform (device-global by documentation, or unmeasured) it is
        kept unchanged. This is the DECISION half of the same fix #697 applied to the
        reporting surfaces - _auto_gpu_layers / _auto_ctx_max / _check_vram all read
        through here."""
        # One debug line per read, stating raw/source/corrected (rule 5): every
        # value here feeds a sizing decision, and the silent form of this chain
        # is what let the worker-side reading stay broken, invisibly, from the
        # worker split until 2026-07-22.
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

        Never raises: a correction that cannot be made must degrade to "use the
        uncorrected reading", never crash a model load (rule 5, the same fail-safe
        posture as the isolated probe). Only acts where
        gpu_usage.raw_reading_is_process_scoped() is True - Windows + a ROCm/HIP
        torch build, and equally the torch-less processes whose readings come from
        the resident bundled HIP runtime (the GGUF worker deciding a context
        grow: same measured blindness, answered via
        discover.native_hip_runtime_resident()) - so NVIDIA / Linux / Vulkan
        reads - device-global by documentation, or unmeasured - are left exactly
        as before."""
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

    @classmethod
    def _total_vram_bytes(cls) -> Optional[int]:
        """Total VRAM in bytes on the configured main GPU device, or None when
        not measurable. The hard physical ceiling: unlike free VRAM, nothing
        can be freed to raise it, so a load that needs more than this can
        never fit on this device - see _check_vram()."""
        return cls._free_total_vram_bytes()[1]

    @classmethod
    def _split_free_total_bytes(cls) -> "tuple[Optional[int], Optional[int], int]":
        """``(free, total, devices)`` summed across the 2+ devices a configured
        multi-GPU split will spread this load over, or ``(None, None, 0)`` when
        no combined budget applies and the caller must fall back to the
        single-device readings above.

        WHY a combined budget: ``discover.apply_gpu_split`` tensor-splits the
        WEIGHTS across the configured devices, and llama.cpp places each
        layer's KV cache on the device that holds the layer (confirmed in a
        live 2-device native load log: per-layer KV buffers landed on each
        split device in the split ratio) - so for a split load, weights AND KV
        both draw on the split's combined capacity, and budgeting the whole
        model against the single main GPU (what ``_free_vram_bytes`` /
        ``_total_vram_bytes`` measure) defeats the split's headline case: a
        model too big for one card that fits combined. The admission gate
        (``discover.vram_capacity`` + ``gpu_split_shortfall``, see
        ``http_server.switch_engine``) already learned this; this helper
        brings the backend's own deeper preflight to the same ceiling.

        Answers ``(None, None, 0)`` - "no combined budget, use the
        single-device reading" - in every case where a combined figure would
        be dishonest or unsafe to fetch:

        - The native llama/ggml runtime is loaded IN THIS PROCESS (we are the
          GgufWorker / an isolated child, where ``_check_context_fit`` runs
          mid-generation): ``discover.list_gpus``'s probe does ``import
          torch``, the exact DLL-identity conflict ``_free_total_vram_bytes``
          documents and guards against, and a multi-second GPU probe has no
          place deep in the decode path. No probe is attempted at all. The
          single-device fallback stays honest there by construction: the
          isolated native probe declines to answer on a 2+-GPU-device box
          (``_loader._resolve_gpu_memory``), so a genuinely split load can
          never be mis-judged against one device's reading in the worker.
        - No ``gpu_split_indices`` configured: the common single-GPU case,
          answered from config alone with NO hardware probe (mirrors
          ``vram_capacity``'s own short-circuit).
        - The probe did not complete fresh this call (non-``GPU_PROBE_OK``):
          the served figure may be a frozen last-known-good value, and sizing
          or refusing a load from stale data is the rule-5 gap the admission
          gate's probe-freshness contract exists to prevent
          (``gpu_split_shortfall``'s docstring). ``wait_for_inflight=True``
          with the default cold-init-tolerant deadline makes this rare (we
          are always off the event loop here - model loads run in an
          executor/CLI thread), so a non-OK status means a genuinely
          wedged/contended driver.
        - Fewer than 2 split devices are detected (``vram_capacity``'s
          ``combined_only`` contract returns ``{}``): the vulkan build's
          honest-unknown (list_gpus is structurally blind to its device
          space - GPU-SPLIT-VKINDEX), a degraded/stale split config, or a
          box where detection is unmeasurable. The single-device fallback is
          then today's behavior, which on those boxes is itself honestly
          unmeasurable or correct.
        - A test double patched ``vram_capacity`` without the opt-in kwargs
          (TypeError) or with a plain dict lacking the ``"devices"`` key:
          treated as "no combined reading" rather than crashing or
          mis-reading a single-GPU stand-in as combined (the same
          double-tolerance posture as ``vram._vram_free_reading``).

        Never raises: a combined reading is an upgrade, and failing to fetch
        one must never break a load that worked before it existed (surfaced
        at debug, rule 5)."""
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            return None, None, 0
        try:
            from localm.config import load_config
            cfg = load_config()
            if not cfg.get("gpu_split_indices"):
                return None, None, 0
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
                # A plain-dict double (no return_status support) models a
                # completed probe, as it always did before the status channel.
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

        On a box with an applied multi-GPU split, ``free``/``total`` here are
        the split's COMBINED figures (see _split_free_total_bytes) and the
        refusal wording names the split, not "this GPU" - a model that fits
        combined must not be refused against one card's capacity.
        """
        # Use the resolved offload count when load() already picked it (auto), else
        # the configured value (also covers a direct _check_vram() call in tests).
        gpu_layers = (self.effective_gpu_layers
                      if self.effective_gpu_layers is not None
                      else self.n_gpu_layers)
        if gpu_layers == 0:
            return  # CPU-only run, VRAM is irrelevant
        # Budget against the COMBINED capacity of an applied multi-GPU split:
        # weights and per-layer KV both spread across the split devices, so the
        # single main-GPU reading under-budgets exactly the split's headline
        # case (a model too big for one card that fits combined) - see
        # _split_free_total_bytes for the full contract and fallbacks.
        free, total, split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
            if free is None:
                return  # can't measure (no torch / no GPU) - nothing useful to say
            total = self._total_vram_bytes()
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
        if total is not None and need > total:
            # On a split box the single-GPU wording ("this GPU only has...")
            # would be factually wrong - the ceiling being exceeded is the
            # split's combined one, so say exactly that.
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
        console.print(
            f"[yellow]⚠ Low VRAM:[/yellow] this model needs roughly "
            f"[bold]{need / 1024**3:.1f} GB[/bold] ({ctx_hint}) but only "
            f"[bold]{free / 1024**3:.1f} GB[/bold] is free{where}.\n"
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
        # Combined split budget first (per-layer KV spreads across the split
        # devices with the weights - see _check_vram). Inside the WORKER - the
        # only production caller of this method - the helper answers
        # (None, None, 0) without probing (its native_lib_loaded guard: no
        # torch/list_gpus in the decode path), and the single-device fallback
        # below stays honest on a split box by construction: the isolated
        # native probe declines at 2+ GPU devices (_loader._resolve_gpu_memory),
        # so a split load is never mis-judged against one device's free here.
        free, _split_total, _split_devices = self._split_free_total_bytes()
        if free is None:
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
        # The decision, stated (rule 5): this runs in the worker where nothing
        # else surfaces what the grow was judged against.
        from localm.debuglog import logger as _dbg
        _dbg.debug("ctx-grow fit: target=%d free=%d per_token=%d charge=%d "
                   "kv_in_ram=%s -> KV in %s", n_ctx, free, per_token, charge,
                   kv_in_ram, "VRAM" if charge <= free else "system RAM")
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

        Budget = free VRAM - model weights - fixed overhead - the configured
        embedder's expected footprint (see embedder_ctx_reservation_bytes:
        the embedder loads AFTER the chat model, and a window sized without
        that reservation hands its room to the KV cache, so the later
        embedder load oversubscribes the card and WDDM-thrashes generation).
        The KV cost per token is estimated from the model's size class
        (larger models have more layers and wider KV heads; sliding-window
        models need less, so the estimate is deliberately conservative). The
        result is clamped to a sane range and rounded to whole KiB of tokens.

        The reservation applies HERE only, deliberately not in
        _auto_gpu_layers: chat weights keep VRAM priority (a partial chat
        offload to protect the embedder would slow the primary workload on
        every token), while the context window is the flexible resource -
        measured live, full weights + a smaller window + the embedder ran at
        31.9 tok/s where the unreserved window ran at 5.0.

        ``capped`` applies the _AUTO_CTX_MAX safety clamp on top of the crude
        KV estimate. It is lifted only when the user explicitly asked for an
        unlimited ceiling (n_ctx_max=0): then "auto" means the full VRAM-derived
        budget, honoring their "grow until VRAM" choice. The _AUTO_CTX_MIN floor
        always applies.

        On an applied multi-GPU split the budget starts from the split's
        COMBINED free VRAM (see _split_free_total_bytes): weights and per-layer
        KV both spread across the split, so the single main-GPU reading would
        collapse the ceiling to n_ctx on exactly the box the split exists for.
        """
        free, _split_total, _split_devices = self._split_free_total_bytes()
        if free is None:
            free = self._free_vram_bytes()
        if free is None:
            return self._AUTO_CTX_FALLBACK
        model = self._model_bytes()
        budget = (free - model - self._VRAM_OVERHEAD_BYTES
                  - embedder_ctx_reservation_bytes())
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
        VRAM is not measurable by ANY path - neither torch.cuda nor the isolated
        native probe (_free_vram_bytes' loader.gpu_memory_isolated() fallback,
        which answers independently of torch.cuda) can answer, e.g. no GPU is
        present at all or its backend/daemon is unreachable - the caller then
        falls back honestly to the configured value instead of guessing a
        precise offload it could not compute).

        Returns 99 ("all") when the whole model plus its KV cache and overhead fit
        in free VRAM; otherwise the largest layer count whose weight share fits the
        GPU budget left after reserving the KV cache + overhead (conservative: the
        KV cache is charged wholly to the GPU). 0 means even that budget is gone -
        run entirely on CPU, still a working (slow) load, the extreme end of the
        promised RAM offload.

        "Free VRAM" is the applied multi-GPU split's COMBINED free when one is
        measurable (see _split_free_total_bytes) - sizing a split load against
        the main GPU alone silently halved the offload on exactly the
        model-bigger-than-one-card case the split exists for."""
        free, _split_total, _split_devices = self._split_free_total_bytes()
        if free is None:
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
            # Unmeasurable VRAM (neither torch.cuda nor the isolated native probe
            # can answer - e.g. no GPU present, or the probe daemon itself is
            # unreachable) still needs a working default: offload via the display
            # driver, letting the model try to fit as configured. Keep it a
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
