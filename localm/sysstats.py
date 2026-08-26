# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-effort live system stats for the GUI hardware monitor.

Pure and NEVER raises: a probe that fails just omits its field, so the status
bar degrades gracefully (e.g. VRAM still shows on a box without psutil). Safe
to call on a short GUI poll - CPU/RAM are cheap; the two GPU-touching probes
below run throttled and single-flighted on their own background thread so a
slow driver/subprocess call never blocks the poll itself (see _vram/_gpu_util):
  * CPU % and RAM via psutil (the optional ``[monitor]`` extra)
  * VRAM via localm.discover.vram_capacity (combined across a configured
    multi-GPU split, else the single main GPU - torch -> nvidia-smi ->
    Windows registry)
  * GPU utilisation % best-effort via nvidia-smi (NVIDIA only; omitted elsewhere)
"""

from __future__ import annotations

import threading
import time


def _clamped_field_deltas(t1, t2):
    """Per-field ``t2 - t1`` from two same-shaped ``psutil.cpu_times()`` ntuples,
    each trimmed to >= 0. CPU-time counters are supposed to only increase, but
    some fields (Windows ``interrupt``/``dpc``, some Linux counters) can
    occasionally regress even while total CPU time moves forward, so each field is
    trimmed to zero independently - the same shape as psutil's own
    ``_cpu_times_deltas``."""
    return type(t2)(*(max(0.0, float(b) - float(a)) for a, b in zip(t1, t2)))


def _busy_total(times) -> tuple[float, float]:
    """``(busy, total)`` CPU-seconds from a ``psutil.cpu_times()``-shaped ntuple -
    either an absolute snapshot or a delta from :func:`_clamped_field_deltas` (the
    formula is identical either way; this mirrors psutil's own
    ``_cpu_busy_time``/``_cpu_tot_time``, which run on both shapes too).
    ``iowait``/``guest``/``guest_nice`` exist only on some platforms (Linux); the
    ``getattr`` defaults keep this correct on Windows and macOS, which have
    neither, so ``busy`` there is simply ``total - idle``."""
    total = float(sum(times))
    # Linux already counts guest/guest_nice inside user/nice; subtract them so the
    # total is not double-counted (mirrors psutil._cpu_tot_time).
    total -= float(getattr(times, "guest", 0.0) or 0.0)
    total -= float(getattr(times, "guest_nice", 0.0) or 0.0)
    # iowait is a wait, not work: fold it into idle like psutil._cpu_busy_time does.
    idle = float(times.idle) + float(getattr(times, "iowait", 0.0) or 0.0)
    return total - idle, total


class _CpuMeter:
    """Non-blocking CPU-utilisation meter that keeps its OWN previous snapshot.

    ``psutil.cpu_percent(interval=None)`` measures CPU% "since the previous call"
    off psutil's process-global state, and the GUI polls /api/stats off a
    MULTI-thread executor (get_plugin_executor), which chops that shared window
    into unpredictable slices.

    This meter does not use that implicit state. It stores its own previous
    ``(times, monotonic)`` snapshot and derives the percent from the CLAMPED
    per-field delta over a real, known window (see :func:`_clamped_field_deltas`).
    The first reading (no baseline yet) reports ``None`` so the caller omits the
    CPU field for that one poll instead of showing a made-up number."""

    # Two samples closer than this (seconds) form too short a window to trust -
    # rapid or concurrent polls, where the delta is noise; reuse the last value.
    _MIN_INTERVAL = 0.1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prev: tuple[object, float] | None = None  # times, monotonic
        self._last: float | None = None                  # last computed percent

    def percent(self, times, now: float) -> float | None:
        """CPU% over the interval since the previous sample, or ``None`` when there
        is no trustworthy baseline yet. *times* is a ``psutil.cpu_times()`` ntuple;
        *now* is a monotonic timestamp. Deterministic given its inputs plus the prior
        state, so it is unit-testable with fabricated snapshots."""
        with self._lock:
            prev = self._prev
            if prev is None:                       # first poll: seed, no baseline yet
                self._prev = (times, now)
                return None
            prev_times, prev_now = prev
            if now - prev_now < self._MIN_INTERVAL:
                return self._last                  # window too short; keep last value
            self._prev = (times, now)
            busy_delta, total_delta = _busy_total(_clamped_field_deltas(prev_times, times))
            if total_delta <= 0:                   # no CPU time advanced (or clock skew)
                return self._last
            pct = max(0.0, min(100.0, busy_delta / total_delta * 100.0))
            self._last = round(pct, 1)
            return self._last


# Process-global meter: the status bar polls repeatedly, so its previous snapshot
# must survive across requests.
_cpu_meter = _CpuMeter()


def _cpu_ram() -> dict:
    """{"cpu": {...}, "ram": {...}} via psutil, or {} when psutil is absent."""
    try:
        import psutil
    except Exception:
        return {}
    out: dict = {}
    try:
        # CPU% is derived from our own previous snapshot over a real window, not
        # from psutil's since-last-call global. Non-blocking: cpu_times() is an
        # instantaneous read. The first poll has no baseline, so pct is None and
        # the CPU field is omitted rather than fabricated.
        pct = _cpu_meter.percent(psutil.cpu_times(), time.monotonic())
        if pct is not None:
            out["cpu"] = {"percent": pct}
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        out["ram"] = {
            "used": int(vm.total - vm.available),
            "total": int(vm.total),
            "percent": round(float(vm.percent), 1),
        }
    except Exception:
        pass
    return out


def _vram_reading_trusted(info: dict, status) -> bool:
    """Whether a ``vram_capacity``/``vram_info`` reading's ``free`` may be shown as
    the board's CURRENT free VRAM. True only when the reading is BOTH:

      * fresh - ``status`` is GPU_PROBE_OK, i.e. a probe actually completed. A
        GPU_PROBE_TIMEOUT/BUSY reading is a served last-known-good value (possibly
        frozen from an earlier state), not a current measurement.
      * device-global - ``free_scope`` is FREE_SCOPE_DEVICE, i.e. it counts every
        process's VRAM. On Windows + an AMD ROCm/HIP torch build the driver query is
        blind to other processes (FREE_SCOPE_PROCESS); since every GGUF loads in an
        isolated worker the model's own VRAM is invisible there, so a process-scoped
        ``free`` overstates what is available.

    A stale or process-scoped reading shows total-only instead (board capacity is
    always true). Linux/NVIDIA tag every live reading FREE_SCOPE_DEVICE, so this
    never withholds there."""
    from localm.discover import FREE_SCOPE_DEVICE, GPU_PROBE_OK
    return (status == GPU_PROBE_OK
            and info.get("free_scope") == FREE_SCOPE_DEVICE
            and info.get("free") is not None)


# VRAM enumeration is throttled and single-flighted here, not inside
# list_gpus()/vram_capacity(), which carry no TTL cache of their own because
# switch_engine's eviction-wait loop needs a live reading.
_VRAM_REFRESH_INTERVAL_S = 10.0   # much longer than _GPU_UTIL_REFRESH_INTERVAL_S:
                                  # this probe is far more expensive and its answer
                                  # changes on the timescale of plugging in
                                  # hardware, not of a ~2.5s UI poll.

_vram_lock = threading.Lock()
_vram_inflight = False
_vram_last: dict | None = None      # last COMPLETED reading. May be {} (looked,
                                    # found nothing); None means no attempt has
                                    # landed yet. A FAILED attempt never writes here.
_vram_last_at: float | None = None  # monotonic time of the last COMPLETED attempt,
                                    # success or failure, so a persistently failing
                                    # probe is throttled too.
_vram_ready = threading.Event()     # set once, after the first completed attempt
                                    # lands; never cleared. Lets a one-shot caller
                                    # block for a real reading.


def _compute_vram() -> dict:
    """The actual vram_capacity() call and the dict-shaping around it, split
    out of :func:`_vram` so :func:`_vram_probe` can run it on a background
    thread. May raise - :func:`_vram_probe` is the exception boundary, the
    same split :func:`_gpu_util_probe`/``subprocess.run`` use below.

    ``used``/``percent`` are included ONLY when the free reading is trustworthy
    (see :func:`_vram_reading_trusted`); a stale or process-blind reading shows
    ``total`` alone."""
    from localm.discover import (FREE_SCOPE_DEVICE as _FREE_SCOPE_DEVICE,
                                 last_known_gpus, vram_capacity)
    info, status = vram_capacity(return_status=True)
    total = info.get("total")
    if not total:
        return {}
    vram: dict = {"total": int(total)}
    trusted = _vram_reading_trusted(info, status)
    if trusted:
        used = max(0, int(total) - int(info["free"]))
        vram["used"] = used
        vram["percent"] = round(used / int(total) * 100, 1)

    # PER-DEVICE breakdown. The aggregate above does not describe one card on a
    # multi-GPU board: with a gpu_split_indices configured it SUMS the split
    # devices, and with no split it falls back to vram_info(), the single MAIN GPU.
    #
    # No extra probes: this reads the reading vram_capacity() just took (see
    # last_known_gpus). It sits inside _compute_vram, which is single-flighted onto
    # a background thread and throttled to _VRAM_REFRESH_INTERVAL_S.
    #
    # The SAME trust gate applies per device: a process-scoped or stale reading
    # overstates free, so those devices report total only.
    try:
        # last_known_gpus(), not list_gpus(): vram_capacity() above has just driven
        # a probe, and list_gpus has no TTL cache, so calling it here would spawn a
        # second torch subprocess for data already produced.
        #
        # Source order matters. last_known_gpus() comes from list_gpus(), which
        # enumerates only via torch.cuda or nvidia-smi and never calls the Vulkan
        # loader, so it cannot see a device visible only through Vulkan.
        # native_device_inventory() is the ggml runtime's own registry and sees
        # whatever backend is loaded, Vulkan included; it is the fallback because it
        # needs the native lib resident.
        raw = last_known_gpus()
        if not raw:
            try:
                from localm.discover import _apply_device_global_free
                from localm.inference.backends.llamacpp._loader import (
                    native_device_inventory)
                raw = list(native_device_inventory() or [])
                # The registry returns a raw driver `free` with no free_scope tag,
                # and on Windows with AMD that counts only this process's
                # allocations. _apply_device_global_free corrects it and tags the
                # scope, so the per-card gate below has something true to read.
                _apply_device_global_free(raw)
            except Exception:
                raw = []
        devices = []
        for g in raw:
            t = g.get("total")
            if not t:
                continue
            d = {"index": g.get("index"), "total": int(t)}
            if g.get("name"):
                d["name"] = g["name"]
            # Both halves are required:
            #   `trusted`  - the reading is fresh, i.e. a probe completed rather
            #                than a last-known-good value being served. Board-wide.
            #   free_scope - whether THIS card's free counts every process. Per
            #                device; a device list from the native registry never
            #                went through vram_capacity at all.
            if (g.get("free") is not None and trusted
                    and g.get("free_scope") == _FREE_SCOPE_DEVICE):
                u = max(0, int(t) - int(g["free"]))
                d["used"] = u
                d["percent"] = round(u / int(t) * 100, 1)
            devices.append(d)
        # Only worth sending when it says something the aggregate does not.
        if len(devices) > 1:
            vram["devices"] = devices
    except Exception as e:
        # The per-device breakdown never costs the aggregate, which was already
        # computed above. Its failure shows as a missing breakdown.
        from localm.debuglog import logger
        logger.debug("_vram: per-device breakdown unavailable: %s", e)
    return {"vram": vram}


def _vram_probe() -> None:
    """The actual (blocking) vram_capacity() call, which drives list_gpus()'s
    out-of-process torch probe. Runs on its OWN single-flighted daemon thread,
    started by :func:`_vram` - never call this directly.

    On a genuine exception the cache is left UNTOUCHED, never overwritten with
    an empty reading: an error is "could not look", not "confirmed nothing
    there". list_gpus' own TIMEOUT/BUSY/INCONCLUSIVE statuses are NOT this case
    - vram_capacity() already degrades those to a total-only or last-known-good
    dict via :func:`_vram_reading_trusted`, so a clean return here, whatever its
    status, IS a completed attempt worth caching. The refresh-window timer
    advances on a raised exception too, so a persistently erroring probe backs
    off at the same cadence as a healthy one.

    This has no epoch/retirement guard against a straggler thread writing late,
    and needs none: it calls vram_capacity() and blocks on IT, and
    vram_capacity() is guaranteed by list_gpus()'s own deadline to return within
    that bounded deadline, so this thread can never become an abandoned
    straggler."""
    global _vram_inflight, _vram_last, _vram_last_at
    computed = None
    try:
        computed = _compute_vram()
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("_vram: probe raised unexpectedly: %s", e)
    with _vram_lock:
        if computed is not None:
            _vram_last = computed
        _vram_last_at = time.monotonic()
        _vram_inflight = False
    _vram_ready.set()


def _vram(wait_first: bool = False) -> dict:
    """{"vram": {"used"?, "total", "percent"?}} - combined across a configured
    multi-GPU split, else the single main GPU, or {} when unmeasurable or
    before the first reading has landed.

    NEVER blocks the calling (poll) thread on the underlying torch/nvidia-smi
    probe: the actual vram_capacity() call runs single-flighted on its own
    background thread (:func:`_vram_probe`), throttled to at most once per
    :data:`_VRAM_REFRESH_INTERVAL_S`, and this function always returns
    immediately with the last completed reading - omitting the section (never
    fabricating one) until a reading has landed.

    *wait_first*: for a ONE-SHOT caller that will never poll again to pick up
    a reading that lands later (e.g. the MCP system_stats tool, unlike the
    GUI's repeating ~2.5s poll) - when no reading has ever landed AND a probe
    is genuinely in flight, block for up to the underlying probe's own
    cold-init-tolerant deadline instead of omitting VRAM. Default False keeps
    the non-blocking contract for every repeating-poll caller."""
    global _vram_inflight
    now = time.monotonic()
    start_probe = False
    with _vram_lock:
        stale = (_vram_last_at is None
                 or now - _vram_last_at >= _VRAM_REFRESH_INTERVAL_S)
        probe_running = _vram_inflight
        if stale and not _vram_inflight:
            _vram_inflight = True
            start_probe = True
        last = _vram_last
    if start_probe:
        try:
            threading.Thread(target=_vram_probe, name="localm-vram-probe",
                             daemon=True).start()
            probe_running = True
        except Exception as e:
            # Could not spawn the probe thread: clear the in-flight guard so a
            # later call can retry.
            from localm.debuglog import logger
            logger.debug("_vram: could not start probe thread: %s", e)
            with _vram_lock:
                _vram_inflight = False
    if last is None and wait_first and probe_running:
        # Only wait when a probe is genuinely running: if the thread spawn failed,
        # nothing will ever set _vram_ready.
        from localm.discover import _GPU_PROBE_DEADLINE
        if _vram_ready.wait(timeout=_GPU_PROBE_DEADLINE + 1.0):
            with _vram_lock:
                last = _vram_last
    return last if last is not None else {}


_GPU_UTIL_REFRESH_INTERVAL_S = 2.0    # do not re-probe more often than the GUI polls

_gpu_util_lock = threading.Lock()
_gpu_util_inflight = False
_gpu_util_last: float | None = None      # last successfully read percent
_gpu_util_last_at: float | None = None   # monotonic time of the last COMPLETED
                                         # attempt, success or failure, so a
                                         # no-nvidia-smi box is throttled too


def _gpu_util_probe() -> None:
    """The actual (blocking) nvidia-smi call. Runs on its OWN daemon thread,
    started single-flighted by :func:`_gpu_util` - never call this directly,
    it can take up to its subprocess timeout on a slow/wedged driver.

    KNOWN RESIDUAL RISK, not guarded: CPython's ``subprocess.run(..., timeout=N)``
    on Windows, after a ``TimeoutExpired``, calls ``process.kill()`` then calls
    ``communicate()`` a SECOND time with NO timeout to drain the killed process's
    pipes (``Lib/subprocess.py``'s ``run()``, the ``if _mswindows:`` branch). If a
    killed nvidia-smi's pipes never close, that second call - and this whole
    thread - can hang past the stated 4s indefinitely, leaving
    ``_gpu_util_inflight`` stuck True forever. The reading then freezes at its
    last value; nothing user-facing blocks."""
    global _gpu_util_inflight, _gpu_util_last, _gpu_util_last_at
    import subprocess
    pct = None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if p.returncode == 0 and p.stdout.strip():
            pct = round(float(p.stdout.strip().splitlines()[0].strip()), 1)
    except Exception:
        pass
    with _gpu_util_lock:
        if pct is not None:
            _gpu_util_last = pct
        _gpu_util_last_at = time.monotonic()
        _gpu_util_inflight = False


def _gpu_util() -> dict:
    """{"gpu": {"percent": N}} via nvidia-smi (NVIDIA), or {} on any other box
    or before the first reading has landed.

    The non-NVIDIA path is PER VENDOR, not one shared counter. Source order:
    nvidia-smi where it exists, then ADL's own whole-GPU activity sensor on AMD,
    then the vendor-neutral WDDM counter for everything else.

    The AMD branch is required because the vendor-neutral counter is blind to the
    workload localm itself creates: ROCm/HIP compute appears on no WDDM engine, so
    that counter can report another process's video encoder as GPU load while the
    card is saturated. Every figure this returns is WHOLE-GPU - all work on the
    board, whichever process caused it - never localm's own share.

    NEVER blocks the calling (poll) thread on nvidia-smi: the subprocess call
    runs single-flighted on its own daemon thread (:func:`_gpu_util_probe`),
    and this function always returns immediately with the last completed
    reading, omitting the field (never fabricating 0%) until one has landed."""
    global _gpu_util_inflight
    now = time.monotonic()
    start_probe = False
    with _gpu_util_lock:
        stale = (_gpu_util_last_at is None
                 or now - _gpu_util_last_at >= _GPU_UTIL_REFRESH_INTERVAL_S)
        if stale and not _gpu_util_inflight:
            _gpu_util_inflight = True
            start_probe = True
        last = _gpu_util_last
    if start_probe:
        try:
            threading.Thread(target=_gpu_util_probe, name="localm-gpu-util-probe",
                              daemon=True).start()
        except Exception as e:
            # Could not spawn the probe thread: clear _gpu_util_inflight here so a
            # later call can retry. No epoch gate is needed - Thread.start() itself
            # raised, so no thread exists that could write a late reading.
            from localm.debuglog import logger
            logger.debug("_gpu_util: could not start probe thread: %s", e)
            with _gpu_util_lock:
                _gpu_util_inflight = False
    if last is None:
        # Vendor-neutral WDDM fallback: works on AMD and Intel, and on NVIDIA when
        # nvidia-smi is absent. Reads the busiest engine on the main GPU's adapter
        # via a persistent PDH query. Returns {} on its first call, since a rate
        # counter has nothing to rate against yet.
        #
        # ADL is tried first on AMD: the WDDM fold does not track that card's
        # actual load and can report an unrelated process's video encoder instead.
        # It is unreliable there, not dead, so it stays as the fallback.
        try:
            from localm.gpu_usage import amd_whole_gpu_activity
            amd = amd_whole_gpu_activity()
            if amd is not None:
                return {"gpu": {"percent": amd}}
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("_gpu_util: ADL utilisation unavailable: %s", e)
        try:
            from localm.gpu_usage import adapter_utilisation
            per_adapter = adapter_utilisation()
            if per_adapter:
                # Non-AMD Windows boards, where there is no ADL and this is the
                # only device-global source, and the fallback when ADL refuses.
                #
                # No LUID->device mapping is attempted: the payload carries one
                # system-wide gpu.percent, and the LUID carries no device identity.
                # The per-card breakdown keys off the device list instead.
                #
                # This is the busiest ENGINE TYPE, so it under-reports compute a
                # vendor does not publish through the WDDM counters and can be
                # carried by an unrelated process's engine.
                return {"gpu": {"percent": max(per_adapter.values())}}
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("_gpu_util: WDDM utilisation unavailable: %s", e)
        return {}

    return {"gpu": {"percent": last}}


def system_stats(*, wait_first_vram: bool = False) -> dict:
    """Combined live stats; any unmeasurable section is simply absent.

    *wait_first_vram*: pass True only for a one-shot caller that will not
    poll again (see :func:`_vram`'s ``wait_first``) - blocks up to the GPU
    probe's own deadline so a cold first call gets a real VRAM reading
    instead of omitting it. The GUI's repeating status-bar poll must never
    pass this: it self-heals within one refresh window, and blocking it would
    stall the plugin executor."""
    out: dict = {}
    out.update(_cpu_ram())
    out.update(_vram(wait_first=wait_first_vram))
    out.update(_gpu_util())
    return out


# VRAM footprint estimate -------------------------------------------------- #
# The overhead term comes from localm.vram. The KV-cache term prefers the exact
# per-token cost read from the model's GGUF header and falls back to
# GgufBackend._bytes_per_token's size-class rule when no header could be read.
# Approximate: callers should label it an estimate.
from localm.vram import VRAM_OVERHEAD_BYTES as _VRAM_OVERHEAD_BYTES


def estimate_vram(model_bytes: int, n_ctx: int,
                  n_gpu_layers: int = 99, n_layers: int | None = None,
                  kv_bytes_per_token: int = 0, moe_pinned_bytes: int = 0) -> dict:
    """Rough VRAM footprint (bytes) to load a GGUF model at *n_ctx* with
    *n_gpu_layers* offloaded. Returns a breakdown {weights, kv_cache, overhead,
    needed} so the UI can show where the memory goes. A model/ctx of 0 yields 0
    needed (nothing to load).

    *kv_bytes_per_token*, when > 0, is the exact per-token KV cost the caller
    read from the model's GGUF header via gguf_kv_bytes_per_token(path) - this
    function only ever sees a raw byte count, not a path, so it cannot read the
    header itself (the /api/vram-estimate route does that read off the event
    loop and passes the result in). 0 means no header reading was possible
    (missing file, unreadable header, or an unresolved attention shape); this
    then falls back to the size-class heuristic by calling
    GgufBackend._bytes_per_token, the same one VramSizingMixin._kv_bytes_per_token
    falls back to post-load.

    *moe_pinned_bytes*, when > 0, is the caller's own read of
    gguf_moe_pinned_expert_bytes(path, n_cpu_moe) - the exact byte count of
    routed-expert tensors an n_cpu_moe load pins to system RAM instead of
    VRAM (see llamacpp/_sizing.py's VramSizingMixin._effective_model_bytes_for_vram,
    which applies the identical discount to the preflight that decides whether
    a load is even attempted). Same "0 means no signal, do nothing" contract as
    kv_bytes_per_token - this function cannot read a header itself."""
    model_bytes = max(0, int(model_bytes or 0))
    n_ctx = max(0, int(n_ctx or 0))
    # The MoE-pinned share never touches VRAM regardless of n_gpu_layers - it
    # is subtracted from the file's total BEFORE the GPU-offload fraction
    # below is applied, mirroring _effective_model_bytes_for_vram's ordering.
    model_bytes = max(0, model_bytes - max(0, int(moe_pinned_bytes or 0)))
    # Fraction of the weights placed on the GPU. Without the model's true layer
    # count, treat >= 99 (the "all" sentinel) as full; otherwise scale linearly.
    if n_gpu_layers is None or n_gpu_layers < 0:
        frac = 1.0
    elif n_layers and n_layers > 0:
        frac = min(1.0, n_gpu_layers / n_layers)
    else:
        frac = 1.0 if n_gpu_layers >= 99 else min(1.0, n_gpu_layers / 99)
    weights = int(model_bytes * frac)
    # KV cache grows with context; bytes-per-token is the model's real
    # attention shape when the caller could read it, else the size-class guess.
    if kv_bytes_per_token > 0:
        bytes_per_token = kv_bytes_per_token
    else:
        from localm.inference.backends.gguf import GgufBackend
        bytes_per_token = GgufBackend._bytes_per_token(model_bytes)
    kv_cache = int(n_ctx * bytes_per_token)
    overhead = _VRAM_OVERHEAD_BYTES if (weights or kv_cache) else 0
    return {"weights": weights, "kv_cache": kv_cache, "overhead": overhead,
            "needed": weights + kv_cache + overhead}
