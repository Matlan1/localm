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
    occasionally regress even while total CPU time moves forward - psutil's own
    ``_cpu_times_deltas`` trims each field to zero rather than let one regressing
    field corrupt the aggregate, and this mirrors that exactly (verified against
    psutil 7.2.2 source) so the percent derived from it matches what a blocking
    ``cpu_percent(interval=..)`` would report even across a regressing field."""
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
    off psutil's process-global state, so its FIRST reading (and any reading whose
    since-last-call window happened to span a burst, e.g. server startup or a model
    load) is fabricated: it can report 100% on a busy-starting box or 0% on an idle
    one, neither a real measurement. The GUI polls /api/stats off a MULTI-thread
    executor (get_plugin_executor), which chops that shared window into unpredictable
    slices on top of it.

    So we do not rely on that implicit state. We store our own previous
    ``(times, monotonic)`` snapshot and derive the percent from the CLAMPED
    per-field delta over a real, known window (see :func:`_clamped_field_deltas`
    - computing an aggregate busy/total per snapshot and subtracting THOSE would
    let one regressing field, e.g. Windows ``interrupt``, corrupt the whole
    percentage; clamping per-field first, like psutil itself does, does not).
    The first reading (no baseline yet) reports ``None`` so the caller omits the
    CPU field for that one poll rather than showing a made-up number (AGENTS.md
    rule 5: never present a fabricated value as a live measurement)."""

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
# must survive across requests (that persistence is the whole point of the fix).
_cpu_meter = _CpuMeter()


def _cpu_ram() -> dict:
    """{"cpu": {...}, "ram": {...}} via psutil, or {} when psutil is absent."""
    try:
        import psutil
    except Exception:
        return {}
    out: dict = {}
    try:
        # Derive CPU% from OUR own previous snapshot over a real window, not from
        # psutil's fragile since-last-call global (see _CpuMeter). Non-blocking:
        # cpu_times() is an instantaneous read, so the request never stalls. The
        # first poll has no baseline yet -> pct is None -> omit the CPU field this
        # once; the frontend renders only the sections it receives, so it shows the
        # rest and the CPU figure appears (correct) on the next 2.5s poll, rather
        # than a fabricated 0 or 100.
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
        isolated worker (#606) the model's own VRAM is invisible, so a process-scoped
        ``free`` overstates what is available - a fresh reading can still be wrong.

    Presenting a stale or process-scoped ``free`` as the board's live figure is
    exactly the "report success when the measurement did not hold" that AGENTS.md
    rule 5 forbids, so those readings show total-only instead (board capacity is
    always true). Linux/NVIDIA tag every live reading FREE_SCOPE_DEVICE by
    documentation, so this never withholds there."""
    from localm.discover import FREE_SCOPE_DEVICE, GPU_PROBE_OK
    return (status == GPU_PROBE_OK
            and info.get("free_scope") == FREE_SCOPE_DEVICE
            and info.get("free") is not None)


# VRAM enumeration is throttled/single-flighted exactly like GPU utilisation
# below, and for a stronger reason: discover.list_gpus() deliberately carries
# NO TTL cache of its own (see the module note above it in discover.py) -
# switch_engine's eviction-wait loop polls it for a LIVE "free" reading, and a
# stale value there would defeat that guard and over-evict. So the cache
# cannot live inside list_gpus()/vram_capacity() itself; it has to sit here,
# scoped to this one best-effort status-bar reading, leaving every other
# caller of list_gpus()/vram_capacity() untouched. Without it, every GUI poll
# paid the full out-of-process torch probe cost synchronously (measured
# 2.1-3.5s typical, up to the 15s cold-init-tolerant deadline on a wedged
# driver) on the shared plugin executor (localm.executor) - roughly one probe
# per poll in the field, and each one occupies a worker thread that RAG/web/
# voice/coder tool calls also depend on.
_VRAM_REFRESH_INTERVAL_S = 10.0   # Deliberately much longer than
                                  # _GPU_UTIL_REFRESH_INTERVAL_S: that probe is
                                  # a cheap nvidia-smi call (<200ms), this one
                                  # is 10-20x more expensive AND its answer
                                  # (GPU count/capacity) changes on the
                                  # timescale of plugging in hardware, not of a
                                  # ~2.5s UI poll - a hot-plugged GPU still
                                  # shows up within one refresh window, never
                                  # permanently stale.

_vram_lock = threading.Lock()
_vram_inflight = False
_vram_last: dict | None = None      # last COMPLETED reading. May legitimately
                                     # be {} ("looked, found nothing"); None
                                     # means no attempt has ever landed yet
                                     # ("could not look" - see _vram_probe).
                                     # Those two must stay distinguishable
                                     # (AGENTS.md rule 5), so a FAILED attempt
                                     # never writes into this.
_vram_last_at: float | None = None  # monotonic time of the last COMPLETED
                                     # attempt, success or failure - throttles
                                     # a persistently failing probe too, not
                                     # just successful readings (mirrors
                                     # _gpu_util_last_at).


def _compute_vram() -> dict:
    """The actual vram_capacity() call and the dict-shaping around it, split
    out of :func:`_vram` so :func:`_vram_probe` can run it on a background
    thread. May raise - :func:`_vram_probe` is the exception boundary, the
    same split :func:`_gpu_util_probe`/``subprocess.run`` use below.

    ``used``/``percent`` are included ONLY when the free reading is trustworthy
    (see :func:`_vram_reading_trusted`); a stale or process-blind reading shows
    ``total`` alone rather than a wrong used/free, so the status bar never
    presents a number localm cannot stand behind as current fact."""
    from localm.discover import vram_capacity
    info, status = vram_capacity(return_status=True)
    total = info.get("total")
    if not total:
        return {}
    vram: dict = {"total": int(total)}
    if _vram_reading_trusted(info, status):
        used = max(0, int(total) - int(info["free"]))
        vram["used"] = used
        vram["percent"] = round(used / int(total) * 100, 1)
    return {"vram": vram}


def _vram_probe() -> None:
    """The actual (blocking) vram_capacity() call, which drives list_gpus()'s
    out-of-process torch probe. Runs on its OWN single-flighted daemon thread,
    started by :func:`_vram` - never call this directly.

    On a genuine exception the cache is left UNTOUCHED rather than overwritten
    with an empty reading: an error is "could not look", never "confirmed
    nothing there" - collapsing the two is the exact bug class
    ``.claude/rules/diff-review-discipline.md`` item 3 documents for this same
    subsystem (list_gpus' own TIMEOUT/BUSY/INCONCLUSIVE statuses are not this
    case - vram_capacity() already degrades those to an honest total-only or
    last-known-good dict via :func:`_vram_reading_trusted`, so a clean return
    here, whatever its status, IS a completed attempt worth caching). The
    refresh-window timer still advances on a raised exception too, so a
    persistently erroring probe backs off at the same cadence as a healthy one
    instead of retrying on every single poll (mirrors _gpu_util_probe).

    Unlike list_gpus()'s own probe machinery, this has no epoch/retirement
    guard against a straggler thread writing late: list_gpus() needs one
    because the NATIVE torch/HIP call it starts can hang uninterruptibly
    forever, so it abandons that call and returns early, leaving a thread
    that may write whenever (or never). This function never does that - it
    calls vram_capacity() and blocks on IT, and vram_capacity() (via
    list_gpus()'s own deadline) is guaranteed to return within that same
    bounded deadline regardless of what its own native probe does. So this
    thread is always bounded and never becomes an abandoned straggler
    itself; there is nothing here for an epoch to fence out."""
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


def _vram() -> dict:
    """{"vram": {"used"?, "total", "percent"?}} - combined across a configured
    multi-GPU split, else the single main GPU, or {} when unmeasurable or
    before the first reading has landed.

    NEVER blocks the calling (poll) thread on the underlying torch/nvidia-smi
    probe: the actual vram_capacity() call runs single-flighted on its own
    background thread (:func:`_vram_probe`), throttled to at most once per
    :data:`_VRAM_REFRESH_INTERVAL_S`, and this function always returns
    immediately with the last completed reading - omitting the section (never
    fabricating one) until a reading has landed, the same "omit rather than
    fabricate" contract ``_cpu_ram``/``_gpu_util`` already use in this module."""
    global _vram_inflight
    now = time.monotonic()
    start_probe = False
    with _vram_lock:
        stale = (_vram_last_at is None
                 or now - _vram_last_at >= _VRAM_REFRESH_INTERVAL_S)
        if stale and not _vram_inflight:
            _vram_inflight = True
            start_probe = True
        last = _vram_last
    if start_probe:
        try:
            threading.Thread(target=_vram_probe, name="localm-vram-probe",
                             daemon=True).start()
        except Exception as e:
            # Could not spawn the probe thread: never leave the in-flight guard
            # stuck True with nothing left to clear it - a LATER call must be
            # able to retry (mirrors _gpu_util() and discover.py's list_gpus()
            # handling of the identical failure).
            from localm.debuglog import logger
            logger.debug("_vram: could not start probe thread: %s", e)
            with _vram_lock:
                _vram_inflight = False
    return last if last is not None else {}


_GPU_UTIL_REFRESH_INTERVAL_S = 2.0    # do not re-probe more often than the GUI polls

_gpu_util_lock = threading.Lock()
_gpu_util_inflight = False
_gpu_util_last: float | None = None      # last successfully read percent
_gpu_util_last_at: float | None = None   # monotonic time of the last COMPLETED
                                          # attempt (success or failure) - throttles
                                          # retries on a no-nvidia-smi box too, not
                                          # just successful readings


def _gpu_util_probe() -> None:
    """The actual (blocking) nvidia-smi call. Runs on its OWN daemon thread,
    started single-flighted by :func:`_gpu_util` - never call this directly,
    it can take up to its subprocess timeout on a slow/wedged driver.

    KNOWN RESIDUAL RISK, deliberately not guarded further: CPython's
    ``subprocess.run(..., timeout=N)`` on Windows, after a ``TimeoutExpired``,
    calls ``process.kill()`` then calls ``communicate()`` a SECOND time with NO
    timeout to drain the killed process's pipes (verified in
    ``Lib/subprocess.py``'s ``run()``, the ``if _mswindows:`` branch). If a
    killed nvidia-smi's pipes never close, that second call - and this whole
    thread - can hang past the stated 4s indefinitely, leaving
    ``_gpu_util_inflight`` stuck True forever (the reading simply freezes at
    its last value; nothing user-facing blocks, unlike before this fix). This
    is not new: the pre-fix code shared the identical subprocess timeout and
    the identical exposure. Accepted rather than fixed with a retirement/epoch
    mechanism (see the no-epoch note in :func:`_gpu_util`) because the field is
    explicitly best-effort/nice-to-have and no caller ever blocks on it either
    way - a genuinely wedged nvidia-smi that never releases its pipes is rare
    enough that the fix's cost (a second timed layer, e.g. killing this thread
    itself) is not worth it unless it is ever observed for real."""
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

    VRAM (above) is the metric that matters for fitting a model; utilisation is a
    nice-to-have, and a clean CSV probe only exists for NVIDIA. AMD/Intel boxes
    simply omit it rather than ship a fragile, often-wrong parse.

    NEVER blocks the calling (poll) thread on nvidia-smi: the subprocess call
    runs single-flighted on its own daemon thread (:func:`_gpu_util_probe`),
    and this function always returns immediately with the last completed
    reading, omitting the field (never fabricating 0%) until one has landed -
    the same "omit rather than fabricate" contract the CPU-percent meter and
    the VRAM used/percent gate already use elsewhere in this module. The old
    version ran a bare per-poll ``subprocess.run(..., timeout=4)`` (the poll
    interval is 2.5s), which spawned a fresh nvidia-smi on every single poll
    with no cap on how many could be in flight at once, and could park the
    polling thread on a slow driver call. This is a standalone efficiency/
    robustness fix for this one reading; it is unrelated to the GPU-enumeration
    freeze diagnosed for issue #833 (a cold ``import torch`` holding the
    Windows loader lock across thread creation process-wide, fixed separately
    in ``discover.py``/``_torch_gpu_probe.py`` - a caller-side timeout or cache
    cannot fix that class of stall, since nothing a caller does bounds a lock
    the OS holds on another thread's behalf)."""
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
            # Could not spawn the probe thread (e.g. OS thread exhaustion): the
            # thread that would have cleared _gpu_util_inflight never ran, so
            # clear it here - never leave it stuck True with nothing left to
            # reset it (a LATER call must be able to retry). Mirrors discover.py's
            # list_gpus() handling of the identical failure, minus its epoch
            # gate: that gate exists to fence out a late write from a probe
            # thread ABANDONED after overrunning its deadline (a thread that IS
            # still running and could write after a caller gives up on it).
            # There is no such thread here - Thread.start() itself raised, so
            # no thread was ever created, and nothing can write into
            # _gpu_util_last/_gpu_util_last_at after this reset. An epoch would
            # be dead machinery implying a retirement mechanism this module
            # does not have. Surfaced at debug (AGENTS.md rule 5) rather than
            # propagating, matching this module's documented "NEVER raises"
            # contract.
            from localm.debuglog import logger
            logger.debug("_gpu_util: could not start probe thread: %s", e)
            with _gpu_util_lock:
                _gpu_util_inflight = False
    if last is None:
        return {}
    return {"gpu": {"percent": last}}


def system_stats() -> dict:
    """Combined live stats; any unmeasurable section is simply absent."""
    out: dict = {}
    out.update(_cpu_ram())
    out.update(_vram())
    out.update(_gpu_util())
    return out


# VRAM footprint estimate -------------------------------------------------- #
# Single-sourced from localm.vram (the same constant GgufBackend._check_vram and
# discover.fit_label use) for the overhead term. The KV-cache term prefers the
# exact per-token cost read from the model's own GGUF header
# (gguf_kv_bytes_per_token - the same attention-shape source
# VramSizingMixin._kv_bytes_per_token uses before a load) and falls back to
# GgufBackend._bytes_per_token's size-class rule only when the caller could not
# read a header. Approximate by nature - callers should label it an estimate.
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
    then falls back to the size-class heuristic, kept as the LAST resort by
    calling GgufBackend._bytes_per_token rather than reimplementing it, so
    there is exactly one size-class formula in the codebase - the same one
    VramSizingMixin._kv_bytes_per_token falls back to post-load.

    *moe_pinned_bytes*, when > 0, is the caller's own read of
    gguf_moe_pinned_expert_bytes(path, n_cpu_moe) - the exact byte count of
    routed-expert tensors an n_cpu_moe load pins to system RAM instead of
    VRAM (see llamacpp/_sizing.py's VramSizingMixin._effective_model_bytes_for_vram,
    which applies the identical discount to the preflight that decides whether
    a load is even attempted). Without this, this estimate double-counted
    exactly what that preflight fix corrected: a Mixture-of-Experts model
    with its experts pinned to system RAM would still show the WHOLE file as
    VRAM-needed here, overstating the readout even after the load itself
    would have succeeded. Same "0 means no signal, do nothing" contract as
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
