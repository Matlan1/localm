# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-effort live system stats for the GUI hardware monitor.

Pure and NEVER raises: a probe that fails just omits its field, so the status
bar degrades gracefully (e.g. VRAM still shows on a box without psutil). Cheap
enough to call on a short GUI poll:
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


def _vram() -> dict:
    """{"vram": {"used"?, "total", "percent"?}} - combined across a configured
    multi-GPU split, else the single main GPU, or {} when unmeasurable.

    ``used``/``percent`` are included ONLY when the free reading is trustworthy
    (see :func:`_vram_reading_trusted`); a stale or process-blind reading shows
    ``total`` alone rather than a wrong used/free, so the status bar never presents
    a number localm cannot stand behind as current fact."""
    try:
        from localm.discover import vram_capacity
        info, status = vram_capacity(return_status=True)
    except Exception:
        return {}
    total = info.get("total")
    if not total:
        return {}
    vram: dict = {"total": int(total)}
    if _vram_reading_trusted(info, status):
        used = max(0, int(total) - int(info["free"]))
        vram["used"] = used
        vram["percent"] = round(used / int(total) * 100, 1)
    return {"vram": vram}


def _gpu_util() -> dict:
    """{"gpu": {"percent": N}} via nvidia-smi (NVIDIA), or {} on any other box.

    VRAM (above) is the metric that matters for fitting a model; utilisation is a
    nice-to-have, and a clean CSV probe only exists for NVIDIA. AMD/Intel boxes
    simply omit it rather than ship a fragile, often-wrong parse."""
    import subprocess
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if p.returncode == 0 and p.stdout.strip():
            pct = float(p.stdout.strip().splitlines()[0].strip())
            return {"gpu": {"percent": round(pct, 1)}}
    except Exception:
        pass
    return {}


def system_stats() -> dict:
    """Combined live stats; any unmeasurable section is simply absent."""
    out: dict = {}
    out.update(_cpu_ram())
    out.update(_vram())
    out.update(_gpu_util())
    return out


# VRAM footprint estimate -------------------------------------------------- #
# Single-sourced from localm.vram (the same constant GgufBackend._check_vram and
# discover.fit_label use) plus the bytes-per-token rule mirrored from
# _auto_ctx_max, so the number the GUI shows matches how the loader actually
# reasons about fit. Approximate by nature - callers should label it an estimate.
from localm.vram import VRAM_OVERHEAD_BYTES as _VRAM_OVERHEAD_BYTES


def estimate_vram(model_bytes: int, n_ctx: int,
                  n_gpu_layers: int = 99, n_layers: int | None = None) -> dict:
    """Rough VRAM footprint (bytes) to load a GGUF model at *n_ctx* with
    *n_gpu_layers* offloaded. Returns a breakdown {weights, kv_cache, overhead,
    needed} so the UI can show where the memory goes. A model/ctx of 0 yields 0
    needed (nothing to load)."""
    model_bytes = max(0, int(model_bytes or 0))
    n_ctx = max(0, int(n_ctx or 0))
    # Fraction of the weights placed on the GPU. Without the model's true layer
    # count, treat >= 99 (the "all" sentinel) as full; otherwise scale linearly.
    if n_gpu_layers is None or n_gpu_layers < 0:
        frac = 1.0
    elif n_layers and n_layers > 0:
        frac = min(1.0, n_gpu_layers / n_layers)
    else:
        frac = 1.0 if n_gpu_layers >= 99 else min(1.0, n_gpu_layers / 99)
    weights = int(model_bytes * frac)
    # KV cache grows with context; bytes-per-token scales with model size,
    # clamped to a plausible band (matches _auto_ctx_max).
    bytes_per_token = min(max(model_bytes // 100_000, 16_000), 512_000)
    kv_cache = int(n_ctx * bytes_per_token)
    overhead = _VRAM_OVERHEAD_BYTES if (weights or kv_cache) else 0
    return {"weights": weights, "kv_cache": kv_cache, "overhead": overhead,
            "needed": weights + kv_cache + overhead}
