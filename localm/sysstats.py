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


def _cpu_ram() -> dict:
    """{"cpu": {...}, "ram": {...}} via psutil, or {} when psutil is absent."""
    try:
        import psutil
    except Exception:
        return {}
    out: dict = {}
    try:
        # interval=None is non-blocking: it measures since the previous call, so
        # the very first poll reads ~0 and subsequent polls are real. Right for a
        # repeating status-bar poll; a blocking interval would stall the request.
        out["cpu"] = {"percent": round(float(psutil.cpu_percent(interval=None)), 1)}
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
