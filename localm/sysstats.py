"""Best-effort live system stats for the GUI hardware monitor.

Pure and NEVER raises: a probe that fails just omits its field, so the status
bar degrades gracefully (e.g. VRAM still shows on a box without psutil). Cheap
enough to call on a short GUI poll:
  * CPU % and RAM via psutil (the optional ``[monitor]`` extra)
  * VRAM via localm.discover.vram_info (torch -> nvidia-smi -> Windows registry)
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


def _vram() -> dict:
    """{"vram": {"used"?, "total", "percent"?}} for the largest GPU, or {}."""
    try:
        from localm.discover import vram_info
        info = vram_info()
    except Exception:
        return {}
    total = info.get("total")
    if not total:
        return {}
    vram: dict = {"total": int(total)}
    free = info.get("free")
    if free is not None:
        used = max(0, int(total) - int(free))
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
# Deliberately mirrors the GGUF backend's own heuristic (gguf.GgufBackend:
# _VRAM_OVERHEAD_BYTES and the bytes-per-token rule in _auto_ctx_max) so the
# number the GUI shows matches how the loader actually reasons about fit. It is
# approximate by nature - callers should label it as an estimate.
_VRAM_OVERHEAD_BYTES = int(1.5e9)


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
