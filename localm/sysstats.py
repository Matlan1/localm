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
