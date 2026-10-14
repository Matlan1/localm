# SPDX-License-Identifier: AGPL-3.0-or-later
"""The VRAM budget a stable-diffusion.cpp load gets (``sd_ctx_params.max_vram``).

On Windows the ROCm and Vulkan runtimes report free VRAM for their own process
only. For those, ``vram_budget`` turns localm's device-global free-VRAM reading
(``discover.list_gpus``) into a ``max_vram`` setting, so the runtime keeps to
the VRAM that is actually free and streams weights from RAM when the model does
not fit next to what other programs hold. Everywhere else it returns None and
the runtime sizes the load from its own reading.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional

PROCESS_SCOPED_BACKENDS = ("rocm", "vulkan")

MARGIN_BYTES = 512 * 1024 ** 2

MIN_BUDGET_BYTES = 256 * 1024 ** 2

_GIB = 1024 ** 3


@dataclass(frozen=True)
class Budget:
    """``setting`` is the ``max_vram`` value; ``free`` and ``total`` are the
    device-global readings it came from, in bytes, summed over the GPUs."""
    setting: str
    free: int
    total: int


def _platform() -> str:
    return sys.platform


def _gib(n: int) -> str:
    return f"{max(n, MIN_BUDGET_BYTES) / _GIB:.2f}"


def vram_budget(backend: str, *, gpus: Optional[list] = None, status: Optional[str] = None,
                platform: Optional[str] = None) -> Optional[Budget]:
    """The budget for a *backend* runtime load, or None when the runtime's own
    reading is used: not Windows, a backend outside ``PROCESS_SCOPED_BACKENDS``,
    a GPU probe that did not complete, a reading that is not device-global, or
    several GPUs on Vulkan (whose device order is not the probe's).

    Each GPU's budget is its device-global free VRAM less ``MARGIN_BYTES``, at
    least ``MIN_BUDGET_BYTES``. One GPU gives a plain GiB value; several ROCm
    GPUs give ``ROCm<index>=<GiB>`` assignments. *gpus* and *status* replace the
    live ``discover.list_gpus`` reading and *platform* replaces
    ``sys.platform``. Never raises: a failed probe returns None."""
    if (platform or _platform()) != "win32" or backend not in PROCESS_SCOPED_BACKENDS:
        return None
    from localm import discover
    if gpus is None:
        try:
            gpus, status = discover.list_gpus(return_status=True, wait_for_inflight=True)
        except Exception as e:  # noqa: BLE001
            from localm.debuglog import logger
            logger.warning("native media VRAM budget: the GPU probe failed (%s: %s); the "
                           "runtime sizes the load from its own reading",
                           type(e).__name__, e)
            return None
    if status != discover.GPU_PROBE_OK or not gpus:
        return None
    usable = [g for g in gpus if isinstance(g, dict) and isinstance(g.get("free"), int)
              and isinstance(g.get("total"), int) and isinstance(g.get("index"), int)
              and g.get("free_scope") == discover.FREE_SCOPE_DEVICE]
    if not usable or len(usable) != len(gpus):
        return None
    if len(usable) > 1 and backend != "rocm":
        return None
    free = sum(g["free"] for g in usable)
    total = sum(g["total"] for g in usable)
    if len(usable) == 1:
        setting = _gib(usable[0]["free"] - MARGIN_BYTES)
    else:
        setting = ",".join(f"ROCm{g['index']}={_gib(g['free'] - MARGIN_BYTES)}"
                           for g in sorted(usable, key=lambda g: g["index"]))
    return Budget(setting=setting, free=free, total=total)


def with_budget(ctx: dict, backend: str, say: Callable[[str], None]) -> dict:
    """*ctx* with ``max_vram`` set from :func:`vram_budget`, saying the free
    VRAM through *say*; *ctx* unchanged (the same object) when there is no
    budget. Call it with no worker of this process holding VRAM."""
    b = vram_budget(backend)
    if b is None:
        return ctx
    from localm.debuglog import logger
    logger.debug("native media VRAM budget: max_vram=%s (free %d of %d bytes)",
                 b.setting, b.free, b.total)
    say(f"Free VRAM: {b.free / _GIB:.1f} of {b.total / _GIB:.1f} GB.")
    return dict(ctx, max_vram=b.setting)
