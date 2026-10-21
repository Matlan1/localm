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
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

PROCESS_SCOPED_BACKENDS = ("rocm", "vulkan")

MARGIN_BYTES = 512 * 1024 ** 2

MIN_BUDGET_BYTES = 256 * 1024 ** 2

RELEASE_TIMEOUT_S = 5.0

_GIB = 1024 ** 3

UNBUDGETED_RETRY = ("The free VRAM is too little for this model to run within it. Loading it "
                    "again without that limit; Windows may move part of it to system "
                    "memory, which is slower.")


@dataclass(frozen=True)
class Budget:
    """``setting`` is the ``max_vram`` value, or None when the free VRAM could
    not be measured (``reason`` says why); ``free`` and ``total`` are the
    device-global readings in bytes, summed over the GPUs (0 without a setting)."""
    setting: Optional[str]
    free: int = 0
    total: int = 0
    reason: Optional[str] = None


def _platform() -> str:
    return sys.platform


def _gib(n: int) -> str:
    return f"{max(n, MIN_BUDGET_BYTES) / _GIB:.2f}"


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def applies(backend: str, platform: Optional[str] = None) -> bool:
    """Whether a *backend* load on this platform gets a budget: Windows and a
    backend in ``PROCESS_SCOPED_BACKENDS``."""
    return (platform or _platform()) == "win32" and backend in PROCESS_SCOPED_BACKENDS


def _norm_name(name) -> str:
    return " ".join(str(name or "").split()).casefold()


def _runtime_gpus(devices) -> Optional[list]:
    """``[(name, description), ...]`` for the non-CPU devices of a runtime's
    device list (``Runtime.devices``), or None when *devices* is None."""
    if devices is None:
        return None
    out = []
    for d in devices:
        if isinstance(d, (list, tuple)) and d:
            name = str(d[0]).strip()
            desc = str(d[1]).strip() if len(d) > 1 else ""
            if name and name.upper() != "CPU":
                out.append((name, desc))
    return out


def _per_device_setting(usable: list, runtime_gpus: Optional[list]):
    """``(setting, matched GPUs, reason)`` assigning a budget to each runtime
    device by name. A runtime device's budget comes from the GPUs of *usable*
    whose name equals its description, the smallest free VRAM among them when
    several do; a runtime device no GPU matches gets no assignment. *setting* is
    None, with the reason, when the runtime's devices are not known or none of
    them matches a GPU."""
    if runtime_gpus is None:
        return None, [], "the runtime's device list is not known"
    parts = []
    matched: dict = {}
    for name, desc in runtime_gpus:
        same = [g for g in usable if _norm_name(g.get("name")) == _norm_name(desc)]
        if not same:
            continue
        parts.append(f"{name}={_gib(min(g['free'] for g in same) - MARGIN_BYTES)}")
        for g in same:
            matched[g["index"]] = g
    if not parts:
        return None, [], "none of the runtime's devices matches a GPU localm measured"
    return ",".join(parts), list(matched.values()), None


def vram_budget(backend: str, *, gpus: Optional[list] = None, status: Optional[str] = None,
                platform: Optional[str] = None,
                devices: Optional[list] = None) -> Optional[Budget]:
    """The budget for a *backend* runtime load. None when the runtime's own
    reading is used (``applies`` is False). A ``Budget`` without a setting, with
    the reason, when the GPU probe fails or does not complete, a reading is not
    device-global or is incomplete, there are several GPUs on Vulkan, or
    several GPUs on ROCm cannot be matched to the runtime's devices.

    Each GPU's budget is its device-global free VRAM less ``MARGIN_BYTES``, at
    least ``MIN_BUDGET_BYTES``. One GPU gives a plain GiB value. Several ROCm
    GPUs give each runtime device ``<device name>=<GiB>`` (``ROCm0=...``),
    matched to a GPU by name (:func:`_per_device_setting`); *devices* is the
    runtime's
    ``[name, description]`` device list (``Runtime.devices``). *gpus* and
    *status* replace the live ``discover.list_gpus`` reading and *platform*
    replaces ``sys.platform``. Never raises."""
    if not applies(backend, platform):
        return None
    from localm import discover
    if gpus is None:
        try:
            gpus, status = discover.list_gpus(return_status=True, wait_for_inflight=True)
        except Exception as e:  # noqa: BLE001
            return Budget(None, reason=f"the GPU probe failed: {type(e).__name__}: {e}")
    if status != discover.GPU_PROBE_OK:
        return Budget(None, reason=f"the GPU probe did not complete ({status})")
    if not gpus:
        return Budget(None, reason="the GPU probe found no GPU")
    usable = [g for g in gpus if isinstance(g, dict) and _is_int(g.get("free"))
              and _is_int(g.get("total")) and _is_int(g.get("index"))
              and g.get("free_scope") == discover.FREE_SCOPE_DEVICE]
    if len(usable) != len(gpus):
        return Budget(None, reason="a GPU's reading covers only this process or is incomplete")
    if len(usable) > 1 and backend != "rocm":
        return Budget(None, reason="several GPUs on the Vulkan runtime")
    if len(usable) == 1:
        g = usable[0]
        return Budget(setting=_gib(g["free"] - MARGIN_BYTES), free=g["free"],
                      total=g["total"])
    setting, matched, reason = _per_device_setting(usable, _runtime_gpus(devices))
    if setting is None:
        return Budget(None, reason=reason)
    return Budget(setting=setting, free=sum(g["free"] for g in matched),
                  total=sum(g["total"] for g in matched))


def with_budget(ctx: dict, backend: str, say: Callable[[str], None],
                b: Optional[Budget] = None, *, devices: Optional[list] = None) -> dict:
    """*ctx* with ``max_vram`` set from *b* (default: a fresh :func:`vram_budget`
    for the runtime's *devices*), saying the free VRAM through *say*. *ctx*
    unchanged (the same object) without a setting; when the reading failed,
    *say* and the debug log get the reason. Call it with no worker of this
    process holding VRAM."""
    if b is None:
        b = vram_budget(backend, devices=devices)
    if b is None:
        return ctx
    from localm.debuglog import logger
    if b.setting is None:
        logger.warning("native media VRAM budget: free VRAM not measured (%s); the %s "
                       "runtime sizes the load from its own reading", b.reason, backend)
        say(f"Free VRAM could not be measured ({b.reason}); the runtime sizes the model "
            "from its own reading, which does not count VRAM other programs use.")
        return ctx
    logger.debug("native media VRAM budget: max_vram=%s (free %d of %d bytes)",
                 b.setting, b.free, b.total)
    say(f"Free VRAM: {b.free / _GIB:.1f} of {b.total / _GIB:.1f} GB.")
    return dict(ctx, max_vram=b.setting)


def reload_ctx(runner, ctx: dict, backend: str, say: Callable[[str], None],
               cancel_check: Optional[Callable[[], bool]] = None, *,
               devices: Optional[list] = None) -> dict:
    """Stop *runner*'s worker and return *ctx* for the next load, with its
    budget (:func:`with_budget`) for the runtime's *devices*. When the stopped worker held a model and the
    budget applies, the free VRAM is read before the stop, and the budget is
    the first reading after it that has risen (``vram.wait_for_vram_release``,
    polling for ``RELEASE_TIMEOUT_S``); when none rises, the last successful
    reading after the stop. Raises ``SdCancelled`` when *cancel_check* returns
    True before a reading or after the stop."""
    from .runner import SdCancelled

    def check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise SdCancelled("Generation cancelled.")

    before = None
    if applies(backend) and runner.is_alive() and runner.loaded_key is not None:
        check_cancel()
        b = vram_budget(backend, devices=devices)
        before = b.free if b is not None and b.setting is not None else None
    runner.shutdown()
    check_cancel()
    if before is None:
        return with_budget(ctx, backend, say, devices=devices)
    last: list = [None]
    good: list = [None]

    def read_free() -> Optional[int]:
        check_cancel()
        b = vram_budget(backend, devices=devices)
        last[0] = b
        if b is None or b.setting is None:
            return None
        good[0] = b
        return b.free

    from localm.vram import wait_for_vram_release
    released, _final = wait_for_vram_release(read_free, before_bytes=before,
                                             timeout_s=RELEASE_TIMEOUT_S)
    if released is False:
        from localm.debuglog import logger
        logger.warning("native media VRAM budget: free VRAM did not rise within %.0f s "
                       "of stopping the previous model", RELEASE_TIMEOUT_S)
    return with_budget(ctx, backend, say, good[0] or last[0], devices=devices)


def call_within_budget(runner, call: Callable[[], T], reload: Callable[[dict], object],
                       say: Callable[[str], None], *,
                       on_retry: Optional[Callable[[], None]] = None,
                       cancel_check: Optional[Callable[[], bool]] = None) -> T:
    """``call()``, the generation on *runner*'s loaded model. When it raises
    ``SdBudgetRefused`` and the model was loaded with a ``max_vram`` budget,
    calls *on_retry*, says ``UNBUDGETED_RETRY``, stops the worker, passes the
    loaded context without ``max_vram`` to *reload* (which loads it) and
    returns ``call()`` once more. Any other failure, and a refusal of a load
    without a budget, propagates. Raises ``SdCancelled`` when *cancel_check*
    returns True after the refusal, before the worker is stopped or after."""
    from .runner import SdBudgetRefused, SdCancelled

    def check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise SdCancelled("Generation cancelled.")

    try:
        return call()
    except SdBudgetRefused as e:
        loaded = dict(runner.loaded_ctx or {})
        if not loaded.get("max_vram"):
            raise
        from localm.debuglog import logger
        logger.info("native media VRAM budget: the runtime refused to run within "
                    "max_vram=%s (%s); reloading without a budget", loaded["max_vram"], e)
    check_cancel()
    if on_retry is not None:
        on_retry()
    say(UNBUDGETED_RETRY)
    loaded.pop("max_vram", None)
    runner.shutdown()
    check_cancel()
    reload(loaded)
    return call()
