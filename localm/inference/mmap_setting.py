# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``use_mmap`` setting: its three values and the one line that reports
what a load did with it.

``auto`` lets the loader decide, ``on`` memory-maps the model file so only the
pages a run touches are resident, ``off`` reads the whole model into memory.
"""

from typing import Optional

USE_MMAP_AUTO = "auto"
USE_MMAP_ON = "on"
USE_MMAP_OFF = "off"
USE_MMAP_MODES = (USE_MMAP_AUTO, USE_MMAP_ON, USE_MMAP_OFF)

MMAP_FROM_DISK_NOTE = ("model may not fit in available RAM, running from "
                       "disk-backed memory, first tokens slower")


def coerce_use_mmap(val) -> Optional[str]:
    """*val* as one of USE_MMAP_MODES (case and surrounding spaces ignored), or
    None when it is not one of them. A bool is not a mode: None, never
    "on"/"off"."""
    if isinstance(val, str):
        low = val.strip().lower()
        if low in USE_MMAP_MODES:
            return low
    return None


def resolve_use_mmap(cfg: dict) -> str:
    """The ``use_mmap`` mode *cfg* holds, always one of USE_MMAP_MODES. An
    absent value is ``auto``; a present value that is not a mode is logged at
    WARNING and read as ``auto``."""
    raw = cfg.get("use_mmap", USE_MMAP_AUTO)
    if raw is None or raw == "":
        return USE_MMAP_AUTO
    mode = coerce_use_mmap(raw)
    if mode is not None:
        return mode
    from localm.debuglog import logger as _dbg
    _dbg.warning("use_mmap is set but not one of %s (%r); using %s",
                 ", ".join(USE_MMAP_MODES), raw, USE_MMAP_AUTO)
    return USE_MMAP_AUTO


def describe_mmap(setting: str, effective: Optional[bool],
                  forced_by_ram: bool = False) -> Optional[str]:
    """One plain line (no markup) for what a load did with memory-mapping, or
    None when there is nothing to say.

    *effective* is whether the load memory-mapped the model (None when the
    loader did not report it). *forced_by_ram* marks an ``auto`` load that
    turned mmap on because the model may not fit in available RAM. Nothing is said for
    any other ``auto`` load, or when *effective* is unknown. An explicit ``on``
    or ``off`` is confirmed, and a load that did the opposite says so."""
    if effective is None:
        return None
    if setting == USE_MMAP_AUTO:
        return f"mmap on: {MMAP_FROM_DISK_NOTE}" if effective and forced_by_ram else None
    if effective:
        return "mmap on" if setting == USE_MMAP_ON else "mmap on, although use_mmap is off"
    return "mmap off" if setting == USE_MMAP_OFF else "mmap off, although use_mmap is on"
