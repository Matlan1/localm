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
MMAP_CPU_FROM_DISK_NOTE = ("the weights kept in system RAM may not fit in "
                           "available RAM, CPU layers read them from disk-backed "
                           "memory, every token slower")


def unmapped_exceeds_ram_note(host_bytes: int, ram_available: int) -> str:
    """The warning for a ``use_mmap`` off load whose *host_bytes* of system-RAM
    weights do not fit the *ram_available* bytes of available RAM."""
    gib = 1024 ** 3
    return (f"use_mmap is off and this load reads {host_bytes / gib:.1f} GB of "
            f"weights into system RAM with {ram_available / gib:.1f} GB "
            f"available: the load may fail or swap. Set use_mmap to auto to "
            f"map them from disk instead.")


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
                  forced_by_ram: bool = False, cpu_layers: bool = False) -> Optional[str]:
    """One plain line (no markup) for what a load did with memory-mapping, or
    None when there is nothing to say.

    *effective* is whether the load memory-mapped the model (None when the
    loader did not report it). *forced_by_ram* marks an ``auto`` load mapped
    while the weights it keeps in system RAM may not fit available RAM, and
    *cpu_layers* one of those whose CPU layers read them on every token
    (``MMAP_CPU_FROM_DISK_NOTE`` instead of ``MMAP_FROM_DISK_NOTE``). Nothing
    is said for any other ``auto`` load, or when *effective* is unknown. An
    explicit ``on`` or ``off`` is confirmed, and a load that did the opposite
    says so."""
    if effective is None:
        return None
    if setting == USE_MMAP_AUTO:
        if not (effective and forced_by_ram):
            return None
        return f"mmap on: {MMAP_CPU_FROM_DISK_NOTE if cpu_layers else MMAP_FROM_DISK_NOTE}"
    if effective:
        return "mmap on" if setting == USE_MMAP_ON else "mmap on, although use_mmap is off"
    return "mmap off" if setting == USE_MMAP_OFF else "mmap off, although use_mmap is on"
