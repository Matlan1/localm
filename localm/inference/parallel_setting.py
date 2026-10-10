# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``parallel_slots`` setting: how many chat requests one loaded GGUF model
answers at the same time.

``auto`` lets the load choose (``PARALLEL_AUTO_SLOTS`` for most models, fewer
when the extra state a slot needs would not fit in VRAM, one for a model with a
draft source); a number from 1 to ``PARALLEL_MAX`` asks for that many. Every
slot shares the one context window.
"""

from __future__ import annotations

from typing import Optional, Union

PARALLEL_AUTO = "auto"
PARALLEL_AUTO_SLOTS = 4
PARALLEL_MAX = 16

ParallelSetting = Union[str, int]


def coerce_parallel_slots(val) -> Optional[ParallelSetting]:
    """*val* as ``"auto"`` or an int from 1 to PARALLEL_MAX, or None when it is
    neither. Digit strings count as numbers; a bool is not a number."""
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if 1 <= val <= PARALLEL_MAX else None
    if isinstance(val, str):
        low = val.strip().lower()
        if low == PARALLEL_AUTO:
            return PARALLEL_AUTO
        if low.isdigit():
            return coerce_parallel_slots(int(low))
    return None


def resolve_parallel_slots(cfg: dict) -> ParallelSetting:
    """The ``parallel_slots`` setting *cfg* holds. An absent or empty value is
    ``auto``; a value that is neither ``auto`` nor 1-PARALLEL_MAX is logged at
    WARNING and read as ``auto``."""
    raw = cfg.get("parallel_slots", PARALLEL_AUTO)
    if raw is None or raw == "":
        return PARALLEL_AUTO
    value = coerce_parallel_slots(raw)
    if value is not None:
        return value
    from localm.debuglog import logger as _dbg
    _dbg.warning("parallel_slots is set but is neither %s nor a number from 1 "
                 "to %d (%r); using %s", PARALLEL_AUTO, PARALLEL_MAX, raw,
                 PARALLEL_AUTO)
    return PARALLEL_AUTO
