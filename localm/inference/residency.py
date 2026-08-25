# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-model residency policy, shared by every serving layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# A loaded model costs more VRAM than its file: KV cache, compute buffers and
# allocator slack. The same 1.2 factor switch_engine has always used.
VRAM_REQUIRED_FACTOR = 1.2
# Never fill the card to the brim - leave room for the KV cache to grow with
# context and for whatever else shares the device (compositor, another app).
DEFAULT_HEADROOM_BYTES = 1024 ** 3  # 1 GB
# Used only when a registered model's path is neither a file nor a directory, so
# no real size can be read. Deliberately pessimistic: it makes the admit check
# HARDER to pass, so an unreadable model errs toward single-resident.
UNKNOWN_FOOTPRINT_BYTES = 4 * 1024 ** 3


# Ceiling on the recursive walk below. A real model directory holds tens of files
# (shards, tokenizer, config); thousands means we are not looking at a model. The
# walk used to be unbounded, so a caller who could name a directory could spend
# the server's time enumerating it (CodeQL 73-76) - a deep or huge tree, or one
# whose contents change under us, cost an arbitrary amount of work on the calling
# thread. Stopping early only makes the measured size SMALLER, and a smaller
# footprint is the conservative direction for every caller: it can only make a
# model look easier to fit, never harder, and the admit decision is re-checked
# against real VRAM at load time.
_FOOTPRINT_MAX_FILES = 10_000


def model_footprint_bytes(model_path: Any) -> int:
    """On-disk size of a model: a single GGUF file, or a sharded HF directory."""
    from localm.debuglog import logger     # function-scoped, as elsewhere here
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        total = 0
        seen = 0
        for f in p.rglob("*"):
            # Counted BEFORE the is_file()/stat() filters, so the bound is on
            # ENTRIES WALKED, not on files successfully measured. Counting only
            # measured files left the walk unbounded for the two trees that cost
            # the most and measure the least: one made entirely of directories,
            # and one whose entries all fail stat(). Both would `continue` forever
            # without ever reaching the ceiling.
            seen += 1
            if seen > _FOOTPRINT_MAX_FILES:
                logger.debug(
                    "footprint: stopped after walking %d entries under %s; this "
                    "does not look like a model directory, reporting the partial "
                    "size", seen - 1, p)
                break
            try:
                if not f.is_file():
                    continue
                total += f.stat().st_size
            except OSError as e:
                # One unreadable entry (a broken link, a race with a delete, a
                # permission hole) must not abort the whole measurement, but it
                # is not silently ignored either (AGENTS.md rule 5).
                logger.debug("footprint: skipping unreadable %s: %s", f, e)
                continue
        return total
    return UNKNOWN_FOOTPRINT_BYTES


def required_vram_bytes(file_size: int) -> int:
    """VRAM a model of ``file_size`` bytes is expected to occupy once loaded."""
    return int(file_size * VRAM_REQUIRED_FACTOR)


def fits_alongside_residents(
    *,
    free_vram: Optional[int],
    vram_required: int,
    probe_ok: bool,
    headroom: int = DEFAULT_HEADROOM_BYTES,
    shortfall: Sequence = (),
    is_process_scoped: bool = False,
) -> bool:
    """True when the model may load with ZERO eviction, alongside resident peers."""
    if not probe_ok or free_vram is None or is_process_scoped:
        return False
    if shortfall:
        return False
    return free_vram >= vram_required + headroom


def pick_eviction_victim(
    lru: Iterable[str],
    engines: Mapping[str, Any],
    *,
    requested: Optional[str] = None,
    pinned: Iterable[str] = (),
) -> Optional[str]:
    """Least-recently-used resident model that is SAFE to evict, or None."""
    pinned_set = set(pinned)
    for candidate in lru:
        if requested is not None and candidate == requested:
            continue
        if candidate in pinned_set:
            continue
        engine = engines.get(candidate)
        if engine is None:
            continue
        if getattr(engine, "active_requests", 0) != 0:
            continue
        if getattr(engine, "unloading", False) is True:
            continue
        return candidate
    return None


def resident_cap(config: Optional[Mapping] = None) -> Optional[int]:
    """``max_resident_models``, or None for 'no cap' (the default)."""
    cfg = _config(config)
    raw = cfg.get("max_resident_models")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        _warn("max_resident_models=%r is not an integer; ignoring the cap", raw)
        return None
    if raw < 1:
        _warn("max_resident_models=%r is below 1; ignoring the cap", raw)
        return None
    return raw


def pinned_model_names(config: Optional[Mapping] = None) -> frozenset:
    """``pinned_models``: display names never chosen as an eviction victim."""
    cfg = _config(config)
    raw = cfg.get("pinned_models")
    if not raw:
        return frozenset()
    if not isinstance(raw, (list, tuple)):
        _warn("pinned_models=%r is not a list; ignoring it", raw)
        return frozenset()
    names = {n for n in raw if isinstance(n, str) and n}
    # Count what was actually REJECTED, not raw-minus-set: a duplicate pin
    # ("a,a") collapses in the set but was not rejected, and reporting it as
    # one would be a warning about something that never happened.
    dropped = sum(1 for n in raw if not (isinstance(n, str) and n))
    if dropped > 0:
        _warn("pinned_models: ignored %d entr%s that were not names",
              dropped, "y" if dropped == 1 else "ies")
    return frozenset(names)


def exceeds_resident_cap(
    resident: Iterable[str], requested: str, cap: Optional[int]
) -> bool:
    """True when admitting ``requested`` would leave more models resident than the cap allows, so a peer must be evicted even though VRAM may well fit."""
    if cap is None:
        return False
    names = set(resident)
    if requested in names:
        return False
    return len(names) + 1 > cap


def _config(config: Optional[Mapping]) -> Mapping:
    if config is not None:
        return config
    from localm.config import load_config
    return load_config()


def _warn(msg: str, *args) -> None:
    from localm.debuglog import logger
    logger.warning("residency: " + msg, *args)
