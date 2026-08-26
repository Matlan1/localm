# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Multi-model residency policy, shared by every serving layer.

Both the HTTP server (``http_server.switch_engine``) and the MCP server plugin
(``plugins.mcpserver.server.EngineCache``) decide the same two questions:

    1. ADMIT: can this model load ALONGSIDE the already-resident ones, with no
       eviction at all?
    2. EVICT: if not, which resident peer is the safe one to free first?

The rules live here so a change to the admit margin or the victim-safety
conditions lands in one place and both servers get it.

The policy is conservative in the PERMIT direction. Stacking a second model on
top of a resident one is only allowed on a reading that was actually taken
(``probe_ok``), that the box can produce (``free_vram is not None``), that counts
every process's VRAM rather than just the caller's own (``not
is_process_scoped``), and that clears the model's requirement plus a fixed
headroom, with no per-device split shortfall. Anything else falls back to
single-resident. A wrong PERMIT ends in a native OOM or a driver TDR, not a
graceful error.

Two optional knobs sit on top of the VRAM arithmetic, both OFF by default:

    max_resident_models  cap the number of concurrent resident chat models
    pinned_models        names that are never chosen as an eviction victim
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# Multiplier from on-disk model size to the VRAM a model is expected to occupy
# once loaded.
VRAM_REQUIRED_FACTOR = 1.2
# VRAM the admit check leaves free on the device.
DEFAULT_HEADROOM_BYTES = 1024 ** 3  # 1 GB
# Footprint reported when a registered model's path is neither a file nor a
# directory, so no real size can be read.
UNKNOWN_FOOTPRINT_BYTES = 4 * 1024 ** 3


# Ceiling on the recursive walk in model_footprint_bytes. Stopping early only
# makes the measured size smaller, never larger.
_FOOTPRINT_MAX_FILES = 10_000


def model_footprint_bytes(model_path: Any) -> int:
    """On-disk size of a model: a single GGUF file, or a sharded HF directory.

    The directory walk is bounded (see _FOOTPRINT_MAX_FILES).
    """
    from localm.debuglog import logger     # function-scoped import
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        total = 0
        seen = 0
        for f in p.rglob("*"):
            # Counted before the is_file()/stat() filters, so the bound is on
            # entries walked, not on files successfully measured.
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
                # An unreadable entry is skipped and logged; the walk continues.
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
    """
    True when the model may load with ZERO eviction, alongside resident peers.

    Every condition is a PERMIT-direction guard:

    ``probe_ok``      the reading was actually taken. A frozen last-known-good
                      that reads HIGH would otherwise admit a load on top of
                      resident peers and OOM. The same stale value in the LOW
                      direction only costs a spurious refusal, so the REFUSE
                      direction is laxer.
    ``free_vram``     is not None: the box can report free VRAM at all. A
                      CPU-only / GGUF-only box reports nothing, and "nothing"
                      never reads as "plenty".
    ``is_process_scoped`` the reading counts only the CALLING process's own
                      VRAM allocations, not the whole device (see
                      ``discover.FREE_SCOPE_PROCESS``). Every resident model
                      lives in its OWN isolated worker subprocess
                      (backends/gguf.py), so a process-scoped reading is blind
                      to exactly the VRAM this check accounts for and can only
                      over-report free space. Treated the same as "cannot
                      measure".
    ``+ headroom``    the requirement alone is not enough of a margin.
    ``not shortfall`` aggregate free can clear the bar while ONE device of a
                      configured split is short: the GGUF backend divides a
                      model by a static ratio with no live per-device check of
                      its own. See ``discover.gpu_split_shortfall``.

    Callers that cannot measure, whose probe was inconclusive, or whose reading
    is process-scoped, get False and should fall back to single-resident rather
    than stacking until the driver OOMs.
    """
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
    """
    Least-recently-used resident model that is SAFE to evict, or None.

    ``lru`` is least-recently-used first. A candidate is skipped when it is:

    - the model being loaded (never evict what we are here to make room for);
    - pinned by the user;
    - serving requests (``active_requests``); or
    - already mid-unload (``unloading``). Another unload path keeps its entry
      listed until the native free completes, so without this guard eviction can
      call ``unload()`` a SECOND time on one native context - a double free that
      the per-model semaphore does not serialise, because eviction takes no lock
      on its victim. ``active_requests == 0`` alone does not catch it.
    """
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
    """
    ``max_resident_models``, or None for "no cap" (the default).

    None means the VRAM arithmetic alone decides how many models stay resident.
    A cap of 1 forces strict single-resident.

    An unusable value (not an int, or < 1) is IGNORED with a warning, never
    coerced.
    """
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
    """
    ``pinned_models``: display names never chosen as an eviction victim.

    Pinning only PROTECTS; it never loads anything on its own. A non-list value,
    or a non-string entry, is dropped with a warning.
    """
    cfg = _config(config)
    raw = cfg.get("pinned_models")
    if not raw:
        return frozenset()
    if not isinstance(raw, (list, tuple)):
        _warn("pinned_models=%r is not a list; ignoring it", raw)
        return frozenset()
    names = {n for n in raw if isinstance(n, str) and n}
    # Counts the entries actually rejected, not raw length minus the set size:
    # a duplicate pin collapses in the set without having been rejected.
    dropped = sum(1 for n in raw if not (isinstance(n, str) and n))
    if dropped > 0:
        _warn("pinned_models: ignored %d entr%s that were not names",
              dropped, "y" if dropped == 1 else "ies")
    return frozenset(names)


def exceeds_resident_cap(
    resident: Iterable[str], requested: str, cap: Optional[int]
) -> bool:
    """
    True when admitting ``requested`` would leave more models resident than the
    cap allows, so a peer must be evicted even though VRAM may well fit.

    Reloading a model that is ALREADY resident never exceeds the cap.
    """
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
