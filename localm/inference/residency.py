# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Multi-model residency policy, shared by every serving layer.

Both the HTTP server (``http_server.switch_engine``) and the MCP server plugin
(``plugins.mcpserver.server.EngineCache``) decide the same two questions:

    1. ADMIT: can this model load ALONGSIDE the already-resident ones, with no
       eviction at all?
    2. EVICT: if not, which resident peer is the safe one to free first?

The rules used to live only inside ``switch_engine``, so the MCP server had no
way to reach them and shipped single-resident instead. They live here now so the
two cannot drift: a change to the admit margin or the victim-safety conditions
lands in one place and both servers get it.

The policy is deliberately conservative in the PERMIT direction. Stacking a
second model on top of a resident one is only allowed on a reading that was
actually taken (``probe_ok``), that the box can actually produce (``free_vram is
not None``), and that clears the model's requirement plus a fixed headroom, with
no per-device split shortfall. Anything else falls back to single-resident. See
``fits_alongside_residents`` for why each of those is load-bearing: the failure
mode of a wrong PERMIT is a native OOM or a driver TDR, not a graceful error.

Two optional knobs sit on top of the VRAM arithmetic, both OFF by default so a
tight-VRAM box behaves exactly as it did before this module existed:

    max_resident_models  cap the number of concurrent resident chat models
    pinned_models        names that are never chosen as an eviction victim
"""

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


def model_footprint_bytes(model_path: Any) -> int:
    """On-disk size of a model: a single GGUF file, or a sharded HF directory."""
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
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
) -> bool:
    """
    True when the model may load with ZERO eviction, alongside resident peers.

    Every condition is a PERMIT-direction guard, and each one has already cost
    somebody a bad load:

    ``probe_ok``      the reading was actually taken. A frozen last-known-good
                      that happens to read HIGH would otherwise admit a load on
                      top of resident peers and OOM. (The same stale value in
                      the LOW direction only costs a spurious refusal, which is
                      why the REFUSE direction can be laxer than this.)
    ``free_vram``     is not None: the box can report free VRAM at all. A
                      CPU-only / GGUF-only box reports nothing, and "nothing"
                      must never read as "plenty".
    ``+ headroom``    the requirement alone is not enough of a margin.
    ``not shortfall`` aggregate free can clear the bar while ONE device of a
                      configured split is short, because the GGUF backend
                      divides a model by a static ratio with no live per-device
                      check of its own. See ``discover.gpu_split_shortfall``.

    Callers that cannot measure, or whose probe was inconclusive, get False and
    should fall back to single-resident rather than stacking until the driver
    OOMs.
    """
    if not probe_ok or free_vram is None:
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

    None means the VRAM arithmetic alone decides how many models stay resident,
    which is the behaviour that shipped before the knob existed. A cap of 1
    restores strict single-resident on a box whose owner does not want localm
    using the headroom the arithmetic says is there.

    An unusable value (not an int, or < 1) is IGNORED with a warning rather than
    silently coerced: a typo that quietly became "cap 1" would look like a
    performance regression with no way to trace it.
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
    or a non-string entry, is dropped with a warning (an ignored pin that
    silently failed to protect is exactly the kind of hidden problem rule 5 is
    about).
    """
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
