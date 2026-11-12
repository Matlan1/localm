# SPDX-License-Identifier: AGPL-3.0-or-later
"""How each routing candidate would run on this machine now.

``assess`` turns a VRAM reading, a RAM reading and the resident models into a
``capability_routing.Fit`` per candidate: its fit class (``FIT_*``) and which
models the user chose would be unloaded to reach it. A GGUF candidate is sized
by its backend's ``predict_placement``, the same computation its load runs;
any other backend by its offload needs or, failing that, its file size.

Nothing here loads, evicts or reads the GPU. The caller passes the readings,
so the same function serves the HTTP server, the MCP server and a re-check
under the load lease."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

from localm.inference import residency
from localm.inference.capability_routing import (FIT_CPU_ONLY, FIT_DISK,
                                                 FIT_EXPERTS_RAM, FIT_GPU,
                                                 FIT_PARTIAL, FIT_RESIDENT,
                                                 FIT_UNSIZED, Fit)

# Same margin the GGUF mmap decision keeps free in system RAM.
_RAM_HEADROOM = 2 * 1024 ** 3

UNLOAD_ALWAYS = "always"
UNLOAD_NEVER = "never"


@dataclass(frozen=True)
class VramReading:
    """One GPU reading. *free*/*total* are bytes (None when unknown);
    *usable* is True when *free* is a trusted device-wide figure; *devices*
    is how many GPUs a load spreads over; *gpu_present* is False on a machine
    without a GPU."""

    free: Optional[int]
    total: Optional[int]
    usable: bool
    devices: int = 1
    gpu_present: bool = True


@dataclass(frozen=True)
class Resident:
    """A loaded model. *hold* is the VRAM it holds (0 or less: unknown);
    *user_chosen* is False only for a model routing itself loaded;
    *evictable* is False for a pinned model or one mid-unload;
    *reads_from_disk* is True when its own placement reads weights from disk."""

    name: str
    hold: int
    user_chosen: bool
    evictable: bool = True
    reads_from_disk: bool = False


def classify(engine, free: Optional[int], reading: VramReading,
             ram_available: Optional[int], ram_total: Optional[int]) -> tuple[int, int]:
    """``(FIT_* class, bytes read from disk while answering)`` for loading
    *engine* with *free* bytes of VRAM free on its GPU."""
    if reading.gpu_present and (free is None or not reading.usable):
        return FIT_UNSIZED, 0
    gpu_free = int(free or 0) if reading.gpu_present else 0
    predict = getattr(engine, "predict_placement", None)
    placement = None
    if callable(predict):
        try:
            if reading.devices >= 2:
                placement = predict(None, None, ram_available, ram_total)
            else:
                placement = predict(gpu_free, reading.total, ram_available, ram_total)
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("routing fit: predicting %s failed (%s: %s)",
                       getattr(engine, "display_name", "?"), type(exc).__name__, exc)
            placement = None
    if isinstance(placement, dict):
        if placement.get("reads_from_disk"):
            return FIT_DISK, int(placement.get("cpu_read_bytes") or 0)
        if int(placement.get("layers") or 0) <= 0:
            return (FIT_CPU_ONLY if reading.gpu_present else FIT_PARTIAL), 0
        if placement.get("full"):
            return (FIT_EXPERTS_RAM if int(placement.get("n_cpu_moe") or 0) > 0
                    else FIT_GPU), 0
        return FIT_PARTIAL, 0
    return _classify_by_size(engine, gpu_free, reading, ram_available)


def _classify_by_size(engine, free: int, reading: VramReading,
                      ram_available: Optional[int]) -> tuple[int, int]:
    """The fit class from the backend's offload needs, else the file size,
    for a backend that cannot predict its placement."""
    size = _file_bytes(engine)
    if size <= 0:
        return FIT_UNSIZED, 0
    full = minimum = None
    needs = getattr(engine, "gpu_offload_needs", None)
    if callable(needs):
        try:
            answer = needs()
        except Exception:
            answer = None
        if isinstance(answer, tuple) and answer and answer[0]:
            full, minimum = int(answer[0]), int(answer[1] or answer[0])
    if full is None:
        full = minimum = residency.required_vram_bytes(size) + residency.DEFAULT_HEADROOM_BYTES
    if reading.gpu_present and free >= full:
        return FIT_GPU, 0
    if _experts_read_from_disk(engine):
        return FIT_DISK, max(0, size - free)
    in_ram = max(0, size - free) if reading.gpu_present else size
    if ram_available is None:
        return FIT_UNSIZED, 0
    if in_ram + _RAM_HEADROOM > ram_available:
        return FIT_DISK, in_ram
    if reading.gpu_present and free >= minimum:
        return FIT_EXPERTS_RAM, 0
    if reading.gpu_present and free <= 0:
        return FIT_CPU_ONLY, 0
    return FIT_PARTIAL, 0


def _file_bytes(engine) -> int:
    path = getattr(engine, "model_path", None)
    if not path:
        return 0
    try:
        return int(residency.model_footprint_bytes(path))
    except Exception:
        return 0


def _experts_read_from_disk(engine) -> bool:
    """Whether *engine* is a checkpoint whose routed experts are read from its
    files when it does not fit."""
    path = getattr(engine, "model_path", None)
    if not path:
        return False
    try:
        from localm.inference.backends._hf_expert_files import worth_using
        return bool(worth_using(str(path), 0))
    except Exception:
        return False


def _tiers(residents: Iterable[Resident], unload_policy: str
           ) -> list[tuple[int, tuple[str, ...], Optional[int]]]:
    """``(tier, user models unloaded, VRAM freed)`` options for one load: free
    VRAM as it is; then also unloading the idle models routing loaded; then
    also the models the user chose. Only ``unload_for_room`` "always" unloads
    anything for a routed load. Freed is None when a hold is unknown."""
    out: list[tuple[int, tuple[str, ...], Optional[int]]] = [(0, (), 0)]
    if unload_policy != UNLOAD_ALWAYS:
        return out
    evictable = [r for r in residents if r.evictable]
    routed = [r for r in evictable if not r.user_chosen]
    user = [r for r in evictable if r.user_chosen]

    def freed(rs: Sequence[Resident]) -> Optional[int]:
        return None if any(r.hold <= 0 for r in rs) else sum(r.hold for r in rs)

    if routed:
        out.append((1, (), freed(routed)))
    if user:
        both = routed + user
        out.append((2, tuple(r.name for r in user), freed(both)))
    return out


def assess(names: Sequence[str], *, residents: Mapping[str, Resident],
           build: Callable[[str], object], reading: VramReading,
           ram_available: Optional[int], ram_total: Optional[int],
           unload_policy: str, peers: Iterable[str] = ()) -> dict[str, Fit]:
    """A ``Fit`` for each of *names*.

    A resident model or one a peer answers is ``FIT_RESIDENT`` (``FIT_DISK``
    when its own placement reads from disk). Any other is built with *build*
    (an unloaded engine) and classified at each option ``_tiers`` allows; the
    best option usable without asking wins, else the best class, else the
    cheapest. An option that unloads models adds their VRAM to the reading's
    free figure; it is left out when a freed amount is unknown or the load
    spreads over two or more GPUs (a freed amount has no device). Blocking:
    reads model files."""
    peers = frozenset(peers)
    out: dict[str, Fit] = {}
    tiers = _tiers(residents.values(), unload_policy)
    for name in names:
        if name in peers:
            out[name] = Fit(FIT_RESIDENT)
            continue
        here = residents.get(name)
        if here is not None:
            out[name] = Fit(FIT_DISK if here.reads_from_disk else FIT_RESIDENT)
            continue
        try:
            engine = build(name)
        except Exception as exc:
            from localm.debuglog import logger as _dbg
            _dbg.debug("routing fit: cannot build %s to size it (%s: %s)",
                       name, type(exc).__name__, exc)
            out[name] = Fit(FIT_UNSIZED, error=f"{type(exc).__name__}: {exc}")
            continue
        size = _file_bytes(engine)
        options = []
        for tier, evicts, freed in tiers:
            if tier and (freed is None or reading.devices >= 2):
                continue
            free = None if reading.free is None else reading.free + int(freed or 0)
            cls, disk = classify(engine, free, reading, ram_available, ram_total)
            options.append(Fit(cls, evicts_user=evicts, load_bytes=size,
                               disk_read_bytes=disk))
        out[name] = min(options, key=lambda f: (f.needs_ask, f.cls, len(f.evicts_user)))
    return out
