# SPDX-License-Identifier: AGPL-3.0-or-later
"""Admission decisions for ``http_server.switch_engine``.

Everything here is pure: it takes a VRAM reading, the load's budget and the
resident registries as arguments and returns a value. Nothing awaits, probes
hardware, reads module state, logs or mutates a registry. ``switch_engine``
performs every effect these decisions call for (probing, freeing the embedder,
asking a peer instance, cancelling a busy model, detaching and unloading a
victim, loading, committing).

A load attempt runs the eviction loop: take a ``VramProbe``, ask
``decide_admission`` what it allows, and either load, evict the idle victim it
named, or walk the exhaustion ladder (embedder, probe verdict, peer
cooperation, busy victim, final verdict) until the model may load or the
request is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from localm.inference import residency

# decide_admission() actions.
ADMIT = "admit"                      # fits alongside residents, within the cap
ADMIT_OVER_CAP = "admit_over_cap"    # fits, only the resident cap wanted room, nothing idle to evict
EVICT_IDLE = "evict_idle"            # evict AdmissionDecision.victim, then re-probe
EXHAUSTED = "exhausted"              # does not fit and no idle chat model is evictable

# exhausted_probe_verdict() answers.
LOAD_BEST_EFFORT = "load_best_effort"   # the box cannot report free VRAM at all
RETRY_PROBE = "retry_probe"             # the probe was inconclusive and a retry is left
GIVE_UP_PROBE = "give_up_probe"         # the probe stayed inconclusive through every retry
KEEP_FREEING = "keep_freeing"           # a real reading: continue down the ladder

# final_exhaustion_verdict() answers.
REFUSE_SPLIT_SHORTFALL = "refuse_split_shortfall"   # a static split share is short on a device
CONFIRM_DEGRADED_LOAD = "confirm_degraded_load"     # explicit switch: ask before a partial offload
DEFER_TO_BACKEND = "defer_to_backend"               # load and let the backend size itself

# EvictionStep kinds.
REPROBE = "reprobe"            # something was freed or retried: take a new reading
PROCEED_TO_LOAD = "proceed"    # stop evicting and load
RETURN_RESULT = "result"       # switch_engine returns EvictionStep.result
EVICT = "evict"                # evict EvictionStep.victim, then re-probe

_MB = 1024 ** 2


@dataclass(frozen=True)
class LoadBudget:
    """What one load of ``name`` has to fit, fixed for the whole load attempt.

    ``vram_required`` is the whole-model estimate
    (``residency.required_vram_bytes``), ``headroom`` the margin demanded on
    top of it, ``resident_cap`` the ``max_resident_models`` setting (None for
    no cap), ``pinned`` the ``pinned_models`` names and ``check_split_fit``
    whether the configured split's per-device shares are checked (a GGUF load).
    """

    name: str
    vram_required: int
    headroom: int
    resident_cap: Optional[int]
    pinned: frozenset
    check_split_fit: bool

    @property
    def needed_bytes(self) -> int:
        """``vram_required + headroom``: the bar every VRAM comparison uses."""
        return self.vram_required + self.headroom


@dataclass(frozen=True)
class VramProbe:
    """One VRAM reading taken by the eviction loop.

    ``free`` is the free-VRAM figure (None when the box cannot report one),
    ``probe_ok`` whether the probe completed, ``process_scoped`` whether
    ``free`` counts only this process's allocations
    (``discover.FREE_SCOPE_PROCESS``), ``shortfall`` the configured split's
    short devices (``discover.gpu_split_shortfall``, empty when not checked)
    and ``shares_adaptive`` whether those shares were the live
    free-VRAM-proportional ones rather than static ratios. ``implicit_split``
    is True when ``free`` is summed over the GPUs llama.cpp's default split
    uses (``discover.implicit_split_free``) rather than one device's reading.
    """

    free: Optional[int]
    probe_ok: bool
    process_scoped: bool
    shortfall: Any
    shares_adaptive: bool
    implicit_split: bool = False

    @property
    def measurable(self) -> bool:
        """The reading carries a free-VRAM figure."""
        return self.free is not None

    @property
    def cannot_measure(self) -> bool:
        """The probe completed and the box reported no free figure. Permanent:
        the box cannot measure, so a load proceeds best-effort."""
        return self.probe_ok and not self.measurable

    @property
    def inconclusive(self) -> bool:
        """The probe did not complete. Transient: free VRAM is unknown, so a
        load is retried and then refused, never admitted on it."""
        return not self.probe_ok


@dataclass(frozen=True)
class AdmissionDecision:
    """What ``decide_admission`` allows for one reading.

    ``action`` is one of ADMIT, ADMIT_OVER_CAP, EVICT_IDLE or EXHAUSTED;
    ``victim`` is set only for EVICT_IDLE. ``vram_ok`` and ``over_cap`` are the
    two inputs the action was derived from.
    """

    action: str
    vram_ok: bool
    over_cap: bool
    victim: Optional[str] = None


@dataclass
class EvictionAttempt:
    """Mutable bookkeeping for one load attempt's eviction loop.

    ``switch_engine`` updates it as it performs effects; the decisions here only
    read it. Each bounded step (embedder eviction, each peer instance, the busy
    victim) runs at most once per attempt, and inconclusive probes are retried
    ``_INCONCLUSIVE_LOAD_RETRIES`` times, so the loop always terminates.
    ``started`` is the ``time.monotonic()`` reading the attempt began at.
    """

    started: float
    asked_peers: set = field(default_factory=set)
    embedder_attempted: bool = False
    busy_attempted: bool = False
    inconclusive_retries: int = 0


@dataclass(frozen=True)
class EvictionStep:
    """What the eviction loop does next after the exhaustion ladder.

    ``kind`` is REPROBE, PROCEED_TO_LOAD, RETURN_RESULT (``result`` is the dict
    ``switch_engine`` returns) or EVICT (``victim`` is evicted; ``force_busy``
    evicts it even while a request is still pinned on it).
    """

    kind: str
    victim: Optional[str] = None
    force_busy: bool = False
    result: Optional[dict] = None


def decide_admission(probe: VramProbe, budget: LoadBudget, lru: Iterable[str],
                     engines: Mapping[str, Any]) -> AdmissionDecision:
    """Decide whether ``budget.name`` may load on this reading, and if not,
    which idle resident model to evict first.

    ADMIT when the model fits alongside the residents
    (``residency.fits_alongside_residents``) and stays within the resident cap.
    Otherwise the least-recently-used safe victim
    (``residency.pick_eviction_victim``) is EVICT_IDLE; with no victim it is
    ADMIT_OVER_CAP when VRAM fits and only the cap wanted room, and EXHAUSTED
    when VRAM does not fit. The victim is only looked up when admission fails.
    """
    over_cap = residency.exceeds_resident_cap(lru, budget.name, budget.resident_cap)
    vram_ok = residency.fits_alongside_residents(
        free_vram=probe.free, vram_required=budget.vram_required,
        probe_ok=probe.probe_ok, headroom=budget.headroom,
        shortfall=probe.shortfall, is_process_scoped=probe.process_scoped)
    if vram_ok and not over_cap:
        return AdmissionDecision(ADMIT, vram_ok, over_cap)
    victim = residency.pick_eviction_victim(
        lru, engines, requested=budget.name, pinned=budget.pinned)
    if victim is None and vram_ok:
        return AdmissionDecision(ADMIT_OVER_CAP, vram_ok, over_cap)
    if victim is None:
        return AdmissionDecision(EXHAUSTED, vram_ok, over_cap)
    return AdmissionDecision(EVICT_IDLE, vram_ok, over_cap, victim)


def exhausted_probe_verdict(probe: VramProbe, *, retries_used: int,
                            max_retries: int) -> str:
    """With no idle chat model left to evict: LOAD_BEST_EFFORT when the box
    cannot measure free VRAM, RETRY_PROBE when the probe was inconclusive and
    fewer than *max_retries* retries were used, GIVE_UP_PROBE when it stayed
    inconclusive through them, and KEEP_FREEING for a real reading."""
    if probe.cannot_measure:
        return LOAD_BEST_EFFORT
    if probe.inconclusive:
        if retries_used < max_retries:
            return RETRY_PROBE
        return GIVE_UP_PROBE
    return KEEP_FREEING


def pin_blocks_peer_cooperation(budget: LoadBudget, lru: Iterable[str],
                                engines: Mapping[str, Any]) -> bool:
    """True when ``pinned_models`` is the only reason no local model could be
    evicted: some model would be evictable if pins were ignored. A peer
    instance is then not asked to give up its models for a local preference."""
    return bool(budget.pinned) and residency.pick_eviction_victim(
        lru, engines, requested=budget.name) is not None


def busy_victim_candidate(budget: LoadBudget, lru: Iterable[str],
                          engines: Mapping[str, Any], *, preempt: bool,
                          already_attempted: bool) -> Optional[str]:
    """The serving model to cancel and evict as the last local resort, or None.

    Only an explicit switch (*preempt*) may interrupt a generation, and only
    once per load attempt. The candidate is
    ``residency.pick_busy_eviction_victim``: never the requested model, a
    pinned one or one already mid-unload."""
    if not preempt or already_attempted:
        return None
    return residency.pick_busy_eviction_victim(
        lru, engines, requested=budget.name, pinned=budget.pinned)


def final_exhaustion_verdict(probe: VramProbe, *, preempt: bool, force: bool) -> str:
    """With local, peer and busy eviction all exhausted on a measurable reading:

    - REFUSE_SPLIT_SHORTFALL when a split device is short and the shares are
      static (pinned ratios, or the equal fallback).
    - CONFIRM_DEGRADED_LOAD for an explicit switch (*preempt*) without *force*.
    - DEFER_TO_BACKEND otherwise, including a shortfall on adaptive shares:
      the load goes ahead and the backend sizes its own GPU offload.
    """
    if probe.shortfall and not probe.shares_adaptive:
        return REFUSE_SPLIT_SHORTFALL
    if preempt and not force:
        return CONFIRM_DEGRADED_LOAD
    return DEFER_TO_BACKEND


def split_shortfall_refusal(name: str, shortfall: Iterable[Mapping]) -> str:
    """The 503 detail naming each short split device."""
    detail = "; ".join(
        f"GPU {d['index']} needs ~{d['needed'] // _MB} MB, "
        f"{d['free'] // _MB} MB free" for d in shortfall)
    return (f"Not enough VRAM on the configured split "
            f"device(s) to load '{name}' ({detail}).")


def degraded_load_confirm(name: str, vram_required: int, free: int) -> dict:
    """The confirm_required result asking before a partial-offload load."""
    return {
        "status": "confirm_required", "model": name,
        "detail": (
            f"'{name}' does not fit the estimated free VRAM "
            f"(need ~{vram_required // _MB} MB, "
            f"{free // _MB} MB free) even after "
            "eviction; loading it anyway will let the backend "
            "fall back to partial CPU offload, which is slower"),
    }


def busy_victim_confirm(name: str, victim: str, in_use: str) -> dict:
    """The confirm_required result asking before evicting a model still in use.
    *in_use* describes who is using *victim*."""
    return {
        "status": "confirm_required", "model": name,
        "detail": (
            f"loading '{name}' needs to free "
            f"'{victim}', which is {in_use}"),
    }


def inconclusive_restart_reason(name: str, attempts: int) -> str:
    """The reason handed to the hang alarm's self-restart."""
    return (f"GPU probe still inconclusive after "
            f"{attempts} attempts loading '{name}'")


def inconclusive_restarting_refusal(name: str) -> str:
    """The 503 detail when a still-inconclusive probe triggered a self-restart."""
    return (f"Cannot load '{name}' right now: the server "
            "detected a stuck GPU check and is restarting "
            "automatically. This page will reconnect once it "
            "comes back up.")


def inconclusive_refusal(name: str, attempts: int, elapsed: float) -> str:
    """The 503 detail when the probe stayed inconclusive and no self-restart
    was available."""
    return (f"Cannot load '{name}': tried measuring free VRAM "
            f"{attempts} times over about "
            f"{elapsed:.0f}s without a conclusive reading, could not "
            "free anything, and automatic recovery is unavailable. "
            "Please file a bug report if this keeps happening.")
