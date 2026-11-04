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

# unload_for_room setting values.
UNLOAD_ALWAYS = "always"   # unload this instance's idle models without asking
UNLOAD_ASK = "ask"         # ask before unloading anything
UNLOAD_NEVER = "never"     # load beside the other models with what is left
UNLOAD_POLICIES = (UNLOAD_ALWAYS, UNLOAD_ASK, UNLOAD_NEVER)

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
    ``plugin_backend`` is True for a model a plugin-registered backend runs;
    such a load, like a GGUF one, is sized by its backend before the first
    eviction pass.
    ``backend_need`` is the free VRAM the backend's own sizing needs to put
    every layer on the GPU (``Engine.gpu_offload_needs``), or None when
    it is not known. ``backend_sized`` is True when ``backend_need`` is the
    load's complete need, its compute overhead included (a GGUF load), so it
    replaces the whole-model estimate. ``min_need`` is the free VRAM the
    backend's least demanding placement that keeps every layer on the GPU
    needs (a Mixture-of-Experts model with its routed experts in system RAM),
    or None when it is not known or equals ``backend_need``.
    ``unload_policy`` is the ``unload_for_room`` setting (``UNLOAD_POLICIES``).
    ``experts_on_disk`` is True for a checkpoint whose routed experts the backend reads from
    its files as they are needed when it does not fit.
    """

    name: str
    vram_required: int
    headroom: int
    resident_cap: Optional[int]
    pinned: frozenset
    check_split_fit: bool
    backend_need: Optional[int] = None
    plugin_backend: bool = False
    backend_sized: bool = False
    min_need: Optional[int] = None
    unload_policy: str = UNLOAD_ALWAYS
    experts_on_disk: bool = False

    @property
    def needed_bytes(self) -> int:
        """The free VRAM a load must see to be admitted beside resident models
        with every layer and every expert on the GPU: ``backend_need`` when
        ``backend_sized``, else the larger of ``vram_required + headroom`` and
        ``backend_need``. The bar every other free-VRAM comparison uses, except
        the split's per-device check (``split_share_bytes``) and
        ``plan_bytes``."""
        if self.backend_sized and self.backend_need:
            return self.backend_need
        return max(self.vram_required + self.headroom, self.backend_need or 0)

    @property
    def plan_bytes(self) -> int:
        """The free VRAM below which the load moves layers to the CPU:
        ``min_need`` when it is known and below ``needed_bytes`` (between the
        two, the backend keeps routed experts in system RAM and every layer on
        the GPU), else ``needed_bytes``."""
        if self.min_need is not None and 0 < self.min_need < self.needed_bytes:
            return self.min_need
        return self.needed_bytes

    @property
    def keeps_experts_in_ram(self) -> bool:
        """True when ``plan_bytes`` is below ``needed_bytes``: some free VRAM
        between them loads every layer on the GPU with routed experts in system
        RAM."""
        return self.plan_bytes < self.needed_bytes

    @property
    def split_share_bytes(self) -> int:
        """``vram_required + headroom``: the amount the configured split's
        per-device shares are checked against."""
        return self.vram_required + self.headroom

    @property
    def whole_model_bytes(self) -> int:
        """The larger of ``vram_required`` and ``backend_need``: the VRAM the
        load is reported to need."""
        return max(self.vram_required, self.backend_need or 0)


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
    read it. Each bounded step (embedder eviction, the stop of or wait for this
    process's native media runtimes, each peer instance, the busy
    victim, the wait for a busy model to go idle, the longer wait for an evicted
    model's VRAM release) runs at most once per attempt, and inconclusive probes
    are retried ``_INCONCLUSIVE_LOAD_RETRIES`` times, so the loop always
    terminates. ``started`` is the ``time.monotonic()`` reading the attempt began
    at. ``deferred_to_backend`` is set when the load goes ahead below the
    whole-model estimate because nothing more could be evicted.
    """

    started: float
    asked_peers: set = field(default_factory=set)
    embedder_attempted: bool = False
    media_attempted: bool = False
    busy_attempted: bool = False
    busy_waited: bool = False
    release_wait_extended: bool = False
    inconclusive_retries: int = 0
    deferred_to_backend: bool = False


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


def unload_policy(cfg: Mapping) -> str:
    """The ``unload_for_room`` setting of *cfg*: one of ``UNLOAD_POLICIES``,
    ``UNLOAD_ALWAYS`` when unset or not one of them."""
    value = cfg.get("unload_for_room") if isinstance(cfg, Mapping) else None
    return value if value in UNLOAD_POLICIES else UNLOAD_ALWAYS


def effective_unload_policy(policy: str, *, preempt: bool, force: bool,
                            keep_others: bool) -> str:
    """The unload policy one load applies: ``UNLOAD_NEVER`` when the caller
    chose to load beside the other models (*keep_others*); ``UNLOAD_ALWAYS``
    after the user confirmed unloading (*force*); ``UNLOAD_NEVER`` for an
    ``UNLOAD_ASK`` load that is not an explicit switch (*preempt*), which has
    nobody to ask; else *policy*."""
    if keep_others:
        return UNLOAD_NEVER
    if force:
        return UNLOAD_ALWAYS
    if policy == UNLOAD_ASK and not preempt:
        return UNLOAD_NEVER
    return policy


def decide_admission(probe: VramProbe, budget: LoadBudget, lru: Iterable[str],
                     engines: Mapping[str, Any]) -> AdmissionDecision:
    """Decide whether ``budget.name`` may load on this reading, and if not,
    which idle resident model to evict first.

    ADMIT when free VRAM covers ``budget.needed_bytes`` alongside the
    residents (``residency.fits_alongside_residents``) and the load stays
    within the resident cap.
    Otherwise the least-recently-used safe victim
    (``residency.pick_eviction_victim``) is EVICT_IDLE; with no victim it is
    ADMIT_OVER_CAP when VRAM fits and only the cap wanted room, and EXHAUSTED
    when VRAM does not fit. The victim is only looked up when admission fails.
    """
    over_cap = residency.exceeds_resident_cap(lru, budget.name, budget.resident_cap)
    vram_ok = residency.fits_alongside_residents(
        free_vram=probe.free, vram_required=budget.needed_bytes,
        probe_ok=probe.probe_ok, headroom=0,
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


def idle_wait_candidate(budget: LoadBudget, lru: Iterable[str],
                        engines: Mapping[str, Any], *, preempt: bool,
                        already_waited: bool) -> Optional[str]:
    """The serving model a non-explicit load waits for, so it can be evicted
    once idle instead of loading beside it, or None.

    Only a load that is not an explicit switch (*preempt* False) waits: an
    explicit switch cancels the busy model instead (``busy_victim_candidate``).
    Once per load attempt. The candidate is
    ``residency.pick_busy_eviction_victim``: never the requested model, a
    pinned one or one already mid-unload."""
    if preempt or already_waited:
        return None
    return residency.pick_busy_eviction_victim(
        lru, engines, requested=budget.name, pinned=budget.pinned)


def evictable_residents(budget: LoadBudget, lru: Iterable[str],
                        engines: Mapping[str, Any]) -> list:
    """Every resident chat model an explicit switch to ``budget.name`` may
    evict, idle or busy, least recently used first: the same exclusions as
    ``residency.pick_busy_eviction_victim`` (never the requested model, a
    pinned one, one no longer registered or one already mid-unload)."""
    out = []
    for candidate in lru:
        if candidate == budget.name or candidate in budget.pinned:
            continue
        engine = engines.get(candidate)
        if engine is None or getattr(engine, "unloading", False) is True:
            continue
        out.append(candidate)
    return out


def free_after_eviction(probe: VramProbe, holds: Iterable[int]) -> Optional[int]:
    """The free VRAM expected once every model holding *holds* (bytes each,
    0 when unknown) is evicted: ``probe.free`` plus their sum.

    None when it cannot be estimated: the reading is inconclusive, carries no
    free figure, counts only this process's allocations, shows a split
    shortfall, or any hold is unknown."""
    holds = list(holds)
    if (not probe.probe_ok or probe.free is None or probe.process_scoped
            or probe.shortfall or any(h <= 0 for h in holds)):
        return None
    return probe.free + sum(holds)


def experts_in_ram_fit(probe: VramProbe, budget: LoadBudget) -> bool:
    """True when the load fits *probe* with routed experts in system RAM and
    every layer on the GPU (``budget.keeps_experts_in_ram`` and a trusted
    device-wide reading of at least ``budget.plan_bytes`` free, with no split
    shortfall), so it may load without freeing more."""
    return (budget.keeps_experts_in_ram and probe.probe_ok and probe.free is not None
            and not probe.process_scoped and not probe.shortfall
            and probe.free >= budget.plan_bytes)


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


def degraded_load_confirm(name: str, budget: LoadBudget, free: int,
                          unloaded: Iterable[str] = (), *, shared: bool = False,
                          peers: bool = False) -> dict:
    """The confirm_required result asking before a load that moves layers to
    the CPU: *free* is the free VRAM the load would see, after unloading the
    models named in *unloaded*, the shared embedding, reranker and speech
    models (*shared*) and other localm instances' models (*peers*) when any
    of those is given. The detail names the free VRAM every layer on the GPU
    needs (``budget.plan_bytes``) and, for a Mixture-of-Experts model
    (``budget.keeps_experts_in_ram``), what every expert on the GPU too would
    need (``budget.needed_bytes``).

    Besides ``detail``, the result carries ``reason`` "degraded_load",
    ``plan_mb``, ``full_mb``, ``free_mb``, ``moe``, ``experts_on_disk``, ``unloaded`` (a list),
    ``unloads_shared`` and ``unloads_peers`` for a client that words the
    question itself."""
    unloaded = list(unloaded)
    moe = budget.keeps_experts_in_ram
    plan_mb, full_mb, free_mb = (budget.plan_bytes // _MB, budget.needed_bytes // _MB,
                                 max(0, free) // _MB)
    what = [repr(n) for n in unloaded]
    if shared:
        what.append("the shared embedding, reranker and speech models")
    if peers:
        what.append("the models of other localm instances")
    where = (f"about {free_mb} MB would be free after unloading {', '.join(what)}"
             if what else f"about {free_mb} MB is free")
    if moe:
        need = (f"even with its experts in system RAM, every layer on the GPU "
                f"needs about {plan_mb} MB of free VRAM (about {full_mb} MB with "
                f"every expert on the GPU too)")
    else:
        need = f"every layer on the GPU needs about {plan_mb} MB of free VRAM"
    return {
        "status": "confirm_required", "model": name, "reason": "degraded_load",
        "plan_mb": plan_mb, "full_mb": full_mb, "free_mb": free_mb, "moe": moe,
        "experts_on_disk": budget.experts_on_disk,
        "unloaded": unloaded, "unloads_shared": shared, "unloads_peers": peers,
        "detail": (f"'{name}' does not fit: {need}, and {where}. "
                   + ("Loading it anyway keeps its routed experts on disk and reads them as "
                      "they are needed, so replies are very slow."
                      if budget.experts_on_disk else
                      "Loading it anyway runs some of its layers on the CPU, which is slower.")),
    }


def unload_ask_confirm(name: str, models: Iterable[str], freed: int, *,
                       shared: bool = False, peers: Iterable[str] = (),
                       busy: Iterable[str] = ()) -> dict:
    """The confirm_required result of the "ask" unload policy: loading *name*
    unloads the models named in *models* (those also in *busy* are answering a
    request, which unloading stops), the shared embedding, reranker and speech
    models (*shared*) and the models of the other localm instances named in
    *peers*, which together free about *freed* bytes (0 when unknown).
    Carries ``reason`` "unload_ask", ``unloaded``, ``busy``,
    ``unloads_shared``, ``peers`` and ``freed_mb`` besides ``detail``."""
    models, peers, busy = list(models), list(peers), list(busy)
    what = [repr(n) + (" (stopping the reply it is writing)" if n in busy else "")
            for n in models]
    if shared:
        what.append("the shared embedding, reranker and speech models")
    what += [f"{p} of another localm instance" for p in peers]
    frees = f", which frees about {freed // _MB} MB of VRAM" if freed > 0 else ""
    return {
        "status": "confirm_required", "model": name, "reason": "unload_ask",
        "unloaded": models, "busy": busy, "unloads_shared": shared, "peers": peers,
        "freed_mb": max(0, freed) // _MB,
        "detail": f"Loading '{name}' unloads {', '.join(what)}{frees}.",
    }


def unload_choice_confirm(name: str, budget: LoadBudget, free: int,
                          models: Iterable[str]) -> dict:
    """The confirm_required result offering the choice for an explicit switch
    that does not fully fit beside the loaded models: *free* bytes are free
    with *models* still loaded. The caller offers three answers: load beside
    them with what is left (``keep_others``), unload them for this load only
    (``force``), or cancel. When *free* covers ``budget.plan_bytes``, every
    layer stays on the GPU beside them with routed experts in system RAM
    (``fits_with_experts_in_ram``), and the text names the free VRAM every
    expert on the GPU would need. Carries ``reason`` "unload_choice",
    ``unloaded``, ``plan_mb``, ``full_mb``, ``free_mb``, ``moe`` and
    ``fits_with_experts_in_ram`` besides ``detail``."""
    models = list(models)
    plan_mb, full_mb, free_mb = (budget.plan_bytes // _MB, budget.needed_bytes // _MB,
                                 max(0, free) // _MB)
    named = ", ".join(repr(n) for n in models) or "the other models"
    fits = budget.keeps_experts_in_ram and free >= budget.plan_bytes
    if fits:
        detail = (f"'{name}' needs about {full_mb} MB of free VRAM to keep every expert "
                  f"on the GPU and about {free_mb} MB is free beside {named}. Load it "
                  "beside them with some of its experts in system RAM (slower), or "
                  "unload them for this load.")
    else:
        detail = (f"'{name}' needs about {plan_mb} MB of free VRAM to run every "
                  f"layer on the GPU and about {free_mb} MB is free beside {named}. "
                  "Load it beside them, slower, or unload them for this load.")
    return {
        "status": "confirm_required", "model": name, "reason": "unload_choice",
        "unloaded": models, "plan_mb": plan_mb, "full_mb": full_mb, "free_mb": free_mb,
        "moe": budget.keeps_experts_in_ram, "fits_with_experts_in_ram": fits,
        "detail": detail,
    }


def busy_victim_confirm(name: str, victim: str, in_use: str) -> dict:
    """The confirm_required result asking before evicting a model still in use.
    *in_use* describes who is using *victim*."""
    return {
        "status": "confirm_required", "model": name,
        "detail": (
            f"loading '{name}' needs to free "
            f"'{victim}', which is {in_use}; loading it anyway stops "
            f"'{victim}' and cuts off the reply it is writing"),
    }


def busy_media_confirm(name: str, runtimes: list) -> dict:
    """The confirm_required result asking before loading *name* beside this
    process's native media *runtimes* (display names) while a job runs on
    them."""
    which = " and the ".join(runtimes)
    return {
        "status": "confirm_required", "model": name,
        "detail": (
            f"'{name}' needs the VRAM localm's own {which} holds, which is "
            f"running a job; loading it anyway can put part of '{name}' on the CPU, "
            "which is slower (the job is not stopped)"),
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
