# SPDX-License-Identifier: AGPL-3.0-or-later
"""
OpenAI-compatible HTTP inference server built with FastAPI + uvicorn.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (streaming + non-streaming, multimodal-capable)

Start programmatically:
    from localm.inference.http_server import serve
    serve(engine, host="127.0.0.1", port=8642)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)
from fastapi.security import HTTPBearer

from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback_host  # noqa: F401  (re-export for back-compat)
from localm.inference.backends.base import ModelLoadCancelled
from localm.inference.chat_pipeline import ChatPipeline
from localm.inference import residency
from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatResponse,
    FullChoice, Message, UsageInfo, make_chunk_id,
)

# Map of display name -> Engine instance
_engines: dict[str, Engine] = {}
# Order of model usage (display names, MRU at the end)
_engines_lru: list[str] = []
# Display names currently mid-eviction: detached from _engines/_engines_lru
# already (BUG-9b's fix, so a fast-path lookup correctly sees them as gone),
# but the native free (evict_engine.unload(), an executor call the eviction
# loop awaits) has not completed yet. A concurrent switch_engine/get_engine
# call for THIS SAME name has no other way to see that a free is in flight
# for it: it would otherwise construct-and-load a brand-new engine for the
# name while the stale eviction is still running, race it, and (once the
# stale unload() finally completes) that caller can end up pinning an engine
# that gets freed out from under it - the exact pin-arrives-during-the-
# unload-await hazard BUG-9b closed for the FAST path, reopened for this one
# by #753's fall-through-to-a-real-load-attempt change (confirmed by bisection:
# test_eviction_victim_race.py::test_eviction_victim_not_pinnable_during_native_free
# passes at #752, fails at #753). switch_engine consults this before
# constructing/loading *name* and refuses (503, honest backpressure - the
# test's own documented acceptable outcome) rather than racing it.
_evicting_names: set[str] = set()
# Default/startup model name
_default_model_name: str | None = None
# Active model name (most recently used/loaded)
_active_model_name: str | None = None

# The server's audit log, published by create_app() (see the `global _audit`
# there). Must default to None here so _do_restart's `global _audit` guard
# always resolves to a real (possibly None) value even if it runs in a
# process/interpreter where create_app() was never called first - reading an
# unassigned global raises NameError, not a clean "None" check.
_audit = None

# Inference serialisation - per-model semaphores mapping display name -> Semaphore
_inference_sems: dict[str, asyncio.Semaphore] = {}

# Backward compatibility references
_engine: Engine | None = None
_inference_sem: asyncio.Semaphore | None = None

# The server's running event loop, captured once at lifespan startup so an OFF-loop
# worker thread (notably the jobs runner, which runs on a run_in_executor thread) can
# submit a coroutine back ONTO it via asyncio.run_coroutine_threadsafe. Used to route
# a shared-engine unload through the guarded unload_one_model ON the loop, where
# get_engine and the synchronous request _pin also run - the event loop is the
# serialization point that makes eviction safe (a bare off-loop engine.unload() races
# get_engine's fast path and ignores the in-flight pin). None until a real server
# lifespan runs (a bare create_app() test app / headless import never sets it), so an
# off-loop caller detects "no loop" and degrades safely instead of racing the registry.
_server_loop: "asyncio.AbstractEventLoop | None" = None

# Preemptive model switching (see switch_engine). _switch_desired = most-recent
# switch request; _switch_loading = model whose load is in flight; _switch_cancel
# aborts it. Touched only on the event-loop thread inside switch_engine, except the
# cancel event, which the loader-thread load-progress callback reads
# (threading.Event is thread-safe).
_switch_desired: Optional[str] = None
_switch_loading: Optional[str] = None
_switch_cancel: Optional["threading.Event"] = None

# Cross-install GPU/VRAM coordination (multi-instance, see localm.gpu_registry).
# None until lifespan startup populates it, and ONLY for a real, non-isolated,
# instances.advertise()'d server (app.state.instance_id set, instance_isolated
# falsy). A plain create_app() test app or an --isolated run never sets it, so it
# never touches the shared machine-wide registry (zero-daemon). Shape:
# {"instance_id", "port", "host", "scheme", "token"}.
_gpu_coord: Optional[dict] = None

# Hang watchdog: a monotonic heartbeat bumped every _HEARTBEAT_INTERVAL_S by
# _hang_heartbeat_loop (an async task ON the loop) and read by the off-loop
# watchdog thread + the debug request log. A growing (now - _hb_monotonic)
# means the single event loop has stopped making progress, i.e. something is
# blocking it (the diagnosed hang) - the off-loop watchdog thread compares
# this raw gap against its own multi-second threshold (hang_watchdog_
# threshold(), 10s by default) and is unaffected by the note below.
#
# _loop_lag_seconds() (below) answers a DIFFERENT question and must be used
# for anything reported to a human. Raw (now - _hb_monotonic) saws between 0
# and ~_HEARTBEAT_INTERVAL_S on a perfectly healthy loop - that is just how
# far into the current tick cycle "now" happens to land, not evidence of lag
# (GitHub #955/#950: this raw value was logged as "loop_lag" on every debug
# request, so a healthy server reported up to a full second of fake "lag" on
# a sawtooth, sending the maintainer chasing ghosts). Subtracting the
# interval turns it into a real scheduling-delay figure: ~0 when healthy,
# and positive only when a tick itself was late, i.e. something actually
# blocked the loop.
_HEARTBEAT_INTERVAL_S = 1.0
_hb_monotonic = time.monotonic()


def _loop_lag_seconds() -> float:
    """Real event-loop scheduling delay, in seconds - NOT time-since-last-
    heartbeat-tick (see the comment above _hb_monotonic). ~0.0 on a healthy
    loop; grows only when a heartbeat tick was itself delayed, meaning
    something blocked the loop. This is what gets reported to a human (the
    debug request log, /debug/stacks); the raw gap is for the watchdog's own
    large-threshold hang detection only.

    RESOLUTION LIMIT, by construction: a stall shorter than
    _HEARTBEAT_INTERVAL_S (currently 1.0s) reads as exactly 0.0, identical to
    a perfectly healthy loop - a 1Hz heartbeat cannot see a sub-interval
    block. "loop_lag=0.0" therefore means "no stall LONGER than the
    heartbeat interval was detected", not "the loop was never blocked at
    all". A finer-grained sampler would close this gap at the cost of a
    second background task and more wakeups purely for a diagnostic counter,
    which is not worth it: the hang watchdog (large-threshold, above) already
    owns detecting a real freeze; this value is for correlating a slow
    request with a preceding stall, not for catching sub-second ones."""
    return max(0.0, (time.monotonic() - _hb_monotonic) - _HEARTBEAT_INTERVAL_S)

def _default_engine_factory(name: str) -> Engine:
    from localm.config import load_registry
    from localm.model_manager import get_model_info, get_model_mmproj
    info = get_model_info(name)
    if info is None:
        raise ValueError(f"Model not found: {name}")
    m_path, m_hint = info
    mmproj = get_model_mmproj(name)
    return Engine(
        str(m_path),
        display_name=name if name in load_registry() else m_hint,
        mmproj_path=mmproj,
    )

_engine_factory = _default_engine_factory


def _model_file_size(name: str) -> Optional[int]:
    """Best-effort on-disk size for registered model *name*, or None when not
    resolvable (e.g. under pytest with no registry). Mirrors switch_engine's
    own file_size computation (residency.model_footprint_bytes) for the
    single-file-vs-directory STAT LOGIC, so the VRAM estimate written to the
    coordination registry is consistent with the number switch_engine itself
    used to decide whether eviction was needed - but deliberately NOT for the
    empty-directory return value: model_footprint_bytes returns int (never
    None) because its caller always needs a numeric eviction-admission input,
    even a 0 one; this function feeds a registry field other instances treat
    as "how much VRAM does this peer hold", where a 0 is read as a REAL
    measurement (see gpu_registry.py's vram_estimate_bytes and its one
    consumer's `isinstance(e, int) and e > 0` guard). rglob() matching no
    files - an empty directory, or one whose real weights sit somewhere
    rglob does not look - means the size genuinely was not measured, and
    reporting a suspiciously-precise 0 for "unknown" is exactly the
    collapsed-two-outcomes failure AGENTS.md rule 5 warns about, even though
    today's one consumer happens to treat 0 the same as None already."""
    try:
        from pathlib import Path as _Path
        from localm.model_manager import get_model_info
        info = get_model_info(name)
        if info is None:
            return None
        m_path, _ = info
        if not m_path:
            return None
        p = _Path(m_path)
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return total if total > 0 else None
    except (OSError, TypeError, ValueError):
        return None
    return None


def _current_gpu_index() -> int:
    """The configured main GPU device index (0 when unset/unconfigured) - see
    ``main_gpu_index`` / ``discover.resolve_main_gpu_index``, the same
    resolution ``vram_info()`` and the GGUF backend's own VRAM check use."""
    try:
        from localm.config import load_config
        from localm.discover import resolve_main_gpu_index
        return resolve_main_gpu_index(load_config().get("main_gpu_index"))
    except Exception:
        return 0


def _gpu_registry_sync() -> None:
    """Best-effort: write this instance's current model/VRAM state to the
    cross-install GPU coordination registry (called on every successful model
    load/unload, plus a periodic heartbeat). A no-op when this instance is not
    registered for coordination (``_gpu_coord`` unset - a plain test app or an
    ``--isolated`` run never reaches the shared registry directory at all).

    Never raises into the caller: a registry write failure must not break the
    model load/unload it is piggybacking on (RULE 5 - logged, not silenced)."""
    global _gpu_coord
    if not _gpu_coord:
        return
    try:
        import os as _os
        from localm import gpu_registry
        model = _active_model_name
        vram_bytes = None
        if model:
            size = _model_file_size(model)
            if size is not None:
                vram_bytes = int(size * 1.2)
        gpu_registry.write_entry(
            gpu_registry.registry_dir(),
            instance_id=_gpu_coord["instance_id"],
            pid=_os.getpid(),
            port=_gpu_coord.get("port"),
            host=_gpu_coord.get("host") or "127.0.0.1",
            scheme=_gpu_coord.get("scheme") or "http",
            model=model,
            vram_estimate_bytes=vram_bytes,
            gpu_index=_current_gpu_index(),
            coordination_token=_gpu_coord["token"],
        )
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("gpu-registry sync failed (continuing): %s", e)


def _load_gpu_indices() -> set:
    """Every device whose free VRAM this instance's next model load can actually
    USE - the whole configured split when one is active, else just the main
    device.

    NOT ``{_current_gpu_index()}``: that is an IDENTITY answer ("which one device
    is primary"), and resolve_main_gpu_index(None) returns 0 for an unconfigured
    main_gpu_index even on a box whose split spans 0 AND 1. Weighing peers
    against that single index while weighing VRAM against vram_capacity()'s
    COMBINED split total contradicts itself, and dropped a sibling holding VRAM
    on this instance's own second split device - turning a cooperative unload
    that used to succeed into a 503 (the capacity-vs-identity distinction the
    raw-accessor guard in scripts/check_hygiene.py exists to enforce).

    Known limitation, stated rather than hidden (rule 5): a registry entry
    advertises ONE ``gpu_index`` per instance (see _gpu_registry_sync), so a
    SPLIT peer is represented only by its main device. A split peer whose main
    device is outside our set is therefore still skipped even though it may hold
    VRAM on a device we do use. Widening the entry to a device LIST is a registry
    schema change, out of scope here; the effect is a missed cooperation
    opportunity (the pre-existing 503), never a wrong yank."""
    try:
        from localm.config import load_config
        from localm.discover import resolve_gpu_split
        cfg = load_config()
        pairs = resolve_gpu_split(cfg.get("gpu_split_indices"),
                                  cfg.get("gpu_split_ratios"))
        if len(pairs) >= 2:
            return {idx for idx, _ratio in pairs}
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("could not resolve the configured GPU split for the "
                   "cooperative-unload peer filter (%s); using the main device "
                   "only", e)
    return {_current_gpu_index()}


def _attempt_cooperative_unload(*, needed_bytes: Optional[int] = None,
                                free_bytes: Optional[int] = None,
                                asked: Optional[set] = None) -> bool:
    """Best-effort: ask a live sibling localm instance (found via the
    cross-install GPU-coordination registry) to release its own VRAM, so this
    instance does not have to give up and 503 just because ITS OWN local
    eviction candidates are all busy. Returns True once a peer confirms it
    freed its model.

    Cooperating COSTS the sibling every model it has loaded (the peer runs its
    own ``unload_all_models``), so this is deliberately conservative about when
    it is worth it (REG-454):

    - *asked* (a set of instance_ids, per load attempt) makes each peer
      answerable at most ONCE. The caller re-probes VRAM and calls back on
      success, and a peer that has already released advertises no new VRAM -
      but its entry can keep listing a model anyway (its own post-unload
      registry write is best-effort and may have failed, or it reloaded), and
      ``request_cooperative_unload`` reports success for "already_unloaded"
      too. Without this the caller's ``while True`` would keep re-asking the
      same peer forever, holding the per-model semaphore and re-probing VRAM,
      never progressing.
    - *needed_bytes*/*free_bytes* gate the yank on whether it could actually
      help: if every candidate advertises a VRAM estimate and freeing ALL of
      them still leaves this load short (a model far bigger than the card, or
      a third-party app such as ComfyUI holding the bulk of VRAM), the peers
      would lose their models for nothing and this load would 503 anyway. The
      pre-existing 503 is the honest answer; do not take the sibling down with
      us. An unknown estimate is not proof it cannot help, so it does not veto.

    Only a peer on a device THIS load can use is considered (see
    _load_gpu_indices - the whole configured split, not one index): freeing an
    unrelated card's VRAM cannot make this load fit.

    Only runs when THIS instance itself is registered for coordination
    (``_gpu_coord`` set - never for a plain test app or an ``--isolated`` run,
    so tests and isolated runs never touch the shared machine-wide registry
    directory or make an outbound loopback call). Fully advisory: ANY failure
    (no registry dir, no live peer, request timeout/refusal) is logged and
    returns False - the caller's pre-existing 503 remains the unchanged
    fallback (RULE 5: a failed cooperation attempt must never become a HARDER
    failure than today's baseline, and must never be silenced)."""
    global _gpu_coord
    from localm.debuglog import logger as _dbg
    if not _gpu_coord:
        return False
    try:
        from localm import gpu_registry
    except Exception as e:
        _dbg.debug("gpu_registry unavailable, skipping cooperative unload: %s", e)
        return False
    try:
        peers = gpu_registry.list_gpu_peers(exclude_self_id=_gpu_coord.get("instance_id"))
    except Exception as e:
        _dbg.warning("gpu-registry peer lookup failed (falling back to local-only "
                     "eviction): %s", e)
        return False
    # Only a peer actually holding a model has anything to free.
    holders = [p for p in peers if p.get("model")]
    if asked is not None:
        holders = [p for p in holders if p.get("instance_id") not in asked]
    my_gpus = _load_gpu_indices()
    holders = [p for p in holders
               if p.get("gpu_index") is None or p.get("gpu_index") in my_gpus]
    if not holders:
        return False

    if needed_bytes is not None and free_bytes is not None:
        estimates = [p.get("vram_estimate_bytes") for p in holders]
        if all(isinstance(e, int) and e > 0 for e in estimates):
            reclaimable = sum(estimates)
            if free_bytes + reclaimable < needed_bytes:
                _dbg.info(
                    "cooperative unload skipped: freeing all %d peer(s) on GPU(s) "
                    "%s would reclaim only ~%d MB on top of %d MB free, still short "
                    "of the ~%d MB this load needs - leaving their models alone",
                    len(holders), sorted(my_gpus), reclaimable // 1024 ** 2,
                    free_bytes // 1024 ** 2, needed_bytes // 1024 ** 2)
                return False

    for peer in holders:
        if asked is not None:
            asked.add(peer.get("instance_id"))
        try:
            ok = gpu_registry.request_cooperative_unload(peer)
        except Exception as e:
            _dbg.warning("cooperative-unload request to peer %s failed: %s",
                        peer.get("instance_id"), e)
            continue
        if ok:
            _dbg.info("cooperative unload: peer %s (port %s) released its model "
                      "to free VRAM for this load", peer.get("instance_id"), peer.get("port"))
            return True
        _dbg.warning("peer %s declined/failed cooperative unload", peer.get("instance_id"))
    return False


def _gpu_placement_fields(engine) -> dict:
    """{"gpu_layers_offloaded", "gpu_layers_total", "degraded"} for *engine*'s
    current load, or {} when the backend cannot report placement (no load
    yet, or a backend without a layer-count knob - see Engine.gpu_placement).
    Merged into every switch_engine()/load-route success payload so a caller
    can tell a full GPU load from a silent CPU fallback (AGENTS.md rule 5)
    instead of a bare "loaded"/"already_active" that hides it."""
    placement = getattr(engine, "gpu_placement", None)
    return dict(placement) if placement else {}


async def switch_engine(name: str, make_engine, *, on_active=None, preempt: bool = True) -> dict:
    global _engines, _engines_lru, _active_model_name, _engine_factory, _last_activity_per_model
    global _switch_desired, _switch_loading, _switch_cancel, _engine, _inference_sem

    # Preemption (a newer selection aborts an in-flight load) is SINGLE-slot and
    # belongs to an explicit user switch, not API-routed loads: with preempt=True
    # two concurrent DIFFERENT-model requests cancel each other and the earlier
    # 503s "superseded" (AUDIT-HIGH-3). So get_engine loads preempt=False (loads
    # coexist/queue); only the GUI/CLI switch_model path preempts.
    if preempt:
        _switch_desired = name
        if _switch_cancel is not None and _switch_loading != name:
            _switch_cancel.set()

    if make_engine is not None:
        _engine_factory = make_engine

    sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))

    loop = asyncio.get_running_loop()
    async with sem:
        if preempt and _switch_desired != name:
            return {"status": "superseded", "model": name, "by": _switch_desired}

        if name in _engines and _engines[name].loaded and getattr(_engines[name], "unloading", False) is not True:
            if name in _engines_lru:
                _engines_lru.remove(name)
            _engines_lru.append(name)
            _active_model_name = name
            _engine = _engines[name]
            _inference_sem = sem
            if on_active is not None:
                on_active(name)
            return {"status": "already_active", "model": name,
                    **_gpu_placement_fields(_engines[name])}

        # Perform VRAM check and eviction
        from localm import discover
        # By-symbol AND by-module deliberately: the by-symbol names are re-read from
        # the module on every call (this import is function-scoped), which is what
        # lets tests patch localm.discover.vram_capacity; the module handle is for the
        # constants, which no test patches.
        from localm.discover import gpu_split_shortfall, vram_capacity
        from localm.model_manager import get_model_info
        from localm.config import load_registry
        
        registry = load_registry()
        info = get_model_info(name)
        # file_size feeds the VRAM-eviction estimate below, which is gated on a
        # non-empty registry. So it is only needed for a registered model.
        file_size = 0
        if info is not None:
            m_path, _ = info
            file_size = residency.model_footprint_bytes(m_path)
        elif registry:
            # Registered against a real registry but the files are not on disk:
            # the shipped 404 contract. (Previously papered over with a fabricated
            # path + 4 GB size when pytest was importable, making the 404 path
            # untestable and running eviction math on fiction - AUDIT rule 5 / no
            # facade.)
            raise HTTPException(404, f"Model files not found: {name}")
        # else: empty registry (single-model / direct-path startup) - no size to
        # compute; the eviction block below is skipped for an empty registry.

        # Only perform eviction check if there are registered models
        if registry:
            from localm.inference.engine import _is_gguf
            from localm.inference import embedder as _embedder_mod
            from localm.vram import wait_for_vram_release
            # The admit margin, the victim-safety rules and the two policy knobs
            # live in inference/residency.py, shared with the MCP server's
            # EngineCache so the two serving layers cannot drift apart.
            vram_required = residency.required_vram_bytes(file_size)
            headroom = residency.DEFAULT_HEADROOM_BYTES
            from localm.config import load_config as _load_config
            _cfg = _load_config()
            resident_cap = residency.resident_cap(_cfg)
            pinned = residency.pinned_model_names(_cfg)
            # discover.gpu_split_shortfall's docstring has the full rationale:
            # vram_capacity() alone proves the AGGREGATE combined split free is
            # enough, but the GGUF/llama.cpp backend divides a model by a
            # STATIC per-config ratio with no live per-device capacity check of
            # its own (unlike the HF backend's device_map="auto", which already
            # self-corrects from live per-device free VRAM) - so an asymmetric
            # split (e.g. another already-loaded model sits on one device more
            # than another) can pass the aggregate check while one device's
            # actual share is short, reaching the native loader with too
            # little room on that device. Only applies to a GGUF-backend load.
            check_split_fit = _is_gguf(m_path)
            # Peers already asked to cooperate during THIS load attempt: each is
            # answerable once, so the loop below always makes progress (REG-454).
            asked_peers: set = set()
            # The shared embedder is attempted as an eviction candidate at most
            # once per load attempt, mirroring asked_peers just above: without
            # this bound, a concurrent get_embedder() reload from an unrelated
            # RAG/memory/coder-episode request between two iterations of this
            # loop could repopulate localm.inference.embedder._EMBEDDER between
            # attempts, making this branch fire again instead of the loop
            # converging to either a successful load or the final 503.
            embedder_evict_attempted = False

            while True:
                # Off the event loop: vram_capacity()/gpu_split_shortfall() route
                # through discover.list_gpus(), which is deadline-bounded (PR #541)
                # but still a REAL hardware probe that can take up to that deadline
                # - calling it directly on the loop would stall every other
                # concurrent request on this single-threaded server for that long,
                # the exact class of hang #541 fixed for the GUI routes and the
                # GPU-registry heartbeat. This loop can iterate (and re-probe)
                # multiple times per eviction, so it is just as exposed.
                # _GPU_PROBE_CLI_DEADLINE (now an alias of the module default, which
                # became cold-init-tolerant when the old 4.0s cap was retired): kept
                # explicit because THIS caller's correctness DEPENDS on waiting out a
                # cold driver init, and an explicit deadline documents that and pins
                # it against any future default change. History: this probe once
                # inherited that 4.0s cap while executor-offloaded (the cap existed
                # to guard the event loop, which this call was never on), and a COLD
                # ROCm/CUDA init (4.63s measured on this box, ~6.5s historically)
                # reliably overran it - so the first load after every server start
                # timed out, a fresh process had no last-known-good to serve,
                # free_vram was None -> "unmeasurable" -> the gate SKIPPED ENTIRELY
                # (see the state split below). Waiting out the cold init turns the
                # most common load there is - the first one - from unguarded into
                # properly checked. Retrying at a longer deadline AFTER a timeout
                # cannot do this: an overrun probe is abandoned, not cancelled, so
                # _gpu_probe_inflight stays True and every retry short-circuits to
                # (last-known-good, GPU_PROBE_BUSY) in 0.0s without probing at all -
                # measured. The budget must be spent on the FIRST call or not at all.
                #
                # wait_for_inflight=True (#701) closes the remaining window: when a
                # CONCURRENT probe already holds the slot - the GUI polls /api/stats
                # every 2500ms, whose probe holds _gpu_probe_inflight through the cold
                # init - this call would otherwise get an instant BUSY + stale reading
                # and (nothing to evict on a fresh server) refuse spuriously. Instead
                # it JOINS that in-flight probe and waits on ITS result, up to our
                # deadline. Safe only because we are executor-offloaded here. So "open
                # GUI, click a model" on a cold box now gets a real reading, not a
                # spurious 503.
                v_info, probe_status = await loop.run_in_executor(
                    None, functools.partial(
                        vram_capacity, return_status=True,
                        deadline=discover._GPU_PROBE_CLI_DEADLINE,
                        wait_for_inflight=True))
                free_vram = v_info.get("free")
                # v_info may also carry a "free_scope" tag (FREE_SCOPE_DEVICE vs
                # FREE_SCOPE_PROCESS, PR #697/#700): whether `free` counts all
                # processes or only this one. This gate DELIBERATELY keys only on
                # probe_status, not free_scope, for THIS change's scope. The asymmetry:
                #  - REFUSE direction: ignoring it is unconditionally safe.
                #  - PERMIT direction: a PROCESS-scoped `free` OVER-reports (blind to
                #    other processes), so permitting on it can OOM. As of #700 the label
                #    is accurate (PROCESS means genuinely process-local, no longer
                #    over-firing onto correct multi-NVIDIA), so a future permit-side
                #    caution on a PROCESS reading (prefer single-resident) is POSSIBLE -
                #    left out here deliberately, not because the signal is unusable. See
                #    dev-notes/pr697-followup-review.md.
                # Residual permit-on-PROCESS OOM risk, and why it is small and accepted:
                # when this gate STARTS the probe (no concurrent heartbeat), deadline=
                # _GPU_PROBE_CLI_DEADLINE funds #697's cold device-global source, so
                # `free` is DEVICE-scoped and correct. When it JOINS a concurrent probe
                # (the 2500ms /api/stats heartbeat, wait_for_inflight below), it
                # inherits the STARTER's thinner 4s budget, under which the cold ADL
                # source is skipped - so for ONE cold cycle the joined reading can be
                # PROCESS-scoped and the gate may permit on a blind number. It self-
                # heals: once the ADL source is warm every later reading is DEVICE. This
                # one-cycle window is an accepted transient (surfaced as a known gap in
                # the PR), not silently assumed away.
                # Two states that master conflated under one `measurable` flag, and
                # they want OPPOSITE handling (AGENTS.md rule 5 - do not collapse a
                # benign case into an unknown one):
                #   probe_ok and free is None -> this BOX cannot report free VRAM at
                #       all (CPU-only, GGUF-only without nvidia-smi, the Windows
                #       registry tier). PERMANENT. Best-effort single-resident load is
                #       correct and is the shipped behaviour - refusing would brick it.
                #   not probe_ok            -> THIS PROBE was inconclusive; the box may
                #       well be measurable. TRANSIENT. NOT a licence to skip the gate.
                probe_ok = probe_status == discover.GPU_PROBE_OK
                measurable = free_vram is not None
                cannot_measure = probe_ok and not measurable
                inconclusive = not probe_ok
                # + headroom for consistency with the aggregate check just
                # below, which also demands vram_required + headroom, not
                # bare vram_required - a per-device share should not be held
                # to a thinner margin than the aggregate ceiling it composes
                # with (see gpu_split_shortfall's own docstring: it does not
                # bake in headroom itself, that is the caller's decision).
                # No return_status opt-in, and that is SAFE as of #699:
                # gpu_split_shortfall is status-aware and admits-on-non-OK (a
                # stale/timed-out per-device probe yields an EMPTY shortfall, never a
                # fabricated one), and it omits a device whose reading is blind/stale
                # rather than quoting it. So this call can no longer turn a stale
                # reading into a spurious refusal or a quoted figure. It also runs
                # WARM here: the vram_capacity probe above already paid the cold-init
                # cost, so gpu_split_shortfall's own probe completes fast. The one
                # thing it does NOT get is #701's in-flight JOIN (wait_for_inflight)
                # - a tiny follow-up owned by the gpu_split_shortfall session, and
                # moot in practice given the warm-driver ordering. Passing the
                # deadline/status through is therefore unnecessary.
                #
                # return_shares_adaptive: the refuse-vs-defer branch below MUST know
                # whether the shares were the live auto free-VRAM-proportional ones
                # (their all-short can only mean combined-short -> defer) or static
                # pinned/equal-fallback shares (the pre-feature per-device hazard ->
                # hard 503). Config shape alone cannot answer that: auto can DECLINE
                # (stale index, unmeasurable device) and fall back to equal shares
                # with ratios still unset - see gpu_split_shortfall's docstring.
                shortfall, shares_adaptive = (
                    await loop.run_in_executor(
                        None, functools.partial(
                            gpu_split_shortfall, vram_required + headroom,
                            return_shares_adaptive=True))
                    if check_split_fit else ([], False))
                # PERMIT only on a reading that was actually taken. A frozen
                # last-known-good that happens to read HIGH would otherwise pass here
                # and admit a load with ZERO eviction on top of resident peers - the
                # same stale int that produces a dishonest 503 in the LOW direction
                # produces a native OOM / driver TDR in the HIGH one.
                # `measurable` is folded into the shared predicate (free_vram is
                # not None); it stays a local because the refuse-vs-defer branch
                # below still distinguishes cannot-measure from inconclusive.
                over_cap = residency.exceeds_resident_cap(
                    _engines_lru, name, resident_cap)
                vram_ok = residency.fits_alongside_residents(
                    free_vram=free_vram, vram_required=vram_required,
                    probe_ok=probe_ok, headroom=headroom, shortfall=shortfall)
                if vram_ok and not over_cap:
                    break

                # Make room. Measurable VRAM: evict idle models until the new one
                # fits (also satisfying any configured split device's own share
                # - see check_split_fit above). Cannot measure (default GGUF-only
                # / non-NVIDIA, no "free" from discover.vram_info) or inconclusive:
                # cannot prove it fits alongside others, so fall back to
                # single-resident (evict every idle model first) rather than stacking
                # until the driver OOMs (AUDIT-CRIT-2). The two diverge only once
                # eviction is exhausted, below: cannot-measure loads best-effort,
                # inconclusive refuses.
                # Victim safety (never the requested model, never one that is
                # serving, never one another path is already mid-freeing, never a
                # pinned one) lives in residency.pick_eviction_victim, shared with
                # the MCP server's cache. Its docstring carries the full rationale
                # for each skip, in particular why active_requests==0 alone is not
                # enough to rule out a concurrent double free.
                evict_name = residency.pick_eviction_victim(
                    _engines_lru, _engines, requested=name, pinned=pinned)

                if evict_name is None and vram_ok:
                    # ONLY the resident cap wanted room, and nothing could be
                    # freed (every peer is pinned or serving). VRAM already
                    # fits, so this is a user PREFERENCE going unmet, not a
                    # safety constraint. Load, and say the cap was missed.
                    # Falling through to the exhaustion path below would answer
                    # a preference with a 503 - and, worse, would first ask a
                    # SIBLING localm instance to dump its models to free VRAM
                    # that was never short. Being one model over a soft cap is
                    # strictly better than either.
                    from localm.debuglog import logger as _dbg
                    _dbg.warning(
                        "max_resident_models=%s wanted room for %s but no "
                        "resident model could be evicted (resident=%s, "
                        "pinned=%s); free VRAM is sufficient, so loading it "
                        "anyway over the cap",
                        resident_cap, name, list(_engines_lru), sorted(pinned))
                    break

                if evict_name is None:
                    # Nothing idle among the chat engines. The shared embedder is a
                    # separate lifecycle from _engines (see localm.inference.embedder's
                    # module docstring): it is loaded independently by RAG/memory/
                    # coder-episode callers via get_embedder(), never through
                    # switch_engine, so the LRU scan above can never see it even
                    # though it can hold real, evictable VRAM. Without this, a chat
                    # load that only needed to reclaim the embedder's VRAM fell
                    # straight through to the cooperative-unload / final-503 handling
                    # below having never tried the cheapest, purely-local option
                    # first - a resident-but-idle embedder left over from a RAG/
                    # memory run made every next chat load 503 with "not enough
                    # VRAM" even though the embedder alone was the entire shortfall.
                    # Honor the in-flight-request pin (AUDIT-CRIT-1) exactly like
                    # unload_all_models's own embedder release does: a request
                    # mid-embed() must not have its embedder (and the isolated
                    # worker process it is waiting on) freed out from under it.
                    # reset_embedder(force=False) checks active_requests()==0 and
                    # clears the embedder in ONE locked step, not a separate
                    # active_requests() call followed by an unconditional
                    # reset_embedder() - that used to leave a TOCTOU window, since
                    # IsolatedEmbedder.embed() pins active_requests without taking
                    # embedder._LOCK.
                    #
                    # embedder_evict_attempted (set above the while loop) bounds
                    # this to once per load, same rationale as asked_peers just
                    # below: loaded_dim() answers None again once THIS attempt
                    # clears it, but a concurrent get_embedder() from an unrelated
                    # request could repopulate it before this loop's next pass, so
                    # "it clears itself" alone is not a progress guarantee.
                    if not embedder_evict_attempted:
                        embedder_dim = await loop.run_in_executor(
                            None, _embedder_mod.loaded_dim)
                        if embedder_dim is not None:
                            embedder_evict_attempted = True
                            cleared = await loop.run_in_executor(
                                None, functools.partial(
                                    _embedder_mod.reset_embedder, force=False))
                            if cleared:
                                if measurable and free_vram is not None:
                                    await loop.run_in_executor(
                                        None,
                                        lambda: wait_for_vram_release(
                                            lambda: vram_capacity().get("free"),
                                            before_bytes=free_vram))
                                continue

                    # Nothing idle left to evict.
                    if cannot_measure:
                        # This BOX cannot report free VRAM at all, and every remaining
                        # model is busy (or none loaded): freed what we safely can,
                        # load best-effort (the pre-multi-model behaviour). Correct and
                        # unchanged - refusing here would brick every CPU-only /
                        # GGUF-only / registry-tier box, which can NEVER measure.
                        break
                    if inconclusive:
                        # The probe did not complete, so we do not know. Master reached
                        # this line with free_vram None and took the break above,
                        # loading with NO VRAM CHECK AT ALL - and because a cold driver
                        # init reliably overruns the old 4s cap, that was the FIRST
                        # load after every server start on such a box, silent but for a
                        # logger.debug. Refuse instead: the failure modes are not
                        # symmetric. A wrong permit hands a too-big model to the native
                        # loader, which "can hard-abort the process rather than return
                        # NULL" (gpu_split_shortfall's docstring) - unrecoverable. A
                        # wrong refusal is a 503 the user can retry, and by then the
                        # abandoned probe has usually landed and the driver is warm.
                        # Quotes no figure: we have none we measured (rule 5).
                        #
                        # RARE now. Two ways the probe fails to complete, and both the
                        # long deadline and wait_for_inflight=True above are aimed at
                        # them: (1) a cold init on a fresh process - the deadline waits
                        # it out; (2) a CONCURRENT probe holding the slot (the GUI's
                        # 2500ms /api/stats heartbeat through a cold init) - the join
                        # waits on ITS result instead of taking an instant BUSY. So
                        # reaching here needs the probe to blow the FULL deadline even
                        # after joining - a genuinely stuck/wedged driver, not the
                        # ordinary cold-init or heartbeat-collision cases. Refusing
                        # there is correct: we truly cannot measure, and loading blind
                        # onto a possibly-wedged GPU is the OOM/TDR risk this gate
                        # exists to prevent.
                        raise HTTPException(
                            503, f"Cannot load '{name}': free VRAM could not be "
                            f"measured (the GPU probe did not complete within "
                            f"{discover._GPU_PROBE_CLI_DEADLINE:.0f}s, so the driver "
                            f"may be stuck), and no idle model could be unloaded to "
                            f"make room. Refusing rather than load a model that may "
                            f"not fit. Retry shortly; if this persists, restart the "
                            f"GPU app holding the driver, or unload another model.")
                    # Local eviction exhausted: before giving up, best-effort ask a
                    # sibling localm instance to release ITS VRAM (multi-instance
                    # coordination, see localm.gpu_registry). Off the event loop (it
                    # may make a blocking loopback call). Advisory: any failure falls
                    # through to the 503 below, never a harder failure than baseline.
                    # Bounded and fit-checked (see _attempt_cooperative_unload): each
                    # peer is asked at most once per load attempt, so this loop always
                    # progresses, and a yank that provably could not free enough is
                    # not worth destroying a sibling's models for.
                    # A PIN is a local preference, exactly like the cap above, so
                    # it must not cost a SIBLING instance its models either. If an
                    # unpinned peer WOULD have been evictable, the pin is the only
                    # reason local eviction came up empty - so do not escalate.
                    # Fall through to the backend's own split-aware sizing
                    # (partial offload) instead, which honors the pin locally at
                    # this instance's own expense rather than another's. Unlike
                    # the cap case, VRAM here is genuinely short, so this is a
                    # slower load rather than a free one - still the right trade
                    # against destroying another instance's work.
                    pin_blocked = bool(pinned) and residency.pick_eviction_victim(
                        _engines_lru, _engines, requested=name) is not None
                    if pin_blocked:
                        from localm.debuglog import logger as _dbg
                        _dbg.warning(
                            "pinned_models=%s is the only reason no local model "
                            "could be evicted for %s; NOT asking a peer instance "
                            "to unload - deferring to the backend's own sizing",
                            sorted(pinned), name)
                    cooperated = False if pin_blocked else await loop.run_in_executor(
                        None,
                        lambda: _attempt_cooperative_unload(
                            needed_bytes=vram_required + headroom,
                            free_bytes=free_vram, asked=asked_peers))
                    if cooperated:
                        continue
                    if shortfall and not shares_adaptive:
                        # Aggregate may well be enough - it is specifically the
                        # configured split's per-device share that is short, so name
                        # the device(s), not a generic aggregate message. (Said "or
                        # unmeasurable" until it was shown unreachable: shortfall is
                        # always [] when free is unmeasurable, because list_gpus()
                        # DROPS a device that fails to report rather than emitting
                        # free=None, so the per-device `free is None` skip above and an
                        # unmeasurable aggregate cannot coexist. Also unreachable now
                        # via the inconclusive branch, which empties shortfall.)
                        #
                        # STATIC shares only (pinned ratios, or auto declined into
                        # the equal fallback - a stale configured index, a device
                        # without a free reading): this stays a hard refusal even
                        # though everything below it defers to the backend, because
                        # with static shares apply_gpu_split() divides the model by
                        # a ratio that ignores live per-device free VRAM
                        # (discover.gpu_split_shortfall's own docstring), and the
                        # backend's own sizing (_auto_gpu_layers / _check_vram,
                        # llamacpp/_sizing.py) budgets the split's COMBINED capacity
                        # (_split_free_total_bytes), never any one device's static
                        # share - so this per-device check is still the only gate
                        # that can catch one split device being individually short
                        # while the aggregate fits. Letting a real per-device
                        # shortfall through would trade today's precise, actionable
                        # message for a native worker abort with no such visibility.
                        # Keyed on shares_adaptive, NOT on whether ratios are set in
                        # config: with ratios unset but auto DECLINED, the loader
                        # will apply the same equal fallback the gate just checked,
                        # so this hazard is fully live there (review finding on
                        # this feature's first cut - the config shape alone admitted
                        # exactly that case).
                        detail = "; ".join(
                            f"GPU {d['index']} needs ~{d['needed'] // 1024 ** 2} MB, "
                            f"{d['free'] // 1024 ** 2} MB free" for d in shortfall)
                        raise HTTPException(
                            503, f"Not enough VRAM on the configured split "
                            f"device(s) to load '{name}' ({detail}).")
                    # ADAPTIVE shares (live auto free-VRAM-proportional split): a
                    # non-empty shortfall can only mean the COMBINED estimate is
                    # short (each device's auto share fits its free whenever the
                    # aggregate fits - gpu_split_shortfall computed with the same
                    # auto ratios the loader will pin), so it falls through to
                    # the same defer-to-backend path as the aggregate-only miss
                    # below: the backend's split-aware sizing (#770) is the
                    # accurate judge there, with partial offload available -
                    # exactly the #753 posture the single-GPU path already ships.
                    #
                    # Local + cooperative eviction exhausted, no hard-refusable
                    # (pinned-share) shortfall - what remains is this loop's own
                    # coarse "vram_required = file_size * 1.2" estimate not
                    # being met (combined across an auto split, or single-GPU). That
                    # estimate assumes the WHOLE model lands in VRAM; it has no idea
                    # the backend's own load() already knows how to make a too-big
                    # model fit anyway: GgufBackend's n_gpu_layers_auto (default ON -
                    # _effective_gpu_layers/_auto_gpu_layers/_check_vram in
                    # llamacpp/_sizing.py) sizes how many layers actually fit free
                    # VRAM and puts the rest on system RAM, and HFBackend's
                    # device_map="auto" (hf.py) does the unconditional equivalent -
                    # both already documented as the promise behind a "too-big"
                    # discover.fit_label() badge in the GUI's model browser. Refusing
                    # here, before an engine is ever constructed, broke that promise
                    # for exactly this case (a model that would fit via partial
                    # offload, but not by this loop's whole-model estimate).
                    #
                    # Fall through to a real load attempt instead: _check_vram() is
                    # the accurate, backend-owned final gate - it raises only when
                    # the model genuinely cannot fit even at 0 GPU layers, which the
                    # except handler around new_engine.load() below turns into the
                    # same clean 503 shape this refusal used to be, for every caller.
                    from localm.debuglog import logger as _dbg
                    _dbg.info(
                        "switch_engine: '%s' exceeds the whole-model VRAM estimate "
                        "(need ~%s MB, %s MB free) after eviction - deferring to the "
                        "backend's own load-time sizing instead of refusing",
                        name, vram_required // 1024 ** 2, free_vram // 1024 ** 2)
                    break

                evict_engine = _engines[evict_name]
                free_before = free_vram
                # Defensive re-check right before we commit (approach (a) of the
                # pin-during-unload fix, WITHOUT its victim-semaphore - that would
                # reintroduce the two-switch lock-ordering deadlock below). The LRU
                # scan above already required active_requests==0 and there is no
                # await between it and here, so this cannot currently fire; it is a
                # guard so a future await in the scan cannot silently reopen the
                # window. If it became pinned, abandon this candidate and re-scan.
                if getattr(evict_engine, "active_requests", 0) != 0:
                    continue

                # Detach the victim from the live registry BEFORE the native free,
                # not after (the pin-arrives-during-the-unload-await fix, BUG-9b).
                # Safe to unload without the victim's own semaphore: active_requests
                # == 0 means no request is pinned on it (a request pins its engine
                # for its whole lifetime - AUDIT-CRIT-1), so no decode races the
                # free; taking the victim sem while holding the target sem would risk
                # a two-switch lock-ordering deadlock. But active_requests alone does
                # NOT stop a QUEUED request pinning the victim DURING the unload await
                # (it pins lock-free, after the check passed): while await unload()
                # yields the loop, get_engine's fast path would still see the victim
                # loaded+registered and pin a doomed engine, which then gets freed out
                # from under it and silently auto-reloaded (VRAM over-subscription /
                # a tight-box 500). Removing it from _engines first closes that window
                # - a queued request now misses the fast path and gets a fresh,
                # VRAM-checked reload via switch_engine - and means a concurrent
                # switch_engine(evict_name) rebuilds a fresh engine instead of reusing
                # (and racing load() against) this backend object mid-free. The flag
                # is belt-and-suspenders for any stale reference already resolved.
                # Removal stays pop/guarded (BUG-9a): even pre-await, a concurrent
                # idle/explicit unload could have popped the victim already.
                evict_engine.unloading = True
                _engines.pop(evict_name, None)
                if evict_name in _engines_lru:
                    _engines_lru.remove(evict_name)
                _inference_sems.pop(evict_name, None)
                # If the victim was the active/compat engine, drop those pointers too,
                # so a concurrent get_engine back-compat re-import cannot resurrect the
                # just-detached victim mid-free and nothing keeps serving a being-freed
                # engine as active. switch_engine sets them to the newly-loaded model
                # at the end (or leaves them cleared if the load fails - correct, the
                # old active model is gone). The other unload paths already do this.
                if _active_model_name == evict_name:
                    _active_model_name = None
                if _engine is evict_engine:
                    _engine = None
                    _inference_sem = None
                # See _evicting_names' own docstring: mark THIS name as mid-free so a
                # concurrent switch_engine/get_engine call for it (a queued reload of
                # the very model being evicted) refuses instead of racing a fresh
                # load against this still-running native free. try/finally: every
                # path out of the free below (success or an unexpected exception from
                # evict_engine.unload itself) must clear it, or the name would be
                # permanently unloadable.
                _evicting_names.add(evict_name)
                try:
                    await loop.run_in_executor(None, evict_engine.unload)

                    # Wait for the native VRAM free to land before re-checking, so the
                    # next iteration does not see a stale-low reading and over-evict
                    # (driver-hang guard, AUDIT-MED-11). Only meaningful when measurable.
                    if measurable and free_before is not None:
                        await loop.run_in_executor(
                            None,
                            lambda: wait_for_vram_release(
                                lambda: vram_capacity().get("free"), before_bytes=free_before))
                finally:
                    _evicting_names.discard(evict_name)

        # See _evicting_names' own docstring: *name* itself may be the victim
        # another concurrent switch_engine call just detached and is still
        # natively freeing - a queued reload landing in exactly that window.
        # Refuse rather than construct-and-load a fresh engine that races the
        # still-running free (honest backpressure; the caller may simply
        # retry once the in-flight eviction finishes).
        if name in _evicting_names:
            raise HTTPException(
                503, f"'{name}' is currently being freed by another request; "
                f"retry shortly.")

        if name in _engines:
            new_engine = _engines[name]
        else:
            new_engine = _engine_factory(name)

        cancel = threading.Event()
        # Install the fresh event for EVERY load, preempt or not. An engine object
        # outlives a load (idle-unload keeps it in _engines for lazy reload), so a
        # preempt=True switch that gets superseded leaves its FIRED event on the
        # engine, and nothing else ever clears it. Installing only under preempt
        # meant the next API-routed (preempt=False) load reused that stale SET
        # event, and the backend honours it by aborting the load at once - a
        # permanent spurious 503 "superseded" on every later request for that
        # model (REG-461). Loads of one model are serialized by its own semaphore
        # above, so this cannot clobber a concurrent load's event.
        if hasattr(new_engine, "set_load_cancel"):
            new_engine.set_load_cancel(cancel)
        if preempt:
            # Only an explicit switch REGISTERS the hook globally, so only a newer
            # explicit switch can abort this load; API-routed loads (preempt=False)
            # run to completion, never cancelled by a concurrent different-model load.
            _switch_cancel = cancel
            _switch_loading = name
        try:
            await loop.run_in_executor(None, new_engine.load)
        except ModelLoadCancelled:
            return {"status": "superseded", "model": name, "by": _switch_desired}
        except RuntimeError as exc:
            # The backend's own sizing found the model genuinely cannot fit even
            # with 0 GPU layers (GgufBackend._check_vram - llamacpp/_sizing.py) or
            # its native worker failed outright (GgufBackend._load_native's own
            # wrapper, gguf.py) - both raise a plain, already-informative
            # RuntimeError. Only 2 of the 6 routes that reach switch_engine
            # currently wrap it in their own try/except to get a clean message
            # (the GUI's "load model" button, the coder plugin's model switch);
            # the OpenAI-compatible routes (/v1/chat/completions, /v1/completions,
            # /v1/embeddings, /v1/models/load) do not, so this exact failure used
            # to fall through to the generic "Internal server error" 500 there,
            # discarding the real reason (AGENTS.md rule 5). Converting it to an
            # HTTPException here, once, gives every caller the same clean message
            # the existing VRAM-refusal 503s above already get from Starlette's
            # default HTTPException handling.
            raise HTTPException(503, f"Failed to load '{name}': {exc}") from exc
        finally:
            if preempt and _switch_cancel is cancel:
                _switch_cancel = None
                _switch_loading = None
                
        _engines[name] = new_engine
        # Seed the per-model activity clock the INSTANT this engine becomes
        # resident - not when it first answers a request. Without this,
        # _idle_unload_once's _last_activity_per_model.get(name, _last_activity)
        # fell back to the GLOBAL _last_activity, which holds whatever OTHER
        # model was last used - so a model switched to while the PREVIOUS model
        # was already idle past the TTL looked instantly overdue for eviction,
        # and could be unloaded before it ever served its first response.
        _last_activity_per_model[name] = time.monotonic()
        _engines_lru.append(name)
        _active_model_name = name
        _engine = new_engine
        _inference_sem = sem
        if on_active is not None:
            on_active(name)
        # Cross-install GPU coordination: reflect the newly-active model so a
        # sibling's next VRAM/eviction check sees fresh state. No-op when not
        # registered (see _gpu_registry_sync). Offloaded for the same reason as
        # the heartbeat below: registry file I/O plus, with a non-zero
        # main_gpu_index, a real GPU driver probe - keep it OFF the event loop.
        await loop.run_in_executor(None, _gpu_registry_sync)
        return {"status": "loaded", "model": name,
                **_gpu_placement_fields(new_engine)}


async def get_engine(model_name: str, *, load: bool = True) -> Engine:
    """Resolve the engine for *model_name*, loading it if necessary.

    With ``load=False`` the resolved engine is returned WITHOUT forcing a load -
    for callers like /v1/embeddings whose backend may not need the model resident
    at all (a GGUF backend embeds via the dedicated embedder; AUDIT-MED-13). The
    caller decides whether to load. Registration/resolution (and its 404) still
    apply.
    """
    global _engines, _engines_lru, _active_model_name, _default_model_name, _inference_sems, _engine, _inference_sem

    # Back-compat: if a test or script set _engine directly, import it into the multi-model dicts
    if _engine is not None and _engine.display_name not in _engines:
        _engines[_engine.display_name] = _engine
        _inference_sems[_engine.display_name] = _inference_sem or asyncio.Semaphore(1)
        if _engine.display_name not in _engines_lru:
            _engines_lru.append(_engine.display_name)
        if not _active_model_name:
            _active_model_name = _engine.display_name

    name = (model_name or "").strip()
    if not name or name == "localm":
        name = _active_model_name or _default_model_name
        
    from localm.config import load_registry
    registry = load_registry()
    
    # If no registry is populated, route all requests to the active/loaded engine (classic single-model mode)
    if not registry:
        active = _active_model_name or (_engine.display_name if _engine else None)
        if active:
            name = active
        else:
            name = name or _default_model_name
            if name != _default_model_name and name not in _engines:
                raise HTTPException(503, "No model loaded. Please load a model first.")

    # Only enforce registration check if the registry is not empty
    if registry:
        if name not in registry and name != _default_model_name and name != _active_model_name:
            registered = sorted(registry.keys())
            msg = f"Model '{name}' is not registered."
            if registered:
                msg += f" Registered models in your library: {', '.join(registered)}. Use 'localm pull' to add a new model."
            raise HTTPException(404, msg)

    # A name that resolved to nothing (an empty/"localm" request with no active AND
    # no default model - e.g. `gui --no-model` with a populated registry, or the
    # transient window during an active-model eviction/unload where _active_model_name
    # was just cleared) must be an honest 503, not fall through to switch_engine(None)
    # -> get_model_info(None) -> Path(None) TypeError -> HTTP 500.
    if not name:
        raise HTTPException(503, "No model is loaded and none was specified. "
                            "Load a model first or name one explicitly.")

    # `not unloading`: never hand back (and let the caller pin) an engine that an
    # eviction/unload path is mid-freeing - that is the pin-arrives-during-the-
    # unload-await race. An engine flagged unloading falls through to switch_engine
    # below, which reloads it cleanly under its per-model semaphore (the unloader
    # holds that semaphore for the native free, so the reload serializes AFTER it).
    if name in _engines and _engines[name].loaded and getattr(_engines[name], "unloading", False) is not True:
        if name in _engines_lru:
            _engines_lru.remove(name)
        _engines_lru.append(name)
        _active_model_name = name
        _engine = _engines[name]
        _inference_sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        return _engines[name]

    if not load:
        # Return the engine object WITHOUT loading it: reuse a tracked (possibly
        # unloaded) engine, else build a fresh one via the factory. Does NOT go
        # through switch_engine, so no model is loaded and nothing is evicted.
        return _engines.get(name) or _engine_factory(name)

    res = await switch_engine(name, _engine_factory, preempt=False)
    if res.get("status") == "superseded":
        raise HTTPException(503, f"Model load was superseded by a newer request: {res.get('by')}")

    return _engines[name]


def _add_vram_fields(result: dict, *, before, released, after, before_fresh: bool,
                     before_scope=None) -> None:
    """Add vram_freed/vram_before_bytes/vram_after_bytes to *result* when
    measurable (unchanged from before), plus an honest flag when the reading
    cannot be presented as current fact - rather than asserting a wrong number
    (AGENTS.md rule 5).

    ``before is None`` returns early and adds NOTHING - the benign case (a
    CPU-only box, or the Windows registry tier, which reports total but never
    free) where a completed probe simply has no free reading to give. That is not
    the fault this guards, and must not be dressed up as one: no VRAM telemetry is
    the normal, permanent state there, so saying nothing is the honest answer.

    THREE independent ways this reading can be wrong, and they are not the same bug:

    - NOT FRESH (``before_fresh`` false): the probe timed out or was busy, so the
      'before' value is a stale cached one (PR #693's case; see _vram_free_reading).
    - AFTER UNVERIFIABLE (``released is None``): wait_for_vram_release() could not
      verify the outcome (the 'after' reading went unmeasurable), so ``vram_freed``
      is null rather than a false "VRAM did not drop" - a claim a reading that never
      refreshed cannot support (PR #694's case).
    - NOT DEVICE-SCOPED (``before_scope`` is FREE_SCOPE_PROCESS): the probe was
      perfectly fresh, but on this platform the driver reports only the CALLING
      process's own allocations. Since every GGUF load runs in an isolated worker
      subprocess (backends/gguf.py, #606), the model's VRAM is in another process
      and simply absent from the number - which is why before/after came back
      byte-identical, and vram_freed false, across a load/unload cycle that an OS
      counter showed working perfectly. Measured, see
      dev-notes/vram-cross-process-blindness.md.

    All three are reported through the same flag because they mean the same thing to
    a caller (do not trust this number), but the note says which, so a bug report
    points at the right one."""
    if before is None:
        return
    from localm.discover import FREE_SCOPE_PROCESS
    result.update(vram_freed=released, vram_before_bytes=before, vram_after_bytes=after)
    reasons = []
    if not before_fresh:
        reasons.append(
            "the GPU probe timed out or was busy when this reading was taken, so "
            "it may reflect a stale cached value rather than the current state")
    if released is None:
        reasons.append(
            "the free-VRAM reading went unmeasurable after the unload, so whether "
            "VRAM was actually reclaimed could not be verified")
    if before_scope == FREE_SCOPE_PROCESS:
        reasons.append(
            "this GPU's driver reports only THIS process's own VRAM allocations, "
            "and the model is loaded in a separate worker process, so its memory "
            "is not counted in these figures")
    if reasons:
        result["vram_reading_uncertain"] = True
        result["vram_note"] = (
            "vram_before_bytes/vram_after_bytes/vram_freed may be wrong: "
            + "; ".join(reasons))


async def unload_all_models() -> dict:
    """Release every currently-loaded model from GPU/CPU memory and wait until
    VRAM is actually reclaimed (see ``localm.vram.wait_for_vram_release`` - the
    driver-hang guard: otherwise a media model can load on top of a
    not-yet-freed LLM and exceed total VRAM).

    Extracted from the ``POST /v1/models/unload`` route so it has exactly ONE
    implementation, reused by two callers with two different auth models: the
    owner-scoped ``/v1/models/unload`` route (``MODELS_WRITE``), and the
    coordination-token-gated ``POST /v1/instances/cooperate-unload`` (a
    sibling localm instance asking THIS one to free VRAM - multi-instance GPU
    coordination, see ``localm.gpu_registry``). Behavior is unchanged from the
    original inline implementation."""
    global _active_model_name, _engine, _inference_sem
    loop = asyncio.get_running_loop()
    from localm.vram import (_live_free_vram_bytes, _vram_free_reading,
                             wait_for_vram_release)
    from localm.inference import embedder as _embedder_mod

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    unloaded_models = []
    skipped_in_use = []

    for name in list(_engines.keys()):
        engine = _engines[name]
        if not engine.loaded:
            continue
        # Honor the in-flight-request pin (AUDIT-CRIT-1), like the VRAM-eviction and
        # idle-unload paths: a pinned engine has a request generating (or about to)
        # against it. Unloading it would free VRAM the request immediately reloads
        # (so the reported "freed" total is a lie) and race a use-after-unload; skip
        # it and report it as still in use. Gate on isinstance(int) exactly like
        # _pin/_unpin: a non-int active_requests (a bare test double) is "not pinned".
        active = getattr(engine, "active_requests", 0)
        if isinstance(active, int) and active > 0:
            skipped_in_use.append(name)
            continue
        sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        # Flag BEFORE acquiring the semaphore so no request that arrives after the
        # pin check above can take get_engine's fast path and pin this engine while
        # we free it (the pin-arrives-during-the-unload-await window, BUG-9b); such
        # a request now blocks on the same semaphore and reloads cleanly afterwards.
        # Cleared in finally so the kept-in-_engines engine reloads lazily.
        engine.unloading = True
        try:
            async with sem:
                await loop.run_in_executor(None, engine.unload)
                unloaded_models.append(name)
                if name in _engines_lru:
                    _engines_lru.remove(name)
        finally:
            engine.unloading = False

    # Release the shared embedder too - a separate lifecycle from _engines (see
    # localm.inference.embedder's module docstring): it is loaded independently
    # by RAG/memory/coder-episode callers via get_embedder(), never through
    # switch_engine, so it was previously NEVER freed by "Unload all" even
    # though the GUI reported everything released (only the chat engines'
    # VRAM actually dropped - the embedder's stayed resident).
    #
    # loaded_dim()/active_requests() MUST run in the executor, not directly on
    # this coroutine: get_embedder() can hold embedder._LOCK for the full
    # duration of an IsolatedEmbedder native/subprocess load (up to its load
    # timeout), and both of those accessors block on that same lock. A
    # synchronous call here would freeze the WHOLE event loop - every other
    # request this server is serving - for that entire window, not just this
    # coroutine (confirmed via live reproduction during review, 2026-07-14).
    # Executor-offloading them, like every other blocking call in this
    # function, keeps the wait local to this one coroutine instead.
    embedder_was_loaded = False
    embedder_dim = await loop.run_in_executor(None, _embedder_mod.loaded_dim)
    if embedder_dim is not None:
        # Honor the in-flight-request pin (AUDIT-CRIT-1) for the embedder too,
        # exactly like the chat-engine loop above: a request mid-embed() must
        # not have its embedder (and the isolated worker process it is
        # waiting on) freed out from under it. reset_embedder(force=False)
        # checks active_requests()==0 and clears the embedder in ONE locked
        # step (not two separate executor calls) - a real TOCTOU window used
        # to sit between a standalone active_requests() check and an
        # unconditional reset_embedder(), since IsolatedEmbedder.embed() pins
        # active_requests without taking embedder._LOCK. Skip it and report
        # it alongside the pinned chat engines instead of a lying "unloaded".
        cleared = await loop.run_in_executor(
            None, functools.partial(_embedder_mod.reset_embedder, force=False))
        if cleared:
            embedder_was_loaded = True
        else:
            skipped_in_use.append("embedding model")

    # Update compatibility pointers - but NOT if the active engine was a pinned one
    # we deliberately left loaded (clearing it would strand the in-flight request's
    # active model).
    if _active_model_name not in skipped_in_use:
        _active_model_name = None
        _engine = None
        _inference_sem = None

    released_anything = bool(unloaded_models) or embedder_was_loaded
    if before is not None and released_anything:
        released, after = await loop.run_in_executor(
            None, lambda: wait_for_vram_release(_free, before_bytes=before))
    else:
        released, after = 0, before

    if released_anything:
        status = "unloaded"
    elif skipped_in_use:
        status = "in_use"          # nothing freed: every loaded model is pinned
    else:
        status = "already_unloaded"
    result = {
        "status": status,
        "model": unloaded_models[0] if unloaded_models else "none",
        "unloaded_models": unloaded_models,
        "embedder_unloaded": embedder_was_loaded,
    }
    if skipped_in_use:
        result["skipped_in_use"] = skipped_in_use
    _add_vram_fields(result, before=before, released=released, after=after,
                     before_fresh=before_fresh, before_scope=before_scope)
    # Cross-install GPU coordination: reflect the now-empty/changed state for a
    # sibling's next eviction decision. No-op when not registered. Offloaded:
    # registry file I/O plus, with a non-zero main_gpu_index, a GPU driver probe.
    await loop.run_in_executor(None, _gpu_registry_sync)
    return result


async def _unload_embedder_if_matches(name: str, loop) -> Optional[dict]:
    """If *name* is a registered model whose path matches the currently-loaded
    shared embedder, release it and report the freed VRAM - the targeted-unload
    counterpart to ``unload_all_models``'s embedder release above.

    The embedder is a separate lifecycle from ``_engines`` (see
    ``localm.inference.embedder``'s module docstring): ``unload_one_model``'s
    own ``_engines.get(name)`` lookup can never find it, so without this a
    resident embedding model registered under its own name (the common case:
    a `localm pull`-ed GGUF selected as the embedding model) showed as
    "loaded" on the Models page yet its per-row Unload button was a silent
    no-op. Matched by resolved PATH, not by name/config, so it is correct
    regardless of how ``embedding_model`` was originally resolved (an explicit
    path, a registered name, or a known key) - what matters is which file is
    actually resident. Returns None when *name* is not the embedder, so the
    caller falls back to its normal "already_unloaded" outcome for a genuinely
    untracked/never-loaded chat model."""
    from localm.inference import embedder as _embedder_mod
    # Executor-offloaded, not a direct call: get_embedder() can hold
    # embedder._LOCK for the full duration of an IsolatedEmbedder
    # native/subprocess load, and loaded_path() blocks on that same lock. A
    # synchronous call here would freeze the WHOLE event loop for that window
    # (same hazard as unload_all_models's loaded_dim() call - see its comment).
    emb_path = await loop.run_in_executor(None, _embedder_mod.loaded_path)
    if emb_path is None:
        return None
    from pathlib import Path
    from localm.config import load_registry
    from localm.model_manager import _entry_path
    entry_path = _entry_path(load_registry().get(name))
    if entry_path is None:
        return None
    try:
        if Path(entry_path).resolve() != Path(emb_path).resolve():
            return None
    except OSError:
        return None

    # Honor the in-flight-request pin (AUDIT-CRIT-1): a request mid-embed()
    # must not have its embedder freed out from under it. Report it as still
    # in use instead of a lying "unloaded", matching unload_one_model's own
    # pinned-chat-engine check just below.
    #
    # TWO layers, deliberately, mirroring switch_engine's own chat-engine
    # eviction (its LRU scan's active_requests==0 check, THEN a synchronous
    # "defensive re-check right before we commit" - http_server.py's own
    # comment on that code names it "approach (a) of the pin-during-unload
    # fix"): a cheap active_requests() precheck here avoids paying for the
    # _vram_free_reading() hardware probe below on the common busy case (that
    # probe is NOT executor-offloaded and can block this whole single-
    # threaded event loop for up to discover._GPU_PROBE_DEADLINE seconds - a
    # regression a review caught: folding the busy check ENTIRELY into
    # reset_embedder(force=False) after moving the probe earlier meant a busy
    # embedder paid for the probe every time, where the pre-existing code
    # never reached it). reset_embedder(force=False) is still what actually
    # authorizes the close, atomically re-checking active_requests()==0 under
    # embedder._LOCK in the SAME step as the close (no separate unlocked
    # active_requests() call before an unconditional reset_embedder() - that
    # TOCTOU window is what round 1 of this fix closed, since
    # IsolatedEmbedder.embed() pins active_requests without taking
    # embedder._LOCK) - so the rare case of a pin arriving in the gap between
    # this precheck and the actual close is still caught correctly, just no
    # longer the ONLY thing gating the probe cost. Both calls are executor-
    # offloaded for the same reason as loaded_path() above - each blocks on
    # embedder._LOCK, which get_embedder() can hold for the length of an
    # IsolatedEmbedder load; a synchronous call here would freeze the whole
    # event loop, not just this request.
    embedder_active = await loop.run_in_executor(None, _embedder_mod.active_requests)
    if embedder_active > 0:
        return {"status": "in_use", "model": name, "vram_freed": 0}

    from localm.vram import (_live_free_vram_bytes, _vram_free_reading,
                             wait_for_vram_release)

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    cleared = await loop.run_in_executor(
        None, functools.partial(_embedder_mod.reset_embedder, force=False))
    if not cleared:
        return {"status": "in_use", "model": name, "vram_freed": 0}
    if before is not None:
        released, after = await loop.run_in_executor(
            None, lambda: wait_for_vram_release(_free, before_bytes=before))
    else:
        released, after = 0, before
    result = {"status": "unloaded", "model": name, "was_active": False}
    _add_vram_fields(result, before=before, released=released, after=after,
                     before_fresh=before_fresh, before_scope=before_scope)
    # Offloaded for the same reason as this function's other executor hops above.
    await loop.run_in_executor(None, _gpu_registry_sync)
    return result


async def unload_one_model(name: str) -> dict:
    """Release ONE currently-loaded model from GPU/CPU memory, leaving any
    other loaded models untouched - the targeted counterpart to
    ``unload_all_models()`` (same VRAM-release-wait + gpu-registry-sync
    behavior, just scoped to a single engine). Clears the active-model
    pointers only when *name* was the active model, so unloading a background
    (loaded-but-not-active) model never disturbs the one actually serving
    requests. A *name* that is registered but not currently loaded is a
    no-op success (idempotent, matching unload_all_models()'s "nothing to do"
    case), not an error - callers that need to reject an unknown model name
    outright should check the registry themselves before calling this."""
    global _active_model_name, _engine, _inference_sem
    loop = asyncio.get_running_loop()
    from localm.vram import (_live_free_vram_bytes, _vram_free_reading,
                             wait_for_vram_release)

    engine = _engines.get(name)
    if engine is None or not engine.loaded:
        embedder_result = await _unload_embedder_if_matches(name, loop)
        if embedder_result is not None:
            return embedder_result
        return {"status": "already_unloaded", "model": name}
    # Honor the in-flight-request pin (AUDIT-CRIT-1): an engine a request is
    # generating on must not be unloaded out from under it (it would reload it
    # anyway, making the "freed" report a lie). Report it as in use, not unloaded.
    # isinstance(int) guard matches _pin/_unpin (a bare test double is not pinned).
    active = getattr(engine, "active_requests", 0)
    if isinstance(active, int) and active > 0:
        return {"status": "in_use", "model": name, "vram_freed": 0}

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
    # Flag BEFORE acquiring the semaphore so no request that arrives after the pin
    # check above can fast-path-pin this engine while we free it (the pin-arrives-
    # during-the-unload-await window, BUG-9b); such a request blocks on the same
    # semaphore and reloads cleanly afterwards. Cleared in finally so the
    # kept-in-_engines engine reloads lazily.
    engine.unloading = True
    try:
        async with sem:
            await loop.run_in_executor(None, engine.unload)
            if name in _engines_lru:
                _engines_lru.remove(name)
    finally:
        engine.unloading = False

    was_active = _active_model_name == name
    if was_active:
        _active_model_name = None
        _engine = None
        _inference_sem = None

    if before is not None:
        released, after = await loop.run_in_executor(
            None, lambda: wait_for_vram_release(_free, before_bytes=before))
    else:
        released, after = 0, before

    result = {"status": "unloaded", "model": name, "was_active": was_active}
    _add_vram_fields(result, before=before, released=released, after=after,
                     before_fresh=before_fresh, before_scope=before_scope)
    # Offloaded: registry file I/O plus, with a non-zero main_gpu_index, a GPU
    # driver probe - keep it OFF the event loop (same as the heartbeat).
    await loop.run_in_executor(None, _gpu_registry_sync)
    return result


# Monotonic timestamp of the last inference request, for the optional idle-unload
# loop (config "idle_unload_seconds"). Touched at the start of each inference
# endpoint, like Ollama's keep_alive (measured from the last request).
_last_activity: float = time.monotonic()
_last_activity_per_model: dict[str, float] = {}


def _touch_activity(name: str | None = None) -> None:
    """Record that an inference request just arrived (resets the idle timer)."""
    global _last_activity, _last_activity_per_model
    now = time.monotonic()
    _last_activity = now
    if name:
        _last_activity_per_model[name] = now


def _sanitize_client_context(raw) -> dict:
    """Reduce an untrusted GUI ``client`` payload to a safe, bounded dict for a bug
    report: only known string fields (capped) plus a capped list of console-error
    strings. Anything else is dropped. Returns {} for non-dict / empty input."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for field in ("userAgent", "page", "viewport", "appVersion"):
        val = raw.get(field)
        if isinstance(val, (str, int, float)) and str(val).strip():
            out[field] = str(val)[:500]
    console = raw.get("console")
    if isinstance(console, list):
        errs = [str(e)[:1000] for e in console if isinstance(e, (str, int, float))]
        if errs:
            out["console"] = errs[-40:]
    return out


def _idle_unload_ttl() -> int:
    """Configured idle-unload TTL in seconds (0 = disabled), read live so a
    Settings change applies without a restart. A bad value falls back to 0."""
    try:
        from localm.config import load_config
        return max(0, int(load_config().get("idle_unload_seconds", 0) or 0))
    except (TypeError, ValueError):
        return 0


async def _idle_unload_once(ttl: int) -> bool:
    """One idle check: unload the model if it has been idle for >= ttl seconds.
    Returns True if it unloaded. Does NO sleeping (the loop owns cadence), so the
    decision is unit-testable without waiting.

    The unload runs UNDER the inference semaphore so it can never free the native
    context mid-decode (that crashes the GPU driver), and the idle time is
    re-checked inside the lock so a request that arrived while we waited for the
    lock cancels the unload. The next inference reloads the model lazily."""
    global _active_model_name, _engine, _inference_sem
    if ttl <= 0:
        return False
        
    targets = dict(_engines)
    if _engine is not None and _engine.display_name not in targets:
        targets[_engine.display_name] = _engine
        
    if not targets:
        return False
        
    loop = asyncio.get_running_loop()
    unloaded_any = False
    
    for name, engine in list(targets.items()):
        if engine is None or not engine.loaded:
            continue

        # The _last_activity fallback below is intentional, not a leftover: it is
        # only ever reached for an engine that was assigned to `_engine` directly
        # without going through switch_engine's registration (a test/script
        # setting _engine, or a genuinely single-model/direct-path startup with
        # an empty registry - see switch_engine's own "empty registry" comment).
        # In that mode there is only ONE model, ever, so "the last activity of
        # any request" and "the last activity of THIS model" are the same fact -
        # falling back to it is correct, not a cross-model leak. A model loaded
        # via switch_engine always has its OWN entry from the instant it is
        # registered (seeded there), so in the multi-model case this fallback is
        # never reached at all.
        last_act = _last_activity_per_model.get(name, _last_activity)
        if (time.monotonic() - last_act) < ttl:
            continue
            
        if getattr(engine, "active_requests", 0) > 0:
            continue
            
        sem = _inference_sems.get(name) or _inference_sem or asyncio.Semaphore(1)
        async with sem:
            # Recheck under the lock
            last_act = _last_activity_per_model.get(name, _last_activity)
            if not (engine.loaded and (time.monotonic() - last_act) >= ttl):
                continue
            if getattr(engine, "active_requests", 0) > 0:
                continue
                
            idle_s = int(time.monotonic() - last_act)
            # Flag for the duration of the native free so no request that slips in
            # after the active_requests recheck above can take get_engine's fast
            # path and pin this engine while it is being freed (the pin-arrives-
            # during-the-unload-await window, BUG-9b). Cleared in finally so the
            # kept-in-_engines engine reloads lazily on the next request.
            engine.unloading = True
            try:
                await loop.run_in_executor(None, engine.unload)
            finally:
                engine.unloading = False

            # Keep the (now-unloaded) Engine in _engines so the next request
            # reloads it lazily with its ORIGINAL constructor settings (n_ctx /
            # n_gpu_layers / device / mmproj), and so a direct-path served model
            # (display name not in the registry, unbuildable by the factory) is
            # not lost forever (AUDIT-HIGH-5). Only drop it from the LRU (no VRAM
            # while unloaded); keep its inference semaphore for a concurrent reload.
            if name in _engines_lru:
                _engines_lru.remove(name)

            if _engine is engine:
                _engine = None
            if _active_model_name == name:
                _active_model_name = _engines_lru[-1] if _engines_lru else None
                _engine = _engines[_active_model_name] if _active_model_name else None
                _inference_sem = _inference_sems.get(_active_model_name) if _active_model_name else None
                
            from localm.debuglog import logger as _dbg
            _dbg.info("idle-unload: freed %s after %ds idle (ttl=%ds); it reloads "
                      "on the next request", engine.display_name, idle_s, ttl)
            unloaded_any = True
            # Cross-install GPU coordination: reflect the freed model. No-op when
            # not registered. Offloaded: _gpu_registry_sync does filesystem I/O
            # (and, when a non-zero main_gpu_index is set, a GPU probe via
            # _current_gpu_index) - keep it OFF the event loop.
            await loop.run_in_executor(None, _gpu_registry_sync)

    return unloaded_any


async def _idle_unload_loop() -> None:
    """Free the model from VRAM after `idle_unload_seconds` of no inference.

    Opt-in (default 0 = disabled). Runs as a lifespan background task; the actual
    decision lives in `_idle_unload_once`. A transient error is logged (RULE 5:
    surface, do not swallow) instead of killing the loop."""
    while True:
        ttl = _idle_unload_ttl()
        if ttl <= 0:
            # Disabled: poll occasionally so enabling it at runtime takes effect.
            await asyncio.sleep(30)
            continue
        # Check within the TTL, but not too hot and not too slow.
        await asyncio.sleep(max(5, min(ttl, 30)))
        try:
            await _idle_unload_once(ttl)
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.warning("idle-unload check failed (continuing)", exc_info=True)


async def _gpu_registry_heartbeat_loop() -> None:
    """Keep this instance's cross-install GPU-coordination entry fresh
    (~every 20s), matching the ``_idle_unload_loop`` pattern above. Only
    started when this instance is actually registered for coordination (see
    ``_gpu_coord`` / the lifespan startup below) - a plain test app or an
    ``--isolated`` run never starts this loop at all. A transient failure is
    logged, not fatal (RULE 5): the entry just ages until the next tick, and a
    stale entry is skipped by a peer's own liveness+identity check anyway."""
    while True:
        await asyncio.sleep(20)
        try:
            # Offloaded off the event loop: _gpu_registry_sync does filesystem I/O
            # (registry write) and, when a non-zero main_gpu_index is configured, a
            # GPU driver probe - either could otherwise stall the single loop and
            # freeze the whole WebUI on this 20s tick while the box is idle.
            await asyncio.get_running_loop().run_in_executor(None, _gpu_registry_sync)
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.warning("gpu-registry heartbeat failed (continuing)", exc_info=True)


async def _hang_heartbeat_loop() -> None:
    """Bump _hb_monotonic every _HEARTBEAT_INTERVAL_S so the off-loop watchdog
    thread can tell when the single event loop has stopped making progress (a
    hang), and so _loop_lag_seconds() can report a real scheduling-delay
    figure. The ONLY steady-state cost is one wakeup per interval."""
    global _hb_monotonic
    while True:
        _hb_monotonic = time.monotonic()
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)


def _start_hang_watchdog(threshold: float, trace_path, *, poll: float = 1.0):
    """Start a plain (NON-async) daemon thread that watches the heartbeat. When
    the event loop has not ticked in `threshold`s it is blocked, so dump ALL
    thread stacks to `trace_path` via faulthandler - the only way to see what a
    fully-wedged loop is stuck in, because this thread runs OUTSIDE the loop.
    Polls every `poll`s (tests lower it for speed). Returns (stop_event, thread)
    for teardown; the thread OWNS its file and closes it on exit.

    The trace file is opened LAZILY, only when the first stall is detected, so a
    healthy run (the overwhelming common case, since this is on by default) never
    creates a file at all. Never blocks: it only waits on an Event, subtracts two
    numbers, and appends to a file."""
    import faulthandler
    import traceback

    stop = threading.Event()

    def _run() -> None:
        fh = None
        last_dump = None
        try:
            while not stop.wait(poll):
                lag = time.monotonic() - _hb_monotonic
                if lag < threshold:
                    continue
                now = time.monotonic()
                # Throttle: a long freeze yields a handful of snapshots, not one/sec.
                # `is not None` (not a 0.0 sentinel): time.monotonic() is boot-relative,
                # so a real 0.0 baseline would wrongly suppress the FIRST dump within
                # the first ~30s of uptime.
                if last_dump is not None and now - last_dump < max(30.0, threshold * 3):
                    continue
                last_dump = now
                try:
                    if fh is None:   # lazy: create the file only on a real stall
                        fh = open(trace_path, "a", buffering=1,
                                  encoding="utf-8", errors="backslashreplace")
                    fh.write(
                        f"\n===== LOCALM HANG WATCHDOG: event loop stalled {lag:.1f}s "
                        f"(pid {os.getpid()}, {time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
                    try:
                        faulthandler.dump_traceback(file=fh, all_threads=True)
                    except Exception:
                        # Fallback: pure-Python walk of every thread's frames.
                        for tid, frame in sys._current_frames().items():
                            fh.write(f"\n--- thread {tid} ---\n")
                            fh.write("".join(traceback.format_stack(frame)))
                    fh.flush()
                except Exception:
                    # The watchdog must never crash the process it is diagnosing.
                    pass
        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    t = threading.Thread(target=_run, name="localm-hang-watchdog", daemon=True)
    t.start()
    return stop, t


def _diagnostics_allowed() -> bool:
    """Whether localm may write an AUTOMATIC diagnostic trace right now. True in
    the log/full session modes; in privacy mode ONLY when the user opted into
    keeping diagnostics (config ``keep_diagnostics``). Fail-safe to privacy (no
    trace) when the mode/config cannot be resolved, matching audit.py's default.

    Gates the hang watchdog and the crash-restart breadcrumbs so privacy mode's
    "nothing written automatically" promise holds, while the toggle lets a tester
    keep the diagnostics a bug report needs."""
    try:
        from localm.config import keep_diagnostics_enabled
        if keep_diagnostics_enabled():
            return True
    except Exception:
        pass
    try:
        from localm.audit import SessionMode, effective_mode
        return effective_mode("server") != SessionMode.PRIVACY
    except Exception:
        return False   # fail toward privacy: write no automatic trace


# Optional bearer-token auth - enabled when LOCALM_API_KEY is set.
_bearer_scheme = HTTPBearer(auto_error=False)

# S2: the browser GUI authenticates with an HttpOnly session cookie whose value is
# an OPAQUE session id (localm.sessions), NOT the API key - page JS cannot read it,
# and rolling the key does not invalidate it. Cookie-sourced auth on a state change
# must also carry a CSRF token in this header, an HMAC DERIVED from the session
# (csrf_token_for / _csrf_ok) fetched via GET /api/session, so it is always in
# lockstep with the session and cannot desync (there is NO separate CSRF cookie).
# The Authorization-header path (CLI / SDK / coder) cannot be forged cross-site and
# is therefore CSRF-exempt.
SESSION_COOKIE = "localm_session"
CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Hard cap on any request body, rejected (413) from Content-Length BEFORE the body
# is buffered or parsed, so a large base64 upload cannot be materialized ahead of a
# route's own checks (decode-time OOM DoS on /api/rag/upload + /extract, CWE-400).
# 160 MB fits the largest legitimate upload (100 MB decoded ~= 133 MB base64 + JSON
# wrapper) and rejects larger up front. Read at request time so a test can patch it.
MAX_REQUEST_BODY_BYTES = 160_000_000


class _BodyStreamCapMiddleware:
    """Enforce MAX_REQUEST_BODY_BYTES on the actual bytes received over the
    wire, not the client-supplied Content-Length header. ``Transfer-Encoding:
    chunked`` sends no Content-Length at all, so a plain header check (as a
    ``@app.middleware("http")``/BaseHTTPMiddleware handler would have to do)
    never fires - live-verified: a chunked POST to a CORS-exempt route like
    /v1/chat/completions was fully buffered by FastAPI's own body handling
    (ahead of any auth dependency or pydantic validation) up to ~5.9 GB RSS
    from one unauthenticated connection before this fix (AUD-CHUNKED). A pure
    ASGI middleware class (not the BaseHTTPMiddleware pattern used elsewhere in
    this file) so it wraps the raw ``receive`` callable BEFORE Starlette/
    FastAPI's own body-buffering step ever runs; a BaseHTTPMiddleware handler
    that itself called ``request.body()`` would just reproduce the same
    unbounded read it is trying to bound.

    Once the cap is crossed this does NOT just raise and let the exception
    unwind through FastAPI's own body-parsing (that was tried first): FastAPI
    wraps ANY exception from body reading into a generic 400 "error parsing
    the body" - worse, raising from deep inside receive() surfaces to it as an
    ``ExceptionGroup`` (from the anyio task group `BaseHTTPMiddleware` runs the
    downstream app in), which doesn't match FastAPI's own
    ``except HTTPException: raise`` passthrough, so the 413 never reaches the
    client at all - only a bare TCP reset (confirmed live: uvicorn had unread
    bytes still sitting in the socket's receive buffer when it closed, so the
    OS sent RST instead of completing the response). Instead: tell the inner
    app the body simply ENDED at the cap (bounding what it can ever buffer),
    swallow whatever confused response it tries to send for that truncated
    body, drain and discard the rest of the real stream so the OS can close
    the connection cleanly, and send exactly one authoritative 413 ourselves."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        cl = None
        for name, value in scope.get("headers") or ():
            if name == b"content-length":
                try:
                    cl = int(value)
                except ValueError:
                    cl = None
                break
        if cl is not None and cl > MAX_REQUEST_BODY_BYTES:
            # Fast path: reject BEFORE reading any body bytes off the wire.
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )(scope, receive, send)
            return

        total = 0
        exceeded = False
        real_stream_done = False

        async def _capped_receive():
            nonlocal total, exceeded, real_stream_done
            message = await receive()
            if message["type"] == "http.disconnect":
                real_stream_done = True
                return message
            if message["type"] == "http.request":
                if not message.get("more_body", False):
                    real_stream_done = True
                total += len(message.get("body") or b"")
                if total > MAX_REQUEST_BODY_BYTES:
                    exceeded = True
                    # Tell the inner app the body ends HERE, even though the
                    # real client may still be sending - bounds what it can
                    # ever accumulate instead of leaving it to keep reading a
                    # never-ending stream via this same wrapped receive.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def _suppressing_send(message):
            # The inner app now believes it got a (truncated) body and will
            # try to respond to it - never let that response reach the real
            # client; the 413 below is authoritative once exceeded.
            if not exceeded:
                await send(message)

        try:
            await self._app(scope, _capped_receive, _suppressing_send)
        except Exception:
            if not exceeded:
                raise
            # Suppressed: the inner app choked on a body we deliberately cut
            # short (e.g. invalid JSON at the truncation point) - irrelevant,
            # the request is rejected as too large regardless of what its
            # first MAX_REQUEST_BODY_BYTES happened to contain.

        if exceeded:
            # Drain the rest of the real stream (if the client had not already
            # finished sending it) so the OS does not RST the connection on
            # close over unread bytes still in its receive buffer, which would
            # silently discard the 413 response below. Bytes are discarded
            # immediately, not accumulated, so this cannot reproduce the
            # unbounded-memory bug - but a client that goes silent mid-stream
            # (stops sending, never signals more_body=False, never disconnects)
            # would otherwise leave this loop's `await receive()` blocked
            # forever, trading the memory-exhaustion bug for a connection/task
            # left open indefinitely (AUD-CHUNKED follow-up). Bound BOTH bytes
            # drained AND wall-clock time spent draining; past either ceiling,
            # give up on the graceful drain (a possible RST instead of a clean
            # 413 response is an acceptable trade for not hanging a task).
            drained = 0
            drain_ceiling = MAX_REQUEST_BODY_BYTES * 4
            drain_deadline = time.monotonic() + 30.0
            while not real_stream_done and drained <= drain_ceiling:
                remaining = drain_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if message["type"] == "http.disconnect":
                    real_stream_done = True
                    break
                drained += len(message.get("body") or b"")
                if not message.get("more_body", False):
                    real_stream_done = True
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )(scope, receive, send)


# Scope key under which _DisconnectSignalMiddleware publishes a non-blocking
# "has the client gone?" poll. See the middleware and _generate_full for why an
# endpoint cannot just call request.is_disconnected().
_DISCONNECT_POLL_KEY = "localm.disconnect_poll"


class _DisconnectSignalMiddleware:
    """Publish a working client-disconnect poll for endpoints that need one.

    The four @app.middleware("http") handlers below are BaseHTTPMiddleware, which
    runs the endpoint in a child task fed by a SYNTHETIC receive that never yields
    http.disconnect - so request.is_disconnected() is permanently False for any
    endpoint behind them (confirmed against a real uvicorn client abort). A
    StreamingResponse still learns of a disconnect (Starlette acloses its body
    generator), but a plain non-streaming coroutine does not.

    This is a PURE-ASGI middleware (a BaseHTTPMiddleware here would defeat its own
    purpose) added OUTSIDE that stack, so it keeps the raw ASGI receive - which
    does carry http.disconnect - and stashes a Starlette-style non-blocking peek at
    scope[_DISCONNECT_POLL_KEY]. It never wraps receive/send, so it is transparent
    to every other route; only the non-streaming inference path polls it (see
    _generate_full). WHY here and not fixed globally: converting the auth/origin
    BaseHTTPMiddleware handlers to pure-ASGI is a far larger, riskier change; this
    gives the one path that needs it a correct signal without touching them."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        gone = {"v": False}

        async def poll_disconnected() -> bool:
            # Mirrors Starlette.Request.is_disconnected: peek the raw stream with an
            # immediately-cancelled receive so it never blocks, and latch True once
            # http.disconnect has arrived. Reading the raw receive here is safe: the
            # body is fully read (through the wrapped chain) before the endpoint -
            # and thus this poll - runs, so only http.disconnect remains to consume.
            import anyio
            if gone["v"]:
                return True
            message: dict = {}
            with anyio.CancelScope() as cs:
                cs.cancel()
                message = await receive()
            if message.get("type") == "http.disconnect":
                gone["v"] = True
            return gone["v"]

        scope[_DISCONNECT_POLL_KEY] = poll_disconnected
        await self._app(scope, receive, send)


# SEAMLESS: the session cookie PERSISTS so the user stays signed in across a browser
# or PWA restart (a drop-on-close cookie made the key gate and its "Install
# certificate" step reappear every restart). Browsers clamp lifetime to ~400 days,
# so we ask for that ceiling; escape hatch is /api/session/logout (Settings: blank
# the key and Save).
SESSION_MAX_AGE = 400 * 24 * 3600  # ~400 days (the browser cap)


def _bearer_token(request) -> Optional[str]:
    """Extract a presented bearer token from the raw Authorization header (used
    by the origin/management middleware and the request-aware auth core)."""
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def _request_token(request) -> tuple[Optional[str], str]:
    """Resolve the presented key and where it came from. The Authorization
    header wins (programmatic clients); otherwise the HttpOnly ``localm_session``
    cookie (the browser GUI). Returns ``(token, source)`` with *source* one of
    ``"header"`` / ``"cookie"`` / ``"none"``."""
    header = _bearer_token(request)
    if header:
        return header, "header"
    cookie = (request.cookies.get(SESSION_COOKIE) or "").strip()
    if cookie:
        return cookie, "cookie"
    return None, "none"


def _principal_from_token(token, source):
    """Resolve a presented credential to ``(scopes, key_hash, fs_access)`` or None.

    A ``header`` token is a raw API key -> ``auth.verify()``. A ``cookie`` token is
    now an OPAQUE SESSION ID -> the server-side session store (``localm.sessions``),
    which returns the scope / owning-key / fs-access SNAPSHOT taken at login. So a
    cookie session stays valid across an owner-key roll (the reported bug), and the
    durable key never has to live in the cookie. ``key_hash`` is the sha256 of the
    key that minted the session, so ``principal_id`` over a cookie matches the same
    key presented as a bearer (job ownership parity)."""
    if not token:
        return None
    if source == "cookie":
        from localm import sessions
        rec = sessions.lookup(token)
        if rec is None:
            return None
        held = set(rec.get("scopes", []))
        if scopes.ADMIN not in held:
            # A SCOPED-key session lives only as long as its key: re-validate the
            # owning key against the live keystore every request, so revoking or
            # expiring it cuts the session off (parity with the bearer path's
            # per-request verify()). An owner/ADMIN session is exempt - decoupled
            # from the key VALUE so an owner-key ROLL does not log the owner out.
            from localm.auth import key_hash_live
            if not key_hash_live(rec.get("key_hash")):
                return None
        return (held, rec.get("key_hash"), rec.get("fs_access", "none"))
    from localm.auth import _hash_key, fs_access_for, verify
    held = verify(token)
    if held is None:
        return None
    fs = "host" if scopes.ADMIN in held else fs_access_for(token, "none")
    return held, _hash_key(token), fs


def _csrf_secret(request) -> str:
    """The per-process CSRF secret for this app, created lazily if a standalone
    mount (a bare attach_gui in tests) never ran create_app's setup."""
    st = request.app.state
    sec = getattr(st, "csrf_secret", None)
    if not sec:
        sec = secrets.token_urlsafe(32)
        st.csrf_secret = sec
    return sec


def csrf_token_for(request, sid: str) -> str:
    """The CSRF token for a session id: a deterministic HMAC of the id under the
    per-process secret. Derived from the session, so it is available exactly when
    the session is (delivered to the client via GET /api/session and the shell
    <meta>) and can never desync from it. Empty for an empty sid."""
    if not sid:
        return ""
    return hmac.new(_csrf_secret(request).encode("utf-8"),
                    sid.encode("utf-8"), hashlib.sha256).hexdigest()


def _csrf_ok(request) -> bool:
    """CSRF check for a cookie-authenticated unsafe request: the ``X-CSRF-Token``
    header must equal the token DERIVED from the session cookie (signed with the
    per-process secret). No separate CSRF cookie exists to be cleared or to desync,
    so a valid session always has a usable token. A cross-site page can neither read
    the HttpOnly session cookie nor the token (SOP), and cannot set a non-simple
    header; SameSite=Strict and the same-origin guard remain in force."""
    from localm.auth import ct_equal
    header = request.headers.get(CSRF_HEADER, "")
    sid = (request.cookies.get(SESSION_COOKIE) or "").strip()
    if not header or not sid:
        return False
    # ct_equal, not compare_digest: the header is caller-supplied and latin-1
    # decoded, so a non-ASCII one would raise and turn this 403 into a 500.
    return ct_equal(header, csrf_token_for(request, sid))


def _enforce_request(request: Request, scope: Optional[str]) -> None:
    """Shared auth core (request-aware). Open/dev mode when no key is configured
    anywhere (unless LOCALM_REQUIRE_AUTH forces fail-closed). Otherwise a valid
    key is required - from the Authorization header OR the session cookie - and
    *scope* (None = 'any valid key') must be granted (the owner key implies every
    scope). Cookie-sourced auth on an unsafe method additionally requires a valid
    CSRF token (an HMAC derived from the session)."""
    from localm.auth import any_key_configured, require_auth_enabled
    if not any_key_configured():
        if require_auth_enabled():
            raise HTTPException(
                status_code=401,
                detail="Auth required but no API key configured "
                       "(set one via the launcher or LOCALM_API_KEY)")
        return  # open/dev mode
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    held = prin[0]
    if source == "cookie" and request.method not in _SAFE_METHODS:
        if not _csrf_ok(request):
            raise HTTPException(
                status_code=403,
                detail="Missing or invalid CSRF token for a cookie-"
                       "authenticated state change.")
    if scope is not None and not scopes.grants(held, scope):
        raise HTTPException(status_code=403,
                            detail=f"Key lacks required scope: {scope}")


def _require_auth(request: Request) -> None:
    """Require any valid API key (no specific scope)."""
    _enforce_request(request, None)


def require_scope(scope: str):
    """FastAPI dependency factory: require a key whose scopes grant *scope*.
    Use as ``dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))]``."""
    def dep(request: Request) -> None:
        _enforce_request(request, scope)
    return dep


def caller_scopes(request: Request) -> Optional[set]:
    """The scope set the presented key grants (the owner key -> {ADMIN}), or
    None in open mode / when no valid key is presented. Routes use this to make
    authorisation decisions that depend on *who* the caller is (e.g. only an
    owner/ADMIN principal may mint keys carrying privileged scopes)."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    return prin[0] if prin else None


def principal_id(request: Request) -> Optional[str]:
    """A stable, opaque per-key identity for the CURRENT caller, or None in open
    mode / when no token is presented. It is the same SHA-256 the keystore stores
    (never the plaintext key), so it identifies the key WITHOUT exposing it, and
    is identical whether the key arrives via the Authorization header or the
    session cookie. Used to bind a background job to the key that created it
    (KEY-SCOPE-2), so only that key (or an admin/owner) may stream or cancel it."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    if not token or not token.strip():
        return None
    # Route through _principal_from_token for BOTH sources (AUTH-NETWORK-4):
    # a cookie session for a non-ADMIN key must be re-validated against the
    # live keystore the same way a bearer token is on every request, or a
    # revoked/expired key's still-resident session can keep resolving a key
    # hash here even though the same cookie is already rejected everywhere
    # else auth is enforced.
    prin = _principal_from_token(token, source)
    if prin is not None:
        held, key_hash, _ = prin
        return key_hash
    return None


def memory_principal(request: Request) -> Optional[str]:
    """The identity used to NAMESPACE this caller's chat memory. The owner (an
    ADMIN-scoped key or the owner session) collapses to the shared "owner"
    namespace (returns None -> memory.principal_of maps None to "owner"), so the
    owner's saved memories are not stranded in a per-key-hash namespace that a
    key rotation would orphan (AUDIT-MED-14).

    This is deliberately NOT principal_id: principal_id must keep returning the
    key hash so a background job stays bound to the key that created it
    (KEY-SCOPE-2). Only the memory principal collapses ADMIN/owner to "owner"; a
    non-owner scoped key keeps its own hash namespace here too."""
    from localm import scopes
    held = caller_scopes(request)
    if held is not None and scopes.ADMIN in held:
        return None
    return principal_id(request)


def job_owner_ok(request: Request, job_owner: Optional[str]) -> bool:
    """Whether the caller may stream/cancel a job created by *job_owner*. A job
    with NO recorded owner (created in open mode) is unrestricted; an admin/owner
    key may reach any job; otherwise the caller's principal must match the
    creator's. Pairs with principal_id() stamped at job creation."""
    if job_owner is None:
        return True
    held = caller_scopes(request)
    if held is not None and scopes.ADMIN in held:
        return True
    return principal_id(request) == job_owner


def require_owner(resolve):
    """FastAPI dependency factory: promote a per-route ownership check into a
    Depends()-injectable gate, the same pattern require_scope already uses for
    scope checks - so a new per-owner route cannot omit the check by
    construction (design-audit LM-DA-020, reaffirming LM-DA-SEC-06).

    *resolve* is itself an ordinary FastAPI dependency - its own path/query
    params (e.g. ``job_id``, ``name``) are auto-injected the same as an
    endpoint function's - that returns ``(resource, owner, not_found_detail)``:
    *resource* is the object the route needs (or None if it does not exist),
    *owner* is its recorded creator (None = unrestricted, see job_owner_ok),
    and *not_found_detail* is the 404 message to raise. The SAME 404 is raised
    whether the resource is missing or the caller does not own it - never
    distinguished, so a foreign key cannot even confirm the resource exists
    (KEY-SCOPE-2). On success the gate returns *resource*, so a route can
    declare ``thing = Depends(require_owner(resolve))`` and receive it
    directly. Use as ``dependencies=[Depends(require_owner(resolve))]`` when
    the route does not need the resource itself."""
    def dep(request: Request, resolved=Depends(resolve)):
        resource, owner, not_found_detail = resolved
        if resource is None or not job_owner_ok(request, owner):
            raise HTTPException(404, not_found_detail)
        return resource
    return dep


def effective_fs_access(request: Request) -> str:
    """The caller's effective reach into the SERVER HOST filesystem: "host" (the
    whole disk), "shared" (confined to owner-designated shared roots), or "none"
    (no host FS - device upload only).

    Open mode (loopback owner) and the owner/ADMIN key always resolve to "host";
    any other valid key uses its stored fs_access level (default "none" for a
    legacy key); no valid key -> "none". Filesystem reach is a per-credential dial
    kept deliberately INDEPENDENT of ownership, so an owner can pair one of their
    own devices with a lower-reach key."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return "host"                       # open/dev mode = loopback owner
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        return "none"                       # keys configured, none/invalid presented
    held, _key_hash, fs = prin
    if scopes.ADMIN in held:
        return "host"                       # owner key / owner session
    return fs                               # bearer key or session fs-access snapshot


def require_fs_host(request: Request) -> None:
    """FastAPI dependency: require a caller with FULL host filesystem access
    (owner / open mode / a key explicitly granted fs_access=host). Gates the host
    file/folder browser so a merely config-reading key can no longer enumerate the
    server's disk."""
    _enforce_request(request, None)         # a valid key (or open mode) first
    if effective_fs_access(request) != "host":
        raise HTTPException(
            status_code=403,
            detail="This key does not have host filesystem access")


# Surface mounting (H6 phase 5: on-demand GUI on a running instance).

def mount_gui_surface(app) -> bool:
    """Add the GUI surface (its /api routes + the SPA static mount) to a running
    ``api``-mode app, in place. Idempotent: returns False if a GUI is already
    mounted (a ``full`` instance, or a second call), True if it mounted now.

    Safe at runtime because ``attach_gui`` only appends routes + a ``/`` catch-all
    mount and sets ``app.state`` services - it adds NO middleware (Starlette reads
    ``app.router.routes`` per request, so appended routes take effect immediately;
    only new middleware would need a stack rebuild). The engine + inference
    semaphore are this instance's own (it already loaded the model for /v1), so no
    second model load happens; ``switch_model`` swaps the shared ``_engine`` under
    ``_inference_sem`` exactly as the GUI launcher does."""
    global _engine
    if getattr(app.state, "gui_mounted", False):
        return False

    scheme = getattr(app.state, "instance_scheme", "http")
    port = getattr(app.state, "instance_port", None)
    if not port:
        # advertise() sets instance_port before uvicorn accepts connections, so a
        # real request can never reach here without it; guard anyway so a manual
        # app build fails loudly instead of dialling "http://127.0.0.1:None/v1".
        raise HTTPException(500, "Instance not fully started (no bind port); "
                            "cannot mount the GUI surface yet.")
    self_url = f"{scheme}://127.0.0.1:{port}/v1"

    def active_model() -> str:
        return _engine.display_name if _engine is not None else ""

    def _build_engine(name: str) -> Engine:
        from localm.config import load_registry
        from localm.model_manager import get_model_info, get_model_mmproj
        info = get_model_info(name)
        if info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = info
        # VIS-1: a GUI/registry switch must not drop vision. Carry the model's
        # mmproj (registry-recorded, else a sibling projector next to the GGUF)
        # into the new Engine like the CLI --mmproj flag, else switching silently
        # loses image support.
        mmproj = get_model_mmproj(name)
        return Engine(
            str(m_path),
            display_name=name if name in load_registry() else m_hint,
            mmproj_path=mmproj,
        )

    async def switch_model(name: str) -> dict:
        # Preemptive switch: a newer selection aborts an in-flight load rather
        # than waiting for the abandoned model to finish (see switch_engine).
        return await switch_engine(name, _build_engine)

    from localm.plugins.gui.web import attach_gui
    # Claim the mount BEFORE attaching so a re-entrant/concurrent call cannot
    # double-register the GUI routes; roll the flag back if attach fails. (Today
    # this runs fully synchronously in the request handler, so nothing interleaves;
    # this just makes the invariant explicit.)
    app.state.gui_mounted = True
    try:
        manager = attach_gui(
            app, self_url=self_url, switch_model=switch_model, active_model=active_model)
    except Exception:
        app.state.gui_mounted = False
        raise
    # attach_gui re-affirms app.state.gui_mounted; reflect the surface change in
    # discovery so /whoami and the registry report this is now a full instance.
    app.state.coder_sessions = manager
    app.state.instance_mode = "full"
    app.openapi_schema = None   # force the schema to include the new routes
    try:
        from localm import instances
        from localm.config import home_dir
        instances.set_mode(home_dir(), getattr(app.state, "instance_id", ""), "full")
    except Exception as e:
        # The mount already succeeded; this is best-effort registry sync (so
        # discovery advertises "full" not "api"), not fatal - but now visible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("registry mode not updated to full: %s", e)
    return True


def _dbg_swallow(msg: str, *, level: str = "debug") -> None:
    """Log a swallowed best-effort failure at *level* (with the current exception's
    traceback) without ever raising. The nested-guard pattern already used at the
    update-watchdog site, factored out for the shutdown/restart teardown chain: a
    swallow stays discoverable (rule 5) yet can never itself break the stop/restart
    (the logging call is guarded too)."""
    try:
        from localm.debuglog import logger as _dbg
        getattr(_dbg, level, _dbg.debug)(msg, exc_info=True)
    except Exception:
        pass


def _do_shutdown(*, instance_id: Optional[str] = None) -> None:
    """SRV-4: the actual stop sequence. Unload the model FIRST so the native
    context is freed cleanly (a hard exit while it is loaded segfaults during
    teardown), clear the crash marker so this intentional stop is not reported as
    a crash, then exit the process so the stop is guaranteed (Ctrl+C sometimes
    does nothing). Separated from the route so it can be tested without exiting.

    *instance_id* (app.state.instance_id, set by instances.advertise()) scopes
    the crash-marker clear to THIS instance only - see bugreport.py's
    per-instance-scoping note; omitting it falls back to the legacy shared
    marker name rather than silently skipping the clear."""
    # Unload all engines in the multi-model dictionary
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            # Best-effort clean teardown before exit; a failed unload must not
            # block the stop, but log it so a segfault-on-exit has a breadcrumb.
            _dbg_swallow("engine unload during shutdown failed (non-fatal)")
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            _dbg_swallow("engine unload during shutdown failed (non-fatal)")
    # Also release the shared embedder - a separate lifecycle from _engines (see
    # localm.inference.embedder's module docstring), so a full stop actually
    # frees ALL resident VRAM, not just the chat engines. Same swallow-but-log
    # pattern as the engine unload above: shutdown must complete regardless,
    # but a failure stays discoverable (rule 5).
    try:
        from localm.inference import embedder as _embedder_mod
        # One lock-free call, deliberately NOT active_requests()/reset_embedder():
        # both take the embedder's load lock, which get_embedder() holds for the
        # FULL duration of an embedding-model load - so a stop issued during a
        # load blocked on the guard itself and hung this shutdown, never reaching
        # the worker teardown. release_for_exit() makes the whole decision without
        # that lock: it terminates a busy worker outright (the pinned request
        # cannot be served either way once we exit) and closes an idle one
        # politely. It must run: os._exit below bypasses atexit, and
        # multiprocessing's daemon-child reclamation IS an atexit hook, so a
        # skipped release leaves the worker orphaned with its model resident in
        # VRAM - defeating this path's whole purpose of freeing ALL of it
        # (REG-650).
        _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during shutdown failed (non-fatal)")
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard(instance_id=instance_id)
    except Exception:
        # If the crash marker is NOT cleared, this intentional stop is reported as
        # a crash on the next boot - a false "it crashed". Log it (WARNING) so that
        # misattribution is discoverable instead of silent.
        _dbg_swallow("could not disarm crash guard on shutdown; next boot may "
                     "misreport this intentional stop as a crash", level="warning")
    import os
    os._exit(0)


def _request_shutdown(delay: float = 0.25, *,
                      instance_id: Optional[str] = None) -> None:
    """Run _do_shutdown shortly after returning, so the 200 response flushes to
    the client before the process exits. *instance_id* is forwarded unchanged -
    see _do_shutdown's docstring."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_shutdown(instance_id=instance_id)

    threading.Thread(target=_run, daemon=True).start()


def _restart_argv(port: Optional[int] = None) -> list:
    """The command line to re-launch this server. Always ``python -m localm <args>``
    - the canonical entry the codebase uses - so a restart works regardless of how
    the server was originally started (a console-script .exe, ``-m``, or a script
    path, any of which can make ``sys.argv[0]`` un-re-runnable by the interpreter).

    *port* (the port this instance is ACTUALLY bound to) is appended as an explicit
    ``-p``, so the new process comes back on the same port instead of re-running
    pick_port() and picking a different one. Without it, an instance that was
    auto-bumped off a busy default (started with no -p while another localm held
    8642, so pick_port() gave it 8643) re-execs with no port token at all, calls
    pick_port(None), finds 8642 free again now that the other instance is gone,
    and silently moves - stranding the user's open GUI tab on a dead port, and
    making the post-update watchdog poll the old port until it times out and
    auto-rolls back a perfectly healthy build (REG-605).

    Appending is safe against a user-supplied -p: click takes the LAST occurrence
    of an option, and *port* is the port that value already resolved to, so the
    two agree. Only serve/gui reach a restart, and both accept -p; a caller with
    no known port (a bare create_app() that never advertised) passes None and gets
    the untouched command line."""
    import sys
    argv = [sys.executable, "-m", "localm", *sys.argv[1:]]
    if port:
        argv += ["-p", str(port)]
    return argv


def _do_restart(*, update_watchdog: Optional[dict] = None,
                port: Optional[int] = None,
                instance_id: Optional[str] = None) -> None:
    """R18: restart this server IN PLACE. Unload the model FIRST (clean native
    teardown, like _do_shutdown - a hard re-exec while it is loaded can segfault),
    clear the crash marker so this intentional restart is not reported as a crash,
    then re-exec the same command line so the server comes back on the same port.
    os.execv replaces the process image and does not return on success. Separated
    from the route so it can be tested without actually re-execing.

    *instance_id* (app.state.instance_id, set by instances.advertise()) scopes
    the crash-marker clear to THIS instance only - see _do_shutdown/bugreport.py's
    per-instance-scoping note. The re-exec'd process re-advertises and gets a
    fresh instance_id of its own, so no persistence across the restart is needed.

    *port* is the port this instance is actually bound to (app.state.instance_port,
    set by advertise()); it is pinned into the re-exec command line so "comes back
    on the same port" is TRUE rather than merely intended - see _restart_argv.

    *update_watchdog*, when given (only by the post-update restart path - see
    routes/admin.py's /api/update/apply), is a
    ``{host, port, scheme, expect_version}`` dict describing this instance. A
    DETACHED health-check watchdog is spawned right before the re-exec (LM-DA-011);
    it polls the restarted instance's own /whoami and auto-rolls back if the
    expected version never comes up healthy within its timeout. The plain
    "restart the server" button (/v1/server/restart) calls this with no
    update_watchdog, so it is entirely unaffected.

    NEW-CRASH-NOTICE-USELESS item C: capture free VRAM BEFORE the teardown
    below so the wait after it (once every native free has been issued) can
    confirm the releases actually landed before re-exec spawns a fresh worker
    into them - see that wait's own comment further down for the full
    rationale. Best-effort: an unmeasurable/wedged probe must not block a
    restart the user asked for - vram_capacity() is itself deadline-bounded."""
    free_before = None
    try:
        from localm.discover import vram_capacity
        free_before = vram_capacity().get("free")
    except Exception:
        _dbg_swallow("free-VRAM read before restart failed (non-fatal)")

    # Unload all engines in the multi-model dictionary
    # had_engines asks "is anything ACTUALLY loaded" (worth waiting on), not
    # merely "is the dict non-empty": unload_all_models/idle-unload both KEEP a
    # now-unloaded engine's entry in _engines so a later request reloads it
    # lazily (see their own docstrings), so a dict-non-emptiness check would
    # make EVERY restart on a server that ever idle-unloaded a model pay the
    # wait's full timeout for nothing - the exact "no delay in the common
    # case" claim below would be false. getattr defaults to False so a test
    # double that does not define .loaded (never holding real VRAM) is
    # correctly treated as nothing-to-wait-for.
    had_engines = any(getattr(e, "loaded", False) for e in _engines.values())
    if not had_engines and _engine is not None and _engine not in _engines.values():
        had_engines = bool(getattr(_engine, "loaded", False))
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            # Best-effort clean teardown before re-exec; a failed unload must not
            # block the restart, but log it (a hard re-exec while loaded can
            # segfault, so a breadcrumb helps if that happens).
            _dbg_swallow("engine unload during restart failed (non-fatal)")
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            _dbg_swallow("engine unload during restart failed (non-fatal)")
    # Also release the shared embedder - a separate lifecycle from _engines (see
    # localm.inference.embedder's module docstring), so a restart actually
    # frees ALL resident VRAM before re-exec, not just the chat engines. Same
    # swallow-but-log pattern as the engine unload above: restart must proceed
    # regardless, but a failure stays discoverable (rule 5).
    released_embedder = False
    try:
        from localm.inference import embedder as _embedder_mod
        # Lock-free release - see the matching comment in _do_shutdown above for
        # the full rationale. os.execv is the same case as os._exit: it replaces
        # this process image but does NOT touch the separate worker child, and
        # bypasses atexit, so without this the old worker survives the restart
        # holding VRAM while the restarted server spawns a second one (REG-650).
        released_embedder = _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during restart failed (non-fatal)")

    # Wait for the frees above to actually land before re-exec. The re-exec'd
    # process spawns a brand-new GGUF worker that constructs a fresh
    # llama_context on startup (plugins/gui/cli.py's preload thread) with no
    # idea a restart just freed anything - unlike switch_engine's in-process
    # model swap, which already waits here (its own wait_for_vram_release call,
    # AUDIT-MED-11) before constructing the replacement, this path used to go
    # straight from unload to os.execv with no wait at all. If the GPU driver
    # has not finished reclaiming the just-freed VRAM by the time the fresh
    # worker allocates, that construction races the still-reclaiming driver.
    # NEW-CRASH-NOTICE-USELESS item C: a real 2026-07-26 crash matched exactly
    # this signature (a fresh llama_context dying mid-construction right after
    # a restart, log going dark, empty native trace). Attempts to force the
    # SAME race live (20+ restart cycles, full-VRAM model, idle and under
    # deliberate concurrent GPU contention) did not reproduce a failure on
    # this box/driver - VRAM reclaim measured near-instant here, and the
    # driver's support components were upgraded a few hours before testing, a
    # plausible reason the window is narrower now than on 2026-07-26. It did
    # not need to be observed to be worth closing: switch_engine already
    # treats the identical native free as unsafe to skip this wait for.
    # Skipped when nothing was actually unloaded (a model-less restart), so
    # the common case pays no delay.
    if (had_engines or released_embedder) and free_before is not None:
        try:
            from localm.discover import vram_capacity
            from localm.vram import wait_for_vram_release
            wait_for_vram_release(lambda: vram_capacity().get("free"),
                                  before_bytes=free_before)
        except Exception:
            _dbg_swallow("VRAM-release wait before restart failed (non-fatal)")

    try:
        from localm import bugreport
        bugreport.disarm_crash_guard(instance_id=instance_id)
    except Exception:
        # Same misattribution hazard as _do_shutdown: an uncleared crash marker
        # makes this intentional restart look like a crash next boot. Log it.
        _dbg_swallow("could not disarm crash guard on restart; next boot may "
                     "misreport this intentional restart as a crash", level="warning")

    try:
        global _audit
        if _audit is not None and hasattr(_audit, "close"):
            _audit.close()
    except Exception:
        _dbg_swallow("audit log close during restart failed (non-fatal)")

    try:
        from localm.debuglog import dump_ring_buffer, flush_log_handlers, recent_activity
        # Privacy mode opts out of ALL automatic disk traces, so skip the
        # crash-recovery breadcrumb dumps (ring buffer + pre_restart.log): they are
        # session-derived INFO breadcrumbs written without the user asking. The
        # keep_diagnostics toggle overrides that (a tester who wants a report has
        # opted in); _diagnostics_allowed() folds both in and fails toward privacy
        # (skip) when the mode/config cannot be resolved.
        if _diagnostics_allowed():
            dump_ring_buffer()
            # Also write a clear text log for the bug reporter to ingest directly,
            # in case the JSON buffer fails to load back into memory.
            from localm.config import home_dir
            pre_log = home_dir() / "logs" / "pre_restart.log"
            pre_log.parent.mkdir(parents=True, exist_ok=True)
            pre_log.write_text("\n".join(recent_activity()), encoding="utf-8")
        # Flush all log handlers before os.execv so no buffered lines are lost
        # (Task 1: log durability / save-bug). This flushes already-open handlers;
        # it creates no new trace file, so it is safe in privacy mode too.
        flush_log_handlers()
    except Exception:
        # Best-effort crash-recovery breadcrumbs (ring buffer + pre_restart.log)
        # and the handler flush; a failure here must not block the restart, but it
        # means the next boot has fewer diagnostics, so log rather than swallow.
        _dbg_swallow("pre-restart breadcrumb dump / log flush failed (non-fatal)")

    import os
    import sys

    # NEW-G: no inheritable fd (the debug-log FileHandler, the uvicorn listening
    # socket) must survive into the re-exec'd image, where the freshly loaded
    # ggml/llama runtime warns "Failed to close child file descriptors at 3/4".
    # os.execv does NOT close fds; marking them non-inheritable drops them at exec
    # (POSIX O_CLOEXEC / Windows handle inheritance). Best-effort per fd; stdin/
    # stdout/stderr (0-2) are left inheritable so the new process keeps the console.
    try:
        max_fd = 4096
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if isinstance(soft, int) and 0 < soft < max_fd:
                max_fd = soft
        except Exception:
            pass  # no resource module (Windows) - the 4096 default is plenty
        for fd in range(3, max_fd):
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass  # not an open fd
    except Exception:
        pass  # never let fd hygiene block the restart

    if update_watchdog:
        # Spawned as the LAST step before execv, deliberately: the watchdog's own
        # timeout clock starts when its process begins executing, so spawning as
        # late as possible keeps that clock closest to the actual restart moment.
        try:
            from localm import updater
            updater.spawn_health_watchdog(
                host=update_watchdog["host"], port=update_watchdog["port"],
                scheme=update_watchdog.get("scheme", "http"),
                expect_version=update_watchdog["expect_version"])
        except Exception as e:
            # spawn_health_watchdog() already never raises; this is
            # belt-and-suspenders so even a malformed dict here can never block
            # the restart itself (we do not hide problems - log it, but a broken
            # watchdog must not make updates worse than having none at all).
            try:
                from localm.debuglog import logger as _dbg
                _dbg.warning("update watchdog not started: %s", e)
            except Exception:
                pass

    os.execv(sys.executable, _restart_argv(port))


def _request_restart(delay: float = 0.25, *, update_watchdog: Optional[dict] = None,
                     port: Optional[int] = None,
                     instance_id: Optional[str] = None) -> None:
    """Run _do_restart shortly after returning, so the 200 response flushes to the
    client before the process re-execs (mirrors _request_shutdown). *update_watchdog*,
    *port*, and *instance_id* are forwarded to _do_restart unchanged - see its
    docstring."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_restart(update_watchdog=update_watchdog, port=port,
                   instance_id=instance_id)

    threading.Thread(target=_run, daemon=True).start()


def create_app(engine: Optional[Engine], *, api_landing: bool = False) -> FastAPI:
    global _engine, _inference_sem, _engines, _engines_lru, _default_model_name, _active_model_name, _inference_sems, _last_activity_per_model, _audit
    
    _engines.clear()
    _engines_lru.clear()
    _inference_sems.clear()
    _last_activity_per_model.clear()
    
    if engine is not None:
        _engines[engine.display_name] = engine
        _engines_lru.append(engine.display_name)
        _default_model_name = engine.display_name
        _active_model_name = engine.display_name
        _engine = engine
        _inference_sem = asyncio.Semaphore(1)
        _inference_sems[engine.display_name] = _inference_sem
        _last_activity_per_model[engine.display_name] = time.monotonic()
    else:
        _default_model_name = None
        _active_model_name = None
        _engine = None
        _inference_sem = None

    # Session-persistence mode for this server (privacy -> no traces). One audit
    # log / transcript covers the server lifetime; GUI + API chat traffic flows
    # through /v1/chat/completions and lands here. _audit is published as a
    # module global (unlike _mode/_transcript, kept local and closure-shared
    # with the nested handlers below) because _do_restart is a separate
    # top-level function with no closure access into this frame - without
    # `global _audit` here, its cleanup could never reach the real object.
    from localm.audit import effective_mode, make_audit_log, make_transcript
    _mode = effective_mode("server")
    _audit = make_audit_log(_mode, label="server")
    _transcript = make_transcript(_mode, label="server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem, _inference_sems, _active_model_name, _server_loop
        # Publish the running loop so off-loop worker threads (the jobs runner) can
        # route a shared-engine unload back onto it via run_coroutine_threadsafe -
        # see unload_one_model and the _server_loop comment above.
        _server_loop = asyncio.get_running_loop()
        if _active_model_name:
            _inference_sem = asyncio.Semaphore(1)
            _inference_sems[_active_model_name] = _inference_sem
        # Prune expired browser sessions once at startup so an install that rarely
        # mints new sessions does not accumulate stale rows (create() only prunes
        # opportunistically). Best-effort: a sweep failure must never block startup.
        try:
            from localm import sessions as _sessions
            _sessions.sweep()
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.debug("session sweep at startup failed (non-fatal)", exc_info=True)
        # SRV-3: route an uncaught asyncio task exception through the bug reporter
        # instead of a silent "Task exception was never retrieved". Skipped under
        # pytest so the test runner keeps its own loop handling.
        if "pytest" not in sys.modules:
            try:
                from localm import bugreport
                bugreport.install_asyncio_handler(asyncio.get_running_loop())
            except Exception:
                # The very mechanism meant to surface silent asyncio-task failures
                # must not itself fail silently; log it (mirrors the session-sweep
                # guard just above) so its absence is discoverable.
                from localm.debuglog import logger as _dbg
                _dbg.debug("asyncio exception handler not installed (non-fatal)",
                           exc_info=True)
        # Plugins register() before this loop existed, so loop-dependent plugin
        # work (the jobs scheduler) is queued on the manager; run it now the loop
        # is up - without this no scheduled job fired on a stock start (memory-audit
        # 2026-07-02, critical C2). attach_engine runs after create_app, so the
        # manager resolves at lifespan time.
        _pm = getattr(app.state, "plugin_manager", None)
        if _pm is not None:
            _pm.run_startup_callbacks()
        # Optional idle-unload background task (config "idle_unload_seconds"); it
        # is a cheap no-op while disabled. Cancelled on shutdown so it never
        # outlives the app.
        idle_task = asyncio.create_task(_idle_unload_loop())

        # Hang-capture watchdog: a 1s async heartbeat + an OFF-loop daemon thread
        # that dumps all stacks to a file when the loop stops ticking - the only
        # in-process way to see what a fully-wedged loop is stuck in. On by default
        # in the log/full session modes (the trace file is lazy, created only on a
        # real stall), so an intermittent freeze is captured with zero setup. In
        # PRIVACY mode it stays OFF (no automatic trace) UNLESS the user opted into
        # keeping diagnostics (_diagnostics_allowed) or explicitly forced it on
        # (LOCALM_HANG_WATCHDOG=1). LOCALM_HANG_WATCHDOG=0 opts out entirely. Skipped
        # under pytest so no thread lingers. Best-effort: a startup failure must
        # never block serving.
        hb_task = None
        hang_stop = hang_thread = None
        from localm.debuglog import (
            hang_watchdog_active as _hw_active,
            hang_watchdog_verbose as _hw_verbose,
            hang_watchdog_threshold as _hw_secs,
            hang_trace_path as _hw_path,
        )
        if (_hw_active() and (_hw_verbose() or _diagnostics_allowed())
                and "pytest" not in sys.modules):
            try:
                hb_task = asyncio.create_task(_hang_heartbeat_loop())
                hang_stop, hang_thread = _start_hang_watchdog(_hw_secs(), _hw_path())
                if _hw_verbose():
                    # Explicit opt-in extras: asyncio debug logs any single callback
                    # that hogs the loop past the threshold (names the culprit at a
                    # lower cost than a full stall). Adds per-callback overhead, so
                    # it is NOT part of the default-on path.
                    loop = asyncio.get_running_loop()
                    loop.set_debug(True)
                    loop.slow_callback_duration = 0.5
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("hang watchdog startup failed (continuing): %s", e)

        # Executor thread-pool saturation watch (dev-notes/decisions-2026-07-30-
        # release-gate.md, Q2, the "make exhaustion detectable, not silent" half
        # of the thread-pool-exhaustion fix): a separate off-loop daemon thread,
        # always on (unlike the hang watchdog it sits beside, it logs only pool
        # names and integer counts - no paths, no chat content - so it carries
        # none of the privacy-mode considerations that gate the stack-trace
        # capture above). Skipped under pytest so no thread lingers, matching
        # the hang watchdog's own guard. Best-effort: a startup failure must
        # never block serving.
        sat_stop = sat_thread = None
        if "pytest" not in sys.modules:
            try:
                from localm.inference._executor_health import start_executor_saturation_watch
                sat_stop, sat_thread = start_executor_saturation_watch(
                    asyncio.get_running_loop())
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("executor saturation watch startup failed "
                          "(continuing): %s", e)

        # Cross-install GPU/VRAM coordination (see localm.gpu_registry): register
        # this instance in the machine-wide registry, but ONLY for a real,
        # non-isolated, advertise()'d server (instance_id + port/scheme are set by
        # advertise() before uvicorn accepts connections; a bare create_app() test
        # app or --isolated run never sets instance_id, so this is a no-op that
        # never touches the shared directory). Best-effort: a failure must never
        # block startup (RULE 5: logged, not silenced).
        global _gpu_coord
        gpu_task = None
        _instance_id = getattr(app.state, "instance_id", None)
        _isolated = getattr(app.state, "instance_isolated", False)
        if _instance_id and not _isolated:
            try:
                _gpu_coord = {
                    "instance_id": _instance_id,
                    "port": getattr(app.state, "instance_port", None),
                    "host": getattr(app.state, "bind_host", None) or "127.0.0.1",
                    "scheme": getattr(app.state, "instance_scheme", None) or "http",
                    "token": secrets.token_urlsafe(32),
                }
                app.state.gpu_coordination_token = _gpu_coord["token"]
                _gpu_registry_sync()
                gpu_task = asyncio.create_task(_gpu_registry_heartbeat_loop())
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("gpu-registry startup failed (continuing without "
                          "cross-instance GPU coordination): %s", e)
                _gpu_coord = None

        try:
            yield
        finally:
            idle_task.cancel()
            try:
                await idle_task
            except asyncio.CancelledError:
                pass
            if hb_task is not None:
                hb_task.cancel()
                try:
                    await hb_task
                except asyncio.CancelledError:
                    pass
            if hang_stop is not None:
                # Signal the watchdog thread to stop; it closes its own trace file
                # (if it ever opened one) in its finally.
                hang_stop.set()
                if hang_thread is not None:
                    hang_thread.join(timeout=2)
            if sat_stop is not None:
                sat_stop.set()
                if sat_thread is not None:
                    sat_thread.join(timeout=2)
            if gpu_task is not None:
                gpu_task.cancel()
                try:
                    await gpu_task
                except asyncio.CancelledError:
                    pass
            if _gpu_coord is not None:
                # Best-effort: a crash just leaves the entry to be aged out by
                # a peer's own pid+identity liveness check (same philosophy as
                # instances.py's own registry cleanup).
                try:
                    from localm import gpu_registry
                    gpu_registry.remove_entry(
                        gpu_registry.entry_path(gpu_registry.registry_dir(),
                                                _gpu_coord["instance_id"]))
                except Exception as e:
                    from localm.debuglog import logger as _dbg
                    _dbg.debug("gpu-registry cleanup on shutdown failed: %s", e)
                _gpu_coord = None
            # The loop is stopping - stop advertising it so a late off-loop caller
            # falls back to the safe "no loop" path instead of a dead loop reference
            # (_server_loop is already declared global at the top of lifespan).
            _server_loop = None
            _audit.close()

    app = FastAPI(
        title="localm inference server",
        version="0.1.3",
        lifespan=lifespan,
    )

    # SRV-2: one backstop so an unexpected error in ANY route returns a consistent
    # JSON 500 and is logged, instead of leaking a traceback or bare body. This
    # standardises the response shape and logging so a failing request is a clean
    # 500, never a crash or info leak. (A native fault - a C-extension segfault -
    # cannot be caught in-process; those are prevented at the source, e.g. voice
    # audio is validated before the native path, and surfaced via the crash marker.)
    @app.exception_handler(Exception)
    async def _unhandled_error(request, exc):  # noqa: ANN001 - framework signature
        from localm.debuglog import logger as _dbg
        _dbg.exception("unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error"})

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):  # noqa: ANN001 - framework signature
        # A 422 body must stay serializable. pydantic records the offending value
        # under `input`; when a client sends a NON-FINITE number (NaN / Infinity),
        # Starlette's JSONResponse serializes the error with allow_nan=False and
        # CRASHES into a 500 - so a bad numeric param turned a clean 422 into an
        # unhandled 500 (live-confirmed: `top_k: NaN`, `seed: NaN`, and any float
        # field once allow_inf_nan=False rejects it). Replace non-finite floats in
        # the error detail so the 422 always renders. Same shape as FastAPI's
        # default handler for every other (finite) validation error.
        import math

        from fastapi.encoders import jsonable_encoder

        def _finite_safe(v):
            if isinstance(v, float) and not math.isfinite(v):
                return repr(v)      # "nan" / "inf" / "-inf"
            if isinstance(v, dict):
                return {k: _finite_safe(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_finite_safe(x) for x in v]
            return v

        safe = _finite_safe(jsonable_encoder(exc.errors()))

        def _field(err) -> str:
            # Drop the "body"/"query" container so the user sees the name they
            # actually typed ("max_tokens"), not pydantic's full path.
            parts = [str(p) for p in (err.get("loc") or ())
                     if p not in ("body", "query", "path", "header")]
            return ".".join(parts) or "request"

        def _one(err) -> str:
            name = _field(err)
            msg = (err.get("msg") or "is invalid").strip()
            # pydantic phrases these as "Input should be X", which reads as
            # "max_tokens input should be X" once the field name is prepended.
            # "max_tokens must be X" is the same information, in English.
            if msg.lower().startswith("input should be "):
                msg = "must be " + msg[len("input should be "):]
            elif msg[:1].isupper():
                msg = msg[0].lower() + msg[1:]
            # On a MISSING field pydantic reports the whole request body as
            # `input`, so echoing it back is noise, not evidence.
            got = ("" if err.get("type") == "missing" or "input" not in err
                   else f" (got {err.get('input')!r})")
            # max_tokens=0 is the one worth explaining rather than just reporting:
            # it looks like a legitimate "no limit", and the reason it is refused
            # (0 collides with the engine's internal unlimited sentinel, so it
            # would silently become an unbounded generation) is not guessable from
            # "greater than or equal to 1". Reported live by the maintainer as a
            # raw pydantic dump with no idea what to do about it.
            if name == "max_tokens" and err.get("input") in (0, "0"):
                return ("max_tokens must be 1 or more - 0 is not 'no limit'. "
                        "Omit max_tokens entirely to use the model's default")
            return f"{name} {msg}{got}"

        # `detail` is the human sentence, because every client in this repo does
        # `data.detail || r.statusText` and stringifying pydantic's error LIST
        # there is what produced the unreadable dump users were shown. The
        # structured form is preserved verbatim under `errors` for anything that
        # wants to parse it - nothing is lost, it just stops being the thing a
        # person reads.
        try:
            summary = "; ".join(_one(e) for e in (safe or []))
        except Exception:   # never let error FORMATTING turn a 422 into a 500
            summary = ""
        return JSONResponse(
            status_code=422,
            content={"detail": summary or "Request validation failed",
                     "errors": safe})

    # Chat-pipeline hooks: plugins register inlet/stream/outlet transforms that
    # run on every /v1/chat/completions turn. Created here so it exists before
    # plugins load (attach_engine, below) and stays reachable as
    # request.app.state.chat_pipeline. A pipeline with no hooks is a no-op.
    app.state.chat_pipeline = ChatPipeline()

    # Per-process "shell token" (H5): in open mode the management routes require
    # this token, which the loopback GUI shell injects into the SPA (web.py
    # _gui_index). It gates the no-Origin local-client path that bearer auth /
    # the Origin guard alone do not cover. Per-process so it dies on restart;
    # never persisted.
    app.state.shell_token = secrets.token_urlsafe(32)

    # Per-process CSRF secret. The CSRF token is a deterministic HMAC of the session
    # id (below), so it is present exactly when the session is and CANNOT desync
    # (the old design used a SEPARATE readable cookie a client reset could clear
    # while the HttpOnly session survived, 403-ing every write). Per-process so it
    # dies on restart (client re-fetches from /api/session); never persisted.
    app.state.csrf_secret = secrets.token_urlsafe(32)

    # api-mode landing: a bare `localm serve` has no GUI shell, so GET / would
    # 404. Redirect it to the auto-generated API docs. Only on the api path so
    # it never collides with the GUI's own "/" handler + StaticFiles catch-all.
    if api_landing:
        @app.get("/", include_in_schema=False)
        async def _api_root() -> RedirectResponse:
            return RedirectResponse(url="/docs", status_code=307)

    # Debug mode: log every request with timing to the debug log file
    from localm.debuglog import debug_enabled, logger as _dbg
    if debug_enabled():
        @app.middleware("http")
        async def _log_requests(request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            # loop_lag = real scheduling delay (_loop_lag_seconds, see the
            # comment above _hb_monotonic) - ~0 on a healthy server, and only
            # positive when a preceding event-loop stall pushed the last
            # heartbeat tick late. This is NOT time-since-last-tick, which
            # saws 0..1s even when nothing is wrong (#955/#950). Resolution
            # limit: a stall shorter than _HEARTBEAT_INTERVAL_S also reads 0.0
            # (see _loop_lag_seconds' docstring) - 0.0 means "no stall LONGER
            # than the interval", not "no stall at all".
            _dbg.debug(
                "%s %s -> %d (%.0f ms, loop_lag=%.2fs)",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
                _loop_lag_seconds(),
            )
            return response

    # Loopback-only debug endpoint: every thread's stack + the asyncio task list,
    # for diagnosing a hang/slowdown from the SAME machine on demand. 404'd off
    # loopback. NOTE: served ON the event loop, so it answers only while the loop
    # is alive (a partial stall, a task backlog). A FULLY wedged loop cannot
    # respond here at all - that case is captured by the off-loop watchdog file
    # (LOCALM_HANG_WATCHDOG); this endpoint complements it.
    #
    # THREE gates, because the obvious one is not enough (CodeQL 97):
    #   1. Depends(require_fs_host) - meaningful in PROTECTED mode (a key's
    #      fs_access dial), but in DEFAULT KEYLESS mode effective_fs_access
    #      returns "host" for EVERY caller, so on its own it was a tautology and
    #      this was one of only two fully unauthenticated reads on the server.
    #   2. the open-mode shell-token gate, via _SHELL_TOKEN_GETS below - that is
    #      what makes gate 1 non-vacuous in keyless mode.
    #   3. the bind_host loopback check below. Deliberately on app.state.bind_host
    #      (what the server actually BOUND to) and never request.client.host:
    #      behind portmux the request peer is always 127.0.0.1, so the peer
    #      address cannot distinguish a loopback client from a LAN one.
    # Frame text is path-scrubbed on the way out (see below) - gating it is not a
    # licence to keep emitting the install layout to whoever holds the token.
    @app.get("/debug/stacks", include_in_schema=False,
             dependencies=[Depends(require_fs_host)])
    async def _debug_stacks(request: Request):
        host = getattr(request.app.state, "bind_host", "127.0.0.1")
        if not _is_loopback_host(host):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        import traceback

        from localm.pathscrub import path_scrubber
        # traceback.format_stack emits absolute paths ('File "<install>/localm/
        # inference/http_server.py", line N') which name the install dir and, on
        # a per-user install, the OS account. Redact the DIRECTORY only: the file
        # name, line number, function and source line all survive, so this stays
        # a usable hang diagnosis (rule 5 - scrub, do not mute). Bound once
        # rather than per string: a dump is hundreds of frames and each prefix
        # resolve is a filesystem call.
        scrub = path_scrubber()
        threads = {str(tid): [scrub(line) for line in traceback.format_stack(frame)]
                   for tid, frame in sys._current_frames().items()}
        tasks = []
        try:
            for task in asyncio.all_tasks():
                tasks.append({
                    "name": task.get_name(),
                    "done": task.done(),
                    "stack": [scrub(str(f)) for f in task.get_stack(limit=20)],
                })
        except RuntimeError:
            pass   # no running loop (should not happen inside an async handler)
        # Thread-pool saturation numbers (dev-notes/decisions-2026-07-30-
        # release-gate.md, Q2's "make exhaustion detectable" half) - live
        # here too, not just in the periodic warning log, so a diagnosis in
        # progress does not have to wait for the threshold to trip a line.
        try:
            from localm.inference._executor_health import executors_snapshot
            executors = executors_snapshot(asyncio.get_running_loop())
        except Exception:
            executors = {}
        return {"pid": os.getpid(),
                "loop_lag_s": round(_loop_lag_seconds(), 2),
                "threads": threads, "tasks": tasks, "executors": executors}

    # CORS: localhost-only by default. A wildcard here would let ANY website
    # the user visits call this API from browser JS and read the responses
    # (drive-by GPU use, response exfiltration, /v1/models/unload abuse).
    # Override with config "cors_origins": ["https://app.example"] or "*".
    from localm.config import load_config
    cors_cfg = load_config().get("cors_origins")
    cors_kwargs: dict
    if cors_cfg == "*":
        cors_kwargs = {"allow_origins": ["*"]}
    elif isinstance(cors_cfg, list) and cors_cfg:
        cors_kwargs = {"allow_origins": cors_cfg}
    else:
        cors_kwargs = {
            "allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        }
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["*"],
        allow_headers=["*"],
        **cors_kwargs,
    )

    # CSRF / drive-by guard. The default CORS policy admits ANY localhost:PORT
    # origin, so without this a malicious local web page (a dev server, an npm
    # postinstall server) could drive state-changing endpoints from the user's
    # browser - mint a key, flip require_auth, install a plugin, load/unload the
    # model, browse the filesystem, read files via a plugin route like /api/rag,
    # or drive the coder via /api/coder - even keyless (open-mode scope collapse).
    # So every unsafe-method request must be same-origin (or a configured CORS
    # origin), EXCEPT the OpenAI-compatible inference API, left cross-origin
    # callable for local apps. Allowlist-by-default means a new plugin route is
    # protected the moment it is added. Non-browser clients (CLI / SDK) send no
    # Origin; "cors_origins": "*" opts out entirely.
    _UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    _CROSS_ORIGIN_OK = (
        "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
        # Surface management (phase 5 on-demand GUI mount) is driven by a local
        # process (the attaching `localm gui`), not the browser shell: no Origin,
        # no shell_token. The route does its OWN strict auth (this instance's
        # attach token, or an owner API key) - that, not the same-origin gate, is
        # the real credential, so it is exempt. A cross-origin page still cannot
        # set Authorization without a secret it cannot read, so no CSRF surface.
        "/v1/surfaces/",
        # Multi-instance GPU coordination (localm.gpu_registry): a SIBLING localm
        # instance calls this loopback-only, like surface-management above - no
        # Origin, no shell_token (different process). Its own coordination_token
        # (never the API key/shell token) is the real credential, checked in the
        # route, so the same-origin gate is exempt for the same reason.
        "/v1/instances/",
    )
    _cors_allowlist = cors_cfg if isinstance(cors_cfg, list) else []
    _cors_wildcard = cors_cfg == "*"

    # LM-PT-002 (CWE-200): a short list of UNAUTHENTICATED GETs that disclose host
    # detail and, unlike the /api,/v1 metadata reads below, have NO route-level
    # auth to fall back on. The default CORS policy hands an ACAO to any
    # http(s)://localhost:PORT origin, so without an explicit refusal a drive-by
    # local page could read them cross-origin: /whoami leaks root_dir (an absolute
    # path -> the OS username) on a loopback bind, and /debug/stacks leaks thread
    # stacks. They sit OUTSIDE the /api,/v1 metadata-GET gate, so they are refused
    # here instead - cross-origin, in EVERY mode (they are unauthenticated in
    # protected mode too, so an open-mode-only refusal would miss them).
    _CROSS_ORIGIN_GET_REFUSED = ("/whoami", "/debug/stacks")

    # Same refusal, matched by PREFIX rather than exact path. /api/fs/* is the
    # host filesystem browser: it enumerates the user's disk, which is host
    # detail of exactly the kind above, and it is a GET, so it is exempt from
    # both the CSRF gate (unsafe methods only) and the open-mode shell-token
    # gate. The generic /api,/v1 metadata-GET gate below does cover it, but only
    # inside the `not any_key_configured()` branch - so in PROTECTED mode a
    # cookie-authenticated cross-origin GET would execute. SameSite=strict does
    # not close it either: "site" ignores port, so any other page on a loopback
    # port is same-site, which is the precise actor the comment above at the
    # _cross_origin_refused definition names. Refused in EVERY mode instead.
    _CROSS_ORIGIN_GET_REFUSED_PREFIXES = ("/api/fs/",)
    # Sensitive GETs that sit OUTSIDE the /api,/v1 prefixes the open-mode
    # shell-token gate below keys on, and so were never covered by it (CodeQL
    # 97). /debug/stacks' own Depends(require_fs_host) is a TAUTOLOGY in keyless
    # mode - effective_fs_access returns "host" for everyone when no key is
    # configured - so without this list it was reachable with no credential at
    # all. Listing it here routes it through the same shell-token + cross-origin
    # check as a management read, which is what makes its fs-host gate mean
    # something. /whoami is deliberately NOT here: it is the endpoint the GUI
    # shell calls to discover whether it needs a key at all, so requiring the
    # token to read it would be circular. Its disclosure is handled by the
    # cross-origin refusal above and is a separate, narrower surface.
    # NOTE: enforced only on a LOOPBACK bind - see the comment at token_gated_get
    # below for why answering 403 off loopback would open a new oracle.
    _SHELL_TOKEN_GETS = ("/debug/stacks",)

    def _cross_origin_refused(request) -> bool:
        """True when this request carries an Origin header that is neither
        same-origin nor CORS-allow-listed. Shared by the CSRF check (unsafe
        methods) and the open-mode shell-token gate (AUD-CORSTOKEN): the default
        CORS policy lets any http(s)://localhost:PORT / 127.0.0.1:PORT origin
        READ a matching response, so a hostile local page can steal the shell
        token from a plain cross-origin ``GET /`` and replay it - token
        possession alone does not prove the caller IS the loopback GUI shell.
        "cors_origins": "*" opts OUT of this specific check (AUD-CORSWILD), same
        as it already did for the CSRF check; it does not waive the shell-token
        requirement itself."""
        if _cors_wildcard:
            return False
        origin = request.headers.get("origin")
        if not origin:
            return False
        allowlisted = origin in _cors_allowlist
        host = request.headers.get("host", "")
        same_origin = origin.split("://", 1)[-1] == host
        return not (same_origin or allowlisted)

    @app.middleware("http")
    async def _origin_guard(request, call_next):
        _path = request.url.path
        # Cross-origin refusal (every mode): every state-changing method (CSRF),
        # plus the sensitive GETs in _CROSS_ORIGIN_GET_REFUSED (exact) and
        # _CROSS_ORIGIN_GET_REFUSED_PREFIXES (prefix) - host-detail disclosure,
        # LM-PT-002. All are subject to the same same-origin / CORS-allowlist
        # check.
        if ((request.method in _UNSAFE_METHODS
             or (request.method == "GET"
                 and (_path in _CROSS_ORIGIN_GET_REFUSED
                      or _path.startswith(_CROSS_ORIGIN_GET_REFUSED_PREFIXES))))
                and not _path.startswith(_CROSS_ORIGIN_OK)):
            if _cross_origin_refused(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request refused "
                             "(only same-origin requests or a configured "
                             "'cors_origins' may use this endpoint)."},
                )
            # Open-mode management gate. With no key configured, management
            # routes still require the per-process shell token (injected into the
            # loopback GUI shell), so a no-Origin local client (curl, a script)
            # can no longer mint a key, flip config, install a plugin, load a
            # model, or browse the filesystem unauthenticated. Protected mode (a
            # key exists) is bearer-auth'd on the route. The token is required even
            # for an allowlisted CORS origin: an Origin header is forgeable, so
            # it is not a management credential - a configured external origin must
            # use an API key for state changes.
        is_unsafe = request.method in _UNSAFE_METHODS
        # A _SHELL_TOKEN_GETS path is token-gated only where it is actually
        # SERVED. /debug/stacks 404s off a loopback bind (deliberately hidden),
        # and returning 403 there instead would tell an unauthenticated NETWORK
        # caller that the endpoint exists - a brand new existence oracle opened
        # in the middle of closing a disclosure, since an unknown path under
        # /debug/ 404s. So off loopback, fall through to the handler's own 404.
        # The handler does the loopback check before computing anything, so
        # nothing is spent serving it.
        token_gated_get = (
            request.method == "GET"
            and request.url.path in _SHELL_TOKEN_GETS
            and _is_loopback_host(
                getattr(request.app.state, "bind_host", "127.0.0.1")))
        is_metadata_get = token_gated_get or (
            request.method == "GET"
            and (request.url.path.startswith("/api/")
                 or request.url.path.startswith("/v1/"))
            and request.url.path != "/api/session"
            and not request.url.path.startswith("/v1/models")
        )
        if (is_unsafe or is_metadata_get) and not request.url.path.startswith(_CROSS_ORIGIN_OK):
            from localm.auth import (any_key_configured, ct_equal,
                                     require_auth_enabled)
            if not any_key_configured() and not require_auth_enabled():
                token = getattr(request.app.state, "shell_token", None)
                presented = _bearer_token(request)
                token_ok = ct_equal(presented, token)
                # AUD-CORSTOKEN: an unsafe-method request already passed the
                # same-origin check above (or is exempt as _CROSS_ORIGIN_OK,
                # which never reaches here); a metadata GET never went through
                # that block at all, so it must pass the identical check here -
                # otherwise a token stolen via CORS (the default policy trusts
                # every localhost:PORT origin to READ a response) is directly
                # replayable cross-origin against every /api/*, /v1/* read.
                cross_origin = is_metadata_get and _cross_origin_refused(request)
                if not token_ok or cross_origin:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Open-mode management requires the "
                                 "localm GUI shell on this machine, or an API key "
                                 "(run 'localm key generate')."},
                    )
        return await call_next(request)

    # Security response headers (R41 defense-in-depth). The user-content render
    # path is already XSS-safe via DOMPurify (see dev-notes/SECURITY-xss-render-
    # review-2026-06-23.md); the gap it found was no Content-Security-Policy
    # backstop on the GUI shell. nosniff is enforced everywhere (blocks MIME-sniff
    # into executable HTML). The CSP ships REPORT-ONLY: it never blocks (so it
    # cannot break the inline shell-token script, TTS CDN/HF fetches, the sandboxed
    # artifact iframe, workers) but documents the intended policy and surfaces
    # violations before a later flip to enforcing (which needs a nonce on the
    # inline shell script).
    _CSP_REPORT_ONLY = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://huggingface.co https://*.hf.co "
        "https://cdn.jsdelivr.net; "
        "worker-src 'self' blob:; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault(
            "Content-Security-Policy-Report-Only", _CSP_REPORT_ONLY)
        return resp

    # API-surface disclosure guard. FastAPI's built-in docs (/docs, /redoc,
    # /openapi.json) enumerate every route + schema. Fine on a loopback bind, but
    # on a NETWORK bind it hands an unauthenticated remote a full attack-surface
    # map (every endpoint stays scope-gated, so no access is granted, but it is
    # needless disclosure). Serve docs only on a loopback bind, keyed on the
    # CONFIGURED bind host (never the peer - portmux makes the peer always look
    # loopback). 404 not 403, so they simply do not exist off-loopback. bind_host
    # unset in tests / standalone mount -> default loopback -> docs stay available.
    _DOCS_PATHS = frozenset(
        {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"})

    @app.middleware("http")
    async def _docs_loopback_only(request, call_next):
        if request.url.path in _DOCS_PATHS:
            host = getattr(request.app.state, "bind_host", "127.0.0.1")
            if not _is_loopback_host(host):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return await call_next(request)

    # Outside every BaseHTTPMiddleware handler above (none of which touch the
    # body), so it sees the raw ASGI receive() for its body-size accounting. The
    # _DisconnectSignalMiddleware added right after passes receive() through
    # untouched, so this still gets the raw stream.
    app.add_middleware(_BodyStreamCapMiddleware)
    # Added LAST (== outermost) so its disconnect poll is bound to the raw receive,
    # OUTSIDE the BaseHTTPMiddleware handlers that otherwise mask http.disconnect
    # from the non-streaming inference path (see the class + _generate_full).
    app.add_middleware(_DisconnectSignalMiddleware)

    # Route groups (extracted to localm/inference/routes/*.py).
    # The engine + inference semaphore are module globals read live by the route
    # modules (via `import localm.inference.http_server as _hs`), so a model swap
    # that reassigns them is seen there. Only these session-scoped objects are
    # create_app locals, so they travel to the route groups on ctx. Registration
    # order is irrelevant: FastAPI matches exact path templates by method, and no
    # group has a same-method literal-vs-param path collision.
    from types import SimpleNamespace
    ctx = SimpleNamespace(audit=_audit, transcript=_transcript, mode=_mode)
    from localm.inference.routes import models as _routes_models
    _routes_models.register(app, ctx)
    from localm.inference.routes import system as _routes_system
    _routes_system.register(app, ctx)
    from localm.inference.routes import session as _routes_session
    _routes_session.register(app, ctx)
    from localm.inference.routes import config as _routes_config
    _routes_config.register(app, ctx)
    from localm.inference.routes import keys as _routes_keys
    _routes_keys.register(app, ctx)
    from localm.inference.routes import gpu as _routes_gpu
    _routes_gpu.register(app, ctx)
    from localm.inference.routes import admin as _routes_admin
    _routes_admin.register(app, ctx)
    from localm.inference.routes import chat as _routes_chat
    _routes_chat.register(app, ctx)

    # Plugin engine: load enabled plugins + management API.
    # Wrapped so a plugin-engine failure can never stop the server starting.
    try:
        from localm.plugins.engine import attach_engine
        attach_engine(app, _engine)
    except Exception as e:
        # Server must still start, but make the loss visible: WARNING (not a buried
        # debug line) plus a sentinel so plugin_manager being unset is diagnosable.
        from localm.debuglog import logger as _dbg
        _dbg.warning("plugins unavailable: %s", e)
        _dbg.exception("plugin engine attach failed")
        app.state.plugin_engine_error = str(e)

    return app


def _engine_finish_reason(engine) -> str:
    """Why the last generation ended - "stop" unless the backend reported a
    real string (mocks and minimal engines without the attribute count as stop)."""
    fr = getattr(engine, "last_finish_reason", "stop")
    return fr if isinstance(fr, str) else "stop"


def _ttft_ms(gen_start: float, first_token_at: Optional[float]) -> Optional[float]:
    """Time to first token in milliseconds, or None if nothing was generated."""
    if first_token_at is None:
        return None
    return round((first_token_at - gen_start) * 1000, 1)


def _decode_elapsed(first_token_at: Optional[float], gen_end: float) -> Optional[float]:
    """Wall time spent DECODING: first token -> end of generation. None if nothing
    was generated. This deliberately EXCLUDES model load + prompt prefill (the span
    before the first token), which is reported on its own as ttft_ms - so a cold
    start's multi-second load is never charged against the generation rate."""
    if first_token_at is None:
        return None
    return gen_end - first_token_at


# A floor on plausible per-token decode time, used to reject an implausible decode
# window rather than report a nonsensical rate (see _tokens_per_sec). Verified live
# on real hardware (RX 6900 XT, qwen2.5-0.5b-instruct-q4_k_m) that this floor is
# necessary, not theoretical: under concurrent GPU load from unrelated processes,
# a real HTTP request measured decode_elapsed as low as ~0.2ms for 19-29 tokens,
# reporting 54,786 and 137,701 tok/s. first_token_at is a SINGLE sample - if the
# GPU scheduler delays token 1 (contended) then delivers the rest in an
# uncontended burst once its turn comes, the measured decode window collapses
# toward zero even though every individual timestamp is real. 1ms/token (a 1000
# tok/s ceiling) is deliberately generous: single-stream autoregressive decode is
# memory-bandwidth-bound (reading the quantized weights at least once per token),
# so even a future GPU at multi-TB/s bandwidth plus this architecture's per-token
# IPC marshaling (the GGUF backend relays each token through a subprocess queue)
# is not expected to sustain a single request past this ceiling. Below it, no
# number is more honest than a number that cannot physically be true.
_MIN_SEC_PER_TOKEN = 0.001


def _tokens_per_sec(completion_tokens: int, decode_elapsed: Optional[float]) -> Optional[float]:
    """Decode throughput = generated tokens over the DECODE window only (see
    _decode_elapsed), NOT over total wall time. Folding the model-load/prefill time
    into this rate made the first call after a load report a value ~100x too low
    (e.g. 0.6 tok/s on a warm-64 tok/s GPU), which read as a silent CPU fallback.
    Matches the `localm bench` convention (cli/models.py): "tok/s measures pure
    generation after the first token". None when unmeasurable: a non-positive
    window, fewer than two tokens (one token has no decode interval to time), or a
    window so short it implies a physically implausible rate (see
    _MIN_SEC_PER_TOKEN) - a burst-arrival artifact under GPU contention, not a
    real decode speed."""
    if (decode_elapsed is None or decode_elapsed <= 0 or completion_tokens < 2
            or decode_elapsed < completion_tokens * _MIN_SEC_PER_TOKEN):
        return None
    return round(completion_tokens / decode_elapsed, 2)


def _last_user_text(messages: list) -> str:
    """Text of the most recent user message (for the audit trail)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in (content or [])
                            if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _messages_prompt_text(messages: list) -> str:
    """Flatten chat messages to plain text for prompt token counting (text parts
    of multimodal content included; non-text parts ignored)."""
    return " ".join(
        m.get("content") if isinstance(m.get("content"), str)
        else " ".join(p.get("text", "") for p in (m.get("content") or [])
                      if p.get("type") == "text")
        for m in messages
    )


def _audit_exchange(audit, transcript, messages: list, reply: str) -> None:
    """Record one chat exchange (log/full modes; no-op log in privacy)."""
    if audit is None:
        return
    try:
        user_text = _last_user_text(messages)
        audit.user(user_text)
        audit.llm(reply)
        if transcript is not None:
            transcript.exchange(user_text, reply)
    except Exception as e:
        # In log/full mode a failed write silently drops the record; surface it
        # so the gap is discoverable instead of invisible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("audit/transcript write failed: %s; this exchange was not recorded", e)
        pass  # auditing must never break serving


def _reason_sse(content: str, reasoning: str,
                model_id: str, chunk_id: str, ts: int) -> list:
    """SSE ``data:`` lines for a (content, reasoning) split (H4). Reasoning is
    emitted before content (it precedes the answer); empty parts produce
    nothing, so an ordinary content-only token yields exactly one chunk."""
    from localm.inference.protocol import ChatChunk, ChoiceDelta, StreamChoice
    out = []
    for field, value in (("reasoning_content", reasoning), ("content", content)):
        if not value:
            continue
        chunk = ChatChunk(
            id=chunk_id, created=ts, model=model_id,
            choices=[StreamChoice(delta=ChoiceDelta(**{field: value}))],
        )
        out.append(f"data: {chunk.model_dump_json()}\n\n")
    return out


def _pin(engine) -> None:
    """Mark *engine* as in-use for the current request, the instant the request
    takes ownership of it - call this SYNCHRONOUSLY right after get_engine, with
    no await in between, so the event loop cannot interleave an eviction before
    the pin lands. A pinned engine (active_requests > 0) is skipped by VRAM
    eviction, closing the window where a concurrent model load would unload an
    engine out from under an in-flight request (AUDIT-CRIT-1)."""
    if isinstance(getattr(engine, "active_requests", None), int):
        engine.active_requests += 1


def _unpin(engine) -> None:
    """Release the request pin taken by _pin. Balanced exactly once per request."""
    if isinstance(getattr(engine, "active_requests", None), int):
        engine.active_requests = max(0, engine.active_requests - 1)


@contextmanager
def driving_engine(engine):
    """Pin *engine* busy and touch its activity clock for the DURATION of a
    plugin-driven generation call (memory auto-consolidate, a scheduled job, ...).

    Wrap this around the ACTUAL chat_stream/complete call, never around merely
    resolving or inspecting the engine (checking .loaded, reading a name) - a
    bare property read must not count as activity, or a model nobody is really
    using again stays pinned resident forever. Plugins reach the live engine via
    PluginManager.inference_engine, which is now resolved fresh at every use site
    across several plugins (#959), so inspection-only reads are common; only the
    call that actually drives the model should register as "in use".

    active_requests is NOT optional here even though a timestamp is also touched:
    _idle_unload_once checks the per-model timestamp FIRST and only consults
    active_requests if that already looks stale, so active_requests>0 is what
    actually prevents eviction mid-task across a multi-round loop where
    individual rounds may pause for a while - a timestamp alone cannot, since
    nothing re-touches it between rounds unless every round does so itself.
    Touching the clock again on exit resets the idle countdown to "now" the
    moment the task genuinely finishes, so the model is not instantly eligible
    for eviction the second a long task ends."""
    name = getattr(engine, "display_name", None)
    _touch_activity(name)
    _pin(engine)
    try:
        yield engine
    finally:
        _unpin(engine)
        _touch_activity(name)


async def _pin_engine(engine: Engine, gen: AsyncIterator[str]) -> AsyncIterator[str]:
    """Release the request pin when a streaming response finishes. The pin itself
    is TAKEN by the handler (via _pin) synchronously right after get_engine, so
    the engine stays pinned across the pre-stream setup window too - this wrapper
    only unpins at stream end."""
    try:
        async for chunk in gen:
            yield chunk
    finally:
        # Starlette acloses THIS wrapper on a client disconnect (that is how the
        # pin below gets released). Explicitly aclose the inner stream too so its
        # own cancel/finally runs NOW - releasing the per-model _inference_lock the
        # producer thread holds - instead of one async-generator GC tick later.
        # No-op on a clean finish (the inner generator is already exhausted).
        try:
            await gen.aclose()
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.exception("closing stream generator on unpin failed")
        _unpin(engine)


async def _stream_sse(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    from localm.inference.protocol import ChoiceDelta, StreamChoice
    from localm.textnorm import ThinkSplitter

    chunk_id = make_chunk_id()
    ts = int(time.time())
    think = ThinkSplitter()   # route <think> reasoning into delta.reasoning_content (H4)

    prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

    # Context-limit handling: compact_messages when close to the limit; reserve a
    # 2048-token buffer for compaction overhead + response generation.
    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            # Off the event loop: compact_messages runs a FULL summarization
            # generation (engine.chat_stream holds the per-model inference lock for
            # up to ~1024 tokens). Run directly on the single-threaded loop it would
            # freeze every other request, the heartbeat, and the disconnect watchers
            # for its whole duration - the same event-loop-block class #541 fixed for
            # the GPU probes. So offload it, exactly as the real generation below is.
            _loop = asyncio.get_running_loop()
            new_messages, changed = await _loop.run_in_executor(
                None, compact_messages, messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

    # Role announcement
    role_chunk = ChatChunk(
        id=chunk_id,
        created=ts,
        model=model_id,
        choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    # Run blocking generator in executor so we don't block the event loop
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    import threading

    # A mid-stream client disconnect makes Starlette throw GeneratorExit into this
    # async generator. Without a cancel path the producer thread below would keep
    # driving engine.chat_stream() all the way to end-of-generation, holding
    # llama.py's per-model _inference_lock the whole time and blocking the next
    # request to THIS model. cancel_event lets the disconnect unwind stop it.
    cancel_event = threading.Event()

    def _generate():
        # engine.chat_stream is called INSIDE the try: Engine.chat_stream is not a
        # generator - it eagerly runs the auto-reload (backend.load()) and
        # load_config() before returning the token generator, so it can RAISE here
        # (a reload OOM, a since-removed GGUF). If that raise escaped the try, the
        # thread would die before the finally enqueued the sentinel and the consumer
        # would block forever at `await token_queue.get()` holding the per-model
        # semaphore - a permanent per-model deadlock. Inside the try, the except
        # surfaces it and the finally still enqueues _DONE.
        gen = None
        try:
            gen = engine.chat_stream(messages, **gen_kwargs)
            for token in gen:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Log (full traceback to the debug log) and surface to the client - a
            # silent thread death looks like an empty reply. Deliberately NOT
            # traceback.print_exc(): _dbg.exception already records the trace, an
            # expected condition (e.g. outgrew n_ctx_max) should reach the user as a
            # clean message, and printing it was the historical WinError-6 crash on
            # Windows.
            from localm.debuglog import logger as _dbg
            _dbg.exception("generation thread failed")
            loop.call_soon_threadsafe(
                token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            # Close the generator chain from THIS thread (it is suspended at its
            # yield right now, so close() is safe here - closing it from the
            # event-loop thread would race the in-flight next() and raise
            # "generator already executing"). close() propagates GeneratorExit down
            # through the backend wrappers into llama.py _generate, whose
            # `with self._inference_lock` then exits and frees the lock
            # deterministically - the whole point of the cancel path. gen is None
            # if chat_stream raised eagerly (nothing to close then).
            try:
                if gen is not None:
                    gen.close()
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("closing generation stream failed")
            # Wake the consumer. If the loop is already gone (server shutdown, or a
            # disconnect whose request-loop has since closed) the consumer is gone
            # too, so dropping the sentinel is correct - don't let it surface as an
            # unhandled daemon-thread exception.
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, _DONE)
            except RuntimeError:
                pass

    # Serialise inference - only one request runs at a time
    async with sem:
        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        gen_error: Exception | None = None
        try:
            while True:
                token = await token_queue.get()
                if token is _DONE:
                    break
                if isinstance(token, Exception):
                    gen_error = token
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                # Stream hook transforms the piece before it is recorded and sent,
                # so usage reflects exactly what the client receives.
                if pipeline is not None and ctx is not None and pipeline.has("stream"):
                    token = pipeline.run_stream(token, ctx)
                completion_parts.append(token)
                for data in _reason_sse(*think.feed(token), model_id, chunk_id, ts):
                    yield data
        finally:
            # Signal the producer to stop. On a clean finish this is a no-op: the
            # thread already exited after _DONE, so t.join() below returns at once.
            # On a disconnect (GeneratorExit raised at the yield above) it makes the
            # thread break its loop, close the generator chain, and release
            # _inference_lock instead of running to end-of-generation. GeneratorExit
            # then keeps propagating, so t.join() below is skipped - the daemon
            # thread self-terminates within ~one token of the cancel.
            cancel_event.set()

        t.join()
        gen_end = time.perf_counter()
        # Release any tail held back while disambiguating a partial <think> tag.
        for data in _reason_sse(*think.flush(), model_id, chunk_id, ts):
            yield data

    if gen_error is not None:
        err_chunk = ChatChunk.token(
            f"\n[inference error: {gen_error}]", model_id, chunk_id, ts)
        yield f"data: {err_chunk.model_dump_json()}\n\n"

    streamed = "".join(completion_parts)
    # Outlet runs after every chunk has been sent, so it cannot alter the live
    # stream (a stream hook does that). Here it only shapes the recorded reply
    # (audit / transcript / side-effects); usage stays tied to what was streamed.
    reply = streamed
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        reply = await pipeline.run_outlet(streamed, messages, ctx)

    _audit_exchange(audit, transcript, messages, reply)

    # Count tokens on the streamed text - what the client actually received
    completion_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_tokens, streamed)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(
            completion_tokens, _decode_elapsed(first_token_at, gen_end)),
        context_capacity=engine.context_capacity(),
    )
    # Honesty: a mid-stream error must not report a clean "stop" on the terminal
    # frame. The error text was already streamed, but a PROGRAMMATIC client keys
    # off finish_reason, so mark it "error" to make the failure machine-detectable.
    finish_reason = "error" if gen_error is not None else _engine_finish_reason(engine)
    done = ChatChunk.done(model_id, chunk_id, ts, usage=usage,
                          finish_reason=finish_reason)
    yield f"data: {done.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_sse_completion(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    chunk_id = make_chunk_id()
    ts = int(time.time())
    # *messages* arrive already inlet-transformed; count tokens on what
    # inference sees (matches the chat path).
    prompt_tokens = await asyncio.get_running_loop().run_in_executor(
        None, engine.count_tokens, _messages_prompt_text(messages))

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()

    import threading

    # See _stream_sse: a mid-stream disconnect must stop the producer thread so it
    # releases llama.py's per-model _inference_lock instead of running to
    # end-of-generation and blocking the next request to this model.
    cancel_event = threading.Event()

    def _generate():
        # chat_stream INSIDE the try: it eagerly runs the auto-reload before
        # returning the generator, so an eager raise must not escape the try and
        # orphan the consumer (see the fuller note in _stream_sse).
        gen = None
        try:
            gen = engine.chat_stream(messages, **gen_kwargs)
            for token in gen:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Surface an inference failure to the client instead of letting this
            # daemon thread die (an uncaught death fires a crash report and looks
            # like an empty reply). _dbg.exception logs the full trace (same
            # contract as the chat-completions path).
            from localm.debuglog import logger as _dbg
            _dbg.exception("completion generation thread failed")
            loop.call_soon_threadsafe(token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            # Close the generator chain from this (suspended) thread so a cancel
            # propagates GeneratorExit into llama.py _generate and frees
            # _inference_lock (see the fuller note in _stream_sse). gen is None if
            # chat_stream raised eagerly (nothing to close then).
            try:
                if gen is not None:
                    gen.close()
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("closing completion generation stream failed")
            # See _stream_sse: tolerate a gone loop when waking the consumer.
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)
            except RuntimeError:
                pass

    async with sem:
        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        gen_error: Exception | None = None
        try:
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                if isinstance(token, Exception):
                    gen_error = token
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                # Stream hook transforms each piece before it is recorded and sent,
                # so usage and the audit trail reflect what the client receives.
                if pipeline is not None and ctx is not None and pipeline.has("stream"):
                    token = pipeline.run_stream(token, ctx)
                completion_parts.append(token)
                chunk = {
                    "id": chunk_id, "object": "text_completion.chunk",
                    "created": ts, "model": model_id,
                    "choices": [{"text": token, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            # No-op on a clean finish (thread already exited after the sentinel);
            # on a disconnect it stops the producer so _inference_lock is released.
            cancel_event.set()

        t.join()
        gen_end = time.perf_counter()

    if gen_error is not None:
        err = {
            "id": chunk_id, "object": "text_completion.chunk",
            "created": ts, "model": model_id,
            "choices": [{"text": f"\n[inference error: {gen_error}]",
                         "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    streamed = "".join(completion_parts)
    # Outlet shapes only the recorded reply (the live stream already went out);
    # then record the exchange (audit + transcript), exactly like chat.
    reply = streamed
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        reply = await pipeline.run_outlet(streamed, messages, ctx)
    _audit_exchange(audit, transcript, messages, reply)

    completion_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_tokens, streamed)
    # Honesty (mirrors the chat path): a mid-stream error is reported as "error",
    # not "stop", so a client keying off finish_reason detects the failure even
    # though the error text was already streamed as a visible chunk.
    done = {
        "id": chunk_id, "object": "text_completion.chunk",
        "created": ts, "model": model_id,
        "choices": [{"text": "", "index": 0,
                     "finish_reason": ("error" if gen_error is not None else "stop")}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_ms": _ttft_ms(gen_start, first_token_at),
            "tokens_per_sec": _tokens_per_sec(
                completion_tokens, _decode_elapsed(first_token_at, gen_end)),
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def _generate_full(engine, messages: list, request=None, *,
                         timing: Optional[dict] = None, **gen_kwargs) -> str:
    """Consume a whole (non-streaming) generation in an executor while watching for
    a client disconnect, and return the accumulated text.

    ``timing``, when passed, is populated with ``first_token_at`` (a perf_counter
    stamp of when the FIRST token arrived) so the caller can report ttft_ms and a
    decode-window throughput even though this path does not stream to the client:
    the handler still drives ``engine.chat_stream`` internally, so the first token
    boundary IS observable here. Left absent by callers that do not report metrics.

    A non-streaming handler is a plain coroutine, and Starlette does NOT cancel it
    when the client disconnects (unlike a StreamingResponse, whose async generator
    it acloses - the hook the _stream_sse fix relies on). So without a cancel path,
    an aborted request with a large/unlimited max_tokens leaves the executor thread
    driving engine.chat_stream() to end-of-generation while holding llama.py's
    per-model _inference_lock, blocking the NEXT request to this model (and this
    coroutine keeps the per-model semaphore too). This is the non-streaming twin of
    the _stream_sse cancel path (PR #540).

    We poll a disconnect signal on the loop (resolved just below - NOT plain
    request.is_disconnected(), which the app's BaseHTTPMiddleware stack defeats)
    and, on disconnect, set a threading.Event the worker checks each token; the
    worker then gen.close()s the chain, cascading GeneratorExit through the backend
    wrappers into llama.py _generate, whose `with self._inference_lock` exits and
    frees the lock now rather than at end-of-generation. Returns the partial text
    produced before the abort (the caller's response is discarded anyway once the
    client is gone).
    """
    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()

    # Resolve a working disconnect poll. In the real server the endpoint sits
    # behind BaseHTTPMiddleware, which makes request.is_disconnected() permanently
    # False (its synthetic receive never yields http.disconnect), so
    # _DisconnectSignalMiddleware publishes one bound to the raw receive under
    # scope[_DISCONNECT_POLL_KEY]. Fall back to request.is_disconnected for a bare
    # request (no middleware - unit tests, or a caller that never disconnects).
    poll = None
    if request is not None:
        scope = getattr(request, "scope", None)
        if isinstance(scope, dict):
            poll = scope.get(_DISCONNECT_POLL_KEY)
        if poll is None:
            poll = getattr(request, "is_disconnected", None)

    def _run() -> str:
        gen = engine.chat_stream(messages, **gen_kwargs)
        parts: list[str] = []
        try:
            for token in gen:
                if cancel_event.is_set():
                    break
                if timing is not None and "first_token_at" not in timing:
                    timing["first_token_at"] = time.perf_counter()
                parts.append(token)
        finally:
            # Close from THIS (suspended) worker thread so GeneratorExit propagates
            # through the backend wrappers into llama.py _generate, whose
            # `with self._inference_lock` then exits - freeing the lock
            # deterministically (see _stream_sse for the fuller rationale). Closing
            # it from the event-loop thread would race the in-flight next().
            try:
                gen.close()
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("closing non-stream generation stream failed")
        return "".join(parts)

    fut = loop.run_in_executor(None, _run)

    async def _watch_disconnect() -> None:
        # The poll is a non-blocking peek (it cancels the receive immediately), so
        # polling it every 0.1s is cheap. poll is None for a caller with no request
        # / no disconnect signal, in which case this loop is an inert wait for the
        # generation to finish.
        try:
            while not fut.done():
                if poll is not None and await poll():
                    cancel_event.set()
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A watcher failure must NEVER cancel a live generation; fall back to
            # the pre-fix behaviour (run to completion) and surface why (rule 5),
            # rather than risk truncating a good reply on a transient poll error.
            from localm.debuglog import logger as _dbg
            _dbg.exception("non-stream disconnect watcher failed")

    watcher = asyncio.ensure_future(_watch_disconnect())
    try:
        return await fut
    finally:
        # Also stop the worker if the handler coroutine itself is cancelled (server
        # shutdown, a timeout middleware): a no-op on the normal path, where the
        # worker already returned and fut is done, so it cannot truncate a good
        # reply. Then retire the watcher.
        cancel_event.set()
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass


def _memory_used_header(ctx) -> dict:
    """F11 observability: render the memory plugin's per-turn recall (stashed in
    ``ctx.state`` by its inlet) into a response-header dict so a client can show a
    "used N memories" chip and the recall degrade reason. Empty when memory did
    not run for this turn (plugin disabled, privacy mode, recall off). The value is
    compact ASCII JSON, header-safe: ``{"n":<int>,"degrade":<reason|null>,"items":
    [{"id","text","source","kind"}...]}`` - json.dumps(ensure_ascii=True) escapes
    newlines and non-ASCII, so the blob is a single header-legal line."""
    if ctx is None:
        return {}
    used = getattr(ctx, "state", {}).get("memory_used")
    if used is None:
        return {}
    payload = {"n": len(used),
               "degrade": ctx.state.get("memory_degrade_reason"),
               "items": used}
    try:
        blob = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return {}
    return {"X-Localm-Memory": blob}


async def _complete(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    request=None,
    **gen_kwargs,
):
    # Call the engine's real methods directly. The previous hasattr-guarded
    # fallbacks (100 prompt tokens, 4096 capacity, an "ok" completion, 10
    # completion tokens) let a method-less mock pass through, so a broken engine
    # returned a fabricated 200 instead of surfacing the failure (AUDIT rule 5 /
    # no facade).
    prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            # Off the event loop (see the same fix in _stream_sse): compaction runs a
            # full generation and must not block the single-threaded loop.
            _loop = asyncio.get_running_loop()
            new_messages, changed = await _loop.run_in_executor(
                None, compact_messages, messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

    # Serialise inference - only one request runs at a time
    gen_error: Exception | None = None
    timing: dict = {}
    async with sem:
        gen_start = time.perf_counter()
        # Cancelable on client disconnect so an aborted request releases the
        # per-model _inference_lock (and this semaphore) instead of generating to
        # end-of-budget behind the next request's back.
        try:
            text = await _generate_full(engine, messages, request,
                                        timing=timing, **gen_kwargs)
        except RuntimeError as e:
            # A generation FAILURE (not enough free VRAM for this prompt, a
            # conversation that outgrew n_ctx_max, a native decode error) is raised
            # as RuntimeError; it must reach the client as a clean reply, never a
            # raw HTTP 500 - the non-streaming twin of the streaming path's
            # gen_error handling. Catch ONLY RuntimeError, not Exception: a broken
            # engine (e.g. a method-less mock -> AttributeError) is a real bug that
            # must surface loudly, not be masked as an "inference error" (rule 5);
            # and CancelledError (client disconnect) must not be swallowed either.
            from localm.debuglog import logger as _dbg
            _dbg.exception("non-streaming generation failed")
            gen_error = e
            text = f"\n[inference error: {e}]"
        gen_end = time.perf_counter()
    first_token_at = timing.get("first_token_at")

    # Outlet fully controls the returned content in the non-streaming path (but a
    # failed generation surfaces its error verbatim, not reshaped by the outlet).
    if gen_error is None and pipeline is not None and ctx is not None and pipeline.has("outlet"):
        text = await pipeline.run_outlet(text, messages, ctx)

    _audit_exchange(audit, transcript, messages, text)

    # Split the model's <think> reasoning out of the visible answer into a
    # separate field (H4), so API clients get clean content (token count stays on
    # the full generated text - reasoning was still generated).
    from localm.textnorm import split_think
    answer, reasoning = split_think(text)

    completion_tokens = await asyncio.get_running_loop().run_in_executor(
        None, engine.count_tokens, text)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        # A non-streaming handler still drives chat_stream internally, so the first
        # token boundary IS observable: report ttft_ms too, and compute tok/s over
        # the decode window only (never folding the cold-start load into the rate).
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(
            completion_tokens, _decode_elapsed(first_token_at, gen_end)),
    )

    response = ChatResponse(
        id=make_chunk_id(),
        created=int(time.time()),
        model=model_id,
        choices=[
            FullChoice(
                message=Message(role="assistant", content=answer,
                                reasoning_content=reasoning or None),
                finish_reason=("error" if gen_error is not None
                               else _engine_finish_reason(engine)),
            )
        ],
        usage=usage,
    )
    return JSONResponse(response.model_dump())


def _protocol_messages_to_dicts(messages: List[Message]) -> list:
    """Convert Pydantic Message objects to plain dicts for backends."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
        else:
            parts = []
            for part in msg.content:
                if hasattr(part, "text"):
                    parts.append({"type": "text", "text": part.text})
                elif hasattr(part, "image_url"):
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url},
                    })
                elif hasattr(part, "input_audio"):
                    parts.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": part.input_audio.data,
                            "format": part.input_audio.format,
                        },
                    })
            result.append({"role": msg.role, "content": parts})
    return result


def run_advertised(app, host: str, port: int, *, mode: str,
                    ssl_certfile: Optional[str] = None,
                    ssl_keyfile: Optional[str] = None,
                    project: Optional[str] = None,
                    isolated: bool = False,
                    log_level: Optional[str] = None) -> None:
    """Advertise *app* in the instance registry and serve it - blocks until
    Ctrl+C.

    This is the shared "advertise, then run_server" tail used by both
    ``serve()`` below (the api-only production path, which also owns the
    ``create_app`` call) and ``localm gui``'s CLI (``plugins/gui/cli.py``),
    which needs the ``app`` object available earlier than this to wire its
    own GUI-only routes/state, so it cannot delegate the ``create_app`` call
    itself here - only this tail was actually duplicated between the two.

    ``mode`` is the instance-registry surface (``"api"`` or ``"full"``).
    ``log_level`` defaults to ``debuglog.uvicorn_log_level()`` when omitted.
    """
    from localm import instances, portmux
    from localm.config import home_dir
    if log_level is None:
        from localm.debuglog import uvicorn_log_level
        log_level = uvicorn_log_level()
    scheme = "https" if ssl_certfile else "http"

    # SRV-CTRLC: no custom Win32 Ctrl+C handler here. A previous one resolved the
    # loop on the control-handler OS thread - NOT the serving loop (portmux's
    # asyncio.run makes a fresh loop) - so loop.stop() never fired, yet it returned
    # True, eating the event and defeating uvicorn's own SIGINT shutdown: it ATE
    # Ctrl+C and the server hung. Removing it lets Ctrl+C flow through uvicorn
    # (KeyboardInterrupt caught in portmux.run_server), kept responsive on Windows
    # by portmux's SRV-6 loop-wakeup. Verified live (Ctrl+Break).

    with instances.advertise(app, home_dir(), host=host, port=port, mode=mode,
                             scheme=scheme, project=project, isolated=isolated):
        # On a TLS bind, also catch a plain-http request on the same port with an
        # https redirect (issue 8); plain binds are a direct uvicorn.run. SRV-5:
        # in debug mode uvicorn logs at "info" so the console shows requests.
        portmux.run_server(app, host=host, port=port, log_level=log_level,
                           ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8642,
          ssl_certfile: Optional[str] = None,
          ssl_keyfile: Optional[str] = None,
          project: Optional[str] = None,
          isolated: bool = False, *,
          mode: str = "api") -> None:
    """Start the server - blocks until Ctrl+C. The real production startup
    path: both ``localm serve`` and ``localm gui`` end up here, the latter via
    ``run_advertised`` above (it builds ``app`` itself, to attach GUI-only
    routes/state before advertising, then reuses the shared tail).

    The caller resolves *port* up front (``config.pick_port``): the default
    auto-bumps through localm's range, while an explicit ``--port`` is honored or
    refused, never silently relocated. By here it is already a concrete free port
    to bind.

    When ``ssl_certfile`` / ``ssl_keyfile`` are given (built-in TLS, NET-1), the
    server speaks HTTPS on this port; a plain-HTTP request to it then fails the
    TLS handshake (effectively refused) rather than crossing the network in
    cleartext.

    ``mode`` is the instance-registry surface (``"api"`` or ``"full"``).

    Advertises itself in the instance registry (H6 phase 3/4) so a future
    launch can discover and attach to it; ``isolated`` keeps it invisible to
    discovery.
    """
    app = create_app(engine, api_landing=True)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    run_advertised(app, host, port, mode=mode, ssl_certfile=ssl_certfile,
                   ssl_keyfile=ssl_keyfile, project=project, isolated=isolated)

