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
from typing import AsyncIterator, List, NamedTuple, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)
from fastapi.security import HTTPBearer

from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback_host  # noqa: F401  (re-export for back-compat)
from localm.inference.backends.base import (
    ContextCapacityExceededError,
    EmbedBatchTooLargeError,
    GrammarUnsupportedError,
    ImageDecodeUnavailable,
    InvalidGrammarError,
    ModelLoadCancelled,
    TriggerValidatorUnavailableError,
    UnsupportedInputError,
    VisionInputError,
)
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
# Display names currently mid-eviction: already detached from _engines and
# _engines_lru, so a fast-path lookup sees them as gone, but the native free
# (evict_engine.unload(), an executor call the eviction loop awaits) has not
# completed. A concurrent switch_engine or get_engine call for THIS SAME name has
# no other way to see that a free is in flight for it, and would otherwise
# construct and load a brand-new engine for the name while the stale eviction is
# still running, then end up pinning an engine that gets freed out from under it.
# switch_engine consults this before constructing or loading *name* and refuses
# with a 503 rather than racing it.
_evicting_names: set[str] = set()
# Default/startup model name
_default_model_name: str | None = None
# Active model name (most recently used/loaded)
_active_model_name: str | None = None
# The name _active_model_name held immediately before a full eviction
# (unload_all_models) cleared it. That eviction KEEPS the Engine in _engines so it
# reloads lazily, and without this the only thing that could still NAME it was
# _default_model_name, which is write-once at startup and never updated by a model
# switch, so an unnamed request would resolve to the startup model rather than the
# one actually in use. Only ever consulted via _resolve_unnamed_model_name, AFTER
# _active_model_name, and cleared by switch_engine on every successful activation,
# so a later eviction on a different path cannot resolve it to a stale name from an
# unrelated eviction.
_last_active_model_name: str | None = None

# The server's audit log, published by create_app() (see the `global _audit`
# there). Defaults to None here so _do_restart's `global _audit` always resolves to
# a real, possibly None, value even in a process where create_app() never ran:
# reading an unassigned global raises NameError rather than giving a clean None.
_audit = None

# Inference serialisation - per-model semaphores mapping display name -> Semaphore
_inference_sems: dict[str, asyncio.Semaphore] = {}

# Backward compatibility references
_engine: Engine | None = None
_inference_sem: asyncio.Semaphore | None = None

# The server's running event loop, captured once at lifespan startup so an OFF-loop
# worker thread, notably the jobs runner on a run_in_executor thread, can submit a
# coroutine back ONTO it via asyncio.run_coroutine_threadsafe. Used to route a
# shared-engine unload through the guarded unload_one_model ON the loop, where
# get_engine and the synchronous request _pin also run: the event loop is the
# serialization point that makes eviction safe, and a bare off-loop
# engine.unload() races get_engine's fast path and ignores the in-flight pin. None
# until a real server lifespan runs, so an off-loop caller detects no-loop and
# degrades safely instead of racing the registry.
_server_loop: "asyncio.AbstractEventLoop | None" = None

# Preemptive model switching (see switch_engine). _switch_desired is the most
# recent switch request, _switch_loading the model whose load is in flight, and
# _switch_cancel aborts it. Touched only on the event-loop thread inside
# switch_engine, except the cancel event, which the loader thread's load-progress
# callback reads; threading.Event is thread-safe.
_switch_desired: Optional[str] = None
_switch_loading: Optional[str] = None
_switch_cancel: Optional["threading.Event"] = None

# Cross-install GPU/VRAM coordination (multi-instance, see localm.gpu_registry).
# None until lifespan startup populates it, and ONLY for a real, non-isolated,
# instances.advertise()'d server (app.state.instance_id set, instance_isolated
# falsy). A plain create_app() test app or an --isolated run never sets it, so it
# never touches the shared machine-wide registry. Shape: instance_id, port, host,
# scheme, token.
_gpu_coord: Optional[dict] = None

# Hang watchdog: a monotonic heartbeat bumped every _HEARTBEAT_INTERVAL_S by
# _hang_heartbeat_loop, an async task ON the loop, and read by the off-loop
# watchdog thread, the debug request log and GET /debug/stacks. A growing
# (now - _hb_monotonic) means the single event loop has stopped making progress.
# The off-loop watchdog thread compares this raw gap against its own multi-second
# threshold (hang_watchdog_threshold(), 10s by default) and is unaffected by the
# note below.
#
# The heartbeat TASK's own startup, in lifespan below, is gated only on pytest not
# being loaded, NOT on the watchdog thread's privacy or env gate: it is pure
# in-memory bookkeeping with no I/O and nothing observable outside this process, so
# it carries none of the privacy considerations that gate the stack-dump-to-disk
# thread, and every reader below needs it regardless of which of them is active.
#
# _loop_lag_seconds() below answers a DIFFERENT question and is what must be used
# for anything reported to a human. Raw (now - _hb_monotonic) saws between 0 and
# about _HEARTBEAT_INTERVAL_S on a perfectly healthy loop, which is just how far
# into the current tick cycle now happens to land. Subtracting the interval turns
# it into a real scheduling-delay figure: about 0 when healthy, positive only when
# a tick itself was late.
#
# None, rather than a real monotonic value, UNTIL THE HEARTBEAT TASK'S OWN FIRST
# TICK, and never seeded at import time. _hang_heartbeat_loop() is only
# created once lifespan() starts and only updates this on its first turn on the
# loop; an import-time seed leaves a cold-start window, from import to first tick,
# where (now - _hb_monotonic) measures elapsed-since-import rather than scheduling
# delay - a number that grows with wall-clock time regardless of what the loop is
# doing. Both readers below treat None as no-reading-yet and report None or skip
# the check rather than inventing a number.
_HEARTBEAT_INTERVAL_S = 1.0
_hb_monotonic: Optional[float] = None


def _loop_lag_seconds() -> Optional[float]:
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
    all". The hang watchdog (large-threshold, above) owns detecting a real
    freeze; this value is for correlating a slow request with a preceding
    stall, not for catching sub-second ones.

    COLD START, before the heartbeat task's first tick (_hb_monotonic is
    still None): returns None, never 0.0, which is the same reading as
    "healthy". Every caller must render None as explicitly unavailable and
    never reuse the "0.0 = no stall longer than the interval" reading for a
    state that has no reading at all - see the debug request log and
    /debug/stacks call sites."""
    if _hb_monotonic is None:
        return None
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
    uses to decide whether eviction was needed - but NOT for the
    empty-directory return value: model_footprint_bytes returns int (never
    None) because its caller always needs a numeric eviction-admission input,
    even a 0 one; this function feeds a registry field other instances treat
    as "how much VRAM does this peer hold", where a 0 is read as a REAL
    measurement (see gpu_registry.py's vram_estimate_bytes and its one
    consumer's `isinstance(e, int) and e > 0` guard). rglob() matching no
    files - an empty directory, or one whose real weights sit somewhere
    rglob does not look - means the size was not measured, so it reports
    None; a 0 there would read as a measured zero."""
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
    model load/unload it is piggybacking on. Logged, never silenced."""
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
    COMBINED split total contradicts itself and drops a sibling holding VRAM
    on this instance's own second split device, turning a cooperative unload
    into a 503. That capacity-vs-identity distinction is what the raw-accessor
    guard in scripts/check_hygiene.py enforces.

    Known limitation: a registry entry advertises ONE ``gpu_index`` per instance
    (see _gpu_registry_sync), so a SPLIT peer is represented only by its main
    device. A split peer whose main device is outside our set is therefore
    skipped even though it may hold VRAM on a device we do use. The effect is a
    missed cooperation opportunity (a 503), never a wrong yank."""
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
    own ``unload_all_models``), so it is conservative about when it is worth
    it:

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
    returns False, so the caller's 503 remains the fallback. A failed
    cooperation attempt is never silenced and never a harder failure than that
    503."""
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
    can tell a full GPU load from a silent CPU fallback instead of reading a
    bare "loaded"/"already_active"."""
    placement = getattr(engine, "gpu_placement", None)
    return dict(placement) if placement else {}


async def switch_engine(name: str, make_engine, *, on_active=None, preempt: bool = True) -> dict:
    global _engines, _engines_lru, _active_model_name, _last_active_model_name, _engine_factory, _last_activity_per_model
    global _switch_desired, _switch_loading, _switch_cancel, _engine, _inference_sem

    # Preemption (a newer selection aborts an in-flight load) is SINGLE-slot and
    # belongs to an explicit user switch, not API-routed loads: with preempt=True,
    # two concurrent DIFFERENT-model requests cancel each other and the earlier one
    # 503s as superseded. get_engine therefore loads with preempt=False, so loads
    # coexist and queue; only the GUI/CLI switch_model path preempts.
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
            # A real active model again, so any name remembered from a past
            # eviction is no longer needed (see _last_active_model_name).
            _last_active_model_name = None
            _engine = _engines[name]
            _inference_sem = sem
            if on_active is not None:
                on_active(name)
            return {"status": "already_active", "model": name,
                    **_gpu_placement_fields(_engines[name])}

        # Perform VRAM check and eviction
        from localm import discover
        # By-symbol AND by-module: the by-symbol names are re-read from
        # the module on every call, since this import is function-scoped, which is
        # what lets tests patch localm.discover.vram_capacity; the module handle is
        # for the constants, which no test patches.
        from localm.discover import gpu_split_shortfall, vram_capacity
        from localm.model_manager import get_model_info
        from localm.config import load_registry
        
        registry = load_registry()
        info = get_model_info(name)
        # file_size feeds the VRAM-eviction estimate below, which is gated on a
        # non-empty registry, so it is only needed for a registered model.
        file_size = 0
        if info is not None:
            m_path, _ = info
            file_size = residency.model_footprint_bytes(m_path)
        elif registry:
            # Registered against a real registry but the files are not on disk: the
            # shipped 404 contract, rather than a fabricated path and size that
            # would run the eviction math on fiction.
            raise HTTPException(404, f"Model files not found: {name}")
        # else: empty registry (single-model or direct-path startup), so there is no
        # size to compute and the eviction block below is skipped.

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
            # vram_capacity() alone proves only that the AGGREGATE combined split
            # free is enough, while the GGUF/llama.cpp backend divides a model by a
            # STATIC per-config ratio with no live per-device capacity check of its
            # own, unlike the HF backend's device_map=auto, which self-corrects from
            # live per-device free VRAM. So an asymmetric split can pass the
            # aggregate check while one device's actual share is short. Applies only
            # to a GGUF-backend load. See discover.gpu_split_shortfall.
            check_split_fit = _is_gguf(m_path)
            # Peers already asked to cooperate during THIS load attempt: each is
            # answerable once, so the loop below always makes progress.
            asked_peers: set = set()
            # The shared embedder is attempted as an eviction candidate at most once
            # per load attempt, mirroring asked_peers above: without the bound, a
            # concurrent get_embedder() reload from an unrelated request between two
            # iterations could repopulate embedder._EMBEDDER and make this branch
            # fire again instead of converging to a load or the final 503.
            embedder_evict_attempted = False

            while True:
                # Off the event loop: vram_capacity() and gpu_split_shortfall()
                # route through discover.list_gpus(), which is deadline-bounded but
                # still a REAL hardware probe that can take up to that deadline, so
                # calling it on the loop would stall every other concurrent request
                # for that long. This loop can re-probe several times per eviction.
                #
                # _GPU_PROBE_CLI_DEADLINE, an alias of the module default, is passed
                # explicitly because THIS caller's correctness depends on waiting out
                # a cold driver init, and an explicit deadline pins that against a
                # future default change. Retrying at a longer deadline after a
                # timeout cannot substitute: an overrun probe is abandoned rather
                # than cancelled, so _gpu_probe_inflight stays True and every retry
                # short-circuits to (last-known-good, GPU_PROBE_BUSY) without
                # probing. The budget must be spent on the FIRST call or not at all.
                #
                # wait_for_inflight=True closes the remaining window: when a
                # CONCURRENT probe already holds the slot - the GUI polls /api/stats
                # every 2500ms, and its probe holds _gpu_probe_inflight through the
                # cold init - this call would otherwise take an instant BUSY plus a
                # stale reading and, with nothing to evict on a fresh server, refuse
                # spuriously. Instead it JOINS that in-flight probe and waits on its
                # result, up to this deadline. Safe only because this call is
                # executor-offloaded.
                v_info, probe_status = await loop.run_in_executor(
                    None, functools.partial(
                        vram_capacity, return_status=True,
                        deadline=discover._GPU_PROBE_CLI_DEADLINE,
                        wait_for_inflight=True))
                free_vram = v_info.get("free")
                # v_info also carries a free_scope tag (FREE_SCOPE_DEVICE vs
                # FREE_SCOPE_PROCESS): whether `free` counts all processes or only
                # this one. It is threaded into the PERMIT decision via
                # residency.fits_alongside_residents's is_process_scoped, because
                # every resident model lives in its OWN isolated worker subprocess,
                # so a PROCESS-scoped reading is structurally blind to exactly the
                # VRAM this check accounts for and can only ever OVER-report free
                # space. Hence PERMIT-only, never a REFUSE gate:
                #  - REFUSE direction: ignoring scope is unconditionally safe, since
                #    a number wrong in the LOW direction only costs a spurious
                #    refusal.
                #  - PERMIT direction: a PROCESS-scoped free that over-reports
                #    headroom is the direction that OOMs, so it is gated exactly
                #    like an unmeasurable reading.
                free_scope = v_info.get("free_scope")
                # The scope check above closes every avenue into a PROCESS-scoped
                # permit regardless of cause, so nothing below gates on the specific
                # one: joining a concurrent probe could once inherit a thinner
                # cold-start budget, cold-skipping discover.py's device-global
                # correction (_apply_device_global_free) so the joined reading came
                # back PROCESS-scoped. Every deadline is now unified at 15.0s, and
                # the cold-skip guard in discover.py still exists for a future
                # short-deadline caller, which this check would catch anyway by
                # reading the resulting free_scope tag directly.
                # Two states that want OPPOSITE handling and must not be collapsed
                # into one measurable flag:
                #   probe_ok and free is None -> this BOX cannot report free VRAM at
                #       all (CPU-only, GGUF-only without nvidia-smi, the Windows
                #       registry tier). PERMANENT, so a best-effort single-resident
                #       load is correct; refusing would brick it.
                #   not probe_ok            -> THIS PROBE was inconclusive and the
                #       box may well be measurable. TRANSIENT, and not a licence to
                #       skip the gate.
                probe_ok = probe_status == discover.GPU_PROBE_OK
                measurable = free_vram is not None
                cannot_measure = probe_ok and not measurable
                inconclusive = not probe_ok
                # Plus headroom, for consistency with the aggregate check just below,
                # which also demands vram_required + headroom rather than bare
                # vram_required: a per-device share is not held to a thinner margin
                # than the aggregate ceiling it composes with. gpu_split_shortfall
                # does not bake in headroom itself; that is the caller's decision.
                # No return_status opt-in, which is safe because gpu_split_shortfall
                # is status-aware and admits on non-OK: a stale or timed-out
                # per-device probe yields an EMPTY shortfall rather than a fabricated
                # one, and a device whose reading is blind or stale is omitted rather
                # than quoted. It also runs WARM here, since the vram_capacity probe
                # above already paid the cold-init cost. It does not get the
                # in-flight JOIN (wait_for_inflight), which is moot given that warm
                # ordering.
                #
                # return_shares_adaptive: the refuse-vs-defer branch below MUST know
                # whether the shares were the live auto free-VRAM-proportional ones,
                # whose all-short can only mean combined-short and so defers, or
                # static pinned or equal-fallback shares, which take the hard 503.
                # Config shape alone cannot answer that: auto can DECLINE on a stale
                # index or an unmeasurable device and fall back to equal shares with
                # ratios still unset.
                shortfall, shares_adaptive = (
                    await loop.run_in_executor(
                        None, functools.partial(
                            gpu_split_shortfall, vram_required + headroom,
                            return_shares_adaptive=True))
                    if check_split_fit else ([], False))
                # PERMIT only on a reading that was actually taken. A frozen
                # last-known-good that happens to read HIGH would otherwise pass and
                # admit a load with zero eviction on top of resident peers: the same
                # stale int that produces a dishonest 503 in the LOW direction
                # produces a native OOM or driver TDR in the HIGH one. `measurable`
                # is folded into the shared predicate and stays a local because the
                # refuse-vs-defer branch below distinguishes cannot-measure from
                # inconclusive.
                over_cap = residency.exceeds_resident_cap(
                    _engines_lru, name, resident_cap)
                vram_ok = residency.fits_alongside_residents(
                    free_vram=free_vram, vram_required=vram_required,
                    probe_ok=probe_ok, headroom=headroom, shortfall=shortfall,
                    is_process_scoped=free_scope == discover.FREE_SCOPE_PROCESS)
                if vram_ok and not over_cap:
                    break

                # Make room. With measurable VRAM, evict idle models until the new
                # one fits, also satisfying any configured split device's own share.
                # When the box cannot measure, or this probe was inconclusive, it
                # cannot be proven to fit alongside others, so this falls back to
                # single-resident and evicts every idle model rather than stacking
                # until the driver OOMs. The two diverge only once eviction is
                # exhausted below: cannot-measure loads best-effort, inconclusive
                # refuses.
                #
                # Victim safety - never the requested model, never one that is
                # serving, never one another path is already mid-freeing, never a
                # pinned one - lives in residency.pick_eviction_victim, shared with
                # the MCP server's cache.
                evict_name = residency.pick_eviction_victim(
                    _engines_lru, _engines, requested=name, pinned=pinned)

                if evict_name is None and vram_ok:
                    # ONLY the resident cap wanted room, and nothing could be freed
                    # because every peer is pinned or serving. VRAM already fits, so
                    # this is a user PREFERENCE going unmet rather than a safety
                    # constraint: the model loads and the missed cap is logged.
                    # Falling through to the exhaustion path below would answer a
                    # preference with a 503, and would first ask a SIBLING localm
                    # instance to dump its models to free VRAM that was never short.
                    from localm.debuglog import logger as _dbg
                    _dbg.warning(
                        "max_resident_models=%s wanted room for %s but no "
                        "resident model could be evicted (resident=%s, "
                        "pinned=%s); free VRAM is sufficient, so loading it "
                        "anyway over the cap",
                        resident_cap, name, list(_engines_lru), sorted(pinned))
                    break

                if evict_name is None:
                    # Nothing idle among the chat engines. The shared embedder has
                    # a separate lifecycle from _engines: it is loaded
                    # independently by RAG, memory and coder-episode callers via
                    # get_embedder(), never through switch_engine, so the LRU scan
                    # above can never see it even though it can hold real,
                    # evictable VRAM.
                    #
                    # Honor the in-flight-request pin exactly like
                    # unload_all_models's own embedder release: a request
                    # mid-embed() must not have its embedder, and the isolated
                    # worker process it is waiting on, freed out from under it.
                    # reset_embedder(force=False) checks active_requests()==0 and
                    # clears the embedder in ONE locked step, closing the TOCTOU
                    # window a separate active_requests() call would leave, since
                    # IsolatedEmbedder.embed() pins active_requests without taking
                    # embedder._LOCK.
                    #
                    # embedder_evict_attempted, set above the while loop, bounds
                    # this to once per load, for the same reason as asked_peers
                    # below: loaded_dim() answers None again once this attempt
                    # clears it, but a concurrent get_embedder() could repopulate
                    # it before the next pass, so self-clearing is not a progress
                    # guarantee.
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
                        # This BOX cannot report free VRAM at all and every
                        # remaining model is busy, or none is loaded: free what can
                        # safely be freed and load best-effort. Refusing here would
                        # brick every CPU-only, GGUF-only or registry-tier box,
                        # which can never measure.
                        break
                    if inconclusive:
                        # The probe did not complete, so free VRAM is unknown.
                        # Refuse rather than load with no VRAM check: the failure
                        # modes are not symmetric. A wrong permit hands a too-big
                        # model to the native loader, which can hard-abort the
                        # process rather than return NULL, and is unrecoverable; a
                        # wrong refusal is a 503 the caller can retry, by which
                        # time the abandoned probe has usually landed and the
                        # driver is warm. No figure is quoted, because none was
                        # measured.
                        #
                        # Two ways the probe fails to complete, and both the long
                        # deadline and wait_for_inflight=True above target them:
                        # a cold init on a fresh process, which the deadline waits
                        # out, and a CONCURRENT probe holding the slot, which the
                        # join waits on instead of taking an instant BUSY. So
                        # reaching here needs the probe to blow the FULL deadline
                        # even after joining, which is a stuck driver rather than
                        # an ordinary cold init or heartbeat collision.
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
                    # coordination, see localm.gpu_registry). Off the event loop,
                    # since it may make a blocking loopback call. Advisory: any
                    # failure falls through to the 503 below. Bounded and
                    # fit-checked (see _attempt_cooperative_unload): each peer is
                    # asked at most once per load attempt, so this loop always
                    # progresses, and a yank that could not free enough is skipped.
                    #
                    # A PIN is a local preference, exactly like the cap above, so
                    # it must not cost a SIBLING instance its models. If an
                    # unpinned peer would have been evictable, the pin is the only
                    # reason local eviction came up empty, so no peer is asked and
                    # this falls through to the backend's own split-aware sizing,
                    # which honors the pin at this instance's own expense.
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
                        # The aggregate may well be enough: it is specifically the
                        # configured split's per-device share that is short, so the
                        # message names the device(s) rather than the aggregate.
                        # shortfall is always empty when free is unmeasurable,
                        # because list_gpus() DROPS a device that fails to report
                        # rather than emitting free=None.
                        # STATIC shares only (pinned ratios, or auto declined into
                        # the equal fallback after a stale configured index or a
                        # device without a free reading): a hard refusal, even
                        # though everything below defers to the backend, because
                        # with static shares apply_gpu_split() divides the model by
                        # a ratio that ignores live per-device free VRAM, and the
                        # backend's own sizing budgets the split's COMBINED
                        # capacity (_split_free_total_bytes) rather than any one
                        # device's static share. So this per-device check is the
                        # only gate that catches one split device being
                        # individually short while the aggregate fits. Keyed on
                        # shares_adaptive, NOT on whether ratios are set in config:
                        # with ratios unset but auto declined, the loader applies
                        # the same equal fallback this gate just checked.
                        detail = "; ".join(
                            f"GPU {d['index']} needs ~{d['needed'] // 1024 ** 2} MB, "
                            f"{d['free'] // 1024 ** 2} MB free" for d in shortfall)
                        raise HTTPException(
                            503, f"Not enough VRAM on the configured split "
                            f"device(s) to load '{name}' ({detail}).")
                    # ADAPTIVE shares (live auto free-VRAM-proportional split): a
                    # non-empty shortfall can only mean the COMBINED estimate is
                    # short, since each device's auto share fits its free whenever
                    # the aggregate fits, gpu_split_shortfall having been computed
                    # with the same auto ratios the loader will pin. It therefore
                    # falls through to the same defer-to-backend path as the
                    # aggregate-only miss below, where the backend's split-aware
                    # sizing judges and partial offload is available.
                    # Local and cooperative eviction exhausted, with no
                    # hard-refusable pinned-share shortfall. What remains is this
                    # loop's coarse whole-model estimate (file_size * 1.2) not
                    # being met, combined across an auto split or on a single GPU.
                    # That estimate assumes the WHOLE model lands in VRAM, while
                    # the backend's own load() can already make a too-big model
                    # fit: GgufBackend's n_gpu_layers_auto, on by default, sizes
                    # how many layers fit free VRAM and puts the rest on system
                    # RAM, and HFBackend's device_map=auto does the unconditional
                    # equivalent.
                    #
                    # Fall through to a real load attempt instead: _check_vram() is
                    # the accurate, backend-owned final gate, raising only when the
                    # model genuinely cannot fit even at 0 GPU layers, which the
                    # except handler around new_engine.load() below turns into the
                    # same clean 503 shape for every caller.
                    from localm.debuglog import logger as _dbg
                    _dbg.info(
                        "switch_engine: '%s' exceeds the whole-model VRAM estimate "
                        "(need ~%s MB, %s MB free) after eviction - deferring to the "
                        "backend's own load-time sizing instead of refusing",
                        name, vram_required // 1024 ** 2, free_vram // 1024 ** 2)
                    break

                evict_engine = _engines[evict_name]
                free_before = free_vram
                # Defensive re-check right before committing, without the victim's
                # own semaphore, which would reintroduce the two-switch
                # lock-ordering deadlock below. The LRU scan above already required
                # active_requests==0 and there is no await between it and here, so
                # this cannot currently fire; it guards against a future await in
                # the scan reopening the window. If the candidate became pinned, it
                # is abandoned and the scan runs again.
                if getattr(evict_engine, "active_requests", 0) != 0:
                    continue

                # Detach the victim from the live registry BEFORE the native free,
                # not after. Unloading without the victim's own semaphore is safe:
                # active_requests == 0 means no request is pinned on it, since a
                # request pins its engine for its whole lifetime, so no decode races
                # the free, and taking the victim sem while holding the target sem
                # would risk a two-switch lock-ordering deadlock. active_requests
                # alone does NOT stop a QUEUED request pinning the victim DURING the
                # unload await, because it pins lock-free after the check passed:
                # while await unload() yields the loop, get_engine's fast path would
                # still see the victim loaded and registered and pin a doomed
                # engine. Removing it from _engines first closes that window, so a
                # queued request misses the fast path and gets a fresh,
                # VRAM-checked reload, and a concurrent switch_engine(evict_name)
                # rebuilds a fresh engine rather than racing load() against this
                # backend object mid-free. The flag covers any stale reference
                # already resolved. Removal stays pop-and-guarded: even pre-await, a
                # concurrent idle or explicit unload could have popped the victim.
                evict_engine.unloading = True
                _engines.pop(evict_name, None)
                if evict_name in _engines_lru:
                    _engines_lru.remove(evict_name)
                _inference_sems.pop(evict_name, None)
                # If the victim was the active or compat engine, those pointers are
                # dropped too, so a concurrent get_engine back-compat re-import
                # cannot resurrect the just-detached victim mid-free and nothing
                # keeps serving a being-freed engine as active. switch_engine sets
                # them to the newly-loaded model at the end, or leaves them cleared
                # if the load fails. The other unload paths do the same.
                if _active_model_name == evict_name:
                    _active_model_name = None
                if _engine is evict_engine:
                    _engine = None
                    _inference_sem = None
                # See _evicting_names: this marks THIS name as mid-free, so a
                # concurrent switch_engine or get_engine call for it - a queued
                # reload of the very model being evicted - refuses instead of racing
                # a fresh load against this still-running native free. try/finally,
                # because every path out of the free below must clear it or the name
                # would be permanently unloadable.
                _evicting_names.add(evict_name)
                try:
                    await loop.run_in_executor(None, evict_engine.unload)

                    # Wait for the native VRAM free to land before re-checking, so
                    # the next iteration does not see a stale-low reading and
                    # over-evict. Only meaningful when measurable.
                    if measurable and free_before is not None:
                        await loop.run_in_executor(
                            None,
                            lambda: wait_for_vram_release(
                                lambda: vram_capacity().get("free"), before_bytes=free_before))
                finally:
                    _evicting_names.discard(evict_name)

        # See _evicting_names: *name* itself may be the victim another concurrent
        # switch_engine call just detached and is still natively freeing, with a
        # queued reload landing in exactly that window. Refuse rather than
        # construct and load a fresh engine that races the still-running free; the
        # caller may retry once the in-flight eviction finishes.
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
        # outlives a load, since idle-unload keeps it in _engines for lazy reload,
        # so a preempt=True switch that gets superseded leaves its FIRED event on
        # the engine and nothing else clears it; the next preempt=False load would
        # then reuse that stale SET event and the backend would abort the load at
        # once. Loads of one model are serialized by its own semaphore above, so
        # this cannot clobber a concurrent load's event.
        if hasattr(new_engine, "set_load_cancel"):
            new_engine.set_load_cancel(cancel)
        if preempt:
            # Only an explicit switch REGISTERS the hook globally, so only a newer
            # explicit switch can abort this load; API-routed loads (preempt=False)
            # run to completion and are never cancelled by a concurrent load of a
            # different model.
            _switch_cancel = cancel
            _switch_loading = name
        try:
            await loop.run_in_executor(None, new_engine.load)
        except ModelLoadCancelled:
            return {"status": "superseded", "model": name, "by": _switch_desired}
        except RuntimeError as exc:
            # The backend's own sizing found the model cannot fit even with 0 GPU
            # layers (GgufBackend._check_vram), or its native worker failed
            # outright (GgufBackend._load_native): both raise a plain,
            # already-informative RuntimeError. Only the GUI load-model button and
            # the coder plugin's model switch wrap switch_engine in their own
            # try/except; the OpenAI-compatible routes do not, so converting it to
            # an HTTPException here, once, gives every caller the same clean
            # message the VRAM-refusal 503s above already get.
            raise HTTPException(503, f"Failed to load '{name}': {exc}") from exc
        finally:
            if preempt and _switch_cancel is cancel:
                _switch_cancel = None
                _switch_loading = None
                
        _engines[name] = new_engine
        # Seed the per-model activity clock the INSTANT this engine becomes
        # resident, not when it first answers a request. Otherwise
        # _idle_unload_once's _last_activity_per_model.get(name, _last_activity)
        # falls back to the GLOBAL _last_activity, which holds whatever OTHER model
        # was last used, so a model switched to while the previous one was already
        # idle past the TTL looks instantly overdue for eviction.
        _last_activity_per_model[name] = time.monotonic()
        _engines_lru.append(name)
        _active_model_name = name
        # See the already-active fast path above: there is a real active model
        # again, so any name remembered from a past eviction is stale.
        _last_active_model_name = None
        _engine = new_engine
        _inference_sem = sem
        if on_active is not None:
            on_active(name)
        # Cross-install GPU coordination: reflect the newly-active model so a
        # sibling's next VRAM or eviction check sees fresh state. A no-op when not
        # registered (see _gpu_registry_sync). Offloaded because it does registry
        # file I/O plus, with a non-zero main_gpu_index, a real GPU driver probe.
        await loop.run_in_executor(None, _gpu_registry_sync)
        return {"status": "loaded", "model": name,
                **_gpu_placement_fields(new_engine)}


def _resolve_unnamed_model_name() -> str | None:
    """The model name an unnamed (or ``"localm"``-named) request currently
    resolves to - read-only, no loading or registry validation.

    Shared by get_engine's own fallback below and by ``GET /health``, so both
    agree on what "recoverable" means: /health must not report "no model" for
    a state chat already knows how to fix on the next request, and it must not
    duplicate this chain and risk the two silently drifting apart.
    ``_last_active_model_name`` covers the gap ``_default_model_name`` alone
    cannot: that one is write-once at startup (create_app) and never updated
    by a model switch, so a model switched to after boot and then evicted
    (unload_all_models keeps its Engine in _engines for lazy reload) would
    otherwise resolve back to the STARTUP model instead of the one actually in
    use."""
    return _active_model_name or _last_active_model_name or _default_model_name


async def get_engine(model_name: str | None, *, load: bool = True) -> Engine:
    """Resolve the engine for *model_name*, loading it if necessary.

    With ``load=False`` the resolved engine is returned WITHOUT forcing a load -
    for callers like /v1/embeddings whose backend may not need the model resident
    at all (a GGUF backend embeds via the dedicated embedder). The caller decides
    whether to load. Registration/resolution (and its 404) still apply.
    """
    global _engines, _engines_lru, _active_model_name, _default_model_name, _last_active_model_name, _inference_sems, _engine, _inference_sem

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
        name = _resolve_unnamed_model_name()

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
        if (name not in registry and name != _default_model_name
                and name != _active_model_name and name != _last_active_model_name):
            registered = sorted(registry.keys())
            msg = f"Model '{name}' is not registered."
            if registered:
                msg += f" Registered models in your library: {', '.join(registered)}. Use 'localm pull' to add a new model."
            raise HTTPException(404, msg)

    # A name that resolved to nothing (an empty or localm-named request with no
    # active model, no remembered last-active and no default) is answered with a
    # 503 rather than falling through to switch_engine(None) and a TypeError 500.
    if not name:
        raise HTTPException(503, "No model is loaded and none was specified. "
                            "Load a model first or name one explicitly.")

    # `not unloading`: never hand back, and let the caller pin, an engine an
    # eviction or unload path is mid-freeing. An engine flagged unloading falls
    # through to switch_engine below, which reloads it cleanly under its per-model
    # semaphore; the unloader holds that semaphore for the native free, so the
    # reload serializes after it.
    if name in _engines and _engines[name].loaded and getattr(_engines[name], "unloading", False) is not True:
        if name in _engines_lru:
            _engines_lru.remove(name)
        _engines_lru.append(name)
        _active_model_name = name
        _engine = _engines[name]
        _inference_sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        return _engines[name]

    if not load:
        # Return the engine object WITHOUT loading it: reuse a tracked, possibly
        # unloaded, engine, else build a fresh one via the factory. Does not go
        # through switch_engine, so no model is loaded and nothing is evicted.
        return _engines.get(name) or _engine_factory(name)

    res = await switch_engine(name, _engine_factory, preempt=False)
    if res.get("status") == "superseded":
        raise HTTPException(503, f"Model load was superseded by a newer request: {res.get('by')}")

    return _engines[name]


def _add_vram_fields(result: dict, *, before, released, after, before_fresh: bool,
                     before_scope=None) -> None:
    """Add vram_freed/vram_before_bytes/vram_after_bytes to *result* when
    measurable, plus an honest flag when the reading cannot be presented as
    current fact, so a wrong number is never asserted.

    ``before is None`` returns early and adds NOTHING - the benign case (a
    CPU-only box, or the Windows registry tier, which reports total but never
    free) where a completed probe simply has no free reading to give. That is not
    the fault this guards, and must not be dressed up as one: no VRAM telemetry is
    the normal, permanent state there, so saying nothing is the honest answer.

    THREE independent ways this reading can be wrong, and they are not the same bug:

    - NOT FRESH (``before_fresh`` false): the probe timed out or was busy, so the
      'before' value is a stale cached one (see _vram_free_reading).
    - AFTER UNVERIFIABLE (``released is None``): wait_for_vram_release() could not
      verify the outcome (the 'after' reading went unmeasurable), so ``vram_freed``
      is null rather than a false "VRAM did not drop", which a reading that never
      refreshed cannot support.
    - NOT DEVICE-SCOPED (``before_scope`` is FREE_SCOPE_PROCESS): the probe was
      perfectly fresh, but on this platform the driver reports only the CALLING
      process's own allocations. Since every GGUF load runs in an isolated worker
      subprocess (backends/gguf.py), the model's VRAM is in another process and
      absent from the number, so before/after come back byte-identical and
      vram_freed false across a load/unload cycle that really worked.

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

    ONE implementation, shared by two callers with two different auth models:
    the owner-scoped ``/v1/models/unload`` route (``MODELS_WRITE``), and the
    coordination-token-gated ``POST /v1/instances/cooperate-unload`` (a sibling
    localm instance asking THIS one to free VRAM - multi-instance GPU
    coordination, see ``localm.gpu_registry``)."""
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
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
        # Honor the in-flight-request pin, like the VRAM-eviction and idle-unload
        # paths: a pinned engine has a request generating against it, so unloading
        # it would free VRAM the request immediately reloads and race a
        # use-after-unload. It is skipped and reported as still in use. Gated on
        # isinstance(int) exactly like _pin/_unpin, so a non-int active_requests
        # from a bare test double counts as not pinned.
        active = getattr(engine, "active_requests", 0)
        if isinstance(active, int) and active > 0:
            skipped_in_use.append(name)
            continue
        sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        # Flag BEFORE acquiring the semaphore, so a request arriving after the pin
        # check above cannot take get_engine's fast path and pin this engine while
        # it is being freed; such a request blocks on the same semaphore and
        # reloads cleanly afterwards. Cleared in finally, so the engine kept in
        # _engines reloads lazily.
        engine.unloading = True
        try:
            async with sem:
                await loop.run_in_executor(None, engine.unload)
                unloaded_models.append(name)
                if name in _engines_lru:
                    _engines_lru.remove(name)
        finally:
            engine.unloading = False

    # Release the shared embedder too: it has a separate lifecycle from _engines
    # and is loaded independently by RAG, memory and coder-episode callers via
    # get_embedder(), never through switch_engine.
    #
    # loaded_dim() and active_requests() MUST run in the executor rather than
    # directly on this coroutine: get_embedder() can hold embedder._LOCK for the
    # full duration of an IsolatedEmbedder native or subprocess load, up to its
    # load timeout, and both accessors block on that same lock, so a synchronous
    # call would freeze the whole event loop for that window. Offloading keeps the
    # wait local to this coroutine.
    embedder_was_loaded = False
    embedder_dim = await loop.run_in_executor(None, _embedder_mod.loaded_dim)
    if embedder_dim is not None:
        # Honor the in-flight-request pin for the embedder too, exactly like the
        # chat-engine loop above: a request mid-embed() must not have its embedder,
        # and the isolated worker process it is waiting on, freed out from under
        # it. reset_embedder(force=False) checks active_requests()==0 and clears
        # the embedder in ONE locked step rather than two separate executor calls,
        # closing the TOCTOU window that a standalone active_requests() check plus
        # an unconditional reset_embedder() would leave, since
        # IsolatedEmbedder.embed() pins active_requests without taking
        # embedder._LOCK. Otherwise it is skipped and reported alongside the pinned
        # chat engines rather than reported as unloaded.
        cleared = await loop.run_in_executor(
            None, functools.partial(_embedder_mod.reset_embedder, force=False))
        if cleared:
            embedder_was_loaded = True
        else:
            skipped_in_use.append("embedding model")

    # Update compatibility pointers, but NOT when the active engine was a pinned
    # one left loaded: clearing it would strand the in-flight request's active
    # model.
    if _active_model_name not in skipped_in_use:
        if _active_model_name:
            # The Engine stays in _engines above for a lazy reload on the next
            # request. Its NAME is kept alive too, or nothing can resolve an
            # unnamed request back to it (see _last_active_model_name and
            # _resolve_unnamed_model_name).
            _last_active_model_name = _active_model_name
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
    # Cross-install GPU coordination: reflect the changed state for a sibling's
    # next eviction decision. A no-op when not registered. Offloaded because it
    # does registry file I/O plus, with a non-zero main_gpu_index, a GPU probe.
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
    a `localm pull`-ed GGUF selected as the embedding model) shows as
    "loaded" on the Models page while its per-row Unload button does nothing.
    Matched by resolved PATH, not by name/config, so it is correct
    regardless of how ``embedding_model`` was originally resolved (an explicit
    path, a registered name, or a known key) - what matters is which file is
    actually resident. Returns None when *name* is not the embedder, so the
    caller falls back to its normal "already_unloaded" outcome for a genuinely
    untracked/never-loaded chat model."""
    from localm.inference import embedder as _embedder_mod
    # Executor-offloaded rather than called directly: get_embedder() can hold
    # embedder._LOCK for the full duration of an IsolatedEmbedder native or
    # subprocess load, and loaded_path() blocks on that same lock, so a
    # synchronous call here would freeze the whole event loop for that window.
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

    # Honor the in-flight-request pin: a request mid-embed() must not have its
    # embedder freed out from under it, so it is reported as still in use rather
    # than as unloaded.
    #
    # TWO layers. The cheap active_requests() precheck here avoids paying for the
    # _vram_free_reading() hardware probe below on the common busy case; that probe
    # is NOT executor-offloaded and can block this single-threaded event loop for
    # up to discover._GPU_PROBE_DEADLINE seconds. reset_embedder(force=False) is
    # what actually authorizes the close, atomically re-checking
    # active_requests()==0 under embedder._LOCK in the SAME step as the close, so a
    # pin arriving in the gap between this precheck and the close is still caught;
    # an unlocked active_requests() call before an unconditional reset_embedder()
    # would be a TOCTOU window, since IsolatedEmbedder.embed() pins
    # active_requests without taking embedder._LOCK. Both calls are
    # executor-offloaded for the same reason as loaded_path() above.
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
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
    loop = asyncio.get_running_loop()
    from localm.vram import (_live_free_vram_bytes, _vram_free_reading,
                             wait_for_vram_release)

    engine = _engines.get(name)
    if engine is None or not engine.loaded:
        embedder_result = await _unload_embedder_if_matches(name, loop)
        if embedder_result is not None:
            return embedder_result
        return {"status": "already_unloaded", "model": name}
    # Honor the in-flight-request pin: an engine a request is generating on must
    # not be unloaded out from under it, since it would reload anyway and make the
    # freed report a lie. Reported as in use instead. The isinstance(int) guard
    # matches _pin/_unpin, so a bare test double counts as not pinned.
    active = getattr(engine, "active_requests", 0)
    if isinstance(active, int) and active > 0:
        return {"status": "in_use", "model": name, "vram_freed": 0}

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
    # Flag BEFORE acquiring the semaphore, so a request arriving after the pin
    # check above cannot fast-path-pin this engine while it is being freed; such a
    # request blocks on the same semaphore and reloads cleanly afterwards. Cleared
    # in finally, so the engine kept in _engines reloads lazily.
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
        if _active_model_name:
            # The Engine stays in _engines above for a lazy reload on the next
            # request. Its NAME is kept alive too, same as unload_all_models, or
            # nothing can resolve an unnamed request back to it (see
            # _last_active_model_name and _resolve_unnamed_model_name).
            _last_active_model_name = _active_model_name
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
    # driver probe, kept off the event loop as in the heartbeat.
    await loop.run_in_executor(None, _gpu_registry_sync)
    return result


# Monotonic timestamp of the last inference request, for the optional idle-unload
# loop (config idle_unload_seconds). Touched at the start of each inference
# endpoint, so the idle window is measured from the last request.
_last_activity: float = time.monotonic()
_last_activity_per_model: dict[str, float] = {}


def _touch_activity(name: str | None = None) -> None:
    """Record that an inference request just arrived (resets the idle timer)."""
    global _last_activity, _last_activity_per_model
    now = time.monotonic()
    _last_activity = now
    if name:
        _last_activity_per_model[name] = now


def rekey_loaded_model(old_name: str, new_name: str) -> bool:
    """Re-key every in-memory record of a loaded model's identity after its
    registry entry was renamed *old_name* -> *new_name*, so a still-loaded/
    serving engine is not orphaned under its old name.

    This is load-bearing, not cosmetic: ``active_model()`` reads
    ``_engine.display_name``, and the GUI's remove-model guard is exactly
    ``req.model == active_model()`` - without this re-key, renaming the
    active model would leave that guard comparing the NEW registry name
    against the engine's stale OLD display_name, so it would never match and
    the GUI could delete the file out from under the model still serving
    requests.

    A synchronous, in-memory-only op (dict/list mutation, no I/O, no
    ``await``), so it is safe to call directly from an async route body: on
    a single-threaded event loop nothing else can interleave between the pop
    and the re-insert. Returns whether an ENGINE was re-keyed: False when
    *old_name* had none, which is the common case, since renaming a model
    that was never loaded in this process needs no engine bookkeeping.

    The startup and last-active POINTERS are corrected either way, before
    that early return: they are wrong the moment the registry moves,
    regardless of what is in the engine map. On a server started with the
    renamed model, a stale ``_default_model_name`` puts a ghost row for the
    old name into ``GET /v1/models`` (list_models adds the startup name when
    the registry lacks it) and makes ``switch_engine``'s registration check
    at ``name != _default_model_name`` accept a request for a name the
    registry no longer has instead of answering 404.
    ``_last_active_model_name`` is the same shape one step along:
    ``_resolve_unnamed_model_name`` falls back to it after a full eviction,
    so an unnamed request would resolve to a name that no longer exists."""
    global _active_model_name, _default_model_name, _last_active_model_name
    if _default_model_name == old_name:
        _default_model_name = new_name
    if _last_active_model_name == old_name:
        _last_active_model_name = new_name
    engine = _engines.pop(old_name, None)
    if engine is None:
        return False
    engine.display_name = new_name
    _engines[new_name] = engine
    if old_name in _engines_lru:
        _engines_lru[_engines_lru.index(old_name)] = new_name
    if _active_model_name == old_name:
        _active_model_name = new_name
    sem = _inference_sems.pop(old_name, None)
    if sem is not None:
        _inference_sems[new_name] = sem
    ts = _last_activity_per_model.pop(old_name, None)
    if ts is not None:
        _last_activity_per_model[new_name] = ts
    return True


async def rename_registered_model(model: str, new_name_raw: str) -> dict:
    """Move a registry entry to a new name AND re-key the live engine, in that
    order, in ONE process. The single entry point every rename route uses.

    The re-key is not bookkeeping that can be added later by whoever remembers:
    until it runs, the registry holds the new name while the engine map is
    still keyed on the old one, and every name-keyed check downstream then asks
    about a name nothing is under. Pairing the two here means a route cannot
    perform half of it. Renaming from OUTSIDE this process cannot do the second
    half at all, so ``localm rename`` asks a running server to call this instead
    of moving the registry entry itself.

    Raises HTTPException (404 unregistered, 409 name taken, 400 rename failed)
    so both the /v1 and the GUI route answer identically. Returns the response
    body: status, the old name, the sanitized new name, and the migration
    notes, which must reach the caller rather than only the server log - a user
    has no other way to learn that e.g. a per-project coder config still names
    the old model.
    """
    from localm.config import load_registry
    from localm.executor import get_plugin_executor
    from localm.model_manager import _sanitize_name, rename_model_with_notes

    registry = load_registry()
    if model not in registry:
        raise HTTPException(404, f"Model not registered: {model}")
    # Sanitizing happens server-side, so the collision check and the response both
    # speak the sanitized name, not the raw text the caller sent.
    new_name = _sanitize_name(new_name_raw)
    if new_name != model and new_name in registry:
        raise HTTPException(409, f"Name already taken: {new_name}")
    loop = asyncio.get_running_loop()
    try:
        renamed, notes = await loop.run_in_executor(
            get_plugin_executor(), rename_model_with_notes, model, new_name_raw)
    except Exception as e:
        raise HTTPException(400, f"Rename failed: {e}")
    if not renamed:
        # rename_model_with_notes distinguishes vanished from name-taken in its
        # own console output, but only the bool crosses the executor boundary, so
        # the losing race is re-derived here.
        if model not in load_registry():
            raise HTTPException(404, f"Model not registered: {model}")
        raise HTTPException(409, f"Name already taken: {new_name}")
    # Synchronous and in-memory only, with no await, so it is safe to call
    # directly on the event loop once the executor call above returns.
    rekey_loaded_model(model, new_name)
    return {"status": "renamed", "model": model, "new_name": new_name,
            "notes": notes}


class ModelFileHold(NamedTuple):
    """Why a model removal must be refused, and which loaded engine is the
    reason.

    *reason* is None when that engine is PROVEN to hold the file (both paths
    resolved and named the same file). Otherwise it says why holding could not
    be ruled out, and the removal is refused anyway - see
    :func:`loaded_engine_holding_model_file` for why an unknown is a refusal.
    The caller reports it, so it has to read as a sentence fragment after
    "'<key>' is loaded and ".
    """
    key: str
    reason: str | None = None


def loaded_engine_holding_model_file(
        model: str, registry: dict | None = None) -> "ModelFileHold | None":
    """Whether removing registry entry *model* would delete a file that a LOADED
    engine is holding. Returns None only when that is POSITIVELY RULED OUT;
    otherwise a :class:`ModelFileHold` naming the engine responsible.

    IDENTITY BY FILE PATH, NOT BY NAME. A guard that asks "is a loaded engine
    keyed under THIS NAME" is the same question only while every renamer re-keys
    the engine map. ``localm rename`` runs in a SEPARATE PROCESS and cannot
    reach into this one's memory, so after a CLI rename the engine is still
    keyed on the old name, a name-keyed guard misses, and the route goes on to
    spawn ``localm rm`` and delete the GGUF out from under a live, serving
    engine. This guard asks the question the deletion itself turns on, so a
    caller that does not re-key is covered by construction.

    IT FAILS CLOSED. The question is not "do these two paths compare equal" but
    "can I establish that nothing live is using this file", and the two differ
    on every input where the comparison cannot be made at all:

    * the engine's recorded ``model_path`` is missing or empty;
    * resolving the engine's path raises (a UNC or network path whose share is
      momentarily unreachable, a permission error, an embedded NUL);
    * THIS model's own registry path fails to resolve.

    Each of those is "I could not tell", and each REFUSES, naming the engine and
    the reason. A refused delete costs the user one command; a deleted model
    file is gone.

    The refusal is scoped, not blanket: with nothing loaded at all,
    nothing can be holding the file, so the answer is a certain None however
    unresolvable the paths are. Only a box that actually has a model resident
    can hit an uncertain refusal.

    Alias-aware, mirroring remove_model: while another registered name still
    points at the file, the removal keeps the bytes and only drops the name, so
    there is nothing to refuse. Pass *registry* to reuse a load the caller has
    already done.

    Does filesystem I/O (Path.resolve on registry paths, which can block on a
    UNC entry), so callers on the event loop run it in the executor.
    """
    from pathlib import Path

    from localm.config import load_registry
    from localm.model_manager import (
        _entry_path, find_aliases_by_path, resolve_deletion_target)

    reg = load_registry() if registry is None else registry
    entry = reg.get(model)
    if not isinstance(entry, dict):
        return None
    epath = _entry_path(entry)
    if epath is None:
        # A corrupt entry has no usable path, and remove_model's own
        # corrupt-entry branch drops the NAME without touching any file. No
        # deletion means nothing to guard.
        return None
    path = Path(epath)

    # Snapshot: _engines is mutated by loads and evictions on the event loop while
    # this runs in a worker thread, so iterating it live would raise.
    candidates = [(k, e) for k, e in list(_engines.items())
                  if getattr(e, "loaded", False)]
    # _engine is normally the same object as _engines[active], but a startup
    # (`localm serve <path.gguf>`) or test-injected engine can sit outside the map
    # while still holding the file.
    if (_engine is not None and getattr(_engine, "loaded", False)
            and not any(e is _engine for _, e in candidates)):
        candidates.append((getattr(_engine, "display_name", "") or model, _engine))
    if not candidates:
        # Nothing is resident, so nothing can be holding the file. Answered FIRST,
        # before any of the filesystem work below can fail and force a cautious
        # refusal, so a flaky network share never blocks tidying the library on a
        # box with no model loaded.
        return None

    # This probe stands ahead of BOTH filesystem users below, and the order is
    # load-bearing: find_aliases_by_path resolves too, and does not catch for the
    # path it was handed, only for sibling entries, so calling it first would turn
    # an unresolvable path into a 500 out of this guard.
    try:
        Path(path).resolve()
    except (OSError, ValueError):
        # resolve_deletion_target answers None for THREE different situations -
        # unresolvable, not under the models dir, already gone - and only this one
        # is an unknown. That collapse is correct for remove_model, which does not
        # delete in any of them, and wrong here, where None means go ahead.
        return ModelFileHold(
            candidates[0][0],
            "this model's registered file path could not be resolved")
    if any(a != model for a in find_aliases_by_path(path, reg)):
        return None
    target = resolve_deletion_target(path)
    if target is None:
        # Reached only for the two CERTAIN cases: the file is outside the models
        # dir, or it is already gone. Either way the removal drops the name and
        # deletes nothing.
        return None

    want = os.path.normcase(str(target))
    unknown: "ModelFileHold | None" = None
    for key, engine in candidates:
        mpath = getattr(engine, "model_path", None)
        if not mpath:
            unknown = unknown or ModelFileHold(
                key, "it has no recorded model file path")
            continue
        try:
            held = Path(mpath).resolve()
        except (OSError, ValueError):
            unknown = unknown or ModelFileHold(
                key, "its model file path could not be resolved")
            continue
        # normcase because the two paths reach here by different routes, one from
        # the registry and one from whatever the engine was constructed with, and
        # on Windows those can differ in case or separator.
        if os.path.normcase(str(held)) == want:
            return ModelFileHold(key)
    # A proven holder outranks an unknown one: both refuse, but only the first can
    # say which engine has the file. Hence the full pass rather than returning the
    # first unknown as soon as it is seen.
    return unknown


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
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
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

        # The _last_activity fallback is reached only for an engine assigned to
        # _engine directly without going through switch_engine's registration (a
        # test or script setting _engine, or a single-model direct-path startup
        # with an empty registry). In that mode there is only ONE model, so the
        # last activity of any request and the last activity of THIS model are the
        # same fact. A model loaded via switch_engine always has its own entry from
        # the instant it is registered, so the multi-model case never reaches this.
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
            # Flag for the duration of the native free, so a request that slips in
            # after the active_requests recheck above cannot take get_engine's fast
            # path and pin this engine while it is being freed. Cleared in finally,
            # so the engine kept in _engines reloads lazily on the next request.
            engine.unloading = True
            try:
                await loop.run_in_executor(None, engine.unload)
            finally:
                engine.unloading = False

            # Keep the now-unloaded Engine in _engines so the next request reloads
            # it lazily with its ORIGINAL constructor settings (n_ctx,
            # n_gpu_layers, device, mmproj), and so a direct-path served model,
            # whose display name is not in the registry and cannot be rebuilt by
            # the factory, is not lost. Only the LRU entry is dropped, since it
            # holds no VRAM while unloaded; its inference semaphore is kept for a
            # concurrent reload.
            if name in _engines_lru:
                _engines_lru.remove(name)

            if _engine is engine:
                _engine = None
            if _active_model_name == name:
                if _active_model_name:
                    # Same as unload_one_model and unload_all_models: capture the
                    # name before it is possibly cleared below, so a still-idle
                    # server with nothing left in _engines_lru can resolve an
                    # unnamed request (see _last_active_model_name and
                    # _resolve_unnamed_model_name). Harmless when the LRU fallback
                    # below keeps a real active model, which wins over
                    # _last_active_model_name automatically.
                    _last_active_model_name = _active_model_name
                _active_model_name = _engines_lru[-1] if _engines_lru else None
                _engine = _engines[_active_model_name] if _active_model_name else None
                _inference_sem = _inference_sems.get(_active_model_name) if _active_model_name else None
                
            from localm.debuglog import logger as _dbg
            _dbg.info("idle-unload: freed %s after %ds idle (ttl=%ds); it reloads "
                      "on the next request", engine.display_name, idle_s, ttl)
            unloaded_any = True
            # Cross-install GPU coordination: reflect the freed model. A no-op when
            # not registered. Offloaded because _gpu_registry_sync does filesystem
            # I/O and, when a non-zero main_gpu_index is set, a GPU probe via
            # _current_gpu_index.
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
    logged, not fatal: the entry ages until the next tick, and a stale entry is
    skipped by a peer's own liveness+identity check anyway."""
    while True:
        await asyncio.sleep(20)
        try:
            # Offloaded off the event loop: _gpu_registry_sync does filesystem I/O
            # for the registry write and, when a non-zero main_gpu_index is
            # configured, a GPU driver probe, either of which could stall the
            # single loop on this 20s tick.
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
                if _hb_monotonic is None:
                    # Cold start: the heartbeat task has not ticked even once, so
                    # there is no prior tick to measure a stall against and this
                    # poll is skipped rather than dumping against a fabricated
                    # baseline. Without the check, subtracting from None raises an
                    # uncaught TypeError on the very first poll and kills this
                    # daemon thread, which from outside is indistinguishable from a
                    # healthy quiet one.
                    continue
                lag = time.monotonic() - _hb_monotonic
                if lag < threshold:
                    continue
                now = time.monotonic()
                # Throttle: a long freeze yields a handful of snapshots rather
                # than one per second. Compared with `is not None` rather than a
                # 0.0 sentinel, since time.monotonic() is boot-relative and a real
                # 0.0 baseline would suppress the FIRST dump early in uptime.
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


# Human-facing hooks for the hang ALARM (localm/inference/_hang_alarm.py, the
# detect/surface/recover pipeline, distinct from the forensic stack-dump thread
# above). The GUI replaces these with the native status window's own error and
# ready transitions via set_hang_surface(); headless serve keeps the console
# default so the terminal running the server shows the state change. The alarm
# itself already emits logger.critical before calling these, so the hooks are
# only the human-visible half.
def _default_hang_surface(text: str) -> None:
    try:
        from localm.console import console
        console.print(f"[bold red]HANG ALARM:[/bold red] {text}")
    except Exception:
        pass


def _default_hang_recovered() -> None:
    try:
        from localm.console import console
        console.print("[green]Hang alarm cleared - server responding again.[/green]")
    except Exception:
        pass


_hang_surface_hooks: dict = {
    "surface": _default_hang_surface,
    "recovered": _default_hang_recovered,
}


def set_hang_surface(surface, recovered) -> None:
    """Route hang-alarm surfacing somewhere a user actually looks (the GUI
    wires the native status window here). Must be thread-safe callables: the
    alarm invokes them from its own daemon thread."""
    _hang_surface_hooks["surface"] = surface
    _hang_surface_hooks["recovered"] = recovered


# The running event loop, captured at lifespan startup for _hang_dump's
# async-task section, which a plain thread cannot resolve on its own.
_hang_dump_loop = None


def _hang_dump(reason: str) -> None:
    """Forensic stack snapshot on a NEW hang incident, into the same
    per-run trace file the stall watchdog uses - under the same privacy gate
    (_diagnostics_allowed), because it writes stack frames to disk. The alarm
    calls this once per distinct incident; detection/surfacing themselves are
    NOT gated on this (they write nothing sensitive anywhere).

    Two sections: every THREAD (faulthandler - where a wedged executor job
    is visible), then every asyncio TASK
    with its await stack - because a purely-async wedge (a coroutine parked
    on an await that never completes) does not exist on any thread and is
    invisible to faulthandler. The task section is gathered ON the loop
    (call_soon_threadsafe) with a short wait: in the starvation class the
    loop is healthy by definition so it returns instantly, and in the
    frozen-loop class it times out and is skipped - the thread section
    already shows the freeze itself."""
    if not _diagnostics_allowed():
        return
    import faulthandler
    from localm.debuglog import hang_trace_path
    with open(hang_trace_path(), "a", encoding="utf-8",
              errors="backslashreplace") as fh:
        fh.write(f"\n===== LOCALM HANG ALARM: {reason} (pid {os.getpid()}, "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
        faulthandler.dump_traceback(file=fh, all_threads=True)
        loop = _hang_dump_loop
        if loop is None or loop.is_closed():
            return
        lines: list = []
        done = threading.Event()

        def _capture_tasks() -> None:
            # Task.get_stack() reports only the OUTERMOST coroutine's suspension
            # frame and never follows cr_await, which for a wedged request shows
            # the middleware wrapper and hides the handler line. The await chain
            # is walked by hand instead, so the record ends at the exact line the
            # coroutine is parked on, plus the primitive it is awaiting.
            try:
                for task in asyncio.all_tasks(loop):
                    lines.append(f"\n--- task {task.get_name()} ---\n")
                    obj = task.get_coro()
                    depth = 0
                    while obj is not None and depth < 40:
                        frame = (getattr(obj, "cr_frame", None)
                                 or getattr(obj, "gi_frame", None)
                                 or getattr(obj, "ag_frame", None))
                        if frame is not None:
                            code = frame.f_code
                            lines.append('  File "%s", line %d, in %s\n' % (
                                code.co_filename, frame.f_lineno,
                                code.co_name))
                        nxt = (getattr(obj, "cr_await", None)
                               or getattr(obj, "gi_yieldfrom", None)
                               or getattr(obj, "ag_await", None))
                        if nxt is None and frame is None:
                            lines.append(f"  awaiting: {obj!r}\n")
                        obj = nxt
                        depth += 1
            except Exception:
                lines.append("    (task capture failed)\n")
            finally:
                done.set()

        try:
            loop.call_soon_threadsafe(_capture_tasks)
        except RuntimeError:
            return   # loop shutting down
        if done.wait(2.0):
            fh.write("\n----- asyncio tasks (await stacks) -----\n")
            fh.writelines(lines)
        else:
            fh.write("\n----- asyncio tasks: NOT CAPTURED (loop did not "
                     "respond in 2s - consistent with a frozen loop; see "
                     "the thread section above) -----\n")


def _hang_restart_action(app) -> None:
    """Recovery action for the hang alarm: the same in-place re-exec restart
    as the tray Restart button (_do_restart), hardened for a process that is
    currently misbehaving. The graceful path (engine unloads, embedder
    release, VRAM-release wait) is lock-free and bounded, but "bounded" is a
    claim about code on a box that is provably wedged, so it gets a hard
    window, after which the re-exec happens anyway with
    only the steps that cannot block: the lock-free embedder-worker release
    (an orphaned worker survives execv holding VRAM), the crash-marker disarm
    (so the next boot does not misreport this recovery as a crash), and a log
    flush."""
    port = getattr(app.state, "instance_port", None)
    instance_id = getattr(app.state, "instance_id", None)

    def _graceful() -> None:
        try:
            _do_restart(port=port, instance_id=instance_id)
        except Exception:
            _dbg_swallow("graceful restart during hang recovery failed; "
                         "forcing re-exec", level="warning")

    t = threading.Thread(target=_graceful, name="localm-hang-restart",
                         daemon=True)
    t.start()
    t.join(45.0)
    # _do_restart ends in os.execv, which never returns, so reaching this line at
    # all means the graceful path raised or wedged. Force it.
    from localm.debuglog import logger as _dbg
    _dbg.critical("graceful restart did not complete in 45s; forcing re-exec")
    try:
        from localm.inference import embedder as _embedder_mod
        _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during forced restart failed")
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard(instance_id=instance_id)
    except Exception:
        _dbg_swallow("crash-guard disarm during forced restart failed")
    try:
        from localm.debuglog import flush_log_handlers
        flush_log_handlers()
    except Exception:
        pass
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    os.execv(sys.executable, _restart_argv(port))


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

# The browser GUI authenticates with an HttpOnly session cookie whose value is an
# OPAQUE session id (localm.sessions), NOT the API key: page JS cannot read it,
# and rolling the key does not invalidate it. Cookie-sourced auth on a state
# change must also carry a CSRF token in this header, an HMAC DERIVED from the
# session (csrf_token_for / _csrf_ok) fetched via GET /api/session, so the two
# stay in lockstep and there is NO separate CSRF cookie. The
# Authorization-header path (CLI, SDK, coder) cannot be forged cross-site and is
# CSRF-exempt.
SESSION_COOKIE = "localm_session"
CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Hard cap on any request body, rejected with 413 from Content-Length BEFORE the
# body is buffered or parsed, so a large base64 upload cannot be materialized
# ahead of a route's own checks. 160 MB fits the largest legitimate upload (100 MB
# decoded, about 133 MB base64 plus the JSON wrapper). Read at request time, so a
# test can patch it.
MAX_REQUEST_BODY_BYTES = 160_000_000


class _BodyStreamCapMiddleware:
    """Enforce MAX_REQUEST_BODY_BYTES on the actual bytes received over the
    wire, not the client-supplied Content-Length header. ``Transfer-Encoding:
    chunked`` sends no Content-Length at all, so a plain header check (as a
    ``@app.middleware("http")``/BaseHTTPMiddleware handler would have to do)
    never fires: a chunked POST to a CORS-exempt route like
    /v1/chat/completions is otherwise fully buffered by FastAPI's own body
    handling, ahead of any auth dependency or pydantic validation, from one
    unauthenticated connection. A pure
    ASGI middleware class (not the BaseHTTPMiddleware pattern used elsewhere in
    this file) so it wraps the raw ``receive`` callable BEFORE Starlette/
    FastAPI's own body-buffering step ever runs; a BaseHTTPMiddleware handler
    that itself called ``request.body()`` would just reproduce the same
    unbounded read it is trying to bound.

    Once the cap is crossed this does NOT just raise and let the exception
    unwind through FastAPI's own body-parsing: FastAPI
    wraps ANY exception from body reading into a generic 400 "error parsing
    the body" - worse, raising from deep inside receive() surfaces to it as an
    ``ExceptionGroup`` (from the anyio task group `BaseHTTPMiddleware` runs the
    downstream app in), which doesn't match FastAPI's own
    ``except HTTPException: raise`` passthrough, so the 413 never reaches the
    client at all - only a bare TCP reset, because uvicorn still has unread
    bytes in the socket's receive buffer when it closes and the OS sends RST
    instead of completing the response. Instead: tell the inner
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
                    # Tell the inner app the body ends HERE, even though the real
                    # client may still be sending, so it cannot keep accumulating
                    # from this same wrapped receive.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def _suppressing_send(message):
            # The inner app believes it got a truncated body and will try to
            # respond to it; that response never reaches the real client, because
            # the 413 below is authoritative once exceeded.
            if not exceeded:
                await send(message)

        try:
            await self._app(scope, _capped_receive, _suppressing_send)
        except Exception:
            if not exceeded:
                raise
            # Suppressed: the inner app choked on a body this middleware cut short,
            # for instance invalid JSON at the truncation point. The request is
            # rejected as too large regardless of what it contained.

        if exceeded:
            # Drain the rest of the real stream, if the client had not already
            # finished sending it, so the OS does not RST the connection on close
            # over unread bytes still in its receive buffer, which would discard
            # the 413 below. Bytes are discarded immediately rather than
            # accumulated. Both the bytes drained AND the wall-clock time spent
            # draining are bounded, so a client that goes silent mid-stream cannot
            # block this loop's receive() forever; past either ceiling the graceful
            # drain is abandoned and a possible RST is accepted.
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
# has-the-client-gone poll. See the middleware and _generate_full for why an
# endpoint cannot just call request.is_disconnected().
_DISCONNECT_POLL_KEY = "localm.disconnect_poll"


class _DisconnectSignalMiddleware:
    """Publish a working client-disconnect poll for endpoints that need one.

    The four @app.middleware("http") handlers below are BaseHTTPMiddleware, which
    runs the endpoint in a child task fed by a SYNTHETIC receive that never yields
    http.disconnect - so request.is_disconnected() is permanently False for any
    endpoint behind them. A StreamingResponse still learns of a disconnect
    (Starlette acloses its body generator), but a plain non-streaming coroutine
    does not.

    This is a PURE-ASGI middleware (a BaseHTTPMiddleware here would defeat its own
    purpose) added OUTSIDE that stack, so it keeps the raw ASGI receive - which
    does carry http.disconnect - and stashes a Starlette-style non-blocking peek at
    scope[_DISCONNECT_POLL_KEY]. It never wraps receive/send, so it is transparent
    to every other route; only the non-streaming inference path polls it (see
    _generate_full)."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        gone = {"v": False}

        async def poll_disconnected() -> bool:
            # Mirrors Starlette.Request.is_disconnected: peek the raw stream with
            # an immediately-cancelled receive so it never blocks, and latch True
            # once http.disconnect has arrived. Reading the raw receive here is
            # safe because the body is fully read, through the wrapped chain,
            # before the endpoint and therefore this poll runs, so only
            # http.disconnect remains to consume.
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


# The session cookie PERSISTS, so the user stays signed in across a browser or PWA
# restart. Browsers clamp lifetime to about 400 days, which is the ceiling asked
# for here; the escape hatch is /api/session/logout.
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


def _session_minted_by_owner_key(rec, token=None) -> bool:
    """Whether *rec* was minted by the OWNER KEY itself, as opposed to a minted
    (and therefore revocable) keystore key.

    Asked POSITIVELY and answered from the owner key alone. Two ways, in cost
    order, and both are proofs rather than inferences:

    1. the ``owner_key_minted`` stamp recorded at login (``sessions.create``);
    2. for a record written before that field existed, the recorded ``key_hash``
       still equalling the live owner key's digest. Only the owner key's own
       digest can match that, so a keystore key's session can never satisfy it,
       and no ABSENT-vs-False distinction on the stamp is needed.

    On (2) the proof is written back (``remember_owner_key_minted``) while it
    still holds: after an owner-key roll the recorded hash matches neither the new
    owner key nor any keystore entry, so without the back-fill a pre-existing
    owner session would start failing the re-check below and be signed out on the
    roll - the exact behaviour the exemption exists to prevent.

    NOT derived from the scope set, and NOT from a keystore read:

    - **ADMIN is not the question.** The owner may mint ADMIN-scoped KEYSTORE
      keys, which stay revocable, so "holds ADMIN" is not "is the owner".
    - **Nothing here reads the keystore**, so ``_load_keystore()``'s fail-OPEN
      behaviour (``[]`` on OSError/ValueError) cannot promote anything, and no
      answer is derived from a NEGATIVE such as ``not key_hash_live``."""
    if rec.get("owner_key_minted") is True:
        return True
    from localm.auth import _hash_key, _legacy_owner_identity, ct_equal, get_api_key
    owner_key = get_api_key()
    if not owner_key:
        return False
    kh = rec.get("key_hash")
    if not kh:
        return False
    # The LEGACY unsalted digest counts too: the owner key's identity moved to a
    # salted KDF, and a session minted before that upgrade still carries the old
    # value until relink_key_hash rewrites it.
    if not (ct_equal(kh, _hash_key(owner_key))
            or ct_equal(kh, _legacy_owner_identity(owner_key))):
        return False
    if token:
        from localm import sessions
        sessions.remember_owner_key_minted(token)
    return True


def _valid_session(token):
    """The session record behind a presented cookie *token*, or None if it does not
    resolve to a session this server still honours.

    THE single gate for reading anything off a cookie session. It exists so that
    every consumer of a session attribute goes through the same re-validation
    rather than each calling ``sessions.lookup()`` and re-deciding: a bare lookup
    returns a record that this function would REJECT (a key's session whose key has
    since been revoked or expired), so a second reader written against ``lookup``
    would honour a session that auth already refuses everywhere else."""
    if not token:
        return None
    from localm import sessions
    rec = sessions.lookup(token)
    if rec is None:
        return None
    if not _session_exempt_from_key_recheck(rec, token):
        # A KEYSTORE key's session lives only as long as its key, so the owning
        # key is re-validated against the live keystore every request and revoking
        # or expiring it cuts the session off, matching the bearer path's
        # per-request verify().
        from localm.auth import key_hash_live
        if not key_hash_live(rec.get("key_hash")):
            return None
    return rec


def _session_exempt_from_key_recheck(rec, token=None) -> bool:
    """Whether *rec* may skip the per-request keystore liveness re-check.

    The exemption is for the OWNER KEY, not for ADMIN: the owner key is not a
    keystore entry and a session is decoupled from the key VALUE, so an owner-key
    ROLL must not log the owner out. Keying it on the SCOPE SET alone would hand
    the same exemption to any ADMIN-scoped KEYSTORE key, which is revocable by
    design, and removing the exemption outright signs the owner out on a roll.

    ADMIN is NECESSARY but not SUFFICIENT. It
    is not a return to inferring the owner from the scope set - the owner proof
    below is what actually grants the exemption. ADMIN is required ALONGSIDE it
    because a record claiming to be owner-minted while carrying narrower scopes is
    self-contradictory: every mint site records the owner key's own scope snapshot,
    which is ADMIN. No mint site can produce that combination, so only a tampered
    or corrupted store can, and requiring both means one flipped boolean is not
    enough to buy a session that can never be revoked."""
    if scopes.ADMIN not in set(rec.get("scopes", [])):
        return False
    if _session_minted_by_owner_key(rec, token):
        return True
    # No key identity at all. The re-check asks whether a REVOCABLE KEYSTORE
    # CREDENTIAL behind this session is still live; a session that records no key
    # has no such credential, so key_hash_live(None) would reject it for the wrong
    # reason.
    return not rec.get("key_hash")


def _principal_from_token(token, source):
    """Resolve a presented credential to ``(scopes, key_hash, fs_access,
    rag_roots)`` or None.

    A ``header`` token is a raw API key -> ``auth.verify()``. A ``cookie`` token is
    an OPAQUE SESSION ID -> the server-side session store (``localm.sessions``),
    which returns the scope / owning-key / fs-access / rag-roots SNAPSHOT taken at
    login, so a cookie session stays valid across an owner-key roll and the
    durable key never lives in the cookie. ``key_hash`` is the
    sha256 of the key that minted the session, so ``principal_id`` over a cookie
    matches the same key presented as a bearer (job ownership parity). ``rag_roots``
    is that credential's per-key RAG-indexing folder allowlist (see
    ``auth.rag_roots_for`` / ``effective_rag_roots``); empty means no per-key
    restriction, exactly like ``fs_access``'s "none" is not the ADMIN answer."""
    if not token:
        return None
    if source == "cookie":
        rec = _valid_session(token)
        if rec is None:
            return None
        return (set(rec.get("scopes", [])), rec.get("key_hash"),
                rec.get("fs_access", "none"), list(rec.get("rag_roots", []) or []))
    from localm.auth import _hash_key, fs_access_for, rag_roots_for, verify
    held = verify(token)
    if held is None:
        return None
    fs = "host" if scopes.ADMIN in held else fs_access_for(token, "none")
    rag_roots = [] if scopes.ADMIN in held else rag_roots_for(token, [])
    return held, _hash_key(token), fs, rag_roots


def caller_minted_by_owner_key(request: Request) -> bool:
    """True when this caller's COOKIE SESSION was minted by the owner key itself.

    Answers the one question a frozen ``key_hash`` cannot survive an owner-key roll
    to answer: was the credential behind this session the owner key, or a minted
    (and therefore revocable) keystore key? ``sessions.create`` records that as a
    POSITIVE proof at login - a constant-time plaintext compare against
    ``auth.get_api_key()`` - because after a roll the two are
    indistinguishable.

    Narrow, and every clause matters:

    - **False for a BEARER caller**, who has no session. That path already answers
      correctly by comparing the presented key's value, and ``verify()`` rejects a
      revoked or expired key first, so there is nothing here to add.
    - **Reads through ``_valid_session``**, so a scoped-key session whose key has
      been revoked or has expired is rejected before its record is ever consulted.
    - **Never consults the keystore itself**, so it cannot be flipped by a
      transient unreadable/corrupt ``auth.json``: ``_load_keystore()`` fails OPEN
      (returns ``[]``), and a privilege answer must never be derived from that.
    - **Never consults the scope set.** ADMIN is grantable to a keystore key, which
      stays revocable; only the recorded key-VALUE proof counts.

    This reports an attribute of an ALREADY-AUTHENTICATED session; it is not an
    authentication step and grants nothing on its own."""
    token, source = _request_token(request)
    if source != "cookie":
        return False
    rec = _valid_session(token)
    # Shares _session_minted_by_owner_key with the exemption gate rather than
    # re-reading the raw stamp, so is-this-the-owner's-session has exactly one
    # answer. Otherwise a pre-upgrade owner session, which has no stamp and is
    # recognised by value, would be exempt from the keystore re-check yet not
    # count as the owner here.
    return bool(rec) and _session_minted_by_owner_key(rec, token)


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
    session cookie. Used to bind a background job to the key that created it, so
    only that key (or an admin/owner) may stream or cancel it."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    if not token or not token.strip():
        return None
    # Route through _principal_from_token for BOTH sources: a cookie session for a
    # non-ADMIN key is re-validated against the live keystore the same way a
    # bearer token is on every request, so a revoked or expired key's still-
    # resident session cannot keep resolving a key hash here while the same cookie
    # is rejected everywhere else.
    prin = _principal_from_token(token, source)
    if prin is not None:
        held, key_hash, _fs, _rag_roots = prin
        return key_hash
    return None


def memory_principal(request: Request) -> Optional[str]:
    """The identity used to NAMESPACE this caller's chat memory. The owner (an
    ADMIN-scoped key or the owner session) collapses to the shared "owner"
    namespace (returns None -> memory.principal_of maps None to "owner"), so the
    owner's saved memories are not stranded in a per-key-hash namespace that a
    key rotation would orphan.

    This is NOT principal_id: principal_id keeps returning the key hash, so a
    background job stays bound to the key that created it. Only the memory
    principal collapses ADMIN/owner to "owner"; a non-owner scoped key keeps its
    own hash namespace here too."""
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
    Depends()-injectable gate, the same pattern require_scope uses for scope
    checks, so a new per-owner route cannot omit the check by construction.

    *resolve* is itself an ordinary FastAPI dependency - its own path/query
    params (e.g. ``job_id``, ``name``) are auto-injected the same as an
    endpoint function's - that returns ``(resource, owner, not_found_detail)``:
    *resource* is the object the route needs (or None if it does not exist),
    *owner* is its recorded creator (None = unrestricted, see job_owner_ok),
    and *not_found_detail* is the 404 message to raise. The SAME 404 is raised
    whether the resource is missing or the caller does not own it - never
    distinguished, so a foreign key cannot even confirm the resource exists.
    On success the gate returns *resource*, so a route can
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
    legacy key); no valid key -> "none". Filesystem reach is a per-credential
    dial, INDEPENDENT of ownership, so an owner can pair one of their own
    devices with a lower-reach key."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return "host"                       # open/dev mode = loopback owner
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        return "none"                       # keys configured, none/invalid presented
    held, _key_hash, fs, _rag_roots = prin
    if scopes.ADMIN in held:
        return "host"                       # owner key / owner session
    return fs                               # bearer key or session fs-access snapshot


def effective_rag_roots(request: Request) -> list:
    """The caller's effective per-key RAG-indexing folder allowlist: a list of
    folder-path strings, or ``[]`` meaning NO per-key restriction (the caller
    falls back to the global ``rag_allowed_roots`` policy that already applies to
    everyone - see ``rag.store.indexing_policy``/``confine_index_path``).

    Exactly the same shape as ``effective_fs_access``: open mode (loopback owner)
    and the owner/ADMIN key always resolve to ``[]`` (unrestricted - a per-key
    allowlist exists to confine a LESSER credential, never the owner's own); any
    other valid key uses its stored rag_roots list (default ``[]`` for a legacy
    key or one that never had one set); no valid key -> ``[]`` (the caller is
    then refused elsewhere in the request pipeline, same as a missing fs_access
    check - this function only ever narrows an already-authorized caller)."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return []                           # open/dev mode = loopback owner
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        return []
    held, _key_hash, _fs, rag_roots = prin
    if scopes.ADMIN in held:
        return []                           # owner key / owner session
    return rag_roots                        # bearer key or session snapshot


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


# Surface mounting: an on-demand GUI on a running instance.

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
        # real request cannot reach here without it; this guard makes a manual app
        # build fail loudly instead of dialling a port of None.
        raise HTTPException(500, "Instance not fully started (no bind port); "
                            "cannot mount the GUI surface yet.")
    # Follow the real bind: a server bound only on ::1, or on one specific
    # interface, has nothing listening on the IPv4 loopback, so a hardcoded
    # 127.0.0.1 self-call would dial an address that is not there.
    from localm.bindhost import self_connect_host, url_host
    _host = url_host(self_connect_host(getattr(app.state, "bind_host", None)))
    self_url = f"{scheme}://{_host}:{port}/v1"

    def active_model() -> str:
        return _engine.display_name if _engine is not None else ""

    def _build_engine(name: str) -> Engine:
        from localm.config import load_registry
        from localm.model_manager import get_model_info, get_model_mmproj
        info = get_model_info(name)
        if info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = info
        # Carry the model's mmproj (registry-recorded, else a sibling projector
        # next to the GGUF) into the new Engine, like the CLI --mmproj flag, so a
        # GUI or registry switch does not drop image support.
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
    # Claim the mount BEFORE attaching, so a re-entrant or concurrent call cannot
    # double-register the GUI routes; the flag rolls back if attach fails.
    app.state.gui_mounted = True
    try:
        manager = attach_gui(
            app, self_url=self_url, switch_model=switch_model, active_model=active_model)
    except Exception:
        app.state.gui_mounted = False
        raise
    # attach_gui re-affirms app.state.gui_mounted; the surface change is reflected
    # in discovery so /whoami and the registry report a full instance.
    app.state.coder_sessions = manager
    app.state.instance_mode = "full"
    app.openapi_schema = None   # force the schema to include the new routes
    try:
        from localm import instances
        from localm.config import home_dir
        instances.set_mode(home_dir(), getattr(app.state, "instance_id", ""), "full")
    except Exception as e:
        # The mount already succeeded; this registry sync, which makes discovery
        # advertise full rather than api, is best-effort but visible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("registry mode not updated to full: %s", e)
    return True


def _dbg_swallow(msg: str, *, level: str = "debug") -> None:
    """Log a swallowed best-effort failure at *level* (with the current exception's
    traceback) without ever raising. The nested-guard pattern already used at the
    update-watchdog site, factored out for the shutdown/restart teardown chain: a
    swallow stays discoverable yet can never itself break the stop/restart (the
    logging call is guarded too)."""
    try:
        from localm.debuglog import logger as _dbg
        getattr(_dbg, level, _dbg.debug)(msg, exc_info=True)
    except Exception:
        pass


def _do_shutdown(*, instance_id: Optional[str] = None) -> None:
    """The actual stop sequence. Unload the model FIRST so the native
    context is freed cleanly (a hard exit while it is loaded segfaults during
    teardown), clear the crash marker so this intentional stop is not reported as
    a crash, then exit the process so the stop is guaranteed (Ctrl+C sometimes
    does nothing). Separated from the route so it can be tested without exiting.

    *instance_id* (app.state.instance_id, set by instances.advertise()) scopes
    the crash-marker clear to THIS instance only - see bugreport.py's
    per-instance-scoping note; omitting it falls back to the legacy shared
    marker name rather than silently skipping the clear."""
    # Stop the child processes of any in-flight background job FIRST. A start_cli
    # job runs `python -m localm <cmd>` as a real child (a model pull, a runtime
    # provision, a ComfyUI setup): os._exit below bypasses atexit, the job worker
    # thread is a daemon so its finally may never run, and the Popen carries no
    # creationflags, so without this the child is ABANDONED. A child that writes
    # nothing to stdout survives and keeps working untracked; one that flushes
    # output dies at its next write on the broken pipe, mid-operation and with no
    # cleanup. See jobs.terminate_children_for_exit.
    #
    # FIRST in the sequence, before the engine and embedder teardown below,
    # because a media child can itself hold VRAM and any child can keep writing to
    # the data dir.
    #
    # The registry is left saying running rather than cancelled: the next start
    # reconciles those rows to interrupted.
    try:
        from localm.plugins.gui.jobs import terminate_children_for_exit
        _killed = terminate_children_for_exit()
        if _killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d in-flight job child process(es) on shutdown", _killed)
    except Exception:
        _dbg_swallow("terminating job child processes during shutdown failed "
                     "(non-fatal); a child may be left running")
    # Unload all engines in the multi-model dictionary
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            # Best-effort clean teardown before exit; a failed unload does not
            # block the stop, but is logged so a segfault-on-exit has a breadcrumb.
            _dbg_swallow("engine unload during shutdown failed (non-fatal)")
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            _dbg_swallow("engine unload during shutdown failed (non-fatal)")
    # Also release the shared embedder, which has a separate lifecycle from
    # _engines, so a full stop frees ALL resident VRAM and not just the chat
    # engines. Same swallow-but-log pattern as the engine unload above: shutdown
    # completes regardless, and a failure stays discoverable.
    try:
        from localm.inference import embedder as _embedder_mod
        # One lock-free call, and NOT active_requests() or
        # reset_embedder(): both take the embedder's load lock, which
        # get_embedder() holds for the FULL duration of an embedding-model load,
        # so a stop issued during a load would block here and never reach the
        # worker teardown. release_for_exit() decides without that lock: it
        # terminates a busy worker outright and closes an idle one politely. It
        # must run, because os._exit below bypasses atexit and multiprocessing's
        # daemon-child reclamation IS an atexit hook, so a skipped release leaves
        # the worker orphaned with its model resident in VRAM.
        _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during shutdown failed (non-fatal)")
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard(instance_id=instance_id)
    except Exception:
        # An uncleared crash marker makes this intentional stop report as a crash
        # on the next boot, so the failure is logged at WARNING rather than left
        # silent.
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
    auto-rolls back a perfectly healthy build.

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
    """Restart this server IN PLACE. Unload the model FIRST (clean native
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
    DETACHED health-check watchdog is spawned right before the re-exec;
    it polls the restarted instance's own /whoami and auto-rolls back if the
    expected version never comes up healthy within its timeout. The plain
    "restart the server" button (/v1/server/restart) calls this with no
    update_watchdog, so it is entirely unaffected.

    Free VRAM is captured BEFORE the teardown below, so the wait after it (once
    every native free has been issued) can confirm the releases landed before
    re-exec spawns a fresh worker into them - see that wait's own comment
    further down. Best-effort: an unmeasurable or wedged probe must not block a
    restart the user asked for, and vram_capacity() is deadline-bounded."""
    # Stop the child processes of any in-flight background job FIRST. A start_cli
    # job runs `python -m localm <cmd>` as a real child (a model pull, a runtime
    # provision, a ComfyUI setup): os.execv below bypasses atexit, the job worker
    # thread is a daemon so its finally may never run, and the Popen carries no
    # creationflags, so without this the child is ABANDONED. A child that writes
    # nothing to stdout survives and keeps working untracked; one that flushes
    # output dies at its next write on the broken pipe, mid-operation and with no
    # cleanup. See jobs.terminate_children_for_exit.
    #
    # FIRST in the sequence, before the engine and embedder teardown below,
    # because a media child can itself hold VRAM and any child can keep writing to
    # the data dir.
    #
    # The registry is left saying running rather than cancelled: the next start
    # reconciles those rows to interrupted.
    try:
        from localm.plugins.gui.jobs import terminate_children_for_exit
        _killed = terminate_children_for_exit()
        if _killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d in-flight job child process(es) on restart", _killed)
    except Exception:
        _dbg_swallow("terminating job child processes during restart failed "
                     "(non-fatal); a child may be left running")

    free_before = None
    try:
        from localm.discover import vram_capacity
        free_before = vram_capacity().get("free")
    except Exception:
        _dbg_swallow("free-VRAM read before restart failed (non-fatal)")

    # Unload all engines in the multi-model dictionary
    # had_engines asks whether anything is ACTUALLY loaded and worth waiting on,
    # not merely whether the dict is non-empty: unload_all_models and idle-unload
    # both KEEP a now-unloaded engine's entry in _engines so a later request
    # reloads it lazily, so a dict-non-emptiness check would make every restart on
    # a server that ever idle-unloaded a model pay the wait's full timeout.
    # getattr defaults to False, so a test double without .loaded is treated as
    # nothing to wait for.
    had_engines = any(getattr(e, "loaded", False) for e in _engines.values())
    if not had_engines and _engine is not None and _engine not in _engines.values():
        had_engines = bool(getattr(_engine, "loaded", False))
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            # Best-effort clean teardown before re-exec; a failed unload does not
            # block the restart, but is logged, since a hard re-exec while loaded
            # can segfault.
            _dbg_swallow("engine unload during restart failed (non-fatal)")
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            _dbg_swallow("engine unload during restart failed (non-fatal)")
    # Also release the shared embedder, which has a separate lifecycle from
    # _engines, so a restart frees ALL resident VRAM before re-exec and not just
    # the chat engines. Same swallow-but-log pattern as the engine unload above:
    # the restart proceeds regardless, and a failure stays discoverable.
    released_embedder = False
    try:
        from localm.inference import embedder as _embedder_mod
        # Lock-free release - see the matching comment in _do_shutdown above.
        # os.execv replaces this process image but does NOT touch the separate
        # worker child, and bypasses atexit, so without this the old worker
        # survives the restart holding VRAM while the restarted server spawns a
        # second one.
        released_embedder = _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during restart failed (non-fatal)")

    # Wait for the frees above to land before re-exec. The re-exec'd process
    # spawns a brand-new GGUF worker that constructs a fresh llama_context on
    # startup with no idea a restart just freed anything, so if the GPU driver has
    # not finished reclaiming the just-freed VRAM by then, that construction races
    # the still-reclaiming driver. switch_engine's in-process model swap already
    # waits on the identical native free. Skipped when nothing was actually
    # unloaded (a model-less restart), so the common case pays no delay.
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
        # makes this intentional restart look like a crash on the next boot.
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
        # Privacy mode opts out of ALL automatic disk traces, so the
        # crash-recovery breadcrumb dumps (ring buffer and pre_restart.log) are
        # skipped: they are session-derived INFO breadcrumbs written without the
        # user asking. keep_diagnostics overrides that, and _diagnostics_allowed()
        # folds both in and fails toward privacy when the mode or config cannot be
        # resolved.
        if _diagnostics_allowed():
            dump_ring_buffer()
            # Also write a plain text log for the bug reporter to ingest directly,
            # in case the JSON buffer fails to load back into memory.
            from localm.config import home_dir
            pre_log = home_dir() / "logs" / "pre_restart.log"
            pre_log.parent.mkdir(parents=True, exist_ok=True)
            pre_log.write_text("\n".join(recent_activity()), encoding="utf-8")
        # Flush all log handlers before os.execv so no buffered lines are lost.
        # This flushes already-open handlers and creates no new trace file, so it
        # is safe in privacy mode too.
        flush_log_handlers()
    except Exception:
        # Best-effort crash-recovery breadcrumbs (ring buffer and pre_restart.log)
        # plus the handler flush. A failure does not block the restart, but leaves
        # the next boot with fewer diagnostics, so it is logged.
        _dbg_swallow("pre-restart breadcrumb dump / log flush failed (non-fatal)")

    import os
    import sys

    # No inheritable fd (the debug-log FileHandler, the uvicorn listening socket)
    # may survive into the re-exec'd image, where the freshly loaded ggml/llama
    # runtime warns about failing to close child file descriptors. os.execv does
    # NOT close fds; marking them non-inheritable drops them at exec (POSIX
    # O_CLOEXEC, Windows handle inheritance). Best-effort per fd; stdin, stdout
    # and stderr are left inheritable so the new process keeps the console.
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
        # Spawned as the LAST step before execv: the watchdog's own timeout clock
        # starts when its process begins executing, so spawning late keeps that
        # clock closest to the actual restart moment.
        try:
            from localm import updater
            updater.spawn_health_watchdog(
                host=update_watchdog["host"], port=update_watchdog["port"],
                scheme=update_watchdog.get("scheme", "http"),
                expect_version=update_watchdog["expect_version"])
        except Exception as e:
            # spawn_health_watchdog() already never raises; this guard keeps a
            # malformed dict here from blocking the restart. Logged, not silenced.
            try:
                from localm.debuglog import logger as _dbg
                _dbg.warning("update watchdog not started: %s", e)
            except Exception:
                pass

    # The re-exec'd process must NOT auto-open a new browser tab: this is a
    # restart, not a fresh launch, and the tab the user already has shows a
    # reconnect overlay that polls and reloads itself in place once this process
    # is back up. plugins/gui/cli.py's browser-open step checks and consumes this
    # flag, so it never leaks into a later, genuinely fresh launch of the same
    # process tree.
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
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
    global _engine, _inference_sem, _engines, _engines_lru, _default_model_name, _active_model_name, _last_active_model_name, _inference_sems, _last_activity_per_model, _audit

    _engines.clear()
    _engines_lru.clear()
    _inference_sems.clear()
    _last_activity_per_model.clear()
    # A fresh app boot must not carry over a name remembered from a previous
    # create_app() call in the same process (test reuse, a restart).
    _last_active_model_name = None

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

    # Session-persistence mode for this server (privacy means no traces). One
    # audit log and transcript cover the server lifetime; GUI and API chat traffic
    # both flow through /v1/chat/completions and land here. _audit is a module
    # global, unlike _mode and _transcript which stay local and closure-shared
    # with the nested handlers below, because _do_restart is a separate top-level
    # function with no closure access into this frame.
    from localm.audit import effective_mode, make_audit_log, make_transcript
    _mode = effective_mode("server")
    _audit = make_audit_log(_mode, label="server")
    _transcript = make_transcript(_mode, label="server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem, _inference_sems, _active_model_name, _server_loop
        # Publish the running loop so off-loop worker threads (the jobs runner)
        # can route a shared-engine unload back onto it via
        # run_coroutine_threadsafe - see unload_one_model.
        _server_loop = asyncio.get_running_loop()
        if _active_model_name:
            _inference_sem = asyncio.Semaphore(1)
            _inference_sems[_active_model_name] = _inference_sem
        # Prune expired browser sessions once at startup, since create() prunes
        # only opportunistically. Best-effort: a sweep failure does not block
        # startup.
        try:
            from localm import sessions as _sessions
            _sessions.sweep()
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.debug("session sweep at startup failed (non-fatal)", exc_info=True)
        # Route an uncaught asyncio task exception through the bug reporter
        # instead of a silent never-retrieved warning. Skipped under pytest so the
        # test runner keeps its own loop handling.
        if "pytest" not in sys.modules:
            try:
                from localm import bugreport
                bugreport.install_asyncio_handler(asyncio.get_running_loop())
            except Exception:
                # The mechanism meant to surface silent asyncio-task failures must
                # not itself fail silently, so its absence is logged.
                from localm.debuglog import logger as _dbg
                _dbg.debug("asyncio exception handler not installed (non-fatal)",
                           exc_info=True)
        # Plugins register() before the loop exists, so loop-dependent plugin
        # work (the jobs scheduler) is queued on the manager and run now the loop
        # is up. attach_engine runs after create_app, so the manager resolves at
        # lifespan time.
        _pm = getattr(app.state, "plugin_manager", None)
        if _pm is not None:
            _pm.run_startup_callbacks()
        # Optional idle-unload background task (config idle_unload_seconds), a
        # cheap no-op while disabled. Cancelled on shutdown so it never outlives
        # the app.
        idle_task = asyncio.create_task(_idle_unload_loop())

        # The heartbeat (_hb_monotonic) and the watchdog's disk-writing stall dump
        # are on two separate gates. The heartbeat is pure in-memory bookkeeping -
        # _hang_heartbeat_loop writes nothing to disk, logs nothing and sends
        # nothing - so it carries none of the privacy considerations that gate the
        # stack-dump file below, and runs whenever pytest is not loaded,
        # independent of mode. THREE readers depend on it: the debug request log,
        # GET /debug/stacks, and the watchdog thread's own stall check. Without it
        # _hb_monotonic stays None and _loop_lag_seconds() reports a permanent
        # 0.00, indistinguishable from healthy. Best-effort: a startup failure does
        # not block serving.
        hb_task = None
        if "pytest" not in sys.modules:
            try:
                hb_task = asyncio.create_task(_hang_heartbeat_loop())
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("heartbeat task startup failed (continuing): %s", e)

        # The stack-dump-on-stall THREAD keeps its own privacy gate: on by default
        # in the log and full session modes, where the trace file is lazy and
        # created only on a real stall; in PRIVACY mode it stays OFF unless the
        # user opted into keeping diagnostics (_diagnostics_allowed) or forced it
        # on with LOCALM_HANG_WATCHDOG=1. LOCALM_HANG_WATCHDOG=0 opts out entirely.
        # Skipped under pytest so no thread lingers.
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
                hang_stop, hang_thread = _start_hang_watchdog(_hw_secs(), _hw_path())
                if _hw_verbose():
                    # Explicit opt-in extras: asyncio debug logs any single
                    # callback that hogs the loop past the threshold, naming the
                    # culprit. It adds per-callback overhead, so it is not part of
                    # the default-on path.
                    loop = asyncio.get_running_loop()
                    loop.set_debug(True)
                    loop.slow_callback_duration = 0.5
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("hang watchdog startup failed (continuing): %s", e)

        # Executor thread-pool saturation watch: a separate off-loop daemon
        # thread, always on. Unlike the hang watchdog beside it, it logs only pool
        # names and integer counts - no paths, no chat content - so it carries none
        # of the privacy-mode considerations that gate the stack-trace capture
        # above. Skipped under pytest so no thread lingers. Best-effort: a startup
        # failure does not block serving.
        sat_stop = sat_thread = None
        if "pytest" not in sys.modules:
            try:
                from localm.inference._executor_health import start_executor_saturation_watch
                # anyio's default thread pool, which
                # fastapi.concurrency.run_in_threadpool always uses, can only be
                # resolved from INSIDE a running event loop, so it is captured
                # once here and handed to the plain background thread below, which
                # cannot fetch it itself. A capture failure degrades to the anyio
                # pool reporting as unobservable, logged rather than claimed
                # healthy.
                anyio_limiter = None
                try:
                    import anyio.to_thread
                    anyio_limiter = anyio.to_thread.current_default_thread_limiter()
                except Exception as e:
                    from localm.debuglog import logger as _dbg
                    _dbg.debug("could not capture anyio's default thread "
                              "limiter (continuing, its pool will report as "
                              "unobservable): %s", e)
                sat_stop, sat_thread = start_executor_saturation_watch(
                    asyncio.get_running_loop(), anyio_limiter=anyio_limiter)
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("executor saturation watch startup failed "
                          "(continuing): %s", e)

        # Hang ALARM: detect a hung server, surface it where a user looks, and by
        # default auto-restart when the hang is provably a defect. Complements the
        # forensic stack-dump watchdog above rather than replacing it: that one is
        # privacy-gated because it writes stacks to disk, while this one writes
        # nothing sensitive anywhere and runs in every mode
        # (LOCALM_HANG_RECOVERY=off opts out). Skipped under pytest; tests drive
        # HangAlarm directly.
        hang_alarm = None
        if "pytest" not in sys.modules:
            try:
                from localm.debuglog import hang_watchdog_threshold
                from localm.inference import _hang_alarm as _ha

                def _alarm_probe_target():
                    port = getattr(app.state, "instance_port", None)
                    if not port:
                        return None
                    host = _ha._probe_host(getattr(app.state, "bind_host", None))
                    return host, int(port)

                _mode_now = _ha.recovery_mode()
                if _mode_now != "off":
                    global _hang_dump_loop
                    _hang_dump_loop = asyncio.get_running_loop()
                    hang_alarm = _ha.HangAlarm(
                        heartbeat_gap=lambda: (
                            None if _hb_monotonic is None
                            else time.monotonic() - _hb_monotonic),
                        inflight=_ha.tracker().observe,
                        probe_target=_alarm_probe_target,
                        surface=lambda text: _hang_surface_hooks["surface"](text),
                        recovered=lambda: _hang_surface_hooks["recovered"](),
                        restart=lambda reason: _hang_restart_action(app),
                        dump=_hang_dump,
                        surface_after=hang_watchdog_threshold(),
                        restart_after=_ha.restart_after_seconds(),
                        starvation_after=_ha.starvation_seconds(),
                        allow_restart=(_mode_now == "restart"),
                    ).start()
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("hang alarm startup failed (continuing): %s", e)

        # Cross-install GPU/VRAM coordination (see localm.gpu_registry): register
        # this instance in the machine-wide registry, but ONLY for a real,
        # non-isolated, advertise()'d server. instance_id, port and scheme are set
        # by advertise() before uvicorn accepts connections, so a bare
        # create_app() test app or an --isolated run never sets instance_id and
        # this is a no-op that never touches the shared directory. Best-effort: a
        # failure is logged and does not block startup.
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
                # Sweep entries left behind by an instance that crashed or was
                # killed without reaching its own shutdown cleanup below.
                from localm import gpu_registry
                gpu_registry.reap_stale(gpu_registry.registry_dir(),
                                        self_id=_instance_id)
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
            if hang_alarm is not None:
                hang_alarm.stop()
            if gpu_task is not None:
                gpu_task.cancel()
                try:
                    await gpu_task
                except asyncio.CancelledError:
                    pass
            if _gpu_coord is not None:
                # Best-effort: a crash just leaves the entry on disk. No live peer
                # trusts it (list_gpu_peers checks pid and identity), and the next
                # instance to start reaps it via gpu_registry.reap_stale above.
                try:
                    from localm import gpu_registry
                    gpu_registry.remove_entry(
                        gpu_registry.entry_path(gpu_registry.registry_dir(),
                                                _gpu_coord["instance_id"]))
                except Exception as e:
                    from localm.debuglog import logger as _dbg
                    _dbg.debug("gpu-registry cleanup on shutdown failed: %s", e)
                _gpu_coord = None
            # The loop is stopping, so stop advertising it: a late off-loop caller
            # then takes the safe no-loop path instead of a dead loop reference.
            _server_loop = None
            _audit.close()

    app = FastAPI(
        title="localm inference server",
        version="0.1.5rc3",
        lifespan=lifespan,
    )

    # One backstop so an unexpected error in ANY route returns a consistent JSON
    # 500 and is logged, instead of leaking a traceback or a bare body. A native
    # fault, such as a C-extension segfault, cannot be caught in-process; those
    # are prevented at the source and surfaced via the crash marker.
    @app.exception_handler(Exception)
    async def _unhandled_error(request, exc):  # noqa: ANN001 - framework signature
        from localm.debuglog import logger as _dbg
        _dbg.exception("unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error"})

    # Logs an HTTPException's detail next to its status, so a refusal is
    # diagnosable from the debug log rather than only from the client's response.
    #
    # Gated on debug_enabled() like the request and timing lines it sits beside,
    # NOT on debug_content_enabled(): an HTTPException detail is server-authored
    # operational text, never chat content, and nothing here reads the request
    # body.
    #
    # Registered for starlette's HTTPException (fastapi subclasses it and
    # registers the starlette class as its own key), then DELEGATED to fastapi's
    # own handler, so the response - status, body shape, and any
    # WWW-Authenticate or Retry-After headers - stays byte-identical. This is a
    # logging seam, not a response change.
    from starlette.exceptions import HTTPException as _StarletteHTTPException

    @app.exception_handler(_StarletteHTTPException)
    async def _log_http_exception(request, exc):  # noqa: ANN001 - framework signature
        from fastapi.exception_handlers import http_exception_handler
        from localm.debuglog import debug_enabled, logger as _dbg
        if debug_enabled():
            # Truncated: a detail can be long by design, and this names the cause
            # rather than mirroring the whole body into the log.
            detail = str(getattr(exc, "detail", "") or "")
            if len(detail) > 500:
                detail = detail[:500] + " ...[truncated]"
            _dbg.debug("%s %s refused %d: %s", request.method,
                       request.url.path, exc.status_code, detail)
        return await http_exception_handler(request, exc)

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):  # noqa: ANN001 - framework signature
        # A 422 body must stay serializable. pydantic records the offending value
        # under `input`, and when a client sends a NON-FINITE number (NaN,
        # Infinity) Starlette's JSONResponse serializes with allow_nan=False and
        # raises, turning a clean 422 into a 500. Non-finite floats are replaced
        # in the error detail so the 422 always renders.
        #
        # The 422 body must also stay BOUNDED. That same verbatim `input` puts a
        # deeply nested request body into the error object, and jsonable_encoder
        # walks it recursively, so a nested body raises RecursionError inside this
        # handler and the 422 becomes an opaque 500.
        import math

        from fastapi.encoders import jsonable_encoder

        _MAX_ERR_DEPTH = 20     # far past anything a real API request nests
        _ELIDED = "...[nested value elided]"

        def _depth_capped(v, depth: int = 0):
            """Prune the error object BEFORE `jsonable_encoder` ever sees it.

            The ORDER is the fix. Pruning afterwards cannot work, because the
            encoder is what recurses: it would blow the stack before any depth
            limit downstream of it got the chance to apply."""
            if depth >= _MAX_ERR_DEPTH:
                return _ELIDED
            if isinstance(v, dict):
                return {k: _depth_capped(x, depth + 1) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_depth_capped(x, depth + 1) for x in v]
            return v

        def _finite_safe(v, depth: int = 0):
            if depth >= _MAX_ERR_DEPTH:
                return _ELIDED
            if isinstance(v, float) and not math.isfinite(v):
                return repr(v)      # "nan" / "inf" / "-inf"
            if isinstance(v, dict):
                return {k: _finite_safe(x, depth + 1) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_finite_safe(x, depth + 1) for x in v]
            return v

        # _finite_safe still runs AFTER the encoder: the encoder itself can
        # PRODUCE a non-finite float (a Decimal NaN becomes float nan), so moving
        # it before would reopen the NaN 500. Only the DEPTH prune runs first.
        try:
            safe = _finite_safe(jsonable_encoder(_depth_capped(exc.errors())))
        except RecursionError:
            # SAFETY NET, not the fix: it catches a shape the prune above cannot
            # reach (a deeply nested object that is not a dict, list or tuple,
            # which passes through untouched and the encoder then recurses into).
            # It does not remove the CPU cost, so it is not a substitute for the
            # prune. Logged rather than silent: firing means a shape got past the
            # prune.
            from localm.debuglog import logger as _dbg
            _dbg.warning("validation error for %s was too deeply nested to "
                         "encode even after depth-capping; returned a 422 "
                         "without the structured detail", request.url.path)
            safe = []

        def _field(err) -> str:
            # Drop the body/query container so the user sees the name they
            # actually typed, not pydantic's full path.
            parts = [str(p) for p in (err.get("loc") or ())
                     if p not in ("body", "query", "path", "header")]
            return ".".join(parts) or "request"

        def _one(err) -> str:
            name = _field(err)
            msg = (err.get("msg") or "is invalid").strip()
            # pydantic phrases these as Input should be X, which reads badly once
            # the field name is prepended, so it is rewritten to must be X.
            if msg.lower().startswith("input should be "):
                msg = "must be " + msg[len("input should be "):]
            elif msg[:1].isupper():
                msg = msg[0].lower() + msg[1:]
            # On a MISSING field pydantic reports the whole request body as
            # `input`, so it is not echoed back.
            got = ("" if err.get("type") == "missing" or "input" not in err
                   else f" (got {err.get('input')!r})")
            # max_tokens=0 gets its own sentence: it looks like a legitimate no
            # limit, and 0 collides with the engine's internal unlimited sentinel,
            # which is not guessable from a greater-than-or-equal-to-1 message.
            if name == "max_tokens" and err.get("input") in (0, "0"):
                return ("max_tokens must be 1 or more - 0 is not 'no limit'. "
                        "Omit max_tokens entirely to use the model's default")
            return f"{name} {msg}{got}"

        # `detail` is the human sentence, because clients render
        # data.detail or the status text, and stringifying pydantic's error LIST
        # there produces an unreadable dump. The structured form is preserved
        # verbatim under `errors` for anything that parses it.
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

    # Per-process shell token: in open mode the management routes require this
    # token, which the loopback GUI shell injects into the SPA (web.py
    # _gui_index). It gates the no-Origin local-client path that bearer auth and
    # the Origin guard do not cover. Per-process, so it dies on restart, and never
    # persisted.
    app.state.shell_token = secrets.token_urlsafe(32)

    # Per-process CSRF secret. The CSRF token is a deterministic HMAC of the
    # session id below, so it is present exactly when the session is and cannot
    # desync from it. Per-process, so it dies on restart and the client re-fetches
    # from /api/session; never persisted.
    app.state.csrf_secret = secrets.token_urlsafe(32)

    # api-mode landing: a bare `localm serve` has no GUI shell, so GET / would
    # 404. It is redirected to the auto-generated API docs. Only on the api path,
    # so it never collides with the GUI's own / handler and StaticFiles catch-all.
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
            # loop_lag is the real scheduling delay (_loop_lag_seconds), about 0
            # on a healthy server and positive only when a preceding event-loop
            # stall pushed the last heartbeat tick late. It is NOT
            # time-since-last-tick. A stall shorter than _HEARTBEAT_INTERVAL_S
            # also reads 0.0, which means no stall LONGER than the interval, not
            # no stall at all. None, before the heartbeat's first tick, renders as
            # n/a rather than 0.0.
            lag = _loop_lag_seconds()
            lag_str = f"{lag:.2f}s" if lag is not None else "n/a"
            _dbg.debug(
                "%s %s -> %d (%.0f ms, loop_lag=%s)",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
                lag_str,
            )
            return response

    # Loopback-only debug endpoint: every thread's stack plus the asyncio task
    # list. 404s off loopback. Served ON the event loop, so it answers only while
    # the loop is alive (a partial stall, a task backlog); a fully wedged loop is
    # captured by the off-loop watchdog file (LOCALM_HANG_WATCHDOG) instead.
    #
    # THREE gates:
    #   1. Depends(require_fs_host) - meaningful in PROTECTED mode (a key's
    #      fs_access dial); in DEFAULT KEYLESS mode effective_fs_access returns
    #      host for EVERY caller, so on its own it is a tautology.
    #   2. the open-mode shell-token gate, via _SHELL_TOKEN_GETS below, which is
    #      what makes gate 1 non-vacuous in keyless mode.
    #   3. the bind_host loopback check below, on app.state.bind_host (what the
    #      server actually BOUND to) and never request.client.host: behind portmux
    #      the request peer is always 127.0.0.1, so the peer address cannot
    #      distinguish a loopback client from a LAN one.
    # Frame text is path-scrubbed on the way out.
    @app.get("/debug/stacks", include_in_schema=False,
             dependencies=[Depends(require_fs_host)])
    async def _debug_stacks(request: Request):
        host = getattr(request.app.state, "bind_host", "127.0.0.1")
        if not _is_loopback_host(host):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        import traceback

        from localm.pathscrub import path_scrubber
        # traceback.format_stack emits absolute paths, which name the install dir
        # and, on a per-user install, the OS account. Only the DIRECTORY is
        # redacted: the file name, line number, function and source line all
        # survive. The scrubber is bound once rather than per string, since a dump
        # is hundreds of frames and each prefix resolve is a filesystem call.
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
        # Thread-pool saturation numbers, served here as well as in the periodic
        # warning log, so a diagnosis in progress does not have to wait for the
        # threshold to trip a line. This handler is async with a running loop, so
        # it fetches anyio's default thread limiter fresh on every call.
        try:
            from localm.inference._executor_health import executors_snapshot
            import anyio.to_thread
            executors = executors_snapshot(
                asyncio.get_running_loop(),
                anyio_limiter=anyio.to_thread.current_default_thread_limiter())
        except Exception:
            executors = {}
        # None, before the heartbeat's first tick, renders as JSON null rather
        # than 0.0.
        lag = _loop_lag_seconds()
        return {"pid": os.getpid(),
                "loop_lag_s": round(lag, 2) if lag is not None else None,
                "threads": threads, "tasks": tasks, "executors": executors}

    # CORS: localhost-only by default. A wildcard would let any website the user
    # visits call this API from browser JS and read the responses. Override with
    # config cors_origins, either a list of origins or a wildcard.
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
    # origin, so every unsafe-method request must be same-origin (or a configured
    # CORS origin), EXCEPT the OpenAI-compatible inference API, which stays
    # cross-origin callable for local apps. Allowlist-by-default, so a new plugin
    # route is protected the moment it is added. Non-browser clients (CLI, SDK)
    # send no Origin; cors_origins set to a wildcard opts out entirely.
    _UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    # Every entry below is a full route path, not a directory prefix, and is
    # matched with str.startswith(), so a prefix entry would also exempt every
    # FUTURE route added under it. Keep in sync with _BESPOKE_GATED_ROUTES in the
    # kernel route scope contract.
    _CROSS_ORIGIN_OK = (
        "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
        # Surface management (on-demand GUI mount) is driven by a local process,
        # the attaching `localm gui`, not the browser shell: no Origin, no
        # shell_token. The route does its OWN strict auth (this instance's attach
        # token, or an owner API key), which is the real credential, so it is
        # exempt. A cross-origin page cannot set Authorization without a secret it
        # cannot read. Also listed in _BESPOKE_GATED_ROUTES; keep both in sync.
        "/v1/surfaces/gui",
        # Multi-instance GPU coordination (localm.gpu_registry): a SIBLING localm
        # instance calls this loopback-only, like surface management above - no
        # Origin, no shell_token, since it is a different process. Its own
        # coordination_token, never the API key or shell token, is the real
        # credential and is checked in the route. Also listed in
        # _BESPOKE_GATED_ROUTES; keep both in sync.
        "/v1/instances/cooperate-unload",
    )
    _cors_allowlist = frozenset(cors_cfg) if isinstance(cors_cfg, list) else frozenset()
    _cors_wildcard = cors_cfg == "*"

    # A short list of UNAUTHENTICATED GETs that disclose host detail and, unlike
    # the /api and /v1 metadata reads below, have NO route-level auth to fall back
    # on. The default CORS policy hands an ACAO to any http(s)://localhost:PORT
    # origin: /whoami discloses root_dir, an absolute path, on a loopback bind,
    # and /debug/stacks discloses thread stacks. They sit OUTSIDE the /api and /v1
    # metadata-GET gate, so they are refused here instead, cross-origin, in EVERY
    # mode.
    _CROSS_ORIGIN_GET_REFUSED = ("/whoami", "/debug/stacks")

    # Same refusal, matched by PREFIX rather than exact path. /api/fs/* is the
    # host filesystem browser: it enumerates the user's disk, and it is a GET, so
    # it is exempt from both the CSRF gate (unsafe methods only) and the open-mode
    # shell-token gate. The generic /api and /v1 metadata-GET gate below covers it
    # only inside the not-any_key_configured branch, so in PROTECTED mode a
    # cookie-authenticated cross-origin GET would execute. SameSite=strict does
    # not close it either: site ignores port, so any other page on a loopback port
    # is same-site. Refused in EVERY mode instead.
    _CROSS_ORIGIN_GET_REFUSED_PREFIXES = ("/api/fs/",)
    # Sensitive GETs that sit OUTSIDE the /api and /v1 prefixes the open-mode
    # shell-token gate below keys on. /debug/stacks' own Depends(require_fs_host)
    # is a TAUTOLOGY in keyless mode, since effective_fs_access returns host for
    # everyone when no key is configured, so listing it here routes it through the
    # same shell-token and cross-origin check as a management read. /whoami is
    # NOT here: the GUI shell calls it to discover whether it needs a
    # key at all, so requiring the token to read it would be circular; its
    # disclosure is handled by the cross-origin refusal above. Enforced only on a
    # LOOPBACK bind - see the comment at token_gated_get below.
    _SHELL_TOKEN_GETS = ("/debug/stacks",)

    def _cross_origin_refused(request) -> bool:
        """True when this request carries an Origin header that is neither
        same-origin nor CORS-allow-listed. Shared by the CSRF check (unsafe
        methods) and the open-mode shell-token gate: the default
        CORS policy lets any http(s)://localhost:PORT / 127.0.0.1:PORT origin
        READ a matching response, so a hostile local page can steal the shell
        token from a plain cross-origin ``GET /`` and replay it - token
        possession alone does not prove the caller IS the loopback GUI shell.
        "cors_origins": "*" opts OUT of this specific check, the same as for the
        CSRF check; it does not waive the shell-token requirement itself."""
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
        # Cross-origin refusal (every mode): every state-changing method, plus the
        # sensitive GETs in _CROSS_ORIGIN_GET_REFUSED (exact) and
        # _CROSS_ORIGIN_GET_REFUSED_PREFIXES (prefix). All are subject to the same
        # same-origin / CORS-allowlist check.
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
            # Open-mode management gate. With no key configured, management routes
            # still require the per-process shell token, injected into the loopback
            # GUI shell, so a no-Origin local client (curl, a script) cannot mint a
            # key, flip config, install a plugin, load a model or browse the
            # filesystem unauthenticated. Protected mode (a key exists) is
            # bearer-auth'd on the route. The token is required even for an
            # allowlisted CORS origin, since an Origin header is forgeable and is
            # therefore not a management credential.
        is_unsafe = request.method in _UNSAFE_METHODS
        # A _SHELL_TOKEN_GETS path is token-gated only where it is actually
        # SERVED. /debug/stacks 404s off a loopback bind, and returning 403 there
        # instead would tell an unauthenticated NETWORK caller that the endpoint
        # exists, since an unknown path under /debug/ 404s. Off loopback this
        # falls through to the handler's own 404, and the handler does its loopback
        # check before computing anything.
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
                # A keyless LOCAL process (localm status, the MCP
                # server_activity tool) has no way to obtain shell_token, which
                # is per-process, never persisted, and only injected into the
                # browser-served SPA. It does have this instance's own attach
                # token from the per-instance registry file (0600, owner-only,
                # read via instances.attach_target/snapshot), so that is accepted
                # here too. A browser has no filesystem access to that file and
                # can never present it, unlike shell_token, which the
                # cross-origin check below guards.
                #
                # This gate covers every open-mode management route (minting a
                # key, changing config, unloading a model), so either token
                # authorizes all of them, not just activity reads.
                inst_token = getattr(request.app.state, "instance_token", None)
                presented = _bearer_token(request)
                token_ok = ct_equal(presented, token) or (
                    bool(inst_token) and ct_equal(presented, inst_token))
                # An unsafe-method request already passed the same-origin check
                # above (or is exempt as _CROSS_ORIGIN_OK, which never reaches
                # here); a metadata GET never went through that block, so it must
                # pass the identical check here, or a token stolen via CORS is
                # replayable cross-origin against every /api/* and /v1/* read.
                # Applies to both token kinds: a real CLI/MCP client sends no
                # Origin header at all, so it costs the legitimate case nothing.
                cross_origin = is_metadata_get and _cross_origin_refused(request)
                if not token_ok or cross_origin:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Open-mode management requires the "
                                 "localm GUI shell on this machine, or an API key "
                                 "(run 'localm key generate')."},
                    )
        return await call_next(request)

    # Security response headers. nosniff is enforced everywhere and blocks a
    # MIME-sniff into executable HTML. The CSP enforces rather than reports, so
    # DOMPurify is not the only XSS barrier on the shell.
    #
    # script-src carries a PER-REQUEST nonce rather than unsafe-inline, so the
    # shell's own inline scripts run and an injected one cannot. Adding
    # unsafe-inline alongside would do nothing: a policy containing a nonce makes
    # browsers IGNORE unsafe-inline entirely.
    #
    # style-src keeps unsafe-inline, and the nonce CANNOT replace it. CSP3's
    # is-element-nonceable algorithm covers script, style and link ELEMENTS only;
    # an inline style ATTRIBUTE is reachable only by unsafe-inline, or by
    # unsafe-hashes plus a hash per distinct attribute value, and index.html
    # carries 52 such attributes. Under a strict style-src, 32 elements that must
    # start hidden render visible. KaTeX is unaffected, styling via CSSOM, which
    # CSP does not govern, and so are the app's own el.style writes. The cost is
    # that DOMPurify passes a model-authored style ATTRIBUTE through, so a reply
    # can restyle its own subtree; img-src and connect-src still deny the CSS
    # url() exfiltration path.
    #
    # form-action none because nothing in this GUI submits a form: there is no
    # form element in static/, and every mutation goes through fetch().
    # form-action is a NAVIGATION directive with no default-src fallback, so
    # omitting it allows submission ANYWHERE, and DOMPurify's default
    # ALLOWED_TAGS includes form, so a model reply rendering one survives
    # sanitisation with its remote action intact and no CSP violation raised.
    # none rather than self, since there is no legitimate same-origin submission
    # either, which also closes the same-origin CSRF shape against localm's own
    # /api.
    #
    # NO CDN ORIGIN IS LISTED, AND NOTHING NEEDS ONE. The tts plugin's Kokoro
    # bundle pulls the onnxruntime-web backend with a dynamic import(), which is
    # a MODULE SCRIPT and so governed by script-src rather than connect-src. That
    # runtime is vendored and served from this origin
    # (localm/plugins/builtin/tts/static/vendor/onnxruntime/, pointed at by the
    # plugin's own wasm_paths default). Do not add a CDN origin back to make a
    # TTS load error go away: that error means the vendored runtime did not
    # resolve.
    #
    # The wasm-unsafe-eval token is REQUIRED: compiling any WebAssembly needs an
    # explicit grant, and onnxruntime-web is WebAssembly on both its wasm and
    # webgpu paths, so without it no backend can start at all. It permits
    # WebAssembly compilation only and NOT dynamic evaluation of JavaScript, so it
    # is tighter than the broader token the browser error text names. Do not
    # widen it.
    #
    # blob: in script-src is REQUIRED once the page is cross-origin isolated:
    # isolation gives onnxruntime-web SharedArrayBuffer, so it switches to its
    # THREADED build and loads its worker as a blob: module. worker-src with
    # blob: is NOT sufficient, because the dynamic import of that blob module is
    # governed by script-src. A blob: URL can only be minted by same-origin script
    # that is already executing, and the nonce still gates what may execute.
    _CSP_PREFIX = ("default-src 'self'; "
                   "script-src 'self' blob: 'wasm-unsafe-eval' 'nonce-")
    _CSP_SUFFIX = (
        "'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        # blob: here for the same reason it is in img-src, script-src and
        # worker-src: the GUI mints these with its OWN URL.createObjectURL and
        # reads them back (fetch for send-to-chat and copy-image, and the video
        # and audio players). A blob: URL is same-origin-scoped and cannot be
        # pointed at a remote origin, so it grants no exfiltration path.
        #
        # media-src MUST be spelled out. Without it media falls back to
        # default-src, which has no blob:, and the failure is SILENT: assigning a
        # blocked src fires an error EVENT on the element rather than throwing, so
        # a try/except around it cannot see it and the player sits dead.
        # huggingface.co and the hf.co subdomains are the MODEL weights: chat
        # models are server-side, but the tts plugin fetches Kokoro's ONNX in the
        # browser and caches it there. The onnxruntime runtime is vendored and
        # same-origin.
        "connect-src 'self' blob: https://huggingface.co https://*.hf.co; "
        "media-src 'self' blob:; "
        "worker-src 'self' blob:; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        # See the form-action note above: a NAVIGATION directive with no
        # default-src fallback. Nothing in the GUI submits a form.
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        # The nonce is minted BEFORE call_next, because the shell route stamps
        # this exact value onto its inline script tags while it builds the body,
        # so the value must already exist when the handler runs. A header set
        # afterwards from a value the handler never saw would match nothing.
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault(
            "Content-Security-Policy", _CSP_PREFIX + nonce + _CSP_SUFFIX)
        # CROSS-ORIGIN ISOLATION, so onnxruntime-web can use more than one thread.
        # Without both of these the document is not isolated, SharedArrayBuffer is
        # unavailable, and onnxruntime falls back to numThreads=1, at which
        # synthesis runs slower than playback and a long reply stutters.
        #
        # credentialless rather than require-corp: require-corp demands a CORP
        # header on EVERY cross-origin subresource, which localm does not control
        # for huggingface.co. credentialless sends those requests WITHOUT
        # credentials, which is sufficient for isolation and correct here, since
        # none of localm's cross-origin fetches are authenticated.
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        resp.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
        return resp

    # API-surface disclosure guard. FastAPI's built-in docs (/docs, /redoc,
    # /openapi.json) enumerate every route and schema, so they are served only on
    # a loopback bind, keyed on the CONFIGURED bind host and never the peer, since
    # portmux makes the peer always look loopback. 404 rather than 403, so they
    # simply do not exist off-loopback. bind_host unset (tests, standalone mount)
    # defaults to loopback, so docs stay available.
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
    # Outermost of all: in-flight and progress bookkeeping for the hang alarm's
    # starvation detector. Pure ASGI and content-free (counts and clocks only),
    # so a request wedged in ANY inner layer still shows as in flight.
    from localm.inference._hang_alarm import RequestProgressMiddleware
    app.add_middleware(RequestProgressMiddleware)

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
    was generated. EXCLUDES model load + prompt prefill (the span before the
    first token), which is reported on its own as ttft_ms, so a cold start's
    multi-second load is never charged against the generation rate."""
    if first_token_at is None:
        return None
    return gen_end - first_token_at


# Floor on plausible per-token decode time, used by _tokens_per_sec to reject an
# implausible decode window rather than report a nonsensical rate.
# first_token_at is a SINGLE sample, so a GPU scheduler that delays token 1 and
# then delivers the rest in an uncontended burst collapses the measured window
# toward zero even though every timestamp is real. 1ms per token, a 1000 tok/s
# ceiling, is above what single-stream memory-bandwidth-bound decode plus this
# architecture's per-token IPC marshaling can sustain for one request.
_MIN_SEC_PER_TOKEN = 0.001


def _tokens_per_sec(completion_tokens: int, decode_elapsed: Optional[float]) -> Optional[float]:
    """Decode throughput = generated tokens over the DECODE window only (see
    _decode_elapsed), NOT over total wall time: folding the model-load/prefill
    span into this rate makes the first call after a load report a value orders
    of magnitude too low, which reads as a silent CPU fallback.
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
        # In log/full mode a failed write drops the record, so it is logged
        # rather than left invisible.
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
    engine out from under an in-flight request."""
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
    PluginManager.inference_engine, which is resolved fresh at every use site
    across several plugins, so inspection-only reads are common; only the call
    that actually drives the model registers as "in use".

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
        # Starlette acloses THIS wrapper on a client disconnect, which is how the
        # pin below is released. The inner stream is aclosed explicitly too, so
        # its own cancel/finally runs now and releases the per-model
        # _inference_lock the producer thread holds, rather than one
        # async-generator GC tick later. A no-op on a clean finish.
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
    prompt_tokens: Optional[int] = None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    from localm.inference.protocol import ChoiceDelta, StreamChoice
    from localm.textnorm import ThinkSplitter

    chunk_id = make_chunk_id()
    ts = int(time.time())
    think = ThinkSplitter()   # routes <think> reasoning into delta.reasoning_content

    if prompt_tokens is None:
        prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

        # Context-limit handling: compact_messages when close to the limit,
        # reserving a 2048-token buffer for compaction and response generation.
        capacity = engine.context_capacity()
        if isinstance(capacity, int) and capacity > 0 and len(messages) > 3:
            buffer = max(2048, int(capacity * 0.10))
            if capacity - prompt_tokens < buffer:
                from localm.inference.compact import compact_messages
                def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                    return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
                # Off the event loop: compact_messages runs a FULL summarization
                # generation and engine.chat_stream holds the per-model inference
                # lock for up to about 1024 tokens, which on the single-threaded
                # loop would freeze every other request, the heartbeat and the
                # disconnect watchers for its whole duration.
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
    # async generator. cancel_event lets that unwind stop the producer thread
    # below, which would otherwise drive engine.chat_stream() to end-of-generation
    # holding llama.py's per-model _inference_lock and blocking the next request
    # to THIS model.
    cancel_event = threading.Event()

    def _generate():
        # engine.chat_stream is called INSIDE the try: it is not a generator, and
        # eagerly runs the auto-reload (backend.load()) and load_config() before
        # returning the token generator, so it can RAISE here (a reload OOM, a
        # since-removed GGUF). A raise escaping the try would kill the thread
        # before the finally enqueued the sentinel, leaving the consumer blocked
        # forever at token_queue.get() while holding the per-model semaphore.
        gen = None
        try:
            gen = engine.chat_stream(messages, **gen_kwargs)
            for token in gen:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Logged with a full traceback to the debug log and surfaced to the
            # client: a silent thread death looks like an empty reply. NOT
            # traceback.print_exc(): _dbg.exception already records the trace, and
            # an expected condition such as outgrowing n_ctx_max reaches the user
            # as a clean message.
            from localm.debuglog import logger as _dbg
            _dbg.exception("generation thread failed")
            loop.call_soon_threadsafe(
                token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            # Close the generator chain from THIS thread, which is suspended at
            # its yield right now; closing it from the event-loop thread would
            # race the in-flight next() and raise generator-already-executing.
            # close() propagates GeneratorExit down through the backend wrappers
            # into llama.py _generate, whose `with self._inference_lock` then
            # exits and frees the lock deterministically. gen is None when
            # chat_stream raised eagerly.
            try:
                if gen is not None:
                    gen.close()
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("closing generation stream failed")
            # Wake the consumer. If the loop is already gone (server shutdown, or
            # a disconnect whose request-loop has since closed) the consumer is
            # gone too, so the sentinel is dropped rather than surfacing as an
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
            # Signal the producer to stop. A no-op on a clean finish: the thread
            # already exited after _DONE, so t.join() below returns at once. On a
            # disconnect (GeneratorExit raised at the yield above) it makes the
            # thread break its loop, close the generator chain and release
            # _inference_lock. GeneratorExit then keeps propagating, so t.join()
            # is skipped and the daemon thread self-terminates within about one
            # token.
            cancel_event.set()

        t.join()
        gen_end = time.perf_counter()
        # Release any tail held back while disambiguating a partial <think> tag.
        for data in _reason_sse(*think.flush(), model_id, chunk_id, ts):
            yield data

    if gen_error is not None:
        err_chunk = ChatChunk.token(
            inference_error_text(gen_error), model_id, chunk_id, ts)
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
    # A mid-stream error reports finish_reason error rather than a clean stop,
    # so a programmatic client can detect the failure; the error text itself was
    # already streamed.
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
    prompt_tokens: Optional[int] = None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    chunk_id = make_chunk_id()
    ts = int(time.time())
    # *messages* arrive already inlet-transformed; count tokens on what
    # inference sees (matches the chat path) if not already provided.
    if prompt_tokens is None:
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
        # orphan the consumer.
        gen = None
        try:
            gen = engine.chat_stream(messages, **gen_kwargs)
            for token in gen:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Surface an inference failure to the client instead of letting this
            # daemon thread die: an uncaught death fires a crash report and looks
            # like an empty reply. _dbg.exception logs the full trace.
            from localm.debuglog import logger as _dbg
            _dbg.exception("completion generation thread failed")
            loop.call_soon_threadsafe(token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            # Close the generator chain from this (suspended) thread so a cancel
            # propagates GeneratorExit into llama.py _generate and frees
            # _inference_lock. gen is None when chat_stream raised eagerly.
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
            "choices": [{"text": inference_error_text(gen_error),
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
    # A mid-stream error is reported as error, not stop, so a client keying off
    # finish_reason detects the failure even though the error text was already
    # streamed as a visible chunk.
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
    the _stream_sse cancel path.

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
            # Close from THIS (suspended) worker thread so GeneratorExit
            # propagates through the backend wrappers into llama.py _generate,
            # whose `with self._inference_lock` then exits and frees the lock
            # deterministically. Closing it from the event-loop thread would race
            # the in-flight next().
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
            # A watcher failure must NEVER cancel a live generation: the
            # generation runs to completion and the failure is logged.
            from localm.debuglog import logger as _dbg
            _dbg.exception("non-stream disconnect watcher failed")

    watcher = asyncio.ensure_future(_watch_disconnect())
    try:
        return await fut
    finally:
        # Also stop the worker if the handler coroutine itself is cancelled
        # (server shutdown, a timeout middleware). A no-op on the normal path,
        # where the worker already returned and fut is done. Then retire the
        # watcher.
        cancel_event.set()
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass


def _memory_used_header(ctx) -> dict:
    """Observability: render the memory plugin's per-turn recall (stashed in
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


# The backend error contract, in ONE table so the two non-streaming handlers
# cannot drift apart. Every entry is a ValueError subclass a backend raises to
# carry a reason the caller can act on, mapped to the status that says whose
# problem it is.
#
# ORDER IS LOAD-BEARING, so this is a sequence and not a dict:
# ImageDecodeUnavailable and VisionInputError are both UnsupportedInputError
# subclasses, and TriggerValidatorUnavailableError is an InvalidGrammarError, so
# each subclass must stay ABOVE its parent or the parent's status wins.
#
# 503 for TriggerValidatorUnavailableError: a saturated probe pool clears on its
# own, so it is the one entry here that is not permanent.
#
# 501 for ImageDecodeUnavailable: the request is fine and the caller can do
# nothing about it, and the missing decoder will not appear on a retry.
_BACKEND_ERROR_STATUS: tuple = (
    (ImageDecodeUnavailable, 501),
    (VisionInputError, 400),
    (UnsupportedInputError, 400),
    (GrammarUnsupportedError, 400),
    (TriggerValidatorUnavailableError, 503),
    (InvalidGrammarError, 400),
    (EmbedBatchTooLargeError, 413),
    (ContextCapacityExceededError, 413),
)

# The same classes as a plain tuple, for use as an except clause. Derived from
# the table rather than written out again, so a class in one is never missing
# from the other; a caught class the table does not map would raise
# HTTPException(None, ...).
_BACKEND_ERROR_TYPES: tuple = tuple(t for t, _ in _BACKEND_ERROR_STATUS)


def backend_error_status(exc: BaseException) -> Optional[int]:
    """Status for a backend error the caller can act on, or ``None`` when this is
    not one of them.

    ``None`` is the important half: it means the exception falls through to the
    generic handler and becomes an opaque 500, which is the CORRECT outcome for a
    genuine bug. NOT written as ``except ValueError`` at the call sites - every
    class above IS a ValueError, so a broad catch would also swallow an
    unrelated ValueError from a real defect and report it to the user as their
    own bad input.
    """
    for exc_type, status in _BACKEND_ERROR_STATUS:
        if isinstance(exc, exc_type):
            return status
    return None


def inference_error_text(exc: BaseException) -> str:
    """The `[inference error: ...]` body a FAILED generation is rendered as, on
    every one of the four generation paths.

    ONE implementation for all four, so the string is stated in one place.

    THE PATHS ARE SCRUBBED. A mid-generation
    RuntimeError is not always a tidy "not enough free VRAM" sentence: the GGUF
    loader raises `Failed to load model: <absolute path>` with a native stderr
    tail appended, and an auto-reload inside chat_stream can surface exactly
    that here. Handing a client the machine's directory layout is the
    disclosure `pathscrub` exists for, and `bugreport.py` already names
    scrub_paths as the rule for a response to a lower-privileged caller.

    scrub_paths REDACTS, it does not mute: the reason, the file name and the
    line number survive, and only the leading directories are replaced. A caller
    still learns what failed without learning where this machine keeps its
    files.
    """
    from localm.pathscrub import scrub_paths
    return f"\n[inference error: {scrub_paths(str(exc))}]"


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
    prompt_tokens: Optional[int] = None,
    **gen_kwargs,
):
    # Call the engine's real methods directly: a method-less engine raises here
    # rather than falling back to fabricated token counts and a canned
    # completion.
    capacity = engine.context_capacity()
    if prompt_tokens is None:
        prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

        if isinstance(capacity, int) and capacity > 0 and len(messages) > 3:
            buffer = max(2048, int(capacity * 0.10))
            if capacity - prompt_tokens < buffer:
                from localm.inference.compact import compact_messages
                def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                    return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
                # Off the event loop: compaction runs a full generation and must
                # not block the single-threaded loop.
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
        # Cancelable on client disconnect, so an aborted request releases the
        # per-model _inference_lock and this semaphore instead of generating to
        # the end of its budget.
        try:
            text = await _generate_full(engine, messages, request,
                                        timing=timing, **gen_kwargs)
        except _BACKEND_ERROR_TYPES as e:
            # A backend refusal the CALLER can act on (an image this vision model
            # could not process, a grammar the deferred check rejected at
            # sampler-build time, a missing image decoder). Each is a ValueError,
            # so this arm must sit above the RuntimeError catch below and the
            # generic Exception backstop, or a bare internal-server-error detail
            # replaces the refusal.
            #
            # Raised as an HTTPException rather than rendered inline like the
            # RuntimeError case below: nothing was generated, so the status is
            # the report. The streaming path cannot match the STATUS, its role
            # chunk having already put a 200 on the wire, but it carries the same
            # reason and marks finish_reason as error.
            raise HTTPException(backend_error_status(e), str(e))
        except RuntimeError as e:
            # A generation FAILURE (not enough free VRAM for this prompt, a
            # conversation that outgrew n_ctx_max, a native decode error) is
            # raised as RuntimeError and reaches the client as a clean reply,
            # never a raw HTTP 500. Catches ONLY RuntimeError: a broken engine (a
            # method-less mock raising AttributeError) must surface loudly, and
            # CancelledError from a client disconnect must not be swallowed.
            from localm.debuglog import logger as _dbg
            _dbg.exception("non-streaming generation failed")
            gen_error = e
            text = inference_error_text(e)
        gen_end = time.perf_counter()
    first_token_at = timing.get("first_token_at")

    # Outlet fully controls the returned content in the non-streaming path (but a
    # failed generation surfaces its error verbatim, not reshaped by the outlet).
    if gen_error is None and pipeline is not None and ctx is not None and pipeline.has("outlet"):
        text = await pipeline.run_outlet(text, messages, ctx)

    _audit_exchange(audit, transcript, messages, text)

    # Split the model's <think> reasoning out of the visible answer into a
    # separate field, so API clients get clean content. The token count stays on
    # the full generated text.
    from localm.textnorm import split_think
    answer, reasoning = split_think(text)

    completion_tokens = await asyncio.get_running_loop().run_in_executor(
        None, engine.count_tokens, text)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        # A non-streaming handler still drives chat_stream internally, so the
        # first-token boundary is observable: ttft_ms is reported too, and tok/s
        # is computed over the decode window only, never folding in the load.
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(
            completion_tokens, _decode_elapsed(first_token_at, gen_end)),
        context_capacity=capacity,
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

    # No custom Win32 Ctrl+C handler here: portmux's asyncio.run creates a fresh
    # loop, so a control-handler OS thread cannot stop the serving loop, and
    # returning True from one eats the event and defeats uvicorn's own SIGINT
    # shutdown. Ctrl+C flows through uvicorn instead (KeyboardInterrupt caught in
    # portmux.run_server), kept responsive on Windows by portmux's loop wakeup.

    with instances.advertise(app, home_dir(), host=host, port=port, mode=mode,
                             scheme=scheme, project=project, isolated=isolated):
        # On a TLS bind, a plain-http request on the same port is answered with
        # an https redirect; plain binds are a direct uvicorn.run. In debug mode
        # uvicorn logs at info so the console shows requests.
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

    When ``ssl_certfile`` / ``ssl_keyfile`` are given (built-in TLS), the
    server speaks HTTPS on this port; a plain-HTTP request to it then fails the
    TLS handshake (effectively refused) rather than crossing the network in
    cleartext.

    ``mode`` is the instance-registry surface (``"api"`` or ``"full"``).

    Advertises itself in the instance registry so a future launch can discover
    and attach to it; ``isolated`` keeps it invisible to discovery.
    """
    app = create_app(engine, api_landing=True)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    run_advertised(app, host, port, mode=mode, ssl_certfile=ssl_certfile,
                   ssl_keyfile=ssl_keyfile, project=project, isolated=isolated)

