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
import dataclasses
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
from typing import AsyncIterator, Callable, NamedTuple, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback_host  # noqa: F401  (re-export for back-compat)
from localm.inference.backends.base import (
    ChatTemplateMissingError,
    ContextCapacityExceededError,
    EmbedBatchTooLargeError,
    GrammarUnsupportedError,
    ImageDecodeUnavailable,
    InvalidGrammarError,
    ModelLoadCancelled,
    PretokenizerUnsafeInputError,
    TriggerValidatorUnavailableError,
    UnsupportedInputError,
    VisionInputError,
)
from localm.inference import residency, switch_admission
from localm.inference.engine import Engine
from localm.inference.routing_latch import RoutingLatch
from localm.inference.stop_sequences import StopFilter, apply_stop
from localm.inference.protocol import (
    COMPACTING_STATUS, LOADING_MODEL_STATUS, ChatChunk, ChatResponse, ChoiceDelta,
    FullChoice, Message, MtpUsage, PROCESSING_PROMPT_STATUS, SpeculationUsage, STATUS_CODE_BY_TEXT,
    StreamChoice, UsageInfo, WAITING_FOR_MODEL_STATUS, make_chunk_id,
)

# Models whose last load failed; capability routing leaves them out of its
# candidates (see localm.inference.routing_latch).
_routing_latch = RoutingLatch()


class LoadSkipped(HTTPException):
    """503 raised instead of a load that was refused because the model's last
    load failed and that failure still applies. ``skipped`` is the latch's
    record for the model."""

    def __init__(self, skipped) -> None:
        super().__init__(503, skipped.describe())
        self.skipped = skipped


# Map of display name -> Engine instance
_engines: dict[str, Engine] = {}
# Order of model usage (display names, MRU at the end)
_engines_lru: list[str] = []
# Display names currently mid-eviction: detached from _engines/_engines_lru
# already (so a fast-path lookup correctly sees them as gone), but the native
# free (evict_engine.unload(), an executor call the eviction loop awaits) has
# not completed yet. A concurrent switch_engine/get_engine call for THIS SAME
# name has no other way to see that a free is in flight for it: it would
# otherwise construct-and-load a brand-new engine for the name while the stale
# eviction is still running, race it, and (once the stale unload() finally
# completes) end up pinning an engine that gets freed out from under it.
# switch_engine consults this before constructing/loading *name* and refuses
# (503, honest backpressure) rather than racing it.
_evicting_names: set[str] = set()
# Default/startup model name
_default_model_name: str | None = None
# Active model name (most recently used/loaded)
_active_model_name: str | None = None
# The name _active_model_name held immediately before a full eviction
# (unload_all_models) cleared it. That eviction KEEPS the Engine in _engines so
# it reloads lazily, and _default_model_name is write-once at startup
# (create_app) and never updated by a model switch, so without this nothing
# would still NAME the model actually in use. Only ever consulted via
# _resolve_unnamed_model_name, AFTER _active_model_name; cleared by
# switch_engine on every successful activation, so a later eviction on a
# DIFFERENT path (idle-unload, a single-model unload) can never resolve it to a
# stale name from an unrelated, long-past eviction.
_last_active_model_name: str | None = None

# The server's audit log, published by create_app() (see the `global _audit`
# there). Must default to None here so _do_restart's `global _audit` guard
# always resolves to a real (possibly None) value even if it runs in a
# process/interpreter where create_app() was never called first - reading an
# unassigned global raises NameError, not a clean "None" check.
_audit = None

# Inference serialisation - per-model semaphores mapping display name -> Semaphore
_inference_sems: dict[str, asyncio.Semaphore] = {}

# Bounds the dedicated-embedder /v1/embeddings path to ONE default-pool worker
# at a time. Not an _inference_sems entry: those follow the chat engines'
# lifecycle (popped on evict, renamed on rename). Cleared with them in
# create_app. See test_dedicated_embed_path_holds_one_pool_worker_at_a_time.
_embedder_sem: asyncio.Semaphore | None = None


def _get_embedder_sem() -> asyncio.Semaphore:
    """The dedicated-embedder semaphore, created on first use on the serving loop."""
    global _embedder_sem
    if _embedder_sem is None:
        _embedder_sem = asyncio.Semaphore(1)
    return _embedder_sem

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
_server_loop: asyncio.AbstractEventLoop | None = None

# Preemptive model switching (see switch_engine). _switch_desired = most-recent
# switch request; _switch_loading = model whose load is in flight; _switch_cancel
# aborts it. Touched only on the event-loop thread inside switch_engine and its
# _switch_load helper, except the cancel event, which the loader-thread
# load-progress callback reads (threading.Event is thread-safe).
_switch_desired: Optional[str] = None
_switch_loading: Optional[str] = None
_switch_cancel: Optional[threading.Event] = None

# Cross-install GPU/VRAM coordination (multi-instance, see localm.gpu_registry).
# None until lifespan startup populates it, and ONLY for a real, non-isolated,
# instances.advertise()'d server (app.state.instance_id set, instance_isolated
# falsy). A plain create_app() test app or an --isolated run never sets it, so it
# is invisible to sibling instances. Shape: {"instance_id", "port", "host", "scheme"}.
_gpu_coord: Optional[dict] = None

# The coder plugin's SessionManager, published as a module global (mirroring
# _gpu_coord above) so a free function like unload_one_model can ask "is any
# live coder session bound to this model" without needing app/request
# threaded through it. None until mount_gui_surface() attaches the GUI (a
# bare create_app() test app, or an --isolated/API-only instance, never
# mounts it, so this stays None on those). Set alongside the identical
# app.state.coder_sessions assignment in mount_gui_surface - same object,
# reachable two ways for two different kinds of caller.
_coder_session_manager = None

# The running server's HangAlarm instance (see localm.inference._hang_alarm),
# None until lifespan startup constructs one, None again once that lifespan
# shuts down, and None whenever recovery is disabled
# (LOCALM_HANG_RECOVERY=off) or the process is under pytest. Read by
# switch_engine to trigger the same seamless self-restart the loop-freeze and
# transport-death detectors use, for a GPU probe still wedged after its own
# in-request retries.
_hang_alarm_instance = None

# Hang watchdog: a monotonic heartbeat bumped every _HEARTBEAT_INTERVAL_S by
# _hang_heartbeat_loop (an async task ON the loop) and read by the off-loop
# watchdog thread + the debug request log + GET /debug/stacks. A growing (now
# - _hb_monotonic) means the single event loop has stopped making progress,
# i.e. something is blocking it - the off-loop watchdog thread compares this
# raw gap against its own multi-second threshold (hang_watchdog_threshold(),
# 10s by default) and is unaffected by the note below.
#
# The heartbeat TASK's own startup (lifespan, below) is gated only on
# "pytest" not in sys.modules - NOT on the watchdog thread's privacy/env gate.
# It is pure in-memory bookkeeping (no I/O, nothing persisted or observable
# outside this process), so it carries none of the privacy considerations
# that gate the stack-dump-to-disk thread, and every reader below needs it
# regardless of which of them is actually active.
#
# _loop_lag_seconds() (below) answers a DIFFERENT question and must be used
# for anything reported to a human. Raw (now - _hb_monotonic) saws between 0
# and ~_HEARTBEAT_INTERVAL_S on a perfectly healthy loop - that is just how
# far into the current tick cycle "now" happens to land, not evidence of lag.
# Subtracting the interval turns it into a real scheduling-delay figure: ~0
# when healthy, and positive only when a tick itself was late, i.e. something
# actually blocked the loop.
#
# None (not a time.monotonic() value) UNTIL THE HEARTBEAT TASK'S OWN FIRST
# TICK, and never seeded at import time. _hang_heartbeat_loop() is only created
# once lifespan() starts, and only updates this on its first actual turn on the
# loop; an import-time seed makes "now - _hb_monotonic" measure
# elapsed-since-import across the cold-start window (import -> first tick,
# which a slow startup/model-load can stretch to a minute-plus), a number that
# grows with wall-clock time regardless of what the loop is doing. Both readers
# below (_loop_lag_seconds, the watchdog thread) treat None as "no reading
# yet" - report None / skip the check - rather than inventing a number from a
# timestamp that was never real.
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
    all". A finer-grained sampler would close this gap at the cost of a
    second background task and more wakeups purely for a diagnostic counter,
    which is not worth it: the hang watchdog (large-threshold, above) already
    owns detecting a real freeze; this value is for correlating a slow
    request with a preceding stall, not for catching sub-second ones.

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
    used to decide whether eviction was needed - but NOT for the
    empty-directory return value: model_footprint_bytes returns int (never
    None) because its caller always needs a numeric eviction-admission input,
    even a 0 one, while this function feeds a registry field other instances
    treat as "how much VRAM does this peer hold", where a 0 is read as a REAL
    measurement (see gpu_registry.py's vram_estimate_bytes and its one
    consumer's `isinstance(e, int) and e > 0` guard). rglob() matching no
    files - an empty directory, or one whose real weights sit somewhere
    rglob does not look - means the size genuinely was not measured, so this
    returns None rather than a suspiciously-precise 0."""
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
            skipped = residency.alternate_layout_files(p)
            total = sum(f.stat().st_size for f in p.rglob("*")
                        if f.is_file() and f not in skipped)
            return total if total > 0 else None
    except (OSError, TypeError, ValueError):
        return None
    return None


def _current_gpu_index() -> int:
    """The device the next GGUF load reads its VRAM from (0 when nothing
    selects one) - ``discover.resolve_load_gpu_index``, the same resolution
    ``vram_info()`` and the GGUF backend's own VRAM check use, validated
    against ``discover.last_gpu_reading()`` so it never probes."""
    try:
        from localm.config import load_config
        from localm.discover import last_gpu_reading, resolve_load_gpu_index
        return resolve_load_gpu_index(load_config(), gpus=last_gpu_reading() or [],
                                      quiet=True)
    except Exception:
        return 0


def _loaded_gpu_index(name: Optional[str]) -> Optional[int]:
    """The device loaded model *name* runs on alone (its backend's
    ``load_gpu_index``), or None when it is not loaded on one device or that
    is not recorded."""
    engine = _engines.get(name) if name else None
    idx = getattr(getattr(engine, "_backend", None), "load_gpu_index", None)
    return idx if isinstance(idx, int) and not isinstance(idx, bool) else None


def _loaded_model_identities() -> list:
    """Every loaded chat model as ``{"name", "path", "size", "sha256"}`` for the
    cross-install coordination registry. ``path`` is the resolved model file or
    directory, ``size`` its byte size when it is a file, ``sha256`` the
    registry-recorded digest; each is None when unknown. Best-effort: an entry
    that cannot be resolved is listed by name alone."""
    from localm import peer_routing
    try:
        from localm.config import load_registry
        reg = load_registry()
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("gpu-registry: registry unreadable for model identities: %s", e)
        reg = {}
    return [{"name": name, **peer_routing.local_identity(reg, name)}
            for name, eng in list(_engines.items()) if getattr(eng, "loaded", False)]


def _gpu_status() -> Optional[dict]:
    """This instance's live coordination status, served to sibling instances on
    ``GET /v1/instances/status`` and read by the in-process VRAM-holder hint:
    ``{instance_id, pid, port, host, scheme, model, models,
    vram_estimate_bytes, gpu_index}``. None when this instance does not
    coordinate (``_gpu_coord`` unset: a plain test app or an ``--isolated`` run).

    Blocking: sizes the loaded model files and may probe the GPU driver for the
    device index, so call it off the event loop. Never raises."""
    coord = _gpu_coord
    if not coord:
        return None
    try:
        import os as _os
        loaded = [n for n, e in list(_engines.items()) if getattr(e, "loaded", False)]
        model = _active_model_name or (loaded[0] if loaded else None)
        vram_bytes = None
        sizes = [_model_file_size(n) for n in (loaded or ([model] if model else []))]
        if sizes and all(sz is not None for sz in sizes):
            vram_bytes = int(sum(sizes) * 1.2)
        gpu_index = _loaded_gpu_index(model)
        if gpu_index is None:
            gpu_index = _current_gpu_index()
        return {
            "instance_id": coord["instance_id"],
            "pid": _os.getpid(),
            "port": coord.get("port"),
            "host": coord.get("host") or "127.0.0.1",
            "scheme": coord.get("scheme") or "http",
            "model": model,
            "models": _loaded_model_identities(),
            "vram_estimate_bytes": vram_bytes,
            "gpu_index": gpu_index,
        }
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("gpu status unavailable (continuing): %s", e)
        return None


def _load_gpu_indices() -> set:
    """Every device whose free VRAM this instance's next model load can actually
    USE - the whole configured split when one resolves to 2+ devices, else the
    GPUs llama.cpp's default split spreads a GGUF load over
    (``discover.implicit_split_gpus`` on the last reading, the devices
    ``_switch_probe_vram`` sums), else the one device ``_current_gpu_index``
    names. Never probes beyond ``resolve_gpu_split``'s own reading.

    NOT ``{_current_gpu_index()}``: that is an IDENTITY answer ("which one device
    is primary"), and resolve_main_gpu_index(None) returns 0 for an unconfigured
    main_gpu_index even on a box whose split spans 0 AND 1. Weighing peers
    against that single index while weighing VRAM against vram_capacity()'s
    COMBINED split total contradicts itself, and drops a sibling holding VRAM on
    this instance's own second split device, turning a cooperative unload into a
    503. This is the capacity-vs-identity distinction the raw-accessor guard in
    scripts/check_hygiene.py enforces.

    Known limitation: a registry entry
    advertises ONE ``gpu_index`` per instance (see _gpu_status), so a
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
        from localm.discover import implicit_split_gpus
        kept = implicit_split_gpus(cfg)
        if kept is not None:
            return {d.get("index") for d in kept}
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
    own ``unload_all_models``), so this is conservative about when it is worth
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
    so tests and isolated runs never probe or call another instance). Fully
    advisory: ANY failure (no live peer, request timeout/refusal) is logged and
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
    yet, or a backend without a layer-count knob - see Engine.gpu_placement),
    plus the ``Engine.mmap_state`` fields (``use_mmap``, ``mmap``,
    ``mmap_from_disk``, ``mmap_note``) when the load reported them.
    Merged into every switch_engine()/load-route success payload so a caller
    can tell a full GPU load from a silent CPU fallback instead of a bare
    "loaded"/"already_active" that hides it."""
    placement = getattr(engine, "gpu_placement", None)
    fields = dict(placement) if placement else {}
    mmap_state = getattr(engine, "mmap_state", None)
    if isinstance(mmap_state, dict):
        fields.update(mmap_state)
    return fields


# How many more probe attempts an inconclusive VRAM reading with nothing left
# to evict gets in switch_engine, and the pause between them, before the load
# is refused with a 503.
_INCONCLUSIVE_LOAD_RETRIES = 2
_INCONCLUSIVE_LOAD_RETRY_DELAY = 1.5

# How long a load that is not an explicit switch waits for a busy resident
# model to finish its requests, so that model can be evicted instead of the new
# one loading beside it on whatever VRAM is left.
_BUSY_VICTIM_IDLE_WAIT_S = 30.0

# How much longer _switch_free_victim waits for an evicted model's VRAM when
# the first wait_for_vram_release saw no release.
_VICTIM_RELEASE_EXTRA_WAIT_S = 25.0


async def switch_engine(name: str, make_engine, *, on_active=None, preempt: bool = True,
                        force: bool = False, activate: bool = True,
                        skip_if_latched: bool = False,
                        on_status: Optional[Callable[[str], None]] = None) -> dict:
    """Make model *name* resident and, with *activate*, the active model.

    *make_engine*, when not None, becomes the engine factory used to build an
    engine for a name not registered in ``_engines``. *on_active* is called
    with *name* when it becomes the active model. *preempt* marks an explicit
    user switch: it supersedes an earlier explicit switch that is still
    loading, and may cancel a busy resident model or ask before a degraded
    load. *force* skips those confirmations. With *activate* False the model
    becomes active only when nothing else would answer an unnamed request.
    With *skip_if_latched* (a load that capability routing chose, not one the
    user asked for) a model that is not resident is not loaded when its last
    load failed and that failure still applies once the model's semaphore is
    held; ``LoadSkipped`` is raised instead. *on_status*, when given, is called
    on the event loop thread with ``WAITING_FOR_MODEL_STATUS`` while the load
    waits for a busy model to finish and with ``LOADING_MODEL_STATUS`` when it
    goes on.

    A load that is not an explicit switch waits up to ``_BUSY_VICTIM_IDLE_WAIT_S``
    for a busy resident model that stands in its way to go idle, then evicts it
    (``_switch_wait_for_busy_victim``). A resident model left partly on the CPU
    by an earlier load is reloaded instead of reused, once, when a model that
    held the VRAM it lacked is idle or gone (``_placement_heal_due``) and a
    fresh VRAM reading is conclusive; an explicit switch reuses it as it is.
    During that reload an unnamed request still resolves to it.

    Returns a dict whose ``status`` is one of:

    - ``already_active``: *name* was already loaded; merged with
      ``_gpu_placement_fields``.
    - ``loaded``: *name* was loaded by this call; merged with
      ``_gpu_placement_fields``.
    - ``superseded``: a newer explicit switch, named in ``by``, replaced this one.
    - ``cancelled``: the load was cancelled for another ``reason``.
    - ``confirm_required``: an explicit switch without *force* would have to
      evict a model still in use, or load below the whole-model VRAM
      estimate; ``detail`` says which.

    Raises HTTPException 404 when *name* is registered but its files are not
    found, and 503 when a static split device is short of VRAM, the VRAM probe
    stays inconclusive, *name* is still being freed by another request, or the
    backend fails to load it. ``LoadSkipped`` (a 503) is raised for
    *skip_if_latched*.

    Loads of one model are serialized by its ``_inference_sems`` semaphore.
    The admission decisions come from ``localm.inference.switch_admission``;
    the ``_switch_*`` helpers below perform the effects.
    """
    global _engine_factory, _switch_desired

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

        healing = None
        was_active = False
        heal_budget = heal_probe = None
        resident = _engines.get(name)
        if _is_reusable(resident):
            if not preempt and _placement_heal_due(name, resident):
                try:
                    heal_budget = _switch_load_budget(name)
                except HTTPException as exc:
                    from localm.debuglog import logger as _dbg
                    _dbg.info("switch_engine: not reloading '%s' for a better placement, "
                              "it is used as loaded: %s", name, exc.detail)
            if heal_budget is not None:
                heal_budget = await _switch_with_backend_need(loop, heal_budget, resident)
                heal_probe = await _switch_probe_vram(loop, heal_budget)
                if not (heal_probe.probe_ok and heal_probe.measurable
                        and _engines.get(name) is resident
                        and _placement_heal_due(name, resident)):
                    heal_budget = None
            if heal_budget is None and _engines.get(name) is resident and _is_reusable(resident):
                _switch_reuse_resident(name, sem, activate=activate, on_active=on_active)
                return {"status": "already_active", "model": name,
                        **_gpu_placement_fields(resident)}
            if heal_budget is not None:
                # No await since _placement_heal_due read active_requests == 0.
                healing = resident
                was_active = _switch_detach_victim(name, resident, activate=False,
                                                   keep_sem=True)

        if healing is None and skip_if_latched and _routing_latch.failure(name) is not None:
            latched = await loop.run_in_executor(None, _routing_latch.skipped, [name])
            if name in latched:
                raise LoadSkipped(latched[name])

        built = None
        if healing is not None:
            budget = heal_budget
        else:
            budget = _switch_load_budget(name)
            if budget is not None and budget.check_split_fit:
                sized = _engines.get(name)
                if sized is None:
                    built = sized = _engine_factory(name)
                budget = await _switch_with_backend_need(loop, budget, sized)
        attempt = switch_admission.EvictionAttempt(started=time.monotonic())
        evictions: list[VictimRelease] = []
        if healing is not None:
            release = await _switch_free_for_reload(loop, name, healing, heal_probe,
                                                    heal_budget.needed_bytes)
            attempt.release_wait_extended = release.extended
            evictions.append(release)
        if budget is not None:
            early = await _switch_make_room(loop, budget, preempt=preempt,
                                            force=force, activate=activate,
                                            attempt=attempt, evictions=evictions,
                                            on_status=on_status)
            if early is not None:
                return early

        # *name* may itself be a victim another call detached and is still freeing.
        if name in _evicting_names:
            raise HTTPException(
                503, f"'{name}' is currently being freed by another request; "
                f"retry shortly.")

        if healing is not None:
            new_engine = healing
        elif name in _engines:
            new_engine = _engines[name]
        else:
            new_engine = built if built is not None else _engine_factory(name)
        interrupted = await _switch_load(loop, name, new_engine, preempt=preempt)
        if interrupted is not None:
            return interrupted

        _switch_commit(name, new_engine, sem, activate=activate or was_active,
                       on_active=on_active)
        _log_switch_placement(name, new_engine, evictions)
        _record_placement_heal(name, new_engine,
                               budget.pinned if budget is not None else frozenset(),
                               evictions, allowed=healing is None,
                               deferred=attempt.deferred_to_backend)
        return {"status": "loaded", "model": name,
                **_gpu_placement_fields(new_engine)}


def _is_reusable(engine) -> bool:
    """Whether *engine* is a loaded resident model that is not being unloaded."""
    return (engine is not None and engine.loaded
            and getattr(engine, "unloading", False) is not True)


def _switch_reuse_resident(name: str, sem, *, activate: bool, on_active) -> None:
    """Move resident *name* to the most-recently-used end of ``_engines_lru``
    and, with *activate*, make it the active model: ``_active_model_name``,
    ``_engine`` and ``_inference_sem`` point at it, ``_last_active_model_name``
    is cleared and *on_active* is called."""
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
    if name in _engines_lru:
        _engines_lru.remove(name)
    _engines_lru.append(name)
    if activate:
        _active_model_name = name
        _last_active_model_name = None
        _engine = _engines[name]
        _inference_sem = sem
        if on_active is not None:
            on_active(name)


def _switch_load_budget(name: str) -> switch_admission.LoadBudget | None:
    """The VRAM budget a load of *name* has to fit, read from the model
    registry, the model's on-disk footprint and the residency settings.

    Returns None for an empty registry (single-model or direct-path startup):
    no eviction runs then. Raises HTTPException 404 when the registry is not
    empty and *name*'s files cannot be resolved."""
    from localm.config import load_config, load_registry
    from localm.model_manager import get_model_info

    registry = load_registry()
    info = get_model_info(name)
    file_size = 0
    if info is not None:
        m_path, _ = info
        file_size = residency.model_footprint_bytes(m_path)
    elif registry:
        raise HTTPException(404, f"Model files not found: {name}")
    if not registry:
        return None

    from localm.inference.engine import _is_gguf
    vram_required = residency.required_vram_bytes(file_size)
    cfg = load_config()
    return switch_admission.LoadBudget(
        name=name,
        vram_required=vram_required,
        headroom=residency.DEFAULT_HEADROOM_BYTES,
        resident_cap=residency.resident_cap(cfg),
        pinned=residency.pinned_model_names(cfg),
        check_split_fit=_is_gguf(m_path),
    )


async def _switch_with_backend_need(loop, budget: switch_admission.LoadBudget,
                                    engine) -> switch_admission.LoadBudget:
    """*budget* with ``backend_need`` set to *engine*'s
    ``full_offload_vram_bytes()``, read off the event loop. *budget* is
    returned unchanged when *engine* has no such method or it answers None.
    An exception from it is logged at WARNING and *budget* is returned
    unchanged."""
    size = getattr(engine, "full_offload_vram_bytes", None)
    if not callable(size):
        return budget
    from localm.debuglog import logger as _dbg
    try:
        need = await loop.run_in_executor(None, size)
    except Exception as exc:
        _dbg.warning("switch_engine: could not size a full GPU offload of '%s' "
                     "(%s: %s); admitting it on the whole-model estimate",
                     budget.name, type(exc).__name__, exc)
        return budget
    if not need:
        return budget
    sized = dataclasses.replace(budget, backend_need=int(need))
    _dbg.debug("switch_engine: '%s' needs ~%s MB free for a full GPU offload; "
               "admission bar %s MB (estimate %s MB + headroom %s MB)",
               budget.name, need // 1024 ** 2, sized.needed_bytes // 1024 ** 2,
               budget.vram_required // 1024 ** 2, budget.headroom // 1024 ** 2)
    return sized


async def _switch_make_room(loop, budget: switch_admission.LoadBudget, *,
                            preempt: bool, force: bool, activate: bool,
                            attempt: Optional[switch_admission.EvictionAttempt] = None,
                            evictions: Optional[list] = None,
                            on_status: Optional[Callable[[str], None]] = None
                            ) -> Optional[dict]:
    """Run the eviction loop until ``budget.name`` may load.

    Each iteration takes a VRAM reading, asks
    ``switch_admission.decide_admission`` what it allows, and then loads,
    evicts the idle victim it named, or walks the exhaustion ladder
    (``_switch_exhaustion_ladder``). An evicted victim is detached from every
    live registry, natively unloaded, and the loop re-probes; a victim that is
    no longer registered, is being unloaded or is serving a request by then is
    skipped and the loop re-probes. The ``VictimRelease`` of each eviction is
    appended to *evictions* when given. *attempt* is the load attempt's
    ``EvictionAttempt`` (a fresh one when None). *on_status* is passed to
    ``_switch_exhaustion_ladder``.

    Returns None when the load may proceed, or a ``confirm_required`` result
    for switch_engine to return. Raises HTTPException 503 when a static split
    device is short or the probe stays inconclusive.

    There is no await between choosing a victim, re-checking its pin and
    detaching it. See test_eviction_victim_not_pinnable_during_native_free.
    """
    from localm.inference import embedder as _embedder_mod

    if attempt is None:
        attempt = switch_admission.EvictionAttempt(started=time.monotonic())
    while True:
        probe = await _switch_probe_vram(loop, budget)
        decision = switch_admission.decide_admission(
            probe, budget, _engines_lru, _engines)
        if decision.action == switch_admission.ADMIT:
            return None
        if decision.action == switch_admission.ADMIT_OVER_CAP:
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "max_resident_models=%s wanted room for %s but no "
                "resident model could be evicted (resident=%s, "
                "pinned=%s); free VRAM is sufficient, so loading it "
                "anyway over the cap",
                budget.resident_cap, budget.name, list(_engines_lru),
                sorted(budget.pinned))
            return None

        victim, force_busy = decision.victim, False
        if decision.action == switch_admission.EXHAUSTED:
            step = await _switch_exhaustion_ladder(
                loop, probe, budget, attempt, _embedder_mod,
                preempt=preempt, force=force, on_status=on_status)
            if step.kind == switch_admission.REPROBE:
                continue
            if step.kind == switch_admission.PROCEED_TO_LOAD:
                return None
            if step.kind == switch_admission.RETURN_RESULT:
                return step.result
            victim, force_busy = step.victim, step.force_busy

        victim_engine = _engines.get(victim)
        if victim_engine is None or getattr(victim_engine, "unloading", False) is True:
            continue
        # Re-check the pin without taking the victim's semaphore: holding the
        # target's semaphore while taking the victim's can deadlock two
        # concurrent switches. A pinned victim is skipped and the loop
        # re-probes; force_busy evicts it anyway.
        if not force_busy and getattr(victim_engine, "active_requests", 0) != 0:
            continue
        _switch_detach_victim(victim, victim_engine, activate=activate)
        release = await _switch_free_victim(
            loop, victim, victim_engine, probe, needed=budget.needed_bytes,
            extra_wait=not attempt.release_wait_extended)
        attempt.release_wait_extended = attempt.release_wait_extended or release.extended
        if evictions is not None:
            evictions.append(release)


async def _switch_probe_vram(loop, budget: switch_admission.LoadBudget
                             ) -> switch_admission.VramProbe:
    """Take one VRAM reading off the event loop: ``vram_capacity()`` and, when
    ``budget.check_split_fit``, the configured split's
    ``gpu_split_shortfall()`` for ``budget.split_share_bytes``.

    ``vram_capacity`` is given the full CLI deadline on this first call and
    joins a probe already in flight (``wait_for_inflight``); joining is only
    safe because the call runs in an executor thread.

    For a GGUF load (``budget.check_split_fit``) with no configured split, a
    fresh reading of the GPUs llama.cpp's default split spreads over (2+, or
    the one discrete GPU beside integrated ones) is judged by their summed
    free VRAM (``discover.implicit_split_free``, the budget the backend sizes
    the load against), and the probe is marked ``implicit_split``."""
    from localm import discover
    from localm.discover import gpu_split_shortfall, implicit_split_free, vram_capacity

    v_info, probe_status = await loop.run_in_executor(
        None, functools.partial(
            vram_capacity, return_status=True,
            deadline=discover._GPU_PROBE_CLI_DEADLINE,
            wait_for_inflight=True))
    free = v_info.get("free")
    process_scoped = v_info.get("free_scope") == discover.FREE_SCOPE_PROCESS
    implicit = None
    if budget.check_split_fit and probe_status == discover.GPU_PROBE_OK:
        implicit = await loop.run_in_executor(None, implicit_split_free)
    if implicit is not None:
        free = implicit["free"]
        process_scoped = implicit.get("free_scope") == discover.FREE_SCOPE_PROCESS
    shortfall, shares_adaptive = (
        await loop.run_in_executor(
            None, functools.partial(
                gpu_split_shortfall, budget.split_share_bytes,
                return_shares_adaptive=True))
        if budget.check_split_fit else ([], False))
    return switch_admission.VramProbe(
        free=free, probe_ok=probe_status == discover.GPU_PROBE_OK,
        process_scoped=process_scoped, shortfall=shortfall,
        shares_adaptive=shares_adaptive, implicit_split=implicit is not None)


def _probe_free_reader(probe: switch_admission.VramProbe):
    """A callable re-reading free VRAM as the same quantity as ``probe.free``:
    the summed free of llama.cpp's default-split GPUs for an
    ``implicit_split`` probe (None when a fresh reading of them is not
    available), else ``vram_capacity()``'s free."""
    from localm import discover

    if not probe.implicit_split:
        return lambda: discover.vram_capacity().get("free")

    def _read() -> Optional[int]:
        gpus, status = discover._list_gpus_reading()
        if status != discover.GPU_PROBE_OK:
            return None
        info = discover.implicit_split_free(gpus=gpus)
        return info.get("free") if info is not None else None
    return _read


async def _switch_exhaustion_ladder(loop, probe: switch_admission.VramProbe,
                                    budget: switch_admission.LoadBudget,
                                    attempt: switch_admission.EvictionAttempt,
                                    embedder_mod, *, preempt: bool,
                                    force: bool,
                                    on_status: Optional[Callable[[str], None]] = None
                                    ) -> switch_admission.EvictionStep:
    """Find room once no idle chat model is evictable, in this order: free the
    idle shared embedder; act on the probe verdict (load best-effort when the
    box cannot measure, retry or refuse when the probe was inconclusive); ask
    a sibling localm instance to free its VRAM; cancel and claim a busy
    resident model (explicit switch) or wait for it to go idle
    (``_switch_wait_for_busy_victim``, any other load); then refuse, ask, or
    defer to the backend's own sizing
    (``switch_admission.final_exhaustion_verdict``).

    Returns the loop's next ``switch_admission.EvictionStep``. Raises
    HTTPException 503 when a static split device is short or the probe stayed
    inconclusive through every retry."""
    step = switch_admission.EvictionStep
    if not attempt.embedder_attempted:
        if await _switch_evict_embedder(loop, probe, attempt, embedder_mod):
            return step(switch_admission.REPROBE)

    verdict = switch_admission.exhausted_probe_verdict(
        probe, retries_used=attempt.inconclusive_retries,
        max_retries=_INCONCLUSIVE_LOAD_RETRIES)
    if verdict == switch_admission.LOAD_BEST_EFFORT:
        return step(switch_admission.PROCEED_TO_LOAD)
    if verdict == switch_admission.RETRY_PROBE:
        attempt.inconclusive_retries += 1
        await asyncio.sleep(_INCONCLUSIVE_LOAD_RETRY_DELAY)
        return step(switch_admission.REPROBE)
    if verdict == switch_admission.GIVE_UP_PROBE:
        raise _switch_inconclusive_failure(budget.name, attempt)

    if await _switch_ask_peers(loop, probe, budget, attempt):
        return step(switch_admission.REPROBE)

    busy = await _switch_claim_busy_victim(budget, attempt, preempt=preempt, force=force)
    if busy is not None:
        return busy

    idle = await _switch_wait_for_busy_victim(budget, attempt, preempt=preempt,
                                              on_status=on_status)
    if idle is not None:
        return idle

    final = switch_admission.final_exhaustion_verdict(probe, preempt=preempt, force=force)
    if final == switch_admission.REFUSE_SPLIT_SHORTFALL:
        raise HTTPException(
            503, switch_admission.split_shortfall_refusal(budget.name, probe.shortfall))
    if final == switch_admission.CONFIRM_DEGRADED_LOAD:
        return step(switch_admission.RETURN_RESULT,
                    result=switch_admission.degraded_load_confirm(
                        budget.name, budget.whole_model_bytes, probe.free))
    from localm.debuglog import logger as _dbg
    _dbg.info(
        "switch_engine: '%s' exceeds the whole-model VRAM estimate "
        "(need ~%s MB, %s MB free) after eviction - deferring to the "
        "backend's own load-time sizing instead of refusing",
        budget.name, budget.whole_model_bytes // 1024 ** 2, probe.free // 1024 ** 2)
    attempt.deferred_to_backend = True
    return step(switch_admission.PROCEED_TO_LOAD)


async def _switch_evict_embedder(loop, probe: switch_admission.VramProbe,
                                 attempt: switch_admission.EvictionAttempt,
                                 embedder_mod) -> bool:
    """Free the shared embedder (``localm.inference.embedder``) when it is
    loaded and no request is using it. Returns True when it was freed, after
    waiting for the VRAM release when *probe* was measurable; the wait's
    outcome is logged at DEBUG.

    Sets ``attempt.embedder_attempted`` once a loaded embedder is found, so it
    is tried at most once per load attempt. ``reset_embedder(force=False)``
    checks for in-flight requests and clears in one locked step."""
    from localm.debuglog import logger as _dbg
    from localm.vram import wait_for_vram_release

    embedder_dim = await loop.run_in_executor(None, embedder_mod.loaded_dim)
    if embedder_dim is None:
        return False
    attempt.embedder_attempted = True
    cleared = await loop.run_in_executor(
        None, functools.partial(embedder_mod.reset_embedder, force=False))
    if not cleared:
        return False
    if probe.measurable:
        released, after = await loop.run_in_executor(
            None,
            lambda: wait_for_vram_release(
                _probe_free_reader(probe), before_bytes=probe.free))
        _dbg.debug("switch_engine: freed the embedder; VRAM release %s (%s -> %s MB free)",
                   {True: "confirmed", False: "not seen"}.get(released, "not verifiable"),
                   probe.free // 1024 ** 2,
                   after // 1024 ** 2 if after is not None else "?")
    return True


def _switch_inconclusive_failure(name: str,
                                 attempt: switch_admission.EvictionAttempt
                                 ) -> HTTPException:
    """The 503 for a VRAM probe still inconclusive after every retry. First
    asks the hang alarm (``_hang_alarm_instance``) for a self-restart; the
    detail says whether one was triggered. See
    TestSwitchEngineEscalatesToSelfRestartWhenStillInconclusive."""
    attempts = attempt.inconclusive_retries + 1
    elapsed = time.monotonic() - attempt.started
    restarting = (
        _hang_alarm_instance.trigger_restart(
            switch_admission.inconclusive_restart_reason(name, attempts))
        if _hang_alarm_instance is not None else False)
    if restarting:
        return HTTPException(503, switch_admission.inconclusive_restarting_refusal(name))
    return HTTPException(503, switch_admission.inconclusive_refusal(name, attempts, elapsed))


async def _switch_ask_peers(loop, probe: switch_admission.VramProbe,
                            budget: switch_admission.LoadBudget,
                            attempt: switch_admission.EvictionAttempt):
    """Ask a sibling localm instance to release its VRAM
    (``_attempt_cooperative_unload``, off the event loop, each peer at most once
    per load attempt via ``attempt.asked_peers``). Returns its truthy result
    when a peer released its model.

    Not attempted when ``pinned_models`` is the only reason no local model
    could be evicted: that logs a warning and returns False."""
    if switch_admission.pin_blocks_peer_cooperation(budget, _engines_lru, _engines):
        from localm.debuglog import logger as _dbg
        _dbg.warning(
            "pinned_models=%s is the only reason no local model "
            "could be evicted for %s; NOT asking a peer instance "
            "to unload - deferring to the backend's own sizing",
            sorted(budget.pinned), budget.name)
        return False
    return await loop.run_in_executor(
        None,
        lambda: _attempt_cooperative_unload(
            needed_bytes=budget.needed_bytes,
            free_bytes=probe.free, asked=attempt.asked_peers))


async def _switch_claim_busy_victim(budget: switch_admission.LoadBudget,
                                    attempt: switch_admission.EvictionAttempt,
                                    *, preempt: bool, force: bool
                                    ) -> switch_admission.EvictionStep | None:
    """Cancel every generation on a busy resident model and claim it as the
    eviction victim, for an explicit switch only and once per load attempt
    (``switch_admission.busy_victim_candidate``).

    Returns None when there is no candidate. Without *force*, waits for its
    pins to clear (``_wait_for_pin_clear``) and returns a ``confirm_required``
    result step when they do not. Otherwise returns an EVICT step whose
    ``force_busy`` is *force*: evicted even if a pin is still held."""
    busy_name = switch_admission.busy_victim_candidate(
        budget, _engines_lru, _engines, preempt=preempt,
        already_attempted=attempt.busy_attempted)
    if busy_name is None:
        return None
    attempt.busy_attempted = True
    busy_engine = _engines[busy_name]
    residency.cancel_all(busy_name)
    if not force and not await _wait_for_pin_clear(busy_engine):
        return switch_admission.EvictionStep(
            switch_admission.RETURN_RESULT,
            result=switch_admission.busy_victim_confirm(
                budget.name, busy_name, _in_use_description(busy_name, busy_engine)))
    return switch_admission.EvictionStep(
        switch_admission.EVICT, victim=busy_name, force_busy=force)


async def _switch_wait_for_busy_victim(budget: switch_admission.LoadBudget,
                                       attempt: switch_admission.EvictionAttempt,
                                       *, preempt: bool,
                                       on_status: Optional[Callable[[str], None]] = None
                                       ) -> switch_admission.EvictionStep | None:
    """For a load that is not an explicit switch, wait up to
    ``_BUSY_VICTIM_IDLE_WAIT_S`` for the busy resident model
    ``switch_admission.idle_wait_candidate`` names to finish its requests,
    once per load attempt. Nothing is cancelled.

    Returns an EVICT step for that model when it went idle, a REPROBE step when
    another request removed it or began unloading it during the wait, else None
    (no candidate, or still busy when the wait ended). *on_status* is called
    with ``WAITING_FOR_MODEL_STATUS`` before the wait and
    ``LOADING_MODEL_STATUS`` after it."""
    from localm.debuglog import logger as _dbg
    busy_name = switch_admission.idle_wait_candidate(
        budget, _engines_lru, _engines, preempt=preempt,
        already_waited=attempt.busy_waited)
    if busy_name is None:
        return None
    attempt.busy_waited = True
    busy_engine = _engines[busy_name]
    _dbg.info("switch_engine: '%s' needs the VRAM '%s' holds; waiting up to %.0fs "
              "for '%s' to finish its request(s) so it can be evicted",
              budget.name, busy_name, _BUSY_VICTIM_IDLE_WAIT_S, busy_name)
    if on_status is not None:
        on_status(WAITING_FOR_MODEL_STATUS)
    try:
        idle = await _wait_for_pin_clear(busy_engine, timeout=_BUSY_VICTIM_IDLE_WAIT_S,
                                         poll_interval=0.25)
    finally:
        if on_status is not None:
            on_status(LOADING_MODEL_STATUS)
    if not idle:
        _dbg.info("switch_engine: '%s' is still busy after %.0fs; '%s' loads beside "
                  "it", busy_name, _BUSY_VICTIM_IDLE_WAIT_S, budget.name)
        return None
    if _engines.get(busy_name) is not busy_engine or not _is_reusable(busy_engine):
        _dbg.info("switch_engine: '%s' was freed by another request while '%s' "
                  "waited for it", busy_name, budget.name)
        return switch_admission.EvictionStep(switch_admission.REPROBE)
    return switch_admission.EvictionStep(switch_admission.EVICT, victim=busy_name)


def _switch_detach_victim(victim: str, engine, *, activate: bool,
                          keep_sem: bool = False) -> bool:
    """Mark eviction victim *victim* ``unloading`` and remove it from
    ``_engines``, ``_engines_lru``, ``_inference_sems`` (unless *keep_sem*) and
    the active-model pointers, before its native unload. A request then misses
    get_engine's fast path for it, and a concurrent switch to *victim* builds a
    fresh engine. With *activate* False, an active victim is kept in
    ``_last_active_model_name`` as the name an unnamed request resolves to.
    Returns whether *victim* was the active model.

    *keep_sem* is for a caller that holds *victim*'s own semaphore and reloads
    it: a request for *victim* then waits on that semaphore instead of building
    a second engine.

    Must run with no await since the victim's pin was last checked. See
    test_eviction_victim_not_pinnable_during_native_free."""
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
    was_active = _active_model_name == victim
    engine.unloading = True
    _engines.pop(victim, None)
    if victim in _engines_lru:
        _engines_lru.remove(victim)
    if not keep_sem:
        _inference_sems.pop(victim, None)
    if _active_model_name == victim:
        if not activate:
            _last_active_model_name = victim
        _active_model_name = None
    if _engine is engine:
        _engine = None
        _inference_sem = None
    return was_active


class VictimRelease(NamedTuple):
    """The outcome of freeing one eviction victim (``_switch_free_victim``).

    ``released`` is ``wait_for_vram_release``'s verdict: True once free VRAM
    rose, False when it did not rise before the wait ended, None when it could
    not be verified (no measurable reading, or a process-scoped one that did
    not see a rise). ``before`` and ``after`` are the free-VRAM readings around
    the free in bytes (None when unmeasurable), ``seconds`` the time from the
    start of the unload to the verdict, ``expected`` whether the victim was
    expected to free enough VRAM for the wait to see it and ``extended``
    whether the longer second wait ran."""

    victim: str
    released: Optional[bool]
    before: Optional[int]
    after: Optional[int]
    seconds: float
    expected: bool = True
    extended: bool = False

    def describe(self) -> str:
        """``evicting '<victim>': <outcome>`` for a log line."""
        if self.released is True:
            outcome = "VRAM release confirmed"
        elif self.released is False and not self.expected:
            outcome = "no VRAM release seen, none expected"
        elif self.released is False:
            outcome = "VRAM release NOT confirmed"
        else:
            outcome = "VRAM release could not be verified"
        if self.before is not None and self.after is not None:
            outcome += (f" ({self.before // 1024 ** 2} -> {self.after // 1024 ** 2} MB free"
                        f" in {self.seconds:.1f}s)")
        else:
            outcome += f" (after {self.seconds:.1f}s)"
        return f"evicting '{self.victim}': {outcome}"


def _victim_vram_estimate(engine) -> int:
    """Bytes of VRAM *engine*'s current load is expected to hold: the
    ``residency.required_vram_bytes`` of its model files, scaled by the share
    of its layers on the GPU when ``gpu_placement`` reports it. 0 when its
    model path is unknown, does not exist or cannot be read (logged at DEBUG).
    Reads the filesystem; never raises OSError."""
    from pathlib import Path
    path = getattr(engine, "model_path", None)
    try:
        if not path or not Path(path).exists():
            return 0
        size = residency.model_footprint_bytes(path)
    except OSError as exc:
        from localm.debuglog import logger as _dbg
        _dbg.debug("switch_engine: could not size %s for its VRAM release: %s", path, exc)
        return 0
    placement = getattr(engine, "gpu_placement", None)
    if isinstance(placement, dict) and placement.get("gpu_layers_total"):
        size = size * placement.get("gpu_layers_offloaded", 0) // placement["gpu_layers_total"]
    return residency.required_vram_bytes(size)


async def _switch_free_victim(loop, victim: str, engine,
                              probe: switch_admission.VramProbe,
                              needed: Optional[int] = None,
                              extra_wait: bool = True) -> VictimRelease:
    """Natively unload detached victim *victim* off the event loop, then wait
    for its VRAM to be released when *probe* was measurable, so the next
    reading is not stale. *victim* is in ``_evicting_names`` for the whole
    free, and is removed again even when ``unload()`` raises.

    A release is expected when ``_victim_vram_estimate`` is at least the rise
    ``wait_for_vram_release`` looks for. When the first wait sees none, a
    release was expected, *extra_wait* is True, the reading is device-global
    and free VRAM is still below *needed* (the bytes the pending load needs), it
    waits up to ``_VICTIM_RELEASE_EXTRA_WAIT_S`` more. A process-scoped reading
    that saw no rise reports the release as not verifiable.

    Returns the ``VictimRelease``. An expected release that was not confirmed
    is logged at WARNING, one that could not be verified or needed the extra
    wait at INFO, any other outcome at DEBUG."""
    from localm.debuglog import logger as _dbg
    from localm.vram import _MIN_RELEASE_RISE, wait_for_vram_release

    started = time.monotonic()
    released, after = None, None
    extended = False
    _evicting_names.add(victim)
    try:
        expected = await loop.run_in_executor(None, _victim_vram_estimate, engine)
        expect_rise = expected >= _MIN_RELEASE_RISE
        await loop.run_in_executor(None, engine.unload)
        if probe.measurable:
            reader = _probe_free_reader(probe)
            released, after = await loop.run_in_executor(
                None, lambda: wait_for_vram_release(reader, before_bytes=probe.free))
            if (released is False and expect_rise and extra_wait
                    and not probe.process_scoped
                    and needed is not None and after is not None and after < needed):
                extended = True
                released, after = await loop.run_in_executor(
                    None, lambda: wait_for_vram_release(
                        reader, before_bytes=probe.free,
                        timeout_s=_VICTIM_RELEASE_EXTRA_WAIT_S))
            if released is False and probe.process_scoped:
                released = None
    finally:
        _evicting_names.discard(victim)
    result = VictimRelease(victim, released, probe.free, after,
                           time.monotonic() - started, expect_rise, extended)
    if released is False and expect_rise:
        _dbg.warning("switch_engine: %s; free VRAM did not rise, so the next "
                     "reading may still count its memory", result.describe())
    elif released is None or extended:
        _dbg.info("switch_engine: %s%s", result.describe(),
                  " after an extended wait" if extended else "")
    else:
        _dbg.debug("switch_engine: %s", result.describe())
    return result


async def _switch_load(loop, name: str, engine, *, preempt: bool) -> Optional[dict]:
    """Run ``engine.load()`` off the event loop with a fresh cancel event.

    Returns None when the load succeeded, a ``superseded`` result when a newer
    explicit switch cancelled it, or a ``cancelled`` result for any other
    ``ModelLoadCancelled``. Raises HTTPException 503 with the backend's message
    when ``load()`` raises RuntimeError.

    Every load installs a new event on the engine, which may still hold the
    set event of an earlier superseded switch. See
    test_api_load_of_reused_engine_after_preempted_switch_succeeds. Only an
    explicit switch (*preempt*) publishes it as ``_switch_cancel`` for the
    duration of the load, so only a newer explicit switch can abort it.

    A load that raises is recorded in ``_routing_latch`` (a cancelled or
    superseded one is not) and a load that succeeds clears the model's record.
    The load fingerprint the record is made under is computed off the event
    loop, before ``engine.load`` starts."""
    global _switch_cancel, _switch_loading
    fingerprint = None
    cancel = threading.Event()
    if hasattr(engine, "set_load_cancel"):
        engine.set_load_cancel(cancel)
    if preempt:
        _switch_cancel = cancel
        _switch_loading = name
    try:
        fingerprint = await loop.run_in_executor(
            None, _routing_latch.fingerprint, name)
        await loop.run_in_executor(None, engine.load)
    except ModelLoadCancelled as e:
        if preempt and _switch_desired != name:
            return {"status": "superseded", "model": name, "by": _switch_desired}
        return {"status": "cancelled", "model": name, "reason": str(e)}
    except RuntimeError as exc:
        _routing_latch.record_failure(name, exc, fingerprint=fingerprint)
        raise HTTPException(503, f"Failed to load '{name}': {exc}") from exc
    except Exception as exc:
        _routing_latch.record_failure(name, f"{type(exc).__name__}: {exc}",
                                      fingerprint=fingerprint)
        raise
    finally:
        if preempt and _switch_cancel is cancel:
            _switch_cancel = None
            _switch_loading = None
    _routing_latch.record_success(name)
    return None


def _switch_commit(name: str, engine, sem, *, activate: bool, on_active) -> None:
    """Register freshly loaded *engine* as resident model *name*: seed its
    ``_last_activity_per_model`` idle clock, append it to ``_engines_lru`` and,
    with *activate* or when nothing else would answer an unnamed request, make
    it the active model (``_last_active_model_name`` cleared, *on_active*
    called)."""
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
    _engines[name] = engine
    _last_activity_per_model[name] = time.monotonic()
    _engines_lru.append(name)
    if activate or _resolve_unnamed_model_name() is None:
        _active_model_name = name
        _last_active_model_name = None
        _engine = engine
        _inference_sem = sem
        if on_active is not None:
            on_active(name)


class PlacementHeal(NamedTuple):
    """Why a load left partly on the CPU may get a full placement later
    (``_record_placement_heal``): ``blockers`` are the resident models, other
    than the loaded one and the ``pinned_models``, that held VRAM when it
    loaded beside them; ``release_unconfirmed`` is True when an eviction made
    for it did not see the VRAM release it expected."""

    blockers: frozenset
    release_unconfirmed: bool


def _record_placement_heal(name: str, engine, pinned, evictions, *,
                           allowed: bool, deferred: bool) -> None:
    """Set ``engine.placement_heal`` after switch_engine committed a load of
    *name*: a ``PlacementHeal`` when the load landed partly on the CPU with its
    layer count sized from free VRAM (``Engine.gpu_sizing`` mode "auto") and
    that sizing put the part there: fewer layers than all on the GPU, or routed
    experts it chose to keep in system RAM (``n_cpu_moe_auto``),
    *allowed* is True and it has a blocker or an unconfirmed release (from
    *evictions*); None otherwise. Resident models are blockers only when
    *deferred* (the load went ahead below the whole-model estimate because
    nothing more could be evicted); models in *pinned* never are."""
    placement = getattr(engine, "gpu_placement", None)
    sizing = getattr(engine, "gpu_sizing", None)
    heal = None
    if (allowed and isinstance(placement, dict) and placement.get("degraded")
            and isinstance(sizing, dict) and sizing.get("mode") == "auto"
            and (placement.get("gpu_layers_offloaded", 0) < placement.get("gpu_layers_total", 0)
                 or sizing.get("n_cpu_moe_auto"))):
        blockers = (frozenset(n for n in _engines_lru if n != name and n not in pinned)
                    if deferred else frozenset())
        unconfirmed = any(e.released is False and e.expected for e in evictions)
        if blockers or unconfirmed:
            heal = PlacementHeal(blockers, unconfirmed)
    engine.placement_heal = heal


def _placement_heal_due(name: str, engine) -> bool:
    """Whether resident *engine* (model *name*) should be reloaded before it
    serves a request: it is still placed partly on the CPU, no request holds
    it, and its ``placement_heal`` names an unconfirmed release or a blocker
    that is now gone, unloaded or idle. Reads state only."""
    heal = getattr(engine, "placement_heal", None)
    if not isinstance(heal, PlacementHeal):
        return False
    placement = getattr(engine, "gpu_placement", None)
    if not (isinstance(placement, dict) and placement.get("degraded")):
        return False
    if getattr(engine, "active_requests", 0) != 0:
        return False
    if heal.release_unconfirmed:
        return True
    for blocker in heal.blockers:
        other = _engines.get(blocker)
        if other is None or not other.loaded:
            return True
        if (getattr(other, "active_requests", 0) == 0
                and getattr(other, "unloading", False) is not True):
            return True
    return False


async def _switch_free_for_reload(loop, name: str, engine,
                                  probe: switch_admission.VramProbe,
                                  needed: int) -> VictimRelease:
    """Unload resident *engine* (model *name*, already detached with its
    semaphore kept) so switch_engine can load it again at a better placement.
    Logs why at INFO, frees it like an eviction victim (``_switch_free_victim``
    against *probe*, the VRAM reading taken before the detach, for a load that
    needs *needed* bytes) and clears its ``unloading`` flag and
    ``placement_heal``. Returns the ``VictimRelease``."""
    from localm.debuglog import logger as _dbg
    heal = getattr(engine, "placement_heal", None)
    placement = getattr(engine, "gpu_placement", None) or {}
    why = ("its earlier eviction's VRAM release was not confirmed"
           if heal is not None and heal.release_unconfirmed
           else "a model that held the VRAM it lacked is idle or gone")
    from localm.inference.engine import describe_gpu_placement
    _dbg.info("switch_engine: reloading '%s' (%s) for a full GPU placement: %s",
              name, describe_gpu_placement(placement), why)
    try:
        return await _switch_free_victim(loop, name, engine, probe, needed=needed)
    finally:
        engine.unloading = False
        engine.placement_heal = None


def _describe_load_placement(name: str, engine, evictions=()) -> str:
    """One line naming where *engine*'s load of *name* placed its layers
    (``Engine.gpu_placement``), how the layer count was chosen
    (``Engine.gpu_sizing``), the mmap note when the load has one
    (``Engine.mmap_state``) and the outcome of each eviction that made room
    for it (``VictimRelease.describe``)."""
    placement = getattr(engine, "gpu_placement", None)
    sizing = getattr(engine, "gpu_sizing", None)
    placement = placement if isinstance(placement, dict) else None
    sizing = sizing if isinstance(sizing, dict) else None
    if placement:
        from localm.inference.engine import describe_gpu_placement
        where = describe_gpu_placement(placement)
        if placement.get("degraded"):
            where += " (slower)"
    else:
        where = "GPU layer placement not reported by this backend"
    parts = [f"loaded '{name}': {where}"]
    if sizing:
        detail = f"n_ctx {sizing.get('n_ctx')}"
        if sizing.get("mode") == "auto" and sizing.get("free_bytes") is not None:
            mb = 1024 ** 2
            need = (sizing.get("model_bytes", 0) + sizing.get("kv_bytes", 0)
                    + sizing.get("overhead_bytes", 0))
            detail += (f", sized against {sizing['free_bytes'] // mb} MB free VRAM"
                       f" (full offload needs ~{need // mb} MB: weights "
                       f"{sizing.get('model_bytes', 0) // mb} + KV "
                       f"{sizing.get('kv_bytes', 0) // mb} + overhead "
                       f"{sizing.get('overhead_bytes', 0) // mb})")
            if sizing.get("cause"):
                detail += f" - {sizing['cause']}"
        elif sizing.get("mode") == "unmeasurable":
            detail += f", free VRAM not measurable, n_gpu_layers {sizing.get('layers')}"
        elif sizing.get("mode") == "configured":
            detail += f", n_gpu_layers {sizing.get('layers')} as configured"
        else:
            detail += f", n_gpu_layers {sizing.get('layers')}"
        parts.append(detail)
    mmap_state = getattr(engine, "mmap_state", None)
    if isinstance(mmap_state, dict) and mmap_state.get("mmap_note"):
        parts.append(mmap_state["mmap_note"])
    parts.extend(e.describe() for e in evictions)
    return "; ".join(parts)


def _log_switch_placement(name: str, engine, evictions=()) -> None:
    """Log ``_describe_load_placement`` for a load switch_engine just
    committed: at WARNING when fewer than all layers landed on the GPU, else
    at INFO."""
    from localm.debuglog import logger as _dbg
    placement = getattr(engine, "gpu_placement", None)
    line = "switch_engine: " + _describe_load_placement(name, engine, evictions)
    if isinstance(placement, dict) and placement.get("degraded"):
        _dbg.warning(line)
    else:
        _dbg.info(line)


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
    (unload_all_models keeps its Engine in _engines for lazy reload, but used
    to lose its name) would otherwise silently resolve back to the STARTUP
    model instead of the one actually in use."""
    return _active_model_name or _last_active_model_name or _default_model_name


def _model_is_pinned(model_name: str | None) -> bool:
    """Whether the request named a model EXPLICITLY.

    The one discriminator behind the never-swap-an-explicit-pin rule, and the
    same test get_engine already uses to decide whether to resolve an unnamed
    request (peer_routing in routes/chat.py derives the same one independently).
    It lives here as a named function so capability routing cannot drift from
    the resolution path it gates: an empty/absent model, or the "localm"
    sentinel, is "no preference"; anything else is the user's choice."""
    name = (model_name or "").strip()
    return bool(name) and name != "localm"


def plan_capability_route(model_name: str | None, messages: list,
                          required_capabilities=None, *, pin_model=None,
                          min_context=None):
    """The capability-routing decision for one request, without applying it.

    Blocking (registry read plus per-candidate capability probes), so an async
    caller must run it in an executor.

    Returns a ``RoutingDecision``. Its ``resolved`` differs from ``current``
    only when the request is not pinned: *pin_model* decides that when set,
    otherwise ``_model_is_pinned`` does, and ``plan_route`` refuses to move a
    pinned one, so a pinned request gets a decision that reports the gap and
    changes nothing. With ``pin_model=False`` a named *model_name* is the
    request's preferred model and is ``current``.

    Needs are derived from the request wherever the request already states them
    - an image part means vision, the prompt's estimated size means a context
    window - plus *required_capabilities* and *min_context* for what nothing in
    an OpenAI-shaped request can express. A loaded engine confirmed to accept
    images does not count vision as a gap, whatever the registry records.

    Models with an accepted peer route count as resident, since answering with
    one loads nothing here.

    A model whose last load failed is left out of the candidates while
    ``_routing_latch`` holds it, and the decision's ``skipped`` names it. The
    model the request names, the active one and a model with an accepted peer
    route are never affected. The latch is read only once the request has a
    gap and is not pinned."""
    from localm import peer_routing
    from localm.inference import capability_routing as _cr
    from localm.model_manager import capabilities as _caps

    needs = _cr.request_needs(messages or [], required=required_capabilities or (),
                              min_context=min_context)
    named = (model_name or "").strip()
    if named == "localm":
        named = ""
    pinned = bool(pin_model) if pin_model is not None else _model_is_pinned(model_name)
    current = named or _resolve_unnamed_model_name()
    # list() first: this runs in an executor thread while the event loop can
    # be adding or evicting engines, and iterating the live dict would raise.
    live = list(_engines.items())
    resident = [n for n, e in live if getattr(e, "loaded", False)]
    peer_routes = peer_routing.list_routes()
    resident += [n for n in peer_routes if n not in resident]
    known = {}
    cur_engine = dict(live).get(current) if current else None
    if (cur_engine is not None and getattr(cur_engine, "loaded", False)
            and getattr(cur_engine, "supports_images", False) is True):
        known[_caps.VISION] = True

    def skip_set():
        return {n: s for n, s in _routing_latch.skipped().items()
                if n not in peer_routes}

    return _cr.plan_route(current, needs, pinned=pinned, resident=resident,
                          current_known=known, skip=skip_set,
                          mode=_cr.configured_mode())


async def get_engine(model_name: str | None, *, load: bool = True,
                     activate: bool = True, skip_if_latched: bool = False,
                     on_status: Optional[Callable[[str], None]] = None) -> Engine:
    """Resolve the engine for *model_name*, loading it if necessary.

    With ``load=False`` the resolved engine is returned WITHOUT forcing a load -
    for callers like /v1/embeddings whose backend may not need the model resident
    at all (a GGUF backend embeds via the dedicated embedder). The
    caller decides whether to load. Registration/resolution (and its 404) still
    apply.

    With ``activate=False`` the engine serves this one request without becoming
    the model an unnamed request resolves to: the active/default model is left
    as it was (see ``switch_engine``). Capability routing uses this.

    With ``skip_if_latched`` a model that is not resident is not loaded when its
    last load failed and that failure still applies: ``LoadSkipped`` is raised.
    Capability routing sets it for the models it chose.

    A resident model left partly on the CPU whose ``_placement_heal_due`` holds
    is passed to switch_engine, which reloads it, instead of being returned as
    it is. *on_status* is passed to switch_engine.
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

    # A name that resolved to nothing (an empty/"localm" request with no active,
    # no remembered last-active, AND no default model - e.g. `gui --no-model`
    # with a populated registry and nothing ever loaded) must be an honest
    # 503, not fall through to switch_engine(None) -> get_model_info(None) ->
    # Path(None) TypeError -> HTTP 500.
    if not name:
        raise HTTPException(503, "No model is loaded and none was specified. "
                            "Load a model first or name one explicitly.")

    # `not unloading`: never hand back (and let the caller pin) an engine that an
    # eviction/unload path is mid-freeing - that is the pin-arrives-during-the-
    # unload-await race. An engine flagged unloading falls through to switch_engine
    # below, which reloads it cleanly under its per-model semaphore (the unloader
    # holds that semaphore for the native free, so the reload serializes AFTER it).
    if (name in _engines and _engines[name].loaded
            and getattr(_engines[name], "unloading", False) is not True
            and not (load and _placement_heal_due(name, _engines[name]))):
        if name in _engines_lru:
            _engines_lru.remove(name)
        _engines_lru.append(name)
        if activate:
            _active_model_name = name
            _engine = _engines[name]
            _inference_sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        return _engines[name]

    if not load:
        # Return the engine object WITHOUT loading it: reuse a tracked (possibly
        # unloaded) engine, else build a fresh one via the factory. Does NOT go
        # through switch_engine, so no model is loaded and nothing is evicted.
        return _engines.get(name) or _engine_factory(name)

    res = await switch_engine(name, _engine_factory, preempt=False,
                              activate=activate, skip_if_latched=skip_if_latched,
                              on_status=on_status)
    if res.get("status") == "superseded":
        raise HTTPException(503, f"Model load was superseded by a newer request: {res.get('by')}")
    if res.get("status") == "cancelled":
        raise HTTPException(503, f"Model load was cancelled: {res.get('reason')}")

    return _engines[name]


def _add_vram_fields(result: dict, *, before, released, after, before_fresh: bool,
                     before_scope=None) -> None:
    """Add vram_freed/vram_before_bytes/vram_after_bytes to *result* when
    measurable (unchanged from before), plus an honest flag when the reading
    cannot be presented as current fact, rather than asserting a wrong
    number.

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
      is null rather than a false "VRAM did not drop", a claim a reading that
      never refreshed cannot support.
    - NOT DEVICE-SCOPED (``before_scope`` is FREE_SCOPE_PROCESS): the probe was
      perfectly fresh, but on this platform the driver reports only the CALLING
      process's own allocations. Since every GGUF load runs in an isolated worker
      subprocess (backends/gguf.py), the model's VRAM is in another process and
      simply absent from the number, so before/after read byte-identical, and
      vram_freed false, across a load/unload cycle that did free VRAM.

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


async def _unload_engine_off_loop(loop, engine, on_unloaded) -> None:
    """Run ``engine.unload()`` on the loop's default executor and call
    ``on_unloaded()`` on the loop once it has completed.

    Cancellation of the awaiting task: an unload still queued behind other
    executor work is cancelled and never runs; an unload already running is
    waited for (the caller's ``engine.unloading`` flag and per-model semaphore
    stay held), ``on_unloaded()`` still runs when it succeeded, and the
    CancelledError is then re-raised. A failure of a waited-for unload is
    reported at WARNING. See test_cancelling_a_running_unload_keeps_its_bookkeeping."""
    started = threading.Event()
    abandoned = threading.Event()
    finished = asyncio.Event()
    outcome: dict = {}

    def _run():
        started.set()
        try:
            if abandoned.is_set():
                return
            engine.unload()
            outcome["unloaded"] = True
        except BaseException as e:
            outcome["exc"] = e
            raise
        finally:
            try:
                loop.call_soon_threadsafe(finished.set)
            except RuntimeError:
                # The loop is closed: no coroutine is left to wake.
                pass

    try:
        await loop.run_in_executor(None, _run)
    except asyncio.CancelledError:
        abandoned.set()
        if started.is_set():
            # Further cancellation requests arriving while the native unload
            # runs are absorbed; the one caught above is re-raised afterwards.
            while True:
                try:
                    await finished.wait()
                    break
                except asyncio.CancelledError:
                    continue
            if "exc" in outcome:
                from localm.debuglog import logger as _dbg
                _dbg.warning("unloading %s failed after its caller stopped waiting: %s",
                             getattr(engine, "display_name", engine), outcome["exc"])
            if outcome.get("unloaded"):
                on_unloaded()
        raise
    if outcome.get("unloaded"):
        on_unloaded()


async def _wait_for_pin_clear(engine, *, timeout: float = 2.0,
                              poll_interval: float = 0.05) -> bool:
    """Poll ``engine.active_requests`` until it reads 0, or *timeout* elapses.
    Meant to run right after ``residency.cancel_all()``: a generation that
    was only still running because nobody had told it to stop (the caller's
    own just-cancelled request, most commonly) clears within a token or two,
    so this resolves the common "still generating" complaint without ever
    surfacing a confirm box for it. Returns whether the pin is clear."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        active = getattr(engine, "active_requests", 0)
        if not (isinstance(active, int) and active > 0):
            return True
        await asyncio.sleep(poll_interval)
    active = getattr(engine, "active_requests", 0)
    return not (isinstance(active, int) and active > 0)


def _coder_sessions_using(name: str) -> list[str]:
    """Labels for live coder sessions currently BUSY and bound to model
    *name* - informational only, for a confirm_required message. Empty when
    no coder GUI is mounted (an --isolated/API-only instance never sets
    _coder_session_manager) or none match. Never raises: a best-effort
    attribution probe must not turn an unload into a 500."""
    mgr = _coder_session_manager
    if mgr is None:
        return []
    try:
        infos = mgr.list(is_owner=True)
    except Exception:
        return []
    return [f"a coder session in {info.get('cwd') or info.get('id')}"
           for info in infos if info.get("busy") and info.get("model") == name]


def _in_use_description(name: str, engine) -> str:
    """Human-readable "who is using this" for a confirm_required response.
    Best-effort: names what is KNOWN (a busy coder session bound to this
    model) and folds whatever residency can only see as a bare pin count
    into a generic remainder, rather than double-counting a coder session's
    own request against the raw active_requests total."""
    users = _coder_sessions_using(name)
    active = getattr(engine, "active_requests", 0)
    remaining = active if isinstance(active, int) else 0
    other = max(0, remaining - len(users))
    parts = list(users)
    if other:
        parts.append(f"{other} other active request{'s' if other != 1 else ''}")
    return " and ".join(parts) if parts else "still in use"


async def unload_all_models(*, force: bool = False) -> dict:
    """Release every currently-loaded model from GPU/CPU memory and wait until
    VRAM is actually reclaimed (see ``localm.vram.wait_for_vram_release`` - the
    driver-hang guard: otherwise a media model can load on top of a
    not-yet-freed LLM and exceed total VRAM).

    Extracted from the ``POST /v1/models/unload`` route so it has exactly ONE
    implementation, reused by two callers with two different auth models: the
    owner-scoped ``/v1/models/unload`` route (``MODELS_WRITE``), and the
    requester-vouched ``POST /v1/instances/cooperate-unload`` (a
    sibling localm instance asking THIS one to free VRAM - multi-instance GPU
    coordination, see ``localm.gpu_registry``). Behavior is unchanged from the
    original inline implementation."""
    global _active_model_name, _last_active_model_name, _engine, _inference_sem
    loop = asyncio.get_running_loop()
    from localm.vram import (_live_free_vram_bytes, _vram_free_reading,
                             wait_for_vram_release)
    from localm.inference import embedder as _embedder_mod

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    unloaded_models = []
    skipped_in_use = []
    confirm_required: dict[str, str] = {}

    def _reset_active_pointers():
        global _active_model_name, _last_active_model_name, _engine, _inference_sem
        if _active_model_name:
            # The Engine stays in _engines above for exactly this: a lazy
            # reload on the next request. Keep its NAME alive too, or nothing
            # can resolve an unnamed request back to it (see
            # _last_active_model_name / _resolve_unnamed_model_name).
            _last_active_model_name = _active_model_name
        _active_model_name = None
        _engine = None
        _inference_sem = None

    try:
        embedder_was_loaded = await _unload_engines_and_embedder(
            loop, _embedder_mod, unloaded_models, skipped_in_use,
            confirm_required=confirm_required, force=force)
    except asyncio.CancelledError:
        # The caller stopped waiting: an engine already unloaded above still
        # gets the pointer reset the normal path does below.
        if _active_model_name in unloaded_models:
            _reset_active_pointers()
        raise

    # Update compatibility pointers - but NOT if the active engine was a pinned
    # one left loaded above, since clearing it would strand the in-flight
    # request's active model.
    if _active_model_name not in skipped_in_use:
        _reset_active_pointers()

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
    if confirm_required:
        # Present only for a model that is STILL in use after cancel_all()
        # was given its grace period - describes what to tell the caller
        # before it retries this same call with force=True.
        result["confirm_required"] = confirm_required
    _add_vram_fields(result, before=before, released=released, after=after,
                     before_fresh=before_fresh, before_scope=before_scope)
    return result


async def _unload_engines_and_embedder(loop, _embedder_mod, unloaded_models,
                                       skipped_in_use, *,
                                       confirm_required: dict, force: bool = False) -> bool:
    """The releasing half of ``unload_all_models``: unload every loaded, unpinned
    chat engine (appending to *unloaded_models* / *skipped_in_use*), then the
    shared embedder. Returns whether the embedder was released."""
    for name in list(_engines.keys()):
        engine = _engines[name]
        if not engine.loaded:
            continue
        # Honor the in-flight-request pin, like the VRAM-eviction and
        # idle-unload paths: a pinned engine has a request generating (or about to)
        # against it. Unloading it would free VRAM the request immediately reloads
        # (so the reported "freed" total is a lie) and race a use-after-unload; skip
        # it and report it as still in use. Gate on isinstance(int) exactly like
        # _pin/_unpin: a non-int active_requests (a bare test double) is "not pinned".
        active = getattr(engine, "active_requests", 0)
        if isinstance(active, int) and active > 0:
            residency.cancel_all(name)
            if not force and not await _wait_for_pin_clear(engine):
                skipped_in_use.append(name)
                confirm_required[name] = _in_use_description(name, engine)
                continue
            # Either the grace period cleared it (the common case: the
            # caller's own just-stopped generation), or force=True proceeds
            # regardless - engine.unload() below still forcibly kills the
            # worker if something is genuinely still running.
        sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        # Flag BEFORE acquiring the semaphore so no request that arrives after the
        # pin check above can take get_engine's fast path and pin this engine while
        # we free it (the pin-arrives-during-the-unload-await window); such a
        # request then blocks on the same semaphore and reloads cleanly afterwards.
        # Cleared in finally so the kept-in-_engines engine reloads lazily.
        engine.unloading = True
        try:
            async with sem:
                def _unloaded(name=name):
                    unloaded_models.append(name)
                    if name in _engines_lru:
                        _engines_lru.remove(name)
                await _unload_engine_off_loop(loop, engine, _unloaded)
        finally:
            engine.unloading = False

    # Release the shared embedder too - a separate lifecycle from _engines (see
    # localm.inference.embedder's module docstring): it is loaded independently
    # by RAG/memory/coder-episode callers via get_embedder(), never through
    # switch_engine, so nothing else in "Unload all" would reach it.
    #
    # loaded_dim()/active_requests() MUST run in the executor, not directly on
    # this coroutine: get_embedder() can hold embedder._LOCK for the full
    # duration of an IsolatedEmbedder native/subprocess load (up to its load
    # timeout), and both of those accessors block on that same lock. A
    # synchronous call here would freeze the WHOLE event loop - every other
    # request this server is serving - for that entire window, not just this
    # coroutine. Executor-offloading them, like every other blocking call in
    # this function, keeps the wait local to this one coroutine instead.
    embedder_was_loaded = False
    embedder_dim = await loop.run_in_executor(None, _embedder_mod.loaded_dim)
    if embedder_dim is not None:
        # Honor the in-flight-request pin for the embedder too, exactly like
        # the chat-engine loop above: a request mid-embed() must not have its
        # embedder (and the isolated worker process it is waiting on) freed out
        # from under it. reset_embedder(force=False) checks active_requests()==0
        # and clears the embedder in ONE locked step, never two separate
        # executor calls, which would leave a TOCTOU window since
        # IsolatedEmbedder.embed() pins active_requests without taking
        # embedder._LOCK. Skip it and report it alongside the pinned chat
        # engines instead of a lying "unloaded".
        cleared = await loop.run_in_executor(
            None, functools.partial(_embedder_mod.reset_embedder, force=False))
        if cleared:
            embedder_was_loaded = True
        else:
            skipped_in_use.append("embedding model")
    return embedder_was_loaded


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

    # Honor the in-flight-request pin: a request mid-embed() must not have its
    # embedder freed out from under it. Report it as still in use instead of a
    # lying "unloaded", matching unload_one_model's own pinned-chat-engine check
    # just below.
    #
    # TWO layers, mirroring switch_engine's own chat-engine eviction (its LRU
    # scan's active_requests==0 check, THEN a synchronous defensive re-check
    # right before it commits): the cheap active_requests() precheck here avoids
    # paying for the _vram_free_reading() hardware probe below on the common busy
    # case, since that probe is NOT executor-offloaded and can block this whole
    # single-threaded event loop for up to discover._GPU_PROBE_DEADLINE seconds.
    # reset_embedder(force=False) is what actually authorizes the close,
    # atomically re-checking active_requests()==0 under embedder._LOCK in the
    # SAME step as the close - never a separate unlocked active_requests() call
    # ahead of an unconditional reset_embedder(), which would leave a TOCTOU
    # window since IsolatedEmbedder.embed() pins active_requests without taking
    # embedder._LOCK - so a pin arriving in the gap between this precheck and
    # the actual close is still caught. Both calls are executor-offloaded for
    # the same reason as loaded_path() above: each blocks on embedder._LOCK,
    # which get_embedder() can hold for the length of an IsolatedEmbedder load,
    # and a synchronous call here would freeze the whole event loop.
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
    return result


async def unload_one_model(name: str, *, force: bool = False) -> dict:
    """Release ONE currently-loaded model from GPU/CPU memory, leaving any
    other loaded models untouched - the targeted counterpart to
    ``unload_all_models()`` (same VRAM-release-wait + gpu-registry-sync
    behavior, just scoped to a single engine). Clears the active-model
    pointers only when *name* was the active model, so unloading a background
    (loaded-but-not-active) model never disturbs the one actually serving
    requests. A *name* that is registered but not currently loaded is a
    no-op success (idempotent, matching unload_all_models()'s "nothing to do"
    case), not an error - callers that need to reject an unknown model name
    outright should check the registry themselves before calling this.

    A pinned engine is no longer refused outright: ``residency.cancel_all``
    is broadcast first, and a short grace period lets a generation that was
    only running because nobody had told it to stop (the caller's own
    just-cancelled request, most commonly) clear on its own - the common
    case resolves with no confirm box at all. If it is still in use after
    that, the owner's explicit action is final: pass *force* to evict
    immediately regardless of what is running (the underlying
    ``engine.unload()`` forcibly kills the isolated worker if it does not
    exit within its own grace period); without it, this returns
    ``{"status": "confirm_required", ...}`` describing what was found using
    it, for the caller to show a confirmation before retrying with force."""
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
    # Honor the in-flight-request pin: an engine a request is
    # generating on must not be unloaded out from under it (it would reload it
    # anyway, making the "freed" report a lie). Report it as in use, not unloaded.
    # isinstance(int) guard matches _pin/_unpin (a bare test double is not pinned).
    active = getattr(engine, "active_requests", 0)
    if isinstance(active, int) and active > 0:
        residency.cancel_all(name)
        if not force and not await _wait_for_pin_clear(engine):
            return {"status": "confirm_required", "model": name,
                    "detail": _in_use_description(name, engine)}
        # Cleared during the grace period, or force=True - proceed below.

    _free = _live_free_vram_bytes

    before, before_fresh, before_scope = _vram_free_reading()
    sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
    # Flag BEFORE acquiring the semaphore so no request that arrives after the pin
    # check above can fast-path-pin this engine while we free it (the pin-arrives-
    # during-the-unload-await window); such a request blocks on the same
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
        if _active_model_name:
            # The Engine stays in _engines above for exactly this: a lazy
            # reload on the next request. Keep its NAME alive too, same as
            # unload_all_models, or nothing can resolve an unnamed request
            # back to it (see _last_active_model_name / _resolve_unnamed_model_name).
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
    that early return, because they are wrong the moment the registry moves
    regardless of what is in the engine map. Measured live, on a server
    started with the renamed model: a stale ``_default_model_name`` puts a
    ghost row for the old name into ``GET /v1/models`` (list_models adds the
    startup name when the registry lacks it) and makes ``get_engine``'s
    registration check at ``name != _default_model_name`` accept a request
    for a name the registry no longer has, instead of answering the honest
    404. ``_last_active_model_name`` is the same shape one step along:
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
    half at all, which is why ``localm rename`` asks a running server to call
    this rather than moving the registry entry behind its back.

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
    # Sanitizing happens server-side, so the collision check and the eventual
    # response must both speak the sanitized name, not the raw text the caller
    # sent: prechecking the raw name would let a collision through, and
    # answering with the raw name would name an entry that does not exist.
    new_name = _sanitize_name(new_name_raw)
    if new_name != model and new_name in registry:
        raise HTTPException(409, f"Name already taken: {new_name}")
    loop = asyncio.get_running_loop()
    try:
        renamed, notes = await loop.run_in_executor(
            get_plugin_executor(), rename_model_with_notes, model, new_name_raw)
    except Exception as e:
        raise HTTPException(400, f"Rename failed: {e}") from e
    if not renamed:
        # rename_model_with_notes distinguishes "vanished" from "name taken" in
        # its own console output, but only the bool crosses the executor
        # boundary - re-derive which race it lost.
        if model not in load_registry():
            raise HTTPException(404, f"Model not registered: {model}")
        raise HTTPException(409, f"Name already taken: {new_name}")
    # Synchronous, in-memory only (no await) - safe to call directly on the
    # event loop right after the executor call above returns.
    rekey_loaded_model(model, new_name)
    return {"status": "renamed", "model": model, "new_name": new_name,
            "notes": notes}


def loaded_engine_holding_model_file(model: str, registry: dict | None = None):
    """Whether removing registry entry *model* would delete a file that a LOADED
    engine in THIS PROCESS is holding. Returns None only when that is
    POSITIVELY RULED OUT; otherwise a
    :class:`~localm.model_manager.registry.ModelFileHold` naming the engine
    responsible.

    Turns this process's residents (``_engines`` plus the ``_engine``
    singleton, which can hold a startup engine outside the map) into
    ``(key, model_path)`` pairs and delegates the actual hold policy to
    :func:`localm.model_manager.registry.engine_holding_model_file` - the
    single implementation shared with the MCP server's own ``EngineCache``,
    so the two processes cannot independently drift on the same question.

    Pass *registry* to reuse a load the caller has already done. Does
    filesystem I/O, so callers on the event loop run it in the executor.
    """
    from localm.config import load_registry
    from localm.model_manager.registry import engine_holding_model_file

    reg = load_registry() if registry is None else registry
    # Snapshot: _engines is mutated by loads/evictions on the event loop while
    # this runs in a worker thread, and iterating it live would raise.
    engines = [(k, e) for k, e in list(_engines.items())
               if getattr(e, "loaded", False)]
    # _engine is normally the same object as _engines[active], but a startup
    # (`localm serve <path.gguf>`) or test-injected engine can sit outside the
    # map - and it is holding the file just as hard.
    if (_engine is not None and getattr(_engine, "loaded", False)
            and not any(e is _engine for _, e in engines)):
        engines.append((getattr(_engine, "display_name", "") or model, _engine))
    candidates = [(k, getattr(e, "model_path", None)) for k, e in engines]
    return engine_holding_model_file(model, reg, candidates)


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

        # The _last_activity fallback below is intentional, not a leftover: it is
        # only ever reached for an engine that was assigned to `_engine` directly
        # without going through switch_engine's registration (a test/script
        # setting _engine, or a genuinely single-model/direct-path startup with
        # an empty registry - see _switch_load_budget's docstring).
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
            # during-the-unload-await window). Cleared in finally so the
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
            # not lost. Only drop it from the LRU (no VRAM
            # while unloaded); keep its inference semaphore for a concurrent reload.
            if name in _engines_lru:
                _engines_lru.remove(name)

            if _engine is engine:
                _engine = None
            if _active_model_name == name:
                if _active_model_name:
                    # Same reasoning as unload_one_model/unload_all_models: capture
                    # the name before it is possibly cleared below, so a still-idle
                    # server with nothing left in _engines_lru can still resolve an
                    # unnamed request (see _last_active_model_name /
                    # _resolve_unnamed_model_name). Harmless when the LRU fallback
                    # below keeps a real active model instead, since that value
                    # wins over _last_active_model_name automatically.
                    _last_active_model_name = _active_model_name
                _active_model_name = _engines_lru[-1] if _engines_lru else None
                _engine = _engines[_active_model_name] if _active_model_name else None
                _inference_sem = _inference_sems.get(_active_model_name) if _active_model_name else None
                
            from localm.debuglog import logger as _dbg
            _dbg.info("idle-unload: freed %s after %ds idle (ttl=%ds); it reloads "
                      "on the next request", engine.display_name, idle_s, ttl)
            unloaded_any = True

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


async def _mmproj_backfill_once() -> None:
    """Run ``model_manager.sync_models_dir()`` (its default
    ``backfill_mmproj=True``) exactly once, off the event loop - a bounded
    one-shot, not a loop like its lifespan siblings. Any outcome is logged,
    never printed to a console."""
    loop = asyncio.get_running_loop()
    from localm.debuglog import logger as _dbg
    try:
        from localm.model_manager import sync_models_dir
        result = await loop.run_in_executor(None, sync_models_dir)
        if result.mmproj_backfilled:
            _dbg.info("vision-projector backfill: %d model(s) backfilled",
                      result.mmproj_backfilled)
        if result.note:
            _dbg.debug("model sync note: %s", result.note)
    except Exception as e:
        _dbg.debug("vision-projector backfill failed (continuing): %s", e)



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
                    # Cold start: the heartbeat task has not ticked even once
                    # yet (see the comment above _hb_monotonic's declaration).
                    # There is no prior tick to measure a stall AGAINST, so
                    # skip rather than dump against a fabricated baseline -
                    # the same "no reading yet, never a fake one" choice
                    # _loop_lag_seconds() makes.
                    #
                    # NOT COSMETIC: proven by reverting this guard and running
                    # this loop against a real cold start. Without it, `lag =
                    # time.monotonic() - _hb_monotonic` raises an uncaught
                    # TypeError (subtracting from None) on the very first
                    # poll, which crashes this daemon thread outright - a dead
                    # thread looks identical to a healthy quiet one from the
                    # outside, so hang detection would be silently disabled
                    # for the rest of the process with no signal to anyone.
                    # Do not remove this check as redundant.
                    continue
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


# Human-facing hooks for the hang ALARM (localm/inference/_hang_alarm.py -
# the detect/surface/recover pipeline, distinct from the forensic stack-dump
# thread above). The GUI replaces these with the native
# status window's own red-error/ready transitions via set_hang_surface();
# headless serve keeps the console default so the terminal running the
# server still shows the state change. logger.critical is already emitted by
# the alarm itself before calling these, so the hooks only need to be the
# human-visible half.
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
# async-task section (a plain thread cannot resolve it on its own).
_hang_dump_loop = None


def _hang_dump(reason: str) -> None:
    """Forensic stack snapshot on a NEW hang incident, into the same
    per-run trace file the stall watchdog uses - under the same privacy gate
    (_diagnostics_allowed), because it writes stack frames to disk. The alarm
    calls this once per distinct incident; detection/surfacing themselves are
    NOT gated on this (they write nothing sensitive anywhere).

    Two sections: every THREAD (faulthandler - where a wedged executor job is
    visible), then every asyncio TASK with its await stack, because a
    purely-async wedge (a coroutine parked
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
            # Task.get_stack() reports only the OUTERMOST coroutine's
            # suspension frame (it never follows cr_await), which for a
            # wedged request shows the middleware wrapper and hides the
            # handler line that actually matters. Walk the await chain by
            # hand so the record ends at the exact line the coroutine is
            # parked on, plus the primitive it is awaiting.
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
    release, VRAM-release wait) is lock-free and bounded by design, but
    "bounded" is a claim about code on a box that is provably wedged, so it
    gets a hard window, after which the re-exec happens anyway with only the
    steps that cannot block: the lock-free embedder-worker release (an
    orphaned worker survives execv holding VRAM), the crash-marker disarm (so
    the next boot does not misreport this recovery as a crash), a log flush,
    and the same fd non-inheritance marking (_mark_fds_noninheritable) the
    graceful path uses before its own os.execv."""
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
    # _do_restart ends in os.execv, which never returns - so reaching this
    # line at all means the graceful path raised or wedged. Force it.
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
    _mark_fds_noninheritable()
    _set_restart_env()
    os.execv(sys.executable, _execv_argv(_restart_argv(port)))


def _diagnostics_allowed() -> bool:
    """Whether localm may write an AUTOMATIC diagnostic trace right now: the
    log/full session modes, or privacy mode with ``keep_diagnostics`` on.
    Alias of :func:`localm.audit.diagnostics_allowed`, which also gates the
    crash guard's trace file and crash report (localm.bugreport)."""
    from localm.audit import diagnostics_allowed
    return diagnostics_allowed()


# Optional bearer-token auth - enabled when LOCALM_API_KEY is set.
_bearer_scheme = HTTPBearer(auto_error=False)

# The browser GUI authenticates with an HttpOnly session cookie whose value is
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
            # Suppressed: the inner app choked on the body cut short above
            # (e.g. invalid JSON at the truncation point). The request is
            # rejected as too large regardless of what its first
            # MAX_REQUEST_BODY_BYTES happened to contain.

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
            # left open indefinitely. Bound BOTH bytes
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
                except TimeoutError:
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


def _session_minted_by_owner_key(rec, token=None) -> bool:
    """Whether *rec* was minted by the OWNER KEY itself, as opposed to a minted
    (and therefore revocable) keystore key.

    Asked POSITIVELY and answered from the owner key alone. Two ways, in cost
    order, and both are proofs rather than inferences:

    1. the ``owner_key_minted`` stamp recorded at login (``sessions.create``);
    2. for a record written before that field existed, the recorded ``key_hash``
       still equalling the live owner key's digest. Only the owner key's own
       digest can match that, so a keystore key's session can never satisfy it -
       which is why this needs no way to tell an ABSENT stamp from a False one.

    On (2) the proof is written back (``remember_owner_key_minted``) while it
    still holds: after an owner-key roll the recorded hash matches neither the new
    owner key nor any keystore entry, so without the back-fill a pre-existing
    owner session would start failing the re-check below and be signed out on the
    roll - the exact behaviour the exemption exists to prevent.

    NOT derived from the scope set, and NOT from a keystore read:

    - **ADMIN is not the question.** The owner may mint ADMIN-scoped KEYSTORE
      keys, which stay revocable; treating "holds ADMIN" as "is the owner" is what
      gave such a key an exemption it was never entitled to.
    - **Nothing here reads the keystore**, so ``_load_keystore()``'s fail-OPEN
      behaviour (``[]`` on OSError/ValueError) cannot promote anything, and no
      answer is derived from a NEGATIVE such as ``not key_hash_live``. Both of
      those shapes produced privilege escalations in the jobs plugin's equivalent
      check (see ``builtin/jobs/plug.py``)."""
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
        # A KEYSTORE key's session lives only as long as its key: re-validate the
        # owning key against the live keystore every request, so revoking or
        # expiring it cuts the session off (parity with the bearer path's
        # per-request verify()).
        from localm.auth import key_hash_live
        if not key_hash_live(rec.get("key_hash")):
            return None
    return rec


def _session_exempt_from_key_recheck(rec, token=None) -> bool:
    """Whether *rec* may skip the per-request keystore liveness re-check.

    The exemption is for the OWNER KEY, not for ADMIN. It exists because the owner
    key is not a keystore entry and a session is decoupled from the key VALUE, so
    an owner-key ROLL must not log the owner out. Keying it on the SCOPE SET alone
    handed the same exemption to any ADMIN-scoped KEYSTORE key - which is revocable
    by design - so revoking such a key did not reliably end its cookie, and if the
    store cleanup also failed the cookie kept working indefinitely. Removing the
    exemption outright is not the fix either: that reintroduces the owner signing
    themselves out, which is the whole reason it is here.

    ADMIN is NECESSARY but not SUFFICIENT, and that conjunction is deliberate. It
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
    # has no such credential, so the question does not apply to it and answering it
    # with key_hash_live(None) -> False would reject the session for the wrong
    # reason. Distinct from the defect above, where the session DOES name a live
    # keystore entry and is exactly the thing that must stay revocable.
    return not rec.get("key_hash")


def _principal_from_token(token, source):
    """Resolve a presented credential to ``(scopes, key_hash, fs_access,
    rag_roots)`` or None.

    A ``header`` token is a raw API key -> ``auth.verify()``. A ``cookie`` token is
    now an OPAQUE SESSION ID -> the server-side session store (``localm.sessions``),
    which returns the scope / owning-key / fs-access / rag-roots SNAPSHOT taken at
    login. So a cookie session stays valid across an owner-key roll (the reported
    bug), and the durable key never has to live in the cookie. ``key_hash`` is the
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
    ``auth.get_api_key()`` - because after a roll the two are indistinguishable.

    Narrow, and every clause of that is load-bearing:

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
    # re-reading the raw stamp, so "is this the owner's session" has exactly one
    # answer. Without that, a pre-upgrade owner session (no stamp, recognised by
    # value) would be exempt from the keystore re-check yet not count as the owner
    # here - two notions of the same thing, disagreeing on the same record.
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
    session cookie. Used to bind a background job to the key that created it
    so only that key (or an admin/owner) may stream or cancel it."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    if not token or not token.strip():
        return None
    # Route through _principal_from_token for BOTH sources: a cookie session
    # for a non-ADMIN key must be re-validated against the
    # live keystore the same way a bearer token is on every request, or a
    # revoked/expired key's still-resident session can keep resolving a key
    # hash here even though the same cookie is already rejected everywhere
    # else auth is enforced.
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

    This is NOT principal_id: principal_id must keep returning the key hash so a
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
    Depends()-injectable gate, the same pattern require_scope already uses for
    scope checks - so a new per-owner route cannot omit the check by
    construction.

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
    dial kept INDEPENDENT of ownership, so an owner can pair one of their own
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


def preview_allowed(request: Request) -> bool:
    """Whether THIS caller is offered the sandboxed artifact preview canvas.

    False when ``gui_preview_enabled`` is off. Otherwise True unless
    ``gui_preview_owner_only`` is set and the caller is not the owner. Open mode
    (no key configured, loopback owner) and an ADMIN key are the owner; every
    other valid key, and an absent or invalid one, are not.

    This decides what the GUI is OFFERED, not what a reply CONTAINS: the block
    is already in the caller's own DOM either way.
    """
    from localm.auth import any_key_configured
    from localm.config import load_config
    cfg = load_config()
    if not cfg.get("gui_preview_enabled", True):
        return False
    if not cfg.get("gui_preview_owner_only", False):
        return True
    if not any_key_configured():
        return True                         # open/dev mode = loopback owner
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        return False
    held, _key_hash, _fs, _rag_roots = prin
    return scopes.ADMIN in held


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


# Surface mounting: on-demand GUI on a running instance.

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
    ``_inference_sem`` exactly as the GUI launcher does.

    If ``attach_gui`` does not return, ``app.router.routes`` and ``app.state`` are
    restored to what they were before the call, and the exception propagates."""
    global _engine, _coder_session_manager, _gui_mounted_live
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
        # A GUI/registry switch must not drop vision. Carry the model's
        # mmproj (registry-recorded, else a sibling projector next to the GGUF)
        # into the new Engine like the CLI --mmproj flag, else switching silently
        # loses image support.
        mmproj = get_model_mmproj(name)
        return Engine(
            str(m_path),
            display_name=name if name in load_registry() else m_hint,
            mmproj_path=mmproj,
        )

    async def switch_model(name: str, *, force: bool = False) -> dict:
        # Preemptive switch: a newer selection aborts an in-flight load rather
        # than waiting for the abandoned model to finish (see switch_engine).
        return await switch_engine(name, _build_engine, force=force)

    from localm.plugins.gui.web import attach_gui
    # Marks the GUI mounted before attaching; unless attach_gui returns, the routes
    # and app.state are put back as they were. See
    # test_a_mount_that_fails_part_way_leaves_the_app_as_it_was.
    routes_before = len(app.router.routes)
    state_before = {key: app.state[key] for key in app.state}
    app.state.gui_mounted = True
    attached = False
    try:
        manager = attach_gui(
            app, self_url=self_url, switch_model=switch_model, active_model=active_model)
        attached = True
    finally:
        if not attached:
            del app.router.routes[routes_before:]
            for key in list(app.state):
                if key not in state_before:
                    del app.state[key]
            for key, value in state_before.items():
                app.state[key] = value
    # attach_gui re-affirms app.state.gui_mounted; reflect the surface change in
    # discovery so /whoami and the registry report this is now a full instance.
    app.state.coder_sessions = manager
    _coder_session_manager = manager
    _gui_mounted_live = True
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
    swallow stays discoverable yet can never itself break the stop/restart (the
    logging call is guarded too)."""
    try:
        from localm.debuglog import logger as _dbg
        getattr(_dbg, level, _dbg.debug)(msg, exc_info=True)
    except Exception:
        pass


def _shutdown_teardown(*, instance_id: Optional[str] = None) -> None:
    """The stop sequence, WITHOUT the process exit.

    Stops in-flight job children (both the GUI's and the coder plugin's own
    background shell/agent jobs), closes every live GUI coder session
    (waiting a few seconds at most), stops any localm-launched ComfyUI instance,
    unloads the model so the native context is freed cleanly (a hard exit
    while it is loaded segfaults during teardown), releases the shared
    embedder, and clears the crash marker so this intentional stop is not
    reported as a crash on the next boot.

    Safe to run twice: every step tolerates already-stopped state.

    *instance_id* (app.state.instance_id, set by instances.advertise()) scopes
    the crash-marker clear to THIS instance only - see
    localm/bugreport/crash_guard.py's per-instance-scoping note; omitting it
    falls back to the legacy shared marker name rather than silently skipping
    the clear."""
    # Stop the child processes of any in-flight background job FIRST. A start_cli
    # job runs `python -m localm <cmd>` as a real child (a model pull, a runtime
    # provision, a ComfyUI setup): os._exit below bypasses atexit, the job worker
    # thread is a daemon so its finally may never run, and the Popen carries no
    # creationflags - so without this the child is simply ABANDONED: a child
    # that writes nothing to stdout survives and keeps working untracked, while
    # one that flushes output dies at its next write on the broken pipe,
    # mid-operation and with no cleanup. See
    # jobs.terminate_children_for_exit for both arms.
    #
    # FIRST in the sequence, before the engine and embedder teardown below,
    # because a media child can itself hold VRAM and any child can keep writing
    # to the data dir - both of which the teardown below is trying to finish.
    #
    # The registry is left saying "running" rather than cancelled: the next
    # start reconciles those rows to "interrupted", which is the honest word for
    # a server that stopped while work was in flight.
    try:
        from localm.plugins.gui.jobs import terminate_children_for_exit
        _killed = terminate_children_for_exit()
        if _killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d in-flight job child process(es) on shutdown", _killed)
    except Exception:
        _dbg_swallow("terminating job child processes during shutdown failed "
                     "(non-fatal); a child may be left running")
    # The coder plugin's own background shell/agent jobs are a SEPARATE
    # registry (localm.plugins.coder.background.JobRegistry) that reaps
    # itself through an atexit hook, same as the one above - and os._exit
    # bypasses atexit exactly the same way. Left uncalled, a coder background
    # shell command or sub-agent outlives the server that started it.
    try:
        from localm.plugins.coder.background import terminate_all_for_exit
        _coder_killed = terminate_all_for_exit()
        if _coder_killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d coder background job(s) on shutdown", _coder_killed)
    except Exception:
        _dbg_swallow("terminating coder background jobs during shutdown failed "
                     "(non-fatal); a job may be left running")
    # Live GUI coder sessions are closed the way a graceful stop closes
    # them. See test_a_failure_closing_sessions_does_not_block_the_stop.
    try:
        from localm.plugins.coder.sessions import close_all_for_exit
        _sessions_closed = close_all_for_exit()
        if _sessions_closed:
            from localm.debuglog import logger as _dbg
            _dbg.info("closed %d coder session(s) on shutdown", _sessions_closed)
    except Exception:
        _dbg_swallow("closing coder sessions during shutdown failed "
                     "(non-fatal); a session may not record its end",
                     level="warning")
    # Any ComfyUI instance localm itself launched (image/music/video, each
    # possibly its own api_url) runs in a detached process group so
    # stop_comfy() can kill its whole tree on demand - which also means it
    # does NOT die on its own when this process exits. Left unstopped it
    # keeps running, orphaned, holding whatever VRAM/RAM its last job loaded.
    try:
        from localm.media.comfy_client import stop_all_spawned_comfy
        _comfy_stopped = stop_all_spawned_comfy()
        if _comfy_stopped:
            from localm.debuglog import logger as _dbg
            _dbg.info("stopped %d localm-launched ComfyUI instance(s) on shutdown",
                      _comfy_stopped)
    except Exception:
        _dbg_swallow("stopping localm-launched ComfyUI instance(s) during shutdown "
                     "failed (non-fatal); one may be left running")
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
    # but a failure stays discoverable.
    try:
        from localm.inference import embedder as _embedder_mod
        # One lock-free call, never active_requests()/reset_embedder(): both
        # take the embedder's load lock, which get_embedder() holds for the FULL
        # duration of an embedding-model load, so a stop issued during a load
        # would block on the guard itself and hang this shutdown before reaching
        # the worker teardown. release_for_exit() makes the whole decision
        # without that lock: it terminates a busy worker outright (the pinned
        # request cannot be served either way once this process exits) and closes
        # an idle one politely. It must run: os._exit below bypasses atexit, and
        # multiprocessing's daemon-child reclamation IS an atexit hook, so a
        # skipped release leaves the worker orphaned with its model resident in
        # VRAM.
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


def _announce_stopping() -> None:
    """Print the line that tells a console user the stop has begun.

    Best-effort: a console that cannot be written to never blocks the stop."""
    try:
        from localm.console import console
        console.print("[dim]Stopping localm...[/dim]")
    except Exception:
        pass


def _do_shutdown(*, instance_id: Optional[str] = None) -> None:
    """The full stop: _shutdown_teardown, then exit the process. Used by the
    GUI/tray Stop button and the shutdown route. Separated from the route so it
    can be tested without exiting."""
    _shutdown_teardown(instance_id=instance_id)
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


# The GUI surface this process shows the user, recorded by set_restart_ui:
# "window", "browser", or None.
_restart_ui: Optional[str] = None

# True once mount_gui_surface has mounted the GUI onto this process's running
# API-only app.
_gui_mounted_live: bool = False

_GUI_MOUNTED_ENV = "LOCALM_RESTART_GUI_MOUNTED"


def set_restart_ui(ui: Optional[str]) -> None:
    """Record the GUI surface this process shows the user: "window" for the
    native app window, "browser" for a browser tab, None for neither.
    _set_restart_env hands it to a restart's re-exec'd process."""
    global _restart_ui
    _restart_ui = ui


def _set_restart_env() -> None:
    """Set the environment a restart's re-exec'd process inherits:
    LOCALM_RESTART_IN_PROGRESS to the surface recorded by set_restart_ui, or to
    "1" when none is recorded, and LOCALM_RESTART_GUI_MOUNTED to "1" when the
    GUI was mounted live onto this process's API-only app (removed otherwise)."""
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = _restart_ui or "1"
    if _gui_mounted_live:
        os.environ[_GUI_MOUNTED_ENV] = "1"
    else:
        os.environ.pop(_GUI_MOUNTED_ENV, None)


def _remount_gui(app) -> None:
    """Mount the GUI onto *app*, the API-only app of a process that a restart
    re-exec'd from an instance whose GUI was mounted live. A failed mount is
    logged as a warning and the API keeps serving."""
    try:
        mount_gui_surface(app)
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.warning("could not mount the GUI after the restart; serving the "
                     "API only: %s", e)


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


def _execv_argv(argv: list) -> list:
    """The argv list to hand os.execv. On Windows each element is wrapped in
    the quoting the child's command-line parser reverses; on every other
    platform the list is returned unchanged."""
    import subprocess
    if os.name != "nt":
        return list(argv)
    return [subprocess.list2cmdline([a]) for a in argv]


def _mark_fds_noninheritable() -> None:
    """Mark every open fd >= 3 non-inheritable. Call this immediately before
    ANY os.execv() in this module - os.execv does NOT close fds on its own,
    so an inherited listening socket or log FileHandler survives into the
    re-exec'd image otherwise (POSIX O_CLOEXEC / Windows handle inheritance
    is what actually drops them at exec). Best-effort per fd; stdin/stdout/
    stderr (0-2) are left inheritable so the new process keeps the console.
    Shared by _do_restart's graceful path and _hang_restart_action's forced
    fallback - see test_hang_restart_forced_fallback_marks_fds_non_inheritable."""
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
    the crash-marker clear to THIS instance only - see _do_shutdown and
    localm/bugreport/crash_guard.py's per-instance-scoping note. The re-exec'd
    process re-advertises and gets a fresh instance_id of its own, so no
    persistence across the restart is needed.

    *port* is the port this instance is actually bound to (app.state.instance_port,
    set by advertise()); it is pinned into the re-exec command line so "comes back
    on the same port" is TRUE rather than merely intended - see _restart_argv.

    *update_watchdog*, when given (only by the post-update restart path - see
    routes/admin.py's /api/update/apply), is a
    ``{host, port, scheme, expect_version}`` dict describing this instance. A
    DETACHED health-check watchdog is spawned right before the re-exec; it
    polls the restarted instance's own /whoami and auto-rolls back if the
    expected version never comes up healthy within its timeout. The plain
    "restart the server" button (/v1/server/restart) calls this with no
    update_watchdog, so it is entirely unaffected.

    Free VRAM is captured BEFORE the teardown below so the wait after it (once
    every native free has been issued) can confirm the releases actually landed
    before re-exec spawns a fresh worker into them. Best-effort: an
    unmeasurable or wedged probe must not block a restart the user asked for,
    and vram_capacity() is itself deadline-bounded."""
    # Stop the child processes of any in-flight background job FIRST. A start_cli
    # job runs `python -m localm <cmd>` as a real child (a model pull, a runtime
    # provision, a ComfyUI setup): os.execv below bypasses atexit, the job worker
    # thread is a daemon so its finally may never run, and the Popen carries no
    # creationflags - so without this the child is simply ABANDONED: a child
    # that writes nothing to stdout survives and keeps working untracked, while
    # one that flushes output dies at its next write on the broken pipe,
    # mid-operation and with no cleanup. See
    # jobs.terminate_children_for_exit for both arms.
    #
    # FIRST in the sequence, before the engine and embedder teardown below,
    # because a media child can itself hold VRAM and any child can keep writing
    # to the data dir - both of which the teardown below is trying to finish.
    #
    # The registry is left saying "running" rather than cancelled: the next
    # start reconciles those rows to "interrupted", which is the honest word for
    # a server that stopped while work was in flight.
    try:
        from localm.plugins.gui.jobs import terminate_children_for_exit
        _killed = terminate_children_for_exit()
        if _killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d in-flight job child process(es) on restart", _killed)
    except Exception:
        _dbg_swallow("terminating job child processes during restart failed "
                     "(non-fatal); a child may be left running")
    # Same reasoning as the job-children step above, for the coder plugin's
    # separate background shell/agent registry: os.execv bypasses atexit too,
    # so without this a coder background job survives the re-exec'd server.
    try:
        from localm.plugins.coder.background import terminate_all_for_exit
        _coder_killed = terminate_all_for_exit()
        if _coder_killed:
            from localm.debuglog import logger as _dbg
            _dbg.info("terminated %d coder background job(s) on restart", _coder_killed)
    except Exception:
        _dbg_swallow("terminating coder background jobs during restart failed "
                     "(non-fatal); a job may be left running")
    # Live GUI coder sessions are closed the way a graceful stop closes
    # them. See test_a_failure_closing_sessions_does_not_block_the_stop.
    try:
        from localm.plugins.coder.sessions import close_all_for_exit
        _sessions_closed = close_all_for_exit()
        if _sessions_closed:
            from localm.debuglog import logger as _dbg
            _dbg.info("closed %d coder session(s) on restart", _sessions_closed)
    except Exception:
        _dbg_swallow("closing coder sessions during restart failed "
                     "(non-fatal); a session may not record its end",
                     level="warning")
    # Any ComfyUI instance localm itself launched runs in a detached process
    # group so stop_comfy() can kill its whole tree on demand - which also
    # means it does NOT die on its own when this process re-execs. Left
    # unstopped it keeps running, orphaned, holding whatever VRAM/RAM its
    # last job loaded, alongside the freshly re-exec'd server.
    try:
        from localm.media.comfy_client import stop_all_spawned_comfy
        _comfy_stopped = stop_all_spawned_comfy()
        if _comfy_stopped:
            from localm.debuglog import logger as _dbg
            _dbg.info("stopped %d localm-launched ComfyUI instance(s) on restart",
                      _comfy_stopped)
    except Exception:
        _dbg_swallow("stopping localm-launched ComfyUI instance(s) during restart "
                     "failed (non-fatal); one may be left running")

    # had_engines asks "is anything ACTUALLY loaded" (worth waiting on), not
    # merely "is the dict non-empty": unload_all_models/idle-unload both KEEP a
    # now-unloaded engine's entry in _engines so a later request reloads it
    # lazily (see their own docstrings), so a dict-non-emptiness check would
    # make EVERY restart on a server that ever idle-unloaded a model pay the
    # wait's full timeout for nothing - the exact "no delay in the common
    # case" claim below would be false. getattr defaults to False so a test
    # double that does not define .loaded (never holding real VRAM) is
    # correctly treated as nothing-to-wait-for. Computed before any
    # unload/release below, and used to gate the free-VRAM read that follows.
    had_engines = any(getattr(e, "loaded", False) for e in _engines.values())
    if not had_engines and _engine is not None and _engine not in _engines.values():
        had_engines = bool(getattr(_engine, "loaded", False))

    # Cheap: a lock check, not a probe. Must run before release_for_exit() below.
    embedder_had_something = False
    try:
        from localm.inference import embedder as _embedder_mod
        embedder_had_something = _embedder_mod.loaded_path() is not None
    except Exception:
        _dbg_swallow("embedder loaded-state check during restart failed (non-fatal)")

    # A subprocess-isolated GPU probe when torch is not resident. See
    # test_do_restart_skips_vram_wait_when_nothing_was_loaded.
    free_before = None
    if had_engines or embedder_had_something:
        try:
            from localm.discover import vram_capacity
            free_before = vram_capacity().get("free")
        except Exception:
            _dbg_swallow("free-VRAM read before restart failed (non-fatal)")

    # Unload all engines in the multi-model dictionary
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
    # regardless, but a failure stays discoverable.
    released_embedder = False
    try:
        from localm.inference import embedder as _embedder_mod
        # Lock-free release, as in _do_shutdown above. os.execv is the same case
        # as os._exit: it replaces this process image but does NOT touch the
        # separate worker child, and bypasses atexit, so without this the old
        # worker survives the restart holding VRAM while the restarted server
        # spawns a second one.
        released_embedder = _embedder_mod.release_for_exit()
    except Exception:
        _dbg_swallow("embedder release during restart failed (non-fatal)")

    # Wait for the frees above to actually land before re-exec. The re-exec'd
    # process spawns a brand-new GGUF worker that constructs a fresh
    # llama_context on startup (plugins/gui/cli.py's preload thread) with no
    # idea a restart just freed anything. If the GPU driver has not finished
    # reclaiming the just-freed VRAM by the time the fresh worker allocates,
    # that construction races the still-reclaiming driver, so this waits the
    # same way switch_engine's in-process model swap already waits before
    # constructing its replacement. Skipped when nothing was actually unloaded
    # (a model-less restart), so the common case pays no delay.
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

    _mark_fds_noninheritable()

    if update_watchdog:
        # Spawned as the LAST step before execv: the watchdog's own timeout
        # clock starts when its process begins executing, so spawning as late as
        # possible keeps that clock closest to the actual restart moment.
        try:
            from localm import updater
            updater.spawn_health_watchdog(
                host=update_watchdog["host"], port=update_watchdog["port"],
                scheme=update_watchdog.get("scheme", "http"),
                expect_version=update_watchdog["expect_version"])
        except Exception as e:
            # spawn_health_watchdog() already never raises; this is
            # belt-and-suspenders so even a malformed dict here can never block
            # the restart itself. Logged rather than silenced.
            try:
                from localm.debuglog import logger as _dbg
                _dbg.warning("update watchdog not started: %s", e)
            except Exception:
                pass

    _set_restart_env()
    os.execv(sys.executable, _execv_argv(_restart_argv(port)))


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


def _init_engine_state(engine: Optional[Engine]) -> None:
    """Reset the engine registry for a fresh app, then publish *engine*, when
    given, as the default and active model with its own inference semaphore.
    First step of create_app(): every name here is a module global the route
    groups and the model-lifecycle functions read live."""
    global _engine, _inference_sem, _engines, _engines_lru, _default_model_name, _active_model_name, _last_active_model_name, _inference_sems, _last_activity_per_model, _embedder_sem

    _engines.clear()
    _engines_lru.clear()
    _inference_sems.clear()
    _embedder_sem = None
    _last_activity_per_model.clear()
    _routing_latch.clear()
    # A fresh app boot must never carry over a name remembered from a
    # previous create_app() call in the same process (test reuse, a restart) -
    # see _last_active_model_name's own docstring for why it exists at all.
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


def _init_session_audit():
    """Open the server's audit log and transcript for the "server" session mode,
    publish the audit log as the module global _audit, and return all three as
    the AppContext create_app() hands to the route groups."""
    global _audit

    # Session-persistence mode for this server (privacy -> no traces). One audit
    # log / transcript covers the server lifetime; GUI + API chat traffic flows
    # through /v1/chat/completions and lands here. _audit is published as a
    # module global (unlike _mode/_transcript, which reach the route groups
    # only on the AppContext returned below) because _do_restart is a
    # separate top-level function with no other way to reach it - without
    # `global _audit` here, its cleanup could never reach the real object.
    # The lifespan's shutdown closes it through the same global.
    from localm.audit import effective_mode, make_audit_log, make_transcript
    _mode = effective_mode("server")
    _audit = make_audit_log(_mode, label="server")
    _transcript = make_transcript(_mode, label="server")
    from localm.inference.app_assembly.context import AppContext
    return AppContext(audit=_audit, transcript=_transcript, mode=_mode)


def _make_lifespan():
    """The app's lifespan. Startup publishes the running loop (_server_loop),
    sweeps expired browser sessions, installs the asyncio exception handler,
    runs the plugins' startup callbacks and starts the background services: the
    idle-unload loop, the heartbeat, the stack-dump watchdog, the executor
    saturation watch, the hang alarm, the cross-install GPU registry and the
    mmproj backfill. Most of those stay off under pytest; each guard says which.
    Shutdown stops them, leaves the GPU registry, clears _server_loop and closes
    the module global _audit.

    It lives here, not in app_assembly, because every process-lifetime global
    it writes belongs to this module (ADR-0023)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem, _inference_sems, _active_model_name, _server_loop
        global _hang_dump_loop, _hang_alarm_instance
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
        # Route an uncaught asyncio task exception through the bug reporter
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
        # Plugins register() before the loop exists, so loop-dependent plugin
        # work (the jobs scheduler) is queued on the manager and run here, now
        # the loop is up. attach_engine runs after create_app, so the manager
        # resolves at lifespan time.
        _pm = getattr(app.state, "plugin_manager", None)
        if _pm is not None:
            _pm.run_startup_callbacks()
        # Optional idle-unload background task (config "idle_unload_seconds"); it
        # is a cheap no-op while disabled. Cancelled on shutdown so it never
        # outlives the app.
        idle_task = asyncio.create_task(_idle_unload_loop())

        # Heartbeat (_hb_monotonic) vs. the watchdog's disk-writing stall dump:
        # DECOUPLED, on two different gates. The heartbeat is pure in-memory
        # bookkeeping (_hang_heartbeat_loop writes nothing to disk, logs nothing,
        # sends nothing - see its docstring), so it carries none of the privacy
        # considerations that gate the stack-dump file below and runs whenever
        # "pytest" not in sys.modules, independent of mode. THREE independent
        # readers depend on it: the debug request log (debug_enabled(), below),
        # GET /debug/stacks (its own unconditional reachability gate), and the
        # watchdog thread's own stall check. Starting it inside the watchdog's
        # privacy/env gate instead would leave _hb_monotonic None for the whole
        # process lifetime whenever that gate is off (LOCALM_HANG_WATCHDOG=0, or
        # privacy mode on default config), making _loop_lag_seconds() report a
        # permanent 0.00 indistinguishable from "healthy". Best-effort: a startup
        # failure must never block serving.
        hb_task = None
        if "pytest" not in sys.modules:
            try:
                hb_task = asyncio.create_task(_hang_heartbeat_loop())
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("heartbeat task startup failed (continuing): %s", e)

        # The stack-dump-on-stall THREAD keeps its own, unchanged privacy gate:
        # on by default in the log/full session modes (the trace file is lazy,
        # created only on a real stall); in PRIVACY mode it stays OFF (no
        # automatic trace written to disk) UNLESS the user opted into keeping
        # diagnostics (_diagnostics_allowed) or explicitly forced it on
        # (LOCALM_HANG_WATCHDOG=1). LOCALM_HANG_WATCHDOG=0 opts out entirely.
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

        # Executor thread-pool saturation watch: a separate off-loop daemon
        # thread, always on (unlike the hang watchdog it sits beside, it logs
        # only pool names and integer counts - no paths, no chat content - so it
        # carries none of the privacy-mode considerations that gate the
        # stack-trace capture above). Skipped under pytest so no thread lingers,
        # matching the hang watchdog's own guard. Best-effort: a startup failure
        # must never block serving.
        sat_stop = sat_thread = None
        if "pytest" not in sys.modules:
            try:
                from localm.inference._executor_health import start_executor_saturation_watch
                # anyio's default thread pool (what
                # fastapi.concurrency.run_in_threadpool always uses) can only be
                # resolved from INSIDE a running event loop, so capture it ONCE
                # here and hand the reference to the plain background thread
                # below, which cannot fetch it itself. A capture failure degrades
                # to "anyio pool unobservable", logged rather than silently
                # claimed healthy.
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

        # Stats prewarm: the CPU baseline and the first VRAM / GPU-load probes
        # start now, so the first status-bar poll finds readings. Skipped under
        # pytest like its siblings.
        if "pytest" not in sys.modules:
            try:
                from localm import sysstats as _sysstats
                threading.Thread(target=_sysstats.prewarm, name="localm-stats-prewarm",
                                 daemon=True).start()
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("stats prewarm startup failed (continuing): %s", e)

        # Hang ALARM: detect a hung server, surface it where a
        # user actually looks, and (by default) auto-restart when the hang is
        # provably a defect. Complements - does not replace - the forensic
        # stack-dump watchdog above: that one is privacy-gated because it
        # writes stacks to disk; this one writes nothing sensitive anywhere,
        # so it runs in every mode (LOCALM_HANG_RECOVERY=off opts out).
        # Skipped under pytest like its siblings; tests drive HangAlarm
        # directly.
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
                    _hang_alarm_instance = hang_alarm
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("hang alarm startup failed (continuing): %s", e)

        # Cross-install GPU/VRAM coordination (see localm.gpu_registry): a real,
        # non-isolated, advertise()'d server (instance_id + port/scheme are set by
        # advertise() before uvicorn accepts connections) publishes its live status
        # to sibling instances that ask for it. A bare create_app() test app or an
        # --isolated run never sets instance_id, so it stays invisible. Nothing is
        # written to disk. Best-effort: a failure must never block startup (RULE 5:
        # logged, not silenced).
        global _gpu_coord
        _instance_id = getattr(app.state, "instance_id", None)
        _isolated = getattr(app.state, "instance_isolated", False)
        if _instance_id and not _isolated:
            try:
                _gpu_coord = {
                    "instance_id": _instance_id,
                    "port": getattr(app.state, "instance_port", None),
                    "host": getattr(app.state, "bind_host", None) or "127.0.0.1",
                    "scheme": getattr(app.state, "instance_scheme", None) or "http",
                }
                from localm import gpu_registry
                gpu_registry.set_local_status_provider(_gpu_status)
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("gpu coordination startup failed (continuing without "
                          "cross-instance GPU coordination): %s", e)
                _gpu_coord = None

        # Vision-projector backfill (model_manager.sync_models_dir's
        # backfill_mmproj part): one-shot, off the loop, started here rather
        # than during the synchronous CLI startup sequence so a per-candidate
        # HF lookup never delays accepting connections. Best-effort, logged
        # only; skipped under pytest like its siblings above.
        mmproj_task = None
        if "pytest" not in sys.modules:
            try:
                mmproj_task = asyncio.create_task(_mmproj_backfill_once())
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("mmproj backfill task startup failed (continuing): %s", e)

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
                # Retire this lifespan's alarm as the process's restart authority.
                # See test_lifespan_shutdown_clears_the_hang_alarm_instance.
                if _hang_alarm_instance is hang_alarm:
                    _hang_alarm_instance = None
                    _hang_dump_loop = None
            if mmproj_task is not None:
                mmproj_task.cancel()
                try:
                    await mmproj_task
                except asyncio.CancelledError:
                    pass
            if _gpu_coord is not None:
                try:
                    from localm import gpu_registry
                    gpu_registry.set_local_status_provider(None)
                except Exception as e:
                    from localm.debuglog import logger as _dbg
                    _dbg.debug("gpu coordination cleanup on shutdown failed: %s", e)
                _gpu_coord = None
            # The loop is stopping - stop advertising it so a late off-loop caller
            # falls back to the safe "no loop" path instead of a dead loop reference
            # (_server_loop is already declared global at the top of lifespan).
            _server_loop = None
            _audit.close()

    return lifespan


def create_app(engine: Optional[Engine], *, api_landing: bool = False) -> FastAPI:
    """Assemble the app. The body is the boot order; each step is one
    app-assembly component (localm/inference/app_assembly/, ADR-0023).

    Middleware is added innermost first: every add wraps what was added before
    it, so a request passes the steps of phase 4 bottom-up.
    tests/test_create_app_characterization.py and tests/test_app_assembly.py
    pin the resulting stack."""
    from localm.inference.app_assembly import (
        context, diagnostics, errors, mounting, security, transport)

    # 1. Process state: the engine registry and the server's session audit.
    _init_engine_state(engine)
    ctx = _init_session_audit()

    # 2. The app, with the lifespan that runs the background services.
    app = FastAPI(
        title="localm inference server",
        version="0.2.0",
        lifespan=_make_lifespan(),
        **diagnostics.fastapi_telemetry_off(),
    )

    # 3. Exception handlers, then the app.state the middleware and routes read.
    errors.register_exception_handlers(app)
    context.init_app_state(app)

    # 4. Kernel routes and middleware, innermost middleware first.
    if api_landing:
        mounting.add_api_landing(app)
    diagnostics.add_request_logging(app)     # debug mode only
    diagnostics.add_debug_stacks(app)
    cors_cfg = security.add_cors(app)
    security.add_origin_guard(app, cors_cfg)
    security.add_security_headers(app)
    security.add_docs_loopback_gate(app)
    transport.add_transport_middleware(app)  # outermost

    # 5. Route groups (localm/inference/routes/*.py).
    mounting.mount_route_groups(app, ctx)

    # 6. Plugins, last: a failure here still leaves the kernel serving.
    mounting.attach_plugins(app, engine)

    return app


def _engine_finish_reason(engine) -> str:
    """Why the last generation ended - "stop" unless the backend reported a
    real string (mocks and minimal engines without the attribute count as stop)."""
    fr = getattr(engine, "last_finish_reason", "stop")
    return fr if isinstance(fr, str) else "stop"


def _mtp_usage(engine) -> Optional[MtpUsage]:
    """The last reply's MTP figures, or None when the engine reports none
    (MTP off, a non-GGUF backend, or a minimal engine without the method)."""
    fn = getattr(engine, "mtp_usage", None)
    data = fn() if callable(fn) else None
    if not isinstance(data, dict):
        return None
    try:
        return MtpUsage(**data)
    except (TypeError, ValueError) as exc:
        from localm.debuglog import logger as _dbg
        _dbg.debug("usage.mtp left out: the engine's MTP figures did not validate (%s)",
                   type(exc).__name__)
        return None


def _speculation_usage(engine) -> Optional[SpeculationUsage]:
    """The last reply's speculative-drafting figures for any draft source, or
    None when the engine reports none (no draft source, a non-GGUF backend, or
    a minimal engine without the method)."""
    fn = getattr(engine, "speculation_usage", None)
    data = fn() if callable(fn) else None
    if not isinstance(data, dict):
        return None
    try:
        return SpeculationUsage(**data)
    except (TypeError, ValueError) as exc:
        from localm.debuglog import logger as _dbg
        _dbg.debug("usage.speculation left out: the engine's figures did not "
                   "validate (%s)", type(exc).__name__)
        return None


def _ttft_ms(gen_start: float, first_token_at: Optional[float]) -> Optional[float]:
    """Time to first token in milliseconds, or None if nothing was generated."""
    if first_token_at is None:
        return None
    return round((first_token_at - gen_start) * 1000, 1)


def _decode_elapsed(first_token_at: Optional[float], gen_end: float) -> Optional[float]:
    """Wall time spent DECODING: first token -> end of generation. None if nothing
    was generated. EXCLUDES model load + prompt prefill (the span before the first
    token), which is reported on its own as ttft_ms, so a cold start's multi-second
    load is never charged against the generation rate."""
    if first_token_at is None:
        return None
    return gen_end - first_token_at


# A floor on plausible per-token decode time. An implausible decode window is
# rejected rather than reported as a nonsensical rate (see _tokens_per_sec).
# first_token_at is a SINGLE sample: if the GPU scheduler delays token 1 under
# contention then delivers the rest in an uncontended burst, the measured decode
# window collapses toward zero even though every individual timestamp is real.
# 1ms/token (a 1000 tok/s ceiling) is generous: single-stream autoregressive
# decode is memory-bandwidth-bound (reading the quantized weights at least once
# per token), and the GGUF backend relays each token through a subprocess queue,
# so a single request is not expected to sustain a rate past this ceiling. Below
# it, no number is reported at all.
_MIN_SEC_PER_TOKEN = 0.001


def _tokens_per_sec(completion_tokens: int, decode_elapsed: Optional[float]) -> Optional[float]:
    """Decode throughput = generated tokens over the DECODE window only (see
    _decode_elapsed), NOT over total wall time. Folding the model-load/prefill
    time into this rate makes the first call after a load report a rate orders of
    magnitude too low. Matches the `localm bench` convention (cli/models.py):
    "tok/s measures pure generation after the first token". None when
    unmeasurable: a non-positive window, fewer than two tokens (one token has no
    decode interval to time), or a window so short it implies a physically
    implausible rate (see _MIN_SEC_PER_TOKEN), which is a burst-arrival artifact
    under GPU contention rather than a real decode speed."""
    if (decode_elapsed is None or decode_elapsed <= 0 or completion_tokens < 2
            or decode_elapsed < completion_tokens * _MIN_SEC_PER_TOKEN):
        return None
    return round(completion_tokens / decode_elapsed, 2)


def _last_user_text(messages: list) -> str:
    """Text of the most recent user message (for the audit trail). A row the
    client marked with an ``origin`` (a GUI web tool event sent as user-role
    text, or a prompt the client wrote itself such as the compaction summarise
    request) is skipped: the audit line, and the session log memory
    consolidation learns from, record only what the user wrote."""
    for m in reversed(messages):
        if m.get("role") == "user" and not m.get("origin"):
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


def _debug_prompt_dump(messages: list) -> str:
    """One line per message: index, role, and its text; non-text content parts
    (e.g. an image) are named, not embedded."""
    lines = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            parts = []
            for p in (content or []):
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, dict):
                    parts.append(f"<{p.get('type', 'non-text')}>")
            text = " ".join(parts)
        lines.append(f"[{i}] {m.get('role', '?')}: {text}")
    return "\n".join(lines)


def _log_assembled_prompt(messages: list) -> None:
    """Debug-log the messages about to reach the backend, content-gated on
    debug_content_enabled(). See test_assembled_prompt_debug_capture.py."""
    from localm.debuglog import debug_content_enabled, logger as _dbg
    if debug_content_enabled():
        _dbg.debug("assembled chat prompt (%d message(s)):\n%s",
                    len(messages), _debug_prompt_dump(messages))


def _log_chat_reply(reply: str, finish_reason: str = "stop") -> None:
    """Debug-log the model's chat completion reply, content-gated on
    debug_content_enabled(). See test_assembled_prompt_debug_capture.py."""
    from localm.debuglog import debug_content_enabled, logger as _dbg
    if debug_content_enabled():
        _dbg.debug("chat completion reply (finish_reason=%s):\n%s",
                   finish_reason, reply)


def _turn_outcome(gen_error, finish_reason: str) -> str:
    """The ChatHookContext outcome for a finished generation: "error" when the
    backend raised, "length" when the token budget ran out, else "success"."""
    if gen_error is not None:
        return "error"
    return "length" if finish_reason == "length" else "success"


def _audit_exchange(audit, transcript, messages: list, reply: str,
                    outcome: str = "success") -> None:
    """Record one chat exchange (log/full modes; no-op log in privacy). A
    non-"success" *outcome* is recorded as an audit notice and as a trailing
    ``[finish_reason: ...]`` line in the transcript, so a failed or cut-off
    generation never reads as a short normal reply."""
    if audit is None:
        return
    try:
        user_text = _last_user_text(messages)
        audit.user(user_text)
        audit.llm(reply)
        if outcome != "success":
            audit.notice("finish_reason", outcome)
        if transcript is not None:
            recorded = reply
            if outcome != "success":
                recorded = f"{reply}\n\n[finish_reason: {outcome}]"
            transcript.exchange(user_text, recorded)
    except Exception as e:
        # In log/full mode a failed write silently drops the record; surface it
        # so the gap is discoverable instead of invisible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("audit/transcript write failed: %s; this exchange was not recorded", e)
        pass  # auditing must never break serving


def _reason_sse(content: str, reasoning: str,
                model_id: str, chunk_id: str, ts: int) -> list:
    """SSE ``data:`` lines for a (content, reasoning) split. Reasoning is
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
    engine out from under an in-flight request. The count itself lives in
    residency.pin_engine, the one guarded mutation site for active_requests."""
    residency.pin_engine(engine)


def _unpin(engine) -> None:
    """Release the request pin taken by _pin. Balanced exactly once per request."""
    residency.unpin_engine(engine)


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
    that actually drives the model should register as "in use".

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
            try:
                await gen.aclose()
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("closing stream generator on unpin failed")
        finally:
            _unpin(engine)


# How long a streaming chat request may spend preparing (loading the model,
# running inlet hooks, checking the request) before its stream is opened early
# to report what it is doing.
PREP_STATUS_GRACE_S = 0.4

# Interval between SSE keepalive comments while a request is still preparing.
PREP_KEEPALIVE_S = 15.0


class PrepProgress:
    """The live status text of one chat request's preparation phase.

    ``set`` and ``wait_changed`` must be called on the event loop thread."""

    def __init__(self) -> None:
        self.text: Optional[str] = None
        self._changed = asyncio.Event()

    def set(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self._changed.set()

    async def wait_changed(self, task: asyncio.Future, timeout: float) -> None:
        """Return when the text changes, *task* finishes, or *timeout* passes."""
        self._changed.clear()
        waiter = asyncio.ensure_future(self._changed.wait())
        try:
            await asyncio.wait({task, waiter}, timeout=timeout,
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()


def release_prepared_on_done(task: asyncio.Future, engine_of) -> None:
    """Unpin the engine a preparation *task* pinned, once it finishes, for a
    caller that will never stream it. ``engine_of(result)`` returns that
    engine, or None when the result holds no pin. A failed or cancelled task
    holds no pin."""
    def _release(t: asyncio.Future) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            if not isinstance(exc, HTTPException):
                from localm.debuglog import logger as _dbg
                _dbg.error("chat request preparation failed after its client left",
                           exc_info=exc)
            return
        engine = engine_of(t.result())
        if engine is not None:
            _unpin(engine)
    if task.done():
        _release(task)
    else:
        task.add_done_callback(_release)


async def stream_after_prep(
    task: asyncio.Future,
    progress: PrepProgress,
    model_id: str,
    start_stream: Callable[[object, str], AsyncIterator[str]],
    *,
    engine_of: Callable[[object], Optional[Engine]],
    headers_of: Callable[[object], dict],
) -> AsyncIterator[str]:
    """SSE body for a chat request whose preparation outlasted
    ``PREP_STATUS_GRACE_S``: a role chunk, a status chunk for every change of
    *progress*, then the reply.

    *task* resolves to the prepared request, which holds an engine pin.
    ``start_stream(prepared, chunk_id)`` returns the reply's SSE lines and
    takes over that pin. ``headers_of(prepared)`` is sent as a chunk carrying
    ``localm_headers`` (the response headers the early stream could not set)
    when non-empty. A preparation that raises ends the stream with its message
    as an error reply (``finish_reason: "error"``) whose terminal chunk carries
    ``localm_error: {"status": <HTTP status>, "detail": <message>}``. A client
    that disconnects before the reply starts leaves the preparation running;
    its pin is released when it finishes."""
    chunk_id = make_chunk_id()
    ts = int(time.time())
    handed = False
    try:
        role = ChatChunk(id=chunk_id, created=ts, model=model_id,
                         choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))])
        yield f"data: {role.model_dump_json()}\n\n"
        shown: Optional[str] = None
        while True:
            if progress.text is not None and progress.text != shown:
                shown = progress.text
                status = ChatChunk.status_chunk(shown, model_id, chunk_id, ts)
                yield f"data: {status.model_dump_json()}\n\n"
                continue
            if task.done():
                break
            await progress.wait_changed(task, PREP_KEEPALIVE_S)
            if not task.done() and progress.text == shown:
                yield ": keepalive\n\n"
        try:
            prepared = task.result()
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
            for line in _prep_error_lines(detail, e.status_code, model_id, chunk_id, ts):
                yield line
            return
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.exception("chat request preparation failed")
            for line in _prep_error_lines(
                    inference_error_text(e).strip(), 500, model_id, chunk_id, ts):
                yield line
            return
        meta = headers_of(prepared)
        if meta:
            yield "data: " + json.dumps({
                "id": chunk_id, "object": "chat.completion.chunk", "created": ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "localm_headers": meta,
            }) + "\n\n"
        handed = True
        inner = start_stream(prepared, chunk_id)
        try:
            async for line in inner:
                yield line
        finally:
            await inner.aclose()
    finally:
        if not handed:
            release_prepared_on_done(task, engine_of)


def _prep_error_lines(detail: str, status: int, model_id: str, chunk_id: str,
                      ts: int) -> list:
    """The error reply and terminal lines that end an early-opened stream. Both
    the error text chunk and the terminal chunk carry ``localm_error``, so a
    client that knows it can raise before showing the text as a reply."""
    refusal = {"status": status, "detail": detail}
    err = ChatChunk.token(detail, model_id, chunk_id, ts).model_dump()
    err["localm_error"] = refusal
    done = ChatChunk.done(model_id, chunk_id, ts, finish_reason="error").model_dump()
    done["localm_error"] = refusal
    return ["data: " + json.dumps(err) + "\n\n",
            "data: " + json.dumps(done) + "\n\n",
            "data: [DONE]\n\n"]


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
    compact: bool = False,
    chunk_id: Optional[str] = None,
    role_sent: bool = False,
    **gen_kwargs,
) -> AsyncIterator[str]:
    """Stream the reply to *messages* as SSE ``data:`` lines.

    *chunk_id* reuses the id of a stream the caller already opened, and
    *role_sent* skips the role chunk that caller already sent.

    With *compact* (or, when *prompt_tokens* is not given, when the prompt
    nearly fills the context), the conversation is compacted after the role
    chunk, behind a ``COMPACTING_STATUS`` status chunk. A compacted prompt
    that still does not fit, or whose recount is refused, ends the stream with
    the refusal as an error reply and no generation."""
    from localm.inference.compact import compactable
    from localm.inference.gbnf import think_exit_marker
    from localm.textnorm import ThinkSplitter

    chunk_id = chunk_id or make_chunk_id()
    ts = int(time.time())
    # route <think> reasoning into delta.reasoning_content; a tool call the lazy
    # grammar forced inside an open think block is the reply, not reasoning
    think = ThinkSplitter(exit_marker=think_exit_marker(
        gen_kwargs.get("grammar_lazy"), gen_kwargs.get("grammar_triggers")))
    stop = gen_kwargs.pop("stop", None)
    stopper = StopFilter(stop) if stop else None
    stopped = False
    emitted_content: list[str] = []
    emitted_reasoning: list[str] = []

    if prompt_tokens is None:
        prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)
        compact = compact or (
            _needs_compaction(engine.context_capacity(), prompt_tokens, messages,
                              _engine_is_encoder_decoder(engine))
            and compactable(messages))

    if not role_sent:
        role_chunk = ChatChunk(
            id=chunk_id,
            created=ts,
            model=model_id,
            choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

    if compact:
        compacting = ChatChunk.status_chunk(COMPACTING_STATUS, model_id, chunk_id, ts)
        yield f"data: {compacting.model_dump_json()}\n\n"
        new_messages, changed, _gone = await _compact_for_capacity(engine, messages)
        refusal = ""
        if changed:
            messages = list(new_messages)
            try:
                prompt_tokens = await asyncio.get_running_loop().run_in_executor(
                    None, engine.count_messages_tokens, messages)
            except PretokenizerUnsafeInputError as e:
                refusal = str(e)
                prompt_tokens = None
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.exception("token recount after compaction failed")
                refusal = inference_error_text(e).strip()
                prompt_tokens = None
        capacity = engine.context_capacity()
        if (not refusal and isinstance(capacity, int) and capacity > 0
                and isinstance(prompt_tokens, int) and prompt_tokens > capacity):
            refusal = context_overflow_detail(
                prompt_tokens, capacity, _engine_is_encoder_decoder(engine))
        if refusal:
            if ctx is not None:
                ctx.outcome = "error"
            err_chunk = ChatChunk.token(refusal, model_id, chunk_id, ts)
            yield f"data: {err_chunk.model_dump_json()}\n\n"
            done = ChatChunk.done(model_id, chunk_id, ts, finish_reason="error",
                                  usage=UsageInfo(prompt_tokens=prompt_tokens or 0,
                                                  total_tokens=prompt_tokens or 0,
                                                  context_capacity=capacity))
            yield f"data: {done.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

    if sem.locked():
        waiting_chunk = ChatChunk.status_chunk(
            WAITING_FOR_MODEL_STATUS, model_id, chunk_id, ts)
        yield f"data: {waiting_chunk.model_dump_json()}\n\n"

    # Run blocking generator in executor so we don't block the event loop
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    class _StatusSignal:
        def __init__(self, text: str) -> None:
            self.text = text

    # A mid-stream client disconnect makes Starlette throw GeneratorExit into this
    # async generator. Without a cancel path the producer thread below would keep
    # driving engine.chat_stream() all the way to end-of-generation, holding
    # llama.py's per-model _inference_lock the whole time and blocking the next
    # request to THIS model. cancel_event lets the disconnect unwind stop it.
    # Also registered with residency's cancel broadcast so an unload/switch can
    # trigger the same stop proactively, instead of only reading active_requests
    # once and refusing.
    cancel_event = threading.Event()
    residency.register_cancel(engine.display_name, cancel_event)

    def _generate():
        # engine.chat_stream is called INSIDE the try: Engine.chat_stream is not a
        # generator - it eagerly runs the auto-reload (backend.load()) and
        # load_config() before returning the token generator, so it can RAISE here
        # (a reload OOM, a since-removed GGUF). If that raise escaped the try, the
        # thread would die before the finally enqueued the sentinel and the consumer
        # would block forever at `await token_queue.get()` holding the per-model
        # semaphore - a permanent per-model deadlock. Inside the try, the except
        # surfaces it and the finally still enqueues _DONE.
        _log_assembled_prompt(messages)
        gen = None
        def _on_status(s: str) -> None:
            loop.call_soon_threadsafe(token_queue.put_nowait, _StatusSignal(s))

        try:
            gen_opts = dict(gen_kwargs)
            gen_opts.pop("on_status", None)
            gen = engine.chat_stream(messages, on_status=_on_status, **gen_opts)
            for token in gen:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Log (full traceback to the debug log) and surface to the client - a
            # silent thread death looks like an empty reply. NOT
            # traceback.print_exc(): _dbg.exception already records the trace, an
            # expected condition (e.g. outgrew n_ctx_max) should reach the user as
            # a clean message, and printing it can raise WinError 6 on Windows.
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
        # The backend reports image encoding itself, only when an image is encoded.
        status_chunk = ChatChunk.status_chunk(PROCESSING_PROMPT_STATUS, model_id, chunk_id, ts)
        yield f"data: {status_chunk.model_dump_json()}\n\n"

        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        gen_error: Exception | None = None
        drained = False
        try:
            try:
                while True:
                    token = await token_queue.get()
                    if token is _DONE:
                        break
                    if isinstance(token, _StatusSignal):
                        chunk = ChatChunk.status_chunk(token.text, model_id, chunk_id, ts)
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        continue
                    if isinstance(token, Exception):
                        gen_error = token
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    # Stream hook transforms the piece before it is recorded and sent,
                    # so usage reflects exactly what the client receives.
                    if pipeline is not None and ctx is not None and pipeline.has("stream"):
                        token = pipeline.run_stream(token, ctx)
                    content, reasoning = think.feed(token)
                    if stopper is not None:
                        content = stopper.feed(content)
                        stopped = stopper.hit
                        emitted_content.append(content)
                        emitted_reasoning.append(reasoning)
                    completion_parts.append(token)
                    for data in _reason_sse(content, reasoning, model_id, chunk_id, ts):
                        yield data
                    if stopped:
                        break
                drained = True
            finally:
                # Signal the producer to stop. On a clean finish this is a no-op: the
                # thread already exited after _DONE, so t.join() below returns at once.
                # On a disconnect (GeneratorExit raised at the yield above) it makes the
                # thread break its loop, close the generator chain, and release
                # _inference_lock instead of running to end-of-generation. GeneratorExit
                # then keeps propagating, so t.join() below is skipped - the daemon
                # thread self-terminates within ~one token of the cancel.
                cancel_event.set()
                if not drained and ctx is not None:
                    ctx.outcome = "abort"

            t.join()
        finally:
            # t has fully exited by now on every path that reaches here (the
            # normal one; a GeneratorExit skips straight past to this finally
            # without joining, which is fine - the thread notices cancel_event
            # within a token or two on its own and this registration merely
            # stops being reachable, same as an unpin with nothing pinned).
            residency.unregister_cancel(engine.display_name, cancel_event)
        gen_end = time.perf_counter()
        # Release any tail held back while disambiguating a partial <think> tag
        # or a partial stop sequence.
        if not stopped:
            tail_content, tail_reasoning = think.flush()
            if stopper is not None:
                tail_content = stopper.feed(tail_content)
                stopped = stopper.hit
                if not stopped:
                    tail_content += stopper.flush()
            for data in _reason_sse(tail_content, tail_reasoning, model_id, chunk_id, ts):
                yield data

    error_text = ""
    if gen_error is not None:
        error_text = inference_error_text(gen_error)
        err_chunk = ChatChunk.token(error_text, model_id, chunk_id, ts)
        yield f"data: {err_chunk.model_dump_json()}\n\n"

    streamed = "".join(completion_parts)
    if stopped:
        reasoning_text = "".join(emitted_reasoning)
        streamed = (f"<think>{reasoning_text}</think>" if reasoning_text else "") \
            + "".join(emitted_content)
    # finish_reason is fixed before the outlet phase; ctx.outcome, the audit
    # record and the terminal frame all carry the same value. A mid-stream
    # error reports "error", never a clean "stop".
    finish_reason = ("error" if gen_error is not None
                     else "stop" if stopped else _engine_finish_reason(engine))
    outcome = _turn_outcome(gen_error, finish_reason)
    if ctx is not None:
        ctx.outcome = outcome
    # Outlet runs after every chunk has been sent, so it cannot alter the live
    # stream (a stream hook does that). Here it only shapes the recorded reply
    # (audit / transcript / side-effects); usage stays tied to what was streamed.
    # A failed generation skips the outlet; its recorded reply is the streamed
    # text plus the visible error chunk.
    reply = streamed + error_text
    if (gen_error is None and pipeline is not None and ctx is not None
            and pipeline.has("outlet")):
        reply = await pipeline.run_outlet(streamed, messages, ctx)

    _audit_exchange(audit, transcript, messages, reply, outcome=outcome)
    _log_chat_reply(reply, finish_reason=finish_reason)

    # Count tokens on the streamed text - what the client actually received
    completion_tokens = await _count_streamed_tokens(engine, streamed)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(
            completion_tokens, _decode_elapsed(first_token_at, gen_end)),
        context_capacity=engine.context_capacity(),
        mtp=_mtp_usage(engine),
        speculation=_speculation_usage(engine),
    )
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
    stop = gen_kwargs.pop("stop", None)
    stopper = StopFilter(stop) if stop else None
    stopped = False
    emitted: list[str] = []
    # *messages* arrive already inlet-transformed; count tokens on what
    # inference sees (matches the chat path) if not already provided.
    if prompt_tokens is None:
        prompt_tokens = await asyncio.get_running_loop().run_in_executor(
            None, engine.count_tokens, _messages_prompt_text(messages))

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()

    # See _stream_sse: a mid-stream disconnect must stop the producer thread so it
    # releases llama.py's per-model _inference_lock instead of running to
    # end-of-generation and blocking the next request to this model. Also
    # registered with residency's cancel broadcast - see _stream_sse.
    cancel_event = threading.Event()
    residency.register_cancel(engine.display_name, cancel_event)

    class _StatusSignal:
        def __init__(self, text: str) -> None:
            self.text = text

    def _generate():
        # chat_stream INSIDE the try: it eagerly runs the auto-reload before
        # returning the generator, so an eager raise must not escape the try and
        # orphan the consumer (see the fuller note in _stream_sse).
        _log_assembled_prompt(messages)
        gen = None
        def _on_status(s: str) -> None:
            loop.call_soon_threadsafe(token_queue.put_nowait, _StatusSignal(s))

        try:
            gen_opts = dict(gen_kwargs)
            gen_opts.pop("on_status", None)
            gen = engine.chat_stream(messages, on_status=_on_status, **gen_opts)
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
        drained = False
        try:
            try:
                while True:
                    token = await token_queue.get()
                    if token is None:
                        break
                    if isinstance(token, _StatusSignal):
                        chunk = {
                            "id": chunk_id, "object": "text_completion.chunk",
                            "created": ts, "model": model_id,
                            "choices": [{
                                "text": "", "index": 0, "finish_reason": None,
                                "status": token.text,
                                "status_code": STATUS_CODE_BY_TEXT.get(token.text),
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        continue
                    if isinstance(token, Exception):
                        gen_error = token
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    # Stream hook transforms each piece before it is recorded and sent,
                    # so usage and the audit trail reflect what the client receives.
                    if pipeline is not None and ctx is not None and pipeline.has("stream"):
                        token = pipeline.run_stream(token, ctx)
                    piece = token
                    if stopper is not None:
                        piece = stopper.feed(token)
                        stopped = stopper.hit
                        emitted.append(piece)
                    completion_parts.append(token)
                    if piece or stopper is None:
                        chunk = {
                            "id": chunk_id, "object": "text_completion.chunk",
                            "created": ts, "model": model_id,
                            "choices": [{"text": piece, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    if stopped:
                        break
                drained = True
            finally:
                # No-op on a clean finish (thread already exited after the sentinel);
                # on a disconnect it stops the producer so _inference_lock is released.
                cancel_event.set()
                if not drained and ctx is not None:
                    ctx.outcome = "abort"

            t.join()
        finally:
            residency.unregister_cancel(engine.display_name, cancel_event)
        gen_end = time.perf_counter()

    if stopper is not None and not stopped:
        tail = stopper.flush()
        if tail:
            tail_chunk = {
                "id": chunk_id, "object": "text_completion.chunk",
                "created": ts, "model": model_id,
                "choices": [{"text": tail, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(tail_chunk)}\n\n"

    error_text = ""
    if gen_error is not None:
        error_text = inference_error_text(gen_error)
        err = {
            "id": chunk_id, "object": "text_completion.chunk",
            "created": ts, "model": model_id,
            "choices": [{"text": error_text, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    streamed = "".join(emitted if stopped else completion_parts)
    outcome = _turn_outcome(gen_error, "stop")
    if ctx is not None:
        ctx.outcome = outcome
    # Outlet shapes only the recorded reply (the live stream already went out);
    # then record the exchange (audit + transcript), exactly like chat. A failed
    # generation skips the outlet; its recorded reply includes the error chunk.
    reply = streamed + error_text
    if (gen_error is None and pipeline is not None and ctx is not None
            and pipeline.has("outlet")):
        reply = await pipeline.run_outlet(streamed, messages, ctx)
    _audit_exchange(audit, transcript, messages, reply, outcome=outcome)
    _log_chat_reply(reply, finish_reason=("error" if gen_error is not None else "stop"))

    completion_tokens = await _count_streamed_tokens(engine, streamed)
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


COMPACTION_DISCONNECT_DETAIL = (
    "Client closed the request while the conversation was being compacted.")


def _needs_compaction(capacity, prompt_tokens, messages,
                      encoder_decoder: bool = False) -> bool:
    """True when *prompt_tokens* leaves less than the reply buffer (2048 tokens
    or 10% of *capacity*, whichever is larger) free in *capacity*, for a
    conversation of more than three messages. With *encoder_decoder* the reply
    does not occupy *capacity*, so it is True only when the prompt itself is
    larger than *capacity*."""
    if not (isinstance(capacity, int) and capacity > 0
            and isinstance(prompt_tokens, int) and len(messages) > 3):
        return False
    if encoder_decoder:
        return prompt_tokens > capacity
    return capacity - prompt_tokens < max(2048, int(capacity * 0.10))


def _engine_is_encoder_decoder(engine) -> bool:
    """True only when *engine* reports an encoder-decoder model with a real
    ``True`` (a stand-in engine without the attribute answers False)."""
    return getattr(engine, "encoder_decoder", False) is True


def context_overflow_detail(prompt_tokens: int, capacity: int,
                            encoder_decoder: bool = False) -> str:
    """The refusal text for a prompt larger than the context capacity. With
    *encoder_decoder* the text names the model's one-pass prompt limit instead
    of the context window settings."""
    if encoder_decoder:
        return (f"Prompt ({prompt_tokens} tokens) exceeds the {capacity} tokens "
                f"this encoder-decoder model reads in one pass. Shorten the "
                f"message or start a new chat.")
    return (f"Prompt ({prompt_tokens} tokens) exceeds the model's maximum "
            f"context capacity ({capacity} tokens). Start a new chat, "
            f"or raise it:  localm config n_ctx_max 32768  (or set ctx_auto "
            f"true to size it from free VRAM).")


def _resolve_disconnect_poll(request):
    """The async "has the client gone?" poll for *request*, or ``None``.

    Prefers the poll ``_DisconnectSignalMiddleware`` publishes under
    ``scope[_DISCONNECT_POLL_KEY]`` and falls back to
    ``request.is_disconnected`` for a request without that middleware. ``None``
    when *request* is ``None``."""
    if request is None:
        return None
    poll = None
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        poll = scope.get(_DISCONNECT_POLL_KEY)
    if poll is None:
        poll = getattr(request, "is_disconnected", None)
    return poll


async def _compact_for_capacity(engine, messages: list, request=None
                                ) -> tuple[list, bool, bool]:
    """Compact *messages* with ``compact_messages``, the summary written by
    *engine* with its reasoning channel off.

    Runs in an executor. The summariser generation stops early, and the
    compaction falls back to its digest, when the client disconnects (polled
    through *request* every 0.1 s), when residency broadcasts a cancel for
    this model, or when this coroutine is cancelled (the cancellation is then
    re-raised).

    Returns ``(messages, changed, disconnected)``; *disconnected* is True when
    a client disconnect was observed while compacting."""
    from localm.debuglog import logger as _dbg
    from localm.inference.compact import compact_messages
    _dbg.info("compacting conversation: %d message(s) for %s",
              len(messages), engine.display_name)
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    residency.register_cancel(engine.display_name, cancel)
    poll = _resolve_disconnect_poll(request)
    disconnected = {"v": False}

    def _gen_for_compact(ms: list[dict], max_t: int) -> str:
        parts = []
        gen = engine.chat_stream(ms, max_tokens=max_t, temperature=0.3,
                                 thinking=False)
        try:
            for tok in gen:
                if cancel.is_set():
                    break
                parts.append(tok)
        finally:
            gen.close()
        if cancel.is_set():
            raise RuntimeError("summarisation cancelled")
        return "".join(parts)

    fut = loop.run_in_executor(None, compact_messages, messages, _gen_for_compact)

    async def _watch() -> None:
        try:
            while not fut.done():
                if poll is not None and await poll():
                    disconnected["v"] = True
                    cancel.set()
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.exception("compaction disconnect watcher failed")

    watcher = asyncio.ensure_future(_watch())
    try:
        try:
            new_messages, changed = await fut
        except (asyncio.CancelledError, GeneratorExit):
            cancel.set()
            raise
    finally:
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        residency.unregister_cancel(engine.display_name, cancel)
    _dbg.info("compacted conversation: %d -> %d message(s)%s", len(messages),
              len(new_messages), " (client disconnected)" if disconnected["v"] else "")
    return new_messages, changed, disconnected["v"]


class _StopDetector:
    """Watches a non-streamed generation for a stop sequence in the visible
    reply (reasoning inside ``<think>`` is not searched), so the generation can
    end at the token that completes it."""

    def __init__(self, stops, gen_kwargs: dict) -> None:
        from localm.inference.gbnf import think_exit_marker
        from localm.textnorm import ThinkSplitter
        self._think = ThinkSplitter(exit_marker=think_exit_marker(
            gen_kwargs.get("grammar_lazy"), gen_kwargs.get("grammar_triggers")))
        self._stop = StopFilter(stops)

    def feed(self, token: str) -> bool:
        content, _reasoning = self._think.feed(token)
        self._stop.feed(content)
        return self._stop.hit


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
    coroutine keeps the per-model semaphore too). This is the non-streaming twin
    of the _stream_sse cancel path.

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
    residency.register_cancel(engine.display_name, cancel_event)
    poll = _resolve_disconnect_poll(request)
    stop = gen_kwargs.pop("stop", None)
    stop_detector = _StopDetector(stop, gen_kwargs) if stop else None

    def _run() -> str:
        _log_assembled_prompt(messages)
        gen = engine.chat_stream(messages, **gen_kwargs)
        parts: list[str] = []
        try:
            for token in gen:
                if cancel_event.is_set():
                    break
                if timing is not None and "first_token_at" not in timing:
                    timing["first_token_at"] = time.perf_counter()
                parts.append(token)
                if stop_detector is not None and stop_detector.feed(token):
                    break
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
            # A watcher failure must NEVER cancel a live generation: let it run
            # to completion and log why, rather than risk truncating a good reply
            # on a transient poll error.
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
        residency.unregister_cancel(engine.display_name, cancel_event)


def _capability_route_header(route, placement=None) -> dict:
    """Observability: render a capability-routing decision into a response-header
    dict so a user can answer "why did this request use that model".

    Empty when nothing about the choice needs explaining - no needs stated, or
    the model that would have answered already met them - so an ordinary request
    carries no extra header.

    Reports the gap tri-state as measured: ``"absent"`` for a confirmed no,
    ``"unknown"`` for never inspected. Those are different facts and a client
    must be able to tell them apart, so they are not flattened into one flag.
    Compact ASCII JSON, header-safe:
    ``{"resolved","requested","routed","pinned","gaps":{cap:"absent"|"unknown"},
    "unmet":[...]}``, plus ``"suggested":<model>`` when model autoswitch is
    ``ask`` and another model would have answered, plus ``"load_errors":[...]`` (each cut to 200
    characters) when every capable model failed to load, and, when a model was
    left out because its last load failed, ``"skipped":[{"model","failed_at",
    "retry_at","reason"}]`` (the times in epoch seconds). ``"note"`` carries
    the decision's one-line description (cut to 600 characters) whenever either
    is present. A routed decision whose answering model runs partly on the CPU
    (*placement*, an ``Engine.gpu_placement`` dict, with ``degraded`` true)
    adds ``"placement":{"gpu_layers","total_layers"}``, with
    ``"moe_cpu_layers"`` when routed experts stayed in system RAM."""
    if route is None or not getattr(route, "has_gap", False):
        return {}
    payload = {
        "resolved": route.resolved,
        "requested": route.current,
        "routed": route.routed,
        "pinned": route.pinned,
        "gaps": {c: ("absent" if s is False else "unknown")
                 for c, s in route.gaps.items()},
        "unmet": list(route.unmet),
    }
    suggested = getattr(route, "suggested", None)
    if suggested:
        payload["suggested"] = suggested
    load_errors = getattr(route, "load_errors", ())
    if load_errors:
        payload["load_errors"] = [str(e)[:200] for e in load_errors]
    skipped = getattr(route, "skipped", ())
    if skipped:
        payload["skipped"] = [
            {"model": s.model, "failed_at": int(s.failed_at),
             "retry_at": int(s.retry_at), "reason": s.reason}
            for s in skipped]
    if load_errors or skipped:
        payload["note"] = route.describe()[:600]
    if route.routed and isinstance(placement, dict) and placement.get("degraded"):
        payload["placement"] = {"gpu_layers": placement.get("gpu_layers_offloaded"),
                                "total_layers": placement.get("gpu_layers_total")}
        if placement.get("moe_cpu_layers"):
            payload["placement"]["moe_cpu_layers"] = placement["moe_cpu_layers"]
    try:
        blob = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return {}
    return {"X-Localm-Model-Routing": blob}


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
# cannot drift apart. Every entry is a ValueError subclass raised by a backend to
# carry a reason the caller can act on; each maps to the status that says whose
# problem it is.
#
# ORDER IS LOAD-BEARING and the table is a sequence, not a dict, for exactly that
# reason: ImageDecodeUnavailable and VisionInputError are BOTH UnsupportedInputError
# subclasses, so a base-class-first table would swallow them and report a missing
# image decoder as the caller's bad input. TriggerValidatorUnavailableError sits
# above InvalidGrammarError for the same reason: it IS one, and listed after its
# parent it would answer 400 - blaming the caller's pattern for a validator that
# was too busy to look at it. This is the same arm-ordering hazard documented in
# cli/chat.py's vision handling, and it has its own test.
#
# 503 for TriggerValidatorUnavailableError, and this is the one entry in the table
# that is not permanent: everything else here describes a request or a build that
# will fail identically on a retry, while a saturated probe pool clears on its own
# within seconds. "Service Unavailable" is the only status that says "try again"
# rather than "change something".
#
# 501 for ImageDecodeUnavailable, not 400 and not 503: the request is fine and the
# caller can do nothing about it, so 4xx would blame the wrong party; and the
# missing decoder will not appear on a retry, which is what 503 would promise.
# "Not Implemented" is exactly the permanent, server-side capability gap it is.
_BACKEND_ERROR_STATUS: tuple = (
    (ImageDecodeUnavailable, 501),
    (VisionInputError, 400),
    (UnsupportedInputError, 400),
    (ChatTemplateMissingError, 400),
    (GrammarUnsupportedError, 400),
    (TriggerValidatorUnavailableError, 503),
    (InvalidGrammarError, 400),
    (EmbedBatchTooLargeError, 413),
    (ContextCapacityExceededError, 413),
    (PretokenizerUnsafeInputError, 400),
)

# The same classes as a plain tuple, for use as an `except` clause. Derived from
# the table rather than written out again, so a class added to one is never
# missing from the other - a catch listing a class the table does not map would
# raise HTTPException(None, ...) at the moment it finally fired.
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


async def _count_streamed_tokens(engine, streamed: str) -> int:
    """Token count of text already delivered to the client, for the usage block.

    Falls back to the chars/4 estimate when the tokenizer REFUSES the text: the
    content has been streamed already, so raising here would end the response
    with no terminal chunk, no usage and no ``[DONE]``. A model can emit a run
    the pre-tokenizer aborts on just as a caller can send one.
    """
    from localm.inference.pretokenizer_guard import count_tokens_or_estimate
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, count_tokens_or_estimate, engine.count_tokens, streamed,
        "the generated text")


def inference_error_text(exc: BaseException) -> str:
    """The `[inference error: ...]` body a FAILED generation is rendered as, on
    every one of the four generation paths.

    ONE implementation for all four, so the four cannot state the same fact
    differently.

    THE PATHS ARE SCRUBBED. A mid-generation
    RuntimeError is not always a tidy "not enough free VRAM" sentence: the GGUF
    loader raises `Failed to load model: <absolute path>` with a native stderr
    tail appended, and an auto-reload inside chat_stream can surface exactly
    that here. Handing a client the machine's directory layout is the
    disclosure `pathscrub` exists for, and `localm/bugreport/scrub.py` already
    names scrub_paths as the rule for a response to a lower-privileged caller.

    scrub_paths REDACTS, it does not mute: the cause, the file name and the
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
    # Call the engine's real methods directly, with no hasattr-guarded
    # fallbacks: a method-less engine must surface its failure rather than
    # return a fabricated 200.
    capacity = engine.context_capacity()
    if prompt_tokens is None:
        prompt_tokens = await asyncio.get_running_loop().run_in_executor(None, engine.count_messages_tokens, messages)

        if _needs_compaction(capacity, prompt_tokens, messages,
                             _engine_is_encoder_decoder(engine)):
            new_messages, changed, gone = await _compact_for_capacity(
                engine, messages, request)
            if gone:
                raise HTTPException(499, COMPACTION_DISCONNECT_DETAIL)
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
        except _BACKEND_ERROR_TYPES as e:
            # A backend refusal the CALLER can act on (an image this vision model
            # could not process, a grammar the deferred check finally rejected at
            # sampler-build time, a missing image decoder). Every one of these is a
            # ValueError, so without this arm they would sail past the
            # RuntimeError catch below into the generic Exception backstop and
            # come back as {"detail": "Internal server error"} with its cause
            # thrown away, while the STREAMING twin of this function delivers
            # that same cause to the client.
            #
            # Raised as an HTTPException rather than rendered inline like the
            # RuntimeError case below, because these are not generation failures
            # that produced a partial answer: nothing was generated and the status
            # is the honest report. The streaming path cannot match the STATUS
            # (its role chunk, and therefore the 200 header, is already on the wire
            # before generation starts), but it does carry the same reason and
            # marks finish_reason="error" - so both paths tell the caller what went
            # wrong, which is the property that was actually broken.
            raise HTTPException(backend_error_status(e), str(e)) from e
        except RuntimeError as e:
            # A generation FAILURE (not enough free VRAM for this prompt, a
            # conversation that outgrew n_ctx_max, a native decode error) is raised
            # as RuntimeError; it must reach the client as a clean reply, never a
            # raw HTTP 500 - the non-streaming twin of the streaming path's
            # gen_error handling. Catch ONLY RuntimeError, not Exception: a broken
            # engine (e.g. a method-less mock -> AttributeError) is a real bug
            # that must surface loudly rather than be masked as an "inference
            # error", and CancelledError (client disconnect) must not be
            # swallowed either.
            from localm.debuglog import logger as _dbg
            _dbg.exception("non-streaming generation failed")
            gen_error = e
            text = inference_error_text(e)
        gen_end = time.perf_counter()
    first_token_at = timing.get("first_token_at")

    stop = gen_kwargs.get("stop")
    stopped = False
    if stop and gen_error is None:
        from localm.inference.gbnf import think_exit_marker
        from localm.textnorm import split_think
        visible, reasoning_text = split_think(text, exit_marker=think_exit_marker(
            gen_kwargs.get("grammar_lazy"), gen_kwargs.get("grammar_triggers")))
        visible, stopped = apply_stop(visible, stop)
        if stopped:
            text = (f"<think>{reasoning_text}</think>" if reasoning_text else "") + visible

    finish_reason = ("error" if gen_error is not None
                     else "stop" if stopped else _engine_finish_reason(engine))
    outcome = _turn_outcome(gen_error, finish_reason)
    if ctx is not None:
        ctx.outcome = outcome
    # Outlet fully controls the returned content in the non-streaming path (but a
    # failed generation surfaces its error verbatim, not reshaped by the outlet).
    if gen_error is None and pipeline is not None and ctx is not None and pipeline.has("outlet"):
        text = await pipeline.run_outlet(text, messages, ctx)

    _audit_exchange(audit, transcript, messages, text, outcome=outcome)
    _log_chat_reply(text, finish_reason=finish_reason)

    # Split the model's <think> reasoning out of the visible answer into a
    # separate field, so API clients get clean content (token count stays on
    # the full generated text - reasoning was still generated).
    from localm.inference.gbnf import think_exit_marker
    from localm.textnorm import split_think
    answer, reasoning = split_think(text, exit_marker=think_exit_marker(
        gen_kwargs.get("grammar_lazy"), gen_kwargs.get("grammar_triggers")))

    completion_tokens = await _count_streamed_tokens(engine, text)
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
        context_capacity=capacity,
        mtp=_mtp_usage(engine),
        speculation=_speculation_usage(engine),
    )

    response = ChatResponse(
        id=make_chunk_id(),
        created=int(time.time()),
        model=model_id,
        choices=[
            FullChoice(
                message=Message(role="assistant", content=answer,
                                reasoning_content=reasoning or None),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
    return JSONResponse(response.model_dump())


def _protocol_messages_to_dicts(messages: list[Message]) -> list:
    """Convert Pydantic Message objects to plain dicts for backends. A message's
    ``origin`` marker, when set, is kept as an ``"origin"`` key; an unmarked
    message has no such key."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            spans = getattr(msg, "untrusted_spans", None)
            content = msg.content
            if spans:
                from localm.textguard import GuardedText
                content = GuardedText(content, [tuple(s[:2]) for s in spans
                                                if len(s) >= 2])
            result.append({"role": msg.role, "content": content})
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
        origin = getattr(msg, "origin", None)
        if origin:
            result[-1]["origin"] = origin
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

    When LOCALM_RESTART_GUI_MOUNTED is set (a restart re-exec'd this process
    from an instance whose GUI had been mounted live), the variable is removed
    from the environment and the GUI is mounted onto *app* once the instance is
    advertised, before serving starts.
    """
    from localm import instances, portmux
    from localm.config import home_dir
    # Removed from os.environ before portmux.run_server, whose crash-recovery
    # watchdog inherits os.environ. See
    # test_run_advertised_removes_the_gui_mount_flag_before_serving.
    remount_gui = os.environ.pop(_GUI_MOUNTED_ENV, None) == "1"
    if log_level is None:
        from localm.debuglog import uvicorn_log_level
        log_level = uvicorn_log_level()
    scheme = "https" if ssl_certfile else "http"

    # No custom Win32 Ctrl+C handler here. Such a handler resolves the loop on
    # the control-handler OS thread, NOT the serving loop (portmux's asyncio.run
    # makes a fresh loop), so loop.stop() never fires while the handler still
    # returns True, eating the event and defeating uvicorn's own SIGINT
    # shutdown. Without one, Ctrl+C flows through uvicorn (KeyboardInterrupt
    # caught in portmux.run_server), kept responsive on Windows by portmux's
    # loop-wakeup.

    with instances.advertise(app, home_dir(), host=host, port=port, mode=mode,
                             scheme=scheme, project=project, isolated=isolated):
        try:
            if remount_gui:
                _remount_gui(app)
            # On a TLS bind, also catch a plain-http request on the same port
            # with an https redirect; a plain bind closes a TLS connection opened
            # on its port. In debug mode uvicorn logs at "info" so the console
            # shows requests.
            portmux.run_server(app, host=host, port=port, log_level=log_level,
                               ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
        finally:
            # Serving has ended - Ctrl+C, a signal, or an error. Runs the same
            # stop sequence the GUI/tray Stop button runs. Must not exit the
            # process here: the caller's own finally still closes a tray icon
            # and an mDNS advertisement.
            _announce_stopping()
            _shutdown_teardown(
                instance_id=getattr(getattr(app, "state", None),
                                    "instance_id", None))


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

    Advertises itself in the instance registry so a future
    launch can discover and attach to it; ``isolated`` keeps it invisible to
    discovery.
    """
    app = create_app(engine, api_landing=True)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    run_advertised(app, host, port, mode=mode, ssl_certfile=ssl_certfile,
                   ssl_keyfile=ssl_keyfile, project=project, isolated=isolated)

