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
import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Request
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
from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatResponse,
    FullChoice, Message, UsageInfo, make_chunk_id,
)

# Map of display name -> Engine instance
_engines: dict[str, Engine] = {}
# Order of model usage (display names, MRU at the end)
_engines_lru: list[str] = []
# Default/startup model name
_default_model_name: str | None = None
# Active model name (most recently used/loaded)
_active_model_name: str | None = None

# Inference serialisation - per-model semaphores mapping display name -> Semaphore
_inference_sems: dict[str, asyncio.Semaphore] = {}

# Backward compatibility references
_engine: Engine | None = None
_inference_sem: asyncio.Semaphore | None = None

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
    own file_size computation so the VRAM estimate written to the coordination
    registry is consistent with the number switch_engine itself used to decide
    whether eviction was needed."""
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
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
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


def _attempt_cooperative_unload() -> bool:
    """Best-effort: ask a live sibling localm instance (found via the
    cross-install GPU-coordination registry) to release its own VRAM, so this
    instance does not have to give up and 503 just because ITS OWN local
    eviction candidates are all busy. Returns True once a peer confirms it
    freed its model.

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
    for peer in holders:
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


async def switch_engine(name: str, make_engine, *, on_active=None, preempt: bool = True) -> dict:
    global _engines, _engines_lru, _active_model_name, _engine_factory
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
            
        if name in _engines and _engines[name].loaded:
            if name in _engines_lru:
                _engines_lru.remove(name)
            _engines_lru.append(name)
            _active_model_name = name
            _engine = _engines[name]
            _inference_sem = sem
            if on_active is not None:
                on_active(name)
            return {"status": "already_active", "model": name}

        # Perform VRAM check and eviction
        from pathlib import Path
        from localm.discover import vram_capacity
        from localm.model_manager import get_model_info
        from localm.config import load_registry
        
        registry = load_registry()
        info = get_model_info(name)
        # file_size feeds the VRAM-eviction estimate below, which is gated on a
        # non-empty registry. So it is only needed for a registered model.
        file_size = 0
        if info is not None:
            m_path, _ = info
            p = Path(m_path)
            file_size = p.stat().st_size if p.is_file() else (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 4 * 1024 ** 3)
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
            from localm.vram import wait_for_vram_release
            vram_required = int(file_size * 1.2)
            headroom = 1024 ** 3  # 1GB VRAM headroom

            while True:
                v_info = vram_capacity()
                free_vram = v_info.get("free")
                measurable = free_vram is not None
                if measurable and free_vram >= vram_required + headroom:
                    break

                # Make room. Measurable VRAM: evict idle models until the new one
                # fits. NOT measurable (default GGUF-only / non-NVIDIA, no "free"
                # from discover.vram_info): cannot prove it fits alongside others,
                # so fall back to single-resident (evict every idle model first)
                # rather than stacking until the driver OOMs (AUDIT-CRIT-2).
                evict_name = None
                for candidate in _engines_lru:
                    if candidate == name:
                        continue  # never evict the model we are loading
                    candidate_engine = _engines.get(candidate)
                    if candidate_engine is not None and getattr(candidate_engine, "active_requests", 0) == 0:
                        evict_name = candidate
                        break

                if evict_name is None:
                    # Nothing idle left to evict.
                    if not measurable:
                        # Unmeasurable and every remaining model is busy (or none
                        # loaded): freed what we safely can, load best-effort (the
                        # pre-multi-model behaviour).
                        break
                    # Local eviction exhausted: before giving up, best-effort ask a
                    # sibling localm instance to release ITS VRAM (multi-instance
                    # coordination, see localm.gpu_registry). Off the event loop (it
                    # may make a blocking loopback call). Advisory: any failure falls
                    # through to the 503 below, never a harder failure than baseline.
                    cooperated = await loop.run_in_executor(None, _attempt_cooperative_unload)
                    if cooperated:
                        continue
                    if _engines:
                        raise HTTPException(503, f"VRAM exhausted (cannot load '{name}'). "
                                            "All other loaded models are busy.")
                    raise HTTPException(503, f"Not enough VRAM to load '{name}' "
                                        f"(need ~{vram_required // 1024 ** 2} MB, "
                                        f"{(free_vram or 0) // 1024 ** 2} MB free).")

                evict_engine = _engines[evict_name]
                free_before = free_vram
                # Safe to unload without the victim's own semaphore: active_requests
                # == 0 means no request is pinned on it (a request pins its engine
                # for its whole lifetime - AUDIT-CRIT-1), so no decode races the
                # native free. Taking the victim sem while holding the target sem
                # would also risk a two-switch lock-ordering deadlock.
                await loop.run_in_executor(None, evict_engine.unload)
                del _engines[evict_name]
                _engines_lru.remove(evict_name)
                if evict_name in _inference_sems:
                    del _inference_sems[evict_name]

                # Wait for the native VRAM free to land before re-checking, so the
                # next iteration does not see a stale-low reading and over-evict
                # (driver-hang guard, AUDIT-MED-11). Only meaningful when measurable.
                if measurable and free_before is not None:
                    await loop.run_in_executor(
                        None,
                        lambda: wait_for_vram_release(
                            lambda: vram_capacity().get("free"), before_bytes=free_before))

        if name in _engines:
            new_engine = _engines[name]
        else:
            new_engine = _engine_factory(name)
            
        cancel = threading.Event()
        if preempt:
            # Only an explicit switch registers a load-cancel hook, so only a newer
            # explicit switch can abort this load; API-routed loads (preempt=False)
            # run to completion, never cancelled by a concurrent different-model load.
            _switch_cancel = cancel
            _switch_loading = name
            if hasattr(new_engine, "set_load_cancel"):
                new_engine.set_load_cancel(cancel)
        try:
            await loop.run_in_executor(None, new_engine.load)
        except ModelLoadCancelled:
            return {"status": "superseded", "model": name, "by": _switch_desired}
        finally:
            if preempt and _switch_cancel is cancel:
                _switch_cancel = None
                _switch_loading = None
                
        _engines[name] = new_engine
        _engines_lru.append(name)
        _active_model_name = name
        _engine = new_engine
        _inference_sem = sem
        if on_active is not None:
            on_active(name)
        # Cross-install GPU coordination: reflect the newly-active model so a
        # sibling's next VRAM/eviction check sees fresh state. No-op when not
        # registered (see _gpu_registry_sync).
        _gpu_registry_sync()
        return {"status": "loaded", "model": name}


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

    if name in _engines and _engines[name].loaded:
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
    from localm.discover import vram_capacity
    from localm.vram import wait_for_vram_release

    def _free():
        return vram_capacity().get("free")

    before = _free()
    unloaded_models = []

    for name in list(_engines.keys()):
        engine = _engines[name]
        if not engine.loaded:
            continue
        sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
        async with sem:
            await loop.run_in_executor(None, engine.unload)
            unloaded_models.append(name)
            if name in _engines_lru:
                _engines_lru.remove(name)

    # Update compatibility pointers
    _active_model_name = None
    _engine = None
    _inference_sem = None

    if before is not None and unloaded_models:
        released, after = await loop.run_in_executor(
            None, lambda: wait_for_vram_release(_free, before_bytes=before))
    else:
        released, after = 0, before

    result = {
        "status": "unloaded" if unloaded_models else "already_unloaded",
        "model": unloaded_models[0] if unloaded_models else "none",
        "unloaded_models": unloaded_models
    }
    if before is not None:
        result.update(vram_freed=released,
                      vram_before_bytes=before, vram_after_bytes=after)
    # Cross-install GPU coordination: reflect the now-empty/changed state for a
    # sibling's next eviction decision. No-op when not registered.
    _gpu_registry_sync()
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
    from localm.discover import vram_capacity
    from localm.vram import wait_for_vram_release

    engine = _engines.get(name)
    if engine is None or not engine.loaded:
        return {"status": "already_unloaded", "model": name}

    def _free():
        return vram_capacity().get("free")

    before = _free()
    sem = _inference_sems.setdefault(name, asyncio.Semaphore(1))
    async with sem:
        await loop.run_in_executor(None, engine.unload)
        if name in _engines_lru:
            _engines_lru.remove(name)

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
    if before is not None:
        result.update(vram_freed=released,
                      vram_before_bytes=before, vram_after_bytes=after)
    _gpu_registry_sync()
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
            await loop.run_in_executor(None, engine.unload)

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
            # not registered.
            _gpu_registry_sync()

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
            _gpu_registry_sync()
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.warning("gpu-registry heartbeat failed (continuing)", exc_info=True)

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
            # from the key VALUE so an owner-key ROLL does not log the owner out
            # (the S1 fix).
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
    header = request.headers.get(CSRF_HEADER, "")
    sid = (request.cookies.get(SESSION_COOKIE) or "").strip()
    if not header or not sid:
        return False
    return hmac.compare_digest(header, csrf_token_for(request, sid))


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
    if source == "cookie":
        from localm import sessions
        rec = sessions.lookup(token)
        if rec:
            return rec.get("key_hash")
        return None
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


def _do_shutdown() -> None:
    """SRV-4: the actual stop sequence. Unload the model FIRST so the native
    context is freed cleanly (a hard exit while it is loaded segfaults during
    teardown), clear the crash marker so this intentional stop is not reported as
    a crash, then exit the process so the stop is guaranteed (Ctrl+C sometimes
    does nothing). Separated from the route so it can be tested without exiting."""
    # Unload all engines in the multi-model dictionary
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            pass
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            pass
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard()
    except Exception:
        pass
    import os
    os._exit(0)


def _request_shutdown(delay: float = 0.25) -> None:
    """Run _do_shutdown shortly after returning, so the 200 response flushes to
    the client before the process exits."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_shutdown()

    threading.Thread(target=_run, daemon=True).start()


def _restart_argv() -> list:
    """The command line to re-launch this server. Always ``python -m localm <args>``
    - the canonical entry the codebase uses - so a restart works regardless of how
    the server was originally started (a console-script .exe, ``-m``, or a script
    path, any of which can make ``sys.argv[0]`` un-re-runnable by the interpreter)."""
    import sys
    return [sys.executable, "-m", "localm", *sys.argv[1:]]


def _do_restart() -> None:
    """R18: restart this server IN PLACE. Unload the model FIRST (clean native
    teardown, like _do_shutdown - a hard re-exec while it is loaded can segfault),
    clear the crash marker so this intentional restart is not reported as a crash,
    then re-exec the same command line so the server comes back on the same port.
    os.execv replaces the process image and does not return on success. Separated
    from the route so it can be tested without actually re-execing."""
    # Unload all engines in the multi-model dictionary
    for engine in list(_engines.values()):
        try:
            engine.unload()
        except Exception:
            pass
    # Unload mocked _engine if it is set and wasn't in _engines
    if _engine is not None and _engine not in _engines.values():
        try:
            _engine.unload()
        except Exception:
            pass
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard()
    except Exception:
        pass

    try:
        global _audit
        if _audit is not None and hasattr(_audit, "close"):
            _audit.close()
    except Exception:
        pass

    try:
        from localm.debuglog import dump_ring_buffer, flush_log_handlers, recent_activity
        # Privacy mode opts out of ALL automatic disk traces, so skip the
        # crash-recovery breadcrumb dumps (ring buffer + pre_restart.log): they are
        # session-derived INFO breadcrumbs written without the user asking. If the
        # mode cannot be resolved, fail toward privacy (skip) - matching audit.py's
        # fail-safe-to-privacy default - rather than write a trace the user may have
        # opted out of.
        try:
            from localm.audit import SessionMode, effective_mode
            _privacy = effective_mode("server") == SessionMode.PRIVACY
        except Exception:
            _privacy = True
        if not _privacy:
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
        pass

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

    os.execv(sys.executable, _restart_argv())


def _request_restart(delay: float = 0.25) -> None:
    """Run _do_restart shortly after returning, so the 200 response flushes to the
    client before the process re-execs (mirrors _request_shutdown)."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_restart()

    threading.Thread(target=_run, daemon=True).start()


def create_app(engine: Optional[Engine], *, api_landing: bool = False) -> FastAPI:
    global _engine, _inference_sem, _engines, _engines_lru, _default_model_name, _active_model_name, _inference_sems, _last_activity_per_model
    
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
    # through /v1/chat/completions and lands here.
    from localm.audit import effective_mode, make_audit_log, make_transcript
    _mode = effective_mode("server")
    _audit = make_audit_log(_mode, label="server")
    _transcript = make_transcript(_mode, label="server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem, _inference_sems, _active_model_name
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
                pass
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
            _audit.close()

    app = FastAPI(
        title="localm inference server",
        version="0.1.1",
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
            _dbg.debug(
                "%s %s -> %d (%.0f ms)",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
            )
            return response

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
        if (request.method in _UNSAFE_METHODS
                and not request.url.path.startswith(_CROSS_ORIGIN_OK)):
            # Same-origin / CORS-allowlist check.
            if _cross_origin_refused(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request refused "
                             "(only same-origin requests or a configured "
                             "'cors_origins' may use this endpoint)."},
                )
            # H5: open-mode management gate. With no key configured, management
            # routes still require the per-process shell token (injected into the
            # loopback GUI shell), so a no-Origin local client (curl, a script)
            # can no longer mint a key, flip config, install a plugin, load a
            # model, or browse the filesystem unauthenticated. Protected mode (a
            # key exists) is bearer-auth'd on the route. The token is required even
            # for an allowlisted CORS origin (F2): an Origin header is forgeable, so
            # it is not a management credential - a configured external origin must
            # use an API key for state changes.
        is_unsafe = request.method in _UNSAFE_METHODS
        is_metadata_get = (
            request.method == "GET"
            and (request.url.path.startswith("/api/") or request.url.path.startswith("/v1/"))
            and request.url.path != "/api/session"
            and not request.url.path.startswith("/v1/models")
        )
        if (is_unsafe or is_metadata_get) and not request.url.path.startswith(_CROSS_ORIGIN_OK):
            from localm.auth import any_key_configured, require_auth_enabled
            if not any_key_configured() and not require_auth_enabled():
                token = getattr(request.app.state, "shell_token", None)
                presented = _bearer_token(request)
                token_ok = bool(token and presented
                                 and hmac.compare_digest(presented, token))
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

    # Added LAST (== outermost, sees the raw ASGI receive() before every other
    # middleware/BaseHTTPMiddleware handler above, none of which touch the body).
    app.add_middleware(_BodyStreamCapMiddleware)

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
    from localm.inference.routes import plugins as _routes_plugins
    _routes_plugins.register(app, ctx)
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


def _tokens_per_sec(completion_tokens: int, elapsed: float) -> Optional[float]:
    """Generation throughput, or None when not measurable."""
    if not completion_tokens or elapsed <= 0:
        return None
    return round(completion_tokens / elapsed, 2)


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
    from localm.inference.textnorm import ThinkSplitter

    chunk_id = make_chunk_id()
    ts = int(time.time())
    think = ThinkSplitter()   # route <think> reasoning into delta.reasoning_content (H4)

    prompt_tokens = engine.count_messages_tokens(messages)

    # Context-limit handling: compact_messages when close to the limit; reserve a
    # 2048-token buffer for compaction overhead + response generation.
    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            new_messages, changed = compact_messages(messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = engine.count_messages_tokens(messages)

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
        gen = engine.chat_stream(messages, **gen_kwargs)
        try:
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
            # deterministically - the whole point of the cancel path.
            try:
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
        gen_elapsed = time.perf_counter() - gen_start
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
    completion_tokens = engine.count_tokens(streamed)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(completion_tokens, gen_elapsed),
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
    prompt_tokens = engine.count_tokens(_messages_prompt_text(messages))

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()

    import threading

    # See _stream_sse: a mid-stream disconnect must stop the producer thread so it
    # releases llama.py's per-model _inference_lock instead of running to
    # end-of-generation and blocking the next request to this model.
    cancel_event = threading.Event()

    def _generate():
        gen = engine.chat_stream(messages, **gen_kwargs)
        try:
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
            # _inference_lock (see the fuller note in _stream_sse).
            try:
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
        gen_elapsed = time.perf_counter() - gen_start

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

    completion_tokens = engine.count_tokens(streamed)
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
            "tokens_per_sec": _tokens_per_sec(completion_tokens, gen_elapsed),
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def _complete(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
):
    loop = asyncio.get_running_loop()

    # Call the engine's real methods directly. The previous hasattr-guarded
    # fallbacks (100 prompt tokens, 4096 capacity, an "ok" completion, 10
    # completion tokens) let a method-less mock pass through, so a broken engine
    # returned a fabricated 200 instead of surfacing the failure (AUDIT rule 5 /
    # no facade).
    prompt_tokens = engine.count_messages_tokens(messages)

    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            new_messages, changed = compact_messages(messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = engine.count_messages_tokens(messages)
    def _run():
        return "".join(engine.chat_stream(messages, **gen_kwargs))

    # Serialise inference - only one request runs at a time
    async with sem:
        gen_start = time.perf_counter()
        text = await loop.run_in_executor(None, _run)
        gen_elapsed = time.perf_counter() - gen_start

    # Outlet fully controls the returned content in the non-streaming path.
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        text = await pipeline.run_outlet(text, messages, ctx)

    _audit_exchange(audit, transcript, messages, text)

    # Split the model's <think> reasoning out of the visible answer into a
    # separate field (H4), so API clients get clean content (token count stays on
    # the full generated text - reasoning was still generated).
    from localm.inference.textnorm import split_think
    answer, reasoning = split_think(text)

    completion_tokens = engine.count_tokens(text)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        tokens_per_sec=_tokens_per_sec(completion_tokens, gen_elapsed),
    )

    response = ChatResponse(
        id=make_chunk_id(),
        created=int(time.time()),
        model=model_id,
        choices=[
            FullChoice(
                message=Message(role="assistant", content=answer,
                                reasoning_content=reasoning or None),
                finish_reason=_engine_finish_reason(engine),
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


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8642,
          ssl_certfile: Optional[str] = None,
          ssl_keyfile: Optional[str] = None,
          project: Optional[str] = None,
          isolated: bool = False) -> None:
    """Start the server - blocks until Ctrl+C.

    When ``ssl_certfile`` / ``ssl_keyfile`` are given (built-in TLS, NET-1), the
    server speaks HTTPS on this port; a plain-HTTP request to it then fails the
    TLS handshake (effectively refused) rather than crossing the network in
    cleartext.

    Advertises itself in the instance registry (H6 phase 3/4) as an ``api``
    surface so a future launch can discover and attach to it; ``isolated`` keeps
    it invisible to discovery.
    """
    from localm import instances, portmux
    from localm.config import home_dir

    app = create_app(engine, api_landing=True)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    scheme = "https" if ssl_certfile else "http"

    # SRV-CTRLC: no custom Win32 Ctrl+C handler here. A previous one resolved the
    # loop on the control-handler OS thread - NOT the serving loop (portmux's
    # asyncio.run makes a fresh loop) - so loop.stop() never fired, yet it returned
    # True, eating the event and defeating uvicorn's own SIGINT shutdown: it ATE
    # Ctrl+C and the server hung. Removing it lets Ctrl+C flow through uvicorn
    # (KeyboardInterrupt caught in portmux.run_server), kept responsive on Windows
    # by portmux's SRV-6 loop-wakeup. Verified live (Ctrl+Break).

    with instances.advertise(app, home_dir(), host=host, port=port, mode="api",
                             scheme=scheme, project=project, isolated=isolated):
        # On a TLS bind, also catch a plain-http request on the same port with an
        # https redirect (issue 8); plain binds are a direct uvicorn.run. SRV-5:
        # in debug mode uvicorn logs at "info" so the console shows requests.
        from localm.debuglog import uvicorn_log_level
        portmux.run_server(app, host=host, port=port,
                           log_level=uvicorn_log_level(),
                           ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)

