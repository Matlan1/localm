# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes, registry mutation group: remove, alias, rename, set type
and relocate a registered model."""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import pathsafe
from localm import scopes
from localm.inference.http_server import (principal_id, require_fs_host,
                                          require_scope)
import localm.inference.http_server as _hs
from localm.executor import get_plugin_executor
from localm.plugins.gui.routes.models._context import (ModelRouteContext,
                                                       _require_registered)
from localm.plugins.gui.web import (AdapterAttachRequest, AdapterDetachRequest,
                                    AliasRequest, RelocateModelRequest,
                                    RemoveModelRequest, RenameModelRequest,
                                    SetTypeRequest)


def _base_is_resident(base: str, registry: dict) -> bool:
    """True when a loaded engine in this process runs the model file that the
    registered name *base* points at, under any of its names. Does filesystem
    I/O; callers on the event loop use the executor."""
    from pathlib import Path

    from localm.model_manager import _entry_path, names_same_model
    base_path = _entry_path(registry.get(base))
    for key, engine in list(_hs._engines.items()):
        if not getattr(engine, "loaded", False):
            continue
        if names_same_model(key, base, registry):
            return True
        model_path = getattr(engine, "model_path", None)
        if model_path is None or base_path is None:
            continue
        try:
            if Path(str(model_path)).resolve() == Path(base_path).resolve():
                return True
        except (OSError, ValueError):
            return True
    return False


def register(app: FastAPI, context: ModelRouteContext) -> None:
    active_model = context.active_model
    jobs = context.jobs

    @app.post("/api/models/remove", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_remove(req: RemoveModelRequest, request: Request):
        registry = _require_registered(req.model)
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
        # A model can be resident in VRAM (loaded) without being the ACTIVE one.
        # Without this guard, removing a background-loaded model deletes the file out
        # from under a live Engine that still has it open or mmap'd.
        engine = _hs._engines.get(req.model)
        if engine is not None and engine.loaded:
            raise HTTPException(409, "Cannot remove a loaded model - unload it first")
        # Both guards above ask whether a loaded engine is keyed under this NAME,
        # which misses a model renamed by the separate `localm rename` process. Ask
        # instead whether any live engine holds THAT FILE. Off the event loop because
        # it resolves registry paths.
        loop = asyncio.get_running_loop()
        hold = await loop.run_in_executor(
            get_plugin_executor(), _hs.loaded_engine_holding_model_file,
            req.model, registry)
        if hold is not None:
            # Two different refusals: "still loaded as X" is a fact, while a cautious
            # refusal means a path could not be resolved.
            if hold.reason is None:
                detail = (f"Cannot remove '{req.model}' - its file is still "
                          f"loaded as '{hold.key}'. Unload it first.")
            else:
                detail = (f"Cannot remove '{req.model}' - '{hold.key}' is "
                          f"loaded and {hold.reason}, so it cannot be ruled "
                          f"out as holding this file. Unload it first.")
            raise HTTPException(409, detail)
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"],
                             owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_alias(req: AliasRequest):
        registry = _require_registered(req.model)
        # alias_model stores under the SANITIZED name (a space / slash / colon can
        # never become a raw registry key), so precheck against that same name and
        # report it back.
        from localm.model_manager import _sanitize_name, alias_model
        alias = _sanitize_name(req.alias)
        if alias in registry:
            raise HTTPException(409, f"Name already taken: {alias}")
        loop = asyncio.get_running_loop()
        try:
            created = await loop.run_in_executor(
                get_plugin_executor(), alias_model, req.model, req.alias)
        except Exception as e:
            raise HTTPException(400, f"Alias failed: {e}") from e
        if not created:
            # alias_model returns False for "model vanished" and "name taken" alike,
            # and both were prechecked above, so reaching here means a concurrent
            # writer won the race. Report which one it lost.
            from localm.config import load_registry
            if req.model not in load_registry():
                raise HTTPException(404, f"Model not registered: {req.model}")
            raise HTTPException(409, f"Name already taken: {alias}")
        return {"status": "aliased", "model": req.model, "alias": alias}

    @app.post("/api/models/rename", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_rename(req: RenameModelRequest):
        """Rename a registered model. Unlike alias, the OLD name stops
        working - this MOVES the registration (plus best-effort migrates
        config/jobs/RAG references that named it). Renaming the currently
        ACTIVE (or merely loaded) model is allowed: the live engine is
        re-keyed in place right after the registry move, so it keeps serving
        under its new name instead of being orphaned under the old one.

        The registry move and the re-key are one operation, not two steps a
        route is trusted to perform in order, so both this and the /v1 sibling
        go through the single helper that pairs them."""
        return await _hs.rename_registered_model(req.model, req.new_name)

    @app.post("/api/models/type", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_set_type(req: SetTypeRequest):
        """Change a registered model's type (the one-click set-type control). A
        type='unknown' model is not auto-loaded as chat but stays runnable by name;
        this corrects a mis-detected or bulk-imported model's type."""
        from localm.model_manager import MODEL_TYPES, set_model_type
        _require_registered(req.model)
        if req.model_type not in MODEL_TYPES:
            raise HTTPException(
                400, f"Invalid type: {req.model_type}. "
                     f"One of: {', '.join(sorted(MODEL_TYPES))}")
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            get_plugin_executor(), set_model_type, req.model, req.model_type)
        if not ok:
            raise HTTPException(400, f"Could not set type for {req.model}")
        return {"status": "typed", "model": req.model, "model_type": req.model_type}

    @app.post("/api/models/adapters/attach", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_adapter_attach(req: AdapterAttachRequest):
        """GUI form of `localm adapter attach ADAPTER BASE --scale S`. A refusal
        (different architecture, not a GGUF chat base, invalid scale) is a 400
        carrying the registry's own message. `needs_reload` is true when BASE
        may be resident: adapters are applied when a model loads, so a loaded
        base keeps its previous adapters until it is unloaded and loaded again."""
        from localm.model_manager import AdapterError, attach_adapter
        _require_registered(req.adapter)
        registry = _require_registered(req.base)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                get_plugin_executor(), attach_adapter, req.adapter, req.base, req.scale)
        except AdapterError as e:
            raise HTTPException(400, str(e)) from e
        resident = await loop.run_in_executor(
            get_plugin_executor(), _base_is_resident, req.base, registry)
        return {"status": "attached", "adapter": req.adapter, "base": req.base,
                "scale": req.scale, "needs_reload": resident}

    @app.post("/api/models/adapters/detach", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_adapter_detach(req: AdapterDetachRequest):
        """GUI form of `localm adapter detach ADAPTER`; the adapter stays
        registered. 404 when it is not attached."""
        from localm.model_manager import detach_adapter
        registry = _require_registered(req.adapter)
        base = registry[req.adapter].get("base") if isinstance(registry[req.adapter], dict) else None
        loop = asyncio.get_running_loop()
        detached = await loop.run_in_executor(
            get_plugin_executor(), detach_adapter, req.adapter)
        if not detached:
            raise HTTPException(404, f"'{req.adapter}' is not an attached adapter")
        needs_reload = False
        if isinstance(base, str) and base in registry:
            needs_reload = await loop.run_in_executor(
                get_plugin_executor(), _base_is_resident, base, registry)
        return {"status": "detached", "adapter": req.adapter, "needs_reload": needs_reload}

    @app.post("/api/models/relocate", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_relocate(req: RelocateModelRequest, request: Request):
        """GUI form of `localm relocate MODEL NEW_PATH`: re-point a registered
        model's file after it was MOVED (the CLI's 'missing' row, mirrored here
        as `missing`/`last_path` on /api/models). Keeps the registration, and
        with it the aliases, source and sha256.

        require_fs_host, unconditionally: unlike a pull spec (which may name a
        remote HuggingFace repo), new_path here always names a location on the
        SERVER's own disk, the same capability class as /api/models/scan and
        /api/models/pull-comfy-source. Without that gate a models:write-only key
        (fs_access="none") could use the specific validation error below (does
        not exist / not a GGUF / not a HF dir) as an existence-and-validity
        oracle over the server's filesystem.

        new_path is rejected lexically (pathsafe.reject_unsafe_path_string)
        before anything touches the filesystem, and every filesystem call runs
        in the plugin executor, never on the event loop.
        """
        require_fs_host(request)
        _require_registered(req.model)
        from localm.plugins.gui.routes.admin import _network_drives_allowed
        try:
            pathsafe.reject_unsafe_path_string(
                req.new_path, reject_network_drives=not _network_drives_allowed())
        except ValueError as e:
            raise HTTPException(400, f"Invalid path: {e}") from e
        from localm.model_manager.registry import relocate_model, relocate_target

        def _do():
            p, reason = relocate_target(req.new_path)
            if p is None:
                return {"reason": reason}
            if not relocate_model(req.model, req.new_path):
                return {"reason": None}
            # .resolve() to match what relocate_model actually wrote (it resolves
            # its own, separately-computed `p` before saving), not the
            # pre-resolution path relocate_target returned above.
            return {"path": str(p.resolve())}

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_plugin_executor(), _do)
        if "path" not in result:
            raise HTTPException(400, result["reason"] or f"Could not relocate {req.model}")
        return {"status": "relocated", "model": req.model, "path": result["path"]}
