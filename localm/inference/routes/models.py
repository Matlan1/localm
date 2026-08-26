# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model routes: list, registry detail, and explicit load/unload.

Reads the live engine and inference semaphore from the http_server module
globals, so a model swap that reassigns them is reflected here.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes
from localm.executor import get_plugin_executor


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/models",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def list_models():
        from localm.config import load_registry
        registry = load_registry()
        
        models = []
        now = int(time.time())
        all_model_names = set(registry.keys())
        if _hs._default_model_name and _hs._default_model_name not in all_model_names:
            all_model_names.add(_hs._default_model_name)
            
        for name in sorted(all_model_names):
            engine = _hs._engines.get(name)
            loaded = engine.loaded if engine is not None else False
            models.append({
                "id":       name,
                "object":   "model",
                "created":  now,
                "owned_by": "localm",
                "loaded":   loaded,
                # True for the active model: the default routing target.
                "active":   name == _hs._active_model_name,
            })
            
        return {
            "object": "list",
            "data": models,
        }

    @app.get("/v1/models/{model_id}",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def model_detail(model_id: str):
        """Registry metadata for one model: path, source, size, hash, aliases."""
        from localm.config import load_registry
        from localm.model_manager import _entry_path
        registry = load_registry()
        entry = registry.get(model_id)

        # If it is the default startup model but not in registry, provide a virtual entry
        if entry is None and _hs._default_model_name == model_id:
            # os.path.abspath, not resolve(): strips '..' against the cwd with no
            # filesystem call and no symlink traversal.
            _startup = getattr(_hs._engine, "model_path", "")
            entry = {"path": os.path.abspath(_startup) if _startup else "",
                     "source": "startup"}

        if entry is None:
            raise HTTPException(404, f"Model not registered: {model_id}")

        # A non-dict registry value is not a usable model record: 404, not a 500
        # from entry.get(...) below. A dict entry with a missing / null /
        # non-string / empty path is still rendered as a pathless model.
        if not isinstance(entry, dict):
            raise HTTPException(404, f"Model not registered: {model_id}")

        # _entry_path -> None for a missing / null / non-string / empty path; treat
        # that as pathless ("") so the `if path:` guard below scrubs it.
        epath = _entry_path(entry)
        path = epath if epath is not None else ""
        p = Path(path)
        size = None
        # Tri-state; None omits the key below. Stays None for a pathless entry and
        # for the virtual startup entry, which is absent from ``registry``.
        vision = None
        # Only stat/walk a real path: Path("") resolves to "." and would walk the
        # server's CWD.
        if path:
            # Runs off the event loop; the syscalls below are unbounded (rglob("*")
            # walks a whole directory tree, a UNC path blocks in the SMB redirector).
            def _measure() -> int | None:
                try:
                    if p.is_file():
                        return p.stat().st_size
                    if p.is_dir():
                        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                except OSError:
                    pass
                return None

            # Both blocking probes in one executor hop. model_vision_capability
            # stats the same path, may glob its folder for an mmproj sibling and
            # may read a small JSON, so it must not run on the loop either.
            def _probe() -> tuple:
                from localm.model_manager import model_vision_capability
                return _measure(), model_vision_capability(model_id, reg=registry)

            loop = asyncio.get_running_loop()
            size, vision = await loop.run_in_executor(get_plugin_executor(), _probe)
        aliases = sorted(
            n for n, e in registry.items()
            # Skip a non-dict sibling entry: its .get would AttributeError.
            if isinstance(e, dict) and e.get("path") == path and n != model_id
        )
        out = {
            "id": model_id,
            "object": "model",
            "owned_by": "localm",
            # Basename only, never the absolute path. Backslashes are normalised to
            # "/" first so a Windows-style path also splits correctly on POSIX.
            "path": Path(str(path).replace("\\", "/")).name if path else "",
            "source": entry.get("source", ""),
            "sha256": entry.get("sha256"),
            "size_bytes": size,
            "aliases": aliases,
            "active": _hs._active_model_name == model_id,
            "loaded": model_id in _hs._engines and _hs._engines[model_id].loaded,
            "model_type": entry.get("model_type", "llm"),
        }
        # The "llm" above is a default, not a recorded fact. This flag is emitted
        # only when the entry records no model_type.
        from localm.model_manager import has_recorded_model_type
        if not has_recorded_model_type(entry):
            out["model_type_recorded"] = False
        # true / false / key absent. Absent means the model's files could not be
        # inspected, which is not the same claim as false.
        if vision is not None:
            out["vision"] = vision
        return out

    @app.post("/v1/models/unload",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def unload_model(model: str | None = None):
        """
        Release model(s) from GPU/CPU memory.

        With no `model`, releases EVERY currently-loaded model (unchanged
        default behavior) - call this before a VRAM-intensive task (e.g.
        ComfyUI FLUX generation) so the GPU memory is fully available. With
        `model` set, releases only that one, leaving any other loaded models
        untouched. Either way, the next matching /v1/chat/completions call
        reloads lazily.
        """
        if model:
            from localm.config import load_registry
            if model not in load_registry() and model != _hs._default_model_name:
                raise HTTPException(404, f"Model not registered: {model}")
            return await _hs.unload_one_model(model)
        return await _hs.unload_all_models()

    @app.post("/v1/models/rename",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def rename_model(model: str, new_name: str):
        """
        Rename a registered model, re-keying a loaded engine in place.

        Unlike an alias, the old name stops working: this MOVES the
        registration (and best-effort migrates config/job/RAG references that
        named it). A model that is currently loaded keeps serving, under the
        new name.

        This lives on the always-present /v1 surface rather than only on the
        GUI's /api one because the caller that needs it most is `localm
        rename`, and a headless `localm serve` has no GUI routes. Renaming a
        model from a separate process CANNOT re-key this server's in-memory
        engine map, which would leave the engine orphaned under a name the
        registry no longer has - so the CLI asks the server to do the whole
        rename here instead, where the registry move and the re-key happen in
        one process with no window between them.
        """
        return await _hs.rename_registered_model(model, new_name)

    @app.post("/v1/models/load",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def load_model(model: str | None = None):
        """
        Explicitly reload a model into memory.

        Use this endpoint if you want to pre-warm the model before the first
        inference request. If no model is specified, pre-warms the model an
        unnamed request would currently resolve to.

        THE UNNAMED CASE GOES THROUGH ``_resolve_unnamed_model_name`` rather
        than re-deriving the chain here, and that is the fix rather than a
        tidy-up. The inline version read
        ``_active_model_name or _default_model_name`` and SKIPPED
        ``_last_active_model_name``, which is precisely where an unload parks
        the name of the model that was in use. So the media VRAM handover -
        which unloads chat, generates, then POSTs here with no model name -
        brought back the STARTUP model instead of the one the user had loaded,
        and they returned to chat talking to something they never chose.

        The shared helper already had it right and is used by ``GET /health``
        and ``get_engine``'s own fallback; this route was the odd one out, so
        the duplication WAS the defect.
        """
        name = model or _hs._resolve_unnamed_model_name()
        if not name:
            raise HTTPException(503, "No model specified or configured to load")
            
        already = False
        if name in _hs._engines and _hs._engines[name].loaded:
            already = True

        engine = await _hs.get_engine(name)
        status = "already_loaded" if already else "loaded"
        # Adds gpu_layers_offloaded/gpu_layers_total/degraded when the backend can
        # report them, so a partial GPU offload is distinguishable from a full one.
        return {"status": status, "model": engine.display_name,
                **_hs._gpu_placement_fields(engine)}

    @app.get("/v1/models/{model_id}/hold", include_in_schema=False,
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def model_file_hold(model_id: str):
        """Whether a LOADED engine in THIS server is holding the file that
        removing *model_id* would delete.

        Exists for a caller in ANOTHER PROCESS. ``localm rm`` and the MCP
        ``remove_model`` tool delete registry entries without ever contacting a
        running server, and the guard that answers this question reads this
        process's own engine map, so no amount of care in the other process can
        reach it. Asking over HTTP is the only way those callers can find out,
        and the answer has to come from here rather than be re-derived there: a
        second opinion about whether a file is in use is free to disagree with
        this one, and the disagreement is only ever discovered by a user whose
        model file is gone.

        ``held`` is a tri-state, and the third state is the load-bearing one.
        true/false are answers; ``reason`` non-null alongside ``held: true``
        means holding could not be RULED OUT rather than proven, which the
        guard treats as a refusal. A caller must not collapse those two into
        one message: "your model is in use" and "I could not establish that it
        is not" send a user looking in different places.

        Not in the OpenAPI schema: this is internal coordination between two
        localm processes, not part of the OpenAI-compatible surface.
        """
        from localm.config import load_registry
        registry = load_registry()
        if model_id not in registry:
            raise HTTPException(404, f"Model not registered: {model_id}")
        # Off the event loop: the guard resolves registry paths, and a UNC
        # entry blocks in the SMB redirector (same reason model_detail offloads
        # its stat/rglob).
        loop = asyncio.get_running_loop()
        hold = await loop.run_in_executor(
            get_plugin_executor(), _hs.loaded_engine_holding_model_file,
            model_id, registry)
        if hold is None:
            return {"held": False}
        return {"held": True, "key": hold.key, "reason": hold.reason}
