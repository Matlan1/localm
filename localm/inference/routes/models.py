# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model routes: list, registry detail, and explicit load/unload.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
and inference semaphore from the http_server module globals (a model swap that
reassigns them is reflected here).
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
                # Which model is the ACTIVE one (the default routing target and
                # the one an attaching client should report as loaded). Without
                # this a consumer cannot tell the active model from the merely
                # first-listed one (AUDIT-HIGH-4).
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
            # abspath, because the startup path is whatever the operator typed on
            # the command line and `localm serve ../models/foo.gguf` is a normal
            # invocation. _entry_path (correctly) reads a stored path carrying a
            # '..' component as malformed, which would blank this virtual entry's
            # path and size for a perfectly good running model. Normalising here
            # keeps that rule strict for the REGISTRY - where a '..' really does
            # mean the file was hand-edited or planted - without punishing a
            # relative CLI argument. os.path.abspath, not resolve(): it strips
            # '..' against the cwd with no filesystem call and no symlink
            # traversal, so this stays off the syscall path entirely.
            _startup = getattr(_hs._engine, "model_path", "")
            entry = {"path": os.path.abspath(_startup) if _startup else "",
                     "source": "startup"}

        if entry is None:
            raise HTTPException(404, f"Model not registered: {model_id}")

        # A non-dict registry value (a hand-edited / cross-version corrupt entry) is
        # not a usable model record: 404 rather than a 500 from entry.get(...) below.
        # A dict entry with a missing / null / non-string / empty path is still
        # rendered as a pathless model (see the empty-path guard below and
        # test_model_detail_empty_path_does_not_walk_cwd), never a Path(None)/Path(int)
        # crash. Mirrors #562's _entry_path registry hardening.
        if not isinstance(entry, dict):
            raise HTTPException(404, f"Model not registered: {model_id}")

        # _entry_path -> None for a missing / null / non-string / empty path; treat
        # that as pathless ("") so the `if path:` guard below scrubs it (exactly like
        # the virtual startup entry) instead of Path(None)/Path(int) raising a 500.
        epath = _entry_path(entry)
        path = epath if epath is not None else ""
        p = Path(path)
        size = None
        # Only stat/walk a REAL path. An empty path (a virtual startup entry, or a
        # malformed registry row) makes Path("") resolve to "." -> the server's CWD,
        # whose is_dir() is True and rglob("*") walks the entire working tree: an
        # aggregate-size info leak plus a filesystem-walk DoS. The path field below
        # is already scrubbed to "" for an empty path; guard the size branch to match.
        if path:
            # Off the event loop. This is an `async def` and the path comes out of
            # registry.json, not from this handler, so the syscalls below are
            # unbounded in both directions: rglob("*") walks an entire directory
            # tree, and a UNC path blocks in the Windows SMB redirector (minutes
            # against an unroutable host, plus an outbound authentication attempt
            # from the server process when it IS reachable). Inline, either one
            # stalls every request this server is serving, not just this one.
            def _measure() -> int | None:
                try:
                    if p.is_file():
                        return p.stat().st_size
                    if p.is_dir():
                        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                except OSError:
                    pass
                return None

            loop = asyncio.get_running_loop()
            size = await loop.run_in_executor(get_plugin_executor(), _measure)
        aliases = sorted(
            n for n, e in registry.items()
            # Skip a malformed sibling: a non-dict entry's .get would AttributeError,
            # so one corrupt sibling must not crash a healthy model's detail lookup.
            if isinstance(e, dict) and e.get("path") == path and n != model_id
        )
        return {
            "id": model_id,
            "object": "model",
            "owned_by": "localm",
            # Basename only; never leak the absolute path. Normalise backslashes
            # to "/" first so the guarantee holds for a Windows-style path even on
            # POSIX, where Path(...).name would not split on "\\" and would leak
            # the whole directory (registry entries can carry either separator).
            "path": Path(str(path).replace("\\", "/")).name if path else "",
            "source": entry.get("source", ""),
            "sha256": entry.get("sha256"),
            "size_bytes": size,
            "aliases": aliases,
            "active": _hs._active_model_name == model_id,
            "loaded": model_id in _hs._engines and _hs._engines[model_id].loaded,
            "model_type": entry.get("model_type", "llm"),
        }

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
        inference request. If no model is specified, pre-warms the default model.
        """
        name = model or _hs._active_model_name or _hs._default_model_name
        if not name:
            raise HTTPException(503, "No model specified or configured to load")
            
        already = False
        if name in _hs._engines and _hs._engines[name].loaded:
            already = True

        engine = await _hs.get_engine(name)
        status = "already_loaded" if already else "loaded"
        # gpu_layers_offloaded/gpu_layers_total/degraded, when the backend can
        # report them: a model too big to fully fit VRAM still loads (the
        # backend's own sizing deliberately defers to a partial/zero GPU
        # offload rather than refusing - see llamacpp/_sizing.py), and without
        # this a caller cannot tell that "loaded" apart from a full GPU load
        # (AGENTS.md rule 5 - a silent downgrade must not report unqualified
        # success).
        return {"status": status, "model": engine.display_name,
                **_hs._gpu_placement_fields(engine)}
