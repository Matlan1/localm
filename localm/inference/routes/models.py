# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model routes: list, registry detail, and explicit load/unload.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
and inference semaphore from the http_server module globals (a model swap that
reassigns them is reflected here).
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes


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
        registry = load_registry()
        entry = registry.get(model_id)
        
        # If it is the default startup model but not in registry, provide a virtual entry
        if entry is None and _hs._default_model_name == model_id:
            entry = {"path": getattr(_hs._engine, "model_path", ""), "source": "startup"}
            
        if entry is None:
            raise HTTPException(404, f"Model not registered: {model_id}")
            
        path = entry.get("path", "")
        p = Path(path)
        size = None
        try:
            if p.is_file():
                size = p.stat().st_size
            elif p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except OSError:
            pass
        aliases = sorted(
            n for n, e in registry.items()
            if e.get("path") == path and n != model_id
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
        return {"status": status, "model": engine.display_name}
