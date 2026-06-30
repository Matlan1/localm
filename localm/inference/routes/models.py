# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model routes: list, registry detail, and explicit load/unload.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
and inference semaphore from the http_server module globals (a model swap that
reassigns them is reflected here).
"""

from __future__ import annotations

import asyncio
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
        # The GUI can run with no engine yet (fresh install, empty registry -
        # the user adds a model from the Models page). Report an empty list.
        if _hs._engine is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [
                {
                    "id":       _hs._engine.display_name,
                    "object":   "model",
                    "created":  int(time.time()),
                    "owned_by": "localm",
                    "loaded":   _hs._engine.loaded,
                }
            ],
        }

    @app.get("/v1/models/{model_id}",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def model_detail(model_id: str):
        """Registry metadata for one model: path, source, size, hash, aliases."""
        from localm.config import load_registry
        registry = load_registry()
        entry = registry.get(model_id)
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
            "active": _hs._engine is not None and _hs._engine.display_name == model_id,
            "loaded": _hs._engine is not None
                      and _hs._engine.display_name == model_id and _hs._engine.loaded,
        }

    @app.post("/v1/models/unload",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def unload_model():
        """
        Release the model from GPU/CPU memory.

        Call this before starting a VRAM-intensive task (e.g. ComfyUI FLUX
        generation) so the GPU memory is fully available.  The next call to
        /v1/chat/completions will reload the model automatically.
        """
        if _hs._engine is None:
            raise HTTPException(503, "No engine initialised")
        if not _hs._engine.loaded:
            return {"status": "already_unloaded", "model": _hs._engine.display_name}
        loop = asyncio.get_running_loop()
        # Under the inference semaphore: freeing the native context while a
        # generation is mid-decode crashes the GPU driver (access violation
        # in the HIP runtime). Unload must wait its turn.
        from localm.discover import vram_info
        from localm.vram import wait_for_vram_release

        def _free():
            return vram_info().get("free")

        async with _hs._inference_sem:
            before = _free()
            await loop.run_in_executor(None, _hs._engine.unload)
            # The native context free is deferred: do NOT return until VRAM has
            # actually dropped, or a caller (e.g. ComfyUI media gen) loads its
            # model on top of the not-yet-freed weights, exceeds total VRAM and
            # hangs the GPU driver (TDR). Best-effort: a no-op when VRAM is
            # unmeasurable (before is None) or already freed.
            released, after = await loop.run_in_executor(
                None, lambda: wait_for_vram_release(_free, before_bytes=before))
        result = {"status": "unloaded", "model": _hs._engine.display_name}
        if before is not None:
            result.update(vram_freed=released,
                          vram_before_bytes=before, vram_after_bytes=after)
        return result

    @app.post("/v1/models/load",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def load_model():
        """
        Explicitly reload the model into memory.

        Normally you don't need this - /v1/chat/completions reloads
        automatically if the model was unloaded.  Use this endpoint if you
        want to pre-warm the model before the first inference request.
        """
        if _hs._engine is None:
            raise HTTPException(503, "No engine initialised")
        if _hs._engine.loaded:
            return {"status": "already_loaded", "model": _hs._engine.display_name}
        loop = asyncio.get_running_loop()
        async with _hs._inference_sem:
            await loop.run_in_executor(None, _hs._engine.load)
        return {"status": "loaded", "model": _hs._engine.display_name}
