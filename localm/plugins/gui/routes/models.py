# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes: registry list/load, VRAM estimate, pull/remove/alias, and
HuggingFace discovery.

Extracted verbatim from attach_gui(); behavior unchanged. The active-model
accessor, the model-switch callable, and the background job manager are unpacked
from the register ``ctx`` into ``active_model`` / ``switch_model`` / ``jobs`` once
at the top of register(), so each handler body is identical to the original.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference.http_server import principal_id, require_scope
from localm.plugins.gui.web import (AliasRequest, LoadModelRequest, PullRequest,
                                    RemoveModelRequest)


def register(app: FastAPI, ctx) -> None:
    active_model = ctx.active_model
    switch_model = ctx.switch_model
    jobs = ctx.jobs

    # -------------------------- models ---------------------------- #

    @app.get("/api/models", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_models():
        from localm.config import load_registry
        registry = load_registry()
        current = active_model()
        models = []
        for name, entry in sorted(registry.items()):
            path = Path(entry.get("path", ""))
            size = None
            try:
                if path.is_file():
                    size = path.stat().st_size
            except OSError:
                pass
            models.append({
                "name": name,
                "source": entry.get("source", ""),
                "size_bytes": size,
                "active": name == current,
            })
        return {"models": models, "active": current}

    @app.post("/api/models/load", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def gui_load_model(req: LoadModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        # Route every switch through the coordinator (switch_engine) so a new
        # selection PREEMPTS an in-flight load instead of queuing behind it. The
        # coordinator returns the authoritative status: loaded, already_active, or
        # superseded (a newer selection took over - not an error). The old early
        # "== active_model()" shortcut is dropped so a re-select mid-switch cannot
        # report already_active for a model that is actually being replaced.
        try:
            result = await switch_model(req.model)
        except Exception as e:
            raise HTTPException(500, f"Failed to load {req.model}: {e}")
        # A switch_model that does not report a status (a minimal/legacy callable)
        # still counts as a successful load of the requested model.
        return result if result is not None else {"status": "loaded", "model": req.model}

    @app.get("/api/vram-estimate", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def vram_estimate(model: str = "", n_ctx: int = 4096, n_gpu_layers: int = 99):
        """Approximate VRAM needed to load *model* (defaults to the active one)
        at the given context + GPU-offload, vs free/total VRAM. Powers the live
        readout under the Settings performance sliders. Always 'approximate'."""
        from localm.config import load_registry
        from localm.discover import vram_info
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        entry = load_registry().get(name)
        if entry:
            try:
                p = Path(entry.get("path", ""))
                if p.is_file():
                    model_bytes = p.stat().st_size
            except OSError:
                pass
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers)
        info = vram_info()
        free, total = info.get("free"), info.get("total")
        fits = (est["needed"] <= free) if isinstance(free, int) else None
        return {"model": name, "model_bytes": model_bytes, **est,
                "free": free, "total": total, "fits": fits, "approximate": True}

    # ----------------------- model ops + jobs --------------------- #

    @app.post("/api/models/pull", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull(req: PullRequest, request: Request):
        spec = req.spec.strip()
        if not spec or set(spec) <= {"-"}:
            raise HTTPException(
                400,
                "Enter a model spec: owner/repo, owner/repo:file.gguf, "
                "or an https URL.",
            )
        # Pass the spec after "--" so a value like "-h" or "--help" is treated as
        # the model argument, not parsed by the CLI as an option/help flag.
        args = ["pull"]
        if req.name:
            args += ["--name", req.name]
        if req.mmproj:
            args += ["--mmproj", req.mmproj]
        args += ["--", spec]
        # Stream structured download progress; suppress huggingface_hub's own
        # tqdm bars (their \r output doesn't line-stream cleanly).
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {spec}", owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_remove(req: RemoveModelRequest, request: Request):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"],
                             owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_alias(req: AliasRequest):
        from localm.config import load_registry
        registry = load_registry()
        if req.model not in registry:
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.alias in registry:
            raise HTTPException(409, f"Name already taken: {req.alias}")
        from localm.model_manager import alias_model
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, alias_model, req.model, req.alias)
        except Exception as e:
            raise HTTPException(400, f"Alias failed: {e}")
        return {"status": "aliased", "model": req.model, "alias": req.alias}

    # ------------------------ model discovery --------------------- #
    # Search HuggingFace for GGUF models and show per-quant "fits your
    # VRAM" badges. User-initiated prelude to a pull (docs/network.md);
    # net_mode=off blocks it like everything else.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF unreachable
        return 422              # bad repo / no GGUF files

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20):
        from localm.discover import DiscoverError, hf_search, vram_info
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: hf_search(q, limit=limit))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        return {"query": q, "results": results, "vram": vram_info()}

    @app.get("/api/discover/files", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_files(repo: str):
        from localm.discover import (DiscoverError, fit_label, hf_gguf_files,
                                     vram_info)
        loop = asyncio.get_running_loop()
        try:
            files = await loop.run_in_executor(
                None, lambda: hf_gguf_files(repo))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        vram = vram_info()
        total = vram.get("total")
        models = []
        mmprojs = []
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
            if "mmproj" in f["file"].lower():
                mmprojs.append(f)
            else:
                models.append(f)
        return {"repo": repo.strip().strip("/"), "files": models, "mmprojs": mmprojs, "vram": vram}
