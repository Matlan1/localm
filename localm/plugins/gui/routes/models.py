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
from localm.inference.http_server import (principal_id, require_scope,
                                          unload_all_models, unload_one_model)
import localm.inference.http_server as _hs
from localm.plugins.gui.web import (AliasRequest, LoadModelRequest, PullRequest,
                                    RemoveModelRequest, SetTypeRequest,
                                    UnloadModelRequest)


def register(app: FastAPI, ctx) -> None:
    active_model = ctx.active_model
    switch_model = ctx.switch_model
    jobs = ctx.jobs

    # -------------------------- models ---------------------------- #

    @app.get("/api/models", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_models(type: str = ""):
        # Plain ``str = ""`` (not ``Optional[str]``) on purpose: this module uses
        # ``from __future__ import annotations``, so an annotation like
        # ``Optional[str]`` is a string forward-ref FastAPI must resolve against
        # this module's globals at route-build time. If ``Optional`` is ever not
        # imported here, that resolution fails silently and the field gets a mock
        # validator that only raises "is not fully defined" on the FIRST request
        # (issue #435). A builtin like ``str`` always resolves, and "" is the
        # same "no filter" sentinel the sibling routes use (q="", model="").
        from localm.config import load_registry
        registry = load_registry()
        current = active_model()
        models = []
        for name, entry in sorted(registry.items()):
            mtype = entry.get("model_type", "llm")
            if type and mtype != type:
                continue
            path = Path(entry.get("path", ""))
            size = None
            try:
                if path.is_file():
                    size = path.stat().st_size
            except OSError:
                pass
            engine = _hs._engines.get(name)
            models.append({
                "name": name,
                "source": entry.get("source", ""),
                "size_bytes": size,
                "active": name == current,
                # Independent of "active": a model can be resident in VRAM
                # (loaded) without being the one currently serving requests -
                # surfaced so the Models page can offer a per-row Unload
                # action on ANY loaded model, not just the active one.
                "loaded": engine.loaded if engine is not None else False,
                "model_type": mtype,
            })
        return {"models": models, "active": current}

    @app.post("/api/models/scan", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def gui_scan_models():
        from localm.model_manager.scan import scan_comfy_models
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(None, scan_comfy_models)
            return {
                "added": res.added,
                "skipped": res.skipped,
                "method": res.method,
            }
        except Exception as e:
            raise HTTPException(500, f"Scan failed: {e}")

    @app.get("/api/models/roles", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_model_roles(request: Request):
        manager = getattr(request.app.state, "plugin_manager", None)
        if manager is None:
            return {"roles": []}
        return {"roles": manager.get_all_model_roles()}

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

    @app.post("/api/models/unload", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def gui_unload_model(req: UnloadModelRequest):
        """Release model(s) from GPU/CPU memory. With no `model` (or an empty
        POST body), unloads everything - the GUI's global "Unload all"
        button. With `model` set, unloads only that one, leaving any other
        loaded models untouched - the GUI's per-row Unload button."""
        if req.model:
            from localm.config import load_registry
            if req.model not in load_registry():
                raise HTTPException(404, f"Model not registered: {req.model}")
            return await unload_one_model(req.model)
        return await unload_all_models()

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

    @app.get("/api/gpus", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_gpus():
        """Every GPU device visible right now, plus the currently configured
        main GPU index. Powers the Settings > Live tuning "Main GPU" selector
        (hidden/disabled when only one device is detected)."""
        from localm.config import load_config
        from localm.discover import list_gpus
        return {"gpus": list_gpus(),
                "main_gpu_index": load_config().get("main_gpu_index")}

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
        if req.store:
            if req.store not in ("copy", "move"):
                raise HTTPException(400, "store must be 'copy' or 'move'")
            args += ["--store", req.store]
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

    @app.post("/api/models/type", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_set_type(req: SetTypeRequest):
        """Change a registered model's type (the one-click set-type control). A
        type='unknown' model is not auto-loaded as chat but stays runnable by name;
        this corrects a mis-detected or bulk-imported model's type."""
        from localm.config import load_registry
        from localm.model_manager import MODEL_TYPES, set_model_type
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model_type not in MODEL_TYPES:
            raise HTTPException(
                400, f"Invalid type: {req.model_type}. "
                     f"One of: {', '.join(sorted(MODEL_TYPES))}")
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, set_model_type, req.model, req.model_type)
        if not ok:
            raise HTTPException(400, f"Could not set type for {req.model}")
        return {"status": "typed", "model": req.model, "model_type": req.model_type}

    # ------------------------ model discovery --------------------- #
    # Search HuggingFace for GGUF and/or HF (transformers) models and show
    # per-quant "fits your VRAM" badges for GGUF files. User-initiated prelude
    # to a pull (docs/network.md); net_mode=off blocks it like everything else.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF unreachable
        return 422              # bad repo / no GGUF files / bad format token

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20, formats: str = "gguf"):
        # `formats` is a CSV of {gguf, hf} from the search-page toggles. Empty
        # tokens are dropped; hf_search raises DiscoverError if none stay valid.
        # hf_backend_available lets the GUI warn (not block) that a transformers
        # model needs the .[gpu] extra to RUN, though it can still be downloaded.
        from localm.discover import (DiscoverError, fit_label,
                                     hf_backend_available, hf_search, vram_info)
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: hf_search(q, limit=limit, formats=wanted))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        vram = vram_info()
        # Attach a VRAM fit badge to results that carry a size estimate (HF results
        # with safetensors param metadata). GGUF results are sized per-file in the
        # /discover/files expander instead. fit_label yields "" when VRAM is unknown;
        # a result with no size estimate keeps no fit (the GUI shows "size unknown").
        total = vram.get("total")
        for r in results:
            if r.get("size_bytes"):
                r["fit"] = fit_label(r["size_bytes"], total)
        return {"query": q, "results": results, "vram": vram,
                "hf_backend_available": hf_backend_available()}

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
