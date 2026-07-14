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
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.debuglog import logger
from localm.inference.http_server import (principal_id, require_scope,
                                          unload_all_models, unload_one_model)
import localm.inference.http_server as _hs
from localm.plugins.executor import get_plugin_executor
from localm.plugins.gui.web import (AliasRequest, ComfyPullRequest,
                                    LoadModelRequest, MediaPreflightRequest,
                                    PullRequest, PullTokenRedeemRequest,
                                    RemoveModelRequest, SetTypeRequest,
                                    UnloadModelRequest, consume_pull_grant)


def _require_registered(model: str, registry: dict | None = None) -> dict:
    """Raise 404 unless *model* is in the registry (GUI-2: the same guard was
    repeated verbatim across five route handlers). Returns the registry, so a
    caller that needs it afterward (model_alias) doesn't load it twice."""
    from localm.config import load_registry
    if registry is None:
        registry = load_registry()
    if model not in registry:
        raise HTTPException(404, f"Model not registered: {model}")
    return registry


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
        from localm.model_manager import _entry_path
        registry = load_registry()
        current = active_model()
        models = []
        for name, entry in sorted(registry.items()):
            epath = _entry_path(entry)
            if epath is None:
                # A hand-corrupted / cross-version registry entry (non-dict, or a
                # null / non-string / empty path). Skip it so one bad row never
                # 500s the whole Models page; the CLI `localm list` shows it as
                # [corrupt] and `localm rm <name>` drops it. Mirrors #562, which
                # routes every registry consumer through _entry_path.
                logger.debug("skipping malformed registry entry %r in /api/models", name)
                continue
            mtype = str(entry.get("model_type", "llm"))
            if type and mtype != type:
                continue
            path = Path(epath)
            size = None
            try:
                if path.is_file():
                    size = path.stat().st_size
            except OSError:
                pass
            engine = _hs._engines.get(name)
            models.append({
                "name": name,
                "source": str(entry.get("source", "")),
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
            res = await loop.run_in_executor(get_plugin_executor(), scan_comfy_models)
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
        _require_registered(req.model)
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
            _require_registered(req.model)
            return await unload_one_model(req.model)
        return await unload_all_models()

    @app.get("/api/vram-estimate", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def vram_estimate(model: str = "", n_ctx: int = 4096, n_gpu_layers: int = 99):
        """Approximate VRAM needed to load *model* (defaults to the active one)
        at the given context + GPU-offload, vs free/total VRAM. Powers the live
        readout under the Settings performance sliders. Always 'approximate'."""
        from localm.config import load_registry
        from localm.discover import vram_capacity
        from localm.model_meta import cached_n_layers
        from localm.model_manager import _entry_path
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        n_layers = None
        # _entry_path returns None for a malformed / corrupt entry (non-dict, or a
        # null / non-string / empty path). The route's own guard below is
        # except OSError, which would NOT catch the AttributeError / TypeError such
        # an entry raises; routing through _entry_path keeps a corrupt entry from
        # 500ing the VRAM readout (model_bytes stays 0 -> still a valid estimate).
        epath = _entry_path(load_registry().get(name))
        if epath is not None:
            try:
                p = Path(epath)
                if p.is_file():
                    model_bytes = p.stat().st_size
                    # A prior load caches the model's true layer count, so a
                    # partial-offload estimate (n_gpu_layers < 99) scales by real
                    # layers instead of the /99 sentinel fallback.
                    n_layers = cached_n_layers(str(p))
            except OSError:
                pass
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers, n_layers=n_layers)
        # vram_capacity() -> list_gpus() probes the GPU driver; keep it OFF the
        # event loop (it is safe-by-construction but still may take up to its
        # deadline on a wedged driver) so a stats read never stalls the WebUI.
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(get_plugin_executor(), vram_capacity)
        free, total = info.get("free"), info.get("total")
        fits = (est["needed"] <= free) if isinstance(free, int) else None
        return {"model": name, "model_bytes": model_bytes, **est,
                "free": free, "total": total, "fits": fits, "approximate": True}

    @app.get("/api/gpus", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_gpus():
        """Every GPU device visible right now, plus the currently configured
        main GPU index and multi-GPU split indices. Powers the Settings >
        Live tuning "Main GPU" selector and "Split across GPUs" checkboxes
        (both hidden/disabled when only one device is detected)."""
        from localm.config import load_config
        from localm.discover import list_gpus
        cfg = load_config()
        # list_gpus() probes the GPU driver; offload it so a wedged/slow driver
        # never blocks the event loop and freezes the whole WebUI.
        loop = asyncio.get_running_loop()
        gpus = await loop.run_in_executor(get_plugin_executor(), list_gpus)
        return {"gpus": gpus,
                "main_gpu_index": cfg.get("main_gpu_index"),
                "gpu_split_indices": cfg.get("gpu_split_indices")}

    # ----------------------- model ops + jobs --------------------- #

    @app.post("/api/models/pull-token/redeem",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull_token_redeem(req: PullTokenRedeemRequest, request: Request):
        """SEC-PULL-CONFIRM: redeem the single-use grant `localm gui --pull`
        minted for its own deep link (see mint_pull_grant/init.js). Only a
        genuine, unused, unexpired grant bound to this EXACT spec succeeds - a
        forged `?pull=` link cannot know the secret, so it 403s here and the
        frontend falls back to requiring an explicit human confirmation."""
        if not consume_pull_grant(request.app, req.spec.strip(), req.token):
            raise HTTPException(403, "Invalid or expired pull token")
        return {"ok": True}

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

    # --------------- ComfyUI missing-model pre-check + curated pull -------- #
    # Read-only pre-check the frontend calls BEFORE submitting a generate job
    # (see images.js/music.js/video.js): does the currently-configured workflow
    # reference any model file ComfyUI doesn't have, and if so, is there a
    # curated HuggingFace source to offer downloading it from? Does not modify
    # generate_image/generate_video/generate_music or their own preflight_models
    # call - this is purely additive.

    class _NullConsole:
        """A do-nothing stand-in for rich.console.Console, so a read-only
        pre-check (possibly polled repeatedly) never spams server-side console
        output the way an actual generation job's progress printing would."""
        def print(self, *a, **kw):
            pass

    def _build_check_workflow(kind: str, overrides: MediaPreflightRequest):
        """Load *kind*'s currently-configured workflow template and shape it
        with the same model-relevant overrides the generate form has pending,
        mirroring the load-template -> _build_*_workflow step generate_image /
        generate_video / generate_music do - minus actually submitting a job.
        input_image is always None: the image/video builders upload it to
        ComfyUI as a real network call, which a read-only check must never do.
        Returns the shaped workflow dict, or raises ValueError for an
        unknown *kind*."""
        if kind == "image":
            from localm.image_gen.comfy import _build_image_workflow
            from localm.image_gen.comfy import _workflow_path
            workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
            _build_image_workflow(
                workflow, prompt="", api_url="", guidance=None, negative_prompt=None,
                cfg=None, seed=0, clip_name1=overrides.clip_name1,
                clip_name2=overrides.clip_name2, lora_name=overrides.lora_name,
                lora_strength_model=1.0, lora_strength_clip=0.5, input_image=None,
                denoise=None, fast_dequant=True, con=_NullConsole())
            return workflow
        if kind == "video":
            from localm.video_gen.comfy import _build_video_workflow
            from localm.video_gen.comfy import _workflow_path
            workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
            _build_video_workflow(
                workflow, prompt="", negative_prompt=None, frames=1, fps=8,
                width=None, height=None, steps=1, cfg=None, seed=0,
                float_type=None, input_image=None, api_url="")
            return workflow
        if kind == "music":
            from localm.music_gen.comfy import _build_music_workflow
            from localm.music_gen.comfy import _workflow_path
            workflow = json.loads(_workflow_path().read_text(encoding="utf-8"))
            _build_music_workflow(
                workflow, tags="", lyrics_text="", duration_seconds=1.0, seed=0,
                steps=1, cfg=1.0, lyrics_strength=1.0,
                ckpt_name=overrides.ckpt_name, float_type=None)
            return workflow
        raise ValueError(f"Unknown media kind: {kind}")

    @app.post("/api/media/{kind}/preflight",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def media_preflight(kind: str, req: MediaPreflightRequest):
        if kind not in ("image", "video", "music"):
            raise HTTPException(404, f"Unknown media kind: {kind}")
        from localm.media.comfy_client import describe_missing_models
        from localm.media.managed_comfy import comfy_models_dest_dir, resolve_comfy_target
        from localm.model_manager.registry import resolve_comfy_model_source

        def _check():
            try:
                workflow = _build_check_workflow(kind, req)
            except Exception as e:
                logger.debug("preflight workflow build failed for %s: %s", kind, e)
                return []
            target = resolve_comfy_target()
            return describe_missing_models(workflow, target.api_url)

        loop = asyncio.get_running_loop()
        missing = await loop.run_in_executor(get_plugin_executor(), _check)

        results = []
        for slot in missing:
            source = resolve_comfy_model_source(slot.filename)
            entry = {
                "class_type": slot.class_type,
                "input_name": slot.input_name,
                "filename": slot.filename,
                "source": None,
                "dest_dir": None,
            }
            if source is not None:
                repo, file = source.spec.rsplit(":", 1)
                dest_dir = comfy_models_dest_dir(source.comfy_subfolder)
                entry["source"] = {
                    "repo": repo, "file": file,
                    "size_bytes": source.size_bytes, "model_type": source.model_type,
                }
                entry["dest_dir"] = str(dest_dir) if dest_dir is not None else None
            results.append(entry)
        return {"missing": results}

    @app.post("/api/models/pull-comfy-source",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull_comfy_source(req: ComfyPullRequest, request: Request):
        from localm.media.managed_comfy import comfy_models_dest_dir
        from localm.model_manager.registry import resolve_comfy_model_source
        source = resolve_comfy_model_source(req.filename.strip())
        if source is None:
            raise HTTPException(400, f"Not a curated download source: {req.filename}")
        dest_dir = comfy_models_dest_dir(source.comfy_subfolder)
        if dest_dir is None:
            raise HTTPException(
                400,
                "No known ComfyUI models folder to download into - set a "
                "ComfyUI working directory in Settings > Media, or enable the "
                "managed ComfyUI instance.")
        args = ["pull", "--type", source.model_type, "--comfy-dest-dir", str(dest_dir),
                "--no-register", "--", source.spec]
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {source.spec}", owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_remove(req: RemoveModelRequest, request: Request):
        _require_registered(req.model)
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"],
                             owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_alias(req: AliasRequest):
        registry = _require_registered(req.model)
        if req.alias in registry:
            raise HTTPException(409, f"Name already taken: {req.alias}")
        from localm.model_manager import alias_model
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                get_plugin_executor(), alias_model, req.model, req.alias)
        except Exception as e:
            raise HTTPException(400, f"Alias failed: {e}")
        return {"status": "aliased", "model": req.model, "alias": req.alias}

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

    async def _run_discover(fn):
        """Run *fn* off the event loop; map DiscoverError to its HTTP status
        (GUI-3: discover_search/discover_files shared this try/except)."""
        from localm.discover import DiscoverError
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(get_plugin_executor(), fn)
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))

    async def _vram_total():
        """Off-thread vram_capacity() plus its extracted 'total' bytes, both
        of which discover_search/discover_files feed into fit_label()."""
        from localm.discover import vram_capacity
        loop = asyncio.get_running_loop()
        vram = await loop.run_in_executor(get_plugin_executor(), vram_capacity)
        return vram, vram.get("total")

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20, formats: str = "gguf"):
        # `formats` is a CSV of {gguf, hf} from the search-page toggles. Empty
        # tokens are dropped; hf_search raises DiscoverError if none stay valid.
        # hf_backend_available lets the GUI warn (not block) that a transformers
        # model needs the .[gpu] extra to RUN, though it can still be downloaded.
        from localm.discover import fit_label, hf_backend_available, hf_search
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        results = await _run_discover(
            lambda: hf_search(q, limit=limit, formats=wanted))
        # Attach a VRAM fit badge to results that carry a size estimate (HF results
        # with safetensors param metadata). GGUF results are sized per-file in the
        # /discover/files expander instead. fit_label yields "" when VRAM is unknown;
        # a result with no size estimate keeps no fit (the GUI shows "size unknown").
        vram, total = await _vram_total()
        for r in results:
            if r.get("size_bytes"):
                r["fit"] = fit_label(r["size_bytes"], total)
        return {"query": q, "results": results, "vram": vram,
                "hf_backend_available": hf_backend_available()}

    @app.get("/api/discover/files", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_files(repo: str):
        from localm.discover import fit_label, hf_gguf_files
        files = await _run_discover(lambda: hf_gguf_files(repo))
        vram, total = await _vram_total()
        models = []
        mmprojs = []
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
            if "mmproj" in f["file"].lower():
                mmprojs.append(f)
            else:
                models.append(f)
        return {"repo": repo.strip().strip("/"), "files": models, "mmprojs": mmprojs, "vram": vram}
