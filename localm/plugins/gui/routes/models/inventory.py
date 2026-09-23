# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes, inventory and runtime group: the registry listing, the
ComfyUI folder scan, plugin model roles, curated pull shortcuts, load/unload,
the embedder warm-up, the VRAM estimate and the GPU list."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.debuglog import logger
from localm.inference._threadpool_timeout import (ThreadCallTimeout,
                                                  run_in_threadpool_bounded)
from localm.inference.http_server import (principal_id, require_fs_host,
                                          require_scope, unload_all_models,
                                          unload_one_model)
import localm.inference.http_server as _hs
from localm.executor import get_plugin_executor
from localm.plugins.gui.routes.models._context import (ModelRouteContext,
                                                       _require_registered)
from localm.plugins.gui.web import (LoadModelRequest, ScanRequest,
                                    UnloadModelRequest)


def register(app: FastAPI, context: ModelRouteContext) -> None:
    active_model = context.active_model
    switch_model = context.switch_model
    jobs = context.jobs

    @app.get("/api/models", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_models(type: str = ""):
        # The annotation stays a builtin so FastAPI can resolve the forward-ref
        # this module's ``from __future__ import annotations`` produces. "" is the
        # no-filter sentinel, as on the sibling routes.
        from localm.config import load_registry
        from localm.model_manager import _entry_path
        from localm.model_manager import has_recorded_model_type as _has_recorded_model_type
        from localm.model_manager import model_vision_capability as _mvc
        from localm.model_manager import capabilities as _caps
        registry = load_registry()
        current = active_model()
        # Fetched once, off the event loop, before the row loop below: loaded_path()
        # blocks on embedder._LOCK, which get_embedder holds for the whole duration
        # of an IsolatedEmbedder load. The embedder's path cannot change mid-request,
        # so one fetch serves every row's comparison below.
        from localm.inference import embedder as _embedder_mod
        loop = asyncio.get_running_loop()
        emb_path = await loop.run_in_executor(get_plugin_executor(), _embedder_mod.loaded_path)
        rows = []
        for name, entry in sorted(registry.items()):
            epath = _entry_path(entry)
            if epath is None:
                # Malformed registry entry (non-dict, or a null / non-string / empty
                # path, or one carrying a '..' component). Skip it so one bad row
                # never 500s the whole Models page.
                logger.debug("skipping malformed registry entry %r in /api/models", name)
                continue
            mtype = str(entry.get("model_type", "llm"))
            if type and mtype != type:
                continue
            rows.append((name, entry, mtype, epath))

        # Stat and resolve every row's registry-supplied path in ONE executor hop,
        # off the event loop, then build the response on the loop from its results.
        # The embedder's own path is resolved in the same hop for the identity
        # comparison below.
        def _probe_rows() -> tuple:
            sizes: dict = {}
            mtimes: dict = {}
            # True when the entry has no file on disk - distinct from sizes[ep] is
            # None, which is also true for a healthy HF model directory. Populated
            # from the same stat() below, not a second filesystem call.
            missing: dict = {}
            resolved: dict = {}
            # Keyed by NAME, not by path: two aliases can share one path, but
            # model_vision_capability() is looked up per registered name.
            vision: dict = {}
            # Trained context window and tool-call support per NAME, as the
            # routing capability readers report them (None = not inspected).
            context_len: dict = {}
            tool_use: dict = {}
            # One projector listing per FOLDER for this request, not per row. Scoped
            # to this call, so a projector added to a folder shows up on the next
            # refresh.
            vision_dirs: dict = {}
            # The per-row resolve() has exactly one consumer, the embedder identity
            # comparison below, so it is skipped entirely when no embedder is loaded.
            emb_resolved = None
            if emb_path is not None:
                try:
                    emb_resolved = Path(emb_path).resolve()
                except (OSError, ValueError):
                    emb_resolved = None
            for _n, _e, _m, ep in rows:
                p = Path(ep)
                # In THIS hop, never on the loop: model_vision_capability() stats the
                # path, may glob the folder for an mmproj sibling, and may read a
                # small JSON. Not short-circuited on a failed stat: an entry with a
                # recorded, present projector answers True even when the model file
                # itself is unreachable.
                vision[_n] = _mvc(_n, reg=registry, dir_cache=vision_dirs)
                context_len[_n] = _caps.model_context_length(_n, reg=registry)
                tool_use[_n] = _caps.model_tool_use_capability(_n, reg=registry)
                # ONE stat() for both size and mtime. mtime is recorded for a
                # directory too (an HF model dir); size stays None for a directory.
                try:
                    st_res = p.stat()
                    sizes[ep] = st_res.st_size if stat.S_ISREG(st_res.st_mode) else None
                    mtimes[ep] = st_res.st_mtime
                    missing[ep] = False
                except (OSError, ValueError):
                    sizes[ep] = None
                    mtimes[ep] = None
                    missing[ep] = True
                if emb_resolved is None:
                    continue
                try:
                    resolved[ep] = p.resolve()
                except (OSError, ValueError):
                    resolved[ep] = None
            return (sizes, mtimes, missing, resolved, emb_resolved, vision,
                    context_len, tool_use)

        (sizes, mtimes, missing_flags, resolved_paths, emb_resolved, vision_caps,
         context_lens, tool_caps) = await loop.run_in_executor(
            get_plugin_executor(), _probe_rows)

        models = []
        for name, entry, mtype, epath in rows:
            size = sizes.get(epath)
            mtime = mtimes.get(epath)
            engine = _hs._engines.get(name)
            loaded = engine.loaded if engine is not None else False
            # A registered model can also be the shared EMBEDDING model, loaded via
            # get_embedder() - a lifecycle separate from _engines, so it never shows
            # up above. Recognised by resolved PATH, so this row's loaded status and
            # its per-row Unload control reflect a resident embedder.
            if not loaded and emb_resolved is not None:
                row_resolved = resolved_paths.get(epath)
                loaded = row_resolved is not None and emb_resolved == row_resolved
            row_out = {
                "name": name,
                "source": str(entry.get("source", "")),
                "size_bytes": size,
                "mtime": mtime,
                "active": name == current,
                # Independent of "active": a model can be resident in VRAM (loaded)
                # without being the one currently serving requests.
                "loaded": loaded,
                "model_type": mtype,
                # entry.get(...) with no default: a model registered before these
                # fields existed has neither key, and that reaches the client as
                # None (unknown) rather than as 0 experts / no architecture.
                "architecture": entry.get("architecture"),
                "expert_count": entry.get("expert_count"),
            }
            # Vision capability as true / false / KEY ABSENT. It is measured from the
            # model's own files on every request, so a model on an unmounted drive or
            # a dead UNC share yields no evidence and the key is omitted. A client
            # renders a pill for true, nothing for false, and nothing for unknown.
            _vis = vision_caps.get(name)
            if _vis is not None:
                row_out["vision"] = _vis
            # Same true / false / KEY ABSENT shape for the other two routing
            # capabilities: the trained context window and tool-call support.
            if context_lens.get(name) is not None:
                row_out["context_length"] = context_lens[name]
            if tool_caps.get(name) is not None:
                row_out["tool_use"] = tool_caps[name]
            # "model_type" above defaults a missing key to "llm" so a legacy entry
            # that predates the field stays selectable for the ?type=llm chat picker.
            # This flag marks that default as a guess rather than a recorded fact,
            # and is emitted only for an entry with no usable recorded type.
            if not _has_recorded_model_type(entry):
                row_out["model_type_recorded"] = False
            # Emitted only on a missing row. `last_path` is the registry's own path
            # string, which gives the GUI relocate control a starting point.
            if missing_flags.get(epath):
                row_out["missing"] = True
                row_out["last_path"] = epath
            models.append(row_out)
        out = {"models": models, "active": current}
        # Models this instance answers through a peer instance on the same
        # machine, keyed by name (see localm.peer_routing).
        from localm import peer_routing
        routes = peer_routing.list_routes()
        if routes:
            out["peer_routes"] = routes
        # The model an UNNAMED request would resolve to when none is currently
        # active: after an idle-unload the Engine stays in _engines for lazy reload
        # and _last_active_model_name records its name, so the next chat message
        # reloads it. Emitted only when there is no active model and a resumable one
        # exists, so a client can tell that state apart from "no model at all"
        # (both report active == "").
        if not current:
            resumable = _hs._resolve_unnamed_model_name()
            if resumable:
                out["resumable"] = resumable
        # The multi-GPU split distribution the ACTIVE model's load applied
        # (GgufBackend.applied_gpu_split - auto free-VRAM-proportional, pinned, or
        # the equal fallback), for the sidebar's loaded-model status. Absent when
        # there is no active engine, no split, or a backend that records none.
        active_engine = _hs._engines.get(current) if current else None
        split = getattr(getattr(active_engine, "_backend", None),
                        "applied_gpu_split", None)
        if split:
            out["active_gpu_split"] = split
        return out

    @app.post("/api/models/scan", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def gui_scan_models(request: Request, req: ScanRequest | None = None):
        """Scan for ComfyUI models and register what it finds. An explicit
        `workdir` is a one-off scan of an arbitrary folder for the guided
        Import-from-ComfyUI flow (never written back to config); with no
        `workdir` (a bodyless POST, or an explicit `{}`) it
        scans whatever `comfy_workdir` is configured. `dry_run` previews
        per-type counts and registers nothing, and stays synchronous (its
        directory walk has no honest total to report progress against - see
        scan_comfy_models's progress_cb docstring).

        A REAL scan (`dry_run` false or absent) runs as a background job
        instead, exactly like a model pull: this returns `{"job_id": ...}`
        immediately and the registration loop reports "registering model N of
        M" via Job.progress() as it goes. GET /api/jobs/{id}/events streams
        it; the final progress event before "end" carries added/skipped/method.

        BOTH forms require `require_fs_host`, called BEFORE either branch below
        so its 403 propagates as-is instead of becoming a generic 500. Either
        one walks a host directory and writes the resulting absolute paths into
        registry.json, a capability equivalent to the host file/folder browser
        (/api/fs/dirs), so a MODELS_WRITE-only key that lacks host filesystem
        access must not reach it. Scanning is authorised by host filesystem
        reach, never by where the folder name came from: `comfy_workdir` being
        admin_only is not what enforces this, this route's own require_fs_host
        is."""
        from localm.model_manager.scan import preview_comfy_models, scan_comfy_models
        import asyncio
        from functools import partial
        require_fs_host(request)
        workdir = req.workdir if req else None
        dry_run = bool(req and req.dry_run)
        if dry_run:
            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(
                    get_plugin_executor(), partial(preview_comfy_models, workdir=workdir))
                return {
                    "dry_run": True,
                    "method": res.method,
                    "counts": res.counts,
                    "already_registered": res.already_registered,
                    "total_new": sum(res.counts.values()),
                }
            except Exception as e:
                raise HTTPException(500, f"Scan failed: {e}")

        def _run_scan(job):
            job.push({"type": "line", "text": "Scanning ComfyUI model folders..."})

            def _cb(done, total, name):
                job.progress(phase="registering", done=done, total=total,
                            unit="models", name=name)

            try:
                res = scan_comfy_models(workdir=workdir, progress_cb=_cb)
            except Exception as e:
                job.push({"type": "line", "text": f"Scan failed: {e}"})
                return False
            job.push({"type": "line", "text":
                     f"Added {res.added} models, skipped {res.skipped} existing."})
            total = res.added + res.skipped
            job.progress(phase="done", done=total, total=total, unit="models",
                        added=res.added, skipped=res.skipped, method=res.method)
            return True

        job = jobs.start_fn("model-scan", _run_scan, owner=principal_id(request))
        return {"job_id": job.id}

    @app.get("/api/models/roles", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_model_roles(request: Request):
        manager = getattr(request.app.state, "plugin_manager", None)
        if manager is None:
            return {"roles": []}
        return {"roles": manager.get_all_model_roles()}

    @app.get("/api/models/shortcuts", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_model_shortcuts():
        """Curated `localm pull <alias>` aliases (MODEL_SHORTCUTS), for the
        Add-a-model dialog's shortcut picker. A fixed local list, not a HuggingFace
        query, so unlike /api/discover/search this needs no network call and works
        under net_mode=off - the pull path already resolves an alias typed into
        pull-spec (resolve_spec()); this just makes the alias keyspace discoverable
        instead of requiring the user to already know it from the CLI docs."""
        from localm.model_manager import MODEL_SHORTCUTS, _SHORTCUT_SIZES
        return {
            "shortcuts": [
                {"alias": alias, "spec": spec, "size": _SHORTCUT_SIZES.get(alias, "")}
                for alias, spec in MODEL_SHORTCUTS.items()
            ]
        }

    @app.post("/api/models/load", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def gui_load_model(req: LoadModelRequest):
        _require_registered(req.model)
        # Route every switch through the coordinator so a new selection preempts an
        # in-flight load instead of queuing behind it. The coordinator returns the
        # authoritative status: loaded, already_active, superseded (a newer
        # selection took over), or cancelled (aborted for any other reason) -
        # none of these is an error.
        try:
            # Omit force entirely when False (the overwhelming common case):
            # switch_model is a pluggable callback, and every test double
            # across the suite still implements the older single-arg
            # contract. Passing force=True is still forwarded when a caller
            # actually asks for it.
            result = await (
                switch_model(req.model, force=True) if req.force
                else switch_model(req.model))
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
            return await unload_one_model(req.model, force=req.force)
        return await unload_all_models(force=req.force)

    @app.post("/api/embedding/warmup",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def embedding_warmup(request: Request):
        """Load the shared embedder NOW, from an explicit user action, instead of
        the first real /v1/embeddings / memory-consolidate / RAG-recall call
        paying the cost. ``localm setup-embeddings`` only pre-fetches the file
        and never warms the singleton, and a restart resets it.

        Reports coarse PARENT-side stage events only; the isolated embedder
        child's own load/embed IPC protocol is untouched. Uses the same
        JobManager/SSE mechanism as model pull/remove."""
        from localm.inference.embedder import (PEEK_TIMEOUT_S, get_embedder,
                                               last_error, loaded_dim)
        # Off the event loop: loaded_dim() takes embedder._LOCK, and get_embedder
        # holds that same lock across an entire IsolatedEmbedder construction (a
        # process spawn plus a native model load, ceiling 300s).
        try:
            already = await run_in_threadpool_bounded(
                loaded_dim, timeout=PEEK_TIMEOUT_S)
        except ThreadCallTimeout:
            # The budget expiring means a load is holding _LOCK right now, so nothing
            # is loaded yet. Fall through and start a job that attaches to that
            # load's result.
            already = None
        if already is not None:
            def _already_warm(job):
                job.push({"type": "line",
                         "text": f"Already warm ({already}-dim)."})
                return True
            job = jobs.start_fn("embedding-warmup", _already_warm,
                                owner=principal_id(request))
            return {"job_id": job.id}

        def _warm(job):
            emb = get_embedder(
                on_progress=lambda msg: job.push({"type": "line", "text": msg}))
            if emb is None:
                why = last_error() or "no embedding model is configured"
                job.push({"type": "line",
                         "text": f"Could not warm up the embedder: {why}"})
                return False
            return True

        job = jobs.start_fn("embedding-warmup", _warm, owner=principal_id(request))
        return {"job_id": job.id}

    @app.get("/api/vram-estimate", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def vram_estimate(model: str = "", n_ctx: int = 4096, n_gpu_layers: int = 99):
        """Approximate VRAM needed to load *model* (defaults to the active one)
        at the given context + GPU-offload, vs free/total VRAM. Powers the live
        readout under the Settings performance sliders. Always 'approximate'."""
        from localm.config import load_config, load_registry
        from localm.discover import vram_capacity
        from localm.model_meta import cached_n_layers
        from localm.model_manager import _entry_path
        from localm.model_manager.gguf import (
            gguf_input_layer_bytes, gguf_kv_bytes_per_token,
            gguf_moe_pinned_expert_bytes)
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        n_layers = None
        kv_bytes_per_token = 0
        moe_pinned_bytes = 0
        input_layer_bytes = 0
        # n_cpu_moe has no GUI slider of its own (unlike n_ctx / n_gpu_layers, which
        # the caller sends as the sliders' live positions), so it is read from the
        # saved config.
        n_cpu_moe = int(load_config().get("n_cpu_moe") or 0)
        # _entry_path returns None for a malformed entry (non-dict, or a null /
        # non-string / empty path). The guard below is except OSError and would not
        # catch the AttributeError / TypeError such an entry raises; model_bytes
        # stays 0, which is still a valid estimate.
        entry = load_registry().get(name)
        epath = _entry_path(entry)
        # expert_count == 0 is a confirmed dense model, so the tensor-info re-parse
        # below is skipped: _apply_cpu_moe's own dense-model guard would find nothing
        # anyway. None (not backfilled, or an unreadable header) and any non-zero
        # count both still need the real read, which yields the byte size this
        # estimate needs.
        known_dense = isinstance(entry, dict) and entry.get("expert_count") == 0
        if epath is not None:
            # Off the event loop: the path comes out of registry.json, so stat() is a
            # blocking syscall on a value this handler did not choose. The GGUF header
            # read for the KV-shape probe below rides the same executor call.
            def _measure(ep: str):
                try:
                    p = Path(ep)
                    if p.is_file():
                        # A prior load caches the model's true layer count, so a
                        # partial-offload estimate (n_gpu_layers < 99) scales by
                        # real layers instead of the /99 sentinel fallback.
                        kv_bpt = 0
                        try:
                            kv_bpt = int(gguf_kv_bytes_per_token(p))
                        except Exception as exc:  # contracted not to raise - surface if it does
                            logger.debug(
                                "gguf KV-shape probe failed (%s) for %s; the VRAM "
                                "estimate falls back to the size-class heuristic",
                                type(exc).__name__, ep)
                        # Unconditional, unlike the MoE probe below: llama.cpp
                        # pins the input layer to the CPU for every load, dense
                        # or MoE, n_cpu_moe set or not.
                        input_bytes = 0
                        try:
                            input_bytes = int(gguf_input_layer_bytes(p) or 0)
                        except Exception as exc:  # contracted not to raise - surface if it does
                            logger.debug(
                                "gguf input-layer probe failed (%s) for %s; "
                                "the VRAM estimate charges the input layer",
                                type(exc).__name__, ep)
                        moe_pinned = 0
                        if n_cpu_moe > 0 and not known_dense:
                            try:
                                moe_pinned = int(gguf_moe_pinned_expert_bytes(
                                    p, n_cpu_moe) or 0)
                            except Exception as exc:  # contracted not to raise - surface if it does
                                logger.debug(
                                    "gguf MoE expert-byte probe failed (%s) for "
                                    "%s; the VRAM estimate charges the whole "
                                    "file (today's behavior)",
                                    type(exc).__name__, ep)
                        return (p.stat().st_size, cached_n_layers(str(p)), kv_bpt,
                                moe_pinned, input_bytes)
                except (OSError, ValueError):
                    pass
                return (model_bytes, n_layers, kv_bytes_per_token,
                        moe_pinned_bytes, input_layer_bytes)

            (model_bytes, n_layers, kv_bytes_per_token, moe_pinned_bytes,
             input_layer_bytes) = await asyncio.get_running_loop().run_in_executor(
                get_plugin_executor(), _measure, epath)
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers, n_layers=n_layers,
                            kv_bytes_per_token=kv_bytes_per_token,
                            moe_pinned_bytes=moe_pinned_bytes,
                            input_layer_bytes=input_layer_bytes)
        # vram_capacity() -> list_gpus() probes the GPU driver; keep it off the event
        # loop so a stats read never stalls the WebUI. return_status=True so a stale
        # (timed-out) or process-blind free reading is not weighed as current. When
        # the reading is untrusted, free is withheld (fits -> None) and the UI shows
        # "free VRAM unknown".
        from localm.sysstats import _vram_reading_trusted
        loop = asyncio.get_running_loop()
        info, status = await loop.run_in_executor(
            get_plugin_executor(), lambda: vram_capacity(return_status=True))
        total = info.get("total")
        free = info.get("free") if _vram_reading_trusted(info, status) else None
        fits = (est["needed"] <= free) if isinstance(free, int) else None
        return {"model": name, "model_bytes": model_bytes, **est,
                "free": free, "total": total, "fits": fits, "approximate": True}

    @app.get("/api/gpus", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def gui_gpus():
        """Every GPU device visible right now, plus the currently configured
        main GPU index and multi-GPU split indices. Powers the Settings >
        Live tuning "Main GPU" selector and "Split across GPUs" checkboxes
        (both hidden/disabled when only one device is detected).

        ``probe_status`` tells the consumer whether ``gpus`` is a FRESH reading
        (``ok`` - an empty list then genuinely means no GPU) or an inconclusive
        one (``timeout``/``busy`` - the driver was wedged or contended, and
        ``gpus`` is a frozen last-known-good or []), so a timed-out probe is
        distinguishable from a GPU-less box.

        ``index_space`` (present only as ``"native"``) says the ``gpus``
        indices are the ones a MODEL LOAD consumes rather than list_gpus()'s
        torch/nvidia-smi numbering - see the opaque-index-space branch below
        (vulkan or sycl); the client labels the numbering accordingly. That
        is llama.cpp's own device list,
        NOT the raw ggml registry order: integrated GPUs and accelerators are
        dropped and the rest renumbered before they get here
        (discover._llama_visible_devices). These numbers are written straight
        into main_gpu_index / gpu_split_indices."""
        from localm.config import load_config
        from localm.discover import GPU_PROBE_OK
        cfg = load_config()
        # The reads below probe the GPU driver (or spawn the probe daemon); offload
        # them so a wedged or slow driver never blocks the event loop.
        # wait_for_inflight joins an in-flight probe instead of bouncing off it with
        # an instant BUSY + [].
        loop = asyncio.get_running_loop()

        def _read_devices():
            # Native-first on a build whose GPU index space is opaque to list_gpus()
            # (vulkan or sycl - see discover._native_gpu_index_space_is_opaque): these
            # selectors write indices the LOADER consumes, and on such a build those
            # live in the native backend's own index space, which list_gpus()
            # (torch.cuda / nvidia-smi) can neither see nor order. The native registry
            # is read via the crash-isolated probe daemon, and a completed read is a
            # conclusive probe (GPU_PROBE_OK). Falls back to list_gpus(), with no
            # index_space claim, when the daemon or registry cannot answer (None).
            from localm.discover import (
                _native_gpu_index_space_is_opaque, list_gpus, native_gpu_devices)
            if _native_gpu_index_space_is_opaque():
                native = native_gpu_devices()
                if native is not None:
                    return native, GPU_PROBE_OK, "native"
            gpus, probe_status = list_gpus(return_status=True, wait_for_inflight=True)
            return gpus, probe_status, None

        gpus, probe_status, index_space = await loop.run_in_executor(
            get_plugin_executor(), _read_devices)
        out = {"gpus": gpus,
               "probe_status": probe_status,
               "main_gpu_index": cfg.get("main_gpu_index"),
               "gpu_split_indices": cfg.get("gpu_split_indices")}
        if index_space:
            out["index_space"] = index_space
        return out
