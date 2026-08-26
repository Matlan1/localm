# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes: registry list/load, VRAM estimate, pull/remove/alias, and
HuggingFace discovery.

The active-model accessor, the model-switch callable, and the background job
manager are unpacked from the register ``ctx`` into ``active_model`` /
``switch_model`` / ``jobs`` once at the top of register().
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import pathsafe
from localm import scopes
from localm.debuglog import logger
from localm.inference._threadpool_timeout import (ThreadCallTimeout,
                                                  run_in_threadpool_bounded)
from localm.inference.http_server import (principal_id, require_fs_host,
                                          require_scope, unload_all_models,
                                          unload_one_model)
import localm.inference.http_server as _hs
from localm.executor import get_plugin_executor
from localm.plugins.gui.web import (AliasRequest, ComfyPullRequest,
                                    LoadModelRequest, MediaPreflightRequest,
                                    PullRequest, PullTokenRedeemRequest,
                                    RelocateModelRequest, RemoveModelRequest,
                                    RenameModelRequest, ScanRequest,
                                    SetTypeRequest, UnloadModelRequest,
                                    consume_pull_grant)


def _spec_names_a_host_path(spec: str) -> bool:
    """True when *spec* TEXTUALLY names a path on the server's filesystem rather
    than a remote HuggingFace/URL spec. Makes no filesystem call.

    `localm pull` registers an existing local path IN PLACE rather than
    downloading it (model_manager/pull.py's is_local_path branch -> add_local
    with store=None), so naming one through POST /api/models/pull writes a
    caller-chosen absolute path into registry.json. That is host filesystem
    reach and belongs behind require_fs_host, not behind MODELS_WRITE alone.

    Textual and existence-INDEPENDENT. It cannot stall (a UNC spec is classified
    without the stat that would block in the SMB redirector - see
    pathsafe.is_unc_or_device_path), it cannot become an existence oracle (the
    authorisation answer is identical whether or not the file is there), and it
    has no TOCTOU. It is therefore BROADER than pull.py's own is_local_path,
    which also requires the path to exist: a non-existent absolute path is
    refused here even though pull.py would treat it as a remote spec.

    RESIDUAL: a relative spec with no ".." component that names an existing FILE
    is still registered in place without this gate firing. Its reach is bounded
    by the server's working directory, and closing it would need a filesystem
    answer."""
    s = spec.strip()
    if not s:
        return False
    if pathsafe.is_unc_or_device_path(s):
        return True
    if s.startswith("~"):
        return True
    # Judge under both Windows and POSIX flavours regardless of host OS: a
    # drive-qualified spec, and the drive-relative form (a drive letter and colon
    # with no separator), are host paths.
    for flavour in (PureWindowsPath, PurePosixPath):
        pure = flavour(s)
        if pure.is_absolute() or pure.drive or pure.root:
            return True
        # A '..' component makes a relative spec a host path.
        if any(part == ".." for part in pure.parts):
            return True
    return False


def _require_registered(model: str, registry: dict | None = None) -> dict:
    """Raise 404 unless *model* is in the registry. Returns the registry, so a
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
        # The annotation stays a builtin so FastAPI can resolve the forward-ref
        # this module's ``from __future__ import annotations`` produces. "" is the
        # no-filter sentinel, as on the sibling routes.
        from localm.config import load_registry
        from localm.model_manager import _entry_path
        from localm.model_manager import has_recorded_model_type as _has_recorded_model_type
        from localm.model_manager import model_vision_capability as _mvc
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
            return sizes, mtimes, missing, resolved, emb_resolved, vision

        sizes, mtimes, missing_flags, resolved_paths, emb_resolved, vision_caps = \
            await loop.run_in_executor(get_plugin_executor(), _probe_rows)

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
        # authoritative status: loaded, already_active, or superseded (a newer
        # selection took over, which is not an error).
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
            gguf_kv_bytes_per_token, gguf_moe_pinned_expert_bytes)
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        n_layers = None
        kv_bytes_per_token = 0
        moe_pinned_bytes = 0
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
                                moe_pinned)
                except (OSError, ValueError):
                    pass
                return model_bytes, n_layers, kv_bytes_per_token, moe_pinned_bytes

            (model_bytes, n_layers, kv_bytes_per_token,
             moe_pinned_bytes) = await asyncio.get_running_loop().run_in_executor(
                get_plugin_executor(), _measure, epath)
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers, n_layers=n_layers,
                            kv_bytes_per_token=kv_bytes_per_token,
                            moe_pinned_bytes=moe_pinned_bytes)
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
        torch/nvidia-smi numbering - see the vulkan branch below; the client
        labels the numbering accordingly. That is llama.cpp's own device list,
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
            # Native-first on the vulkan build: these selectors write indices the
            # LOADER consumes, and on that build those live in ggml-vulkan's own index
            # space, which list_gpus() (torch.cuda / nvidia-smi) can neither see nor
            # order. The native registry is read via the crash-isolated probe daemon,
            # and a completed read is a conclusive probe (GPU_PROBE_OK). Falls back to
            # list_gpus(), with no index_space claim, when the daemon or registry
            # cannot answer (None).
            from localm.discover import _native_backend_has_vulkan, list_gpus, native_gpu_devices
            if _native_backend_has_vulkan():
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

    # ----------------------- model ops + jobs --------------------- #

    @app.post("/api/models/pull-token/redeem",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull_token_redeem(req: PullTokenRedeemRequest, request: Request):
        """Redeem the single-use grant `localm gui --pull` minted for its own
        deep link (see mint_pull_grant/init.js). Only a genuine, unused,
        unexpired grant bound to this EXACT spec succeeds; anything else 403s
        and the frontend falls back to requiring an explicit human
        confirmation."""
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
        # A local spec is registered in place rather than downloaded, so it writes a
        # caller-chosen server path into registry.json. Gate that on host filesystem
        # reach, so a key with fs_access="none" cannot plant absolute or UNC paths.
        # The textual check runs first, so a UNC spec never reaches a stat.
        if _spec_names_a_host_path(spec):
            require_fs_host(request)
        # Fails here rather than starting a second download of the same spec
        # that cannot finish. ADVISORY ONLY, not the guard: reading the job list
        # and then starting a job are two steps, so two requests can both find
        # nothing running and both proceed. What keeps two downloads from
        # writing the same file is the cross-process lock the pull itself takes,
        # which also covers the contender this cannot see - a `localm pull` a
        # user ran in a terminal. This only spares the user a job that would
        # start and immediately refuse.
        label = f"Model pull {spec}"
        if any(j.get("kind") == "pull" and j.get("status") == "running"
               and j.get("label") == label for j in jobs.snapshot()):
            raise HTTPException(
                409, f"Already downloading {spec} - watch the running job "
                     f"instead of starting a second one.")
        # Pass the spec after "--" so a value like "-h" or "--help" is treated as
        # the model argument, not parsed by the CLI as an option/help flag.
        args = ["pull"]
        if req.name:
            args += ["--name", req.name]
        if req.mmproj:
            args += ["--mmproj", req.mmproj]
        if req.sha256:
            args += ["--sha256", req.sha256]
        if req.store:
            if req.store not in ("copy", "move"):
                raise HTTPException(400, "store must be 'copy' or 'move'")
            args += ["--store", req.store]
        if req.model_type:
            from localm.model_manager import MODEL_TYPES
            if req.model_type not in MODEL_TYPES:
                raise HTTPException(
                    400, f"Invalid type: {req.model_type}. "
                         f"One of: {', '.join(sorted(MODEL_TYPES))}")
            args += ["--type", req.model_type]
        args += ["--", spec]
        # Stream structured download progress; suppress huggingface_hub's own
        # tqdm bars.
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {spec}", owner=principal_id(request))
        return {"job_id": job.id}

    # --------------- ComfyUI missing-model pre-check + curated pull -------- #
    # Read-only pre-check the frontend calls before submitting a generate job: does
    # the currently-configured workflow reference any model file ComfyUI does not
    # have, and if so, is there a curated HuggingFace source to download it from.

    class _NullConsole:
        """A do-nothing stand-in for rich.console.Console, so a read-only
        pre-check (possibly polled repeatedly) never spams server-side console
        output the way an actual generation job's progress printing would."""
        def print(self, *a, **kw):
            pass

    def _build_check_workflow(kind: str, overrides: MediaPreflightRequest):
        """Load *kind*'s currently-configured workflow template and shape it
        with the same model-relevant overrides the generate form has pending,
        mirroring the load-template -> apply_model_overrides -> _build_*_workflow
        steps generate_image / generate_video / generate_music do - minus
        actually submitting a job. ``model_overrides`` (the per-slot picks from
        the Workflow panel's model dropdowns) is applied FIRST, exactly like the
        real generate call, so a picked-but-not-installed model is caught here
        too - otherwise this check would silently validate the template's
        default filenames instead of what will actually run. input_image is
        always None: the image/video builders upload it to ComfyUI as a real
        network call, which a read-only check must never do. Returns the shaped
        workflow dict, or raises ValueError for an unknown *kind*."""
        if kind == "image":
            from localm.image_gen.comfy import _build_image_workflow
            from localm.image_gen.comfy import apply_model_overrides, workflow_path
            workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
            if overrides.model_overrides:
                apply_model_overrides(workflow, overrides.model_overrides)
            _build_image_workflow(
                workflow, prompt="", api_url="", guidance=None, negative_prompt=None,
                cfg=None, seed=0, clip_name1=overrides.clip_name1,
                clip_name2=overrides.clip_name2, lora_name=overrides.lora_name,
                lora_strength_model=1.0, lora_strength_clip=0.5, input_image=None,
                denoise=None, fast_dequant=True, con=_NullConsole())
            return workflow
        if kind == "video":
            from localm.video_gen.comfy import _build_video_workflow
            from localm.video_gen.comfy import apply_model_overrides, workflow_path
            workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
            if overrides.model_overrides:
                apply_model_overrides(workflow, overrides.model_overrides)
            _build_video_workflow(
                workflow, prompt="", negative_prompt=None, frames=1, fps=8,
                width=None, height=None, steps=1, cfg=None, seed=0,
                float_type=None, input_image=None, api_url="")
            return workflow
        if kind == "music":
            from localm.music_gen.comfy import _build_music_workflow
            from localm.music_gen.comfy import apply_model_overrides, workflow_path
            workflow = json.loads(workflow_path().read_text(encoding="utf-8"))
            if overrides.model_overrides:
                apply_model_overrides(workflow, overrides.model_overrides)
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
        if kind == "image" and req.lora_name:
            # The same lexical guard plug.py's _validate_lora_name applies to the real
            # generate route, so a path-traversal or UNC shaped value is rejected here
            # too instead of producing a check-workflow missing its LoraLoader node.
            from localm.image_gen.comfy import is_safe_lora_name
            stripped = req.lora_name.strip()
            if not is_safe_lora_name(stripped):
                raise HTTPException(400, "Invalid LoRA name")
            req.lora_name = stripped
        from localm.media.comfy_client import describe_missing_models
        from localm.media.managed_comfy import comfy_models_dest_dir, resolve_comfy_target
        from localm.model_manager.registry import resolve_comfy_model_source

        def _check():
            try:
                workflow = _build_check_workflow(kind, req)
            except Exception as e:
                logger.debug("preflight workflow build failed for %s: %s", kind, e)
                return []
            target = resolve_comfy_target(plugin=kind)
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
                dest_dir = comfy_models_dest_dir(source.comfy_subfolder, plugin=kind)
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
        """Download one CURATED ComfyUI model into the ComfyUI models folder.

        Requires host filesystem access, the same gate /api/models/scan uses on
        the same folder. When the managed ComfyUI instance is not active - the
        DEFAULT state of a fresh install - comfy_models_dest_dir() resolves to
        `<comfy_workdir>/models/<subfolder>`. comfy_workdir is admin_only
        (settings_schema.py, both the core field and its per-plugin twin), but
        this route's own caller only needs MODELS_WRITE, so require_fs_host()
        below is what requires the CALLER triggering the write to independently
        hold host filesystem reach. The curated table fixes the filename and
        subfolder, so this is not an arbitrary-path write, but choosing the
        parent directory is still host filesystem reach and a UNC value draws
        outbound SMB authentication from the server."""
        from localm.media.managed_comfy import comfy_models_dest_dir
        from localm.model_manager.registry import resolve_comfy_model_source
        from localm.plugins.media_config import MEDIA_PLUGINS
        require_fs_host(request)
        source = resolve_comfy_model_source(req.filename.strip())
        if source is None:
            raise HTTPException(400, f"Not a curated download source: {req.filename}")
        # req.plugin is a selector into the server's own per-plugin config, not a
        # path; an unrecognized value falls back to no plugin context (the global
        # comfy_workdir).
        plugin = req.plugin if req.plugin in MEDIA_PLUGINS else None
        dest_dir = comfy_models_dest_dir(source.comfy_subfolder, plugin=plugin)
        if dest_dir is None:
            raise HTTPException(
                400,
                "No ComfyUI folder is configured to download into - set the "
                "'ComfyUI folder' field in Settings > Media (per-plugin or "
                "shared), or enable the managed ComfyUI instance.")
        args = ["pull", "--type", source.model_type, "--comfy-dest-dir", str(dest_dir),
                "--no-register", "--", source.spec]
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {source.spec}", owner=principal_id(request))
        return {"job_id": job.id}

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
            raise HTTPException(400, f"Alias failed: {e}")
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
        """
        require_fs_host(request)
        _require_registered(req.model)
        from localm.model_manager.registry import relocate_model, relocate_target
        p, reason = relocate_target(req.new_path)
        if p is None:
            raise HTTPException(400, reason)
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            get_plugin_executor(), relocate_model, req.model, req.new_path)
        if not ok:
            raise HTTPException(400, f"Could not relocate {req.model}")
        # .resolve() to match what relocate_model actually wrote (it resolves its
        # own, separately-computed `p` before saving), not the pre-resolution path
        # relocate_target returned above.
        return {"status": "relocated", "model": req.model, "path": str(p.resolve())}

    # ------------------------ model discovery --------------------- #
    # Search HuggingFace for GGUF and/or HF (transformers) models and show per-quant
    # "fits your VRAM" badges for GGUF files. net_mode=off blocks it.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF unreachable
        return 422              # bad repo / no GGUF files / bad format token

    async def _run_discover(fn):
        """Run *fn* off the event loop; map DiscoverError to its HTTP status."""
        from localm.discover import DiscoverError
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(get_plugin_executor(), fn)
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))

    async def _vram_total():
        """Off-thread vram_capacity() plus its extracted 'total' bytes, both
        of which discover_search/discover_files feed into fit_label().

        The returned dict's `free` is withheld unless the reading is BOTH fresh
        and device-global (sysstats._vram_reading_trusted, the same gate
        /api/vram-estimate and /api/stats apply). `total` is a static hardware
        fact and stands even under a stale or process-scoped probe."""
        from localm.discover import vram_capacity
        from localm.sysstats import _vram_reading_trusted
        loop = asyncio.get_running_loop()
        info, status = await loop.run_in_executor(
            get_plugin_executor(), lambda: vram_capacity(return_status=True))
        vram = {"total": info.get("total")}
        if _vram_reading_trusted(info, status):
            vram["free"] = info.get("free")
        return vram, vram.get("total")

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20, formats: str = "gguf",
                               types: str = ""):
        # `formats` is a CSV of {gguf, hf} and `types` a CSV of MODEL_TYPES, both from
        # the search-page checkboxes. Empty tokens are dropped; hf_search raises
        # DiscoverError if none stay valid. An empty or absent `types` means an
        # untyped search (model_types=None). hf_backend_available lets the GUI warn
        # (not block) that a safetensors model needs the .[gpu] extra to run.
        from localm.discover import fit_label, hf_backend_available, hf_search
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        wanted_types = [t.strip().lower() for t in types.split(",") if t.strip()]
        model_types = wanted_types or None
        results = await _run_discover(
            lambda: hf_search(q, limit=limit, formats=wanted, model_types=model_types))
        # Attach a VRAM fit badge to results that carry a size estimate (HF results
        # with safetensors param metadata); GGUF results are sized per-file in the
        # /discover/files expander instead. fit_label yields "" when VRAM is unknown,
        # and a result with no size estimate keeps no fit.
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
