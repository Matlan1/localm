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
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import pathsafe
from localm import scopes
from localm.debuglog import logger
from localm.inference.http_server import (principal_id, require_fs_host,
                                          require_scope, unload_all_models,
                                          unload_one_model)
import localm.inference.http_server as _hs
from localm.executor import get_plugin_executor
from localm.plugins.gui.web import (AliasRequest, ComfyPullRequest,
                                    LoadModelRequest, MediaPreflightRequest,
                                    PullRequest, PullTokenRedeemRequest,
                                    RemoveModelRequest, RenameModelRequest,
                                    ScanRequest,
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

    Deliberately textual and existence-INDEPENDENT, for three reasons. It cannot
    stall: a UNC spec is classified without the stat that would block in the SMB
    redirector for minutes (see pathsafe.is_unc_or_device_path). It cannot become
    an existence oracle: the authorisation answer is identical whether or not the
    file is there. And it has no TOCTOU: "may this caller name a host path" is
    not a question whose answer may change between the check and the pull. That
    makes it deliberately BROADER than pull.py's own is_local_path (which also
    requires the path to exist) - a non-existent absolute path is refused here
    even though pull.py would have gone on to treat it as a remote spec, which is
    the safe direction to differ in.

    An earlier revision ALSO probed the filesystem for a relative spec, since
    "owner/repo" and "models/foo.gguf" are the same shape and only a stat can
    tell them apart. That probe was removed: it made the 403-vs-200 answer depend
    on whether the named file exists, i.e. it handed a caller WITHOUT host
    filesystem access an existence oracle for any relative path - the very
    capability this gate exists to withhold. Trading a narrow registration gap for
    a general-purpose oracle is a bad trade, and it also reintroduced the stat
    this function is built to avoid.

    RESIDUAL, stated rather than papered over: a relative spec with no ".."
    component that happens to name an existing FILE is still registered in place
    without this gate firing. Its reach is bounded by the server's working
    directory, it is unchanged from previous behaviour rather than something
    introduced here, and closing it needs a filesystem answer - which is exactly
    what cannot be spent here without rebuilding the oracle."""
    # Trimming here is deliberate and is NOT the over-match that pathsafe's
    # predicate avoids: this value is USER-TYPED (a spec pasted into the GUI
    # arrives with a trailing newline routinely), the route already trimmed it
    # once, and the consequence of over-matching is requiring host filesystem
    # access for an odd-looking spec - the fail-safe direction, for a path the
    # caller named themselves rather than one a remote service supplied.
    s = spec.strip()
    if not s:
        return False
    if pathsafe.is_unc_or_device_path(s):
        return True
    if s.startswith("~"):
        return True
    # Judge under BOTH flavours regardless of host OS: a drive-qualified spec, and
    # the drive-RELATIVE form (a drive letter and colon with no separator), are
    # host paths whoever is running the server, and a POSIX server must not
    # mis-read a Windows-shaped spec as a HuggingFace repo id.
    for flavour in (PureWindowsPath, PurePosixPath):
        pure = flavour(s)
        if pure.is_absolute() or pure.drive or pure.root:
            return True
        # A '..' component makes a RELATIVE spec reach anywhere on the disk from
        # the server's working directory, so it is a host path in every sense
        # that matters here.
        if any(part == ".." for part in pure.parts):
            return True
    return False


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
        # Fetched ONCE, off the event loop, before the row loop below (not per
        # row): get_embedder() can hold embedder._LOCK for the full duration of
        # an IsolatedEmbedder native/subprocess load, and loaded_path() blocks
        # on that same lock. A synchronous call here would freeze the WHOLE
        # event loop - every request this server is serving, not just this one
        # - for that window (same hazard as http_server.unload_all_models's
        # loaded_dim() call). The embedder's path cannot change mid-request, so
        # one fetch correctly serves every row's comparison below.
        from localm.inference import embedder as _embedder_mod
        loop = asyncio.get_running_loop()
        emb_path = await loop.run_in_executor(get_plugin_executor(), _embedder_mod.loaded_path)
        rows = []
        for name, entry in sorted(registry.items()):
            epath = _entry_path(entry)
            if epath is None:
                # A hand-corrupted / cross-version registry entry (non-dict, a
                # null / non-string / empty path, or one carrying a '..'
                # component). Skip it so one bad row never 500s the whole Models
                # page; the CLI `localm list` shows it as [corrupt] and
                # `localm rm <name>` drops it. Mirrors #562, which routes every
                # registry consumer through _entry_path.
                logger.debug("skipping malformed registry entry %r in /api/models", name)
                continue
            mtype = str(entry.get("model_type", "llm"))
            if type and mtype != type:
                continue
            rows.append((name, entry, mtype, epath))

        # Every row below needs a stat and a resolve of a path THIS HANDLER DID NOT
        # CHOOSE - it came out of registry.json. Those are blocking syscalls, and
        # this is an `async def`, so run inline they stall the entire server rather
        # than just this request: a registered UNC path blocks in the Windows SMB
        # redirector (minutes against an unroutable host, and a reachable one also
        # draws an outbound authentication attempt from the server process). Do the
        # whole filesystem pass in ONE executor hop, then build the response on the
        # loop from its results. The embedder's own path is resolved in the same
        # hop for the identity comparison below, for the same reason.
        def _probe_rows() -> tuple:
            sizes: dict = {}
            mtimes: dict = {}
            resolved: dict = {}
            # The per-row resolve() has exactly ONE consumer: the embedder
            # identity comparison below. So it is skipped entirely when no
            # embedder is loaded, rather than spent and dropped. That is not
            # micro-optimisation - resolve() is a syscall, and on a registered
            # UNC row it is a second full SMB-redirector timeout on top of the
            # stat, plus a second outbound authentication attempt against a
            # reachable share. Doing it only when its answer is used halves the
            # cost of exactly the row that is most expensive.
            emb_resolved = None
            if emb_path is not None:
                try:
                    emb_resolved = Path(emb_path).resolve()
                except (OSError, ValueError):
                    emb_resolved = None
            for _n, _e, _m, ep in rows:
                p = Path(ep)
                # ONE stat() for both size and mtime (the previous size-only form
                # called p.is_file() - itself a stat - then p.stat() again on a
                # hit). mtime is recorded for a directory too (an HF model dir),
                # unlike size, which stays None for a dir - a directory's total
                # size needs a recursive walk this probe does not do, but its own
                # mtime is free once stat() has already been called.
                try:
                    st_res = p.stat()
                    sizes[ep] = st_res.st_size if stat.S_ISREG(st_res.st_mode) else None
                    mtimes[ep] = st_res.st_mtime
                except (OSError, ValueError):
                    sizes[ep] = None
                    mtimes[ep] = None
                if emb_resolved is None:
                    continue
                try:
                    resolved[ep] = p.resolve()
                except (OSError, ValueError):
                    resolved[ep] = None
            return sizes, mtimes, resolved, emb_resolved

        sizes, mtimes, resolved_paths, emb_resolved = await loop.run_in_executor(
            get_plugin_executor(), _probe_rows)

        models = []
        for name, entry, mtype, epath in rows:
            size = sizes.get(epath)
            mtime = mtimes.get(epath)
            engine = _hs._engines.get(name)
            loaded = engine.loaded if engine is not None else False
            # A registered model can also be the shared EMBEDDING model, loaded
            # via get_embedder() - a lifecycle entirely separate from _engines
            # (see localm.inference.embedder's module docstring), so it never
            # shows up above. Recognise it by resolved PATH (not name/config)
            # so this row's "loaded" status - and its per-row Unload control,
            # gated on this flag - actually reflect a resident embedder.
            if not loaded and emb_resolved is not None:
                row_resolved = resolved_paths.get(epath)
                loaded = row_resolved is not None and emb_resolved == row_resolved
            models.append({
                "name": name,
                "source": str(entry.get("source", "")),
                "size_bytes": size,
                "mtime": mtime,
                "active": name == current,
                # Independent of "active": a model can be resident in VRAM
                # (loaded) without being the one currently serving requests -
                # surfaced so the Models page can offer a per-row Unload
                # action on ANY loaded model, not just the active one.
                "loaded": loaded,
                "model_type": mtype,
            })
        out = {"models": models, "active": current}
        # The multi-GPU split distribution the ACTIVE model's load actually
        # applied (GgufBackend.applied_gpu_split - auto free-VRAM-proportional,
        # pinned, or the equal fallback), for the sidebar's loaded-model
        # status. Absent whenever there is nothing to show - no active engine,
        # no split, or a backend that does not record one (HF's
        # device_map="auto" placement is torch-internal) - so old clients and
        # single-GPU boxes receive the exact pre-existing payload shape.
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
        `workdir` (the old button's bodyless POST, or an explicit `{}`) it
        scans whatever `comfy_workdir` is configured. `dry_run` previews
        per-type counts and registers nothing.

        BOTH forms require `require_fs_host` - called BEFORE the try/except
        below, so its 403 propagates as-is rather than getting reported as a
        generic 500. Either one walks a host directory and writes the resulting
        absolute paths into registry.json, a capability equivalent to the host
        file/folder browser (/api/fs/dirs), so a MODELS_WRITE-only key that
        lacks host filesystem access must not reach it.

        The gate used to be `if workdir:`, which made that guarantee false for
        the exact case the paragraph above asserts: `comfy_workdir` is a plain
        Widget.FOLDER settable by any config:write key, so a bodyless POST
        scanned a caller-CHOSEN folder with no fs_access check at all. The
        registered rows are permanent and every consumer re-stats them on each
        launch, so a planted UNC row is a lasting event-loop stall plus
        outbound SMB from the server process. Scanning is authorised by host
        filesystem reach, not by where the folder name happened to come
        from."""
        from localm.model_manager.scan import preview_comfy_models, scan_comfy_models
        import asyncio
        from functools import partial
        require_fs_host(request)
        workdir = req.workdir if req else None
        dry_run = bool(req and req.dry_run)
        loop = asyncio.get_running_loop()
        try:
            if dry_run:
                res = await loop.run_in_executor(
                    get_plugin_executor(), partial(preview_comfy_models, workdir=workdir))
                return {
                    "dry_run": True,
                    "method": res.method,
                    "counts": res.counts,
                    "already_registered": res.already_registered,
                    "total_new": sum(res.counts.values()),
                }
            res = await loop.run_in_executor(
                get_plugin_executor(), partial(scan_comfy_models, workdir=workdir))
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

    @app.post("/api/embedding/warmup",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def embedding_warmup(request: Request):
        """Load the shared embedder NOW, from an explicit user action, instead of
        the first real /v1/embeddings / memory-consolidate / RAG-recall call
        silently paying the cost - measured up to two 300s timeout windows (a
        VRAM-eviction wait plus the isolated child's spawn+native-load) on a cold
        server, even after ``localm setup-embeddings`` (which only pre-fetches
        the file, never warms the singleton - a restart resets it too).

        Coarse PARENT-side stage events only (ADR-0004 Unit B): the isolated
        embedder child's own load/embed IPC protocol is untouched. Uses the same
        JobManager/SSE mechanism as model pull/remove - a 9th consumer of an
        already-proven primitive, not a new channel."""
        from localm.inference.embedder import get_embedder, last_error, loaded_dim
        already = loaded_dim()
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
        from localm.config import load_registry
        from localm.discover import vram_capacity
        from localm.model_meta import cached_n_layers
        from localm.model_manager import _entry_path
        from localm.model_manager.gguf import gguf_kv_bytes_per_token
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        n_layers = None
        kv_bytes_per_token = 0
        # _entry_path returns None for a malformed / corrupt entry (non-dict, or a
        # null / non-string / empty path). The route's own guard below is
        # except OSError, which would NOT catch the AttributeError / TypeError such
        # an entry raises; routing through _entry_path keeps a corrupt entry from
        # 500ing the VRAM readout (model_bytes stays 0 -> still a valid estimate).
        epath = _entry_path(load_registry().get(name))
        if epath is not None:
            # Off the event loop, for the same reason as /api/models and
            # /v1/models/{id}: the path comes out of registry.json, not from this
            # handler, so stat() is a blocking syscall on a value this server did
            # not choose. A registered UNC row blocks in the Windows SMB
            # redirector for minutes and draws an outbound authentication attempt
            # from the server process - inline in this `async def` that stalls
            # every request the server is serving, not just this one. The GGUF
            # header read for the KV-shape probe below is the same class of I/O,
            # so it rides the same executor call instead of adding a second one.
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
                        return p.stat().st_size, cached_n_layers(str(p)), kv_bpt
                except (OSError, ValueError):
                    pass
                return model_bytes, n_layers, kv_bytes_per_token

            model_bytes, n_layers, kv_bytes_per_token = await asyncio.get_running_loop().run_in_executor(
                get_plugin_executor(), _measure, epath)
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers, n_layers=n_layers,
                            kv_bytes_per_token=kv_bytes_per_token)
        # vram_capacity() -> list_gpus() probes the GPU driver; keep it OFF the
        # event loop (it is safe-by-construction but still may take up to its
        # deadline on a wedged driver) so a stats read never stalls the WebUI.
        # return_status=True so a stale (timed-out) or process-blind (Windows/AMD,
        # see _vram_reading_trusted) free is not weighed as if it were current: it
        # would make a too-big model read as "fits". When the reading is untrusted,
        # free is withheld (fits -> None) and the UI shows "free VRAM unknown"
        # rather than a confident wrong verdict.
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
        ``gpus`` is a frozen last-known-good or []). Without it, a timed-out
        probe was indistinguishable from a GPU-less box, so the Settings GPU
        controls silently vanished on a slow driver (AGENTS.md rule 5).

        ``index_space`` (present only as ``"native"``) says the ``gpus``
        indices are the NATIVE runtime's own device order rather than
        list_gpus()'s torch/nvidia-smi numbering - see the vulkan branch
        below; the client labels the numbering accordingly."""
        from localm.config import load_config
        from localm.discover import GPU_PROBE_OK
        cfg = load_config()
        # The reads below probe the GPU driver (or spawn the probe daemon);
        # offload them so a wedged/slow driver never blocks the event loop and
        # freezes the whole WebUI. Off-loop and user-initiated (a Settings page
        # open), this caller can afford to wait: join an in-flight probe
        # (wait_for_inflight, #701) rather than bounce off the /api/stats
        # heartbeat's probe with an instant BUSY + [].
        loop = asyncio.get_running_loop()

        def _read_devices():
            # Native-first on the vulkan build: these selectors write indices
            # the LOADER consumes, and on that build those live in
            # ggml-vulkan's own index space, which list_gpus() (torch.cuda /
            # nvidia-smi) can neither see nor order (GPU-SPLIT-VKINDEX) - so
            # selector rows built from list_gpus() stayed hidden on a fully
            # working multi-GPU vulkan box. The native registry, read via the
            # crash-isolated probe daemon, is the one source whose indices
            # mean what a load will do; a completed registry read is a
            # conclusive probe (GPU_PROBE_OK - an empty/short list is a real
            # "that is all there is"). Falls back to the exact pre-existing
            # list_gpus() behavior, with NO index_space claim, when the
            # daemon/registry cannot answer (None).
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
        # A local spec is REGISTERED IN PLACE, not downloaded (pull.py's
        # is_local_path branch -> add_local with store=None), so it writes a
        # caller-chosen server path into registry.json. Gate that on host
        # filesystem reach: MODELS_WRITE alone must not let a key with
        # fs_access="none" plant arbitrary absolute or UNC paths that every
        # consumer then re-stats on every launch. Textual check first, so a UNC
        # spec never reaches a stat (see _spec_names_a_host_path).
        if _spec_names_a_host_path(spec):
            require_fs_host(request)
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
        if req.model_type:
            from localm.model_manager import MODEL_TYPES
            if req.model_type not in MODEL_TYPES:
                raise HTTPException(
                    400, f"Invalid type: {req.model_type}. "
                         f"One of: {', '.join(sorted(MODEL_TYPES))}")
            args += ["--type", req.model_type]
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
        """Download one CURATED ComfyUI model into the ComfyUI models folder.

        Requires host filesystem access for the same reason /api/models/scan
        does, and it is the same folder either way. When the managed ComfyUI
        instance is not active - the DEFAULT state of a fresh install -
        comfy_models_dest_dir() resolves to `<comfy_workdir>/models/<subfolder>`,
        and comfy_workdir carries no admin_only flag, so a plain config:write key
        chooses it. The download then mkdir -p's that directory and streams a
        multi-gigabyte file into it from the server process. The curated table
        fixes the filename and subfolder, so this is not arbitrary-path WRITE -
        but choosing the parent directory is still host filesystem reach, and a
        UNC value draws the same outbound SMB authentication from the server that
        the scan gate exists to prevent.

        This route was the inconsistent survivor when scan and pull were gated:
        same invariant, same config value, same harm, no gate. Fixing one door
        and leaving its neighbour open is how a boundary gets believed to be
        closed when it is not."""
        from localm.media.managed_comfy import comfy_models_dest_dir
        from localm.model_manager.registry import resolve_comfy_model_source
        require_fs_host(request)
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
        # alias_model stores under the SANITIZED name (a space/slash/colon can
        # never become a raw registry key), so precheck against that same name and
        # report it back: prechecking the raw name let a collision through, and
        # answering with the raw name told the user an alias that does not exist
        # (REG-562).
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
            # alias_model returns False for "model vanished" and "name taken"
            # alike, and both were prechecked above, so reaching here means a
            # concurrent writer won the race. Never answer "aliased" for an alias
            # that was not created (AGENTS.md rule 5); say which race it lost.
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
        under its new name instead of being orphaned under the old one."""
        registry = _require_registered(req.model)
        # Same precheck-then-report-the-sanitized-name discipline as alias
        # (REG-562): sanitizing happens server-side, so the collision check and
        # the eventual response must both speak the sanitized name, not the raw
        # text the caller sent.
        from localm.model_manager import _sanitize_name, rename_model_with_notes
        new_name = _sanitize_name(req.new_name)
        if new_name != req.model and new_name in registry:
            raise HTTPException(409, f"Name already taken: {new_name}")
        loop = asyncio.get_running_loop()
        try:
            renamed, notes = await loop.run_in_executor(
                get_plugin_executor(), rename_model_with_notes, req.model, req.new_name)
        except Exception as e:
            raise HTTPException(400, f"Rename failed: {e}")
        if not renamed:
            # rename_model_with_notes itself distinguishes "vanished" from "name
            # taken" via its own console output, but only the return value
            # crosses the executor boundary - re-derive which race it lost the
            # same way alias does.
            from localm.config import load_registry
            if req.model not in load_registry():
                raise HTTPException(404, f"Model not registered: {req.model}")
            raise HTTPException(409, f"Name already taken: {new_name}")
        # Synchronous, in-memory only (no await) - safe to call directly on the
        # event loop right after the executor call above returns.
        _hs.rekey_loaded_model(req.model, new_name)
        # `notes` includes what could be migrated AND what could not (e.g. a
        # per-project .localcoder/config.toml, unreachable from here) - it must
        # reach the caller, not just the server log, or a user has no way to
        # learn their coder config may still name the old model.
        return {"status": "renamed", "model": req.model, "new_name": new_name,
                "notes": notes}

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
    async def discover_search(q: str = "", limit: int = 20, formats: str = "gguf",
                               types: str = ""):
        # `formats` is a CSV of {gguf, hf} and `types` a CSV of MODEL_TYPES, both
        # from the search-page checkboxes. Empty tokens are dropped; hf_search
        # raises DiscoverError if none stay valid. An empty/absent `types` means
        # the legacy untyped search (model_types=None) - byte-for-byte today's
        # request shape, so a caller that predates `types` is unaffected.
        # hf_backend_available lets the GUI warn (not block) that a safetensors
        # model needs the .[gpu] extra to RUN, though it can still be downloaded.
        from localm.discover import fit_label, hf_backend_available, hf_search
        wanted = [f.strip() for f in formats.split(",") if f.strip()]
        wanted_types = [t.strip().lower() for t in types.split(",") if t.strip()]
        model_types = wanted_types or None
        results = await _run_discover(
            lambda: hf_search(q, limit=limit, formats=wanted, model_types=model_types))
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
