# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI model routes, acquisition group: pull-token redemption, the model pull,
the ComfyUI missing-model preflight and the curated ComfyUI download."""

from __future__ import annotations

import asyncio
import json
from pathlib import PurePosixPath, PureWindowsPath

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import pathsafe
from localm import scopes
from localm.debuglog import logger
from localm.inference._threadpool_timeout import (ThreadCallTimeout,
                                                  run_in_threadpool_bounded)
from localm.inference.http_server import (principal_id, require_fs_host,
                                          require_scope)
from localm.executor import get_plugin_executor
from localm.plugins.gui.routes.models._context import ModelRouteContext
from localm.plugins.gui.web import (ComfyPullRequest, ComfySourceLookupRequest,
                                    MediaPreflightRequest, PullRequest,
                                    PullTokenRedeemRequest, consume_pull_grant)

# Longest filename, class_type or input_name a model-source request accepts.
_MAX_SLOT_FIELD = 255
# Seconds a model-source lookup may take before the request gives up on it:
# above comfy_resolve.MAX_HF_REQUESTS times discover._TIMEOUT. See
# test_lookup_timeout_covers_every_request_timing_out.
_LOOKUP_TIMEOUT = 150.0


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


def register(app: FastAPI, context: ModelRouteContext) -> None:
    jobs = context.jobs

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

    def _source_json(download) -> dict:
        """The client-facing view of a comfy_resolve.ComfyDownload."""
        return {"repo": download.repo, "file": download.path,
                "size_bytes": download.size_bytes, "model_type": download.model_type,
                "origin": download.origin}

    def _download_dest(download, plugin):
        """The ComfyUI folder *download* goes into for *plugin*
        (``models/<comfy_subfolder>/<subdir>``), or None when no ComfyUI folder
        is configured (managed_comfy.comfy_models_dest_dir)."""
        from localm.media.managed_comfy import comfy_models_dest_dir
        base = comfy_models_dest_dir(download.comfy_subfolder, plugin=plugin)
        if base is None or not download.subdir:
            return base
        return base.joinpath(*download.subdir.split("/"))

    @app.post("/api/media/{kind}/preflight",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def media_preflight(kind: str, req: MediaPreflightRequest):
        """Read-only pre-check for missing ComfyUI models. Returns
        {"status": "verified"|"unavailable", "missing": [...], "warning": str}.
        "unavailable" means the check itself could not run (the workflow
        template failed to build); "missing" is [] in that case too, so a
        caller must read status rather than infer success from an empty
        list. "verified" means the check ran to completion, whether or not
        it found anything missing."""
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
        from localm.media.managed_comfy import resolve_comfy_target
        from localm.model_manager.comfy_resolve import (
            cached_comfy_download, curated_comfy_download, search_refusal,
        )

        def _check():
            try:
                workflow = _build_check_workflow(kind, req)
            except Exception as e:
                logger.debug("preflight workflow build failed for %s: %s", kind, e)
                return {
                    "status": "unavailable",
                    "missing": [],
                    "warning": f"Could not check {kind} models before generating.",
                }
            target = resolve_comfy_target(plugin=kind)
            missing = describe_missing_models(workflow, target.api_url)

            results = []
            for slot in missing:
                download = (curated_comfy_download(slot.filename)
                            or cached_comfy_download(slot.filename, slot.class_type,
                                                     slot.input_name))
                entry = {
                    "class_type": slot.class_type,
                    "input_name": slot.input_name,
                    "filename": slot.filename,
                    "source": None,
                    "dest_dir": None,
                    "searchable": False,
                    "reason": "",
                    "detail": "",
                }
                if download is not None:
                    # Stays inside this offloaded call. See
                    # tests/test_comfy_media_routes_offloaded.py.
                    dest_dir = _download_dest(download, kind)
                    entry["source"] = _source_json(download)
                    entry["dest_dir"] = str(dest_dir) if dest_dir is not None else None
                else:
                    refusal = search_refusal(slot.filename, slot.class_type,
                                             slot.input_name)
                    entry["searchable"] = refusal is None
                    if refusal is not None:
                        entry["reason"] = refusal.reason
                        entry["detail"] = refusal.detail
                results.append(entry)
            return {"status": "verified", "missing": results, "warning": ""}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_plugin_executor(), _check)

    @app.post("/api/models/comfy-source/lookup",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_comfy_source_lookup(req: ComfySourceLookupRequest):
        """Find where to download one missing ComfyUI workflow model file: the
        curated table, else a HuggingFace search by exact file name
        (comfy_resolve.lookup_comfy_download), which makes network requests.

        Returns {"status", "source", "dest_dir", "reason", "detail"}. status is
        one of comfy_resolve's LOOKUP_* values; source and dest_dir are set only
        when it is "found", and reason only when it is "unsupported"."""
        from localm.model_manager.comfy_resolve import lookup_comfy_download
        from localm.plugins.media_config import MEDIA_PLUGINS
        for value in (req.filename, req.class_type, req.input_name):
            if len(value) > _MAX_SLOT_FIELD:
                raise HTTPException(400, "Field too long")
        plugin = req.plugin if req.plugin in MEDIA_PLUGINS else None

        def _lookup():
            lookup = lookup_comfy_download(req.filename, req.class_type, req.input_name)
            out = {"status": lookup.status, "source": None, "dest_dir": None,
                   "reason": lookup.reason, "detail": lookup.detail}
            if lookup.download is not None:
                out["source"] = _source_json(lookup.download)
                dest = _download_dest(lookup.download, plugin)
                out["dest_dir"] = str(dest) if dest is not None else None
            return out

        try:
            return await run_in_threadpool_bounded(_lookup, timeout=_LOOKUP_TIMEOUT)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Looking up the model source timed out: {e}")

    @app.post("/api/models/pull-comfy-source",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull_comfy_source(req: ComfyPullRequest, request: Request):
        """Download one ComfyUI workflow model file into the ComfyUI models
        folder. The source comes from the curated table, else, when
        class_type and input_name are given, from
        comfy_resolve.lookup_comfy_download for that slot; the client never
        names a repository or path. 400 when neither has a source. When the
        client sends the repo and file it showed the user, 409 when the source
        resolved now differs from them, and nothing is downloaded.

        Requires host filesystem access, the same gate /api/models/scan uses on
        the same folder: the destination is `<comfy_workdir>/models/<subfolder>`
        unless the managed ComfyUI instance is active, and comfy_workdir may be
        a UNC path."""
        from localm.model_manager.comfy_resolve import (
            curated_comfy_download, lookup_comfy_download,
        )
        from localm.plugins.media_config import MEDIA_PLUGINS
        require_fs_host(request)
        filename = req.filename.strip()
        source = curated_comfy_download(filename)
        if source is None and req.class_type and req.input_name:
            if max(len(filename), len(req.class_type),
                   len(req.input_name)) > _MAX_SLOT_FIELD:
                raise HTTPException(400, "Field too long")
            try:
                lookup = await run_in_threadpool_bounded(
                    lookup_comfy_download, filename, req.class_type, req.input_name,
                    timeout=_LOOKUP_TIMEOUT)
            except ThreadCallTimeout as e:
                raise HTTPException(504, f"Looking up the model source timed out: {e}")
            source = lookup.download
        if source is None:
            raise HTTPException(400, f"No download source for: {req.filename}")
        if ((req.repo is not None and req.repo != source.repo)
                or (req.file is not None and req.file != source.path)):
            raise HTTPException(
                409, f"The download source for {req.filename} changed since it was "
                     f"shown; it is now {source.repo} / {source.path}. Search again.")
        # req.plugin is a selector into the server's own per-plugin config, not a
        # path; an unrecognized value falls back to no plugin context (the global
        # comfy_workdir).
        plugin = req.plugin if req.plugin in MEDIA_PLUGINS else None
        try:
            dest_dir = await run_in_threadpool_bounded(
                _download_dest, source, plugin, timeout=20.0)
        except ThreadCallTimeout as e:
            raise HTTPException(
                504, f"Resolving the ComfyUI download destination timed out: {e}")
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
