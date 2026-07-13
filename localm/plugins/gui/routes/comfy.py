# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI managed-ComfyUI routes: set up / status / remove localm's OWN ComfyUI.

Stage S5 (GUI-button slice) of the localm-managed ComfyUI feature. These wire the
GUI's "Set up localm's own ComfyUI" surface to the EXISTING entry points; they add
no provisioning logic of their own (S2/S3 own that in localm/media/managed_comfy*):

  POST /api/comfy/setup           -> dispatch `localm comfy setup` as a background
                                     JOB (a multi-GB install must not block the
                                     request); returns {"job_id"} to stream.
  GET  /api/comfy/managed-status  -> is a managed instance installed, where, and is
                                     localm routing to it (the S1 coexistence state).
  POST /api/comfy/remove          -> delete the managed instance under the data dir
                                     (the shared remove_managed_comfy helper).

Off by default: nothing here changes behaviour until the user opts in. The S1
coexistence toggle (comfy_target) is NOT duplicated here - it stays a Settings
field; this only adds the set-up/status/remove actions around it.

Design: dev-notes/DESIGN-localm-managed-comfyui-2026-07-08.md (decision 8).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from localm import scopes
from localm.inference.http_server import principal_id, require_scope


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs

    @app.get("/api/comfy/managed-status",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def comfy_managed_status():
        """Whether localm's OWN ComfyUI is installed, where it lives, and whether
        media calls currently route to it (the S1 coexistence state). Drives the
        GUI's swap between the Set-up button and the installed/Remove view."""
        from localm.config import load_config
        from localm.media.managed_comfy import (
            MANAGED_COMFY_API_URL, is_managed_comfy_installed, managed_comfy_active,
            managed_comfy_paths,
        )
        cfg = load_config()
        installed = is_managed_comfy_installed()
        paths = managed_comfy_paths()
        return {
            "installed": installed,
            "path": str(paths.root) if installed else None,
            "models_dir": str(paths.models_dir) if installed else None,
            "api_url": MANAGED_COMFY_API_URL,
            "target": cfg.get("comfy_target", "own"),
            "managed_active": managed_comfy_active(cfg),
        }

    @app.post("/api/comfy/setup",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_setup(request: Request, copy_custom_nodes: bool = False):
        """Provision localm's own ComfyUI by running the EXISTING `localm comfy setup`
        entry point as a background job (setup is long + multi-GB, so it must not
        block the request). Streams progress via /api/jobs/{id}/events, like a model
        pull. Refuses (409) when a managed instance already exists - the user removes
        it first (mirrors provision_by_copy's own guard); we never silently clobber."""
        from localm.media.managed_comfy import managed_comfy_paths
        if managed_comfy_paths().root.exists():
            raise HTTPException(
                409, "A managed ComfyUI already exists. Remove it first, then set up "
                "again.")
        # Pass an EXPLICIT custom-nodes flag so the CLI never prompts (its
        # resolve_copy_custom_nodes short-circuits on a set flag) - the job runs with
        # stdin closed and must not hang. Default: a clean start (decision 3).
        flag = "--copy-custom-nodes" if copy_custom_nodes else "--no-custom-nodes"
        job = jobs.start_cli(
            "comfy-setup", ["comfy", "setup", flag],
            host_label="ComfyUI setup", owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/comfy/remove",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_remove(with_models: bool = False):
        """Delete localm's managed ComfyUI (and, with with_models, its managed models
        folder) via the shared remove_managed_comfy helper - the same removal the
        `localm comfy remove` CLI runs. The user's own ComfyUI is never touched. A
        delete that fails is surfaced (500), never reported as success (rule 5); an
        honest no-op is returned when nothing is installed."""
        from localm.media.managed_comfy import remove_managed_comfy
        removed, failed = await run_in_threadpool(remove_managed_comfy, with_models)
        if failed:
            raise HTTPException(500, "Could not remove: " + "; ".join(failed))
        if not removed:
            return {"status": "noop",
                    "message": "No managed ComfyUI is installed.", "removed": []}
        return {"status": "removed", "removed": [str(p) for p in removed]}
