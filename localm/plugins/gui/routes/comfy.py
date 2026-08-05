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
                                     Also reports "corrupt" (an incomplete install
                                     left behind by an abandoned setup attempt) vs
                                     "installing" (a setup job genuinely still running).
  POST /api/comfy/remove          -> delete the managed instance under the data dir
                                     (the shared remove_managed_comfy helper).
  POST /api/comfy/repair          -> clear an INCOMPLETE install (never a genuinely
                                     installed one) and re-run setup, in one action.

Off by default: nothing here changes behaviour until the user opts in. The S1
coexistence toggle (comfy_target) is NOT duplicated here - it stays a Settings
field; this only adds the set-up/status/remove actions around it.

Design: dev-notes/DESIGN-localm-managed-comfyui-2026-07-08.md (decision 8).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import principal_id, require_scope

# remove_managed_comfy() rmtrees a checkout that can include a full venv and
# ComfyUI's custom_nodes - potentially tens of thousands of files - with no
# internal timeout of its own (unlike the urlopen-bound comfy-status/model
# calls elsewhere). 120s is generous for even a large tree on a slow disk;
# genuinely exceeding it means a file lock held by a dead process or a hung
# filesystem call, not ordinary deletion time. Safe against a second
# concurrent rmtree after a client retries past this timeout:
# remove_managed_comfy() now holds its own lock for the whole delete (see
# managed_comfy.py's _remove_lock) regardless of whether the caller is still
# waiting on it.
_REMOVE_TIMEOUT_S = 120.0


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs

    @app.get("/api/comfy/managed-status",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def comfy_managed_status():
        """Whether localm's OWN ComfyUI is installed, where it lives, and whether
        media calls currently route to it (the S1 coexistence state). Drives the
        GUI's swap between the Set-up button, the installed/Remove view, and (new)
        a "needs repair" view.

        `state` distinguishes four cases the plain `installed` boolean alone
        cannot: "not_installed" (nothing here, offer Set up), "installing" (the
        setup job is genuinely still running right now - NOT corrupt, just not
        done yet), "corrupt" (the install dir exists, is_managed_comfy_installed()
        says no completion marker, and no setup job is currently running for
        it - an earlier attempt was abandoned: a crashed process, a closed
        browser tab mid-setup - offer Repair, not a dead-end "already exists"),
        and "installed" (the normal ready state). `installed` (the plain
        boolean) is kept for existing callers."""
        from localm.config import load_config
        from localm.media.managed_comfy import (
            MANAGED_COMFY_API_URL, is_managed_comfy_installed, managed_comfy_active,
            managed_comfy_paths,
        )
        cfg = load_config()
        installed = is_managed_comfy_installed()
        paths = managed_comfy_paths()
        if installed:
            state = "installed"
        elif jobs.has_running("comfy-setup"):
            state = "installing"
        elif paths.root.exists():
            state = "corrupt"
        else:
            state = "not_installed"
        return {
            "installed": installed,
            "state": state,
            "path": str(paths.root) if (installed or state == "corrupt") else None,
            "models_dir": str(paths.models_dir) if installed else None,
            "api_url": MANAGED_COMFY_API_URL,
            "target": cfg.get("comfy_target", "own"),
            "managed_active": managed_comfy_active(cfg),
        }

    def _start_setup_job(request: Request, copy_custom_nodes: bool):
        # Pass an EXPLICIT custom-nodes flag so the CLI never prompts (its
        # resolve_copy_custom_nodes short-circuits on a set flag) - the job runs with
        # stdin closed and must not hang. Default: a clean start (decision 3).
        flag = "--copy-custom-nodes" if copy_custom_nodes else "--no-custom-nodes"
        return jobs.start_cli(
            "comfy-setup", ["comfy", "setup", flag],
            host_label="ComfyUI setup", owner=principal_id(request))

    @app.post("/api/comfy/setup",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_setup(request: Request, copy_custom_nodes: bool = False):
        """Provision localm's own ComfyUI by running the EXISTING `localm comfy setup`
        entry point as a background job (setup is long + multi-GB, so it must not
        block the request). Streams progress via /api/jobs/{id}/events, like a model
        pull. Refuses (409) when a managed instance already exists - the user removes
        it first (or uses /api/comfy/repair for an incomplete one - see
        comfy_managed_status's "corrupt" state); we never silently clobber."""
        from localm.media.managed_comfy import managed_comfy_paths
        if managed_comfy_paths().root.exists():
            raise HTTPException(
                409, "A managed ComfyUI already exists. Remove it first, then set up "
                "again.")
        job = _start_setup_job(request, copy_custom_nodes)
        return {"job_id": job.id}

    @app.post("/api/comfy/repair",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_repair(request: Request, copy_custom_nodes: bool = False):
        """Repair an INCOMPLETE managed ComfyUI (comfy_managed_status's "corrupt"
        state: the checkout dir exists but was never finished - a crashed process
        or a closed browser mid-setup) by removing just the checkout (never the
        models folder - with_models=False, matching remove_managed_comfy's own
        target list) and re-running setup. Config lives outside the checkout
        entirely (LOCALM_HOME's own config.json/registry.json, a sibling of the
        `comfyui` dir, not inside it) and the managed models live in a SEPARATE
        `comfyui-models` sibling dir - neither is touched. Refuses (409) if the
        instance actually reads as installed (never repair-away a real one) or a
        setup job is already running (no double-launch)."""
        from localm.media.managed_comfy import (
            is_managed_comfy_installed, managed_comfy_paths, remove_managed_comfy,
        )
        if is_managed_comfy_installed():
            raise HTTPException(409, "This managed ComfyUI is already installed - "
                                "nothing to repair.")
        if jobs.has_running("comfy-setup"):
            raise HTTPException(409, "A ComfyUI setup is already running.")
        if not managed_comfy_paths().root.exists():
            raise HTTPException(409, "No managed ComfyUI install found to repair - "
                                "use Set up instead.")
        try:
            removed, failed = await run_in_threadpool_bounded(
                remove_managed_comfy, False, timeout=_REMOVE_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Clearing the incomplete install timed out: {e}")
        if failed:
            raise HTTPException(500, "Could not clear the incomplete install: "
                                + "; ".join(failed))
        job = _start_setup_job(request, copy_custom_nodes)
        return {"job_id": job.id, "cleared": [str(p) for p in removed]}

    @app.post("/api/comfy/remove",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_remove(with_models: bool = False):
        """Delete localm's managed ComfyUI (and, with with_models, its managed models
        folder) via the shared remove_managed_comfy helper - the same removal the
        `localm comfy remove` CLI runs. The user's own ComfyUI is never touched. A
        delete that fails is surfaced (500), never reported as success (rule 5); an
        honest no-op is returned when nothing is installed."""
        from localm.media.managed_comfy import remove_managed_comfy
        try:
            removed, failed = await run_in_threadpool_bounded(
                remove_managed_comfy, with_models, timeout=_REMOVE_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Removing the managed ComfyUI timed out: {e}")
        if failed:
            raise HTTPException(500, "Could not remove: " + "; ".join(failed))
        if not removed:
            return {"status": "noop",
                    "message": "No managed ComfyUI is installed.", "removed": []}
        return {"status": "removed", "removed": [str(p) for p in removed]}
