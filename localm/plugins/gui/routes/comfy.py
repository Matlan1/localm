# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI managed-ComfyUI routes: set up / status / remove localm's OWN ComfyUI."""

from __future__ import annotations

import json
from typing import Optional

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
        """Whether localm's OWN ComfyUI is installed, where it lives, and whether media calls currently route to it (the S1 coexistence state)."""
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
        body = {
            "installed": installed,
            "state": state,
            "path": str(paths.root) if (installed or state == "corrupt") else None,
            "models_dir": str(paths.models_dir) if installed else None,
            "api_url": MANAGED_COMFY_API_URL,
            "target": cfg.get("comfy_target", "own"),
            "managed_active": managed_comfy_active(cfg),
        }
        if installed:
            body.update(_update_status(paths.root))
        return body

    def _update_status(root) -> dict:
        """The update-availability half of the status, so the GUI can offer Update only when there is something to update TO and can say WHY when it cannot."""
        from localm.media.managed_comfy_fresh import (COMFYUI_PINNED_COMMIT,
                                                      COMFYUI_PINNED_VERSION)
        from localm.media.managed_comfy_provision import MARKER_FILENAME
        installed_commit = None
        installed_version = None
        try:
            marker = json.loads((root / MARKER_FILENAME).read_text(encoding="utf-8"))
            if isinstance(marker, dict):
                c = marker.get("commit")
                v = marker.get("comfyui_version")
                installed_commit = c if isinstance(c, str) else None
                installed_version = v if isinstance(v, str) else None
        except (OSError, ValueError):
            # An unreadable marker is not fatal: report unknown rather than guessing
            # "up to date", which would hide a genuinely available update (rule 5).
            pass

        updatable = (root / ".git").exists()
        return {
            "pinned_commit": COMFYUI_PINNED_COMMIT,
            "pinned_version": COMFYUI_PINNED_VERSION,
            "installed_commit": installed_commit,
            "installed_version": installed_version,
            # None (not False) when the marker could not be read - "we do not know"
            # is a third answer and must not be collapsed into "no update".
            "update_available": (None if installed_commit is None
                                 else installed_commit != COMFYUI_PINNED_COMMIT),
            "updatable": updatable,
            "update_blocked_reason": (
                "" if updatable else
                "This managed ComfyUI has no git history (it was installed via the "
                "non-git copy fallback), so a pinned-version update is not possible. "
                "Remove it and set it up again to move to the version localm ships."),
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
        """Provision localm's own ComfyUI by running the EXISTING `localm comfy setup` entry point as a background job (setup is long + multi-GB, so it must not block the request)."""
        from localm.media.managed_comfy import managed_comfy_paths
        if managed_comfy_paths().root.exists():
            raise HTTPException(
                409, "A managed ComfyUI already exists. Remove it first, then set up "
                "again.")
        if jobs.has_running("comfy-setup"):
            raise HTTPException(409, "A ComfyUI setup is already running.")
        job = _start_setup_job(request, copy_custom_nodes)
        return {"job_id": job.id}

    @app.post("/api/comfy/repair",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_repair(request: Request, copy_custom_nodes: bool = False):
        """Repair an INCOMPLETE managed ComfyUI (comfy_managed_status's 'corrupt' state: the checkout dir exists but was never finished - a crashed process or a closed browser mid-setup) by removing just the checkout (never the models folder - with_models=False, matching remove_managed_comfy's own target list)..."""
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

    @app.post("/api/comfy/update",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_update(request: Request, reinstall_requirements: bool = False,
                           commit: Optional[str] = None):
        """Move localm's managed ComfyUI to the pinned commit localm ships, by running the EXISTING `localm comfy update` entry point as a background job."""
        from localm.media.managed_comfy import is_managed_comfy_installed
        if not is_managed_comfy_installed():
            raise HTTPException(
                409, "No managed ComfyUI is installed - nothing to update. Set one up "
                "first.")
        if jobs.has_running("comfy-update"):
            raise HTTPException(409, "A ComfyUI update is already running.")
        if jobs.has_running("comfy-setup"):
            raise HTTPException(
                409, "A ComfyUI setup is still running - wait for it to finish, then "
                "update.")
        args = ["comfy", "update"]
        if reinstall_requirements:
            args.append("--reinstall-requirements")
        commit = (commit or "").strip() or None
        if commit:
            args.extend(["--commit", commit])
        job = jobs.start_cli("comfy-update", args,
                             host_label="ComfyUI update", owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/comfy/remove",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_remove(with_models: bool = False):
        """Delete localm's managed ComfyUI (and, with with_models, its managed models folder) via the shared remove_managed_comfy helper - the same removal the `localm comfy remove` CLI runs."""
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
