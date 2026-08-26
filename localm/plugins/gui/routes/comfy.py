# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI managed-ComfyUI routes: set up / status / remove localm's OWN ComfyUI.

These wire the GUI's "Set up localm's own ComfyUI" surface to the EXISTING entry
points and add no provisioning logic of their own (that lives in
localm/media/managed_comfy*):

  POST /api/comfy/setup           -> dispatch `localm comfy setup` as a background
                                     JOB (a multi-GB install must not block the
                                     request); returns {"job_id"} to stream.
  GET  /api/comfy/managed-status  -> is a managed instance installed, where, and is
                                     localm routing to it (the coexistence state).
                                     Also reports "corrupt" (an incomplete install
                                     left behind by an abandoned setup attempt) vs
                                     "installing" (a setup job genuinely still running),
                                     and whether an update is available/possible.
  POST /api/comfy/update          -> dispatch `localm comfy update` as a background
                                     JOB, moving the managed instance to the pinned
                                     commit localm ships (or an advanced ``commit``
                                     override); returns {"job_id"}.
  POST /api/comfy/remove          -> delete the managed instance under the data dir
                                     (the shared remove_managed_comfy helper).
  POST /api/comfy/repair          -> clear an INCOMPLETE install (never a genuinely
                                     installed one) and re-run setup, in one action.

Off by default: nothing here changes behaviour until the user opts in. The
coexistence toggle (comfy_target) is NOT duplicated here - it stays a Settings
field; this only adds the set-up/status/remove actions around it.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import principal_id, require_scope

# remove_managed_comfy() rmtrees a checkout that can include a full venv and
# ComfyUI's custom_nodes - potentially tens of thousands of files - with no
# internal timeout of its own. 120s is generous for even a large tree on a slow
# disk; exceeding it means a file lock held by a dead process or a hung
# filesystem call, not ordinary deletion time. A second concurrent rmtree after
# a client retries past this timeout is prevented by remove_managed_comfy()'s
# own lock (managed_comfy.py's _remove_lock), held for the whole delete
# regardless of whether the caller is still waiting on it.
_REMOVE_TIMEOUT_S = 120.0


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs

    @app.get("/api/comfy/managed-status",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def comfy_managed_status():
        """Whether localm's OWN ComfyUI is installed, where it lives, and whether
        media calls currently route to it (the coexistence state). Drives the
        GUI's swap between the Set-up button, the installed/Remove view, and a
        "needs repair" view.

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
        """The update-availability half of the status, so the GUI can offer Update
        only when there is something to update TO and can say WHY when it cannot.

        CHEAP and side-effect-free: it reads the provisioning marker and tests for
        a `.git` dir. It does NOT shell out to git; this runs on the event loop on
        every settings-page poll.

        ``updatable`` is an ADVISORY pre-flight, not the gate. The authoritative
        refusal lives in update_managed_comfy(), which resolves HEAD with real git
        and returns its own message; this only lets the GUI say so BEFORE the user
        starts a job that would fail minutes later. A `.git` dir present but
        unusable still reaches that real check unchanged."""
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
            # "up to date", which would hide a genuinely available update.
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
        # stdin closed and must not hang. Defaults to a clean start.
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
        comfy_managed_status's "corrupt" state); an existing instance is never
        clobbered.

        Also refuses (409) when a setup job is ALREADY running: the on-disk check
        alone is not enough, since the checkout directory does not exist yet for the
        first few moments of a fresh setup, so two overlapping "Set up" clicks (e.g.
        two browser tabs) would both pass that check and double-launch the setup CLI
        onto the same target checkout."""
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

    @app.post("/api/comfy/update",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def comfy_update(request: Request, reinstall_requirements: bool = False,
                           commit: Optional[str] = None):
        """Move localm's managed ComfyUI to the pinned commit localm ships, by running
        the EXISTING `localm comfy update` entry point as a background job. An update
        re-checks-out the source and can reinstall requirements, so it takes minutes
        and must not block the request - same shape as /api/comfy/setup, streamed via
        /api/jobs/{id}/events.

        ``reinstall_requirements`` forwards `--reinstall-requirements`. Off by
        default, matching the CLI: a partial pip upgrade is not reliably
        rollback-able, so only the git source rollback is guaranteed. It is
        reachable from here because a pin that advances across a dependency change
        otherwise leaves a checkout that moved without its new deps.

        ``commit`` forwards `--commit <sha>`, the CLI's own "advanced: update to a
        specific ComfyUI commit instead of the shipped pin" testing knob
        (localm/cli/comfy.py). Not validated here: update_managed_comfy() resolves
        and checks out whatever it is given and reports a bad ref through the job's
        own output.

        Refuses (409) when nothing is installed, when an update is already running, or
        while a setup/repair job is still in flight - never two writers on one checkout.
        The non-git refusal is NOT duplicated here: update_managed_comfy() owns it and
        reports it through the job's own output (the GUI also pre-warns from
        managed-status's advisory ``updatable``)."""
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
        """Delete localm's managed ComfyUI (and, with with_models, its managed models
        folder) via the shared remove_managed_comfy helper - the same removal the
        `localm comfy remove` CLI runs. The user's own ComfyUI is never touched. A
        delete that fails is surfaced (500), never reported as success; an honest
        no-op is returned when nothing is installed."""
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
