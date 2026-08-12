# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI llama.cpp runtime-update routes: check for, and apply, a newer
provisioned build - the same space as /api/update/* (localm/inference/routes/
admin.py) but for the NATIVE runtime `localm setup-llama` provisions, not the
Python source tree.

  GET  /api/runtime/check   -> read-only: is a different build available for
                               the currently installed backend, per
                               setup_llama.check_runtime_update().
  POST /api/runtime/update  -> dispatch `localm setup-llama --backend <installed>
                               --force --yes` as a background JOB (a real
                               download + native load-test can take a while,
                               same shape as /api/comfy/update); returns
                               {"job_id"} to stream.

Why a background job and not a blocking call like /api/update/apply: this is
closer in shape to /api/comfy/update (a multi-step operation worth streaming
progress for) than to the code-tree swap, which is comparatively quick.

Safety: setup-llama's own _provisioning_lock (localm/setup_llama.py) is what
actually serializes this against a concurrent `localm update` re-provision or
a user's own terminal `setup-llama` run - a cross-process race this route
introduces as a genuinely new, second trigger onto the same directory (see
diff-review-discipline.md item 26, the identical hazard /api/comfy/update hit
before its own lock existed). jobs.has_running() below is a fast, same-process
UX guard for the common double-click case; the file lock is the real
cross-process guarantee and needs no help from this route.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import principal_id, require_scope

# check_runtime_update() makes a real network request (the same release
# listing a bare `setup-llama` resolves against) when no pin is set, so it is
# bounded like any other outbound call from a request handler rather than
# trusted to return promptly.
_CHECK_TIMEOUT_S = 15.0


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs

    @app.get("/api/runtime/check",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def runtime_check_ep():
        """Whether a different llama.cpp build is available for the backend
        actually installed on this box. Read-only; never provisions anything.
        See setup_llama.check_runtime_update() for the comparison rules
        (pin-aware, amd-rocm uses its fixed tag, never queries a release
        listing needlessly)."""
        from localm import setup_llama
        try:
            return await run_in_threadpool_bounded(
                setup_llama.check_runtime_update, timeout=_CHECK_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Checking the runtime timed out: {e}")

    @app.post("/api/runtime/update",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def runtime_update_ep(request: Request):
        """Re-provision the llama.cpp runtime for the currently installed
        backend, by running the EXISTING `localm setup-llama` entry point as a
        background job (--force so it actually replaces the build; --yes so
        the unattended subprocess never blocks on a prompt nothing will
        answer - see _apply_update.post_swap_command's docstring for the same
        reasoning applied to the full updater's runtime class).

        Refuses (409) when nothing is provisioned yet (there is nothing to
        UPDATE - that is initial setup, a different action) or when a runtime
        update is already running. A concurrent `localm update` re-provision
        or a bare terminal `setup-llama` is refused honestly by the job
        itself (setup-llama's own cross-process provisioning lock), not by
        this route - see the module docstring."""
        from localm import setup_llama
        backend = setup_llama.installed_backend()
        if not backend:
            raise HTTPException(
                409, "No llama.cpp runtime is installed yet - nothing to update. "
                "Run setup first.")
        if jobs.has_running("runtime-update"):
            raise HTTPException(409, "A runtime update is already running.")
        job = jobs.start_cli(
            "runtime-update",
            ["setup-llama", "--backend", backend, "--force", "--yes"],
            host_label="llama.cpp runtime update", owner=principal_id(request))
        return {"job_id": job.id}
