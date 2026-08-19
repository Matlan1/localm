# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI llama.cpp runtime routes: check what is provisioned, and provision it -
the same space as /api/update/* (localm/inference/routes/admin.py) but for the
NATIVE runtime `localm setup-llama` provisions, not the Python source tree.

  GET  /api/runtime/check   -> read-only: is a different build available for
                               the currently installed backend, per
                               setup_llama.check_runtime_update().
  POST /api/runtime/update  -> dispatch `localm setup-llama --backend <b>
                               [--tag <t>] --force --yes` as a background JOB
                               (a real download + native load-test can take a
                               while, same shape as /api/comfy/update); returns
                               {"job_id"} to stream.

WHY THE POST TAKES A BACKEND AND A TAG. It used to hardcode the INSTALLED
backend and refuse (409) when nothing was installed, which made three things
CLI-only for a user who has only the GUI: installing a runtime in the first
place, switching backend, and choosing a build. `localm doctor` and the GUI's
own Settings copy both told such a user to run a command they cannot reach.
Dropping that 409 also removes a trap of its own: a box whose runtime failed to
provision had NO route back, so the one surface still working refused the one
action that would fix it.

Why a background job and not a blocking call like /api/update/apply: this is
closer in shape to /api/comfy/update (a multi-step operation worth streaming
progress for) than to the code-tree swap, which is comparatively quick.

Nothing about provisioning is decided here. The route validates its two inputs
and hands them to the EXISTING command, so the backend-resolution, load-test,
fallback and pin rules stay in setup_llama.py with one implementation - notably
_provision_with_fallback, which is what keeps a backend that cannot load on
this machine from being left installed.

Safety: setup-llama's own _provisioning_lock (localm/setup_llama.py) is what
actually serializes this against a concurrent `localm update` re-provision or
a user's own terminal `setup-llama` run - a cross-process race this route
introduces as a genuinely new, second trigger onto the same directory (see
diff-review-discipline.md item 26, the identical hazard /api/comfy/update hit
before its own lock existed). jobs.has_running() below is a fast, same-process
UX guard for the common double-click case; the file lock is the real
cross-process guarantee and needs no help from this route. The job KIND stays
one string for every variant, so an install, a switch and an update cannot be
started concurrently through this route either.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import principal_id, require_scope
from localm.plugins.gui.web import RuntimeSetupRequest

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
        listing needlessly).

        Unchanged by the POST below gaining a backend/tag body, deliberately:
        the value set that POST accepts is a CONSTANT (setup_llama.BACKENDS),
        so the Settings card carries the option list in its own markup and a
        test binds that markup to the constant, rather than the card being
        unable to offer a backend until a check has succeeded."""
        from localm import setup_llama
        try:
            return await run_in_threadpool_bounded(
                setup_llama.check_runtime_update, timeout=_CHECK_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Checking the runtime timed out: {e}")

    @app.post("/api/runtime/update",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def runtime_update_ep(request: Request,
                                req: Optional[RuntimeSetupRequest] = None):
        """Provision the llama.cpp runtime, by running the EXISTING `localm
        setup-llama` entry point as a background job (--force so it actually
        replaces the build; --yes so the unattended subprocess never blocks on
        a prompt nothing will answer - see _apply_update.post_swap_command's
        docstring for the same reasoning applied to the full updater's runtime
        class).

        The backend is the one asked for, else the one already installed, else
        'auto' - so an empty body still means exactly what it meant before this
        route took one, and a box with nothing provisioned gets the same
        hardware detection a bare `localm setup-llama` performs.

        Refuses (409) only when a runtime job is already running. A concurrent
        `localm update` re-provision or a bare terminal `setup-llama` is
        refused honestly by the job itself (setup-llama's own cross-process
        provisioning lock), not by this route - see the module docstring."""
        from localm import setup_llama
        req = req or RuntimeSetupRequest()

        # Both inputs are validated HERE, against setup_llama's own definitions,
        # so a bad value is a 400 the caller can read rather than a job that
        # starts and then fails - and so the answer cannot differ from the one
        # the command line gives for the same string.
        wanted = (req.backend or "").strip().lower() or None
        if wanted is not None and wanted not in setup_llama.BACKENDS:
            raise HTTPException(
                400, f"Unknown backend {req.backend!r}. One of: "
                     f"{', '.join(setup_llama.BACKENDS)}.")
        tag = (req.tag or "").strip() or None
        if tag is not None and not setup_llama.is_safe_tag(tag):
            raise HTTPException(
                400, f"{req.tag!r} is not a usable release tag. "
                     f"{setup_llama.TAG_HELP}")

        # installed_backend() returns None for TWO different states - nothing
        # provisioned, and provisioned by an install too old to have written the
        # marker (or a hand-placed build). They are indistinguishable here, and
        # the fallback to "auto" therefore re-detects hardware on the second one
        # too, which can land on a different backend than the unmarked build on
        # disk. That is deliberate rather than overlooked: it is exactly what a
        # bare `localm setup-llama` does on the same box, the card has already
        # told the user nothing is recorded as installed, and there is no
        # recorded choice to override. An EXPLICIT backend always wins over both.
        installed = setup_llama.installed_backend()
        backend = wanted or installed or "auto"
        if jobs.has_running("runtime-update"):
            raise HTTPException(409, "A runtime update is already running.")

        args = ["setup-llama", "--backend", backend]
        if tag:
            args += ["--tag", tag]
        args += ["--force", "--yes"]
        job = jobs.start_cli(
            "runtime-update", args,
            host_label=("llama.cpp runtime update" if installed
                        else "llama.cpp runtime setup"),
            owner=principal_id(request))
        return {"job_id": job.id}
