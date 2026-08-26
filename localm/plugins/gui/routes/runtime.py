# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI llama.cpp runtime routes: check what is provisioned, and provision it -
the same space as /api/update/* (localm/inference/routes/admin.py) but for the
NATIVE runtime `localm setup-llama` provisions, not the Python source tree.

  GET  /api/runtime/check   -> read-only: is a different build available for
                               the currently installed backend, per
                               setup_llama.check_runtime_update().
  POST /api/runtime/update  -> dispatch `localm setup-llama --backend <b>
                               [--tag <t> | --rollback] --force --yes` as a
                               background JOB (a real download + native
                               load-test can take a while, same shape as
                               /api/comfy/update); returns {"job_id"} to stream.

The POST takes a backend, a tag and a rollback flag, so a GUI-only user can
install a runtime in the first place, switch backend, choose a build, and go
back to the build that worked before a bad upstream release. It does not refuse
when nothing is installed.

Nothing about provisioning is decided here. The route validates its inputs and
hands them to the EXISTING command, so the backend-resolution, load-test,
fallback and pin rules stay in setup_llama.py - notably
_provision_with_fallback, which keeps a backend that cannot load on this
machine from being left installed.

Safety: setup-llama's own _provisioning_lock (localm/setup_llama.py) is what
serializes this against a concurrent `localm update` re-provision or a user's
own terminal `setup-llama` run. jobs.has_running() below is a fast,
same-process guard for the double-click case; the file lock is the
cross-process guarantee. The job KIND is one string for every variant, so an
install, a switch and an update cannot be started concurrently through this
route either.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.http_server import principal_id, require_scope
from localm.plugins.gui.web import RuntimeSetupRequest

# check_runtime_update() makes a real network request (the same release listing
# a bare `setup-llama` resolves against) when no pin is set, so the call is
# bounded.
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

        The response carries no backend list: the set the POST accepts is the
        constant setup_llama.BACKENDS, which the Settings card carries in its
        own markup."""
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
        a prompt nothing will answer).

        The backend is the one asked for, else the one already installed, else
        'auto', so an empty body gets the same hardware detection a bare
        `localm setup-llama` performs. `rollback` pins and installs the previous
        build recorded for that backend instead of `tag`'s explicit one; the two
        are mutually exclusive, same as the CLI.

        Refuses (409) only when a runtime job is already running. A concurrent
        `localm update` re-provision or a bare terminal `setup-llama` is refused
        by the job itself, via setup-llama's own cross-process provisioning
        lock."""
        from localm import setup_llama
        req = req or RuntimeSetupRequest()

        # Both inputs are validated here against setup_llama's own definitions,
        # so a bad value is a 400 rather than a job that starts and then fails.
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
        rollback = bool(req.rollback)
        if rollback and tag is not None:
            raise HTTPException(
                400, "'tag' and 'rollback' both choose a build; send only "
                     "one. Rollback goes to the previous recorded build; "
                     "tag names one.")

        # installed_backend() returns None for TWO states - nothing provisioned,
        # and provisioned by an install too old to have written the marker (or a
        # hand-placed build) - which are indistinguishable here, so the fallback
        # to "auto" re-detects hardware in both cases and can land on a
        # different backend than an unmarked build on disk. An EXPLICIT backend
        # always wins over both.
        installed = setup_llama.installed_backend()
        if rollback:
            # "auto" is the CLI's hardware-detect default, not a real backend to
            # hold history for, so it is treated as "not named" here and falls
            # through to whatever is installed - the same as
            # setup_llama._apply_version_request does for
            # `--backend auto --rollback`.
            backend = wanted if wanted and wanted != "auto" else installed
            if not backend:
                raise HTTPException(
                    400, "Rollback needs to know which backend - nothing is "
                         "recorded as installed on this machine. Name one "
                         "explicitly.")
            if backend == "amd-rocm":
                raise HTTPException(
                    400, "The amd-rocm backend cannot be rolled back: its "
                         "build is fixed by the localm release, not chosen "
                         "from upstream llama.cpp releases.")
            prev = setup_llama.previous_tag(backend)
            if not prev:
                raise HTTPException(
                    400, f"No earlier llama.cpp build is recorded for the "
                         f"{backend} backend, so there is nothing to roll "
                         "back to.")
        else:
            backend = wanted or installed or "auto"
        if jobs.has_running("runtime-update"):
            raise HTTPException(409, "A runtime update is already running.")

        args = ["setup-llama", "--backend", backend]
        if rollback:
            args += ["--rollback"]
        elif tag:
            args += ["--tag", tag]
        args += ["--force", "--yes"]
        job = jobs.start_cli(
            "runtime-update", args,
            host_label=("llama.cpp runtime update" if installed
                        else "llama.cpp runtime setup"),
            owner=principal_id(request))
        return {"job_id": job.id}
