# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI llama.cpp runtime routes: check what is provisioned, and provision it - the same space as /api/update/* (localm/inference/routes/admin.py) but for the NATIVE runtime `localm setup-llama` provisions, not the Python source tree."""

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
        """Whether a different llama.cpp build is available for the backend actually installed on this box."""
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
        """Provision the llama.cpp runtime, by running the EXISTING `localm setup-llama` entry point as a background job (--force so it actually replaces the build; --yes so the unattended subprocess never blocks on a prompt nothing will answer - see _apply_update.post_swap_command's docstring for the same rea..."""
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
        rollback = bool(req.rollback)
        if rollback and tag is not None:
            raise HTTPException(
                400, "'tag' and 'rollback' both choose a build; send only "
                     "one. Rollback goes to the previous recorded build; "
                     "tag names one.")

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
        if rollback:
            # "auto" is not a real backend to hold history for - it is the
            # CLI's own hardware-detect default - so it is treated as "not
            # named" here too, exactly like setup_llama._apply_version_request
            # treats an explicit `--backend auto --rollback`: fall through to
            # whatever is installed rather than looking up history for a
            # backend called "auto" (which can never exist).
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
