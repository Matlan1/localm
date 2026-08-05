# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background-job routes: discover what is running, stream one, cancel one.

  GET  /api/activity              - what this server is doing (no id needed)
  GET  /api/jobs/{id}/events      - stream one job (SSE)
  POST /api/jobs/{id}/cancel      - cancel one job

KERNEL routes, despite the path this module still sits at. They are registered
from ``attach_engine``, not ``attach_gui``, so a headless ``localm serve``
answers them too (ADR-0008); the package location is historical and moving it
would churn six test modules for no behaviour change. ``register()`` therefore
takes the ``JobManager`` directly rather than the GUI's ``ctx`` namespace,
unlike the sibling route groups here.

/api/activity is the only one that does not need an id the caller already
holds, which is the whole reason it exists: a job id is handed out once, in the
body of the POST that started the job, so a second client or a reloaded tab had
no way to reach state the server was recording all along.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from localm.inference.http_server import _require_auth, job_owner_ok, require_owner


def register(app: FastAPI, jobs) -> None:
    """Mount the job SSE + cancel routes over *jobs* (a JobManager).

    Takes the manager DIRECTLY rather than the GUI's ``ctx`` namespace, unlike
    the sibling route groups in this package: since ADR-0008 these routes are
    registered from ``attach_engine`` so they exist on a headless ``localm
    serve`` too, and the kernel has no ctx to hand over.
    """

    def _resolve_job(job_id: str):
        """require_owner() resolver: the GUI-tracked job named by the job_id
        path param (JobManager, NOT the scheduled-jobs plugin's JobStore)."""
        job = jobs.get(job_id)
        return job, (job.owner if job else None), f"No such job: {job_id}"

    # FastAPI dependency: fetch + enforce ownership, Depends()-injectable so
    # job_events/job_cancel cannot omit the check by construction (LM-DA-020).
    # Same 404 whether the job is missing or belongs to someone else, so a
    # non-owner cannot even confirm the (unguessable) id exists (KEY-SCOPE-2).
    owned_job = require_owner(_resolve_job)

    @app.get("/api/activity", dependencies=[Depends(_require_auth)])
    async def activity(request: Request):
        """What this server is doing right now, plus what it recently finished.

        THE POINT (ADR-0008): every other way to reach a job needs an id the
        caller already holds, and that id is handed out exactly once, in the
        body of the POST that started the job. So a second browser tab, a
        second device, or the same tab after a reload could not discover that a
        model pull was running even though the server knew - the state was
        recorded and simply unreachable. This is the one route that answers
        "what is happening" without being told what to look for.

        NOT under /api/jobs. That prefix already belongs to the scheduled-jobs
        plugin, whose jobs are recurring TASK DEFINITIONS in a persisted store,
        a different concept in a different id space under a different gate
        (require_scope("jobs")). Adding an in-flight listing there would put two
        meanings on one prefix.

        No capability scope, deliberately: this returns a strict subset of what
        the caller may already stream by id, so gating it more tightly than
        /api/jobs/{id}/events would be theatre. Ownership is enforced with the
        same job_owner_ok the events and cancel routes use.

        MIND THE DEFAULT CONFIGURATION. With no owner key and no keystore -
        which is how localm runs out of the box - principal_id() returns None,
        so every job is unowned and job_owner_ok admits any authenticated
        caller. That is the correct behaviour for a single-owner local server
        and it is NOT a filter failure, but it does mean this list is
        per-principal only on a KEYED server. Nothing downstream may assume
        otherwise.

        ``now`` is this SERVER's clock at the moment of the reply, and it is
        here so a client can render an age without inventing one. created_at
        and finished_at are server epoch seconds; a client that computed
        "running for 12 minutes" from its OWN clock would be wrong by whatever
        the two clocks disagree by, and a phone or a drifted box disagrees by
        real amounts. Since created_at exists precisely so a user can tell a
        six-second operation from a six-hour one, durations WILL be rendered,
        so the reference clock has to travel with them. Compute ages as
        ``now - created_at``, never against a local clock.
        """
        return {"now": time.time(),
                "operations": jobs.snapshot(
                    visible=lambda owner: job_owner_ok(request, owner))}

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(_require_auth)])
    async def job_events(job=Depends(owned_job)):
        async def _stream():
            # Each connection gets its own subscriber queue (fed the full history
            # plus every event pushed from here on) so concurrent viewers of the
            # same job each see the complete stream, instead of racing to drain
            # one shared queue between them.
            # Imported lazily, not at module scope: these routes are now mounted
            # from attach_engine, and a top-level import would drag the whole GUI
            # web module into a headless `localm serve` that never attaches a GUI.
            import localm.plugins.gui.web as _web
            q = job.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=_web._KEEPALIVE_S)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") == "end":
                        return
            finally:
                job.unsubscribe(q)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(_require_auth)])
    async def job_cancel(job=Depends(owned_job)):
        job.cancel()
        return {"status": "cancelling"}
