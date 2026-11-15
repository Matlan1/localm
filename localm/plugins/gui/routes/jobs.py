# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background-job routes: discover what is running, stream one, cancel one.

  GET  /api/activity              - what this server is doing (no id needed)
  GET  /api/jobs/{id}/events      - stream one job (SSE)
  POST /api/jobs/{id}/cancel      - cancel one job

KERNEL routes, despite the path this module sits at. They are registered from
``attach_engine``, not ``attach_gui``, so a headless ``localm serve`` answers
them too. ``register()`` therefore takes the ``JobManager`` directly rather
than the GUI's ``ctx`` namespace, unlike the sibling route groups here.

/api/activity is the only one that does not need an id the caller already
holds: a job id is handed out once, in the body of the POST that started the
job, so a second client or a reloaded tab otherwise has no way to reach state
the server is recording.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import Depends, FastAPI, Request

from localm.inference.http_server import (
    _require_auth, job_owner_ok, path_formatter, require_owner,
)
from localm.pathscrub import map_strings
from localm.streaming_response import StreamingResponse


def activity_snapshot(request: Request, jobs) -> dict:
    """The body of GET /api/activity for *request*'s caller: the operations that
    caller may see (path-scrubbed for a caller other than the local operator),
    the server's recent stalls and the server's clock."""
    rows = jobs.snapshot(visible=lambda owner: job_owner_ok(request, owner))
    show = path_formatter(request)
    if show is not str:
        rows = map_strings(rows, show)
    from localm.inference._stall_watch import recent_stalls
    return {"now": time.time(), "operations": rows, "stalls": recent_stalls()}


def register(app: FastAPI, jobs) -> None:
    """Mount the job SSE + cancel routes over *jobs* (a JobManager).

    Takes the manager DIRECTLY rather than the GUI's ``ctx`` namespace, unlike
    the sibling route groups in this package: these routes are registered from
    ``attach_engine`` so they exist on a headless ``localm serve`` too, and the
    kernel has no ctx to hand over.
    """

    def _resolve_job(job_id: str):
        """require_owner() resolver: the GUI-tracked job named by the job_id
        path param (JobManager, NOT the scheduled-jobs plugin's JobStore)."""
        job = jobs.get(job_id)
        return job, (job.owner if job else None), f"No such job: {job_id}"

    # FastAPI dependency: fetch + enforce ownership, Depends()-injectable so
    # job_events/job_cancel cannot omit the check by construction. Same 404
    # whether the job is missing or belongs to someone else, so a non-owner
    # cannot even confirm the (unguessable) id exists.
    owned_job = require_owner(_resolve_job)

    @app.get("/api/activity", dependencies=[Depends(_require_auth)])
    async def activity(request: Request):
        """What this server is doing right now, plus what it recently finished.

        Returns ``{now, operations, stalls}``. ``operations`` lists the jobs
        the caller may see (running and recently finished), without needing a
        job id. It needs no capability scope beyond authentication, and shows
        only jobs the caller owns. On a server with no owner key and no
        keystore (the default), jobs have no owner and every authenticated
        caller sees all of them.

        ``now`` is the server's clock in epoch seconds at the time of the reply,
        like ``created_at`` and ``finished_at`` in each row; compute an age as
        ``now - created_at``, not against the client's own clock.

        For a caller other than the local operator every string in a row is
        path-scrubbed. ``stalls`` lists the server's recent stalls, oldest
        first, and is empty when stall monitoring is not running.
        """
        return activity_snapshot(request, jobs)

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(_require_auth)])
    async def job_events(request: Request, job=Depends(owned_job)):
        """Stream one job's events as server-sent events. Every connection
        receives the job's full event history, then each new event. For a caller
        other than the local operator every string in an event is
        path-scrubbed."""
        show = path_formatter(request)

        async def _stream():
            # Each connection gets its own subscriber queue (fed the full history
            # plus every event pushed from here on) so concurrent viewers of the
            # same job each see the complete stream, instead of racing to drain
            # one shared queue between them.
            # Imported lazily, not at module scope: these routes are mounted from
            # attach_engine, and a top-level import would drag the whole GUI web
            # module into a headless `localm serve` that never attaches a GUI.
            import localm.plugins.gui.web as _web
            q = job.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=_web._KEEPALIVE_S)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    shown = event if show is str else map_strings(event, show)
                    yield f"data: {json.dumps(shown, ensure_ascii=False)}\n\n"
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
        if job.finished_at is not None:
            return {"status": "not_running", "state": job.status}
        job.cancel()
        return {"status": "cancelling"}
