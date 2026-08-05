# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI job routes: the SSE event stream and cancel for background jobs.

Extracted verbatim from attach_gui(); behavior unchanged. The background
``JobManager`` is unpacked from the register ``ctx`` into ``jobs`` once at the top
of register(), so each handler body is identical to the original. These stay in
the kernel GUI so every plugin's jobs stream through the one shared manager.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from localm.inference.http_server import _require_auth, require_owner


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
