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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import localm.plugins.gui.web as _web
from localm.inference.http_server import _require_auth, job_owner_ok


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(_require_auth)])
    async def job_events(job_id: str, request: Request):
        job = jobs.get(job_id)
        # A job is reachable only by the key that created it (or an admin/owner).
        # Return 404 - not 403 - on an ownership mismatch so a non-owner cannot even
        # confirm the (unguessable) id exists (KEY-SCOPE-2).
        if job is None or not job_owner_ok(request, job.owner):
            raise HTTPException(404, f"No such job: {job_id}")

        async def _stream():
            # Each connection gets its own subscriber queue (fed the full history
            # plus every event pushed from here on) so concurrent viewers of the
            # same job each see the complete stream, instead of racing to drain
            # one shared queue between them.
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
    async def job_cancel(job_id: str, request: Request):
        job = jobs.get(job_id)
        # Same owner-binding as the events stream: only the creating key (or an
        # admin/owner) may cancel; others get an indistinguishable 404.
        if job is None or not job_owner_ok(request, job.owner):
            raise HTTPException(404, f"No such job: {job_id}")
        job.cancel()
        return {"status": "cancelling"}
