# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/live: the status bar's push channel.

One server-sent-event stream per tab carrying the hardware stats and the
activity list that GET /api/stats and GET /api/activity answer one poll at a
time. Frames are single-line JSON ``data:`` events:

  {"type": "stats", "data": {...}}     what /api/stats returns
  {"type": "activity", ...}            what /api/activity returns

A ``: keepalive`` comment goes out whenever nothing else has for
``web._KEEPALIVE_S`` seconds, so a client can tell a quiet server from a dead
connection.

The stats come from the one process-wide sampler (``localm.livefeed``); the
activity list is built per connection because its rows are filtered and
path-scrubbed for each caller. Authentication is ``_require_auth`` only, as for
the two routes this replaces; neither carries a capability scope.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import localm.plugins.gui.web as _web
from localm import livefeed, portmux
from localm.executor import get_plugin_executor
from localm.inference.http_server import _require_auth
from localm.plugins.gui.routes.jobs import activity_snapshot

FRESH_WAIT_S = 2.5     # longest a forced reading waits for the VRAM probe
FRESH_BUDGET_S = 3.5   # longest a forced sample takes in all; above FRESH_WAIT_S
AUTH_RECHECK_S = 5.0   # how often an open stream asks again whether its caller is still allowed
STOP_POLL_S = 1.0      # how often an open stream looks for a server stop


def _frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def make_sampler():
    """The feed's ``sample(fresh)``: ``sysstats.system_stats`` off the event
    loop, or ``sysstats.fresh_stats`` for a forced reading, answering from
    ``sysstats.cached_stats()`` when the probe thread does not return within
    the budget."""
    from localm import sysstats

    async def sample(fresh: bool) -> dict:
        loop = asyncio.get_running_loop()
        if fresh:
            fut = loop.run_in_executor(
                get_plugin_executor(), lambda: sysstats.fresh_stats(timeout=FRESH_WAIT_S))
            budget = FRESH_BUDGET_S
        else:
            fut = loop.run_in_executor(get_plugin_executor(), sysstats.system_stats)
            budget = sysstats.STATS_REPLY_BUDGET_S
        try:
            return await asyncio.wait_for(asyncio.shield(fut), budget)
        except TimeoutError:
            return sysstats.cached_stats()

    return sample


def register(app: FastAPI, ctx) -> None:
    feed = livefeed.LiveFeed(make_sampler(), interval_s=livefeed.INTERVAL_S,
                             follow_ups_s=livefeed.FOLLOW_UPS_S)
    livefeed.install(feed)
    app.state.live_feed = feed

    @app.get("/api/live", dependencies=[Depends(_require_auth)])
    async def gui_live(request: Request):
        """Stream hardware stats and activity to the caller as they change.

        Stats are pushed when a displayed value changes, at once after a model
        loads or unloads or a job starts or ends, and again shortly after for a
        driver that frees memory late. Activity is pushed when the caller's list
        of operations changes, and every ``livefeed.INTERVAL_S`` seconds while
        the list has a row (running or recently finished) so the ages the client
        shows, computed against the ``now`` of the latest frame, stay current.

        The stream ends when the server is asked to stop, and when the caller's
        credential stops being accepted (checked every ``AUTH_RECHECK_S``
        seconds): authentication at connect time alone would let a revoked key
        keep reading."""
        async def _stream():
            sub = feed.subscribe()
            loop = asyncio.get_running_loop()
            last_sig = None
            last_activity_at = 0.0
            last_sent = loop.time()
            last_auth = last_sent
            woke = True
            try:
                while True:
                    if portmux.stopping():
                        return
                    if loop.time() - last_auth >= AUTH_RECHECK_S:
                        last_auth = loop.time()
                        try:
                            await asyncio.get_running_loop().run_in_executor(
                                get_plugin_executor(), _require_auth, request)
                        except HTTPException:
                            return
                    if woke:
                        if feed.seq != sub.seen_seq and feed.latest is not None:
                            sub.seen_seq = feed.seq
                            last_sent = loop.time()
                            yield _frame({"type": "stats", "data": feed.latest})
                        payload = activity_snapshot(request, ctx.jobs)
                        sig = json.dumps(
                            {"operations": payload["operations"], "stalls": payload["stalls"]},
                            sort_keys=True, default=str)
                        now = loop.time()
                        listed = bool(payload["operations"])
                        if sig != last_sig or (
                                listed and now - last_activity_at >= livefeed.INTERVAL_S):
                            last_sig = sig
                            last_activity_at = now
                            last_sent = now
                            yield _frame({"type": "activity", **payload})
                    woke = await sub.wait(min(_web._KEEPALIVE_S, STOP_POLL_S))
                    if loop.time() - last_sent >= _web._KEEPALIVE_S:
                        last_sent = loop.time()
                        woke = True
                        yield ": keepalive\n\n"
            finally:
                sub.close()

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
