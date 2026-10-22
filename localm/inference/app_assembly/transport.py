# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pure-ASGI transport middleware. Three are added after every
BaseHTTPMiddleware handler so they wrap them all: the request-body cap, the
disconnect signal and, outermost, the hang alarm's in-flight bookkeeping. One,
the client-left answer, is added before every other middleware, so it is the
innermost."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

import localm.inference.http_server as _hs


class ClientLeftMiddleware:
    """Answer 499 with an empty body for a request whose handler raised
    ``asyncio.CancelledError`` itself, before sending any response, after the
    client disconnected (the poll ``_DisconnectSignalMiddleware`` publishes says
    so). Logs one DEBUG line for it.

    Re-raised unchanged: a cancellation delivered to the task (server shutdown,
    a cancel scope), one raised after a response started, one while the client
    is still connected, and one on a request with no disconnect poll."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def _send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if started or task is None or task.cancelling():
                raise
            poll = scope.get(_hs._DISCONNECT_POLL_KEY)
            if poll is None or not await poll():
                raise
            from localm.debuglog import logger as _dbg
            _dbg.debug("%s %s: the client left before the response; answered 499",
                       scope.get("method"), scope.get("path"))
            await send({"type": "http.response.start", "status": 499,
                        "headers": [(b"content-length", b"0")]})
            await send({"type": "http.response.body", "body": b""})


def add_client_left_answer(app: FastAPI) -> None:
    """Add :class:`ClientLeftMiddleware`. Call before any other middleware is
    added: it must sit inside every BaseHTTPMiddleware handler, which turns a
    handler's CancelledError into ``RuntimeError("No response returned.")``."""
    app.add_middleware(ClientLeftMiddleware)


def add_transport_middleware(app: FastAPI) -> None:
    """Add the three, innermost first."""
    # Outside every BaseHTTPMiddleware handler added before it (none of which touch the
    # body), so it sees the raw ASGI receive() for its body-size accounting. The
    # _DisconnectSignalMiddleware added right after passes receive() through
    # untouched, so this still gets the raw stream.
    app.add_middleware(_hs._BodyStreamCapMiddleware)
    # Added LAST (== outermost) so its disconnect poll is bound to the raw receive,
    # OUTSIDE the BaseHTTPMiddleware handlers that otherwise mask http.disconnect
    # from the non-streaming inference path (see the class + _generate_full).
    app.add_middleware(_hs._DisconnectSignalMiddleware)
    # Outermost of all: in-flight/progress bookkeeping for the hang alarm's
    # starvation detector. Pure ASGI and content-free (counts and
    # clocks only); sits outside everything so a request wedged in ANY inner
    # layer still shows as in flight.
    from localm.inference._hang_alarm import RequestProgressMiddleware
    app.add_middleware(RequestProgressMiddleware)
