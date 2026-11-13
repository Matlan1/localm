# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pure-ASGI transport middleware. Three are added after every
BaseHTTPMiddleware handler so they wrap them all: the request-body cap, the
disconnect signal and, outermost, the hang alarm's in-flight bookkeeping. Two
are added before every other middleware, so they are the innermost: the request
privacy scope, then the client-left answer around it."""

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


class RequestPrivacyMiddleware:
    """Run a request whose session mode is privacy while the server's is not
    (``http_server.request_session_mode``: an owner caller's
    ``X-Localm-Session-Mode``, or its ``X-Client-Id`` surface) inside
    ``audit.request_privacy``, from the route handler to the last byte of the
    response. A mode that cannot be resolved counts as privacy and is logged
    as a warning. The mode is resolved off the event loop, and only for a
    request that sends ``X-Localm-Session-Mode`` or names the chat or coder
    surface in ``X-Client-Id``; any other request has the server's mode."""

    def __init__(self, app) -> None:
        self.app = app

    @staticmethod
    def _may_narrow(scope) -> bool:
        from localm.audit import CLIENT_ID_HEADER, SESSION_MODE_HEADER, _CALLER_SURFACES
        mode_name = SESSION_MODE_HEADER.lower().encode()
        client_name = CLIENT_ID_HEADER.lower().encode()
        for name, value in scope.get("headers") or ():
            if name == mode_name:
                return True
            if name == client_name and (
                    value.decode("latin-1").strip().lower() in _CALLER_SURFACES):
                return True
        return False

    @staticmethod
    def _is_privacy(scope) -> bool:
        from localm.audit import SessionMode
        try:
            from starlette.requests import Request
            return _hs.request_session_mode(Request(scope)) == SessionMode.PRIVACY
        except Exception as e:   # noqa: BLE001
            from localm.debuglog import logger as _dbg
            _dbg.warning("could not resolve the session mode of a request (%s: %s); "
                         "treating it as privacy mode", type(e).__name__, e)
            return True

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from fastapi.concurrency import run_in_threadpool
        from localm.audit import request_privacy
        private = (self._may_narrow(scope)
                   and await run_in_threadpool(self._is_privacy, scope))
        with request_privacy(private):
            await self.app(scope, receive, send)


def add_request_privacy(app: FastAPI) -> None:
    """Add :class:`RequestPrivacyMiddleware`. Call before any other middleware
    is added, so it runs in the task that runs the route handler and sends the
    response body."""
    app.add_middleware(RequestPrivacyMiddleware)


def add_client_left_answer(app: FastAPI) -> None:
    """Add :class:`ClientLeftMiddleware`. Call before any other middleware is
    added, except :func:`add_request_privacy`: it must sit inside every
    BaseHTTPMiddleware handler, which turns a
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
    # Outermost of the three: in-flight/progress bookkeeping for the hang
    # alarm's starvation detector. Pure ASGI and content-free (counts and
    # clocks only); sits outside every layer that can hold a request, so a
    # request wedged in ANY inner layer still shows as in flight. Only the
    # metrics counter and the debug request log, both pass-through, wrap it.
    from localm.inference._hang_alarm import RequestProgressMiddleware
    app.add_middleware(RequestProgressMiddleware)
