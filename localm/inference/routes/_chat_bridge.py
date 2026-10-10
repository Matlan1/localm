# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calling the registered ``/v1/chat/completions`` endpoint from another wire
format's route (Ollama, Anthropic Messages, OpenAI Responses), with a derived
request that keeps the caller's headers, app state and disconnect signal."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute


class EndpointMissing(LookupError):
    """The endpoint is not mounted on this app."""


def derived_request(request: Request, path: str, body: bytes) -> Request:
    """*request* re-aimed at *path* with *body* as its whole body. Headers (so
    auth and principal), app and state are the original's; once the body has
    been read, ``receive`` is the original's, so disconnects are still seen."""
    scope = dict(request.scope)
    scope["path"] = path
    scope["raw_path"] = path.encode("ascii")
    scope["path_params"] = {}
    scope["headers"] = [
        (k, v) for k, v in scope["headers"]
        if k.lower() not in (b"content-length", b"content-type")
    ] + [(b"content-type", b"application/json"),
         (b"content-length", str(len(body)).encode("ascii"))]
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await request.receive()

    return Request(scope, receive)


def chat_endpoint(app: FastAPI, path: str):
    """The endpoint function registered for ``POST`` *path*. Raises
    :class:`EndpointMissing` when none is."""
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.path == path and "POST" in (route.methods or ()):
            return route.endpoint
    raise EndpointMissing(f"{path} is not mounted on this server")


async def collect_body(body_iterator) -> bytes:
    """Every chunk of a streaming response body, joined."""
    parts = []
    async for part in body_iterator:
        parts.append(part if isinstance(part, bytes) else part.encode("utf-8"))
    return b"".join(parts)


def validation_text(errors: Sequence[Any]) -> str:
    """The first pydantic validation error as ``<field path>: <message>``."""
    if not errors:
        return "request is invalid"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = first.get("msg", "is invalid")
    return f"{loc}: {msg}" if loc else msg
