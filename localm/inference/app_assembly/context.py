# SPDX-License-Identifier: AGPL-3.0-or-later
"""The per-app context ``create_app()`` builds: the ``AppContext`` the route
groups receive, and the ``app.state`` fields assembly publishes before any
middleware, route or plugin reads them."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI

from localm.inference.chat_pipeline import ChatPipeline

if TYPE_CHECKING:
    from localm.audit import AuditLogT, MarkdownTranscript, SessionMode


@dataclass(frozen=True)
class AppContext:
    """The session-scoped objects every route group receives as ``ctx`` in
    ``register(app, ctx)`` (localm/inference/routes/*.py): the server's audit
    log, its transcript (only in ``full`` mode, else None) and the session mode
    both were opened for. The engine and the inference semaphores are not
    here: they are http_server module globals the route groups read live, so a
    model swap that rebinds them is seen there."""

    audit: AuditLogT
    transcript: Optional[MarkdownTranscript]
    mode: SessionMode


def init_app_state(app: FastAPI) -> None:
    """Publish the per-app state that middleware, routes and plugins read: the
    chat pipeline, the shell token and the CSRF secret."""
    # Chat-pipeline hooks: plugins register inlet/stream/outlet transforms that
    # run on every /v1/chat/completions turn. Created here so it exists before
    # plugins load (attach_plugins, the last assembly phase) and stays reachable as
    # request.app.state.chat_pipeline. A pipeline with no hooks is a no-op.
    app.state.chat_pipeline = ChatPipeline()

    # Per-process "shell token": in open mode the management routes require
    # this token, which the loopback GUI shell injects into the SPA (web.py
    # _gui_index). It gates the no-Origin local-client path that bearer auth /
    # the Origin guard alone do not cover. Per-process so it dies on restart;
    # never persisted.
    app.state.shell_token = secrets.token_urlsafe(32)

    # Per-process CSRF secret. The CSRF token is a deterministic HMAC of the session
    # id (below), so it is present exactly when the session is and CANNOT desync
    # (the old design used a SEPARATE readable cookie a client reset could clear
    # while the HttpOnly session survived, 403-ing every write). Per-process so it
    # dies on restart (client re-fetches from /api/session); never persisted.
    app.state.csrf_secret = secrets.token_urlsafe(32)
