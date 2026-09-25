# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``create_app()`` mounts: the api-mode landing redirect, the kernel route
groups (localm/inference/routes/*.py) and, last, the plugin engine."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from localm.inference.app_assembly.context import AppContext


def add_api_landing(app: FastAPI) -> None:
    """Register the api-mode GET / redirect to /docs."""
    # api-mode landing: a bare `localm serve` has no GUI shell, so GET / would
    # 404. Redirect it to the auto-generated API docs. Only on the api path so
    # it never collides with the GUI's own "/" handler + StaticFiles catch-all.
    @app.get("/", include_in_schema=False)
    async def _api_root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)


def mount_route_groups(app: FastAPI, ctx: AppContext) -> None:
    """Register the kernel route groups."""
    # Route groups (extracted to localm/inference/routes/*.py).
    # The engine + inference semaphore are module globals read live by the route
    # modules (via `import localm.inference.http_server as _hs`), so a model swap
    # that reassigns them is seen there. Only the session-scoped objects travel
    # to the route groups, on ctx (an AppContext). Registration
    # order is irrelevant: FastAPI matches exact path templates by method, and no
    # group has a same-method literal-vs-param path collision.
    from localm.inference.routes import models as _routes_models
    _routes_models.register(app, ctx)
    from localm.inference.routes import system as _routes_system
    _routes_system.register(app, ctx)
    from localm.inference.routes import session as _routes_session
    _routes_session.register(app, ctx)
    from localm.inference.routes import config as _routes_config
    _routes_config.register(app, ctx)
    from localm.inference.routes import keys as _routes_keys
    _routes_keys.register(app, ctx)
    from localm.inference.routes import gpu as _routes_gpu
    _routes_gpu.register(app, ctx)
    from localm.inference.routes import peer_routing as _routes_peer_routing
    _routes_peer_routing.register(app, ctx)
    from localm.inference.routes import admin as _routes_admin
    _routes_admin.register(app, ctx)
    from localm.inference.routes import chat as _routes_chat
    _routes_chat.register(app, ctx)


def attach_plugins(app: FastAPI, engine) -> None:
    """Load the enabled plugins and mount the plugin management API. A failure
    here never stops the server: it is logged as a WARNING plus the traceback,
    and ``app.state.plugin_engine_error`` records it."""
    # Plugin engine: load enabled plugins + management API.
    # Wrapped so a plugin-engine failure can never stop the server starting.
    try:
        from localm.plugins.engine import attach_engine
        attach_engine(app, engine)
    except Exception as e:
        # Server must still start, but make the loss visible: WARNING (not a buried
        # debug line) plus a sentinel so plugin_manager being unset is diagnosable.
        from localm.debuglog import logger as _dbg
        _dbg.warning("plugins unavailable: %s", e)
        _dbg.exception("plugin engine attach failed")
        app.state.plugin_engine_error = str(e)
