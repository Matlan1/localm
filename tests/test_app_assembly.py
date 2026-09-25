# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app-assembly seams of ``create_app()`` (localm/inference/app_assembly/).

tests/test_create_app_characterization.py pins what the assembled app is: its
routes, its registration-order middleware list, handlers, state and globals.
This file pins what the split of ``create_app()`` into assembly steps has to
keep true on top of that:

* the EFFECTIVE middleware order, meaning the ASGI chain Starlette builds from
  the registrations, and the request-level consequences of that order;
* the transport middleware (body cap, disconnect signal, request progress)
  wraps every BaseHTTPMiddleware handler and every route;
* every route group receives the same frozen ``AppContext``, carrying the
  session audit ``http_server`` publishes;
* assembly code reads the names ``http_server`` defines through that module,
  so a monkeypatch on ``localm.inference.http_server`` still reaches it;
* the plugin engine attaches after every kernel route group;
* ``create_app()`` stays a short, readable boot order.
"""

import dataclasses
import inspect
import logging
import re

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference.http_server import create_app

# --------------------------------------------------------------------------- #
# Effective middleware order
# --------------------------------------------------------------------------- #

# Outermost first: the chain a request actually passes through, between
# Starlette's own ServerErrorMiddleware (always outermost) and
# ExceptionMiddleware (always right inside the user middleware).
_EFFECTIVE_CHAIN = (
    ("RequestProgressMiddleware", None),
    ("_DisconnectSignalMiddleware", None),
    ("_BodyStreamCapMiddleware", None),
    ("BaseHTTPMiddleware", "_docs_loopback_only"),
    ("BaseHTTPMiddleware", "_security_headers"),
    ("BaseHTTPMiddleware", "_origin_guard"),
    ("CORSMiddleware", None),
)


def _effective_chain(app) -> tuple:
    """The built ASGI stack from ServerErrorMiddleware down to
    ExceptionMiddleware, both excluded, as ``(class name, dispatch name)``.
    Follows each layer's inner app (``app``, or ``_app`` for
    _BodyStreamCapMiddleware)."""
    node = app.build_middleware_stack()
    names = []
    while node is not None:
        dispatch = getattr(node, "dispatch_func", None)
        names.append((type(node).__name__, getattr(dispatch, "__name__", None)))
        if names[-1][0] == "ExceptionMiddleware":
            break
        inner = getattr(node, "app", None)
        node = inner if inner is not None else getattr(node, "_app", None)
    assert names[0] == ("ServerErrorMiddleware", None), names
    assert names[-1] == ("ExceptionMiddleware", None), names
    return tuple(names[1:-1])


@pytest.mark.parametrize("debug", [False, True], ids=["debug-off", "debug-on"])
def test_effective_middleware_chain(monkeypatch, debug):
    if debug:
        monkeypatch.setenv("LOCALM_DEBUG", "1")
    else:
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    expected = _EFFECTIVE_CHAIN + (
        (("BaseHTTPMiddleware", "_log_requests"),) if debug else ())
    assert _effective_chain(create_app(None)) == expected


def test_transport_middleware_is_the_outermost_three():
    """The three pure-ASGI layers wrap every BaseHTTPMiddleware handler, in
    this order: request progress outermost (a request wedged in any inner
    layer still counts as in flight), then the disconnect signal (bound to the
    raw receive, which BaseHTTPMiddleware would mask), then the body cap (sees
    the raw body stream before anything buffers it)."""
    chain = _effective_chain(create_app(None))
    first_base_http = next(i for i, (cls, _) in enumerate(chain)
                           if cls == "BaseHTTPMiddleware")
    assert chain[:first_base_http] == (
        ("RequestProgressMiddleware", None),
        ("_DisconnectSignalMiddleware", None),
        ("_BodyStreamCapMiddleware", None),
    )


def test_a_handler_runs_inside_the_transport_middleware():
    from localm.inference._hang_alarm import tracker

    app = create_app(None)
    seen = {}

    @app.get("/assembly/probe")
    async def _probe(request: Request):
        seen["disconnect_poll"] = callable(request.scope.get("localm.disconnect_poll"))
        seen["in_flight"] = [label for _, label in tracker().observe()[0]]
        return {}

    assert TestClient(app).get("/assembly/probe").status_code == 200
    assert seen["disconnect_poll"] is True
    assert "GET /assembly/probe" in seen["in_flight"]


# --------------------------------------------------------------------------- #
# Consequences of the order a request can observe
# --------------------------------------------------------------------------- #

_CROSS_ORIGIN = {"Origin": "http://localhost:9999"}


def test_body_cap_answers_before_the_origin_guard(monkeypatch):
    """The body cap sits outside the origin guard, so an oversized cross-origin
    POST is refused as too large, not as cross-origin, and its 413 leaves
    before the security headers are added."""
    monkeypatch.setattr(hs, "MAX_REQUEST_BODY_BYTES", 64)
    r = TestClient(create_app(None)).post(
        "/api/plugins/refresh", content=b"x" * 65,
        headers={**_CROSS_ORIGIN, "Content-Type": "application/json"})
    assert (r.status_code, r.json()) == (413, {"detail": "Request body too large."})
    assert "content-security-policy" not in r.headers


def test_docs_guard_answers_before_the_security_headers():
    """The docs guard is the outermost BaseHTTPMiddleware handler: off a
    loopback bind its 404 leaves without passing the security headers."""
    app = create_app(None)
    app.state.bind_host = "0.0.0.0"
    r = TestClient(app).get("/docs")
    assert (r.status_code, r.json()) == (404, {"detail": "Not Found"})
    assert "content-security-policy" not in r.headers
    assert "x-content-type-options" not in r.headers


def test_security_headers_wrap_the_origin_guard_and_cors():
    """A cross-origin request CORS admits gets both the CORS grant and the
    security headers; one the origin guard refuses gets the security headers
    and no CORS grant."""
    client = TestClient(create_app(None))
    origin = {"Origin": "http://localhost:5173"}
    allowed = client.get("/v1/models", headers=origin)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "'nonce-" in allowed.headers["content-security-policy"]
    refused = client.post("/api/plugins/refresh", headers=origin)
    assert refused.status_code == 403
    assert "'nonce-" in refused.headers["content-security-policy"]
    assert "access-control-allow-origin" not in refused.headers


def test_request_log_is_innermost(monkeypatch, caplog):
    """In debug mode the request log sits inside the origin guard: a request
    the guard refuses never reaches it, a served one is logged."""
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    caplog.set_level(logging.DEBUG, logger="localm")
    client = TestClient(create_app(None))
    assert client.post("/api/plugins/refresh", headers=_CROSS_ORIGIN).status_code == 403
    assert client.get("/v1/models").status_code == 200
    request_line = re.compile(r"^([A-Z]+ \S+ -> \d{3}) \(\d+ ms, loop_lag=")
    logged = [m.group(1) for r in caplog.records
              if (m := request_line.match(r.getMessage()))]
    assert logged == ["GET /v1/models -> 200"]


# --------------------------------------------------------------------------- #
# The AppContext handed to the route groups
# --------------------------------------------------------------------------- #

_ROUTE_GROUPS = ("models", "system", "session", "config", "keys", "gpu",
                 "peer_routing", "admin", "chat")


def test_every_route_group_gets_the_same_frozen_app_context(monkeypatch):
    import importlib

    from localm.audit import effective_mode
    from localm.inference.app_assembly.context import AppContext

    received = {}
    for name in _ROUTE_GROUPS:
        module = importlib.import_module(f"localm.inference.routes.{name}")

        def _spy(app, ctx, _name=name, _real=module.register):
            received[_name] = ctx
            return _real(app, ctx)

        monkeypatch.setattr(module, "register", _spy)

    create_app(None)

    assert list(received) == list(_ROUTE_GROUPS)
    ctx = received["chat"]
    assert all(c is ctx for c in received.values())
    assert type(ctx) is AppContext
    assert ctx.audit is hs._audit
    assert ctx.mode == effective_mode("server")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.mode = None


# --------------------------------------------------------------------------- #
# Assembly code reads http_server's names through the module
# --------------------------------------------------------------------------- #


def test_loopback_decisions_read_is_loopback_host_from_http_server(monkeypatch):
    """Patched after assembly: the docs guard and /debug/stacks look the
    classifier up per request, through http_server."""
    app = create_app(None)
    client = TestClient(app)
    auth = {"Authorization": f"Bearer {app.state.shell_token}"}
    assert client.get("/docs").status_code == 200
    assert client.get("/debug/stacks", headers=auth).status_code == 200
    monkeypatch.setattr(hs, "_is_loopback_host", lambda host: False)
    assert client.get("/docs").status_code == 404
    assert client.get("/debug/stacks", headers=auth).status_code == 404


def test_open_mode_gate_reads_bearer_token_from_http_server(monkeypatch):
    app = create_app(None)
    client = TestClient(app)
    assert client.get("/api/plugins").status_code == 403
    monkeypatch.setattr(hs, "_bearer_token", lambda request: app.state.shell_token)
    assert client.get("/api/plugins").status_code == 200


def test_debug_stacks_gate_is_http_servers_require_fs_host_at_assembly(monkeypatch):
    def _patched_require_fs_host(request: Request) -> None:
        return None

    monkeypatch.setattr(hs, "require_fs_host", _patched_require_fs_host)
    app = create_app(None)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/debug/stacks")
    assert [d.call for d in route.dependant.dependencies] == [_patched_require_fs_host]


# --------------------------------------------------------------------------- #
# Plugins attach last; create_app() stays a readable boot order
# --------------------------------------------------------------------------- #


def test_plugins_attach_after_every_kernel_route_group(monkeypatch):
    kernel_paths_at_attach = set()

    def _spy_attach(app, engine):
        kernel_paths_at_attach.update(getattr(r, "path", None) for r in app.routes)

    monkeypatch.setattr("localm.plugins.engine.attach_engine", _spy_attach)
    create_app(None)
    # One route from the first and one from the last group mounted.
    assert {"/v1/models", "/v1/chat/completions"} <= kernel_paths_at_attach


def test_create_app_stays_a_short_boot_order():
    lines = inspect.getsource(create_app).splitlines()
    assert len(lines) <= 200, (
        f"create_app() is {len(lines)} lines; add a new assembly step to "
        "localm/inference/app_assembly/ and call it from create_app() instead "
        "of growing its body (ADR-0023)")
