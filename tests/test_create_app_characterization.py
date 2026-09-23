# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characterization of the app ``create_app()`` assembles.

Pins the assembled app as behaviour rather than source text: the route set for
both ``api_landing`` shapes, every route's dependency tree with the scope each
gate enforces, the middleware stack order, the exception handlers and the
response each one produces, the ``app.state`` fields assembly sets and the ones
its own middleware, handlers and lifespan read back, the plugin-attach failure
path, and the engine ``create_app(engine)`` publishes to the module globals.
Changing any of these means changing the matching expected table or assertion
in this file.
"""

import inspect
import logging
import sys
from typing import Optional

import pytest
from fastapi import HTTPException, Request
from fastapi.exception_handlers import websocket_request_validation_exception_handler
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Mount, Route, WebSocketRoute, compile_path

from localm.inference.http_server import create_app

# --------------------------------------------------------------------------- #
# Route walk
# --------------------------------------------------------------------------- #


def _label(call) -> str:
    """One dependency as a label: ``scope:<scope>`` for a ``require_scope``
    gate, ``owner`` for a ``require_owner`` gate, ``auth`` for
    ``_require_auth``, ``fs_host`` for ``require_fs_host``, and the callable's
    ``__name__`` for anything else."""
    qualname = getattr(call, "__qualname__", "")
    if qualname == "require_scope.<locals>.dep":
        return "scope:" + inspect.getclosurevars(call).nonlocals["scope"]
    if qualname == "require_owner.<locals>.dep":
        return "owner"
    return {"_require_auth": "auth", "require_fs_host": "fs_host"}.get(
        qualname, getattr(call, "__name__", repr(call)))


def _dependency_labels(dependant) -> tuple:
    """Every dependency in *dependant*'s tree as a label, depth-first, each
    dependency listed before its own sub-dependencies."""
    labels = []
    for sub in dependant.dependencies:
        labels.append(_label(sub.call))
        labels.extend(_dependency_labels(sub))
    return tuple(labels)


def _expand(route) -> list:
    """``(kind, method, path, dependant)`` for each endpoint *route* serves.
    Descends into lazily included routers (``include_router``), whose effective
    dependencies carry the include-level ones. Raises on a route type it cannot
    classify."""
    contexts = getattr(route, "effective_route_contexts", None)
    if callable(contexts):
        out = []
        for ctx in contexts():
            if ctx.starlette_route is not None:
                out.extend(_expand(ctx.starlette_route))
            elif isinstance(ctx.original_route, APIRoute):
                out.extend(("api", method, ctx.path, ctx.dependant)
                           for method in sorted(ctx.methods))
            else:
                raise AssertionError(
                    f"unclassified included route {type(ctx.original_route).__name__}")
        return out
    if isinstance(route, APIRoute):
        return [("api", method, route.path, route.dependant)
                for method in sorted(route.methods)]
    if isinstance(route, APIWebSocketRoute):
        return [("websocket", "WS", route.path, route.dependant)]
    if isinstance(route, WebSocketRoute):
        return [("websocket", "WS", route.path, None)]
    if isinstance(route, Route):
        return [("route", method, route.path, None)
                for method in sorted(route.methods or ())]
    if isinstance(route, Mount):
        return [("mount", "*", route.path, None)]
    raise AssertionError(f"unclassified route type {type(route).__name__}")


def _route_table(app) -> dict:
    """``{(kind, method, path): dependency labels}`` for everything *app*
    serves. Raises when two routes serve the same method and path."""
    table = {}
    for route in app.router.routes:
        for kind, method, path, dependant in _expand(route):
            key = (kind, method, path)
            assert key not in table, f"{method} {path} is served by two routes"
            table[key] = () if dependant is None else _dependency_labels(dependant)
    return table


def _concrete(template: str) -> str:
    """A request path for a route template: every ``{param}`` becomes ``x``."""
    return "/".join("x" if seg.startswith("{") and seg.endswith("}") else seg
                    for seg in template.split("/"))


def _describe(keys) -> str:
    return "\n".join(f"    {key}" for key in sorted(keys)) or "    (none)"


# --------------------------------------------------------------------------- #
# Expected route tables
# --------------------------------------------------------------------------- #

# FastAPI's own docs routes.
_DOCS_ROUTES = {
    ("route", method, path): ()
    for path in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc")
    for method in ("GET", "HEAD")
}

# create_app() itself plus localm/inference/routes/*.py.
_KERNEL_ROUTES = {
    **_DOCS_ROUTES,
    ("api", "GET", "/debug/stacks"): ("fs_host",),
    # routes/models.py
    ("api", "GET", "/v1/models"): ("scope:models:read",),
    ("api", "GET", "/v1/models/{model_id}"): ("scope:models:read",),
    ("api", "POST", "/v1/models/unload"): ("scope:models:write",),
    ("api", "POST", "/v1/models/rename"): ("scope:models:write",),
    ("api", "POST", "/v1/models/load"): ("scope:models:write",),
    ("api", "GET", "/v1/models/{model_id}/hold"): ("scope:models:read",),
    # routes/system.py
    ("api", "GET", "/health"): (),
    ("api", "GET", "/whoami"): (),
    ("api", "POST", "/v1/surfaces/gui"): (),
    ("api", "GET", "/localm-ca.crt"): (),
    # routes/session.py
    ("api", "GET", "/api/session"): (),
    ("api", "POST", "/api/session"): (),
    ("api", "POST", "/api/session/logout"): (),
    ("api", "POST", "/api/auth/key/clear"): ("scope:config:write",),
    ("api", "POST", "/api/auth/key/rotate"): ("scope:admin",),
    # routes/config.py
    ("api", "GET", "/v1/config"): ("scope:config:read",),
    ("api", "GET", "/v1/config/schema"): ("scope:config:read",),
    ("api", "PATCH", "/v1/config"): ("scope:config:write",),
    ("api", "GET", "/v1/media/config"): ("scope:config:read",),
    ("api", "POST", "/v1/media/config/{name}"): ("scope:config:write",),
    ("api", "GET", "/v1/tts/config"): ("scope:config:read",),
    ("api", "POST", "/v1/tts/config"): ("scope:config:write",),
    ("api", "GET", "/v1/plugins/settings"): ("scope:config:read",),
    ("api", "POST", "/v1/plugins/{name}/settings"): ("scope:config:write",),
    ("api", "GET", "/v1/comfy/status"): ("scope:config:read",),
    ("api", "POST", "/v1/comfy/stop"): ("scope:config:write",),
    ("api", "POST", "/v1/comfy/restart"): ("scope:config:write",),
    # routes/keys.py
    ("api", "GET", "/v1/keys"): ("scope:keys:admin", "caller_scopes"),
    ("api", "POST", "/v1/keys"): ("scope:keys:admin", "caller_scopes"),
    ("api", "DELETE", "/v1/keys/{key_id}"): ("scope:keys:admin",),
    # routes/gpu.py
    ("api", "POST", "/v1/instances/cooperate-unload"): (),
    # routes/peer_routing.py
    ("api", "GET", "/v1/models/{model_id}/peer-offer"): ("scope:models:read",),
    ("api", "POST", "/v1/models/{model_id}/peer-route"): ("scope:models:write",),
    ("api", "DELETE", "/v1/models/{model_id}/peer-route"): ("scope:models:write",),
    # routes/admin.py
    ("api", "POST", "/v1/server/shutdown"): ("scope:config:write",),
    ("api", "POST", "/v1/server/restart"): ("scope:config:write",),
    ("api", "POST", "/api/bug-report"): ("scope:config:write",),
    ("api", "GET", "/api/issues"): ("scope:config:read",),
    ("api", "GET", "/api/update/check"): ("scope:config:read",),
    ("api", "GET", "/api/changelog"): ("scope:config:read",),
    ("api", "POST", "/api/update/apply"): ("scope:config:write",),
    ("api", "GET", "/api/update/rollback"): ("scope:config:read",),
    ("api", "POST", "/api/update/rollback"): ("scope:config:write",),
    # routes/chat.py
    ("api", "POST", "/v1/chat/completions"): ("auth",),
    ("api", "POST", "/v1/embeddings"): ("auth",),
    ("api", "POST", "/v1/completions"): ("auth",),
}

# attach_engine(): plugin management, plugin dependency install, background jobs.
_PLUGIN_ENGINE_ROUTES = {
    ("api", "GET", "/api/plugins"): ("scope:plugins:read",),
    ("api", "POST", "/api/plugins/{name}/install"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/install-external"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/{name}/uninstall"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/refresh"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/{name}/refresh"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/{name}/enable"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/{name}/disable"): ("scope:plugins:admin",),
    ("api", "POST", "/api/plugins/{name}/install-deps"): ("scope:plugins:admin",),
    ("api", "GET", "/api/plugins/{name}/install-deps/events"): ("scope:plugins:admin",),
    ("api", "GET", "/api/activity"): ("auth",),
    ("api", "GET", "/api/jobs/{job_id}/events"): ("auth", "owner", "_resolve_job"),
    ("api", "POST", "/api/jobs/{job_id}/cancel"): ("auth", "owner", "_resolve_job"),
}

# The preinstalled chat plugin, mounted through PluginHost.mount_router().
_CHAT_PLUGIN_ROUTES = {
    ("api", "GET", "/api/conversations"): ("scope:chat",),
    ("api", "GET", "/api/conversations/{conv_id}"): ("scope:chat",),
    ("api", "PUT", "/api/conversations/{conv_id}"): ("scope:chat",),
    ("api", "DELETE", "/api/conversations/{conv_id}"): ("scope:chat",),
    ("api", "GET", "/api/prompts"): ("scope:chat",),
    ("api", "PUT", "/api/prompts/{name}"): ("scope:chat",),
    ("api", "DELETE", "/api/prompts/{name}"): ("scope:chat",),
}

_API_LANDING_ROUTES = {("api", "GET", "/"): ()}


def _expected_routes(api_landing: bool) -> dict:
    table = {**_KERNEL_ROUTES, **_PLUGIN_ENGINE_ROUTES, **_CHAT_PLUGIN_ROUTES}
    if api_landing:
        table.update(_API_LANDING_ROUTES)
    return table


_SHAPES = pytest.mark.parametrize(
    "api_landing", [False, True], ids=["gui-shape", "api-landing"])

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@_SHAPES
def test_route_set(api_landing):
    actual = _route_table(create_app(None, api_landing=api_landing)).keys()
    expected = _expected_routes(api_landing).keys()
    assert actual == expected, (
        "create_app() serves a different route set.\n  no longer served:\n"
        + _describe(expected - actual) + "\n  newly served:\n"
        + _describe(actual - expected)
        + "\nUpdate the expected tables in this file if the change is intended.")


@_SHAPES
def test_route_dependencies(api_landing):
    actual = _route_table(create_app(None, api_landing=api_landing))
    expected = _expected_routes(api_landing)
    changed = sorted(
        f"    {key}: expected {expected[key]}, got {actual[key]}"
        for key in actual.keys() & expected.keys() if actual[key] != expected[key])
    assert not changed, (
        "route dependencies changed:\n" + "\n".join(changed)
        + "\nUpdate the expected tables in this file if the change is intended.")


def test_route_walk_reaches_every_documented_operation():
    app = create_app(None, api_landing=True)
    documented = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method.upper() in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
    }
    walked = {(method, path) for kind, method, path in _route_table(app) if kind == "api"}
    assert ("GET", "/api/conversations") in documented
    assert documented <= walked, (
        "the route walk misses operations the OpenAPI schema documents:\n"
        + _describe(documented - walked))


@_SHAPES
def test_no_two_routes_serving_one_method_match_the_same_path(api_landing):
    routes = [(method, path) for kind, method, path in
              _route_table(create_app(None, api_landing=api_landing))
              if kind in ("api", "route")]
    patterns = {path: compile_path(path)[0] for _, path in routes}
    overlaps = sorted(
        f"    {method} {path} is also matched by {other}"
        for method, path in routes
        for other_method, other in routes
        if method == other_method and path != other
        and patterns[other].match(_concrete(path)))
    assert not overlaps, "\n".join(overlaps)


def test_api_landing_root_redirects_to_the_docs():
    landing = TestClient(create_app(None, api_landing=True)).get(
        "/", follow_redirects=False)
    assert (landing.status_code, landing.headers["location"]) == (307, "/docs")
    assert TestClient(create_app(None)).get("/").status_code == 404


# One route per gate label, as (method, template, JSON body). Each request is
# one its handler only reads from or refuses before writing anything.
_GATE_PROBES = {
    "auth": ("POST", "/v1/embeddings", {}),
    "fs_host": ("GET", "/debug/stacks", None),
    "scope:admin": ("POST", "/api/auth/key/rotate", {"key": 0}),
    "scope:chat": ("GET", "/api/conversations", None),
    "scope:config:read": ("GET", "/v1/config", None),
    "scope:config:write": ("PATCH", "/v1/config", []),
    "scope:keys:admin": ("GET", "/v1/keys", None),
    "scope:models:read": ("GET", "/v1/models", None),
    "scope:models:write": ("DELETE", "/v1/models/{model_id}/peer-route", None),
    "scope:plugins:admin": ("POST", "/api/plugins/{name}/enable", None),
    "scope:plugins:read": ("GET", "/api/plugins", None),
}


def test_every_gate_label_is_enforced_with_its_scope():
    from localm import auth
    no_scope_key = auth.create_key("characterization", [])["key"]
    app = create_app(None)
    table = _route_table(app)
    gate_labels = {label for labels in table.values() for label in labels
                   if label.startswith("scope:") or label in ("auth", "fs_host")}
    assert gate_labels == set(_GATE_PROBES), (
        "gate labels in use changed; give every label a probe route in "
        f"_GATE_PROBES: in use {sorted(gate_labels)}")

    client = TestClient(app)
    for label, (method, template, body) in sorted(_GATE_PROBES.items()):
        assert table[("api", method, template)][:1] == (label,), (
            f"{method} {template} no longer starts with the {label} gate")
        url = _concrete(template)
        anonymous = client.request(method, url, json=body)
        assert (anonymous.status_code, anonymous.json()) == (
            401, {"detail": "Invalid or missing API key"}), f"{label}: {method} {url}"
        if label == "auth":
            continue
        refused = client.request(
            method, url, json=body, headers={"Authorization": f"Bearer {no_scope_key}"})
        detail = ("This key does not have host filesystem access" if label == "fs_host"
                  else f"Key lacks required scope: {label[len('scope:'):]}")
        assert (refused.status_code, refused.json()) == (
            403, {"detail": detail}), f"{label}: {method} {url}"


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #

# Outermost first.
_MIDDLEWARE_STACK = (
    ("RequestProgressMiddleware", None),
    ("_DisconnectSignalMiddleware", None),
    ("_BodyStreamCapMiddleware", None),
    ("BaseHTTPMiddleware", "_docs_loopback_only"),
    ("BaseHTTPMiddleware", "_security_headers"),
    ("BaseHTTPMiddleware", "_origin_guard"),
    ("CORSMiddleware", None),
)


def _middleware_stack(app) -> tuple:
    """``app.user_middleware``, outermost first, as ``(class name, dispatch
    function name or None)``."""
    return tuple((m.cls.__name__, getattr(m.kwargs.get("dispatch"), "__name__", None))
                 for m in app.user_middleware)


@_SHAPES
def test_middleware_stack_order(monkeypatch, api_landing):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert _middleware_stack(create_app(None, api_landing=api_landing)) == _MIDDLEWARE_STACK


def test_debug_mode_adds_request_logging_innermost(monkeypatch):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    assert _middleware_stack(create_app(None)) == (
        _MIDDLEWARE_STACK + (("BaseHTTPMiddleware", "_log_requests"),))


def test_origin_guard_refusal_carries_security_headers_and_no_cors_grant():
    client = TestClient(create_app(None))
    origin = {"Origin": "http://localhost:5173"}

    allowed = client.get("/v1/models", headers=origin)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"

    refused = client.post("/api/plugins/refresh", headers=origin)
    assert refused.status_code == 403
    assert refused.json()["detail"].startswith("Cross-origin request refused")
    assert refused.headers["x-content-type-options"] == "nosniff"
    assert refused.headers["content-security-policy"].startswith(
        "default-src 'self'; script-src 'self' blob: 'wasm-unsafe-eval' 'nonce-")
    assert refused.headers["cross-origin-opener-policy"] == "same-origin"
    assert refused.headers["cross-origin-embedder-policy"] == "credentialless"
    assert "access-control-allow-origin" not in refused.headers


def test_security_headers_hand_the_handler_the_nonce_they_send():
    app = create_app(None)

    @app.get("/characterize/nonce")
    async def _nonce(request: Request):
        return {"nonce": request.state.csp_nonce}

    response = TestClient(app).get("/characterize/nonce")
    nonce = response.json()["nonce"]
    assert nonce
    assert f"'nonce-{nonce}'" in response.headers["content-security-policy"]


_DEFAULT_CORS = {"allow_methods": ["*"], "allow_headers": ["*"]}


@pytest.mark.parametrize("cors_origins, expected", [
    (None, {**_DEFAULT_CORS,
            "allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?"}),
    ([], {**_DEFAULT_CORS,
          "allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?"}),
    (["https://app.example"], {**_DEFAULT_CORS, "allow_origins": ["https://app.example"]}),
    ("*", {**_DEFAULT_CORS, "allow_origins": ["*"]}),
], ids=["unset", "empty-list", "list", "wildcard"])
def test_cors_middleware_follows_the_cors_origins_setting(cors_origins, expected):
    from localm.config import load_config, save_config
    if cors_origins is not None:
        cfg = load_config()
        cfg["cors_origins"] = cors_origins
        save_config(cfg)
    cors = [m for m in create_app(None).user_middleware
            if m.cls.__name__ == "CORSMiddleware"]
    assert [m.kwargs for m in cors] == [expected]


# --------------------------------------------------------------------------- #
# Exception handlers
# --------------------------------------------------------------------------- #


class _ProbeBody(BaseModel):
    count: int
    flag: bool
    max_tokens: Optional[int] = Field(default=None, ge=1)


def _app_with_probe_routes():
    """A ``create_app(None)`` app plus three routes that raise into its
    exception handlers."""
    app = create_app(None)

    @app.get("/characterize/boom")
    async def _boom():
        raise RuntimeError("characterization detail that must not leak")

    @app.get("/characterize/refuse")
    async def _refuse(detail: str = "characterization refusal"):
        raise HTTPException(status_code=409, detail=detail, headers={"Retry-After": "7"})

    @app.post("/characterize/validate")
    async def _validate(body: _ProbeBody):
        return {"ok": True}

    return app


def _shell_auth(app) -> dict:
    return {"Authorization": f"Bearer {app.state.shell_token}"}


def test_exception_handler_registry():
    handlers = create_app(None).exception_handlers
    assert set(handlers) == {Exception, StarletteHTTPException,
                             RequestValidationError, WebSocketRequestValidationError}
    assert handlers[WebSocketRequestValidationError] is (
        websocket_request_validation_exception_handler)
    for exc_type in (Exception, StarletteHTTPException, RequestValidationError):
        assert handlers[exc_type].__module__.startswith("localm."), exc_type


def test_unhandled_error_is_a_generic_logged_500(caplog):
    caplog.set_level(logging.ERROR, logger="localm")
    client = TestClient(_app_with_probe_routes(), raise_server_exceptions=False)
    response = client.get("/characterize/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "must not leak" not in response.text
    logged = [r for r in caplog.records
              if r.getMessage() == "unhandled error: GET /characterize/boom"]
    assert len(logged) == 1
    assert logged[0].levelno == logging.ERROR
    assert logged[0].exc_info[0] is RuntimeError


def test_http_exception_keeps_status_detail_and_headers():
    client = TestClient(_app_with_probe_routes())
    refused = client.get("/characterize/refuse")
    assert refused.status_code == 409
    assert refused.json() == {"detail": "characterization refusal"}
    assert refused.headers["retry-after"] == "7"
    missing = client.get("/characterize/missing")
    assert (missing.status_code, missing.json()) == (404, {"detail": "Not Found"})


def test_http_exception_detail_is_logged_only_in_debug_mode(monkeypatch, caplog):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    caplog.set_level(logging.DEBUG, logger="localm")
    client = TestClient(_app_with_probe_routes())

    def refusal_lines():
        return [r.getMessage() for r in caplog.records if " refused " in r.getMessage()]

    client.get("/characterize/refuse")
    assert refusal_lines() == []

    monkeypatch.setenv("LOCALM_DEBUG", "1")
    long_detail = "d" * 600
    response = client.get("/characterize/refuse", params={"detail": long_detail})
    assert response.json() == {"detail": long_detail}
    assert refusal_lines() == [
        f"GET /characterize/refuse refused 409: {'d' * 500} ...[truncated]"]


def test_validation_error_shape():
    app = _app_with_probe_routes()
    client = TestClient(app)
    response = client.post("/characterize/validate", json={"count": "abc"},
                           headers=_shell_auth(app))
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"detail", "errors"}
    assert [e["loc"] for e in body["errors"]] == [["body", "count"], ["body", "flag"]]
    count_part, flag_part = body["detail"].split("; ")
    assert count_part.startswith("count must be ")
    assert count_part.endswith(" (got 'abc')")
    assert flag_part.startswith("flag ")
    assert "(got" not in flag_part


def test_validation_error_names_max_tokens_zero():
    app = _app_with_probe_routes()
    response = TestClient(app).post(
        "/characterize/validate", json={"count": 1, "flag": True, "max_tokens": 0},
        headers=_shell_auth(app))
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "max_tokens must be 1 or more - 0 is not 'no limit'. "
        "Omit max_tokens entirely to use the model's default")


def test_validation_error_renders_a_non_finite_input():
    app = _app_with_probe_routes()
    response = TestClient(app).post(
        "/characterize/validate", content=b'{"count": NaN, "flag": true}',
        headers={**_shell_auth(app), "Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["errors"][0]["input"] == "nan"


def test_validation_error_elides_deeply_nested_input():
    app = _app_with_probe_routes()
    nested = 1
    for _ in range(40):
        nested = [nested]
    response = TestClient(app).post(
        "/characterize/validate", json={"count": nested, "flag": True},
        headers=_shell_auth(app))
    assert response.status_code == 422
    assert "...[nested value elided]" in response.text


# --------------------------------------------------------------------------- #
# app.state
# --------------------------------------------------------------------------- #


def test_assembly_sets_exactly_these_state_fields():
    from localm.inference.chat_pipeline import ChatPipeline
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager

    app = create_app(None)
    assert set(app.state._state) == {
        "chat_pipeline", "csrf_secret", "jobs", "plugin_manager", "shell_token"}
    assert isinstance(app.state.chat_pipeline, ChatPipeline)
    assert isinstance(app.state.plugin_manager, PluginManager)
    assert isinstance(app.state.jobs, JobManager)
    for name in ("shell_token", "csrf_secret"):
        value = getattr(app.state, name)
        assert isinstance(value, str) and len(value) >= 43, name
    assert app.state.shell_token != app.state.csrf_secret

    second = create_app(None)
    assert second.state.shell_token != app.state.shell_token
    assert second.state.csrf_secret != app.state.csrf_secret


@pytest.mark.parametrize("bind_host, served", [
    (None, True), ("127.0.0.1", True), ("0.0.0.0", False), ("192.0.2.10", False),
], ids=["unset", "loopback", "wildcard", "lan"])
def test_docs_and_debug_stacks_follow_app_state_bind_host(bind_host, served):
    app = create_app(None)
    if bind_host is not None:
        app.state.bind_host = bind_host
    client = TestClient(app)
    for path in ("/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"):
        response = client.get(path)
        if served:
            assert response.status_code == 200, path
        else:
            assert (response.status_code, response.json()) == (
                404, {"detail": "Not Found"}), path

    stacks = client.get("/debug/stacks", headers=_shell_auth(app))
    if served:
        assert stacks.status_code == 200
        assert set(stacks.json()) == {"pid", "loop_lag_s", "threads", "tasks", "executors"}
        assert client.get("/debug/stacks").status_code == 403
    else:
        assert (stacks.status_code, stacks.json()) == (404, {"detail": "Not Found"})


def test_open_mode_management_gate_reads_its_tokens_from_app_state():
    app = create_app(None)
    client = TestClient(app)

    def status(token):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return client.get("/api/plugins", headers=headers).status_code

    denied = client.get("/api/plugins")
    assert (denied.status_code, denied.json()) == (403, {
        "detail": "Open-mode management requires the localm GUI shell on this "
                  "machine, or an API key (run 'localm key generate')."})
    assert status("not-the-token") == 403
    assert status(app.state.shell_token) == 200

    app.state.instance_token = "characterization-instance-token"
    assert status("characterization-instance-token") == 200

    previous = app.state.shell_token
    app.state.shell_token = "characterization-replacement-token"
    assert status("characterization-replacement-token") == 200
    assert status(previous) == 403


def test_lifespan_runs_the_plugin_manager_startup_callbacks(monkeypatch):
    app = create_app(None)
    calls = []
    monkeypatch.setattr(app.state.plugin_manager, "run_startup_callbacks",
                        lambda: calls.append("startup"))
    with TestClient(app):
        assert calls == ["startup"]
        assert "gpu_coordination_token" not in app.state._state


# --------------------------------------------------------------------------- #
# Engine registration
# --------------------------------------------------------------------------- #


class _StubEngine:
    display_name = "characterization-model"
    loaded = True


def test_create_app_publishes_its_engine_and_a_later_call_resets_it():
    from localm.inference import http_server as hs

    engine = _StubEngine()
    try:
        app = create_app(engine)
        assert hs._engine is engine
        assert hs._engines == {"characterization-model": engine}
        assert hs._engines_lru == ["characterization-model"]
        assert hs._default_model_name == "characterization-model"
        assert hs._active_model_name == "characterization-model"
        assert hs._inference_sem is not None
        assert hs._inference_sems == {"characterization-model": hs._inference_sem}
        assert set(hs._last_activity_per_model) == {"characterization-model"}
        assert TestClient(app).get("/health").json() == {
            "status": "ok", "model": "characterization-model", "loaded": True}
    finally:
        create_app(None)

    assert hs._engine is None
    assert hs._inference_sem is None
    assert hs._engines == {}
    assert hs._engines_lru == []
    assert hs._inference_sems == {}
    assert hs._last_activity_per_model == {}
    assert hs._default_model_name is None
    assert hs._active_model_name is None


# --------------------------------------------------------------------------- #
# Plugin-attach failure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("breakage", ["attach-raises", "engine-import-fails"])
def test_plugin_attach_failure_keeps_the_kernel_serving(monkeypatch, caplog, breakage):
    if breakage == "attach-raises":
        def _broken_attach(app, engine):
            raise RuntimeError("characterization: plugin engine exploded")

        monkeypatch.setattr("localm.plugins.engine.attach_engine", _broken_attach)
        error_type = RuntimeError
        error = "characterization: plugin engine exploded"
    else:
        monkeypatch.setitem(sys.modules, "localm.plugins.engine", None)
        error_type = ModuleNotFoundError
        error = "import of localm.plugins.engine halted; None in sys.modules"
    caplog.set_level(logging.DEBUG, logger="localm")

    app = create_app(None)

    assert app.state.plugin_engine_error == error
    assert set(app.state._state) == {
        "chat_pipeline", "csrf_secret", "shell_token", "plugin_engine_error"}
    logged = [(r.levelno, r.getMessage(), r.exc_info[0] if r.exc_info else None)
              for r in caplog.records if r.name == "localm"]
    assert (logging.WARNING, f"plugins unavailable: {error}", None) in logged
    assert (logging.ERROR, "plugin engine attach failed", error_type) in logged
    assert _route_table(app).keys() == _KERNEL_ROUTES.keys()
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 200
