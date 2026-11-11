# SPDX-License-Identifier: AGPL-3.0-or-later
"""In debug mode every request gets exactly one request-log line, including a
request a middleware answers itself without passing it on: the open-mode
management gate, the cross-origin refusal, the keyless network gate, the docs
gate, the Host header guard, the body size cap and a refused CORS preflight.
The line carries method, path, status and, for a middleware refusal, a fixed
reason; never a credential. Debug mode does not change what a request gets."""

from __future__ import annotations

import logging
import re
import secrets
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.debuglog as debuglog
from localm import auth
from localm.inference import http_server as hs
from localm.inference.app_assembly import security
from localm.inference.app_assembly.diagnostics import RequestLogMiddleware
from localm.inference.http_server import create_app

_REQUEST_LINE = re.compile(
    r"^([A-Z]+ \S+ -> (\d{3})) \(\d+ ms, loop_lag=(?:\d+\.\d{2}s|n/a)\)"
    r"(?: refused: (.+))?$")
_CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def debug_on(monkeypatch, caplog):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    assert debuglog.debug_enabled(), "test premise: --debug must be ON"
    caplog.set_level(logging.DEBUG, logger="localm")


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    e.loaded = True
    e.chat_stream.side_effect = lambda messages, **k: iter(["ok"])
    return e


def _request_lines(caplog) -> list:
    return [m for r in caplog.records
            if (m := _REQUEST_LINE.match(r.getMessage()))]


def _one_line(caplog, request: str, status: int):
    lines = _request_lines(caplog)
    assert [m.group(1) for m in lines] == [f"{request} -> {status}"], (
        [r.getMessage() for r in caplog.records])
    return lines[0]


def _secret_bearer() -> tuple[str, str]:
    value = "w1-not-a-real-credential-" + secrets.token_hex(12)
    return value, f"Bearer {value}"


def _assert_never_logged(caplog, value: str) -> None:
    assert value not in caplog.text
    assert not any(value in r.getMessage() for r in caplog.records)


def test_the_request_log_is_the_outermost_middleware(debug_on):
    app = create_app(None)
    assert app.user_middleware[0].cls is RequestLogMiddleware
    assert sum(m.cls is RequestLogMiddleware for m in app.user_middleware) == 1


def test_refused_management_call_is_logged_once_with_status_and_reason(debug_on, caplog):
    value, bearer = _secret_bearer()
    client = TestClient(create_app(None))

    r = client.post("/api/plugins/refresh", headers={"Authorization": bearer})

    assert r.status_code == 403
    line = _one_line(caplog, "POST /api/plugins/refresh", 403)
    assert line.group(3) == "open-mode management without the shell token"
    _assert_never_logged(caplog, value)


def test_cross_origin_refusal_is_logged_once(debug_on, caplog):
    client = TestClient(create_app(None))

    r = client.post("/api/plugins/refresh", headers={"Origin": "http://localhost:9999"})

    assert r.status_code == 403
    assert _one_line(caplog, "POST /api/plugins/refresh", 403).group(3) == (
        "cross-origin request")


def test_unauthenticated_call_refused_by_the_keyless_network_gate_is_logged_once(
        debug_on, caplog):
    value, bearer = _secret_bearer()
    app = create_app(None)
    app.state.bind_host = "192.168.0.10"
    client = TestClient(app)

    r = client.get("/v1/models", headers={"Authorization": bearer})

    assert r.status_code == 401
    assert r.json()["detail"] == hs.KEYLESS_NETWORK_DETAIL
    assert _one_line(caplog, "GET /v1/models", 401).group(3) == (
        "no API key is configured on a network bind")
    _assert_never_logged(caplog, value)


def test_unauthenticated_call_refused_by_the_route_is_logged_once(debug_on, caplog):
    auth.set_api_key("w1-strong-test-key-0123456789abcdef")
    value, bearer = _secret_bearer()
    client = TestClient(create_app(_engine()))

    r = client.post("/v1/chat/completions", json=_CHAT, headers={"Authorization": bearer})

    assert r.status_code == 401
    _one_line(caplog, "POST /v1/chat/completions", 401)
    _assert_never_logged(caplog, value)


def test_allowed_call_is_logged_once_without_a_refusal(debug_on, caplog):
    client = TestClient(create_app(None))

    assert client.get("/v1/models").status_code == 200

    assert _one_line(caplog, "GET /v1/models", 200).group(3) is None


def test_docs_refused_off_loopback_is_logged_once(debug_on, caplog):
    app = create_app(None)
    app.state.bind_host = "192.168.0.10"
    client = TestClient(app)

    assert client.get("/docs").status_code == 404

    assert _one_line(caplog, "GET /docs", 404).group(3) == (
        "API docs are served on a loopback bind only")


def test_host_header_refusal_is_logged_once(debug_on, caplog):
    app = create_app(None)
    app.state.host_header_guard = True
    app.state.host_policy = (
        security.HostPolicy(frozenset(), frozenset(), False), time.monotonic())
    client = TestClient(app)

    assert client.get("/health", headers={"host": "evil.example"}).status_code == 400

    assert _one_line(caplog, "GET /health", 400).group(3) == "Host header not allowed"


def test_body_size_refusal_is_logged_once(debug_on, caplog, monkeypatch):
    monkeypatch.setattr(hs, "MAX_REQUEST_BODY_BYTES", 100)
    client = TestClient(create_app(None))

    r = client.post("/v1/chat/completions", content=b"x" * 500,
                    headers={"Content-Type": "application/json"})

    assert r.status_code == 413
    assert _one_line(caplog, "POST /v1/chat/completions", 413).group(3) == (
        "request body too large")


def test_refused_cors_preflight_is_logged_once(debug_on, caplog):
    client = TestClient(create_app(None))

    r = client.options("/v1/chat/completions",
                       headers={"Origin": "http://evil.example",
                                "Access-Control-Request-Method": "POST"})

    assert r.status_code == 400
    assert _one_line(caplog, "OPTIONS /v1/chat/completions", 400).group(3) == (
        "CORS preflight refused")


def test_no_request_line_when_debug_is_off(monkeypatch, caplog):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert not debuglog.debug_enabled(), "test premise: --debug must be OFF"
    caplog.set_level(logging.DEBUG, logger="localm")
    app = create_app(None)
    assert all(m.cls is not RequestLogMiddleware for m in app.user_middleware)

    assert TestClient(app).post("/api/plugins/refresh").status_code == 403

    assert _request_lines(caplog) == []


def _refused_management_response(monkeypatch, debug: bool):
    if debug:
        monkeypatch.setenv("LOCALM_DEBUG", "1")
    else:
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert debuglog.debug_enabled() is debug
    r = TestClient(create_app(None)).post("/api/plugins/refresh")
    return (r.status_code, r.json(),
            r.headers.get(security.SHELL_REJECTED_HEADER),
            r.headers.get("x-content-type-options"))


def test_debug_mode_does_not_change_the_refused_response(monkeypatch):
    off = _refused_management_response(monkeypatch, debug=False)
    on = _refused_management_response(monkeypatch, debug=True)
    assert off[0] == 403
    assert on == off
