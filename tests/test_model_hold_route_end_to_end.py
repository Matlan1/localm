# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /v1/models/{id}/hold and its client, against the REAL route and a REAL socket.

A mocked ``read_model_file_hold`` is satisfied by a route that does not exist,
a scope that refuses, a body shape the client cannot parse, or a URL that never
matches. This file exercises the real thing in two layers:

* The ROUTE, through TestClient, against the real
  ``loaded_engine_holding_model_file`` - so the answer comes from the same
  function the GUI's remove route trusts, not a stand-in.
* The CLIENT, over a real uvicorn socket, so the URL, the auth headers, the
  status handling and the JSON shape are all proven on the wire.

The 404 pair is the load-bearing case and gets both arms. Status alone cannot
tell "this server is too old to have the route" from "this server does not
carry that model", and those are opposite conclusions: the first refuses the
removal, the second lets it proceed. The client reads the BODY to tell them
apart.
"""

from __future__ import annotations

import asyncio
import json
import socket as _socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

import localm.config as config
import localm.model_manager as model_manager
from localm.inference.http_server import create_app


class _FileEngine:
    def __init__(self, name, path, loaded=True):
        self.display_name = name
        self.model_path = str(path)
        self.loaded = loaded


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(model_manager, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    return h


def _model_file(home, filename="m.gguf"):
    p = home / "models" / filename
    p.write_bytes(b"GGUF" + b"\0" * 64)
    return p


def _register(home, entries):
    (home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.fixture
def resident(monkeypatch):
    """Put an engine in the server's OWN map, the way a load does.

    PRECONDITION: call this AFTER the app exists and its lifespan has started,
    never before. ``create_app`` opens with ``_engines.clear()`` and sets
    ``_engine`` to whatever it was handed, so an engine injected first is wiped
    by app construction, leaving exactly the state of a server with nothing
    loaded and a confident all-clear from the guard.
    """
    import localm.inference.http_server as hs

    def _put(name, path):
        eng = _FileEngine(name, path)
        monkeypatch.setattr(hs, "_engines", {name: eng})
        monkeypatch.setattr(hs, "_engine", eng)
        # Read back through the module the guard itself reads, at the point the
        # request is about to be made.
        assert hs._engines[name].loaded
        assert hs._engines[name].model_path == str(path)
        return eng
    return _put


# --------------------------------------------------------------------------
#  The route, against the real guard
# --------------------------------------------------------------------------

def test_route_reports_a_hold_from_the_real_guard(home, resident):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    with TestClient(create_app(None)) as c:
        resident("victim", path)
        r = c.get("/v1/models/victim/hold")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["held"] is True, body
    assert body["key"] == "victim", body
    # reason None means PROVEN, not merely un-ruled-out. The client renders
    # those two as different sentences.
    assert body["reason"] is None, body


def test_route_reports_no_hold_when_nothing_is_resident(home):
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})

    with TestClient(create_app(None)) as c:
        r = c.get("/v1/models/victim/hold")

    assert r.status_code == 200, r.text
    assert r.json() == {"held": False}


def test_route_finds_the_hold_by_path_not_by_name(home, resident):
    """The rename case, end to end through the real route: the engine is keyed
    on a name the registry no longer carries, and the hold is still found."""
    path = _model_file(home)
    _register(home, {"new-name": {"path": str(path), "source": "test"}})
    with TestClient(create_app(None)) as c:
        resident("old-name", path)
        r = c.get("/v1/models/new-name/hold")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["held"] is True, (
        "a loaded engine keyed under the pre-rename name was not recognised "
        "as holding the file")
    assert body["key"] == "old-name", body


def test_route_404s_for_a_model_this_server_does_not_carry(home):
    _register(home, {"other": {"path": str(_model_file(home)), "source": "test"}})

    with TestClient(create_app(None)) as c:
        r = c.get("/v1/models/nope/hold")

    assert r.status_code == 404
    # The client discriminates on this exact prefix to tell a missing MODEL
    # from a missing ROUTE.
    assert r.json()["detail"].startswith("Model not registered"), r.json()


def test_route_does_not_leak_the_model_path(home, resident):
    """This route returns registry NAMES and a fixed reason, never an absolute
    path."""
    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    with TestClient(create_app(None)) as c:
        resident("victim", path)
        r = c.get("/v1/models/victim/hold")

    assert str(path) not in r.text
    assert str(path.parent) not in r.text


# --------------------------------------------------------------------------
#  The client, over a real socket
# --------------------------------------------------------------------------

def _real_server():
    """A real uvicorn instance in open (keyless) mode, with the instance token
    set the way instances.advertise() sets it in production."""
    app = create_app(None)
    app.state.instance_token = "test-instance-token-abcdef123456"

    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning",
                                           lifespan="on"))
    th = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                          daemon=True)
    th.start()
    deadline = time.time() + 10.0
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    return app, port, server, th


def _shutdown(server, th):
    server.should_exit = True
    th.join(timeout=5.0)


def test_client_reads_a_real_hold_over_the_wire(home, resident):
    from localm.selfclient import read_model_file_hold

    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})
    app, port, server, th = _real_server()
    resident("victim", path)
    try:
        state, payload = read_model_file_hold(
            "http", port, "victim", app.state.instance_token)
    finally:
        _shutdown(server, th)

    assert state == "ok", (state, payload)
    assert payload["held"] is True, payload
    assert payload["key"] == "victim", payload


def test_client_reads_a_real_all_clear_over_the_wire(home):
    """The permissive arm on the wire: a real all-clear reaches the client as
    an all-clear, not as "unreachable"."""
    from localm.selfclient import read_model_file_hold

    path = _model_file(home)
    _register(home, {"victim": {"path": str(path), "source": "test"}})

    app, port, server, th = _real_server()
    try:
        state, payload = read_model_file_hold(
            "http", port, "victim", app.state.instance_token)
    finally:
        _shutdown(server, th)

    assert state == "ok", (state, payload)
    assert payload["held"] is False, payload


def test_client_calls_a_model_this_server_does_not_carry_absent(home):
    """404 arm one: a real localm answering about a model it does not have.

    Must be "absent" (that instance is not a holder, removal may proceed), NOT
    "unsupported" (which refuses).
    """
    from localm.selfclient import read_model_file_hold

    _register(home, {"other": {"path": str(_model_file(home)), "source": "test"}})

    app, port, server, th = _real_server()
    try:
        state, payload = read_model_file_hold(
            "http", port, "nope", app.state.instance_token)
    finally:
        _shutdown(server, th)

    assert state == "absent", (state, payload)


def test_client_treats_a_route_less_server_as_unsupported(home):
    """404 arm two: something that answers HTTP but has no such route - an
    older localm. Must be "unsupported", which REFUSES, not "absent".

    Served by a real socket with a real 404 body, because the whole point is
    that status alone cannot separate this from the case above.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from localm.selfclient import read_model_file_hold

    class _Old(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"detail": "Not Found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Old)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        state, payload = read_model_file_hold(
            "http", srv.server_port, "victim", "tok")
    finally:
        srv.shutdown()
        th.join(timeout=5.0)

    assert state == "unsupported", (state, payload)


def test_client_reports_a_dead_port_as_unreachable():
    """The arm that separates fail-closed from fail-open.

    A dead port reaches the caller as "I could not ask", never as an empty or
    false hold. A socket claims a port, closes it, and the test then dials it,
    so the port is one nothing is listening on rather than one guessed at.
    """
    from localm.selfclient import read_model_file_hold

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    state, payload = read_model_file_hold("http", dead_port, "victim", "tok")

    assert state == "unreachable", (state, payload)
