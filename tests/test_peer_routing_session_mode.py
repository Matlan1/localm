# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat answered by another localm instance keeps THIS instance's session
mode: unless that mode is full, the peer must first confirm (``GET /api/session``
with ``X-Localm-Session-Mode``, field ``session_mode``) that it records
the chat no more than this mode allows. Covers the HTTP route forward, route
acceptance, a scheduled chat job on a routed model, and the MCP server's
``--share-loaded-models``. The peer is a real HTTP server on loopback."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from localm import gpu_registry, peer_routing
from localm.audit import SESSION_MODE_HEADER
from localm.inference import http_server as hs
from localm.inference.http_server import create_app

_PEERS: list = []
_RANK = {"privacy": 0, "log": 1, "full": 2}


class _Peer:
    """A peer localm: answers ``GET /api/session`` with ``session_mode``
    (its own mode, narrowed to the requested one when the caller presents
    *owner_key*; omitted when *reports* is False, like an older localm), and
    ``POST /v1/chat/completions`` with a reply. Records every chat body and the
    session-mode header of every request."""

    def __init__(self, own_mode="privacy", owner_key=None, reports=True):
        outer = self
        self.own_mode, self.owner_key, self.reports = own_mode, owner_key, reports
        self.bodies, self.mode_headers, self.config_gets = [], [], 0

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                asked = self.headers.get(SESSION_MODE_HEADER)
                outer.mode_headers.append(asked)
                if self.path.startswith("/api/session"):
                    outer.config_gets += 1
                    mode = outer.own_mode
                    owner = (outer.owner_key is not None and self.headers.get(
                        "Authorization") == f"Bearer {outer.owner_key}")
                    if owner and asked in _RANK and _RANK[asked] < _RANK[mode]:
                        mode = asked
                    body = {"authed": bool(self.headers.get("Authorization")),
                            "required": outer.owner_key is not None}
                    if outer.reports:
                        body["session_mode"] = mode
                    return self._send(200, body)
                self._send(200, {"object": "list", "data": []})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                outer.mode_headers.append(
                    self.headers.get(SESSION_MODE_HEADER))
                outer.bodies.append(body)
                self._send(200, {"model": body.get("model"), "choices": [
                    {"message": {"role": "assistant", "content": "peer answered"},
                     "finish_reason": "stop"}]})

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    _PEERS.clear()
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr("localm.audit._SESSIONS_DIR", tmp_path / "sessions")

    def _live(*, exclude_self_id=None, timeout=0.7):
        return [dict(p) for p in _PEERS
                if p["instance_id"] != exclude_self_id and p["pid"] != os.getpid()]

    monkeypatch.setattr(gpu_registry, "list_gpu_peers", _live)
    hs._gpu_coord = None
    peer_routing._ROUTES.clear()
    yield
    hs._gpu_coord = None
    peer_routing._ROUTES.clear()
    _PEERS.clear()


@pytest.fixture
def peers():
    made = []

    def _make(**kw):
        p = _Peer(**kw)
        made.append(p)
        return p

    yield _make
    for p in made:
        p.close()


def _mode(monkeypatch, mode=None, chat_mode=None):
    import localm.config as cfg
    data = {}
    if mode:
        data["mode"] = mode
    if chat_mode:
        data["chat_mode"] = chat_mode
    monkeypatch.setattr(cfg, "load_config", lambda: dict(data))


def _route(peer, key=""):
    return peer_routing.PeerRoute(model="m", instance_id="peer-1", host="127.0.0.1",
                                  port=peer.port, scheme="http", api_key=key)


class _Req:
    def __init__(self):
        self.headers = {"content-type": "application/json"}

    async def body(self):
        return b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'


def _forward(route):
    async def go():
        return await peer_routing.forward(route, _Req(), "/v1/chat/completions")
    return asyncio.run(go())


def _forward_error(route) -> HTTPException:
    try:
        _forward(route)
    except HTTPException as e:
        return e
    raise AssertionError("forward() sent the request instead of refusing it")


# --------------------------------------------------------------------------- #
#  forward()                                                                   #
# --------------------------------------------------------------------------- #

def test_a_privacy_instance_forwards_to_a_privacy_peer_and_names_the_mode(
        monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")

    resp = _forward(_route(peer))

    assert resp.status_code == 200
    assert len(peer.bodies) == 1
    assert peer.mode_headers[-1] == "privacy"


def test_a_privacy_instance_does_not_forward_to_a_logging_peer(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="log")
    route = _route(peer)
    peer_routing.set_route(route)

    err = _forward_error(route)

    assert peer.bodies == []
    assert err.status_code == 409
    assert f"127.0.0.1:{peer.port}" in err.detail and "log mode" in err.detail
    assert peer_routing.get_route("m") is route


def test_a_peer_that_does_not_report_its_mode_gets_nothing(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy", reports=False)

    err = _forward_error(_route(peer))

    assert peer.bodies == []
    assert err.status_code == 409
    assert "did not confirm" in err.detail


def test_the_owner_key_lets_a_logging_peer_record_less(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="full", owner_key="owner-k")

    resp = _forward(_route(peer, key="owner-k"))

    assert resp.status_code == 200
    assert len(peer.bodies) == 1
    assert peer.mode_headers[-1] == "privacy"


def test_a_scoped_key_cannot_make_a_logging_peer_record_less(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="full", owner_key="owner-k")

    err = _forward_error(_route(peer, key="scoped-k"))

    assert peer.bodies == []
    assert err.status_code == 409


@pytest.mark.parametrize("peer_mode,sent", [("privacy", True), ("log", True),
                                            ("full", False)])
def test_a_log_instance_needs_a_peer_that_records_no_more_than_log(
        monkeypatch, peers, peer_mode, sent):
    _mode(monkeypatch, mode="log")
    peer = peers(own_mode=peer_mode)

    if sent:
        assert _forward(_route(peer)).status_code == 200
        assert len(peer.bodies) == 1
    else:
        assert _forward_error(_route(peer)).status_code == 409
        assert peer.bodies == []


def test_a_full_instance_forwards_without_asking(monkeypatch, peers):
    _mode(monkeypatch, mode="full")
    peer = peers(own_mode="full", reports=False)

    assert _forward(_route(peer)).status_code == 200

    assert peer.config_gets == 0
    assert len(peer.bodies) == 1
    assert peer.mode_headers[-1] == "full"


def test_the_chat_route_refuses_end_to_end(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="log")
    peer_routing.set_route(_route(peer))
    client = TestClient(create_app(None))

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})

    assert peer.bodies == []
    assert r.status_code == 409, r.text
    assert "log mode" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  Accepting a route                                                           #
# --------------------------------------------------------------------------- #

def _offer(peer, iid):
    _PEERS.append({"instance_id": iid, "pid": os.getpid() + 1, "port": peer.port,
                   "host": "127.0.0.1", "scheme": "http", "model": "shared-model",
                   "vram_estimate_bytes": None, "gpu_index": 0})


def _accept(iid):
    with TestClient(create_app(None)) as client:
        return client.post("/v1/models/shared-model/peer-route",
                           headers={"Authorization": f"Bearer {client.app.state.shell_token}"},
                           json={"instance_id": iid})


def test_accepting_a_route_to_a_logging_peer_in_privacy_mode_is_refused(
        monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="log")
    _offer(peer, "logging-peer")

    r = _accept("logging-peer")

    assert peer_routing.get_route("shared-model") is None
    assert r.status_code == 409, r.text
    assert "log mode" in r.json()["detail"]


def test_accepting_a_route_follows_a_private_chat_override(monkeypatch, peers):
    _mode(monkeypatch, mode="log", chat_mode="privacy")
    peer = peers(own_mode="log")
    _offer(peer, "logging-peer-2")

    r = _accept("logging-peer-2")

    assert r.status_code == 409, r.text
    assert peer_routing.get_route("shared-model") is None


def test_accepting_a_route_to_a_privacy_peer_works(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")
    _offer(peer, "private-peer")

    r = _accept("private-peer")

    assert r.status_code == 200, r.text
    assert peer_routing.get_route("shared-model") is not None


# --------------------------------------------------------------------------- #
#  Scheduled chat job on a routed model                                        #
# --------------------------------------------------------------------------- #

def test_a_job_is_not_sent_to_a_peer_that_would_record_it(monkeypatch, peers):
    from localm.plugins.builtin.jobs import runner
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="full")

    with pytest.raises(RuntimeError) as exc:
        runner._peer_served_engine(_route(peer))

    assert "full mode" in str(exc.value)
    assert peer.bodies == []


def test_a_job_sent_to_a_peer_names_the_mode_on_every_request(monkeypatch, peers):
    from localm.plugins.builtin.jobs import runner
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")

    eng = runner._peer_served_engine(_route(peer))
    out = "".join(eng.chat_stream([{"role": "user", "content": "hi"}]))

    assert len(peer.bodies) == 1
    assert peer.mode_headers[-1] == "privacy"
    assert out is not None


# --------------------------------------------------------------------------- #
#  MCP server --share-loaded-models                                            #
# --------------------------------------------------------------------------- #

def _mcp_cache(monkeypatch, peer):
    from localm.plugins.mcpserver.server import EngineCache
    target = {"instance_id": "peer-mcp", "host": "127.0.0.1", "port": peer.port,
              "scheme": "http", "matched_model": "m"}
    monkeypatch.setattr(peer_routing, "find_offer", lambda *a, **k: dict(target))
    monkeypatch.setattr(peer_routing, "registry_name_and_aliases",
                        lambda reg, name: (name, frozenset()))
    monkeypatch.setattr(peer_routing, "local_identity", lambda reg, name: {})
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    monkeypatch.setattr("localm.instances.list_entries", lambda *a, **k: [])
    return EngineCache("m", share_loaded=True, engine_factory=lambda n: None)


def test_mcp_does_not_use_a_peer_that_would_record_the_chat(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="log")
    cache = _mcp_cache(monkeypatch, peer)

    assert cache._peer_engine("m") is None


def test_mcp_uses_a_private_peer_and_names_the_mode(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")
    cache = _mcp_cache(monkeypatch, peer)

    eng = cache._peer_engine("m")
    assert eng is not None
    "".join(eng.chat_stream([{"role": "user", "content": "hi"}]))

    assert peer.mode_headers[-1] == "privacy"


def test_mcp_stops_using_a_peer_that_switched_to_recording(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")
    cache = _mcp_cache(monkeypatch, peer)
    assert cache._peer_engine("m") is not None

    peer.own_mode = "full"

    assert cache._peer_still_answers("m") is False


# --------------------------------------------------------------------------- #
#  The peer side: GET /api/session session_mode, and a real round trip        #
# --------------------------------------------------------------------------- #

def _real_engine():
    from unittest.mock import MagicMock
    engine = MagicMock()
    engine.display_name = "m"
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: (t for t in ["peer", " reply"])
    engine.count_tokens.side_effect = lambda text: max(1, len(text.split()))
    return engine


def _peer_app(tmp_path, monkeypatch, mode):
    _mode(monkeypatch, mode=mode)
    sessions = tmp_path / "peer-sessions"
    sessions.mkdir()
    monkeypatch.setattr("localm.audit._SESSIONS_DIR", sessions)
    return create_app(_real_engine()), sessions


def test_session_reports_the_server_mode_without_a_header(tmp_path, monkeypatch):
    app, _ = _peer_app(tmp_path, monkeypatch, "log")
    with TestClient(app) as c:
        r = c.get("/api/session")
    assert r.json()["session_mode"] == "log"


@pytest.mark.parametrize("asked,expected", [("privacy", "privacy"), ("full", "log"),
                                            ("bogus", "privacy")])
def test_an_open_instance_reports_any_callers_narrowing(tmp_path, monkeypatch,
                                                        asked, expected):
    app, _ = _peer_app(tmp_path, monkeypatch, "log")
    with TestClient(app) as c:
        r = c.get("/api/session", headers={SESSION_MODE_HEADER: asked})
    assert r.status_code == 200, r.text
    assert r.json()["session_mode"] == expected


@pytest.mark.parametrize("who", ["scoped", "nobody"])
def test_a_keyed_instance_ignores_a_non_owners_narrowing(tmp_path, monkeypatch, who):
    from localm import auth
    app, _ = _peer_app(tmp_path, monkeypatch, "log")
    auth.set_api_key("owner-key-session-mode-0123456789")
    headers = {SESSION_MODE_HEADER: "privacy"}
    if who == "scoped":
        scoped = auth.create_key("chatter", ["chat"], allow_privileged=False)
        headers["Authorization"] = f"Bearer {scoped['key']}"
    with TestClient(app) as c:
        r = c.get("/api/session", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["session_mode"] == "log"


def test_a_keyed_instance_reports_its_owners_narrowing(tmp_path, monkeypatch):
    from localm import auth
    app, _ = _peer_app(tmp_path, monkeypatch, "log")
    auth.set_api_key("owner-key-session-mode-0123456789")
    with TestClient(app) as c:
        r = c.get("/api/session", headers={
            SESSION_MODE_HEADER: "privacy",
            "Authorization": "Bearer owner-key-session-mode-0123456789"})
    assert r.json()["session_mode"] == "privacy"


class _Bridge:
    """``requests.get`` / ``requests.post`` for peer_routing, answered by a
    TestClient of a real localm app (the peer)."""

    def __init__(self, client):
        self.client = client

    def get(self, url, headers=None, **kw):
        from urllib.parse import urlsplit
        return self.client.get(urlsplit(url).path, headers=headers or {})

    def post(self, url, data=None, headers=None, **kw):
        from urllib.parse import urlsplit
        resp = self.client.post(urlsplit(url).path, content=data, headers=headers or {})

        class _R:
            status_code = resp.status_code
            headers = resp.headers

            def iter_content(self, chunk_size=None):
                return iter([resp.content])

            def close(self):
                pass

        return _R()


def _round_trip(tmp_path, monkeypatch, peer_mode, local_mode):
    from localm.audit import SessionMode
    app, sessions = _peer_app(tmp_path, monkeypatch, peer_mode)
    with TestClient(app) as peer_client:
        bridge = _Bridge(peer_client)
        monkeypatch.setattr("requests.get", bridge.get)
        monkeypatch.setattr("requests.post", bridge.post)
        monkeypatch.setattr(peer_routing, "_local_request_mode",
                            lambda request: SessionMode(local_mode))
        route = peer_routing.PeerRoute(model="m", instance_id="real-peer",
                                       host="127.0.0.1", port=1, scheme="http",
                                       api_key="")

        async def go():
            resp = await peer_routing.forward(route, _Req(), "/v1/chat/completions")
            return resp, b"".join([c async for c in resp.body_iterator])
        resp, body = asyncio.run(go())
    recorded = "".join(p.read_text(encoding="utf-8") for p in sessions.iterdir()
                       if p.is_file())
    return resp.status_code, body, recorded


def test_a_real_full_mode_peer_records_nothing_for_a_privacy_forward(
        tmp_path, monkeypatch):
    status, body, recorded = _round_trip(tmp_path, monkeypatch, "full", "privacy")

    assert status == 200
    assert b"peer reply" in body
    assert "peer reply" not in recorded and '"hi"' not in recorded


def test_a_real_full_mode_peer_records_a_full_mode_forward(tmp_path, monkeypatch):
    status, body, recorded = _round_trip(tmp_path, monkeypatch, "full", "full")

    assert status == 200
    assert b"peer reply" in body
    assert "peer reply" in recorded and '"hi"' in recorded


# --------------------------------------------------------------------------- #
#  A peer that stops answering or refuses the key is dropped, in every mode    #
# --------------------------------------------------------------------------- #

def test_a_dead_peer_drops_the_route_in_privacy_mode(monkeypatch, peers):
    _mode(monkeypatch, mode="privacy")
    peer = peers(own_mode="privacy")
    route = _route(peer)
    peer_routing.set_route(route)
    peer.close()

    err = _forward_error(route)

    assert err.status_code == 502
    assert "cleared" in err.detail
    assert peer_routing.get_route("m") is None


def test_a_peer_refusing_the_key_drops_the_route_in_privacy_mode(monkeypatch):
    import http.server as hsrv

    class H(hsrv.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = hsrv.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _mode(monkeypatch, mode="privacy")
        route = peer_routing.PeerRoute(model="m", instance_id="peer-1",
                                       host="127.0.0.1", port=srv.server_address[1],
                                       scheme="http", api_key="stale")
        peer_routing.set_route(route)

        err = _forward_error(route)
    finally:
        srv.shutdown()
        srv.server_close()

    assert err.status_code == 502
    assert "HTTP 401" in err.detail
    assert peer_routing.get_route("m") is None


def test_mcp_names_the_mode_checked_last_not_the_one_it_started_with(
        monkeypatch, peers):
    _mode(monkeypatch, mode="full")
    peer = peers(own_mode="full")
    cache = _mcp_cache(monkeypatch, peer)
    eng = cache._peer_engine("m")
    assert eng is not None
    cache._peers["m"] = eng

    _mode(monkeypatch, mode="privacy")
    peer.own_mode = "privacy"
    assert cache._peer_still_answers("m") is True
    "".join(eng.chat_stream([{"role": "user", "content": "hi"}]))

    assert peer.mode_headers[-1] == "privacy"
