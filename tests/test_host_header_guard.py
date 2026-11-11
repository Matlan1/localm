# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Host header guard (security.HostHeaderGuard): a request whose Host names
a host this server does not answer to is refused before any middleware, route,
CORS reply or static file, so a page on a DNS-rebound name cannot reach the
server as same-origin."""

import asyncio
import contextlib
import datetime
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from localm import bindhost
from localm.inference import http_server
from localm.inference.app_assembly import security
from localm.inference.http_server import create_app

_PORT = 8737


@pytest.fixture(autouse=True)
def _restore_engine():
    from localm.inference import http_server as _hs
    saved = _hs._engine
    yield
    _hs._engine = saved


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv(security.ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(security, "own_host_names",
                        lambda certfile=None, refresh=False: {"localhost", "mybox", "mybox.local"})
    return home


def _write_config(home, **values):
    (home / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _app(bind_host="127.0.0.1", *, guard=True, api_landing=False):
    app = create_app(None, api_landing=api_landing)
    from types import SimpleNamespace
    from localm.inference import http_server as _hs
    _hs._engine = SimpleNamespace(display_name="m", loaded=True)
    app.state.bind_host = bind_host
    if guard:
        app.state.host_header_guard = True
    return app


def _get(client, path, host, **headers):
    return client.get(path, headers={"host": host, **headers})


_LOOPBACK_OK = [
    "localhost", f"localhost:{_PORT}", f"LOCALHOST:{_PORT}", f"localhost.:{_PORT}",
    f"LocalHost.:{_PORT}", "127.0.0.1", f"127.0.0.1:{_PORT}", f"127.0.0.2:{_PORT}",
    "[::1]", f"[::1]:{_PORT}", f"[0:0:0:0:0:0:0:1]:{_PORT}", f"app.localhost:{_PORT}",
]

_LOOPBACK_REFUSED = [
    f"evil.example:{_PORT}", "evil.example", f"EVIL.EXAMPLE.:{_PORT}",
    f"localhost.evil.example:{_PORT}", f"127.0.0.1.nip.io:{_PORT}",
    f"192.168.1.5:{_PORT}", f"[fe80::1]:{_PORT}", f"mybox.local:{_PORT}",
    "", " ", f"evil.example@127.0.0.1:{_PORT}", "[::1", f"localhost:{_PORT}/x",
    "localhost..", ".localhost", "local host", f"localhost:{_PORT}:1",
    f"[1:2:3]:{_PORT}", "[1::2::3]",
]


@pytest.mark.parametrize("host", _LOOPBACK_OK)
def test_loopback_bind_accepts_loopback_names_and_addresses(home, host):
    r = _get(TestClient(_app()), "/health", host)
    assert r.status_code == 200, (host, r.text)


@pytest.mark.parametrize("host", _LOOPBACK_REFUSED)
def test_loopback_bind_refuses_every_other_host(home, host):
    r = _get(TestClient(_app()), "/health", host)
    assert r.status_code == 400, (host, r.text)
    body = r.json()
    assert set(body) == {"detail", "error"} and body["detail"] == body["error"]
    assert body["detail"].startswith("Host not allowed")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/v1/models", None),
    ("GET", "/whoami", None),
    ("GET", "/api/tags", None),
    ("GET", "/api/version", None),
    ("POST", "/v1/chat/completions",
     {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("POST", "/v1/embeddings", {"model": "x", "input": "hi"}),
    ("POST", "/api/chat", {"model": "x", "messages": []}),
    ("POST", "/api/generate", {"model": "x", "prompt": "hi"}),
    ("POST", "/v1/messages", {"model": "x", "max_tokens": 1, "messages": []}),
    ("POST", "/v1/models/x/load", {}),
    ("PATCH", "/v1/config", {"n_ctx": 8192}),
])
def test_a_rebound_page_reaches_no_route(home, method, path, body):
    """The reproduced attack: an open-mode loopback server, a page on
    evil.example rebound to 127.0.0.1 sending its own Host and Origin."""
    client = TestClient(_app())
    host = f"evil.example:{_PORT}"
    r = client.request(method, path, json=body,
                       headers={"host": host, "origin": f"http://{host}"})
    assert r.status_code == 400, (path, r.text)
    assert r.json()["detail"] == security.host_refused_detail("evil.example")


def test_the_same_requests_still_work_on_localhost(home):
    client = TestClient(_app())
    host = f"localhost:{_PORT}"
    assert _get(client, "/v1/models", host).status_code == 200
    assert _get(client, "/whoami", host).status_code == 200
    assert _get(client, "/api/tags", host).status_code == 200


def test_a_cors_preflight_from_a_rebound_page_gets_no_cors_grant(home):
    host = f"evil.example:{_PORT}"
    r = TestClient(_app()).options(
        "/v1/chat/completions",
        headers={"host": host, "origin": f"http://{host}",
                 "access-control-request-method": "POST"})
    assert r.status_code == 400
    assert r.headers["content-type"] == "application/json", r.text
    body = r.json()
    assert body == {"detail": security.host_refused_detail("evil.example"),
                    "error": security.host_refused_detail("evil.example")}
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in r.headers


def test_the_api_landing_page_and_static_files_are_refused(home, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "a.txt").write_text("hello", encoding="utf-8")
    app = _app(api_landing=True)
    app.mount("/static-probe", StaticFiles(directory=str(static)))
    client = TestClient(app)
    assert _get(client, "/", "localhost").status_code == 200
    assert _get(client, "/static-probe/a.txt", "localhost").text == "hello"
    assert _get(client, "/", "evil.example").status_code == 400
    r = _get(client, "/static-probe/a.txt", "evil.example")
    assert r.status_code == 400
    assert "hello" not in r.text


def test_an_event_stream_route_is_refused_before_it_streams(home):
    app = _app()
    started = []

    @app.get("/sse-probe")
    async def _sse():
        started.append(True)

        async def _gen():
            yield "data: secret\n\n"
        return StreamingResponse(_gen(), media_type="text/event-stream")

    client = TestClient(app)
    r = _get(client, "/sse-probe", "evil.example")
    assert r.status_code == 400
    assert "secret" not in r.text
    assert started == []
    r = _get(client, "/sse-probe", "localhost")
    assert r.status_code == 200 and "secret" in r.text


def test_a_websocket_from_a_foreign_host_is_closed_before_accept(home):
    app = _app()
    accepted = []

    @app.websocket("/ws-probe")
    async def _ws(websocket: WebSocket):
        accepted.append(True)
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws-probe", headers={"host": "evil.example"}):
            pass
    assert exc.value.code == 1008
    assert accepted == []
    with client.websocket_connect("/ws-probe", headers={"host": "localhost"}) as ws:
        assert ws.receive_text() == "hello"
    assert accepted == [True]


# --------------------------------------------------------------------------- #
# Raw ASGI: missing, repeated and non-UTF-8 Host headers
# --------------------------------------------------------------------------- #

def _run_guard(scope_type, headers, state):
    calls = []
    sent = []

    async def _inner(scope, receive, send):
        calls.append(scope["type"])

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message):
        sent.append(message)

    scope = {"type": scope_type, "path": "/", "method": "GET", "headers": headers,
             "query_string": b"", "app": SimpleNamespace(state=state)}
    asyncio.run(security.HostHeaderGuard(_inner)(scope, _receive, _send))
    return calls, sent


def _state(**kw):
    return SimpleNamespace(host_header_guard=True, bind_host="127.0.0.1", **kw)


@pytest.mark.parametrize("headers", [
    [],
    [(b"host", b"")],
    [(b"host", b"localhost"), (b"host", b"localhost")],
    [(b"host", b"localhost"), (b"host", b"evil.example")],
    [(b"host", "löcalhost".encode("latin-1"))],
])
def test_missing_repeated_or_odd_host_headers_are_refused(headers):
    calls, sent = _run_guard("http", headers, _state())
    assert calls == []
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 400


def test_a_websocket_scope_with_a_foreign_host_is_closed_with_1008():
    calls, sent = _run_guard("websocket", [(b"host", b"evil.example")], _state())
    assert calls == []
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_a_lifespan_scope_passes_through():
    calls, _sent = _run_guard("lifespan", [], _state())
    assert calls == ["lifespan"]


def test_the_guard_is_off_until_the_flag_is_set():
    calls, sent = _run_guard("http", [(b"host", b"evil.example")],
                             SimpleNamespace(bind_host="127.0.0.1"))
    assert calls == ["http"]
    assert sent == []


def test_a_test_app_without_the_flag_answers_any_host(home):
    r = _get(TestClient(_app(guard=False)), "/health", "evil.example")
    assert r.status_code == 200


def test_run_advertised_turns_the_guard_on(monkeypatch, home):
    from localm import instances, portmux
    monkeypatch.delenv(http_server._GUI_MOUNTED_ENV, raising=False)
    app = create_app(None)
    app.state.bind_host = "127.0.0.1"
    seen = {}

    def _run_server(served_app, host, port, **kw):
        seen["guard"] = getattr(served_app.state, "host_header_guard", False)
        seen["certfile"] = served_app.state.tls_certfile
        seen["status"] = _get(TestClient(served_app), "/health", "evil.example").status_code

    monkeypatch.setattr(portmux, "run_server", _run_server)
    monkeypatch.setattr(instances, "advertise",
                        lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(http_server, "_announce_stopping", lambda: None)
    monkeypatch.setattr(http_server, "_shutdown_teardown", lambda **kw: None)
    http_server.run_advertised(app, "127.0.0.1", 8642, mode="api",
                               ssl_certfile="cert.pem", ssl_keyfile="key.pem")
    assert seen == {"guard": True, "certfile": "cert.pem", "status": 400}


# --------------------------------------------------------------------------- #
# Network bind
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("host", [
    "192.168.1.5:8642", "10.0.0.7", "100.101.102.103:8642", "[fe80::1]:8642",
    "[2001:db8::5]", "mybox.local:8642", "MyBox.Local.:8642", "mybox",
    "localhost:8642", "127.0.0.1:8642", "0.0.0.0:8642",
])
def test_network_bind_accepts_addresses_and_own_names(home, host):
    r = _get(TestClient(_app("0.0.0.0")), "/health", host)
    assert r.status_code == 200, (host, r.text)


@pytest.mark.parametrize("host", [
    "evil.example:8642", "mybox.evil.example", "evil.mybox.local", "", "[::1",
])
def test_network_bind_refuses_foreign_names(home, host):
    r = _get(TestClient(_app("0.0.0.0")), "/health", host)
    assert r.status_code == 400, (host, r.text)


def test_a_specific_network_bind_also_accepts_its_own_address(home):
    client = TestClient(_app("192.168.1.5"))
    assert _get(client, "/health", "192.168.1.5:8642").status_code == 200
    assert _get(client, "/health", "mybox.local").status_code == 200
    assert _get(client, "/health", "evil.example").status_code == 400


# --------------------------------------------------------------------------- #
# The allowed_hosts setting
# --------------------------------------------------------------------------- #

def test_allowed_hosts_adds_names_on_a_loopback_bind(home):
    _write_config(home, allowed_hosts=["ai.example.com", "*.lan"])
    client = TestClient(_app())
    assert _get(client, "/health", "AI.example.com:443").status_code == 200
    assert _get(client, "/health", "box.lan").status_code == 200
    assert _get(client, "/health", "a.box.lan").status_code == 400
    assert _get(client, "/health", "lan").status_code == 400
    assert _get(client, "/health", "evil.example").status_code == 400


def test_allowed_hosts_star_accepts_any_well_formed_host(home):
    _write_config(home, allowed_hosts="*")
    client = TestClient(_app())
    assert _get(client, "/health", "evil.example").status_code == 200
    assert _get(client, "/health", "[::1").status_code == 400


def test_the_environment_variable_adds_names(home, monkeypatch):
    monkeypatch.setenv(security.ALLOWED_HOSTS_ENV, "localm, NAS.local ,my_box")
    _write_config(home, allowed_hosts=["ai.example.com"])
    client = TestClient(_app("0.0.0.0"))
    for host in ("localm:8642", "nas.local", "my_box:8642", "ai.example.com"):
        assert _get(client, "/health", host).status_code == 200, host
    assert _get(client, "/health", "nas").status_code == 400


def test_the_environment_variable_star_turns_the_check_off(home, monkeypatch):
    monkeypatch.setenv(security.ALLOWED_HOSTS_ENV, "*")
    assert _get(TestClient(_app()), "/health", "evil.example").status_code == 200


def test_a_hand_edited_string_setting_is_read_as_a_list(home):
    _write_config(home, allowed_hosts="proxy.example, other.example")
    client = TestClient(_app())
    assert _get(client, "/health", "proxy.example").status_code == 200
    assert _get(client, "/health", "other.example").status_code == 200


def test_unusable_hand_edited_entries_are_logged_and_ignored(home, caplog):
    _write_config(home, allowed_hosts=["ok.example", "http://bad.example", 7])
    client = TestClient(_app())
    with caplog.at_level(logging.WARNING, logger="localm.security"):
        assert _get(client, "/health", "ok.example").status_code == 200
    assert _get(client, "/health", "bad.example").status_code == 400
    ignored = [r.getMessage() for r in caplog.records if "ignoring" in r.getMessage()]
    assert any("'http://bad.example'" in m for m in ignored), ignored
    assert any("7" in m for m in ignored), ignored


def test_a_setting_of_the_wrong_type_is_logged_and_ignored(home, caplog):
    _write_config(home, allowed_hosts={"a": 1})
    with caplog.at_level(logging.WARNING, logger="localm.security"):
        assert _get(TestClient(_app()), "/health", "a").status_code == 400
    assert any("expected a list of host names" in r.getMessage()
               for r in caplog.records)


def test_the_refusal_names_the_exact_fix(home):
    r = _get(TestClient(_app()), "/v1/models", "host.docker.internal:8642")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "'host.docker.internal'" in detail
    assert "add host.docker.internal to the 'allowed_hosts' setting" in detail
    assert "LOCALM_ALLOWED_HOSTS=host.docker.internal" in detail
    assert r.json()["error"] == detail


def test_a_malformed_host_gets_the_generic_refusal(home):
    r = _get(TestClient(_app()), "/health", "[::1")
    assert r.json()["detail"] == security.HOST_REFUSED_DETAIL


def test_host_docker_internal_is_not_accepted_by_default(home):
    for bind in ("127.0.0.1", "0.0.0.0"):
        r = _get(TestClient(_app(bind)), "/health", "host.docker.internal:8642")
        assert r.status_code == 400, bind


def test_host_docker_internal_is_accepted_once_listed(home, monkeypatch):
    monkeypatch.setenv(security.ALLOWED_HOSTS_ENV, "host.docker.internal")
    r = _get(TestClient(_app("0.0.0.0")), "/health", "host.docker.internal:8642")
    assert r.status_code == 200


def test_a_tailscale_name_that_comes_up_later_is_accepted_after_the_refresh_interval(
        home, monkeypatch):
    names = {"localhost", "mybox"}
    calls = []

    def _own(certfile=None, refresh=False):
        calls.append(refresh)
        return set(names)

    clock = [1000.0]
    monkeypatch.setattr(security, "own_host_names", _own)
    monkeypatch.setattr(security, "_monotonic", lambda: clock[0])
    client = TestClient(_app("0.0.0.0"))
    assert _get(client, "/health", "mybox.tail1.ts.net").status_code == 400
    names.add("mybox.tail1.ts.net")
    clock[0] += security.HOST_POLICY_REFRESH_SECONDS - 1
    assert _get(client, "/health", "mybox.tail1.ts.net").status_code == 400
    clock[0] += 2
    assert _get(client, "/health", "mybox.tail1.ts.net").status_code == 200
    assert calls == [False, True]
    assert _get(client, "/health", "evil.example").status_code == 400
    assert calls == [False, True]


def test_the_refresh_calls_tailscale_status_again(monkeypatch):
    from localm import netname
    refreshed = []
    monkeypatch.setattr(netname, "tailscale_status",
                        lambda refresh=False: refreshed.append(refresh) or None)
    monkeypatch.setattr(netname, "cert_hostnames", lambda: [])
    security.own_host_names(None, refresh=True)
    security.own_host_names(None)
    assert refreshed == [True]


def test_concurrent_requests_share_one_build(home, monkeypatch):
    import threading
    import time

    import httpx

    built = []
    entered = threading.Event()

    def _slow(*a, **kw):
        built.append(a)
        entered.set()
        time.sleep(0.2)
        return security.HostPolicy(frozenset({"ok.example"}), frozenset(), False)

    monkeypatch.setattr(security, "build_host_policy", _slow)
    app = _app()

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
            hosts = ["ok.example", "evil.example"] * 5
            return await asyncio.gather(*(c.get("/health", headers={"host": h})
                                          for h in hosts))

    replies = asyncio.run(_run())
    assert [r.status_code for r in replies] == [200, 400] * 5
    assert len(built) == 1


def test_the_policy_is_built_once_per_app(home, monkeypatch):
    built = []
    real = security.build_host_policy

    def _counting(*a, **kw):
        built.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(security, "build_host_policy", _counting)
    client = TestClient(_app())
    for _ in range(3):
        assert _get(client, "/health", "evil.example").status_code == 400
    assert _get(client, "/health", "localhost").status_code == 200
    assert built == [("127.0.0.1", None, False)]


def test_only_the_first_refusal_is_a_warning(home, caplog):
    client = TestClient(_app())
    with caplog.at_level(logging.DEBUG, logger="localm.security"):
        for _ in range(3):
            _get(client, "/health", "evil.example")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG
              and "refused a request" in r.getMessage()]
    assert len(warnings) == 1
    assert "'evil.example'" in warnings[0].getMessage()
    assert "allowed_hosts" in warnings[0].getMessage()
    assert len(debugs) == 2


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("  ", None), ("*", "*"),
    ("Proxy.Example.COM., *.lan, [::1]:8642, 10.0.0.5:80, ::1",
     ["proxy.example.com", "*.lan", "::1", "10.0.0.5"]),
    (["a.example", "A.example", " "], ["a.example"]),
])
def test_allowed_hosts_setting_normalises(value, expected):
    from localm.settings_schema import _coerce_allowed_hosts
    assert _coerce_allowed_hosts(value) == expected


@pytest.mark.parametrize("value", [
    "http://x.example", "a b", "*, x.example", "x/y", "*.", ["x", 3],
    ",".join(f"h{i}.example" for i in range(501)),
])
def test_allowed_hosts_setting_refuses_bad_values(value):
    from localm.settings_schema import _coerce_allowed_hosts
    with pytest.raises(ValueError, match="allowed_hosts"):
        _coerce_allowed_hosts(value)


def test_allowed_hosts_goes_through_the_settings_validation():
    from localm.settings_schema import validate_update
    assert validate_update({"allowed_hosts": "a.example, b.example"}) == {
        "allowed_hosts": ["a.example", "b.example"]}


# --------------------------------------------------------------------------- #
# Name helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    ("localhost", "localhost"), ("LOCALHOST:80", "localhost"),
    ("localhost.", "localhost"), ("[::1]:8642", "::1"),
    ("[0:0:0:0:0:0:0:1]", "::1"), ("10.0.0.1:1", "10.0.0.1"),
    ("", None), ("localhost:", None), ("[::1", None), ("a@b", None),
    ("a..b", None), (".a", None), ("a:1:2", None), ("[zz::1]", None),
    ("[1:2:3]", None), ("[1:2:3]:80", None), ("[1::2::3]", None),
    ("My_Box:8642", "my_box"),
])
def test_host_header_name(value, expected):
    assert bindhost.host_header_name(value) == expected


def _self_signed(path, dns_names):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "probe")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in dns_names]), critical=False)
            .sign(key, hashes.SHA256()))
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def test_own_host_names_covers_the_machine_and_the_certificate(monkeypatch, tmp_path):
    from localm import netname
    monkeypatch.setattr(security.socket, "gethostname", lambda: "MyBox")
    monkeypatch.setattr(security.socket, "getfqdn", lambda: "mybox.corp.example.")
    monkeypatch.setattr(netname, "cert_hostnames",
                        lambda: ["localm.local", "mybox.local", "mybox.tail1.ts.net"])
    cert = tmp_path / "cert.pem"
    _self_signed(cert, ["AI.Example.com", "*.lan.example"])
    names = security.own_host_names(str(cert))
    assert {"localhost", "mybox", "mybox.corp.example", "localm.local",
            "mybox.local", "mybox.tail1.ts.net", "ai.example.com",
            "*.lan.example"} <= names


def test_a_certificate_wildcard_matches_one_label(monkeypatch, tmp_path):
    from localm import netname
    monkeypatch.setattr(netname, "cert_hostnames", lambda: [])
    cert = tmp_path / "cert.pem"
    _self_signed(cert, ["*.lan.example"])
    policy = security.build_host_policy("0.0.0.0", str(cert))
    assert policy.allows("box.lan.example")
    assert not policy.allows("a.box.lan.example")
    assert not policy.allows("lan.example")


def test_an_unreadable_certificate_is_a_warning_not_a_crash(monkeypatch, tmp_path, caplog):
    from localm import netname
    monkeypatch.setattr(netname, "cert_hostnames", lambda: [])
    bad = tmp_path / "bad.pem"
    bad.write_text("not a certificate", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="localm.security"):
        names = security.own_host_names(str(bad))
    assert "localhost" in names
    assert any("could not read the certificate names" in r.getMessage()
               for r in caplog.records)
    missing = tmp_path / "missing.pem"
    assert "localhost" in security.own_host_names(str(missing))
