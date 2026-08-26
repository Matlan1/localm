# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /api/instances + POST /api/instances/{id}/stop - the GUI form of
`localm ps` / `localm stop <id>`.

SECURITY: the stop route reaches OUTSIDE the calling instance's own blast
radius (it can end a DIFFERENT process, possibly serving an unrelated project),
unlike /v1/server/shutdown|restart which only ever act on app.state.instance_id.
It is gated on scopes.ADMIN rather than the sibling routes' CONFIG_WRITE, and
TestAdminGate below is the test that catches a regression back to CONFIG_WRITE
or to no gate at all.

TestStopRealHttp runs a REAL uvicorn server on a throwaway loopback port, so
the graceful-shutdown path exercises the real /v1/server/shutdown route and its
real auth gate rather than a mocked `requests` call. Both sides stay in open
mode: LOCALM_HOME, and therefore auth.key, is shared between "my" app and any
real target built in this file, so minting a key anywhere knocks BOTH out of
open mode at once. TestStopGracefulDeclined covers the 401/403-declined branch,
which needs a differently-credentialed target that the shared-home constraint
rules out building for real, so it is mocked at requests.request (what
selfclient.self_request calls). TestStopKillFallback spawns a REAL killable
subprocess, so the direct-kill fallback is proven against a real OS process.
"""

from __future__ import annotations

import asyncio
import socket as _socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.inference.http_server as _hs
from localm import instances
from localm import scopes as S
from localm.inference.http_server import create_app


@pytest.fixture
def instances_app(tmp_path, monkeypatch):
    """Full stack (kernel + GUI) on a throwaway home. instances.py/auth.py both
    resolve LOCALM_HOME lazily via config.home_dir() on every call, so setting
    the env var is enough - unlike the model registry, nothing here caches a
    module-level path constant that would also need patching."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "")
    return app, home


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _admin_key():
    from localm import auth
    return auth.create_key("owner", [S.ADMIN], allow_privileged=True)["key"]


def _config_write_key():
    """Privileged, but NOT the owner - must not reach the stop route."""
    from localm import auth
    return auth.create_key("write-only", [S.CONFIG_WRITE],
                           allow_privileged=True)["key"]


def _config_read_key():
    from localm import auth
    return auth.create_key("reader", [S.CONFIG_READ])["key"]


def _no_scope_key():
    from localm import auth
    return auth.create_key("narrow", [S.MCP])["key"]


def _write_entry(home, **kw):
    """Write a registry entry file directly, for an arbitrary (possibly fake)
    pid - instances.register_instance() always uses THIS process's own
    os.getpid(), so it cannot represent a different target process. Matches
    register_instance()'s own field schema exactly.

    Default pid is THIS TEST PROCESS's own (genuinely alive for the test's
    whole lifetime), not a fake number: both routes call
    instances.reap_stale()/snapshot(reap=True) BEFORE doing anything else, and
    that reaps any entry whose pid does not check out alive, deleting the
    fixture before it can be matched. A test that deliberately wants an
    unreachable or dead pid must pass its own `pid=` AND neutralise the reap
    (monkeypatch instances.reap_stale to a no-op); overriding pid alone is not
    enough."""
    import json
    import os
    defaults = dict(instance_id="aaaa1111bbbb2222", pid=os.getpid(), port=59999,
                    host="127.0.0.1", scheme="http", root_dir="/proj/one",
                    mode="api", version="test", token=None,
                    started="2026-08-20T00:00:00+00:00")
    defaults.update(kw)
    defaults["token"] = defaults["token"] or instances.new_token()
    path = instances.registry_path(home, defaults["instance_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    return defaults["instance_id"]


# --------------------------------------------------------------------------- #
#  GET /api/instances                                                         #
# --------------------------------------------------------------------------- #

class TestInstancesList:
    def test_lists_a_registered_instance(self, instances_app):
        app, home = instances_app
        _write_entry(home, instance_id="cafef00d01234567", root_dir="/proj/a",
                  port=59991, mode="full")
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_config_read_key()))
            assert r.status_code == 200, r.text
            rows = r.json()["instances"]
            assert len(rows) == 1
            row = rows[0]
            assert row["instance_id"] == "cafef00d01234567"
            assert row["root_dir"] == "/proj/a"
            assert row["mode"] == "full"
            assert row["port"] == 59991
            assert row["alive"] is False   # nothing is actually listening
            assert row["self"] is False

    def test_never_leaks_the_attach_token_or_the_registry_file_path(self, instances_app):
        app, home = instances_app
        _write_entry(home, instance_id="secretbearer0001", token="do-not-leak-me")
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_config_read_key()))
            assert r.status_code == 200, r.text
            row = r.json()["instances"][0]
            assert "token" not in row
            assert "_path" not in row
            assert "do-not-leak-me" not in r.text

    def test_flags_the_instance_serving_the_request_as_self(self, instances_app):
        app, home = instances_app
        app.state.instance_id = "cafef00d01234567"
        _write_entry(home, instance_id="cafef00d01234567", root_dir="/proj/self")
        _write_entry(home, instance_id="deadbeef76543210", root_dir="/proj/other",
                  port=59992)
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_config_read_key()))
            assert r.status_code == 200, r.text
            by_id = {row["instance_id"]: row for row in r.json()["instances"]}
            assert by_id["cafef00d01234567"]["self"] is True
            assert by_id["deadbeef76543210"]["self"] is False

    def test_bracketed_ipv6_address(self, instances_app):
        """The Address column brackets an IPv6 literal, so its colons do not
        merge with the port separator."""
        app, home = instances_app
        _write_entry(home, instance_id="v6instance00001", host="::", port=59993)
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_config_read_key()))
            assert r.status_code == 200, r.text
            row = r.json()["instances"][0]
            assert row["address"] == "http://[::]:59993"

    def test_empty_registry_is_an_empty_list_not_an_error(self, instances_app):
        app, home = instances_app
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_config_read_key()))
            assert r.status_code == 200, r.text
            assert r.json()["instances"] == []

    def test_requires_config_read_scope(self, instances_app):
        app, home = instances_app
        with TestClient(app) as c:
            r = c.get("/api/instances", headers=_hdr(_no_scope_key()))
            assert r.status_code == 403, r.text

    def test_succeeds_in_open_mode(self, instances_app):
        app, home = instances_app
        _write_entry(home)
        with TestClient(app) as c:
            r = c.get("/api/instances")
            assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop - matching and validation (no live target)   #
# --------------------------------------------------------------------------- #

class TestStopMatching:
    def test_404_when_no_instance_matches(self, instances_app):
        app, home = instances_app
        with TestClient(app) as c:
            r = c.post("/api/instances/nosuchid/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 404, r.text

    def test_400_when_the_prefix_is_ambiguous(self, instances_app):
        app, home = instances_app
        _write_entry(home, instance_id="aaaa000011112222", port=59994)
        _write_entry(home, instance_id="aaaa000033334444", port=59995)
        with TestClient(app) as c:
            r = c.post("/api/instances/aaaa0000/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 400, r.text
            assert "aaaa000011112222"[:8] in r.text or "matches 2 instances" in r.text

    def test_a_short_unambiguous_prefix_matches(self, instances_app, monkeypatch):
        """Same id-prefix semantics as `localm stop <id>`: a short unambiguous
        prefix resolves to exactly one entry. pid=-1 makes kill_pid report the
        target already gone, so this reaches a real 200 without touching an
        actual process; reap_stale is neutralised too, since pid<=0 reads as
        dead and the fixture would otherwise be deleted before the route can
        match it."""
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        _write_entry(home, instance_id="uniqueprefix00001", pid=-1, port=59996)
        with TestClient(app) as c:
            r = c.post("/api/instances/uniqueprefix/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 200, r.text
            assert r.json()["instance_id"] == "uniqueprefix00001"


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop - the ADMIN gate                             #
# --------------------------------------------------------------------------- #

class TestAdminGate:
    def test_a_config_write_only_key_is_refused(self, instances_app):
        """config:write is privileged but is NOT the owner, and must not reach
        a DIFFERENT instance's process."""
        app, home = instances_app
        _write_entry(home, instance_id="gatedtarget0001", pid=-1, port=59997)
        with TestClient(app) as c:
            r = c.post("/api/instances/gatedtarget0001/stop",
                       headers=_hdr(_config_write_key()))
            assert r.status_code == 403, r.text

    def test_a_config_read_only_key_is_refused(self, instances_app):
        app, home = instances_app
        _write_entry(home, instance_id="gatedtarget0002", pid=-1, port=59998)
        with TestClient(app) as c:
            r = c.post("/api/instances/gatedtarget0002/stop",
                       headers=_hdr(_config_read_key()))
            assert r.status_code == 403, r.text

    def test_an_admin_key_passes_the_gate(self, instances_app, monkeypatch):
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        _write_entry(home, instance_id="gatedtarget0003", pid=-1, port=59989)
        with TestClient(app) as c:
            r = c.post("/api/instances/gatedtarget0003/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 200, r.text

    def test_open_mode_passes_the_gate(self, instances_app, monkeypatch):
        """No key configured anywhere -> the trusted local owner, same as
        every other owner-gated route."""
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        _write_entry(home, instance_id="gatedtarget0004", pid=-1, port=59988)
        with TestClient(app) as c:
            r = c.post("/api/instances/gatedtarget0004/stop")
            assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop - real target, graceful shutdown             #
# --------------------------------------------------------------------------- #

def _wait_sync(cond, want=True, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


class _RealServer:
    def __init__(self, app, port, server, thread):
        self.app = app
        self.port = port
        self.server = server
        self.thread = thread

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


def _start_real_server() -> _RealServer:
    app = create_app(None)
    app.state.instance_token = "real-target-instance-token-0123456789"

    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve(sockets=[lsock]))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    assert _wait_sync(lambda: server.started, True, 10.0), "uvicorn did not start"
    return _RealServer(app, port, server, thread)


class TestStopRealHttp:
    def test_graceful_shutdown_reaches_the_real_target_route(
            self, instances_app, monkeypatch):
        """The real target's /v1/server/shutdown is actually invoked, not merely
        answered 200 by a mock. _do_shutdown's real body ends in os._exit(0), so
        it is replaced with a recorder; the recorder is observed AFTER the
        response returns and BEFORE teardown, since _request_shutdown fires it
        from a 0.25s-delayed background thread.

        No key is minted anywhere in this test: LOCALM_HOME is shared between
        "my" app and the real target server, and any_key_configured() reads that
        SAME shared keystore, so an admin key would also knock the target out of
        open mode. The target's _enforce_request accepts a real registered key
        only and has no notion of an instance attach token, so it would then 401
        the graceful request. Both sides stay in open mode for the
        instance-token fallback under test."""
        app, home = instances_app
        shutdown_calls = []
        monkeypatch.setattr(_hs, "_do_shutdown",
                            lambda **kw: shutdown_calls.append(kw))
        # reap_stale must be neutralised BEFORE pid_alive is forced to always
        # read False two lines below: otherwise the route's own opening
        # reap_stale() call deletes this fixture before the route can match it,
        # and the test fails on an unrelated 404.
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        rs = _start_real_server()
        try:
            iid = "realtargetserver01"
            _write_entry(home, instance_id=iid, pid=999999999, host="127.0.0.1",
                        port=rs.port, root_dir="/proj/real", mode="api",
                        token=rs.app.state.instance_token)
            # A pid that was never real: pid_alive() reads False so the graceful
            # path is what confirms it, and kill_pid must not run.
            monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
            monkeypatch.setattr(
                instances, "kill_pid",
                lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("kill_pid must not run: the graceful "
                                   "shutdown must have been accepted")))
            with TestClient(app) as c:
                r = c.post(f"/api/instances/{iid}/stop")   # open mode: no key anywhere
            assert r.status_code == 200, r.text
            assert r.json()["graceful_denied"] is False
            assert _wait_sync(lambda: len(shutdown_calls) == 1, True, 3.0), (
                "the real target's /v1/server/shutdown route never actually "
                "invoked its shutdown sequence")
        finally:
            rs.stop()

class TestStopGracefulDeclined:
    def test_a_declined_graceful_shutdown_falls_back_to_a_real_kill(
            self, instances_app, monkeypatch):
        """The 401/403 branch, distinct from 'unreachable' (TestStopKillFallback
        covers that against a real closed port): the target answers but
        declines, graceful_denied is reported True, and the route still falls
        through to a real kill.

        The decline is mocked at requests.request (what
        selfclient.self_request calls internally) rather than built from a
        second real server, because LOCALM_HOME - and therefore auth.key - is
        shared between "my" app and any target in this fixture. The kill itself
        stays real (a real subprocess, real instances.kill_pid)."""
        app, home = instances_app
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            iid = "declinedtarget001"
            _write_entry(home, instance_id=iid, pid=proc.pid, host="127.0.0.1",
                        port=59986, root_dir="/proj/declined", mode="api")
            fake_401 = SimpleNamespace(status_code=401, ok=False,
                                       json=lambda: {}, text="")
            with patch("requests.request", return_value=fake_401):
                with TestClient(app) as c:
                    r = c.post(f"/api/instances/{iid}/stop")
            assert r.status_code == 200, r.text
            assert r.json()["graceful_denied"] is True
            assert not instances.pid_alive(proc.pid), (
                "the declined-graceful path must fall back to a real kill")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
#  POST /api/instances/{id}/stop - direct-kill fallback, real process         #
# --------------------------------------------------------------------------- #

class TestStopKillFallback:
    def test_an_unreachable_target_falls_back_to_killing_the_real_process(
            self, instances_app):
        """No server is listening on the registered port at all (connection
        refused, not a slow timeout): the route still confirms the stop by
        falling back to instances.kill_pid against a REAL subprocess."""
        app, home = instances_app
        closed = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        closed.bind(("127.0.0.1", 0))
        dead_port = closed.getsockname()[1]
        closed.close()   # bound-then-closed: nothing answers here

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            iid = "unreachabletarget1"
            _write_entry(home, instance_id=iid, pid=proc.pid, host="127.0.0.1",
                        port=dead_port, root_dir="/proj/unreachable", mode="api",
                        token=None)
            with TestClient(app) as c:
                r = c.post(f"/api/instances/{iid}/stop",
                          headers=_hdr(_admin_key()))
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["status"] == "stopped"
            assert not instances.pid_alive(proc.pid), "the real process must be dead"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def test_stopping_deregisters_the_registry_entry(self, instances_app, monkeypatch):
        app, home = instances_app
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        _write_entry(home, instance_id="deregme00000001", pid=-1, port=59987)
        reg_path = home / "run" / "deregme00000001.json"
        assert reg_path.is_file()
        with TestClient(app) as c:
            r = c.post("/api/instances/deregme00000001/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 200, r.text
        assert not reg_path.exists()


class TestStopUnconfirmed:
    def test_502_when_neither_graceful_nor_kill_confirms(
            self, instances_app, monkeypatch):
        """The ROUTE's response when instances.kill_pid cannot confirm the stop:
        a 502. kill_pid is forced to report failure rather than a real
        unkillable process being constructed."""
        app, home = instances_app
        closed = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        closed.bind(("127.0.0.1", 0))
        dead_port = closed.getsockname()[1]
        closed.close()
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        _write_entry(home, instance_id="neverconfirmed001", pid=999999998,
                  port=dead_port)
        monkeypatch.setattr(instances, "kill_pid", lambda *a, **k: False)
        with TestClient(app) as c:
            r = c.post("/api/instances/neverconfirmed001/stop",
                       headers=_hdr(_admin_key()))
            assert r.status_code == 502, r.text
