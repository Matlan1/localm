# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm unload` and `localm stop` must actually authenticate against the server they manage, in BOTH populations the pre-fix code missed:."""

from __future__ import annotations

import asyncio
import socket as _socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn
from click.testing import CliRunner

import localm.inference.http_server as _hs
from localm import auth, instances
from localm.cli import main
from localm.inference.http_server import create_app


def _entry(iid="abcd1234ef", pid=999999, root="/proj/demo", port=8642,
          token=None):
    e = {"instance_id": iid, "pid": pid, "root_dir": root, "scheme": "http",
         "host": "127.0.0.1", "port": port, "_path": f"/run/{iid}.json"}
    if token is not None:
        e["token"] = token
    return e


def _resp(status_code, body=None):
    body = body if body is not None else {}
    return SimpleNamespace(status_code=status_code, ok=200 <= status_code < 400,
                           json=lambda: body, text=str(body))


# --------------------------------------------------------------------------- #
#  unload: credential selection (mocked requests.post)                        #
# --------------------------------------------------------------------------- #

class TestUnloadCredentialSelection:
    def test_open_mode_uses_the_discovered_instance_token(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(instances, "find_attachable",
                            lambda *a, **k: _entry(token="the-attach-token"))
        captured = {}

        def fake_post(url, headers=None, **kw):
            captured["headers"] = headers
            return _resp(200, {"status": "already_unloaded", "model": "none"})

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["unload"])

        assert res.exit_code == 0, res.output
        assert captured["headers"]["Authorization"] == "Bearer the-attach-token"

    def test_keyed_via_persisted_auth_key_file_only(self, monkeypatch):
        """The exact gap: env-only missed a launcher/`localm key generate` server whose key lives in auth.key, not the environment."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        auth.set_api_key("file-only-key-999")
        monkeypatch.setattr(instances, "find_attachable",
                            lambda *a, **k: _entry())
        captured = {}

        def fake_post(url, headers=None, **kw):
            captured["headers"] = headers
            return _resp(200, {"status": "already_unloaded", "model": "none"})

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["unload"])

        assert res.exit_code == 0, res.output
        assert captured["headers"]["Authorization"] == "Bearer file-only-key-999"

    def test_env_key_wins_over_instance_token(self, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "env-key-1")
        monkeypatch.setattr(instances, "find_attachable",
                            lambda *a, **k: _entry(token="the-attach-token"))
        captured = {}

        def fake_post(url, headers=None, **kw):
            captured["headers"] = headers
            return _resp(200, {"status": "already_unloaded", "model": "none"})

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["unload"])

        assert res.exit_code == 0, res.output
        assert captured["headers"]["Authorization"] == "Bearer env-key-1"

    def test_no_token_available_sends_no_auth_header(self, monkeypatch):
        """Unchanged pre-fix behaviour when the caller genuinely has neither - the registry entry carries no token (an older server, or discovery returning a bare entry)."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(instances, "find_attachable",
                            lambda *a, **k: _entry())  # no token
        captured = {}

        def fake_post(url, headers=None, **kw):
            captured["headers"] = headers
            return _resp(200, {"status": "already_unloaded", "model": "none"})

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["unload"])

        assert res.exit_code == 0, res.output
        assert "Authorization" not in captured["headers"]


# --------------------------------------------------------------------------- #
#  stop: credential selection, PER TARGET (mocked requests.post)              #
# --------------------------------------------------------------------------- #

class TestStopCredentialSelection:
    def test_open_mode_uses_each_targets_own_instance_token(self, monkeypatch):
        """--all can span several instances, each with its OWN attach token - the fallback must be resolved per target, not a single header reused across all of them."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        entries = [
            _entry(iid="aaaa1111", pid=111, port=8642, token="token-a"),
            _entry(iid="bbbb2222", pid=222, port=8643, token="token-b"),
        ]
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: entries)
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        seen = []

        def fake_post(url, headers=None, **kw):
            seen.append((url, headers.get("Authorization")))
            return _resp(200)

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["stop", "--all"])

        assert res.exit_code == 0, res.output
        by_url = dict(seen)
        assert by_url["http://127.0.0.1:8642/v1/server/shutdown"] == "Bearer token-a"
        assert by_url["http://127.0.0.1:8643/v1/server/shutdown"] == "Bearer token-b"

    def test_keyed_via_persisted_auth_key_file_only(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        auth.set_api_key("file-only-key-777")
        entry = _entry()
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [entry])
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        captured = {}

        def fake_post(url, headers=None, **kw):
            captured["headers"] = headers
            return _resp(200)

        with patch("requests.post", side_effect=fake_post), \
             patch("localm.tls.requests_verify", return_value=True):
            res = CliRunner().invoke(main, ["stop", "abcd1234"])

        assert res.exit_code == 0, res.output
        assert captured["headers"]["Authorization"] == "Bearer file-only-key-777"


# --------------------------------------------------------------------------- #
#  Real HTTP: a real loopback server, the real _origin_guard middleware       #
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


def _start_real_server(engine=None) -> _RealServer:
    app = create_app(engine)
    app.state.instance_token = "real-instance-token-0123456789"

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


class TestUnloadRealHttp:
    def test_open_mode_succeeds_via_discovered_instance_token(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        rs = _start_real_server()
        try:
            monkeypatch.setattr(
                instances, "find_attachable",
                lambda *a, **k: _entry(port=rs.port,
                                       token=rs.app.state.instance_token))
            res = CliRunner().invoke(main, ["unload"])
            assert res.exit_code == 0, res.output
            assert "403" not in res.output
            assert "nothing to do" in res.output.lower() or \
                "Nothing was loaded" in res.output
        finally:
            rs.stop()

    def test_keyed_via_persisted_auth_key_file_only_succeeds(self, monkeypatch):
        """The real server is genuinely in KEYED mode (an owner key persisted to auth.key, matching what a real `localm key generate` leaves on disk) - _origin_guard's open-mode branch never engages, and the route's own scope check must accept the CLI's key."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        auth.set_api_key("real-owner-key-abcdef123456")
        rs = _start_real_server()
        try:
            monkeypatch.setattr(
                instances, "find_attachable",
                lambda *a, **k: _entry(port=rs.port))  # no token needed: keyed
            res = CliRunner().invoke(main, ["unload"])
            assert res.exit_code == 0, res.output
            assert "403" not in res.output
            assert "401" not in res.output
        finally:
            rs.stop()


class TestStopRealHttp:
    def test_open_mode_graceful_shutdown_via_discovered_instance_token(
            self, monkeypatch):
        """Not run live by the original report (it would have killed the test server) - exercised here against a real throwaway instance instead."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        shutdown_calls = []
        monkeypatch.setattr(_hs, "_do_shutdown",
                            lambda **kw: shutdown_calls.append(kw))
        rs = _start_real_server()
        try:
            entry = _entry(iid="realsrv1", pid=999999999, port=rs.port,
                           token=rs.app.state.instance_token)
            monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
            monkeypatch.setattr(instances, "list_entries",
                                lambda *a, **k: [entry])
            # A fake pid that was never real: pid_alive() must read False so
            # the graceful path is credited as the confirmation, and kill_pid
            # must never be needed - if it were reached, it would be asked to
            # signal a pid this test does not own.
            monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
            monkeypatch.setattr(
                instances, "kill_pid",
                lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("kill_pid must not run: the graceful "
                                   "shutdown request must have been accepted")))
            res = CliRunner().invoke(main, ["stop", "realsrv1"])
            assert res.exit_code == 0, res.output
            assert "declined an unauthenticated shutdown" not in res.output
            assert _wait_sync(lambda: len(shutdown_calls) == 1, True, 3.0), (
                "the real /v1/server/shutdown route never actually invoked "
                "its shutdown sequence - the auth gate may not have been "
                "reached at all")
        finally:
            rs.stop()
