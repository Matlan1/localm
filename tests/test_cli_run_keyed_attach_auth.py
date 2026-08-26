# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run` (and the shared attach flow it uses) must authenticate to its
OWN server with the OWNER KEY once one is configured, not the raw per-instance
attach token.

``instances.attach_target`` always returns the discovered instance's own
per-instance registry token (a random per-process secret with no keystore
entry). ``_principal_from_token`` (http_server.py) resolves a header-sourced
bearer via ``auth.verify()`` only, which has no notion of instance tokens at
all - so once ANY owner key is configured, presenting the raw instance token
as ``Authorization: Bearer <instance_token>`` always 401s. Sending that value
straight into ``remote_model_status``/``HttpEngine`` makes `localm run` print
"connected ... (no second model load)" and then die with a 401 on any keyed
install.

The same shape applies to ``self_request`` / ``cli/models.py``'s
``unload_cmd``/``stop_cmd`` / ``media/comfy_client.py``'s ``_localm_unload``,
all of which go through ``auth.resolve_bearer_headers``; this file covers the
two attach sites in ``cli/chat.py``, at two levels:

  - a REAL uvicorn server, keyed, proving the actual HTTP round trip succeeds
    with the resolved credential and 401s with the raw instance token alone
    (the control) - the genuinely end-to-end oracle;
  - the CLI's own `run()` control flow (CliRunner, network layer stubbed),
    proving the token that reaches HttpEngine/remote_model_status is the
    OWNER KEY when one is configured and the raw instance token in open mode
    - drives the real credential-SELECTION code, not a mock of it.
"""

import asyncio
import socket as _socket
import threading
import time
from unittest.mock import MagicMock

import pytest
import uvicorn
from click.testing import CliRunner

from localm import auth
from localm.audit import SessionMode
from localm.cli.chat import run
from localm.inference.http_server import create_app


def _wait_sync(cond, want=True, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


def _real_keyed_server(owner_key: str):
    """A real uvicorn instance with an owner key configured, and
    app.state.instance_token set the way instances.advertise() sets it in
    production - the open-mode helper's shape, applied to the keyed case."""
    auth.set_api_key(owner_key)
    app = create_app(None)
    app.state.instance_token = "wrong-instance-token-should-never-auth"

    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve(sockets=[lsock]))

    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    assert _wait_sync(lambda: server.started, True, 10.0), "uvicorn did not start"
    return app, port, server, th


def _shutdown(server, th):
    server.should_exit = True
    th.join(timeout=5.0)


class TestRealKeyedServerAttachAuth:
    def test_raw_instance_token_401s_control(self):
        """CONTROL: the raw instance token, unresolved, is refused by the real
        server - proves the positive test below is really about credential
        RESOLUTION, not a permissive server."""
        app, port, server, th = _real_keyed_server("e2e-owner-key-0123456789")
        try:
            import requests
            r = requests.get(f"http://127.0.0.1:{port}/v1/models",
                             headers={"Authorization":
                                     f"Bearer {app.state.instance_token}"},
                             timeout=5)
            assert r.status_code == 401
        finally:
            _shutdown(server, th)

    def test_resolved_token_gets_past_auth(self):
        app, port, server, th = _real_keyed_server("e2e-owner-key-0123456789")
        try:
            token = auth.resolve_bearer_token(app.state.instance_token)
            assert token == "e2e-owner-key-0123456789"
            import requests
            r = requests.get(f"http://127.0.0.1:{port}/v1/models",
                             headers={"Authorization": f"Bearer {token}"},
                             timeout=5)
            assert r.status_code == 200, r.text
        finally:
            _shutdown(server, th)

    def test_remote_model_status_succeeds_with_resolved_token(self):
        """The exact call cli/chat.py's run() makes to probe the attach
        target, driven for real against a real keyed server."""
        from localm.inference.http_engine import remote_model_status
        app, port, server, th = _real_keyed_server("e2e-owner-key-0123456789")
        try:
            token = auth.resolve_bearer_token(app.state.instance_token)
            state, active = remote_model_status(f"http://127.0.0.1:{port}/v1", token)
            assert state != "unknown", (
                "auth failed against the real server - remote_model_status "
                "cannot distinguish a real failure from a 401")
        finally:
            _shutdown(server, th)

    def test_remote_model_status_fails_with_unresolved_instance_token(self):
        """Sanity/negative: proves the previous test's success is really
        about resolution, not remote_model_status being lenient."""
        from localm.inference.http_engine import remote_model_status
        app, port, server, th = _real_keyed_server("e2e-owner-key-0123456789")
        try:
            state, active = remote_model_status(
                f"http://127.0.0.1:{port}/v1", app.state.instance_token)
            assert state == "unknown"
        finally:
            _shutdown(server, th)


# --------------------------------------------------------------------------- #
#  CLI-level: run()'s own control flow selects the right token               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def patched(monkeypatch):
    """Stub only the network seam (attach discovery, the /v1/models probe, and
    the HTTP engine class) - the credential SELECTION logic under test
    (auth.get_api_key() + resolve_bearer_token) runs for real."""
    fake_target = {"base_url": "http://127.0.0.1:8642/v1", "token": "raw-instance-token"}
    engine_spy = MagicMock(name="HttpEngine")
    captured = {}

    monkeypatch.setattr("localm.instances.attach_target",
                        lambda *a, **k: fake_target)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)

    def fake_status(base_url, token, **kw):
        captured["status_token"] = token
        return "loaded", "SmolLM2-135M"

    monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                        fake_status)
    return engine_spy, captured


def test_run_attach_uses_owner_key_when_configured(patched):
    engine_spy, captured = patched
    auth.set_api_key("real-owner-key-0123456789")

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert captured["status_token"] == "real-owner-key-0123456789"
    assert engine_spy.called
    _, kwargs = engine_spy.call_args
    assert kwargs["token"] == "real-owner-key-0123456789"


def test_run_attach_uses_instance_token_when_open(patched):
    engine_spy, captured = patched
    assert auth.get_api_key() is None      # open mode (no key configured)

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert captured["status_token"] == "raw-instance-token"
    assert engine_spy.called
    _, kwargs = engine_spy.call_args
    assert kwargs["token"] == "raw-instance-token"
