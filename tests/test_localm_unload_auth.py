# SPDX-License-Identifier: AGPL-3.0-or-later
"""The media-gen VRAM unload must authenticate and verify TLS.

The chat model is unloaded before a ComfyUI media gen by POSTing
``/v1/models/unload`` on the local server. That endpoint requires the
models-write scope, so the call must carry the LOCALM_API_KEY bearer token,
and on a built-in-TLS (https) loopback self-call it must trust the install's
own CA. Without either, the unload 401s or SSL-fails, the chat model stays
resident, and the media model loads on top of it.

Also covers the OPEN-MODE (keyless) fallback: with no key configured,
_localm_unload sends the caller's own instance_token instead.
TestUnloadRealHttpOpenMode drives the real _origin_guard gate.
"""

import asyncio
import socket as _socket
import threading
import time
from unittest.mock import MagicMock, patch

import uvicorn

from localm.image_gen import comfy
from localm.inference.http_server import create_app


def _resp(ok=True, payload=None):
    r = MagicMock()
    r.ok = ok
    r.json.return_value = payload if payload is not None else {"status": "unloaded"}
    return r


def test_unload_sends_bearer_and_tls_verify(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(url=url, headers=headers or {}, verify=verify)
        return _resp(payload={"status": "unloaded", "vram_freed": True,
                              "vram_before_bytes": 1, "vram_after_bytes": 2})

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value="/ca/bundle.pem"):
        out = comfy._localm_unload("https://127.0.0.1:8642/v1")

    assert captured["url"] == "https://127.0.0.1:8642/v1/models/unload"
    assert captured["headers"].get("Authorization") == "Bearer secret-key"
    # TLS verification must use the install CA bundle, never be disabled.
    assert captured["verify"] == "/ca/bundle.pem"
    assert out and out["status"] == "unloaded"


def test_unload_no_key_omits_auth_header(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(headers=headers or {})
        return _resp()

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        comfy._localm_unload("http://127.0.0.1:8642/v1")

    assert "Authorization" not in captured["headers"]


def test_unload_noop_without_url(monkeypatch):
    monkeypatch.delenv("LOCALM_URL", raising=False)
    with patch("requests.post") as post:
        assert comfy._localm_unload(None) is None
        post.assert_not_called()


def test_unload_swallows_failure(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "k")
    with patch("requests.post", side_effect=RuntimeError("boom")), \
         patch("localm.tls.requests_verify", return_value=True):
        assert comfy._localm_unload("http://127.0.0.1:8642/v1") is None


def test_unload_uses_instance_token_when_no_key_configured(monkeypatch):
    """In open (keyless) mode the caller's own instance token is sent in place
    of a key."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(headers=headers or {})
        return _resp()

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        comfy._localm_unload("http://127.0.0.1:8642/v1", "inst-token-abc")

    assert captured["headers"]["Authorization"] == "Bearer inst-token-abc"


def test_unload_owner_key_still_wins_over_instance_token(monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, headers=None, timeout=None, verify=None):
        captured.update(headers=headers or {})
        return _resp()

    with patch("requests.post", side_effect=fake_post), \
         patch("localm.tls.requests_verify", return_value=True):
        comfy._localm_unload("http://127.0.0.1:8642/v1", "inst-token-abc")

    assert captured["headers"]["Authorization"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
#  Real HTTP: the actual _origin_guard gate, open (keyless) mode              #
# --------------------------------------------------------------------------- #

def _wait_sync(cond, want=True, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


def _real_open_mode_server():
    """A real uvicorn instance, open (keyless) mode, with
    app.state.instance_token set the way instances.advertise() sets it in
    production."""
    app = create_app(None)
    app.state.instance_token = "test-instance-token-abcdef123456"

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


class TestUnloadRealHttpOpenMode:
    def test_instance_token_gets_past_origin_guard(self, monkeypatch):
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        app, port, server, th = _real_open_mode_server()
        try:
            url = f"http://127.0.0.1:{port}/v1"
            out = comfy._localm_unload(url, app.state.instance_token)
            assert out is not None, (
                "the real open-mode server refused the unload call despite "
                "a valid instance_token - the fallback credential never "
                "reached the wire")
        finally:
            _shutdown(server, th)

    def test_no_token_still_403s_and_returns_none(self, monkeypatch):
        """Negative control: omitting the credential does not bypass the
        gate."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        app, port, server, th = _real_open_mode_server()
        try:
            url = f"http://127.0.0.1:{port}/v1"
            out = comfy._localm_unload(url, None)
            assert out is None
        finally:
            _shutdown(server, th)
