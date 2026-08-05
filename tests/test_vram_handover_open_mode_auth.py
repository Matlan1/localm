# SPDX-License-Identifier: AGPL-3.0-or-later
"""The chat<->media VRAM handover's self-call must not 403 in OPEN mode - the
default configuration (no `localm key generate` run).

selfclient.self_request() built its Authorization header from
localm.auth.get_api_key() ALONE. Open mode has no API key, so it sent NO
Authorization header at all. http_server.py's `_origin_guard` middleware
requires a bearer matching `shell_token` OR `instance_token` for any
unsafe-method /v1 route not listed in `_CROSS_ORIGIN_OK`, and
`/v1/models/unload` / `/v1/models/load` are not on that list - so every
chat<->media VRAM swap (`vram.unload_chat_for_media`/`reload_chat_after_media`)
403'd, permanently, on the default install. vram.py's own module docstring
says this handover exists to stop the chat and media models colliding in VRAM
(a GPU-driver-hang hazard); the protection simply never engaged.

tests/test_vram_reading_honesty.py's TestMediaSwapMessageHonesty patches
`localm.selfclient.self_request` directly, and TestMediaSwapHonorsPinEndToEnd
replaces it with a TestClient-backed fake that supplies the app's own
`shell_token` header itself - both stand exactly at the site the bug lived in
(the header-building inside the real self_request()), so neither could ever
have caught it. These tests drive the REAL `requests`-based self_request()
call over a REAL loopback TCP socket, against the REAL `_origin_guard`
middleware on a REAL keyless server, with no patching of self_request or of
the auth gate.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
import uvicorn

from localm.inference.http_server import create_app
from localm.vram import reload_chat_after_media, unload_chat_for_media


def _wait_sync(cond, want=True, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


@pytest.fixture
def open_mode_server(monkeypatch):
    """A real localm server: no engine, no API key configured (open mode),
    bound to a real loopback TCP port, with its own instance_token set (as
    `localm.instances.advertise()` does for every real running server)."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    app = create_app(None)
    app.state.instance_token = "test-instance-token-0123456789abcdef"

    # Pre-bind an ephemeral port so it is known before the server starts (same
    # pattern as test_stream_disconnect_cancel.py's real-uvicorn tests).
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve(sockets=[lsock]))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        assert _wait_sync(lambda: server.started, True, 10.0), \
            "uvicorn did not start"
        yield app, f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _job_recorder():
    job = MagicMock()
    lines = []
    job.push.side_effect = lambda d: lines.append(d.get("text", ""))
    return job, lines


class TestUnloadChatForMediaRealHttpOpenMode:
    def test_succeeds_over_real_http_via_instance_token(self, open_mode_server):
        app, self_url = open_mode_server
        job, lines = _job_recorder()

        ok = unload_chat_for_media(job, self_url, "test",
                                   instance_token=app.state.instance_token)

        joined = " ".join(lines)
        assert ok, f"real self-call was rejected: {lines}"
        assert "VRAM already free" in joined, lines
        assert "403" not in joined, lines

    def test_still_403s_a_caller_with_no_credential_at_all(self, open_mode_server):
        """Defense in depth: the open-mode gate must keep refusing a caller
        that presents nothing, so this only ever passes via a real credential
        - never because the gate was loosened instead of fixed."""
        app, self_url = open_mode_server
        job, lines = _job_recorder()

        ok = unload_chat_for_media(job, self_url, "test")  # no instance_token

        joined = " ".join(lines)
        assert not ok, f"an uncredentialed self-call must still be refused: {lines}"
        assert "403" in joined, lines


class TestReloadChatAfterMediaRealHttpOpenMode:
    def test_reaches_the_route_via_instance_token(self, open_mode_server):
        """No default model is configured in this fixture, so /v1/models/load
        legitimately 503s ("No model specified") - the point is that it 503s
        rather than 403ing: the request reached real business logic instead of
        being turned away by the auth gate."""
        app, self_url = open_mode_server
        job, lines = _job_recorder()
        s = {"reload_after": True}
        backend = MagicMock()
        backend.free_vram.return_value = True

        reload_chat_after_media(job, self_url, s, backend, "test",
                                instance_token=app.state.instance_token)

        joined = " ".join(lines)
        assert "403" not in joined, lines
        assert "503" in joined, lines

    def test_still_403s_a_caller_with_no_credential_at_all(self, open_mode_server):
        app, self_url = open_mode_server
        job, lines = _job_recorder()
        s = {"reload_after": True}
        backend = MagicMock()
        backend.free_vram.return_value = True

        reload_chat_after_media(job, self_url, s, backend, "test")  # no token

        joined = " ".join(lines)
        assert "403" in joined, lines
