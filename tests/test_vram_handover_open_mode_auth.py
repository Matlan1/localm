# SPDX-License-Identifier: AGPL-3.0-or-later
"""The chat<->media VRAM handover (vram.py's unload_chat_for_media /
reload_chat_after_media) must actually authenticate on a keyless (open-mode,
default) server.

_origin_guard's open-mode management gate (http_server.py) 403s any
unsafe-method call to /v1/models/unload|load that carries neither a key nor
the loopback shell_token/instance_token, and /v1/models/* is not in
_CROSS_ORIGIN_OK. So self_request threads this instance's own attach token
(request.app.state.instance_token, from the 0600 per-instance registry file,
never reachable by a browser), the same way read_activity does.

This file drives a REAL uvicorn server on a real loopback socket and calls the
real self_request -> real HTTP -> real _origin_guard round trip, so a
regression shows up as a real 403.
"""

import asyncio
import socket as _socket
import threading
import time

import uvicorn

from localm import vram
from localm.inference.http_server import create_app


class _FakeJob:
    def __init__(self):
        self.lines = []

    def push(self, ev):
        self.lines.append(ev.get("text", ""))

    def text(self):
        return " | ".join(self.lines)


def _wait_sync(cond, want=True, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(cond()) == want:
            return True
        time.sleep(0.02)
    return bool(cond()) == want


def _real_open_mode_server():
    """A real uvicorn instance on a real loopback socket, open (keyless) mode,
    with app.state.instance_token set the way instances.advertise() sets it
    in production: unconditionally, for the life of the process."""
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


def test_unload_chat_for_media_authenticates_with_instance_token_open_mode():
    """With instance_token threaded through, the real round trip must get PAST
    _origin_guard (no 403) and reach the real /v1/models/unload handler, which
    reports nothing loaded."""
    app, port, server, th = _real_open_mode_server()
    try:
        self_url = f"http://127.0.0.1:{port}/v1"
        job = _FakeJob()
        ok = vram.unload_chat_for_media(job, self_url, "image",
                                        app.state.instance_token)
        text = job.text()
        assert "HTTP 403" not in text, (
            f"unload_chat_for_media was refused by the open-mode gate despite "
            f"a valid instance_token: {text!r}")
        assert ok is True, f"expected a clean (nothing loaded) unload: {text!r}"
        assert "no chat model was loaded" in text.lower()
    finally:
        _shutdown(server, th)


def test_unload_chat_for_media_still_403s_with_no_token_open_mode():
    """Negative case: omitting the credential still 403s, so a real credential
    remains required."""
    app, port, server, th = _real_open_mode_server()
    try:
        self_url = f"http://127.0.0.1:{port}/v1"
        job = _FakeJob()
        ok = vram.unload_chat_for_media(job, self_url, "image", None)
        text = job.text()
        assert ok is False
        assert "HTTP 403" in text
    finally:
        _shutdown(server, th)


def test_reload_chat_after_media_authenticates_with_instance_token_open_mode():
    """The sibling function/endpoint (/v1/models/load). No default model is
    configured in this throwaway app, so the real route 503s past the origin
    guard ("No model specified") - that 503, not a 403, is the proof the gate
    let the authenticated request through."""
    app, port, server, th = _real_open_mode_server()
    try:
        self_url = f"http://127.0.0.1:{port}/v1"
        job = _FakeJob()

        class _Backend:
            def free_vram(self, s):
                return True

        vram.reload_chat_after_media(
            job, self_url, {"reload_after": True}, _Backend(), "image",
            app.state.instance_token)
        text = job.text()
        assert "HTTP 403" not in text, (
            f"reload_chat_after_media was refused by the open-mode gate "
            f"despite a valid instance_token: {text!r}")
    finally:
        _shutdown(server, th)


def test_reload_chat_after_media_still_403s_with_no_token_open_mode():
    app, port, server, th = _real_open_mode_server()
    try:
        self_url = f"http://127.0.0.1:{port}/v1"
        job = _FakeJob()

        class _Backend:
            def free_vram(self, s):
                return True

        vram.reload_chat_after_media(
            job, self_url, {"reload_after": True}, _Backend(), "image", None)
        text = job.text()
        assert "HTTP 403" in text
    finally:
        _shutdown(server, th)
