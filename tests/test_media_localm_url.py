# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm image|video|music` with LOCALM_URL (a server root) unload the chat
model: the request goes to /v1/models/unload, the only path a server answers."""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from localm.media.comfy_client import _localm_unload, localm_v1_base


class _Handler(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        type(self).seen.append(self.path)
        if self.path.startswith("/v1/models/unload"):
            body = b'{"status": "unloaded"}'
            self.send_response(200)
        else:
            body = b'{"detail": "Method Not Allowed"}'
            self.send_response(405)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    _Handler.seen = []
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("given,want", [
    ("http://h:1", "http://h:1/v1"), ("http://h:1/", "http://h:1/v1"),
    ("http://h:1/v1", "http://h:1/v1"), ("http://h:1/v1/", "http://h:1/v1"),
    ("", None), (None, None), ("  ", None)])
def test_v1_base(given, want):
    assert localm_v1_base(given) == want


def test_unload_with_a_server_root_reaches_the_v1_route(server):
    root = f"http://127.0.0.1:{server.server_port}"
    assert _localm_unload(root, "t") == {"status": "unloaded"}
    assert _Handler.seen == ["/v1/models/unload"]


def test_unload_with_a_v1_base_is_unchanged(server):
    base = f"http://127.0.0.1:{server.server_port}/v1"
    assert _localm_unload(base, "t") == {"status": "unloaded"}
    assert _Handler.seen == ["/v1/models/unload"]


def test_unload_reads_a_root_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("LOCALM_URL", f"http://127.0.0.1:{server.server_port}")
    assert _localm_unload(None, "t", media=False) == {"status": "unloaded"}
    assert _Handler.seen == ["/v1/models/unload?media=false"]


def test_music_cli_hands_the_backend_a_v1_base(server, monkeypatch):
    from localm.cli import media

    root = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("LOCALM_URL", root)
    got = {}

    class _Backend:
        @staticmethod
        def _impl(s):
            return type("I", (), {"OUTPUT_SUFFIX": ".flac"})

        @staticmethod
        def supports(s, capability):
            return False

        @staticmethod
        def ensure_available(s, on_progress=None):
            return True, "ready"

        @staticmethod
        def generate(s, tags, out, *, self_url, swap, **kw):
            got["self_url"] = self_url
            got["swap"] = swap
            _localm_unload(self_url, None, media=False)
            return True, "saved"

        @staticmethod
        def free_vram(s):
            return True

    monkeypatch.setattr(media, "_offer_open", lambda path: None)
    media._via_backend(_Backend, {"backend": "native"}, "lofi", Path("o.flac"),
                       "music", lyrics=None, duration=8.0)
    assert got == {"self_url": root + "/v1", "swap": True}
    assert _Handler.seen == ["/v1/models/unload?media=false"]
