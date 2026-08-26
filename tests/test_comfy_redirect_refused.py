# SPDX-License-Identifier: AGPL-3.0-or-later
"""sanitize_comfy_url only screens the CONFIGURED comfy_api_url, at resolution
time. A redirect target is chosen by the remote ComfyUI AFTER that check has
already run, and urllib's default opener follows up to 10 such redirects with no
validation at all, so a hostile or compromised ComfyUI (SECURITY.md: it "may be
another machine, over plain http") could answer any request with a 3xx straight
past the guard, e.g. to a cloud-metadata address. ComfyUI's HTTP API has no
legitimate reason to redirect, so the connection itself refuses every hop
outright.

TWO real (loopback) HTTP stubs: "hop A" plays the compromised/hostile ComfyUI
and answers every request with a 302 to "hop B", which plays the redirect target
and answers with real, valid-looking data. Under the fix, hop B is never
contacted at all; under a regression, hop B's data comes back to the caller
exactly as a real SSRF would deliver it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

from localm.media import comfy_client as c

_HOPB_BYTES = b"HOPB-SECOND-HOP-BYTES"


class _Stub(HTTPServer):
    def __init__(self, handler):
        super().__init__(("127.0.0.1", 0), handler)
        self.hit_paths: list = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _RedirectToHopB(BaseHTTPRequestHandler):
    """Hop A: the hostile/compromised ComfyUI. 302s every request to hop B,
    preserving the path+query so hop B's router matches the same endpoint."""

    hop_b_base = ""   # set per-test via server attribute below

    def log_message(self, *_a):
        pass

    def _redirect(self):
        self.server.hit_paths.append(self.path)
        self.send_response(302)
        self.send_header("Location", f"{self.server.hop_b_base}{self.path}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        self._redirect()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._redirect()


class _HopBHandler(BaseHTTPRequestHandler):
    """Hop B: what the redirect target answers with IF a broken client
    followed it. Real, valid, ComfyUI-shaped responses, so a regression is
    observable as real second-hop data flowing back to the caller."""

    def log_message(self, *_a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, code, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route(self):
        self.server.hit_paths.append(self.path)
        path = urlparse(self.path).path
        if path == "/object_info":
            return self._json(200, {"marker": "HOPB"})
        if path == "/system_stats":
            return self._json(200, {"marker": "HOPB"})
        if path == "/free":
            return self._json(200, {})
        if path == "/interrupt":
            return self._json(200, {})
        if path == "/queue":
            return self._json(200, {})
        if path == "/history":
            return self._json(200, {})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[-1]
            return self._json(200, {pid: {"outputs": {}}})
        if path == "/prompt":
            return self._json(200, {"prompt_id": "hopb-injected-pid"})
        if path == "/upload/image":
            return self._json(200, {"name": "hopb-injected-name.png"})
        if path == "/view":
            return self._raw(200, _HOPB_BYTES)
        return self._json(404, {})

    def do_GET(self):  # noqa: N802
        self._route()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._route()


@pytest.fixture
def hops():
    """(hop_a, hop_b): hop_a redirects everything to hop_b."""
    b = _Stub(_HopBHandler)
    a = _Stub(_RedirectToHopB)
    a.hop_b_base = b.base_url
    ta = threading.Thread(target=a.serve_forever, daemon=True)
    tb = threading.Thread(target=b.serve_forever, daemon=True)
    ta.start()
    tb.start()
    try:
        yield a, b
    finally:
        a.shutdown()
        b.shutdown()
        a.server_close()
        b.server_close()
        ta.join(timeout=5)
        tb.join(timeout=5)


def test_object_info_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    # A refused redirect reads the same as could-not-be-fetched (None) and never
    # surfaces hop B's data.
    assert c.comfy_object_info(a.base_url, timeout=2) is None
    assert b.hit_paths == []          # the redirect target was NEVER dialed
    assert a.hit_paths == ["/object_info"]


def test_comfy_alive_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    assert c._comfy_alive(a.base_url, timeout=2) is False
    assert b.hit_paths == []


def test_free_comfy_vram_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    assert c.free_comfy_vram(a.base_url) is False
    assert b.hit_paths == []


def test_interrupt_comfy_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    assert c.interrupt_comfy(a.base_url) is False
    assert b.hit_paths == []


def test_clear_comfy_history_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    assert c.clear_comfy_history(a.base_url, "some-prompt-id") is False
    assert b.hit_paths == []


def test_submit_prompt_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    kind, value = c.comfy_submit_prompt(a.base_url, {"1": {}}, timeout=2)
    assert kind == c.SUBMIT_ERROR
    assert isinstance(value, c.ComfyRedirectRefused)
    assert b.hit_paths == []          # never picked up hop B's injected prompt_id


def test_fetch_output_refuses_redirect_never_reaches_hop_b(hops, tmp_path):
    a, b = hops
    dest = tmp_path / "out.bin"
    with pytest.raises(c.ComfyRedirectRefused):
        c.comfy_fetch_output(a.base_url, {"filename": "x.png"}, dest, timeout=2)
    assert b.hit_paths == []          # hop B's bytes were never fetched or written
    assert not dest.exists()


def test_upload_image_refuses_redirect_never_reaches_hop_b(hops, tmp_path):
    a, b = hops
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)   # looks_like_image checks the head
    with pytest.raises(c.ComfyRedirectRefused):
        c._upload_image(img, a.base_url)
    assert b.hit_paths == []          # hop B's injected filename never came back


def test_poll_until_done_refuses_redirect_never_reaches_hop_b(hops):
    a, b = hops
    # The loop retries on any exception, so the observable contract is
    # POLL_TIMEOUT with the refusal as the last error, never POLL_FINISHED with
    # hop B's fabricated history entry.
    status, value = c.comfy_poll_until_done(
        a.base_url, "pid-1", max_poll_seconds=0.3, history_timeout=1,
        sleep_seconds=0.05)
    assert status == c.POLL_TIMEOUT
    assert isinstance(value, c.ComfyRedirectRefused)
    assert b.hit_paths == []
