# SPDX-License-Identifier: AGPL-3.0-or-later
"""``stop_comfy`` on a ComfyUI localm did not launch reports what the abort and
free-VRAM requests actually returned. A loopback stand-in answers
``GET /system_stats`` 200 and the POST endpoints with the status each test sets."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.media import comfy_client


@pytest.fixture
def comfy():
    statuses = {"/interrupt": 200, "/queue": 200, "/free": 200}
    posted: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code):
            raw = json.dumps({}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._send(200 if self.path == "/system_stats" else 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            posted.append(self.path)
            self._send(statuses.get(self.path, 404))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, statuses, posted
    httpd.shutdown()
    httpd.server_close()


def test_stop_reports_failure_when_the_interrupt_is_refused(comfy):
    url, statuses, posted = comfy
    statuses["/interrupt"] = 500
    ok, message = comfy_client.stop_comfy(url)
    assert ok is False, message
    assert "/interrupt" in posted
    assert "did not accept" in message
    assert "Aborted" not in message
    assert "left running" in message


def test_stop_reports_the_abort_and_the_freed_vram_when_both_succeed(comfy):
    url, _statuses, posted = comfy
    ok, message = comfy_client.stop_comfy(url)
    assert ok is True, message
    assert "/interrupt" in posted and "/free" in posted
    assert "Aborted the in-flight render" in message
    assert "Freed its VRAM" in message
    assert "left running" in message


def test_stop_says_vram_was_not_freed_when_free_fails(comfy):
    url, statuses, _posted = comfy
    statuses["/free"] = 404
    ok, message = comfy_client.stop_comfy(url)
    assert ok is True, message
    assert "did not free its VRAM" in message
    assert "Freed its VRAM" not in message
