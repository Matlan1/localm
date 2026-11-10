# SPDX-License-Identifier: AGPL-3.0-or-later
"""HttpEngine (``localm run`` attach, routed jobs, the MCP server's shared
model) never follows a redirect: a loopback stand-in answering 307/308 to a
second loopback server must not make the chat body, the token or a probe reach
that second server."""

from __future__ import annotations

import http.server
import threading

import pytest

from localm.inference.http_engine import HttpEngine, remote_model_status


class _Recorder:
    """A loopback server that records every request it receives."""

    def __init__(self):
        outer = self
        self.hits = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _record(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.hits.append((self.command, self.path,
                                   self.headers.get("Authorization"),
                                   self.rfile.read(n) if n else b""))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            do_GET = do_POST = _record

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


class _Redirector:
    """A loopback server that answers every request with *code* pointing at
    *target* (same path)."""

    def __init__(self, code, target):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _redirect(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                self.send_response(code)
                self.send_header("Location", f"{target}{self.path}")
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _redirect

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture(params=[301, 302, 307, 308])
def redirect_pair(request):
    target = _Recorder()
    redir = _Redirector(request.param, f"http://127.0.0.1:{target.port}")
    yield redir, target
    redir.close()
    target.close()


def test_chat_stream_refuses_a_redirect_and_sends_nothing_on(redirect_pair):
    redir, target = redirect_pair
    eng = HttpEngine(f"http://127.0.0.1:{redir.port}/v1", token="secret-token", model="m")
    with pytest.raises(RuntimeError, match="redirect"):
        list(eng.chat_stream([{"role": "user", "content": "private words"}]))
    assert target.hits == []


def test_context_capacity_does_not_follow_a_redirect(redirect_pair):
    redir, target = redirect_pair
    eng = HttpEngine(f"http://127.0.0.1:{redir.port}/v1", token="secret-token", model="m")
    assert eng.context_capacity() is None
    assert target.hits == []


def test_remote_model_status_reads_a_redirect_as_unknown(redirect_pair):
    redir, target = redirect_pair
    assert remote_model_status(f"http://127.0.0.1:{redir.port}/v1",
                               "secret-token") == ("unknown", None)
    assert target.hits == []


def test_chat_stream_still_reaches_a_server_that_answers_directly():
    target = _Recorder()
    try:
        eng = HttpEngine(f"http://127.0.0.1:{target.port}/v1", token="t", model="m")
        list(eng.chat_stream([{"role": "user", "content": "hi"}]))
        assert [(m, p) for m, p, _a, _b in target.hits] == [("POST", "/v1/chat/completions")]
    finally:
        target.close()
