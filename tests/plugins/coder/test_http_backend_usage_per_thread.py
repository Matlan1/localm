# SPDX-License-Identifier: AGPL-3.0-or-later
"""Agents on different threads that share one HTTPBackend each read the usage
of the request their own thread made last."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm.plugins.coder.backends.http import HTTPBackend


@pytest.fixture
def backend():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                    # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tokens = int(body["messages"][-1]["content"])
            payload = json.dumps({
                "choices": [{"message": {"content": "ok", "reasoning_content": f"r{tokens}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tokens, "completion_tokens": 1,
                          "total_tokens": tokens + 1},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield HTTPBackend(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m")
    srv.shutdown()
    srv.server_close()


def _ask(backend, tokens: int) -> None:
    backend.chat([{"role": "user", "content": str(tokens)}])


def test_a_request_on_another_thread_does_not_replace_this_threads_usage(backend):
    _ask(backend, 5000)
    other = threading.Thread(target=_ask, args=(backend, 7))
    other.start()
    other.join()
    assert backend.last_usage["prompt_tokens"] == 5000
    assert backend.last_reasoning == "r5000"


def test_each_thread_reads_the_usage_of_its_own_request(backend):
    seen: dict[str, int] = {}

    def worker():
        _ask(backend, 7)
        seen["worker"] = backend.last_usage["prompt_tokens"]

    _ask(backend, 5000)
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["worker"] == 7
    assert backend.last_usage["prompt_tokens"] == 5000


def test_a_thread_that_made_no_request_sees_no_usage(backend):
    _ask(backend, 5000)
    result: list = []
    t = threading.Thread(target=lambda: result.append(
        (backend.last_usage, backend.last_reply_compacted, backend.last_reasoning)))
    t.start()
    t.join()
    assert result == [({}, False, "")]
