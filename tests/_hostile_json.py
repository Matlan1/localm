# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hostile JSON documents and a real local HTTP server for the regression tests
that feed them to a parser.

``DEEP`` nests past the interpreter's recursion limit (``RecursionError``, which
is a ``RuntimeError``, not a ``ValueError``). ``BIG_INT`` is a number longer than
the integer-conversion digit limit (a ``ValueError`` that is NOT a
``JSONDecodeError``). ``DEEP_CLOSED`` is the same nesting as well-formed JSON.
"""

import contextlib
import http.server
import threading

import pytest

DEEP = "[" * 100_000
BIG_INT = "9" * 5_000
DEEP_CLOSED = "[" * 100_000 + "]" * 100_000
BIG_INT_LIST = "[" + "9" * 5_000 + "]"

HOSTILE = pytest.mark.parametrize("doc", [DEEP, BIG_INT], ids=["deep", "bigint"])
HOSTILE_BYTES = pytest.mark.parametrize(
    "doc", [DEEP.encode(), BIG_INT.encode()], ids=["deep", "bigint"])


def _as_bytes(body):
    return body.encode("utf-8") if isinstance(body, str) else bytes(body)


def port_of(base_url):
    """The port of a ``serve`` base URL."""
    return int(base_url.rsplit(":", 1)[1])


@contextlib.contextmanager
def serve(body, *, status=200, headers=None, content_type="application/json"):
    """A loopback HTTP server answering every request with *body* (str or bytes),
    or, when *body* is a dict, answering each path in it with that path's body
    (and 404 with an empty body for any other path); yields its base URL.
    Stopped on exit."""
    routes = body if isinstance(body, dict) else None
    default = b"" if routes is not None else _as_bytes(body)
    extra = dict(headers or {})

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _reply(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            code, payload = status, default
            if routes is not None:
                path = self.path.split("?", 1)[0]
                if path in routes:
                    payload = _as_bytes(routes[path])
                else:
                    code = 404
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_DELETE = _reply

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
