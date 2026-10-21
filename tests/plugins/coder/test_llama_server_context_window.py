# SPDX-License-Identifier: AGPL-3.0-or-later
"""A llama-server's context window is read from its /props, which carries the
same shape the real server answers with."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localm.plugins.coder.backends.http import HTTPBackend


def _server(routes: dict[str, tuple[int, object]]):
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                     # noqa: N802
            seen.append(self.path)
            status, body = routes.get(self.path, (404, {"detail": "Not Found"}))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def serve():
    servers = []

    def make(routes):
        srv, seen = _server(routes)
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/v1", seen

    yield make
    for s in servers:
        s.shutdown()
        s.server_close()


_PROPS = {"default_generation_settings": {"params": {"seed": 4294967295}, "n_ctx": 32768},
          "total_slots": 4, "build_info": "b9982-99f3dc322"}


def test_a_llama_server_window_comes_from_props(serve):
    url, seen = serve({"/props": (200, _PROPS)})
    backend = HTTPBackend(url, "m")
    assert backend.context_capacity() == 32768
    assert backend.context_capacity() == 32768
    assert seen == ["/v1/config", "/props"]


@pytest.mark.parametrize("routes", [
    {},
    {"/props": (200, {"default_generation_settings": {"n_ctx": "big"}})},
    {"/props": (200, {"default_generation_settings": {"n_ctx": 0}})},
    {"/props": (200, {"other": 1})},
    {"/props": (500, {"error": "x"})},
])
def test_a_server_without_a_usable_props_reports_no_window(serve, routes):
    url, _ = serve(routes)
    assert HTTPBackend(url, "m").context_capacity() is None


def test_a_localm_server_window_is_not_read_from_props(serve):
    url, seen = serve({"/v1/config": (200, {"effective_ctx_max": 8192,
                                            "effective_ctx_model": "m"}),
                       "/props": (200, _PROPS)})
    assert HTTPBackend(url, "m").context_capacity() == 8192
    assert seen == ["/v1/config"]


def test_a_localm_server_without_a_ceiling_is_not_probed_for_props(serve):
    url, seen = serve({"/v1/config": (200, {"effective_ctx_model": "m"}),
                       "/props": (200, _PROPS)})
    assert HTTPBackend(url, "m", localm_server=True).context_capacity() is None
    assert seen == ["/v1/config"]


def test_a_config_naming_another_model_is_not_followed_by_a_props_probe(serve):
    url, seen = serve({"/v1/config": (200, {"effective_ctx_max": 8192,
                                            "effective_ctx_model": "other"}),
                       "/props": (200, _PROPS)})
    backend = HTTPBackend(url, "m")
    assert backend.context_capacity() is None
    assert seen == ["/v1/config"]
