# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/smoke_installed_localm.py judges a server's answers correctly."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_installed_localm.py"
_spec = importlib.util.spec_from_file_location("smoke_installed_localm", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _serve(routes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = routes.get(self.path, (404, b"not found"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def serve():
    servers = []

    def start(routes):
        server = _serve(routes)
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


_GOOD = {
    "/health": (503, b'{"detail":"No engine initialised"}'),
    "/v1/models": (200, json.dumps({"object": "list", "data": []}).encode()),
    "/": (200, b"<!doctype html><html><body>localm</body></html>"),
}


def test_a_server_with_no_model_loaded_passes(serve):
    assert smoke.check_endpoints(serve(_GOOD)) == []


def test_a_healthy_server_with_a_model_passes(serve):
    routes = dict(_GOOD, **{"/health": (200, b'{"status":"ok"}')})
    assert smoke.check_endpoints(serve(routes)) == []


@pytest.mark.parametrize("route,answer,fragment", [
    ("/health", (500, b"boom"), "/health answered 500"),
    ("/health", (503, b"something else"), "unexpected body"),
    ("/v1/models", (404, b"x"), "/v1/models answered 404"),
    ("/v1/models", (200, b"not json"), "not JSON"),
    ("/v1/models", (200, b'{"object":"list","data":5}'), "not a model list"),
    ("/", (200, b"plain text"), "did not serve the GUI page"),
    ("/", (404, b"<html>"), "did not serve the GUI page"),
])
def test_each_wrong_answer_is_reported(serve, route, answer, fragment):
    problems = smoke.check_endpoints(serve(dict(_GOOD, **{route: answer})))
    assert len(problems) == 1 and fragment in problems[0]


def test_a_port_with_nothing_listening_is_not_ready(serve):
    class Dead:
        returncode = None

        def poll(self):
            return None

    base = serve({})
    port = int(base.rsplit(":", 1)[1])
    free = smoke.free_port()
    assert free != port
    why = smoke.wait_ready(f"http://127.0.0.1:{free}", Dead(), smoke.time.monotonic() + 1.5)
    assert why == "the server did not answer /whoami in time"


def test_a_server_that_exited_is_reported_with_its_exit_code(serve):
    class Exited:
        returncode = 3

        def poll(self):
            return 3

    why = smoke.wait_ready("http://127.0.0.1:1", Exited(), smoke.time.monotonic() + 5)
    assert why == "the server exited with code 3 before it answered"
