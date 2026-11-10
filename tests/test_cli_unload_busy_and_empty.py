# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm unload` reports what the server's unload reply says: a busy model the
server left loaded is not "unloaded", `--force` asks the server to stop it, and
an unload of everything with nothing loaded says so. The reply comes from a
loopback stand-in for `POST /v1/models/unload` that records its query."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from click.testing import CliRunner

from localm.cli import main


@pytest.fixture
def server(monkeypatch):
    state = {"reply": {}, "queries": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/v1/models/unload":
                self.send_response(404)
                self.end_headers()
                return
            state["queries"].append(parse_qs(u.query))
            raw = json.dumps(state["reply"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("LOCALM_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    yield state
    httpd.shutdown()
    httpd.server_close()


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_unload_of_a_busy_model_is_not_reported_as_unloaded(server):
    server["reply"] = {"status": "confirm_required", "model": "M",
                       "detail": "M is writing a reply for chat 'demo'."}
    result = CliRunner().invoke(main, ["unload", "M"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "unloaded: M" not in flat
    assert "'M' is busy with a running request and was not unloaded" in flat
    assert "M is writing a reply for chat 'demo'." in flat
    assert "--force" in flat
    assert server["queries"] == [{"model": ["M"]}]


def test_unload_force_asks_the_server_to_stop_the_model(server):
    server["reply"] = {"status": "unloaded", "model": "M", "unloaded_models": ["M"],
                       "unloaded_other": [], "embedder_unloaded": False}
    result = CliRunner().invoke(main, ["unload", "M", "--force"])
    assert result.exit_code == 0, result.output
    assert "unloaded: M" in _flat(result.output)
    assert server["queries"] == [{"model": ["M"], "force": ["true"]}]


def test_unload_without_force_sends_no_force(server):
    server["reply"] = {"status": "unloaded", "model": "M", "unloaded_models": ["M"]}
    CliRunner().invoke(main, ["unload", "M"])
    assert "force" not in server["queries"][0]


def test_unload_everything_with_nothing_loaded_says_nothing_was_loaded(server):
    server["reply"] = {"status": "already_unloaded", "model": "none",
                       "unloaded_models": [], "unloaded_other": [],
                       "embedder_unloaded": False}
    result = CliRunner().invoke(main, ["unload"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "Nothing was loaded." in flat
    assert "'none'" not in flat
    assert server["queries"] == [{}]


def test_unload_of_a_named_model_not_loaded_still_names_it(server):
    server["reply"] = {"status": "already_unloaded", "model": "B"}
    result = CliRunner().invoke(main, ["unload", "B"])
    assert result.exit_code == 0, result.output
    assert "'B' was not loaded - nothing to do." in _flat(result.output)
