# SPDX-License-Identifier: AGPL-3.0-or-later
"""A second `localm run` with another model switches the running server.

A loopback stand-in answers the real `GET /v1/models` and
`POST /v1/models/load` requests `localm run` sends; only attach discovery and
the HTTP engine (the chat stream) are stubbed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli.chat import run

FIRST = "qwen2.5-0.5b"
SECOND = "SmolLM2-135M"


class _Server:
    """A loopback server with one active model. ``POST /v1/models/load`` makes
    a model in *loadable* the active one; any other name answers 404 with the
    real route's detail."""

    def __init__(self, active: str | None, loadable: set):
        self.active = active
        self.loadable = loadable
        self.loads: list[dict] = []
        self.auth: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    rows = ([{"id": outer.active, "loaded": True, "active": True}]
                            if outer.active else [])
                    self._send(200, {"object": "list", "data": rows})
                else:
                    self._send(404, {"detail": "not found"})

            def do_POST(self):
                if self.path != "/v1/models/load":
                    self._send(404, {"detail": "not found"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.loads.append(body)
                outer.auth.append(self.headers.get("Authorization"))
                name = body.get("model")
                if name not in outer.loadable:
                    self._send(404, {"detail": f"Model '{name}' is not registered."})
                    return
                outer.active = name
                self._send(200, {"status": "loaded", "model": name})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    """Start a stand-in serving *active* and point attach discovery at it.
    HttpEngine is a spy; starting a background server fails the test."""
    holder = {}
    engine_spy = MagicMock(name="HttpEngine")
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("server started")))

    def make(active, loadable=(FIRST, SECOND)):
        srv = _Server(active, set(loadable))
        holder["srv"] = srv
        monkeypatch.setattr("localm.instances.attach_target",
                            lambda *a, **k: {"base_url": srv.base_url, "token": None})
        return srv, engine_spy

    yield make
    if "srv" in holder:
        holder["srv"].close()


def _flat(output: str) -> str:
    return " ".join(output.split()).lower()


def test_second_run_with_another_model_switches_the_server(server):
    srv, engine_spy = server(FIRST)

    result = CliRunner().invoke(run, [SECOND, "-p", "hi"])

    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [{"model": SECOND}]
    assert srv.active == SECOND
    assert engine_spy.call_args.args[0] == srv.base_url
    assert engine_spy.call_args.kwargs["model"] == SECOND
    assert f"serves {FIRST}; switching it to {SECOND.lower()}" in flat
    assert "connected to the localm server" in flat
    assert "not the requested" not in flat
    assert "--no-server" not in flat


def test_run_with_the_served_model_reuses_it_without_a_switch(server):
    srv, engine_spy = server(FIRST)

    result = CliRunner().invoke(run, [FIRST, "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert srv.loads == []
    assert engine_spy.call_args.kwargs["model"] == FIRST
    assert "switching" not in _flat(result.output)


def test_a_third_run_with_the_switched_model_reuses_it(server):
    srv, engine_spy = server(FIRST)

    first = CliRunner().invoke(run, [SECOND, "-p", "hi"])
    again = CliRunner().invoke(run, [SECOND, "-p", "hi"])

    assert first.exit_code == 0, first.output
    assert again.exit_code == 0, again.output
    assert srv.loads == [{"model": SECOND}], "the second run of it loads nothing"
    assert engine_spy.call_args.kwargs["model"] == SECOND


def test_a_server_with_no_model_is_switched_to_the_requested_one(server):
    srv, engine_spy = server(None)

    result = CliRunner().invoke(run, [SECOND, "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert srv.loads == [{"model": SECOND}]
    assert engine_spy.call_args.kwargs["model"] == SECOND
    assert "has no model loaded; switching it" in _flat(result.output)


def test_a_model_the_server_cannot_load_runs_in_this_process(server, monkeypatch):
    srv, engine_spy = server(FIRST, loadable=(FIRST,))
    lookups = []

    def lookup(name, *a, **k):
        lookups.append(name)
        return None

    monkeypatch.setattr("localm.cli.chat.get_operator_model_info", lookup)

    result = CliRunner().invoke(run, [SECOND, "-p", "hi"])

    flat = _flat(result.output)
    assert srv.loads == [{"model": SECOND}]
    assert srv.active == FIRST, "a refused switch leaves the server as it was"
    assert not engine_spy.called
    assert lookups == [SECOND], "the model loads in this process"
    assert (f"could not switch to {SECOND.lower()} (http 404: model "
            f"'{SECOND.lower()}' is not registered.); loading it in this "
            f"process") in flat
    assert "--no-server" not in flat
