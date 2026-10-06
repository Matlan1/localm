# SPDX-License-Identifier: AGPL-3.0-or-later
"""REPL /model loads and validates the model on the localm server before it
reports a switch, and the backend's context capacity follows the loaded model.

A real HTTP stub stands in for the localm server: GET /v1/config reports the
loaded model and its context ceiling, POST /v1/models/load loads a registered
model or answers 404.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.backends.http import CoderServerError, HTTPBackend

_CTX = {"model-a": 32768, "model-b": 8192}


class _State:
    def __init__(self):
        self.loaded = "model-a"
        self.load_calls = []
        self.config_calls = 0
        self.report_model = True


def _make_server(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urlparse(self.path).path == "/v1/config":
                state.config_calls += 1
                body = {"effective_ctx_max": _CTX[state.loaded]}
                if state.report_model:
                    body["effective_ctx_model"] = state.loaded
                self._send(200, body)
            else:
                self._send(404, {"detail": "no"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/v1/models/load":
                self._send(404, {"detail": "no"})
                return
            name = parse_qs(url.query).get("model", [""])[0]
            state.load_calls.append(name)
            if name not in _CTX:
                self._send(404, {"detail": f"Model '{name}' is not registered."})
                return
            state.loaded = name
            self._send(200, {"status": "loaded", "model": name})

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def server():
    state = _State()
    srv = _make_server(state)
    yield state, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def _backend(base, model="model-a", **kw):
    return HTTPBackend(base, model, localm_server=True, **kw)


def _agent(backend, cwd):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        return Agent(backend, cwd=cwd, self_verify=False, mode=SessionMode.LOG,
                     auto_approve=True)


def test_context_capacity_is_not_latched_while_the_server_holds_the_old_model(server):
    state, base = server
    backend = _backend(base)
    assert backend.context_capacity() == 32768

    backend.set_model("model-b")
    assert backend.context_capacity() is None
    state.loaded = "model-b"
    assert backend.context_capacity() == 8192
    calls = state.config_calls
    assert backend.context_capacity() == 8192
    assert state.config_calls == calls


def test_a_server_that_does_not_name_the_model_is_still_cached(server):
    state, base = server
    state.report_model = False
    backend = _backend(base)
    assert backend.context_capacity() == 32768
    calls = state.config_calls
    assert backend.context_capacity() == 32768
    assert state.config_calls == calls


def test_load_model_raises_the_servers_explanation_for_an_unknown_model(server):
    state, base = server
    with pytest.raises(CoderServerError, match="not registered"):
        _backend(base).load_model("does-not-exist")


def test_load_model_is_a_noop_for_a_server_that_is_not_localm():
    backend = HTTPBackend("http://127.0.0.1:9/v1", "gpt", localm_server=False)
    with patch("requests.post") as post:
        assert backend.load_model("anything") == {}
    post.assert_not_called()


def test_repl_model_command_refuses_an_unregistered_model(server, tmp_path, capsys):
    from localm.plugins.coder.cli.repl import _handle_command
    state, base = server
    agent = _agent(_backend(base), tmp_path)
    _handle_command("/model does-not-exist", agent)
    out = capsys.readouterr().out
    assert "Model switched" not in out
    assert "not registered" in out
    assert agent.backend.model_id == "model-a"
    assert state.load_calls == ["does-not-exist"]


def test_repl_model_command_loads_the_model_and_budgets_against_it(server, tmp_path, capsys):
    from localm.plugins.coder.cli.repl import _handle_command
    state, base = server
    agent = _agent(_backend(base), tmp_path)
    assert agent.backend.context_capacity() == 32768
    _handle_command("/model model-b", agent)
    out = capsys.readouterr().out
    assert "Model switched to model-b" in out
    assert state.load_calls == ["model-b"]
    assert agent.backend.model_id == "model-b"
    assert agent.backend.context_capacity() == 8192


def test_config_route_names_the_model_the_context_ceiling_belongs_to(tmp_path, monkeypatch):
    import types

    from fastapi.testclient import TestClient

    import localm.config as cfg
    from localm.inference import http_server as hs
    from localm.inference.http_server import create_app

    home = tmp_path / ".localm"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-for-test")
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    auth = {"Authorization": "Bearer owner-key-for-test"}
    with TestClient(create_app(None)) as c:
        monkeypatch.setattr(hs, "_engine", types.SimpleNamespace(
            effective_ctx_max=8192, display_name="model-b"))
        body = c.get("/v1/config", headers=auth).json()
        assert body["effective_ctx_max"] == 8192
        assert body["effective_ctx_model"] == "model-b"
        assert c.patch("/v1/config", json=body, headers=auth).status_code == 200
        monkeypatch.setattr(hs, "_engine", None)
        body = c.get("/v1/config", headers=auth).json()
        assert body["effective_ctx_max"] is None
        assert body["effective_ctx_model"] is None
