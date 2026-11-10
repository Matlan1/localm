# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run` and `localm gui` attach to a running server that serves the
requested model FILE, whatever name the request uses.

A server started from a path keys its model by a display name, so the model
named by that same path, or by a registered name for that file, is the same
model. A loopback stand-in answers the real `/v1/models` and
`/v1/models/{id}` requests; only the HTTP engine (the chat stream) and attach
discovery are stubbed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli.chat import run
from localm.config import save_registry

ACTIVE = "SmolLM2-135M-Instruct-Q4_K_M"


class _StandIn:
    """A loopback server answering GET /v1/models (one active row) and
    GET /v1/models/{id} (the startup model's detail)."""

    def __init__(self, active: str, detail: dict | None, detail_status: int = 200):
        self.active = active
        self.detail = detail
        self.detail_status = detail_status
        self.detail_requests: list[str] = []
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
                    self._send(200, {"object": "list", "data": [
                        {"id": outer.active, "loaded": True, "active": True}]})
                elif self.path.startswith("/v1/models/"):
                    outer.detail_requests.append(unquote(self.path[len("/v1/models/"):]))
                    if outer.detail_status != 200:
                        self._send(outer.detail_status, {"detail": "forbidden"})
                    else:
                        self._send(200, outer.detail)
                else:
                    self._send(404, {"detail": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def model_file(tmp_path):
    f = tmp_path / f"{ACTIVE}.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 60)
    return f


@pytest.fixture
def stand_in(model_file):
    holder = {}

    def make(detail="file", status=200, active=ACTIVE):
        if detail == "file":
            detail = {"id": active, "path": model_file.name,
                      "size_bytes": model_file.stat().st_size, "sha256": None}
        holder["srv"] = _StandIn(active, detail, status)
        return holder["srv"]

    yield make
    if "srv" in holder:
        holder["srv"].close()


@pytest.fixture
def attach(monkeypatch):
    """Attach discovery points at the stand-in; HttpEngine is a spy."""
    engine_spy = MagicMock(name="HttpEngine")
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)

    def to(srv):
        monkeypatch.setattr("localm.instances.attach_target",
                            lambda *a, **k: {"base_url": srv.base_url, "token": None})
    return engine_spy, to


def _flat(output: str) -> str:
    return " ".join(output.split()).lower()


# ------------------------------------------------------------------ #
#  localm run: the same file by another name attaches                  #
# ------------------------------------------------------------------ #

def test_run_by_path_attaches_to_the_server_started_from_that_path(
        attach, stand_in, model_file):
    engine_spy, to = attach
    srv = stand_in()
    to(srv)
    result = CliRunner().invoke(run, [str(model_file), "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert "not the requested" not in _flat(result.output)
    assert engine_spy.called
    assert engine_spy.call_args.kwargs["model"] == ACTIVE
    assert srv.detail_requests == [ACTIVE]


def test_run_by_registered_name_of_that_file_attaches(attach, stand_in, model_file):
    save_registry({"smol": {"path": str(model_file)}})
    engine_spy, to = attach
    to(stand_in())
    result = CliRunner().invoke(run, ["smol", "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert engine_spy.called
    assert engine_spy.call_args.kwargs["model"] == ACTIVE


def test_run_by_path_of_a_different_file_is_still_refused(
        attach, stand_in, model_file, tmp_path):
    other = tmp_path / "other" / model_file.name
    other.parent.mkdir()
    other.write_bytes(b"GGUF" + b"\1" * 200)  # same base name, different size
    engine_spy, to = attach
    to(stand_in())
    result = CliRunner().invoke(run, [str(other), "-p", "hi"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert not engine_spy.called
    assert "not the requested" in flat
    assert "--no-server" in flat


def test_run_by_path_is_refused_when_the_server_identity_cannot_be_read(
        attach, stand_in, model_file):
    engine_spy, to = attach
    to(stand_in(status=403))
    result = CliRunner().invoke(run, [str(model_file), "-p", "hi"])
    assert result.exit_code == 1, result.output
    assert not engine_spy.called


# ------------------------------------------------------------------ #
#  localm run: load options a running server cannot apply              #
# ------------------------------------------------------------------ #

def test_run_attach_names_the_load_options_it_did_not_apply(
        attach, stand_in, model_file):
    engine_spy, to = attach
    to(stand_in())
    result = CliRunner().invoke(
        run, [str(model_file), "-p", "hi", "--ctx", "4096", "-g", "0",
              "--device", "cpu"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert engine_spy.called
    assert "not applied to the running server" in flat
    assert "--ctx" in flat and "--gpu-layers" in flat and "--device" in flat
    assert "--mmproj" not in flat
    assert "--no-server" in flat


def test_run_attach_without_load_options_prints_no_note(attach, stand_in, model_file):
    engine_spy, to = attach
    to(stand_in())
    result = CliRunner().invoke(run, [str(model_file), "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert "not applied" not in _flat(result.output)


# ------------------------------------------------------------------ #
#  localm run: the auto-started server is addressed by its own name    #
# ------------------------------------------------------------------ #

def test_run_autostart_uses_the_new_servers_active_model_name(
        monkeypatch, stand_in, model_file):
    srv = stand_in()
    engine_spy = MagicMock(name="HttpEngine")
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)
    calls = {"n": 0}

    def attach_target(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else {"base_url": srv.base_url, "token": None}

    spawned = []
    monkeypatch.setattr("localm.instances.attach_target", attach_target)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **k: spawned.append(cmd))
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = CliRunner().invoke(run, [str(model_file), "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert spawned and str(model_file) in spawned[0]
    assert engine_spy.called
    assert engine_spy.call_args.kwargs["model"] == ACTIVE
    assert engine_spy.call_args.kwargs["display_name"] == ACTIVE


# ------------------------------------------------------------------ #
#  localm gui: the same file by path attaches                          #
# ------------------------------------------------------------------ #

@pytest.fixture
def gui_running(monkeypatch):
    from localm.plugins.gui import cli as guicli
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")

    def at(srv):
        port = int(srv.base_url.rsplit(":", 1)[1].split("/")[0])
        entry = {"pid": 1, "port": port, "host": "127.0.0.1", "scheme": "http",
                 "mode": "full", "token": None, "root_dir": "/proj"}
        monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: entry)
    return guicli, at


def test_gui_by_path_attaches_to_the_server_serving_that_file(
        gui_running, stand_in, model_file):
    guicli, at = gui_running
    at(stand_in())
    result = CliRunner().invoke(guicli.main, [str(model_file), "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "attaching" in flat
    assert "cannot apply" not in flat


def test_gui_by_path_of_a_different_file_is_refused(
        gui_running, stand_in, model_file, tmp_path):
    other = tmp_path / "elsewhere.gguf"
    other.write_bytes(b"GGUF")
    guicli, at = gui_running
    at(stand_in())
    result = CliRunner().invoke(guicli.main, [str(other), "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "cannot apply" in flat
    assert "attaching" not in flat


def test_run_autostart_refuses_a_server_that_serves_a_different_model(
        monkeypatch, stand_in, model_file, tmp_path):
    srv = stand_in(active="qwen2.5-0.5b",
                   detail={"id": "qwen2.5-0.5b", "path": "qwen.gguf",
                           "size_bytes": 999, "sha256": None})
    engine_spy = MagicMock(name="HttpEngine")
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)
    calls = {"n": 0}

    def attach_target(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else {"base_url": srv.base_url, "token": None}

    monkeypatch.setattr("localm.instances.attach_target", attach_target)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **k: None)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = CliRunner().invoke(run, [str(model_file), "-p", "hi"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert not engine_spy.called
    assert "qwen2.5-0.5b" in flat and "not the requested" in flat
