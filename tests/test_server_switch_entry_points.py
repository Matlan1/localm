# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every command that meets a running localm server with a model of its own
switches that server to the model instead of refusing.

A loopback stand-in answers the real `GET /v1/models`, `GET /v1/models/{id}`,
`GET /v1/config` and `POST /v1/models/load` requests. `localm gui`,
`localm serve`, `localm coder` and the launcher's command lines are driven
through their real click commands; only instance discovery, the coder's HTTP
backend and starting a new server are stubbed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm import auth
from localm.audit import MODE_ENV_VAR
from localm.config import save_registry

FIRST = "qwen2.5-0.5b"
SECOND = "SmolLM2-135M"
REPO_DIR = Path(__file__).resolve().parent.parent


class _Server:
    """A loopback localm stand-in with one active model and a session mode.
    ``POST /v1/models/load`` makes a model in *loadable* the active one; any
    other name answers 404 with the real route's detail."""

    def __init__(self, active, loadable, mode="privacy"):
        self.active = active
        self.loadable = set(loadable)
        self.mode = mode
        self.loads: list = []
        self.load_auth: list = []
        self.get_auth: list = []
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
                outer.get_auth.append(self.headers.get("Authorization"))
                if self.path == "/v1/models":
                    rows = ([{"id": outer.active, "loaded": True, "active": True}]
                            if outer.active else [])
                    self._send(200, {"object": "list", "data": rows})
                elif self.path == "/v1/config":
                    self._send(200, {"effective_mode": outer.mode})
                else:
                    self._send(404, {"detail": "not found"})

            def do_POST(self):
                if self.path != "/v1/models/load":
                    self._send(404, {"detail": "not found"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.loads.append(body.get("model"))
                outer.load_auth.append(self.headers.get("Authorization"))
                name = body.get("model")
                if name not in outer.loadable:
                    self._send(404, {"detail": f"Model '{name}' is not registered."})
                    return
                outer.active = name
                self._send(200, {"status": "loaded", "model": name})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _clean_mode_env():
    os.environ.pop(MODE_ENV_VAR, None)
    yield
    os.environ.pop(MODE_ENV_VAR, None)


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Start a stand-in and point every discovery seam at it. Both models are
    registered here; starting a second server fails the test."""
    holder = {}
    for name in (FIRST, SECOND):
        (tmp_path / f"{name}.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    save_registry({name: {"path": str(tmp_path / f"{name}.gguf")}
                   for name in (FIRST, SECOND)})
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
    monkeypatch.setattr("localm.portmux.run_server", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a second server was started")))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a background server was started")))
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)

    def make(active=FIRST, loadable=(FIRST, SECOND), mode="privacy"):
        srv = _Server(active, loadable, mode)
        holder["srv"] = srv
        entry = {"pid": 1, "port": srv.port, "host": "127.0.0.1", "scheme": "http",
                 "mode": "full", "token": "tok", "root_dir": "/proj"}
        target = {"base_url": srv.base_url, "token": "tok", "port": srv.port,
                  "scheme": "http", "mode": "full"}
        monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: entry)
        monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: target)
        return srv

    yield make
    if "srv" in holder:
        holder["srv"].close()


def _flat(output: str) -> str:
    return " ".join(output.split()).lower()


def _gui(args):
    from localm.plugins.gui import cli as guicli
    return CliRunner().invoke(guicli.main, list(args))


def _serve(args):
    from localm.cli import main as root
    return CliRunner().invoke(root, ["serve", *args])


@pytest.fixture
def coder_backend(monkeypatch):
    """The coder's HTTP backend records what it was built with and stops the
    command there."""
    import localm.plugins.coder.cli._main as cli_main
    built = {}

    def fake_backend(base_url, model, api_key="localm", **kw):
        built.update(base_url=base_url, model=model, api_key=api_key)
        raise SystemExit(0)

    monkeypatch.setattr(cli_main, "HTTPBackend", fake_backend)
    return built


def _coder(args):
    import localm.plugins.coder.cli as ccli
    return CliRunner().invoke(ccli.main, list(args))


# ------------------------------------------------------------------ #
#  localm gui / localm serve                                           #
# ------------------------------------------------------------------ #

def test_gui_with_another_model_switches_the_running_server(server):
    srv = server()
    result = _gui([SECOND, "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert srv.active == SECOND
    assert srv.load_auth == ["Bearer tok"]
    assert "attaching" in flat
    assert f"serves {FIRST}; switching it to {SECOND.lower()}" in flat
    assert f"it now serves {SECOND.lower()}" in flat
    assert "cannot apply" not in flat


def test_gui_with_the_served_model_reuses_it(server):
    srv = server()
    result = _gui([FIRST, "--no-browser"])
    assert result.exit_code == 0, result.output
    assert srv.loads == []
    assert "switching" not in _flat(result.output)


def test_gui_switch_sends_the_owner_key_when_one_is_configured(server):
    srv = server()
    auth.set_api_key("real-owner-key-0123456789")
    result = _gui([SECOND, "--no-browser"])
    assert result.exit_code == 0, result.output
    assert srv.load_auth == ["Bearer real-owner-key-0123456789"]


def test_gui_with_a_model_the_server_cannot_load_attaches_and_exits_1(server):
    srv = server(loadable=(FIRST,))
    result = _gui([SECOND, "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert srv.loads == [SECOND]
    assert srv.active == FIRST
    assert "attaching" in flat
    assert (f"could not switch to {SECOND.lower()} (http 404: model "
            f"'{SECOND.lower()}' is not registered.); it keeps serving "
            f"{FIRST}") in flat
    assert "cannot apply" not in flat


def test_gui_with_load_options_switches_and_names_them_not_applied(server):
    srv = server()
    result = _gui([SECOND, "--ctx", "4096", "-g", "0", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert "with its own settings (--ctx, --gpu-layers not applied)" in flat
    assert "not applied to the running server: --ctx, --gpu-layers" in flat
    assert "--new" in flat
    assert "cannot apply" not in flat


def test_gui_a_switch_is_attempted_when_the_model_cannot_be_read(server, monkeypatch):
    srv = server()
    monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                        lambda *a, **k: ("unknown", None))
    result = _gui([SECOND, "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert "did not say which model it serves; switching it" in flat


def test_gui_with_the_mode_the_server_records_in_attaches(server):
    srv = server(mode="log")
    result = _gui([SECOND, "--mode", "log", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert "cannot apply" not in flat


def test_gui_with_another_session_mode_names_it(server):
    srv = server(mode="full")
    result = _gui([SECOND, "--mode", "privacy", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "--mode privacy (the running server records in full mode)" in flat
    assert srv.loads == []


def test_serve_with_another_model_switches_the_running_server(server):
    srv = server()
    result = _serve([SECOND])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert srv.active == SECOND
    assert "attaching" in flat
    assert "it also serves the gui" in flat


# ------------------------------------------------------------------ #
#  localm coder                                                        #
# ------------------------------------------------------------------ #

def test_coder_with_another_model_switches_the_running_server(server, coder_backend):
    srv = server()
    result = _coder(["--model", SECOND, "hi"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert coder_backend["model"] == SECOND
    assert coder_backend["base_url"] == srv.base_url
    assert f"serves {FIRST}; switching it to {SECOND.lower()}" in flat


def test_coder_with_the_served_model_reuses_it(server, coder_backend):
    srv = server()
    result = _coder(["--model", FIRST, "hi"])
    assert result.exit_code == 0, result.output
    assert srv.loads == []
    assert coder_backend["model"] == FIRST


def test_coder_names_why_a_switch_failed(server, coder_backend):
    srv = server(loadable=(FIRST,))
    result = _coder(["--model", SECOND, "hi"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert coder_backend["model"] == SECOND
    assert f"could not switch to {SECOND.lower()}" in flat
    assert f"requests name {SECOND.lower()}" in flat


OWNER_KEY = "real-owner-key-0123456789"
USER_KEY = "user-given-key-9876543210"


def test_coder_api_key_is_sent_with_the_switch_and_the_chat(server, coder_backend):
    srv = server()
    auth.set_api_key(OWNER_KEY)
    result = _coder(["--model", SECOND, "--api-key", USER_KEY, "hi"])
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert srv.load_auth == [f"Bearer {USER_KEY}"]
    assert srv.get_auth and set(srv.get_auth) == {f"Bearer {USER_KEY}"}
    assert coder_backend["api_key"] == USER_KEY


def test_coder_without_api_key_sends_the_owner_key(server, coder_backend, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    srv = server()
    auth.set_api_key(OWNER_KEY)
    result = _coder(["--model", SECOND, "hi"])
    assert result.exit_code == 0, result.output
    assert srv.load_auth == [f"Bearer {OWNER_KEY}"]
    assert set(srv.get_auth) == {f"Bearer {OWNER_KEY}"}
    assert coder_backend["api_key"] == OWNER_KEY


def test_coder_without_api_key_in_open_mode_sends_the_instance_token(
        server, coder_backend, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    srv = server()
    result = _coder(["--model", SECOND, "hi"])
    assert result.exit_code == 0, result.output
    assert srv.load_auth == ["Bearer tok"]
    assert coder_backend["api_key"] == "tok"


def test_every_request_of_a_switch_carries_the_given_credential(server):
    from localm.inference.server_switch import serve_requested_model
    srv = server()
    auth.set_api_key(OWNER_KEY)
    served = serve_requested_model(srv.base_url, USER_KEY, SECOND)
    assert served.switched and served.name == SECOND
    assert srv.load_auth == [f"Bearer {USER_KEY}"]
    assert srv.get_auth and set(srv.get_auth) == {f"Bearer {USER_KEY}"}


def test_a_switch_without_a_credential_sends_no_authorization(server):
    from localm.inference.server_switch import switch_server_model
    srv = server()
    auth.set_api_key(OWNER_KEY)
    name, why = switch_server_model(srv.base_url, None, SECOND)
    assert (name, why) == (SECOND, None)
    assert srv.load_auth == [None]


# ------------------------------------------------------------------ #
#  The launcher's command lines                                        #
# ------------------------------------------------------------------ #

def _load_launcher():
    path = REPO_DIR / "launcher.pyw"
    loader = importlib.machinery.SourceFileLoader("localm_launcher", str(path))
    spec = importlib.util.spec_from_file_location("localm_launcher", str(path),
                                                  loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher_argv(mode: str, model: str, privacy: str) -> list:
    """The localm arguments the launcher's Launch button builds for *mode*."""
    launcher = _load_launcher()

    class _Var:
        def __init__(self, v):
            self._v = v

        def get(self):
            return self._v

    fields = dict(
        mode=mode, model=model, ctx="", gpu_layers="", port="",
        privacy_global=privacy, privacy_chat=launcher.USE_GLOBAL,
        privacy_coder=launcher.USE_GLOBAL, no_browser=True, host_lan=False,
        debug=False, keep_diagnostics=False, coder_dir="", coder_yes=False)

    class _Fake:
        pass

    fake = _Fake()
    fake.status_msg = lambda *a, **k: None
    for key, val in fields.items():
        setattr(fake, key, _Var(val))
    cmd = launcher.Launcher._build_command(fake)
    return cmd[cmd.index("localm") + 1:]


@pytest.mark.parametrize("mode", ["gui", "serve"])
def test_launcher_gui_and_api_server_switch_the_running_server(server, mode):
    srv = server(mode="log")
    argv = _launcher_argv(mode, SECOND, "log")
    assert argv[0] == "gui" and "--mode" in argv
    result = _gui(argv[1:])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert "attaching" in flat
    assert "cannot apply" not in flat


def test_launcher_chat_switches_the_running_server(server, monkeypatch):
    from localm.cli.chat import run
    srv = server(mode="log")
    engine_spy = MagicMock(name="HttpEngine")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    argv = _launcher_argv("chat", SECOND, "log")
    assert argv[0] == "run"
    result = CliRunner().invoke(run, [*argv[1:], "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert engine_spy.call_args.kwargs["model"] == SECOND


def test_launcher_coder_switches_the_running_server(server, coder_backend):
    srv = server(mode="log")
    argv = _launcher_argv("coder", SECOND, "log")
    assert argv[0] == "coder"
    result = _coder(argv[1:])
    assert result.exit_code == 0, result.output
    assert srv.loads == [SECOND]
    assert coder_backend["model"] == SECOND
