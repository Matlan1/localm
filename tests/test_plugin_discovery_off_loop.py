# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin discovery behind ``/api/plugins`` and ``/api/capabilities`` runs on a
worker thread, so a slow disk does not stall the event loop; a pending plugin
is still imported and registered on the loop thread. ``localm bug-report``
attaches a running server's stall stacks through
``GET /v1/server/stall-stacks``."""

from __future__ import annotations

import asyncio
import socket as _socket
import textwrap
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins import engine as eng
from localm.plugins.engine import PluginManager, attach_engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(eng, "PluginManager", lambda a, **kw: PluginManager(
        a, store_root=tmp_path / "store", installed_root=tmp_path / "installed"))
    return tmp_path


def _make_plugin(root, name):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(f'''
        import asyncio
        from fastapi import APIRouter
        _r = APIRouter()
        SEEN = []

        @_r.get("/api/{name}/ping")
        def ping():
            return {{"pong": True}}

        def register(host):
            try:
                asyncio.get_running_loop()
                SEEN.append("loop")
            except RuntimeError:
                SEEN.append("thread")
            host.mount_router(_r)
    '''), encoding="utf-8")
    return pdir


def _install_from_cli(env, name):
    src = _make_plugin(env / "src", name)
    PluginManager(None, store_root=env / "store", installed_root=env / "installed"
                  ).set_installed_from_dir(src)


def _engine_app():
    app = FastAPI()
    attach_engine(app)

    @app.get("/ping")
    async def ping():
        return {"ok": True}
    return app


def _gui_app():
    from localm.plugins.gui.web import attach_gui
    app = _engine_app()

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
               active_model=lambda: "model-a")
    return app


class _Blocker:
    """Wraps a manager method: the first call runs *real* (before or after
    blocking, per *after*), records whether a loop was running in its thread,
    and waits on ``release``; later calls only record."""

    def __init__(self, real, *, after=False):
        self.real = real
        self.after = after
        self.on_loop: list[bool] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *a, **kw):
        try:
            asyncio.get_running_loop()
            self.on_loop.append(True)
        except RuntimeError:
            self.on_loop.append(False)
        first = len(self.on_loop) == 1
        if first and not self.after:
            self.entered.set()
            self.release.wait(5)
        out = self.real(*a, **kw)
        if first and self.after:
            self.entered.set()
            self.release.wait(5)
        return out


async def _while_blocked(app, path, blocker, during=None):
    """GET *path*; once *blocker* is entered, GET /ping (and call *during*),
    then release. Returns (slow response, ping response, whether ping was
    answered while the slow request was still blocked)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
        slow = asyncio.ensure_future(c.get(path))
        for _ in range(500):
            if blocker.entered.is_set():
                break
            await asyncio.sleep(0.01)
        ping = await asyncio.wait_for(c.get("/ping"), 4)
        answered_while_blocked = not slow.done() and not blocker.release.is_set()
        if during is not None:
            during()
        blocker.release.set()
        return await slow, ping, answered_while_blocked


@pytest.mark.parametrize("build,path", [(_engine_app, "/api/plugins"),
                                        (_gui_app, "/api/capabilities")])
def test_a_blocked_discovery_does_not_hold_the_event_loop(env, build, path):
    app = build()
    mgr = app.state.plugin_manager
    blocker = _Blocker(mgr.discover)
    mgr.discover = blocker
    slow, ping, answered = asyncio.run(_while_blocked(app, path, blocker))
    assert ping.status_code == 200
    assert answered is True
    assert slow.status_code == 200, slow.text
    assert blocker.on_loop and not any(blocker.on_loop), blocker.on_loop


def test_the_plugins_route_answers_what_api_state_reports(env):
    app = _engine_app()
    mgr = app.state.plugin_manager
    _install_from_cli(env, "webq")
    body = TestClient(app).get("/api/plugins").json()
    assert body == mgr.api_state()
    assert next(p for p in body["plugins"] if p["name"] == "webq")["loaded"] is True


@pytest.mark.parametrize("build,path", [(_engine_app, "/api/plugins"),
                                        (_gui_app, "/api/capabilities")])
def test_a_pending_plugin_is_registered_on_the_loop_thread(env, build, path):
    app = build()
    mgr = app.state.plugin_manager
    _install_from_cli(env, "webx")
    with TestClient(app) as client:
        assert client.get(path).status_code == 200
        assert client.get("/api/webx/ping").json() == {"pong": True}
    assert mgr._loaded["webx"][1].SEEN == ["loop"]


def test_a_plugin_disabled_while_discovery_runs_is_not_loaded(env):
    app = _engine_app()
    mgr = app.state.plugin_manager
    _install_from_cli(env, "webd")
    blocker = _Blocker(mgr._enabled_set, after=True)
    mgr._enabled_set = blocker
    slow, _, answered = asyncio.run(_while_blocked(
        app, "/api/plugins", blocker, during=lambda: mgr.disable("webd")))
    assert answered is True
    assert slow.status_code == 200
    assert "webd" not in mgr._loaded
    entry = next(p for p in slow.json()["plugins"] if p["name"] == "webd")
    assert entry["enabled"] is False and entry["loaded"] is False


def _two_installed(env):
    for n in ("pa", "pb"):
        _install_from_cli(env, n)
    return PluginManager(None, store_root=env / "store", installed_root=env / "installed")


def test_a_reader_sees_the_previous_complete_map_while_discover_runs(env, monkeypatch):
    mgr = _two_installed(env)
    mgr.discover()
    assert set(mgr._specs) == {"pa", "pb"}
    real = eng.parse_spec
    entered, release = threading.Event(), threading.Event()

    def slow_parse(path, *a, **kw):
        if path.name == "pb":
            entered.set()
            release.wait(5)
        return real(path, *a, **kw)

    monkeypatch.setattr(eng, "parse_spec", slow_parse)
    t = threading.Thread(target=mgr.discover, daemon=True)
    t.start()
    try:
        assert entered.wait(5)
        seen = set(mgr._specs)
    finally:
        release.set()
        t.join(5)
    assert seen == {"pa", "pb"}
    assert set(mgr._specs) == {"pa", "pb"}


def test_concurrent_discovers_run_one_at_a_time(env, monkeypatch):
    mgr = _two_installed(env)
    real = eng.parse_spec
    active, peak = [0], [0]
    guard = threading.Lock()

    def counting_parse(path, *a, **kw):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return real(path, *a, **kw)

    monkeypatch.setattr(eng, "parse_spec", counting_parse)
    threads = [threading.Thread(target=mgr.discover, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert peak[0] == 1
    assert set(mgr._specs) == {"pa", "pb"}


# --------------------------------------------------------------------------- #
#  GET /v1/server/stall-stacks                                                #
# --------------------------------------------------------------------------- #

def _server_app(monkeypatch, bind_host="127.0.0.1"):
    from unittest.mock import MagicMock

    from localm.inference import _stall_watch as sw
    from localm.inference.http_server import create_app
    monkeypatch.setattr(sw, "recent_stall_stacks", lambda: "ROUTE-STALL-STACKS")
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.loaded = True
    app = create_app(engine)
    if bind_host is not None:
        app.state.bind_host = bind_host
    return app


def test_stall_stacks_are_served_to_the_operator_on_a_loopback_bind(env, monkeypatch):
    app = _server_app(monkeypatch)
    r = TestClient(app).get("/v1/server/stall-stacks",
                            headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    assert r.json() == {"stall_stacks": "ROUTE-STALL-STACKS"}


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "192.168.1.20", None])
def test_stall_stacks_are_refused_unless_bound_to_loopback(env, monkeypatch, bind_host):
    from localm import auth
    auth.set_api_key("owner-key-for-stall-stacks-0123")
    app = _server_app(monkeypatch, bind_host=bind_host)
    r = TestClient(app).get("/v1/server/stall-stacks",
                            headers={"Authorization": "Bearer owner-key-for-stall-stacks-0123"})
    assert r.status_code == 403, r.text
    assert "ROUTE-STALL-STACKS" not in r.text


def test_stall_stacks_need_the_owner_key(env, monkeypatch):
    from localm import auth
    auth.set_api_key("owner-key-for-stall-stacks-0123")
    scoped = auth.create_key("reader", ["config:read", "chat"])["key"]
    app = _server_app(monkeypatch)
    client = TestClient(app)
    r = client.get("/v1/server/stall-stacks", headers={"Authorization": f"Bearer {scoped}"})
    assert r.status_code == 403, r.text
    assert "ROUTE-STALL-STACKS" not in r.text
    r = client.get("/v1/server/stall-stacks",
                   headers={"Authorization": "Bearer owner-key-for-stall-stacks-0123"})
    assert r.status_code == 200 and r.json()["stall_stacks"] == "ROUTE-STALL-STACKS"


# --------------------------------------------------------------------------- #
#  localm bug-report against a real running server                            #
# --------------------------------------------------------------------------- #

class _RealServer:
    def __init__(self, app, port, server, thread):
        self.app, self.port, self.server, self.thread = app, port, server, thread

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


def _start_real_server(monkeypatch, bind_host) -> _RealServer:
    app = _server_app(monkeypatch, bind_host=bind_host)
    app.state.instance_token = "real-instance-token-0123456789"
    app.state.instance_id = "stallsrv01"
    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                              daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    return _RealServer(app, port, server, thread)


def _register(home, rs):
    from localm import instances
    instances.register_instance(home, instance_id="stallsrv01", port=rs.port,
                                host="127.0.0.1", root_dir=str(home), mode="full",
                                token="real-instance-token-0123456789")


def _report_text(home):
    reports = sorted((home / "bug-reports").glob("bug-*.md"))
    assert reports, "no report file written"
    return reports[-1].read_text(encoding="utf-8")


def test_bug_report_attaches_a_running_servers_stall_stacks(cli_runner, monkeypatch):
    import localm.config as cfg
    from localm.cli import main
    home = cfg.HOME_DIR
    rs = _start_real_server(monkeypatch, "127.0.0.1")
    try:
        _register(home, rs)
        result = cli_runner.invoke(main, ["bug-report", "-m", "it froze"])
    finally:
        rs.stop()
    assert result.exit_code == 0, result.output
    text = _report_text(home)
    assert "Server hang trace" in text and "ROUTE-STALL-STACKS" in text
    assert "Could not read the stall record" not in result.output


def test_bug_report_without_a_running_server_is_unchanged(cli_runner):
    import localm.config as cfg
    from localm.cli import main
    result = cli_runner.invoke(main, ["bug-report", "-m", "it froze"])
    assert result.exit_code == 0, result.output
    text = _report_text(cfg.HOME_DIR)
    assert "Server hang trace" not in text
    assert "Could not read the stall record" not in result.output


def test_bug_report_names_a_server_that_refuses(cli_runner, monkeypatch):
    import localm.config as cfg
    from localm.cli import main
    home = cfg.HOME_DIR
    rs = _start_real_server(monkeypatch, "0.0.0.0")
    try:
        _register(home, rs)
        result = cli_runner.invoke(main, ["bug-report", "-m", "it froze"])
    finally:
        rs.stop()
    assert result.exit_code == 0, result.output
    assert "Could not read the stall record" in result.output
    assert "unauthorized 403" in result.output
    assert "ROUTE-STALL-STACKS" not in _report_text(home)
