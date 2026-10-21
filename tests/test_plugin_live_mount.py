# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin installed from the CLI while the server runs is mounted when the
plugin list is next read, not only after a restart."""

import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    return tmp_path


def _make_plugin(root, name, body):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _ping(name):
    return f'''
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/{name}/ping")
        def ping():
            return {{"pong": True}}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass
    '''


def _server(env):
    app = FastAPI()
    app.state.plugin_manager = None
    mgr = PluginManager(app, store_root=env / "store", installed_root=env / "installed")
    mgr.load_enabled()
    return app, mgr


def test_a_plugin_installed_by_the_cli_is_mounted_on_the_next_list(env, monkeypatch):
    from localm.plugins import engine as eng
    app = FastAPI()
    monkeypatch.setattr(eng, "PluginManager", lambda a, **kw: PluginManager(
        a, store_root=env / "store", installed_root=env / "installed"))
    attach_engine(app)
    client = TestClient(app)
    assert client.get("/api/webx/ping").status_code in (404, 405)

    src = _make_plugin(env / "src", "webx", _ping("webx"))
    PluginManager(None, store_root=env / "store", installed_root=env / "installed"
                  ).set_installed_from_dir(src)
    assert client.get("/api/webx/ping").status_code in (404, 405), \
        "the CLI install alone mounts nothing on the running server"

    listing = client.get("/api/plugins").json()
    entry = next(p for p in listing["plugins"] if p["name"] == "webx")
    assert entry["loaded"] is True
    assert client.get("/api/webx/ping").json() == {"pong": True}


def test_a_plugin_that_fails_to_load_is_not_retried_on_every_list(env):
    app, mgr = _server(env)
    src = _make_plugin(env / "src", "badx",
                       'def register(host):\n    raise RuntimeError("boom")\n')
    PluginManager(None, store_root=env / "store", installed_root=env / "installed"
                  ).set_installed_from_dir(src)
    assert mgr.load_enabled_pending() == []
    assert "load failed" in mgr._errors["badx"]
    calls = []
    real = mgr._safe_load
    mgr._safe_load = lambda spec: (calls.append(spec.name), real(spec))[1]
    assert mgr.load_enabled_pending() == []
    assert calls == []


def test_an_already_loaded_plugin_is_not_loaded_twice(env):
    app, mgr = _server(env)
    src = _make_plugin(env / "src", "once", _ping("once"))
    mgr.install_external(src)
    assert mgr.load_enabled_pending() == []
    routes = [r for r in app.router.routes if getattr(r, "path", "") == "/api/once/ping"]
    assert len(routes) == 1
