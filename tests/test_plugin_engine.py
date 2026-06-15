"""Plugin engine: discovery, runtime load/unload (route mount/unmount),
enable/disable persistence, protected plugins, and failure isolation.

Uses a synthetic plugin written to a temp dir; runs open-mode (no API key) so
the mounted, auto-scoped routes are reachable.
"""

import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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


def _make_plugin(root, name, body, *, toml_extra=""):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _ping(name):
    return textwrap.dedent(f'''
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/{name}/ping")
        def ping():
            return {{"pong": True}}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass
    ''')


def test_discover_and_parse(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "alpha", _ping("alpha"),
                 toml_extra='version = "2.1.0"\n[surface]\ntab_id = "alpha"\nlabel = "Alpha"\n')
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    specs = mgr.discover()
    assert "alpha" in specs
    assert specs["alpha"].version == "2.1.0"
    assert specs["alpha"].scope == "alpha"
    assert specs["alpha"].surface.tab_id == "alpha"


def test_enable_mounts_routes_disable_unmounts(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "myplug", _ping("myplug"))
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()

    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 404   # not loaded yet

    mgr.enable("myplug")
    with TestClient(app) as c:
        r = c.get("/api/myplug/ping")
        assert r.status_code == 200 and r.json()["pong"] is True

    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert state["myplug"]["enabled"] and state["myplug"]["loaded"]

    mgr.disable("myplug")
    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 404   # unmounted at runtime

    mgr.enable("myplug")                                       # re-enable = fresh import
    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 200


def test_enable_after_catchall_mount_is_not_shadowed(env, tmp_path):
    """Runtime enable must work even after the GUI mounted a catch-all "/" - the
    host relocates the plugin's routes ahead of it, or Starlette's "/" Mount
    would swallow every request (this is the whole point of enable-without-
    restart living alongside the SPA)."""
    import os as _os
    from fastapi.staticfiles import StaticFiles
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "late", _ping("late"))

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("INDEX", encoding="utf-8")

    app = FastAPI()
    app.mount("/", StaticFiles(directory=str(static), html=True), name="gui")

    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.enable("late")                       # mounted AFTER the catch-all "/"
    with TestClient(app) as c:
        assert c.get("/api/late/ping").status_code == 200
        assert c.get("/").text == "INDEX"    # SPA still served

    mgr.disable("late")
    with TestClient(app) as c:
        assert c.get("/api/late/ping").status_code == 404


def test_enabled_state_persists_in_config(env):
    from localm.plugins.engine import PluginManager
    from localm.config import load_config
    plugins = env / "plugins"
    _make_plugin(plugins, "p1", _ping("p1"))
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.enable("p1")
    assert "p1" in load_config().get("plugins_enabled", [])
    mgr.disable("p1")
    assert "p1" not in load_config().get("plugins_enabled", [])


def test_load_enabled_isolates_failures(env):
    from localm.plugins.engine import PluginManager
    from localm.config import load_config, save_config
    plugins = env / "plugins"
    _make_plugin(plugins, "bad",
                 'def register(host):\n    raise RuntimeError("boom")\ndef unregister():\n    pass\n')
    _make_plugin(plugins, "good", _ping("good"))
    cfg = load_config(); cfg["plugins_enabled"] = ["bad", "good"]; save_config(cfg)
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.load_enabled()                       # must NOT raise despite the bad plugin
    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert state["bad"]["loaded"] is False and state["bad"]["error"]
    assert state["good"]["loaded"] is True
    with TestClient(app) as c:
        assert c.get("/api/good/ping").status_code == 200


def test_protected_plugin_cannot_be_disabled(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "core", _ping("core"), toml_extra="protected = true\n")
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.enable("core")
    with pytest.raises(ValueError):
        mgr.disable("core")


def test_enable_unknown_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(FastAPI(), external_root=env / "plugins", builtin_root=None)
    with pytest.raises(KeyError):
        mgr.enable("nope")


def test_parse_spec_rejects_bad_manifest(tmp_path):
    from localm.plugins.engine import parse_spec
    d = tmp_path / "broken"; d.mkdir()
    (d / "plugin.toml").write_text("[plugin]\n", encoding="utf-8")   # no name
    with pytest.raises(ValueError):
        parse_spec(d)
