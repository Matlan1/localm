# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_on_install_hook_fires(env):
    """The optional on_install lifecycle hook is invoked when a plugin is
    installed (symmetric with on_uninstall). Negative-testable: without the
    engine's _invoke_hook(name, "on_install") call, the marker is never written."""
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    body = '''
        from pathlib import Path
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/hooked/ping")
        def ping():
            return {"pong": True}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass

        def on_install():
            (Path(__file__).parent / "on_install_called").write_text("yes")
    '''
    _make_plugin(plugins, "hooked", body)
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.discover()
    marker = plugins / "hooked" / "on_install_called"
    assert not marker.exists()          # hook has not fired before install
    mgr.install("hooked")
    assert marker.exists()              # on_install fired during install


def test_install_mounts_routes_disable_unmounts(env):
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "myplug", _ping("myplug"))
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()

    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 404   # not installed yet

    mgr.install("myplug")                                      # install = installed + enabled + loaded
    with TestClient(app) as c:
        r = c.get("/api/myplug/ping")
        assert r.status_code == 200 and r.json()["pong"] is True

    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert state["myplug"]["installed"] and state["myplug"]["enabled"]
    assert state["myplug"]["active"] and state["myplug"]["loaded"]

    mgr.disable("myplug")                                      # stays installed, goes inactive
    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 404   # unmounted at runtime
    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert state["myplug"]["installed"] and not state["myplug"]["enabled"]

    mgr.enable("myplug")                                       # re-enable an installed plugin
    with TestClient(app) as c:
        assert c.get("/api/myplug/ping").status_code == 200


def test_api_state_exposes_catalog_commands_and_suggest_flag(env):
    """api_state carries each first-party plugin's command verbs (for the GUI's
    'needs the X plugin' hint) and mirrors the suggest_plugins config toggle."""
    from localm.config import save_config
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(FastAPI(), store_root=env / "store",
                        installed_root=env / "installed")
    state = mgr.api_state()
    assert state["suggest_plugins"] is True
    by_name = {p["name"]: p for p in state["plugins"]}
    # the (available, not installed) image plugin advertises its renamed verb
    assert by_name["image"]["commands"] == ["generate-image"]
    assert by_name["image"]["active"] is False

    save_config({"suggest_plugins": False})
    assert mgr.api_state()["suggest_plugins"] is False


def test_enable_requires_install_first(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "p1", _ping("p1"))               # available, not installed
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=env / "installed")
    with pytest.raises(ValueError):
        mgr.enable("p1")                                  # in the store but not installed


def test_uninstall_external_clears_axes_and_removes_dir(env):
    """Uninstalling a THIRD-PARTY plugin clears both axes AND deletes its copied
    directory (it leaves the catalog entirely)."""
    from localm.config import load_config
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "p1", _ping("p1"))
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.install("p1")
    assert mgr.uninstall("p1") is True
    cfg = load_config()
    assert "p1" not in cfg.get("plugins_installed", [])
    assert "p1" not in cfg.get("plugins_enabled", [])
    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert "p1" not in state                 # external dir deleted -> off the catalog
    assert not (plugins / "p1").exists()


def test_uninstall_builtin_stays_in_catalog(env):
    """Uninstalling a FIRST-PARTY builtin clears both axes but keeps its code in
    the bundled catalog (so it can be reinstalled)."""
    from localm.config import load_config
    from localm.plugins.engine import PluginManager
    builtins = env / "builtins"
    _make_plugin(builtins, "b1", _ping("b1"))
    mgr = PluginManager(FastAPI(), external_root=env / "none", builtin_root=builtins)
    mgr.install("b1")
    assert mgr.uninstall("b1") is True
    cfg = load_config()
    assert "b1" not in cfg.get("plugins_installed", [])
    assert "b1" not in cfg.get("plugins_enabled", [])
    state = {p["name"]: p for p in mgr.api_state()["plugins"]}
    assert "b1" in state and not state["b1"]["installed"]   # still in catalog
    assert (builtins / "b1").exists()


# ---------------------------------------------------------------------------
#  _delete_plugin_data: data_subdir is taken verbatim from a (possibly
#  third-party) manifest. It must NEVER let rmtree escape the data dir via
#  traversal ('../models'), an absolute path, or '.' (the home root itself).
# ---------------------------------------------------------------------------

def _mgr_with_data_subdir(tmp_path, monkeypatch, data_subdir):
    """Build a discovered manager whose plugin 'evil' declares *data_subdir*,
    with LOCALM_HOME pinned to a subdir of tmp_path so siblings can be
    created safely outside the data dir."""
    from localm.plugins.engine import PluginManager
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    plugins = home / "plugins"
    _make_plugin(plugins, "evil", _ping("evil"),
                 toml_extra=f'data_subdir = "{data_subdir}"\n')
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.discover()
    return mgr, home


def test_delete_data_rejects_parent_traversal(tmp_path, monkeypatch):
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, "../victim")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("important", encoding="utf-8")
    mgr._delete_plugin_data(mgr._specs["evil"])
    assert victim.exists() and (victim / "keep.txt").exists()


def test_delete_data_rejects_dot_root(tmp_path, monkeypatch):
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, ".")
    (home / "models").mkdir()
    (home / "models" / "big.gguf").write_text("data", encoding="utf-8")
    mgr._delete_plugin_data(mgr._specs["evil"])
    assert home.exists() and (home / "models" / "big.gguf").exists()


def test_delete_data_deletes_legit_subdir(tmp_path, monkeypatch):
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, "evil_data")
    target = home / "evil_data"
    target.mkdir()
    (target / "cache.bin").write_text("x", encoding="utf-8")
    mgr._delete_plugin_data(mgr._specs["evil"])
    assert not target.exists()


def test_enable_after_catchall_mount_is_not_shadowed(env, tmp_path):
    """Runtime enable must work even after the GUI mounted a catch-all "/" - the
    host relocates the plugin's routes ahead of it, or Starlette's "/" Mount
    would swallow every request (this is the whole point of enable-without-
    restart living alongside the SPA)."""
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
    mgr.install("late")                      # mounted AFTER the catch-all "/"
    with TestClient(app) as c:
        assert c.get("/api/late/ping").status_code == 200
        assert c.get("/").text == "INDEX"    # SPA still served

    mgr.disable("late")
    with TestClient(app) as c:
        assert c.get("/api/late/ping").status_code == 404


def test_install_places_dir_and_persists_enabled(env):
    """install copies the plugin from the store into the installed folder
    (physical 'installed') and enables it; disable keeps it on disk."""
    from localm.plugins.engine import PluginManager
    from localm.config import load_config
    store = env / "store"
    inst = env / "installed"
    _make_plugin(store, "p1", _ping("p1"))
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.install("p1")
    assert (inst / "p1" / "plugin.toml").is_file()       # physically installed
    assert mgr.is_installed("p1")
    assert "p1" in load_config().get("plugins_enabled", [])
    mgr.disable("p1")
    assert (inst / "p1").is_dir()                         # stays installed (on disk)
    assert "p1" not in load_config().get("plugins_enabled", [])


def test_load_enabled_isolates_failures(env):
    from localm.plugins.engine import PluginManager
    from localm.config import load_config, save_config
    plugins = env / "plugins"
    _make_plugin(plugins, "bad",
                 'def register(host):\n    raise RuntimeError("boom")\ndef unregister():\n    pass\n')
    _make_plugin(plugins, "good", _ping("good"))
    cfg = load_config()
    cfg["plugins_enabled"] = ["bad", "good"]   # installed = physical (both on disk)
    save_config(cfg)
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
    mgr.install("core")
    with pytest.raises(ValueError):
        mgr.disable("core")


def test_enable_unknown_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(FastAPI(), external_root=env / "plugins", builtin_root=None)
    with pytest.raises(KeyError):
        mgr.enable("nope")


def test_uninstall_unknown_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(FastAPI(), external_root=env / "plugins", builtin_root=None)
    with pytest.raises(KeyError):
        mgr.uninstall("ghost")


def test_install_rolls_back_on_load_failure(env):
    """A plugin whose register() raises must NOT leave installed/enabled config
    behind - install loads first and only persists on success."""
    from localm.config import load_config
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "broken",
                 'def register(host):\n    raise RuntimeError("boom")\n'
                 'def unregister():\n    pass\n')
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    with pytest.raises(RuntimeError):
        mgr.install("broken")
    cfg = load_config()
    assert "broken" not in cfg.get("plugins_installed", [])
    assert "broken" not in cfg.get("plugins_enabled", [])
    assert not mgr.is_installed("broken")


def test_enable_rolls_back_on_load_failure(env):
    """enable() on an installed plugin whose load fails must not persist enabled."""
    from localm.config import load_config
    from localm.plugins.engine import PluginManager
    store = env / "store"
    inst = env / "installed"
    _make_plugin(store, "broken",
                 'def register(host):\n    raise RuntimeError("boom")\n'
                 'def unregister():\n    pass\n')
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.set_installed_state("broken", True, enable=False)   # copy store->installed, not enabled
    with pytest.raises(RuntimeError):
        mgr.enable("broken")
    assert (inst / "broken").is_dir()                       # stays installed
    assert "broken" not in load_config().get("plugins_enabled", [])  # enable rolled back


def test_set_installed_state_without_app(env):
    """CLI/headless toggle (app is None): install copies store->installed + enables;
    disable keeps it on disk; uninstall removes the dir."""
    from localm.plugins.engine import PluginManager
    store = env / "store"
    inst = env / "installed"
    _make_plugin(store, "p1", _ping("p1"))
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    mgr.set_installed_state("p1", True)
    assert mgr.is_installed("p1") and mgr.is_enabled("p1")
    assert (inst / "p1").is_dir()
    # disable keeps it installed (on disk)
    mgr.set_enabled_state("p1", False)
    assert mgr.is_installed("p1") and not mgr.is_enabled("p1")
    # uninstall removes the dir and clears enabled
    mgr.set_installed_state("p1", False)
    assert not mgr.is_installed("p1") and not mgr.is_enabled("p1")
    assert not (inst / "p1").exists()


def test_set_enabled_state_requires_install(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "p1", _ping("p1"))
    mgr = PluginManager(None, store_root=store, installed_root=env / "installed")
    with pytest.raises(ValueError):
        mgr.set_enabled_state("p1", True)        # available but not installed


def test_set_installed_state_unknown_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(None, external_root=env / "plugins", builtin_root=None)
    with pytest.raises(KeyError):
        mgr.set_installed_state("nope", True)


def test_set_installed_state_protected_cannot_uninstall(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "core", _ping("core"), toml_extra="protected = true\n")
    mgr = PluginManager(None, store_root=store, installed_root=env / "installed")
    mgr.set_installed_state("core", True)
    with pytest.raises(ValueError):
        mgr.set_installed_state("core", False)


def test_set_enabled_state_protected_cannot_disable(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "core", _ping("core"), toml_extra="protected = true\n")
    mgr = PluginManager(None, store_root=store, installed_root=env / "installed")
    mgr.set_installed_state("core", True)
    with pytest.raises(ValueError):
        mgr.set_enabled_state("core", False)


def test_uninstall_protected_raises(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "core", _ping("core"), toml_extra="protected = true\n")
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=env / "installed")
    mgr.install("core")
    with pytest.raises(ValueError):
        mgr.uninstall("core")


def test_set_installed_state_rejects_broken_manifest(env):
    """CLI install copies the plugin, validates it, and rolls back a broken one."""
    from localm.plugins.engine import PluginManager
    store = env / "store"
    inst = env / "installed"
    d = store / "broken"
    d.mkdir(parents=True)
    (d / "plugin.toml").write_text("this is not valid toml [[[", encoding="utf-8")
    mgr = PluginManager(None, store_root=store, installed_root=inst)
    with pytest.raises(ValueError):
        mgr.set_installed_state("broken", True)
    assert not (inst / "broken").exists()       # rolled back, not orphaned


def test_missing_requires(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    _make_plugin(store, "needy", _ping("needy"), toml_extra='requires = ["dep1"]\n')
    _make_plugin(store, "dep1", _ping("dep1"))
    mgr = PluginManager(None, store_root=store, installed_root=env / "installed")
    mgr.set_installed_state("needy", True)
    assert mgr.missing_requires("needy") == ["dep1"]
    mgr.set_installed_state("dep1", True)
    assert mgr.missing_requires("needy") == []


def test_parse_spec_rejects_bad_manifest(tmp_path):
    from localm.plugins.engine import parse_spec
    d = tmp_path / "broken"; d.mkdir()
    (d / "plugin.toml").write_text("[plugin]\n", encoding="utf-8")   # no name
    with pytest.raises(ValueError):
        parse_spec(d)


def test_attach_engine_http_install_lifecycle(env, monkeypatch):
    """The /api/plugins/* management surface over the real shipped builtins:
    nothing installed by default; install -> enable/disable -> uninstall."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)                       # default roots = the real builtins
    with TestClient(app) as c:
        st = {p["name"]: p for p in c.get("/api/plugins").json()["plugins"]}
        assert st["voice"]["installed"] is False        # available, not installed
        assert st["voice"]["loaded"] is False           # nothing active by default

        assert c.post("/api/plugins/voice/install").status_code == 200
        st = {p["name"]: p for p in c.get("/api/plugins").json()["plugins"]}
        assert st["voice"]["installed"] and st["voice"]["active"]

        # enabling a not-installed plugin is a 409 (install first)
        assert c.post("/api/plugins/web/enable").status_code == 409

        assert c.post("/api/plugins/voice/disable").status_code == 200
        st = {p["name"]: p for p in c.get("/api/plugins").json()["plugins"]}
        assert st["voice"]["installed"] and not st["voice"]["enabled"]

        assert c.post("/api/plugins/voice/uninstall").status_code == 200
        st = {p["name"]: p for p in c.get("/api/plugins").json()["plugins"]}
        assert st["voice"]["installed"] is False

        assert c.post("/api/plugins/ghost/install").status_code == 404
        assert c.post("/api/plugins/ghost/uninstall").status_code == 404
