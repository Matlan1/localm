# SPDX-License-Identifier: AGPL-3.0-or-later
"""The plugin engine against the real app: changes another process makes are
picked up on the next plugin list (loads and unloads), lifecycle routes run
their blocking work off the event loop, route clashes, a failed load that is
reinstalled, a known plugin that is not installed, a plugin whose required
plugins are missing, and a settings file that cannot be written."""

import asyncio
import json
import logging
import textwrap
import threading

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.engine import PluginManager, attach_engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("LOCALM_URL", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_plugin(root, name, body, toml_extra=""):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


def _ping(name, path=None):
    path = path or f"/api/{name}/ping"
    return f'''
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("{path}")
        def ping():
            return {{"who": "{name}"}}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass
    '''


def _cli(env):
    """A second process's manager (no app), on the same folders."""
    return PluginManager(None, store_root=env / "store", installed_root=env / "installed")


def _server(env, monkeypatch):
    from localm.plugins import engine as eng
    monkeypatch.setattr(eng, "PluginManager", lambda a, **kw: PluginManager(
        a, store_root=env / "store", installed_root=env / "installed"))
    app = FastAPI()
    mgr = attach_engine(app)
    return app, mgr, TestClient(app)


def _entry(client, name):
    return next(p for p in client.get("/api/plugins").json()["plugins"]
                if p["name"] == name)


def _enabled_on_disk(env):
    f = env / "config.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    return set(data.get("plugins_enabled", []))


# --------------------------------------------------------------------------- #
#  Another process disables or uninstalls a plugin the server has loaded      #
# --------------------------------------------------------------------------- #

def test_a_plugin_disabled_by_another_process_is_unloaded_on_the_next_list(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "webx", _ping("webx")))
    assert _entry(c, "webx")["loaded"] is True
    assert c.get("/api/webx/ping").status_code == 200

    _cli(env).set_enabled_state("webx", False)
    entry = _entry(c, "webx")
    assert entry["loaded"] is False
    assert c.get("/api/webx/ping").status_code == 404


def test_a_plugin_uninstalled_by_another_process_is_unloaded_on_the_next_list(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "webx", _ping("webx")))
    assert _entry(c, "webx")["loaded"] is True

    assert _cli(env).uninstall_report("webx").complete is True
    c.get("/api/plugins")
    assert "webx" not in mgr._loaded
    assert c.get("/api/webx/ping").status_code == 404


def test_an_unreadable_config_unloads_nothing(env, monkeypatch):
    """A config that exists but cannot be read is "could not look", not "no
    plugin is enabled": every loaded plugin stays loaded."""
    import localm.config as cfg
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "webx", _ping("webx")))
    assert _entry(c, "webx")["loaded"] is True

    real = cfg.load_config_checked
    monkeypatch.setattr(cfg, "load_config_checked",
                        lambda: (real()[0] | {"plugins_enabled": []}, False))
    assert mgr.reconcile() == ([], [])
    assert "webx" in mgr._loaded
    assert c.get("/api/webx/ping").status_code in (200, 401)


# --------------------------------------------------------------------------- #
#  Lifecycle work runs off the event loop; register() runs on it               #
# --------------------------------------------------------------------------- #

_THREAD_PROBE = '''
    import asyncio
    import threading

    from fastapi import APIRouter

    try:
        asyncio.get_running_loop()
        IMPORTED_ON_LOOP = True
    except RuntimeError:
        IMPORTED_ON_LOOP = False
    IMPORT_THREAD = threading.current_thread().name
    REGISTERED_ON_LOOP = None

    _r = APIRouter()

    @_r.get("/api/probe/where")
    def where():
        return {"imported_on_loop": IMPORTED_ON_LOOP,
                "registered_on_loop": REGISTERED_ON_LOOP}

    def register(host):
        global REGISTERED_ON_LOOP
        try:
            asyncio.get_running_loop()
            REGISTERED_ON_LOOP = True
        except RuntimeError:
            REGISTERED_ON_LOOP = False
        host.mount_router(_r)

    def unregister():
        pass
'''


def test_a_runtime_load_imports_off_the_loop_and_registers_on_it(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "probe", _THREAD_PROBE))
    assert _entry(c, "probe")["loaded"] is True
    assert c.get("/api/probe/where").json() == {
        "imported_on_loop": False, "registered_on_loop": True}

    assert c.post("/api/plugins/probe/disable").status_code == 200
    assert c.post("/api/plugins/probe/enable").status_code == 200
    assert c.get("/api/probe/where").json() == {
        "imported_on_loop": False, "registered_on_loop": True}


def test_on_loop_runs_the_step_on_the_loop_and_reraises_its_error(env):
    """A step that raises TimeoutError is re-raised, not taken for a wait."""
    mgr = PluginManager(None, store_root=env / "store", installed_root=env / "installed")
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    seen, out = [], {}

    def _step():
        seen.append(threading.current_thread() is t)
        raise TimeoutError("from the step")

    def _worker():
        mgr._op_local.loop = loop
        try:
            mgr._on_loop(_step)
        except TimeoutError as e:
            out["error"] = str(e)
    w = threading.Thread(target=_worker, daemon=True)
    w.start()
    w.join(10)
    loop.call_soon_threadsafe(loop.stop)
    t.join(10)
    loop.close()
    assert not w.is_alive()
    assert seen == [True]
    assert out == {"error": "from the step"}


# --------------------------------------------------------------------------- #
#  Route clashes                                                              #
# --------------------------------------------------------------------------- #

def test_a_third_party_route_that_is_already_served_is_refused(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(
        _make_plugin(env / "src", "dupa", _ping("dupa", "/api/shared/{item}")))
    _cli(env).set_installed_from_dir(
        _make_plugin(env / "src", "dupb", _ping("dupb", "/api/shared/{other}")))
    _cli(env).set_installed_from_dir(
        _make_plugin(env / "src", "kern", _ping("kern", "/api/plugins")))
    state = {p["name"]: p for p in c.get("/api/plugins").json()["plugins"]}
    assert state["dupa"]["loaded"] is True
    assert state["dupb"]["loaded"] is False
    assert "already served" in state["dupb"]["error"]
    assert state["kern"]["loaded"] is False
    assert "already served" in state["kern"]["error"]
    assert c.get("/api/shared/x").json() == {"who": "dupa"}


def test_a_startup_load_cannot_shadow_a_management_route(env, monkeypatch):
    _cli(env).set_installed_from_dir(
        _make_plugin(env / "src", "kern", _ping("kern", "/api/plugins")))
    app, mgr, c = _server(env, monkeypatch)
    assert "kern" not in mgr._loaded
    assert "already served" in mgr._errors["kern"]
    assert "plugins" in c.get("/api/plugins").json()


def test_a_route_registered_after_the_startup_load_unloads_the_clashing_plugin(
        env, monkeypatch):
    """The GUI's routes are registered after attach_engine loaded the plugins;
    the lifespan's run_startup_callbacks finds the clash and unloads it."""
    _cli(env).set_installed_from_dir(
        _make_plugin(env / "src", "early", _ping("early", "/api/late/route")))
    app, mgr, c = _server(env, monkeypatch)
    assert "early" in mgr._loaded

    @app.get("/api/late/route")
    def late():
        return {"who": "kernel"}

    mgr.run_startup_callbacks()
    assert "early" not in mgr._loaded
    assert "already served" in mgr._errors["early"]
    assert c.get("/api/late/route").json() == {"who": "kernel"}


def test_a_first_party_route_clash_is_mounted_and_logged(env, monkeypatch, caplog):
    store = env / "store"
    _make_plugin(store, "fpone", _ping("fpone", "/api/plugins"))
    app, mgr, c = _server(env, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        assert c.post("/api/plugins/fpone/install").status_code == 200
    assert "fpone" in mgr._loaded
    assert any("already served" in r.getMessage() and "fpone" in r.getMessage()
               for r in caplog.records)


# --------------------------------------------------------------------------- #
#  A failed load, then a reinstall                                            #
# --------------------------------------------------------------------------- #

_BROKEN = 'def register(host):\n    raise RuntimeError("boom")\n'


def test_uninstall_forgets_the_load_error_and_a_fixed_reinstall_loads(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "brk", _BROKEN))
    assert "boom" in c.get("/api/plugins").json()["errors"]["brk"]

    assert c.post("/api/plugins/brk/uninstall").status_code == 200
    assert "brk" not in c.get("/api/plugins").json()["errors"]

    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "brk", _ping("brk")))
    assert _entry(c, "brk")["loaded"] is True
    assert c.get("/api/brk/ping").json() == {"who": "brk"}


def test_a_reinstall_by_another_process_retries_a_failed_load(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "brk", _BROKEN))
    assert "boom" in c.get("/api/plugins").json()["errors"]["brk"]
    assert _entry(c, "brk")["loaded"] is False

    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "brk", _ping("brk")),
                                     force=True)
    assert _entry(c, "brk")["loaded"] is True
    assert c.get("/api/brk/ping").json() == {"who": "brk"}


# --------------------------------------------------------------------------- #
#  Disable of a name that is not installed                                    #
# --------------------------------------------------------------------------- #

def test_disable_route_refuses_unknown_and_not_installed_names(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    before = _enabled_on_disk(env)
    r = c.post("/api/plugins/nosuch/disable")
    assert r.status_code == 404
    r = c.post("/api/plugins/memory/disable")
    assert r.status_code == 409
    assert "not installed" in r.json()["detail"]
    assert _enabled_on_disk(env) == before


@pytest.mark.parametrize("args, out, code", [
    (["plugin", "uninstall", "memory"], "was not installed", 0),
    (["plugin", "disable", "memory"], "is not installed", 1),
    (["plugin", "config", "memory"], "memory is not installed", 1),
])
def test_cli_names_a_known_plugin_as_not_installed(env, monkeypatch, args, out, code):
    import localm.cli as climod
    monkeypatch.setattr(climod, "_engine_manager", lambda: _cli(env))
    r = CliRunner().invoke(climod.main, args)
    assert r.exit_code == code, r.output
    assert out in " ".join(r.output.split())
    assert "No such plugin" not in r.output


# --------------------------------------------------------------------------- #
#  A required plugin is missing at install time                               #
# --------------------------------------------------------------------------- #

_NEEDY = 'requires = ["nosuchdep"]\n'


@pytest.mark.parametrize("path", ["from_dir", "external", "store_cli", "store_live"])
def test_an_install_with_a_missing_required_plugin_is_left_disabled(env, path):
    store = env / "store"
    _make_plugin(store, "needy", _ping("needy"), _NEEDY)
    src = _make_plugin(env / "src", "needx", _ping("needx"), _NEEDY)
    app = FastAPI()
    mgr = PluginManager(app, store_root=store, installed_root=env / "installed")
    if path == "from_dir":
        name = mgr.set_installed_from_dir(src).name
    elif path == "external":
        name = mgr.install_external(src).name
    elif path == "store_cli":
        name = "needy"
        mgr.set_installed_state(name, True)
    else:
        name = "needy"
        assert mgr.install(name) is False
    assert mgr.is_installed(name)
    assert not mgr.is_enabled(name)
    assert name not in mgr._loaded


def test_a_forced_reinstall_that_now_misses_a_required_plugin_is_disabled(env):
    app = FastAPI()
    mgr = PluginManager(app, store_root=env / "store", installed_root=env / "installed")
    mgr.install_external(_make_plugin(env / "src", "grow", _ping("grow")))
    assert mgr.is_enabled("grow") and "grow" in mgr._loaded

    mgr.install_external(_make_plugin(env / "src", "grow", _ping("grow"), _NEEDY),
                         force=True)
    assert not mgr.is_enabled("grow")
    assert "grow" not in mgr._loaded


def test_mcp_list_plugins_names_a_plugin_that_cannot_run_here(env, monkeypatch):
    from localm.plugins import catalog
    from localm.plugins.mcpserver.tools import plugin_admin
    monkeypatch.setattr(catalog, "_nvidia_linux", lambda: False)
    reply = plugin_admin.build()["list_plugins"]["handler"]({})
    text = " ".join(c["text"] for c in reply["content"])
    line = next(ln for ln in text.splitlines() if ln.startswith("exllama"))
    assert "which this machine does not have" in line


def _plugin_events(env):
    f = env / "logs" / "management.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines() if f.is_file() else []
    return [(e["kind"], e["plugin"]) for e in map(json.loads, lines)
            if e["kind"].startswith("plugin_")]


def test_the_terminal_install_paths_record_one_event_each(env, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "full")
    _make_plugin(env / "store", "storep", _ping("storep"))
    src = _make_plugin(env / "src", "dirp", _ping("dirp"))
    mgr = _cli(env)
    mgr.set_installed_state("storep", True)
    mgr.set_installed_state("storep", True)
    mgr.set_installed_from_dir(src)
    mgr.set_installed_state("storep", False)
    assert _plugin_events(env) == [("plugin_installed", "storep"),
                                   ("plugin_installed", "dirp"),
                                   ("plugin_uninstalled", "storep")]


def test_install_route_reports_a_plugin_left_disabled(env, monkeypatch):
    _make_plugin(env / "store", "needy", _ping("needy"), _NEEDY)
    app, mgr, c = _server(env, monkeypatch)
    r = c.post("/api/plugins/needy/install")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False and "nosuchdep" in body["detail"]
    assert c.get("/api/needy/ping").status_code == 404


# --------------------------------------------------------------------------- #
#  The settings file cannot be written                                        #
# --------------------------------------------------------------------------- #

def _config_write_fails(monkeypatch, unreadable=False):
    import localm.config as cfg
    from localm.storekit import ConfigUnreadable

    def _fail(mutator):
        if unreadable:
            raise ConfigUnreadable("config.json exists but could not be read")
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(cfg, "update_config", _fail)


@pytest.mark.parametrize("unreadable", [False, True])
def test_enable_unloads_again_when_the_flag_cannot_be_saved(env, monkeypatch, unreadable):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "webx", _ping("webx")),
                                     enable=False)
    _config_write_fails(monkeypatch, unreadable)
    r = c.post("/api/plugins/webx/enable")
    assert r.status_code == 500
    assert "settings file" in r.json()["detail"]
    assert "webx" not in mgr._loaded
    assert c.get("/api/webx/ping").status_code == 404


def test_disable_keeps_the_plugin_when_the_flag_cannot_be_saved(env, monkeypatch):
    app, mgr, c = _server(env, monkeypatch)
    _cli(env).set_installed_from_dir(_make_plugin(env / "src", "webx", _ping("webx")))
    assert _entry(c, "webx")["loaded"] is True
    _config_write_fails(monkeypatch)
    r = c.post("/api/plugins/webx/disable")
    assert r.status_code == 500
    assert "settings file" in r.json()["detail"]
    assert "webx" in mgr._loaded
    assert c.get("/api/webx/ping").status_code == 200


# --------------------------------------------------------------------------- #
#  The first-use record                                                       #
# --------------------------------------------------------------------------- #

_FIRST_USE = _ping("fu") + '''
        CALLS = []

        def on_first_use():
            CALLS.append(1)
'''


def test_a_first_use_record_that_cannot_be_saved_is_logged(env, monkeypatch, caplog):
    _make_plugin(env / "installed", "fu", _FIRST_USE)
    mgr = PluginManager(FastAPI(), store_root=env / "store", installed_root=env / "installed")
    mgr.discover()
    _config_write_fails(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        mgr._load(mgr._specs["fu"])
    assert mgr._loaded["fu"][1].CALLS == [1]
    assert any("could not record that its first-use hook ran" in r.getMessage()
               for r in caplog.records)


def test_a_first_use_record_that_cannot_be_read_is_logged(env, monkeypatch, caplog):
    import localm.config as cfg
    _make_plugin(env / "installed", "fu", _FIRST_USE)
    mgr = PluginManager(FastAPI(), store_root=env / "store", installed_root=env / "installed")
    mgr.discover()

    def _fail():
        raise OSError("unreadable")
    monkeypatch.setattr(cfg, "load_config", _fail)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        mgr._load(mgr._specs["fu"])
    assert mgr._loaded["fu"][1].CALLS == []
    assert any("could not read whether its first-use hook" in r.getMessage()
               for r in caplog.records)


# --------------------------------------------------------------------------- #
#  A first-party plugin this machine cannot run                               #
# --------------------------------------------------------------------------- #

def test_cli_status_and_install_name_a_plugin_that_cannot_run_here(env, monkeypatch):
    import localm.cli as climod
    from localm.plugins import catalog
    monkeypatch.setattr(catalog, "_nvidia_linux", lambda: False)
    monkeypatch.setattr(climod, "_engine_manager",
                        lambda: PluginManager(None, installed_root=env / "installed"))
    r = CliRunner().invoke(climod.main, ["plugin", "status"])
    assert r.exit_code == 0, r.output
    flat = " ".join(r.output.split())
    assert "exllama" in flat and "which this machine does not have" in flat

    r = CliRunner().invoke(climod.main, ["plugin", "install", "--no-deps", "exllama"])
    assert r.exit_code == 0, r.output
    assert "so it will not run models here" in " ".join(r.output.split())
