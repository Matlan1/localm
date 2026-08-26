# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin engine: discovery, runtime load/unload (route mount/unmount),
enable/disable persistence, protected plugins, and failure isolation.

Uses a synthetic plugin written to a temp dir; runs open-mode (no API key) so
the mounted, auto-scoped routes are reachable.
"""

import os
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


def test_plugin_config_is_confined_to_own_block(env):
    """Plugin isolation: a plugin's Host config r/w is CONFINED to its own block.
    It cannot read or tamper with another plugin's persisted config, even though
    plugins are trusted at install time."""
    from unittest.mock import MagicMock

    from localm.config import load_config, save_config
    from localm.plugins.engine import PluginHost

    c = load_config()
    c.setdefault("plugins", {})
    c["plugins"]["alpha"] = {"secret": "alpha-owns-this"}
    c["plugins"]["beta"] = {"secret": "beta-owns-this"}
    save_config(c)

    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)

    # Own config is readable, with or without the explicit own name.
    assert host.plugin_config()["secret"] == "alpha-owns-this"
    assert host.plugin_config("alpha")["secret"] == "alpha-owns-this"

    # A cross-plugin READ is confined to self, never beta's block.
    assert host.plugin_config("beta") == {"secret": "alpha-owns-this"}

    # A cross-plugin WRITE lands in alpha's own block; beta stays untouched.
    host.save_plugin_config("beta", {"secret": "attacker"})
    plugins = load_config()["plugins"]
    assert plugins["beta"]["secret"] == "beta-owns-this"     # NOT tampered
    assert plugins["alpha"]["secret"] == "attacker"          # own block updated

    # An own-config save (no name) works.
    host.save_plugin_config(cfg={"secret": "self-set"})
    assert load_config()["plugins"]["alpha"]["secret"] == "self-set"


def test_register_chat_hook_is_traceable(env):
    """A chat hook sees and transforms every chat turn, an otherwise-invisible
    capability. Its registration is SURFACED (which plugin hooked which phase) so
    an unexpected or compromised plugin hook is discoverable, not silent."""
    from unittest.mock import MagicMock

    from localm.plugins.engine import PluginHost

    app = MagicMock()
    pipeline = MagicMock()
    app.state.chat_pipeline = pipeline
    spec = MagicMock()
    spec.name = "sneaky"
    spec.surface = None
    host = PluginHost(app, MagicMock(), spec)
    host.audit = MagicMock()   # spy on the traceability surface

    def _hook(x, ctx):
        return x

    host.register_chat_hook("inlet", _hook, priority=3)

    # The hook was registered on the pipeline, tagged with the owning plugin...
    pipeline.add_hook.assert_called_once_with("inlet", _hook, priority=3, plugin="sneaky")
    # ...AND the registration is surfaced for traceability (not silent).
    host.audit.assert_called_once_with(
        "chat_hook_registered", {"phase": "inlet", "priority": 3})


def test_register_chat_hook_without_pipeline_is_inert(env):
    """No chat pipeline (a bare-FastAPI test harness): the hook is inert and the
    skip is surfaced, never a crash."""
    from unittest.mock import MagicMock

    from localm.plugins.engine import PluginHost

    app = MagicMock()
    app.state = MagicMock(spec=[])   # no chat_pipeline attribute
    spec = MagicMock()
    spec.name = "p"
    spec.surface = None
    host = PluginHost(app, MagicMock(), spec)
    host.audit = MagicMock()

    host.register_chat_hook("outlet", lambda x, ctx: x)
    host.audit.assert_called_once_with("chat_hook_skipped", {"phase": "outlet"})


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
    # a clean manifest produces no unknown-key warning
    assert "alpha" not in mgr._discover_errors


def test_unknown_manifest_key_warned_but_plugin_loads(env):
    """A misspelled or unknown key in [plugin] or [surface] must be surfaced (it
    means a tab, client module, or flag quietly never materialises) but must NOT
    block the plugin - surface, do not escalate."""
    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    _make_plugin(plugins, "alpha", _ping("alpha"),
                 toml_extra='default_enabld = true\n[surface]\ntab_idd = "alpha"\n')
    app = FastAPI()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    specs = mgr.discover()
    assert "alpha" in specs                       # still parsed, not refused
    warn = mgr._discover_errors.get("alpha", "")
    assert warn.startswith("warning:")
    assert "unknown manifest key" in warn
    assert "[plugin] default_enabld" in warn and "[surface] tab_idd" in warn
    mgr.enable("alpha")                           # and it genuinely loads
    with TestClient(app) as c:
        assert c.get("/api/alpha/ping").json() == {"pong": True}


def test_host_scope_methods_fail_loudly(env):
    """The host has no request context, so has_scope/require_scope cannot check
    anything. They must raise instead of silently allowing - a plugin guard built
    on a silent no-op can never fire."""
    from unittest.mock import MagicMock

    from localm.plugins.engine import PluginHost

    spec = MagicMock()
    spec.name = "alpha"
    host = PluginHost(MagicMock(), MagicMock(), spec)
    with pytest.raises(NotImplementedError):
        host.has_scope("alpha")
    with pytest.raises(NotImplementedError):
        host.require_scope("alpha")


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


def test_hook_failure_is_reported_and_does_not_block_install(env, caplog):
    """A lifecycle hook that RAISES stays best-effort (the install completes),
    but it must not be SILENT: the plugin, the hook name and the underlying
    error are logged.

    WARNING is the load-bearing part of the assertion, not an incidental level:
    the always-on recent-activity ring a bug report dumps is INFO+
    (_RingBufferHandler in localm/debuglog.py), so a debug-level line would
    reach no bug report and the failure would still be effectively hidden."""
    import logging

    from localm.plugins.engine import PluginManager
    plugins = env / "plugins"
    body = '''
        from pathlib import Path
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/boom/ping")
        def ping():
            return {"pong": True}

        def register(host):
            host.mount_router(_r)

        def unregister():
            pass

        def on_install():
            # Marker BEFORE the raise: a hook that was never called at all looks
            # exactly like one whose failure was correctly reported, so the test
            # has to prove the fault actually fired before reading the log.
            (Path(__file__).parent / "on_install_ran").write_text("yes")
            raise RuntimeError("hook exploded")
    '''
    _make_plugin(plugins, "boom", body)
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.discover()
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        mgr.install("boom")            # must NOT propagate the hook's exception

    # 1. The fault fired, and the best-effort contract survived the fix.
    assert (plugins / "boom" / "on_install_ran").exists()
    assert mgr.is_installed("boom") and mgr.is_enabled("boom")

    # 2. It was reported: plugin, hook name and the underlying error, from
    #    OUTSIDE the call (the handler catches Exception broadly, so an
    #    assertion raised inside the hook would be absorbed as an input and the
    #    test would pass in both directions).
    warned = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.WARNING]
    assert any("boom" in m and "on_install" in m and "hook exploded" in m
               for m in warned), warned


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


def test_uninstall_reports_degraded_result_when_rmtree_fails(env, caplog, monkeypatch):
    """A locked or permission-denied installed dir must not be silently
    swallowed. An AV hold or a still-open file handle on Windows leaves
    shutil.rmtree raising OSError while the directory (and its code and data)
    stays on disk - the caller is told, both via a WARNING log and via the
    return value, rather than getting an unconditional ``was_installed``."""
    import logging
    import shutil
    from pathlib import Path

    from localm.config import load_config
    from localm.plugins.engine import PluginManager

    plugins = env / "plugins"
    _make_plugin(plugins, "p1", _ping("p1"))
    mgr = PluginManager(FastAPI(), external_root=plugins, builtin_root=None)
    mgr.install("p1")
    installed_dir = Path(mgr._installed_root) / "p1"
    assert installed_dir.is_dir()

    real_rmtree = shutil.rmtree

    def _boom(path, *a, **k):
        if Path(path) == installed_dir:
            raise OSError(13, "Permission denied", str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", _boom)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        result = mgr.uninstall("p1")

    # The directory genuinely could not be removed: prove it, don't assume it.
    assert installed_dir.is_dir()
    # A bare "was it installed before" answer of True would be a false success
    # report; the outcome must reflect that the removal did not complete.
    assert result is not True
    assert any("p1" in rec.message and "Permission denied" in rec.message
                for rec in caplog.records), (
        "expected a WARNING naming the plugin and the real OSError; got: "
        f"{[rec.message for rec in caplog.records]}")

    # Config state is still updated (disabled/unloaded) even though the files
    # remain - the degraded report is about the directory, not a full no-op.
    cfg = load_config()
    assert "p1" not in cfg.get("plugins_enabled", [])


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
    # A refused delete is a FAILURE, not a silent no-op: the caller (uninstall())
    # must be able to tell "nothing needed deleting" apart from "deletion was
    # refused for safety".
    assert mgr._delete_plugin_data(mgr._specs["evil"]) is False
    assert victim.exists() and (victim / "keep.txt").exists()


def test_delete_data_rejects_dot_root(tmp_path, monkeypatch):
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, ".")
    (home / "models").mkdir()
    (home / "models" / "big.gguf").write_text("data", encoding="utf-8")
    assert mgr._delete_plugin_data(mgr._specs["evil"]) is False
    assert home.exists() and (home / "models" / "big.gguf").exists()


def test_delete_data_deletes_legit_subdir(tmp_path, monkeypatch):
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, "evil_data")
    target = home / "evil_data"
    target.mkdir()
    (target / "cache.bin").write_text("x", encoding="utf-8")
    assert mgr._delete_plugin_data(mgr._specs["evil"]) is True
    assert not target.exists()


def test_delete_data_reports_failure_when_rmtree_fails(tmp_path, monkeypatch):
    """A locked or permission-denied data directory must not be silently
    swallowed. An ``except OSError: pass`` that returns None either way leaves a
    caller with no way to tell a real failure from success."""
    import shutil
    from pathlib import Path
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, "evil_data")
    target = home / "evil_data"
    target.mkdir()
    (target / "cache.bin").write_text("x", encoding="utf-8")

    real_rmtree = shutil.rmtree

    def _boom(path, *a, **k):
        if Path(path) == target:
            raise OSError(13, "Permission denied", str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", _boom)
    assert mgr._delete_plugin_data(mgr._specs["evil"]) is False
    assert target.is_dir() and (target / "cache.bin").exists()


def test_uninstall_delete_data_reports_degraded_result_on_failure(tmp_path, monkeypatch, caplog):
    """uninstall(delete_data=True) folds a failed data deletion into its
    returned bool, not just the installed-dir removal, so a locked data
    directory cannot report a bare True success while the data stays on disk."""
    import logging
    import shutil
    from pathlib import Path

    from localm.config import load_config

    # _mgr_with_data_subdir's external_root IS the installed/discovery dir, so
    # "evil" is already physically installed - no extra provisioning needed.
    mgr, home = _mgr_with_data_subdir(tmp_path, monkeypatch, "evil_data")
    installed_dir = home / "plugins" / "evil"
    assert installed_dir.is_dir()

    target = home / "evil_data"
    target.mkdir()
    (target / "cache.bin").write_text("x", encoding="utf-8")

    real_rmtree = shutil.rmtree

    def _boom(path, *a, **k):
        if Path(path) == target:
            raise OSError(13, "Permission denied", str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", _boom)
    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        result = mgr.uninstall("evil", delete_data=True)

    # The data genuinely was not removed: prove it, don't assume it.
    assert target.is_dir() and (target / "cache.bin").exists()
    assert result is not True, \
        "uninstall() reported success while requested data survived on disk"
    assert any("evil" in rec.message and "Permission denied" in rec.message
                for rec in caplog.records), (
        "expected a WARNING naming the plugin and the real OSError; got: "
        f"{[rec.message for rec in caplog.records]}")
    # The installed directory itself (unaffected by the data-dir failure) is
    # still gone, and config state is still updated - a degraded report is
    # about the DATA, not a full no-op.
    assert not installed_dir.exists()
    cfg = load_config()
    assert "evil" not in cfg.get("plugins_enabled", [])


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


def _flaky_then_ok(name, marker_path):
    """A plugin whose register() mounts a route AND a chat hook, then raises -
    but only the FIRST time (a marker file flips it to a clean success on any
    later attempt, modelling a retry after the first load failed)."""
    return textwrap.dedent(f'''
        import os
        from fastapi import APIRouter
        _r = APIRouter()

        @_r.get("/api/{name}/ping")
        def ping():
            return {{"pong": True}}

        def register(host):
            host.mount_router(_r)
            host.register_chat_hook("inlet", lambda msgs, ctx: msgs)
            marker = {marker_path!r}
            if not os.path.exists(marker):
                open(marker, "w").close()
                raise RuntimeError("boom after partial registration")

        def unregister():
            pass
    ''')


def test_load_failure_after_partial_register_leaves_nothing_live(env, tmp_path):
    """A plugin whose register() mounts a route and a chat hook, then raises,
    must leave NEITHER live: the engine reporting "not loaded" must be true,
    not just claimed. _load() calls host.unmount() before propagating the
    exception, so no route stays reachable and no chat hook keeps firing while
    self._loaded holds no entry. A retry (enable() called again) must not stack
    a second copy on top of the first failed attempt's remnants either."""
    from localm.inference.chat_pipeline import ChatPipeline
    from localm.plugins.engine import PluginManager

    plugins = env / "plugins"
    marker = tmp_path / "flaky.attempted"
    _make_plugin(plugins, "flaky", _flaky_then_ok("flaky", str(marker)))
    app = FastAPI()
    app.state.chat_pipeline = ChatPipeline()
    mgr = PluginManager(app, external_root=plugins, builtin_root=None)
    mgr.discover()

    # include_router() on this FastAPI/Starlette version stores an opaque
    # wrapper object per mount rather than flattening child routes onto
    # app.router.routes, so a mounted route cannot be identified by a `.path`
    # attribute. The route COUNT (relative to the pre-load baseline) is the
    # real, version-independent signal for "how many mounts are attached" -
    # actual reachability is proven separately via TestClient below.
    baseline_routes = len(app.router.routes)

    with pytest.raises(RuntimeError, match="boom after partial registration"):
        mgr._load(mgr._specs["flaky"])
    assert "flaky" not in mgr._loaded
    assert len(app.router.routes) == baseline_routes, \
        "a failed load must not leave its route mount behind"
    assert not app.state.chat_pipeline.has("inlet"), \
        "a failed load must not leave its chat hook firing on every turn"
    with TestClient(app) as c:
        assert c.get("/api/flaky/ping").status_code == 404

    # Retry (e.g. the user clicks enable again): must load clean, not stack a
    # second route/hook on top of the first failed attempt's remnants.
    mgr._load(mgr._specs["flaky"])
    assert "flaky" in mgr._loaded
    assert len(app.router.routes) == baseline_routes + 1, \
        "retry produced a duplicate route mount left over from the failed first attempt"
    assert app.state.chat_pipeline.has("inlet")
    with TestClient(app) as c:
        assert c.get("/api/flaky/ping").status_code == 200


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
    behind - install loads first and only persists on success.

    The plugin here is ALREADY on disk in the installed root, so install() never
    copied it, and its directory is left alone: install must never delete a
    directory it did not create. This test therefore pins the CONFIG rollback
    only. The copy-we-did-create case is
    test_install_rolls_back_a_copy_it_created."""
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
    assert (plugins / "broken").is_dir()      # pre-existing: not ours to delete


def test_install_rolls_back_a_copy_it_created(env):
    """FIRES-CONTROL for the "never roll back a directory we did not create"
    guard: when install() DID copy from the store and the load then fails, the
    copy is still removed. Without this the guard could silently degrade into
    "never delete anything", which would leave a broken half-install behind."""
    from localm.plugins.engine import PluginManager
    store = env / "store"
    inst = env / "installed"
    _make_plugin(store, "badp",
                 'def register(host):\n    raise RuntimeError("boom")\n'
                 'def unregister():\n    pass\n')
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    with pytest.raises(RuntimeError):
        mgr.install("badp")
    assert not (inst / "badp").exists()       # OUR copy was rolled back
    assert (store / "badp").is_dir()          # the store source is untouched


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


def test_uninstall_http_reports_failure_when_removal_incomplete(env, monkeypatch):
    """The route must not report {"status": "uninstalled"} when uninstall()
    itself reports a degraded (False) result - a locked file, an AV hold or a
    permission denial that leaves the plugin's files on disk must not reply 200
    "uninstalled"."""
    import shutil
    from pathlib import Path

    from localm.plugins.engine import attach_engine

    app = FastAPI()
    manager = attach_engine(app)              # default roots = the real builtins
    with TestClient(app) as c:
        assert c.post("/api/plugins/voice/install").status_code == 200

    installed_dir = Path(manager._installed_root) / "voice"
    assert installed_dir.is_dir()

    real_rmtree = shutil.rmtree

    def _boom(path, *a, **k):
        if Path(path) == installed_dir:
            raise OSError(13, "Permission denied", str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", _boom)
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/uninstall")
    assert r.status_code != 200, \
        "the route reported success while the installed directory could not be removed"
    # The removal genuinely did not complete: prove it, don't assume it.
    assert installed_dir.is_dir()


def _make_legacy_plugin(root, name, *, exports='["tool_hello"]'):
    """A THIRD-PARTY legacy plugin source dir: the `entry = ` + `[tools] exports`
    shape the GUI's External plugins card exists for (a CLI command plus coder
    tool exports), as opposed to an engine `register = ` plugin."""
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nversion = "1.2.3"\n'
        f'description = "an external one"\nentry = "{name}_cli:main"\n'
        f'\n[tools]\nexports = {exports}\n', encoding="utf-8")
    (pdir / f"{name}_cli.py").write_text("def main():\n    pass\n", encoding="utf-8")
    return pdir


def test_external_plugin_http_lifecycle(env):
    """The GUI's External plugins card has a live API: install a third-party
    folder over HTTP, see it listed as an installed non-builtin with its tool
    exports, then remove it."""
    from localm.plugins.engine import attach_engine
    src = _make_legacy_plugin(env / "src", "myext")
    app = FastAPI()
    attach_engine(app)
    with TestClient(app) as c:
        def _entry():
            return next((p for p in c.get("/api/plugins").json()["plugins"]
                         if p["name"] == "myext"), None)

        assert _entry() is None                      # not installed yet

        r = c.post("/api/plugins/install-external", json={"source": str(src)})
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "installed", "name": "myext", "version": "1.2.3"}

        e = _entry()
        assert e is not None, "an installed external plugin must be listed"
        assert e["installed"] is True
        assert e["builtin"] is False                  # this is what the card filters on
        assert e["version"] == "1.2.3"
        assert e["description"] == "an external one"
        assert e["tool_exports"] == ["tool_hello"]    # the card's "Tools" column

        assert c.post("/api/plugins/myext/uninstall").status_code == 200
        assert _entry() is None


def test_install_external_rejects_bad_input(env):
    """Negative cases: the route must refuse rather than half-install."""
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    with TestClient(app) as c:
        assert c.post("/api/plugins/install-external", json={}).status_code == 400
        assert c.post("/api/plugins/install-external",
                      json={"source": ""}).status_code == 400
        # a real directory with no plugin.toml
        (env / "empty").mkdir()
        r = c.post("/api/plugins/install-external", json={"source": str(env / "empty")})
        assert r.status_code == 400
        assert "plugin.toml" in r.json()["detail"]
        # a path that does not exist at all
        assert c.post("/api/plugins/install-external",
                      json={"source": str(env / "nope")}).status_code == 400
        # installing the same plugin twice without force is a 400, not a silent
        # overwrite of whatever is already there.
        src = _make_legacy_plugin(env / "src", "dupext")
        assert c.post("/api/plugins/install-external",
                      json={"source": str(src)}).status_code == 200
        r = c.post("/api/plugins/install-external", json={"source": str(src)})
        assert r.status_code == 400
        assert "already installed" in r.json()["detail"]


# ---------------------------------------------------------------------------
#  Plugin id confinement.
#
#  The id arrives unvalidated from the CLI, the MCP tools and the HTTP {name}
#  path param, and _installed_dir()/_store_dir() join it onto a root. Traversal
#  reaches shutil.rmtree (install rollback), the refresh staging rename/rmtree
#  dance, and a marker WRITE outside the plugins root.
# ---------------------------------------------------------------------------

def _traversal_roots(env):
    """store/, installed/ and a SENTINEL dir that `installed/../outside`
    resolves to, so an escape is observable as damage to the sentinel.

    PAYLOAD SAFETY (binding for every test below that reaches rmtree/copytree):
    the escape payloads stay RELATIVE ('../outside') and every root derives from
    tmp_path, so the blast radius is inside the fixture. A run with the guard
    removed executes the unguarded code for real, so a payload naming a real
    location would delete it. tmp_path is itself absolute and drive-qualified on
    Windows, so it exercises the identical "an absolute component REPLACES the
    base" escape with no risk."""
    base = env / "roots"
    store = base / "store"
    store.mkdir(parents=True)
    installed = base / "installed"
    installed.mkdir(parents=True)
    sentinel = _make_plugin(base, "outside", _ping("outside"))
    (sentinel / "keep.txt").write_text("important", encoding="utf-8")
    return store, installed, sentinel


def _sentinel_intact(sentinel):
    return (sentinel.is_dir()
            and (sentinel / "keep.txt").read_text(encoding="utf-8") == "important"
            and (sentinel / "plugin.toml").is_file())


def test_is_valid_plugin_name_rule():
    """The id rule is EXACTLY parse_spec's manifest-name rule, so the name that
    becomes a directory and the name inside the manifest cannot drift."""
    from localm.plugins.engine import _is_valid_plugin_name as ok

    for good in ("chat", "coder", "my-plugin", "_x", "a1"):
        assert ok(good), good
    # PAYLOAD SAFETY: never name a REAL location in a test payload, even here.
    # _is_valid_plugin_name is a pure string predicate and touches no filesystem,
    # so these cannot delete anything, but a NEGATIVE pass runs the UNSAFE code
    # for real. Drive-qualified and absolute vectors use an unmounted letter and
    # a synthetic root; they parse identically.
    for bad in ("", ".", "..", "../outside", "..\\outside", "a/b", "a\\b",
                "Q:/nonexistent", "/nonexistent-root", "my plugin", "1abc",
                ".hidden", "x.y", None, 7):
        assert not ok(bad), bad


@pytest.mark.parametrize("evil", ["../outside", "..\\outside"])
def test_install_with_traversing_id_raises_and_deletes_nothing(env, evil):
    """install() with a traversing id must be refused and delete nothing:
    unguarded, provision early-returns on the traversed dir's own plugin.toml
    and the verify step's rollback then rmtree's it, recursively deleting a
    sibling of the plugins root. Both separator forms must be refused; on POSIX
    only the forward-slash form actually traverses, but the id is illegal - and
    therefore refused - on every platform."""
    from localm.plugins.engine import PluginManager
    store, installed, sentinel = _traversal_roots(env)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    with pytest.raises(ValueError, match="invalid plugin name"):
        mgr.install(evil)
    assert _sentinel_intact(sentinel)
    assert not any(installed.iterdir())          # and nothing landed in the root


@pytest.mark.parametrize("evil", ["../outside", "..\\outside"])
def test_refresh_with_traversing_id_raises_and_writes_nothing(env, evil):
    """The refresh half (CodeQL 32-43): the same unvalidated id drives the
    staging swap and, before that, a provenance-marker WRITE at the traversed
    destination - a file created outside the plugins root."""
    from localm.plugins.engine import PluginManager
    store, installed, sentinel = _traversal_roots(env)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    with pytest.raises(KeyError):
        mgr.refresh(evil)
    assert _sentinel_intact(sentinel)
    assert not (sentinel / ".localm-source.json").exists()   # no marker escaped
    assert sorted(p.name for p in sentinel.iterdir()) == [
        "keep.txt", "plug.py", "plugin.toml"]                # and nothing staged


def test_uninstall_and_enable_with_traversing_id_are_refused(env):
    """uninstall()/enable() gate on _installed_set()/_specs membership, which
    only ever holds single-component basenames - they are not the traversal
    sink, and this test is not a claim that they were. It pins that they keep
    refusing cleanly rather than growing a path join.

    enable() answers a traversing id with KeyError, not
    ValueError('...is not installed; install it first'): _store_dir returns None
    for an illegal id, and an illegal id can never be installed, so the install
    hint would be misleading. Both are clean failures for CLI and HTTP alike."""
    from localm.plugins.engine import PluginManager
    store, installed, sentinel = _traversal_roots(env)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    for evil in ("../outside", "..\\outside"):
        with pytest.raises(KeyError):
            mgr.uninstall(evil)
        with pytest.raises(KeyError):
            mgr.enable(evil)
    assert _sentinel_intact(sentinel)


def test_swap_in_store_copy_refuses_staging_outside_the_installed_root(env):
    """_swap_in_store_copy interpolates the name into two SIBLING basenames
    ('.<name>.refresh.tmp'/'.bak') that drive copytree, two renames and four
    rmtree calls. Today its dest always comes from the validated _installed_dir;
    the guard exists so a future caller passing a raw name cannot silently
    reinstate the escape."""
    from localm.plugins.engine import PluginManager
    store, installed, sentinel = _traversal_roots(env)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)

    with pytest.raises(ValueError, match="staging"):
        mgr._swap_in_store_copy("../outside", sentinel,
                                installed / ".." / "outside", "deadbeef")
    assert _sentinel_intact(sentinel)


def test_http_plugin_routes_404_a_traversing_name(env, monkeypatch):
    """uvicorn unquotes before routing and starlette's path-param regex is
    [^/]+, so '..%5Cx' is delivered to the handler as the ONE segment '..\\x'.
    The route must 404 it (not 400 it, and not act on it) - the handlers catch
    broad Exception, so the guard has to run before their try block.

    On POSIX that id cannot traverse (a backslash is an ordinary character), so
    the sentinel is only file-system proof on Windows. The installed root is
    therefore ALSO asserted unchanged, which is the POSIX-meaningful half: a
    future change that hoisted a mkdir(parents=True) above the id check would
    create '..\\x' there as a literal single-component directory and this test
    would otherwise still pass."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    # attach_engine uses the default roots: installed_root = <home>/plugins,
    # so <installed_root>/../x is <home>/x.
    sentinel = _make_plugin(env, "x", _ping("x"))
    (sentinel / "keep.txt").write_text("important", encoding="utf-8")

    app = FastAPI()
    attach_engine(app)
    installed_root = env / "plugins"
    before = sorted(p.name for p in installed_root.iterdir())
    with TestClient(app) as c:
        for verb in ("install", "uninstall", "refresh", "enable", "disable"):
            r = c.post(f"/api/plugins/..%5Cx/{verb}")
            assert r.status_code == 404, (verb, r.status_code, r.text)
        # a name that is not a path at all but still not a legal id
        assert c.post("/api/plugins/not%20an%20id/install").status_code == 404
    assert _sentinel_intact(sentinel)
    # nothing was created under the installed root either
    assert sorted(p.name for p in installed_root.iterdir()) == before


# ---------------------------------------------------------------------------
#  Third-party source trees are untrusted.
#
#  shutil.copytree defaults to symlinks=False, which DEREFERENCES links: a
#  plugin shipping 'web/notes.txt -> ~/.ssh/id_rsa' would have that file's
#  CONTENTS copied into a directory localm serves from a StaticFiles mount,
#  before mount_static's own resolve()-based escape guard could see it.
# ---------------------------------------------------------------------------

def _symlink_or_skip(link, target, *, dir_link=False):
    try:
        os.symlink(target, link, target_is_directory=dir_link)
    except (OSError, NotImplementedError) as e:   # locked-down Windows host
        pytest.skip(f"cannot create symlinks here: {e}")


def test_install_external_rejects_a_link_escaping_the_source(env):
    from localm.plugins.engine import PluginManager
    src = _make_plugin(env / "src", "linky", _ping("linky"))
    (src / "web").mkdir()
    secret = env / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    _symlink_or_skip(src / "web" / "link", secret)

    installed = env / "installed"
    mgr = PluginManager(None, store_root=env / "store", installed_root=installed)
    with pytest.raises(ValueError, match="points outside"):
        mgr.set_installed_from_dir(src)

    leaked = installed / "linky" / "web" / "link"
    assert not (leaked.is_file() and not leaked.is_symlink()), \
        "the link target's contents were flattened into the installed tree"
    assert not (installed / "linky").exists()


def test_install_external_rejects_an_internal_link_too(env):
    """An INTERNAL link is refused as well, and this is not belt-and-braces.

    An escape-only rule leaves two holes. (1) An ABSOLUTE link whose target sits
    inside the source resolves inside it, so it passes an escape check and is
    then copied verbatim - the installed plugin keeps pointing at the operator's
    source directory and is not self-contained. (2) A directory link back into
    the tree is the cycle case (see the junction test below). Plugin sources
    must be plain files."""
    from localm.plugins.engine import PluginManager
    src = _make_plugin(env / "src", "inner", _ping("inner"))
    (src / "web").mkdir()
    (src / "web" / "real.txt").write_text("payload", encoding="utf-8")
    # absolute, and pointing INSIDE the source - passes any escape-only check
    _symlink_or_skip(src / "web" / "alias.txt", (src / "web" / "real.txt").resolve())

    installed = env / "installed"
    mgr = PluginManager(None, store_root=env / "store", installed_root=installed)
    with pytest.raises(ValueError, match="plain files"):
        mgr.set_installed_from_dir(src)
    assert not (installed / "inner").exists()


def test_install_external_rejects_a_directory_link_cycle(env):
    """A directory link back into the source is the unbounded-copy case, and it
    must be refused by the WALK - not delegated to copytree(symlinks=True)."""
    from localm.plugins.engine import PluginManager
    src = _make_plugin(env / "src", "cyclic", _ping("cyclic"))
    (src / "web").mkdir()
    _symlink_or_skip(src / "web" / "loop", src, dir_link=True)

    installed = env / "installed"
    mgr = PluginManager(None, store_root=env / "store", installed_root=installed)
    with pytest.raises(ValueError, match="plain files"):
        mgr.set_installed_from_dir(src)
    assert not (installed / "cyclic").exists()


@pytest.mark.skipif(os.name != "nt",
                    reason="directory junctions are a Windows-only construct")
def test_install_external_rejects_a_windows_junction_cycle(env):
    """REGRESSION GUARD. The rule is 'any link', not 'any ESCAPING link'.

    shutil.copytree DEMOTES a directory junction to a non-symlink and recurses
    into it (stdlib shutil.py: "Special check for directory junctions, which
    appear as symlinks but we want to recurse"), so symlinks=True neither
    preserves a junction nor bounds a junction cycle: an escape-only rule copies
    nested levels until it fails on path length. A junction also needs NO
    elevation to create, unlike os.symlink, and is_symlink() reports False for
    it, so the walk tests the reparse-point attribute instead."""
    import _winapi
    from localm.plugins.engine import PluginManager
    src = _make_plugin(env / "src", "juncy", _ping("juncy"))
    (src / "web").mkdir()
    _winapi.CreateJunction(str(src), str(src / "web" / "loop"))
    # precondition: this is exactly the shape islink()-based checks miss
    assert not (src / "web" / "loop").is_symlink()

    installed = env / "installed"
    mgr = PluginManager(None, store_root=env / "store", installed_root=installed)
    with pytest.raises(ValueError, match="plain files"):
        mgr.set_installed_from_dir(src)
    assert not (installed / "juncy").exists()


def test_failed_install_never_deletes_a_dir_it_did_not_create(env):
    """A legal id can still name an ALREADY-INSTALLED plugin's directory - on a
    case-insensitive filesystem 'MyTool' is 'mytool'. The id check cannot see
    that (it is a shape rule, not a uniqueness one), so the rollback must not
    delete a directory this install never created."""
    from localm.plugins.engine import PluginManager
    store = env / "store"; store.mkdir(parents=True, exist_ok=True)
    installed = env / "installed"
    # register() raises, so the LOAD-FAILURE rollback site is the one exercised
    # for the exact-case name; a working plugin here would install cleanly.
    real = _make_plugin(installed, "mytool",
                        'def register(host):\n    raise RuntimeError("boom")\n'
                        'def unregister():\n    pass\n')
    (real / "USER_DATA.txt").write_text("irreplaceable", encoding="utf-8")

    mgr = PluginManager(FastAPI(), store_root=store, installed_root=installed)
    # install() has TWO rollback sites (verify-failed and load-failed), so all
    # three variants are pinned. test_install_rolls_back_a_copy_it_created covers
    # a copy this call DID create still being rolled back.
    for variant in ("MyTool", "mytool", "MYTOOL"):
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            mgr.install(variant)          # no store source -> cannot succeed
        assert real.is_dir(), f"{variant} deleted the installed plugin"
        assert (real / "USER_DATA.txt").read_text(encoding="utf-8") == "irreplaceable"


def test_remove_installed_dir_confines_without_demanding_an_identifier(env):
    """The DELETE site confines by resolved parent, not identifier shape, so a
    legitimately-installed directory whose basename is not identifier-shaped (a
    hand-extracted 'coolplugin-1.0') can still be uninstalled. This pins BOTH
    directions: the odd basename is removable, a traversing one is still
    refused."""
    from localm.plugins.engine import PluginManager
    installed = env / "installed"
    odd = _make_plugin(installed, "coolplugin-1.0", _ping("coolplugin"))
    sentinel = _make_plugin(env, "outside", _ping("outside"))
    (sentinel / "keep.txt").write_text("important", encoding="utf-8")
    mgr = PluginManager(None, store_root=env / "store", installed_root=installed)

    # the guard still FIRES on a genuinely different failure
    with pytest.raises(ValueError, match="installed-plugins root"):
        mgr._remove_installed_dir("../outside")
    assert _sentinel_intact(sentinel)

    # ...and no longer blocks a legitimate removal
    assert odd.is_dir()
    mgr._remove_installed_dir("coolplugin-1.0")
    assert not odd.exists()


def test_parse_spec_reads_tool_exports(tmp_path):
    """PluginSpec calls itself a superset of loader.PluginManifest, so it must
    carry [tools] exports too - the GUI reads it straight off api_state now."""
    from localm.plugins.engine import parse_spec
    src = _make_legacy_plugin(tmp_path, "texp", exports='["a", "b"]')
    assert parse_spec(src).tool_exports == ["a", "b"]

    # absent [tools] -> empty, not an error
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "plugin.toml").write_text('[plugin]\nname = "plain"\n', encoding="utf-8")
    assert parse_spec(plain).tool_exports == []

    # malformed -> rejected with the same message shape the legacy loader uses,
    # so the two parsers of this one key cannot drift.
    bad = _make_legacy_plugin(tmp_path / "b", "badexp", exports='"not-a-list"')
    with pytest.raises(ValueError, match="must be a list of strings"):
        parse_spec(bad)
