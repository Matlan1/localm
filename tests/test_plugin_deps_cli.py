# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and CLI behaviour for auto-installing a plugin's pip extras.

The actual pip run and metadata resolution are unit-tested separately; here the
ENGINE methods and the `localm plugin` commands are driven, with the deps layer
mocked so nothing is really installed. A synthetic store plugin ``withdeps``
declares ``requires_extras = ["fakeextra"]``.
"""

import pytest
from click.testing import CliRunner

from localm.plugins import deps


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

    store = tmp_path / "store"
    installed = tmp_path / "installed"
    plugins = {
        "withdeps": 'requires_extras = ["fakeextra"]\n',
        "plain": "",
    }
    for name, extra in plugins.items():
        d = store / name
        d.mkdir(parents=True)
        (d / "plugin.toml").write_text(
            f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{extra}',
            encoding="utf-8")
        (d / "plug.py").write_text(
            "def register(host):\n    pass\n\ndef unregister():\n    pass\n",
            encoding="utf-8")

    import localm.cli as climod
    from localm.plugins.engine import PluginManager

    def mk():
        return PluginManager(None, store_root=store, installed_root=installed)

    monkeypatch.setattr(climod, "_engine_manager", mk)

    # Resolve the fake extra to a fake package; nothing here touches real pip.
    monkeypatch.setattr(
        deps, "extra_requirements",
        lambda e: ["fakepkg>=1.0"] if e == "fakeextra" else [f"localm[{e}]"])

    from types import SimpleNamespace
    return SimpleNamespace(main=climod.main, mk=mk, store=store, installed=installed)


def _all_missing(monkeypatch):
    """Pretend every requirement is missing (so installs run)."""
    monkeypatch.setattr(deps, "missing_requirements", lambda reqs: list(reqs))


def _none_missing(monkeypatch):
    monkeypatch.setattr(deps, "missing_requirements", lambda reqs: [])


def _fake_catalog(monkeypatch):
    """Point catalog.CATALOG at the synthetic plugins so `plugin setup` (which
    validates selections against the catalog) accepts them."""
    from localm.plugins import catalog as cat
    fake = (
        cat.CatalogEntry("chat", "Chat", preinstalled=True, protected=True),
        cat.CatalogEntry("withdeps", "Needs a pip extra", extra="fakeextra"),
        cat.CatalogEntry("plain", "No extras"),
    )
    monkeypatch.setattr(cat, "CATALOG", fake)


def _record_installs(monkeypatch, *, ok=True, error=""):
    calls = []

    def fake(extras, on_progress=None):
        calls.append(list(extras))
        reqs = deps.plugin_requirements(extras)
        return deps.InstallResult(ok=ok, installed=reqs if ok else [],
                                  failed=[] if ok else reqs, error=error)

    monkeypatch.setattr(deps, "install_plugin_extras", fake)
    return calls


# --------------------------------------------------------------------------- #
#  Engine methods                                                             #
# --------------------------------------------------------------------------- #

def test_engine_requirements_and_missing(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    assert mgr.plugin_requirements("withdeps") == ["fakepkg>=1.0"]
    assert mgr.plugin_requirements("plain") == []
    _all_missing(monkeypatch)
    assert mgr.plugin_missing_deps("withdeps") == ["fakepkg>=1.0"]
    _none_missing(monkeypatch)
    assert mgr.plugin_missing_deps("withdeps") == []


def test_engine_all_missing_deps_enabled_only(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)      # installed + enabled
    mgr.set_installed_state("plain", True)
    _all_missing(monkeypatch)
    assert mgr.all_missing_deps() == {"withdeps": ["fakepkg>=1.0"]}
    mgr.set_enabled_state("withdeps", False)       # disabled -> excluded
    assert mgr.all_missing_deps(enabled_only=True) == {}
    assert "withdeps" in mgr.all_missing_deps(enabled_only=False)


def test_engine_install_plugin_deps_passthrough(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    sentinel = deps.InstallResult(ok=True, installed=["fakepkg>=1.0"])
    monkeypatch.setattr(deps, "install_plugin_extras",
                        lambda extras, on_progress=None: sentinel)
    assert mgr.install_plugin_deps("withdeps") is sentinel


def test_api_state_reports_missing_deps(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    _all_missing(monkeypatch)
    entry = {p["name"]: p for p in mgr.api_state()["plugins"]}["withdeps"]
    assert entry["missing_deps"] == ["fakepkg>=1.0"]
    assert entry["requires_extras"] == ["fakeextra"]


# --------------------------------------------------------------------------- #
#  CLI: install / enable honour the flag + config                             #
# --------------------------------------------------------------------------- #

def test_install_with_deps_triggers(env, monkeypatch):
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install", "withdeps", "--with-deps"])
    assert r.exit_code == 0, r.output
    assert calls == [["fakeextra"]]
    assert "Installed dependencies" in r.output


def test_install_no_deps_skips(env, monkeypatch):
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install", "withdeps", "--no-deps"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "install-deps" in r.output          # points at the manual path


def test_install_default_off_by_config(env, monkeypatch):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["auto_install_plugin_deps"] = False
    save_config(cfg)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install", "withdeps"])
    assert r.exit_code == 0 and calls == []
    assert "install-deps" in r.output


def test_install_default_on_by_config(env, monkeypatch):
    # default config has auto_install_plugin_deps True
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install", "withdeps"])
    assert r.exit_code == 0 and calls == [["fakeextra"]]


def test_enable_with_deps(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    mgr.set_enabled_state("withdeps", False)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "enable", "withdeps", "--with-deps"])
    assert r.exit_code == 0, r.output
    assert calls == [["fakeextra"]]


# --------------------------------------------------------------------------- #
#  CLI: install-deps repair command                                           #
# --------------------------------------------------------------------------- #

def test_install_deps_list(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    _all_missing(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install-deps"])
    assert r.exit_code == 0
    assert "withdeps" in r.output and "fakepkg>=1.0" in r.output


def test_install_deps_all(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install-deps", "--all"])
    assert r.exit_code == 0, r.output
    assert calls == [["fakeextra"]]


def test_install_deps_unknown_name(env, monkeypatch):
    r = CliRunner().invoke(env.main, ["plugin", "install-deps", "nope"])
    assert r.exit_code == 1 and "No such plugin" in r.output


def test_install_deps_name_satisfied(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    _none_missing(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "install-deps", "withdeps"])
    assert r.exit_code == 0 and "has its dependencies" in r.output


def test_install_deps_failure_is_nonzero(env, monkeypatch):
    mgr = env.mk()
    mgr.set_installed_state("withdeps", True)
    _all_missing(monkeypatch)
    _record_installs(monkeypatch, ok=False, error="ERROR: boom")
    r = CliRunner().invoke(env.main, ["plugin", "install-deps", "withdeps"])
    assert r.exit_code == 1
    assert "failed" in r.output.lower() and "boom" in r.output


# --------------------------------------------------------------------------- #
#  CLI: setup                                                                 #
# --------------------------------------------------------------------------- #

def test_setup_with_deps_flag(env, monkeypatch):
    _fake_catalog(monkeypatch)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(
        env.main, ["plugin", "setup", "--plugins", "withdeps", "--with-deps"])
    assert r.exit_code == 0, r.output
    assert calls == [["fakeextra"]]


def test_setup_no_deps_flag_notes_pending(env, monkeypatch):
    _fake_catalog(monkeypatch)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(
        env.main, ["plugin", "setup", "--plugins", "withdeps", "--no-deps"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "install-deps --all" in r.output


def test_setup_uses_config_default_on(env, monkeypatch):
    # No --with-deps/--no-deps: the non-interactive setup path follows the
    # auto_install_plugin_deps setting (default True).
    _fake_catalog(monkeypatch)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "setup", "--plugins", "withdeps"])
    assert r.exit_code == 0, r.output
    assert calls == [["fakeextra"]]


def test_setup_uses_config_default_off(env, monkeypatch):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["auto_install_plugin_deps"] = False
    save_config(cfg)
    _fake_catalog(monkeypatch)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)
    r = CliRunner().invoke(env.main, ["plugin", "setup", "--plugins", "withdeps"])
    assert r.exit_code == 0, r.output
    assert calls == [] and "install-deps --all" in r.output


def test_set_auto_deps_roundtrip(env):
    # The persistence helper the interactive prompt uses to remember the choice.
    from localm.cli.plugins import _auto_deps_default, _set_auto_deps
    assert _auto_deps_default() is True             # default
    _set_auto_deps(False)
    assert _auto_deps_default() is False
    _set_auto_deps(True)
    assert _auto_deps_default() is True


def test_scope_deps_warnings(env, monkeypatch):
    mgr = env.mk()
    # Not installed -> warn that the plugin is missing entirely.
    warns = mgr.scope_deps_warnings(["withdeps"])
    assert len(warns) == 1 and "not installed" in warns[0]
    # Installed but missing its pip extra -> warn about the packages.
    mgr.set_installed_state("withdeps", True)
    _all_missing(monkeypatch)
    warns = mgr.scope_deps_warnings(["withdeps"])
    assert len(warns) == 1 and "fakepkg>=1.0" in warns[0]
    # Ready -> no warning; an unrelated non-plugin scope -> no warning.
    _none_missing(monkeypatch)
    assert mgr.scope_deps_warnings(["withdeps", "models:read"]) == []


def test_key_create_warns_on_uninstalled_plugin_scope(env, monkeypatch):
    # key_create builds its own PluginManager(None) over the real catalog; the
    # 'voice' plugin is a known first-party plugin, not installed in this tmp home.
    r = CliRunner().invoke(env.main, ["key", "create", "dash", "-s", "voice"])
    assert r.exit_code == 0, r.output
    assert "voice plugin is not installed" in r.output
    assert "install-deps" in r.output           # exact wording may line-wrap


def test_doctor_reports_missing_plugin_deps(env, monkeypatch, capsys):
    import importlib
    doc = importlib.import_module("localm.cli.doctor")
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(PluginManager, "all_missing_deps",
                        lambda self, enabled_only=True: {"voice": ["faster-whisper>=1.0"]})
    doc._check_plugin_deps()
    out = capsys.readouterr().out
    assert "voice" in out and "install-deps" in out


def test_doctor_ok_when_no_missing_deps(env, monkeypatch, capsys):
    import importlib
    doc = importlib.import_module("localm.cli.doctor")
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(PluginManager, "all_missing_deps",
                        lambda self, enabled_only=True: {})
    doc._check_plugin_deps()
    assert "have theirs" in capsys.readouterr().out


def test_setup_interactive_prompt_records_and_installs(env, monkeypatch):
    """Drive the interactive branch directly (CliRunner's stdin is not a tty, so
    the no-flag path would skip). A faked tty stdin feeds the plugin pick and a
    'yes' to the deps prompt; the choice is persisted and the install runs."""
    import io
    from localm.cli import plugins as plug
    from localm.config import load_config
    _fake_catalog(monkeypatch)
    _all_missing(monkeypatch)
    calls = _record_installs(monkeypatch)

    class _TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(plug.sys, "stdin", _TTY("withdeps\ny\n"))
    # Call the command's underlying function directly (no CliRunner stdin swap).
    plug.plugin_setup.callback(plugins_csv=None, install_all=False,
                               install_defaults=False, with_deps=None)
    assert calls == [["fakeextra"]]
    assert load_config().get("auto_install_plugin_deps") is True
