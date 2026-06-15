"""CLI tests for the engine plugin toggles: `localm plugin enable/disable/status`.

These exercise the new-engine commands wired in localm/cli.py (set_enabled_state +
missing_requires), driven through Click's CliRunner against synthetic plugins."""

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Config isolated to tmp; a synthetic external plugin dir with 'needy'
    (requires dep1) and 'dep1'. _engine_manager is pointed at it."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

    store = tmp_path / "store"          # the available catalog (bundled shelf)
    installed = tmp_path / "installed"  # where install copies them
    for name, extra in (("needy", 'requires = ["dep1"]\n'), ("dep1", "")):
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
    monkeypatch.setattr(
        climod, "_engine_manager",
        lambda: PluginManager(None, store_root=store, installed_root=installed))
    from types import SimpleNamespace
    return SimpleNamespace(main=climod.main, store=store, installed=installed)


def test_install_uninstall_roundtrip(cli_env):
    from localm.config import load_config
    r = CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])
    assert r.exit_code == 0 and "Installed" in r.output
    assert (cli_env.installed / "dep1").is_dir()          # physically installed
    assert "dep1" in load_config().get("plugins_enabled", [])   # enabled by default
    r = CliRunner().invoke(cli_env.main, ["plugin", "uninstall", "dep1"])
    assert r.exit_code == 0 and "Uninstalled" in r.output
    assert not (cli_env.installed / "dep1").exists()      # dir removed
    assert "dep1" not in load_config().get("plugins_enabled", [])


def test_enable_disable_within_installed(cli_env):
    from localm.config import load_config
    CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])
    r = CliRunner().invoke(cli_env.main, ["plugin", "disable", "dep1"])
    assert r.exit_code == 0 and "Disabled" in r.output
    assert (cli_env.installed / "dep1").is_dir()          # stays installed (on disk)
    assert "dep1" not in load_config().get("plugins_enabled", [])
    r = CliRunner().invoke(cli_env.main, ["plugin", "enable", "dep1"])
    assert r.exit_code == 0 and "Enabled" in r.output
    assert "dep1" in load_config().get("plugins_enabled", [])


def test_enable_before_install_is_error(cli_env):
    r = CliRunner().invoke(cli_env.main, ["plugin", "enable", "dep1"])
    assert r.exit_code == 1
    assert "not installed" in r.output.lower()


def test_install_unknown_is_error(cli_env):
    r = CliRunner().invoke(cli_env.main, ["plugin", "install", "ghost"])
    assert r.exit_code == 1
    assert "No such plugin" in r.output


def test_install_warns_missing_requires_with_real_names(cli_env):
    """The dependency warning must name the actual missing plugin AND give the
    exact install command (no literal <name> placeholder)."""
    r = CliRunner().invoke(cli_env.main, ["plugin", "install", "needy"])
    assert r.exit_code == 0
    assert "dep1" in r.output
    assert "localm plugin install dep1" in r.output
    assert "<name>" not in r.output


def test_status_shows_installed_and_available(cli_env):
    CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])
    r = CliRunner().invoke(cli_env.main, ["plugin", "status"])
    assert r.exit_code == 0
    assert "Installed" in r.output and "Available" in r.output
    assert "dep1" in r.output and "needy" in r.output
