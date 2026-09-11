# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI tests for the engine plugin toggles: `localm plugin enable/disable/status`.

These exercise the engine commands wired in localm/cli.py (set_enabled_state +
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


def _wedged_manager_factory(cli_env, **overrides):
    """Build a factory for climod._engine_manager that constructs a FRESH
    PluginManager per call (matching the real one) and applies *overrides* as
    instance attributes on each one built. cli_env's lambda constructs a new
    instance every call, so patching an instance obtained from it is silently
    discarded; the wedge must live in the factory itself."""
    from localm.plugins.engine import PluginManager

    def factory():
        m = PluginManager(None, store_root=cli_env.store, installed_root=cli_env.installed)
        for attr, value in overrides.items():
            setattr(m, attr, value)
        return m
    return factory


def test_uninstall_failed_file_removal_is_not_reported_as_not_installed(cli_env, monkeypatch):
    import localm.cli as climod

    r = CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])
    assert r.exit_code == 0 and (cli_env.installed / "dep1").is_dir()

    monkeypatch.setattr(
        climod, "_engine_manager",
        _wedged_manager_factory(cli_env, _remove_installed_dir=lambda name: False))

    r = CliRunner().invoke(cli_env.main, ["plugin", "uninstall", "dep1"])
    assert (cli_env.installed / "dep1").is_dir(), (
        "the injection did not take: removal succeeded")
    assert "stored data" not in r.output.lower()
    assert "was not installed" not in r.output.lower()
    assert r.exit_code != 0


def test_uninstall_failed_data_delete_does_not_claim_files_stayed(cli_env, monkeypatch, tmp_path):
    import localm.cli as climod

    (cli_env.store / "dep1" / "plugin.toml").write_text(
        '[plugin]\nname = "dep1"\nscope = "dep1"\nregister = "plug"\n'
        'data_subdir = "dep1_data"\n',
        encoding="utf-8")
    data_dir = tmp_path / "dep1_data"
    data_dir.mkdir()
    (data_dir / "private.txt").write_text("USER SECRET", encoding="utf-8")

    r = CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])
    assert r.exit_code == 0 and (cli_env.installed / "dep1").is_dir()

    monkeypatch.setattr(
        climod, "_engine_manager",
        _wedged_manager_factory(cli_env, _delete_plugin_data=lambda spec: False))

    r = CliRunner().invoke(cli_env.main, ["plugin", "uninstall", "dep1", "--delete-data"])
    assert (data_dir / "private.txt").exists()
    assert (data_dir / "private.txt").read_text(encoding="utf-8") == "USER SECRET"
    assert not (cli_env.installed / "dep1").exists()
    assert "files" not in r.output.lower(), (
        f"the message names files that were in fact removed: {r.output!r}")
    assert "was not installed" not in r.output.lower()
    assert r.exit_code != 0


def test_uninstall_benign_not_installed_branch_still_reports_cleanly(cli_env):
    """A directory-name/manifest-name mismatch makes is_installed(<manifest
    name>) False while the directory itself is untouched - the benign branch
    plugin_uninstall_engine's fix newly introduces. Not reachable through
    `plugin install`: a mismatched manifest is refused and rolled back."""
    foo_dir = cli_env.installed / "foo"
    foo_dir.mkdir(parents=True)
    (foo_dir / "plugin.toml").write_text(
        '[plugin]\nname = "bar"\nscope = "bar"\nregister = "plug"\n', encoding="utf-8")
    (foo_dir / "plug.py").write_text(
        "def register(host):\n    pass\n\ndef unregister():\n    pass\n", encoding="utf-8")

    r = CliRunner().invoke(cli_env.main, ["plugin", "uninstall", "bar"])
    assert foo_dir.is_dir()
    assert "was not installed" in r.output.lower()
    assert r.exit_code == 0


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
    """The dependency warning names the actual missing plugin and gives the
    exact install command, with no literal <name> placeholder."""
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


def test_install_from_directory(cli_env, tmp_path):
    """`plugin install <dir>` installs a THIRD-PARTY plugin by path (not a store
    name); re-installing the same dir without --force errors."""
    from localm.config import load_config
    ext = tmp_path / "thirdparty"
    ext.mkdir()
    (ext / "plugin.toml").write_text(
        '[plugin]\nname = "ext1"\nscope = "ext1"\nregister = "plug"\n', encoding="utf-8")
    (ext / "plug.py").write_text(
        "def register(host):\n    pass\n\ndef unregister():\n    pass\n", encoding="utf-8")

    r = CliRunner().invoke(cli_env.main, ["plugin", "install", str(ext)])
    assert r.exit_code == 0 and "Installed" in r.output and "ext1" in r.output
    assert (cli_env.installed / "ext1").is_dir()                 # copied into installed
    assert "ext1" in load_config().get("plugins_enabled", [])    # enabled

    r2 = CliRunner().invoke(cli_env.main, ["plugin", "install", str(ext)])
    assert r2.exit_code == 1 and "already installed" in r2.output.lower()


def test_install_from_directory_surfaces_resolved_scope(cli_env, tmp_path):
    """The CLI install success message shows the resolved scope: the capability
    every route the plugin registers is gated on."""
    ext = tmp_path / "thirdparty2"
    ext.mkdir()
    (ext / "plugin.toml").write_text(
        '[plugin]\nname = "ext2"\nscope = "ext2"\nregister = "plug"\n', encoding="utf-8")
    (ext / "plug.py").write_text(
        "def register(host):\n    pass\n\ndef unregister():\n    pass\n", encoding="utf-8")

    r = CliRunner().invoke(cli_env.main, ["plugin", "install", str(ext)])
    assert r.exit_code == 0
    assert "Granted scope" in r.output and "ext2" in r.output


def test_install_from_directory_rejects_scope_collision(cli_env, tmp_path):
    """A manifest whose scope collides with a first-party plugin's (here
    'dep1', already present in the store) is rejected with an explicit error
    and not installed."""
    ext = tmp_path / "thirdparty3"
    ext.mkdir()
    (ext / "plugin.toml").write_text(
        '[plugin]\nname = "sneaky"\nscope = "dep1"\nregister = "plug"\n', encoding="utf-8")
    (ext / "plug.py").write_text(
        "def register(host):\n    pass\n\ndef unregister():\n    pass\n", encoding="utf-8")

    CliRunner().invoke(cli_env.main, ["plugin", "install", "dep1"])   # already-installed scope owner
    r = CliRunner().invoke(cli_env.main, ["plugin", "install", str(ext)])
    assert r.exit_code == 1
    output = " ".join(r.output.split())            # rich console word-wraps long lines
    assert "already used by installed plugin" in output
    assert "dep1" in output
    assert not (cli_env.installed / "sneaky").exists()


# --------------------------------------------------------------------------- #
#  `plugin setup --plugins`: reject all-junk selections                        #
# --------------------------------------------------------------------------- #

def test_setup_plugins_all_junk_is_error(cli_env):
    """A non-interactive --plugins selection that resolves to nothing exits 1
    rather than reporting a no-op as success."""
    r = CliRunner().invoke(cli_env.main, ["plugin", "setup", "--plugins", "ewew"])
    assert r.exit_code == 1
    assert "no known plugins" in r.output.lower()
    # The bad token is named, and the valid choices are listed.
    assert "ewew" in r.output
    assert "coder" in r.output


def test_setup_plugins_blank_is_a_skip_not_an_error(cli_env):
    """An explicitly blank selection is a deliberate skip and exits 0."""
    r = CliRunner().invoke(cli_env.main, ["plugin", "setup", "--plugins", "   "])
    assert r.exit_code == 0
    assert "no plugins selected" in r.output.lower()
