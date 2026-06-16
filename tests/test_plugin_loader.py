"""Tests for localm.plugins.loader - external plugin discovery and management."""

import sys
import textwrap
from pathlib import Path

import pytest

from localm.plugins.loader import (
    PluginError,
    PluginManifest,
    discover_errors,
    discover_plugins,
    install_plugin,
    load_entry,
    parse_manifest,
    register_external_plugins,
    remove_plugin,
)


VALID_TOML = textwrap.dedent("""\
    [plugin]
    name = "demo"
    version = "1.2.3"
    description = "A demo plugin"
    entry = "demo_cli:main"

    [tools]
    exports = ["tool_hello"]
""")

VALID_ENTRY = textwrap.dedent("""\
    import click

    @click.command()
    def main():
        \"\"\"Demo plugin command.\"\"\"
        click.echo("demo ran")

    def tool_hello():
        return "hello"
""")


def _make_plugin(root: Path, name: str = "demo", toml: str = VALID_TOML,
                 entry_file: str = "demo_cli.py", entry_src: str = VALID_ENTRY) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.toml").write_text(toml, encoding="utf-8")
    (plugin_dir / entry_file).write_text(entry_src, encoding="utf-8")
    return plugin_dir


# ---------------------------------------------------------------------------
#  parse_manifest
# ---------------------------------------------------------------------------

class TestParseManifest:
    def test_valid_manifest(self, tmp_path):
        d = _make_plugin(tmp_path)
        m = parse_manifest(d)
        assert m.name == "demo"
        assert m.version == "1.2.3"
        assert m.entry_module == "demo_cli"
        assert m.entry_attr == "main"
        assert m.tool_exports == ["tool_hello"]

    def test_missing_toml(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(PluginError, match="No plugin.toml"):
            parse_manifest(d)

    def test_invalid_toml_syntax(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "plugin.toml").write_text("[plugin\nname=", encoding="utf-8")
        with pytest.raises(PluginError, match="Invalid TOML"):
            parse_manifest(d)

    def test_missing_plugin_table(self, tmp_path):
        d = tmp_path / "notable"
        d.mkdir()
        (d / "plugin.toml").write_text("[other]\nx = 1\n", encoding="utf-8")
        with pytest.raises(PluginError, match="missing \\[plugin\\] table"):
            parse_manifest(d)

    def test_missing_name(self, tmp_path):
        d = tmp_path / "noname"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nentry = "x:main"\n', encoding="utf-8")
        with pytest.raises(PluginError, match="name is required"):
            parse_manifest(d)

    def test_reserved_name_rejected(self, tmp_path):
        d = tmp_path / "clash"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nname = "pull"\nentry = "x:main"\n', encoding="utf-8")
        with pytest.raises(PluginError, match="clashes with a built-in"):
            parse_manifest(d)

    def test_entry_without_colon_rejected(self, tmp_path):
        d = tmp_path / "badentry"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nname = "ok"\nentry = "nomodule"\n', encoding="utf-8")
        with pytest.raises(PluginError, match="entry must be"):
            parse_manifest(d)

    def test_invalid_name_rejected(self, tmp_path):
        d = tmp_path / "badname"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nname = "has space"\nentry = "x:main"\n', encoding="utf-8")
        with pytest.raises(PluginError, match="invalid plugin name"):
            parse_manifest(d)


# ---------------------------------------------------------------------------
#  discover_plugins
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_empty_root_returns_nothing(self, tmp_path):
        assert discover_plugins(tmp_path / "missing") == []

    def test_finds_valid_plugins(self, tmp_path):
        _make_plugin(tmp_path, "demo")
        found = discover_plugins(tmp_path)
        assert [m.name for m in found] == ["demo"]

    def test_skips_dirs_without_manifest(self, tmp_path):
        (tmp_path / "junk").mkdir()
        _make_plugin(tmp_path, "demo")
        assert [m.name for m in discover_plugins(tmp_path)] == ["demo"]

    def test_invalid_plugin_reported_in_errors(self, tmp_path):
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "plugin.toml").write_text("[plugin]\n", encoding="utf-8")
        assert discover_plugins(tmp_path) == []
        errs = discover_errors(tmp_path)
        assert len(errs) == 1
        assert "name is required" in errs[0]

    def test_engine_plugin_ignored_not_errored(self, tmp_path):
        """An engine-contract plugin (register=, no entry=) shares the installed
        dir but belongs to the plugin engine; the legacy loader skips it silently
        instead of reporting it as a broken legacy plugin."""
        eng = tmp_path / "coder"
        eng.mkdir()
        (eng / "plugin.toml").write_text(
            '[plugin]\nname = "coder"\nscope = "coder"\nregister = "plug"\n',
            encoding="utf-8")
        _make_plugin(tmp_path, "demo")           # a real legacy plugin alongside
        assert [m.name for m in discover_plugins(tmp_path)] == ["demo"]
        assert discover_errors(tmp_path) == []   # engine plugin produced no error


# ---------------------------------------------------------------------------
#  load_entry
# ---------------------------------------------------------------------------

class TestLoadEntry:
    def test_loads_click_command(self, tmp_path):
        d = _make_plugin(tmp_path)
        m = parse_manifest(d)
        cmd = load_entry(m)
        import click
        assert isinstance(cmd, click.Command)

    def test_missing_entry_module(self, tmp_path):
        d = _make_plugin(tmp_path, entry_file="other.py")
        m = parse_manifest(d)
        with pytest.raises(PluginError, match="not found"):
            load_entry(m)

    def test_missing_entry_attr(self, tmp_path):
        d = _make_plugin(tmp_path, entry_src="x = 1\n")
        m = parse_manifest(d)
        with pytest.raises(PluginError, match="entry attribute"):
            load_entry(m)

    def test_import_error_wrapped(self, tmp_path):
        d = _make_plugin(tmp_path, entry_src="raise RuntimeError('boom')\n")
        m = parse_manifest(d)
        with pytest.raises(PluginError, match="failed to import"):
            load_entry(m)
        # broken module must not linger in sys.modules
        assert not any(k.startswith("_localm_plugin_demo") and getattr(
            sys.modules.get(k), "__file__", "").startswith(str(tmp_path))
            for k in list(sys.modules))


# ---------------------------------------------------------------------------
#  register_external_plugins
# ---------------------------------------------------------------------------

class TestRegister:
    def test_registers_command_on_group(self, tmp_path, monkeypatch):
        import click
        monkeypatch.setattr(
            "localm.plugins.loader.plugins_dir", lambda: tmp_path)
        _make_plugin(tmp_path)

        @click.group()
        def grp():
            pass

        warnings = register_external_plugins(grp)
        assert warnings == []
        assert "demo" in grp.commands

    def test_name_clash_warns_and_skips(self, tmp_path, monkeypatch):
        import click
        monkeypatch.setattr(
            "localm.plugins.loader.plugins_dir", lambda: tmp_path)
        _make_plugin(tmp_path)

        @click.group()
        def grp():
            pass

        @grp.command("demo")
        def existing():
            pass

        warnings = register_external_plugins(grp)
        assert len(warnings) == 1
        assert "already registered" in warnings[0]

    def test_broken_plugin_warns_but_does_not_raise(self, tmp_path, monkeypatch):
        import click
        monkeypatch.setattr(
            "localm.plugins.loader.plugins_dir", lambda: tmp_path)
        _make_plugin(tmp_path, entry_src="raise ImportError('nope')\n")

        @click.group()
        def grp():
            pass

        warnings = register_external_plugins(grp)
        assert len(warnings) == 1
        assert "demo" not in grp.commands


# ---------------------------------------------------------------------------
#  install / remove
# ---------------------------------------------------------------------------

class TestInstallRemove:
    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path, monkeypatch):
        self.root = tmp_path / "installed"
        monkeypatch.setattr(
            "localm.plugins.loader.plugins_dir", lambda: self.root)

    def test_install_copies_directory(self, tmp_path):
        src = _make_plugin(tmp_path / "src")
        m = install_plugin(src)
        assert m.path == self.root / "demo"
        assert (self.root / "demo" / "plugin.toml").is_file()
        assert (self.root / "demo" / "demo_cli.py").is_file()

    def test_install_rejects_duplicate_without_force(self, tmp_path):
        src = _make_plugin(tmp_path / "src")
        install_plugin(src)
        with pytest.raises(PluginError, match="already installed"):
            install_plugin(src)

    def test_install_force_overwrites(self, tmp_path):
        src = _make_plugin(tmp_path / "src")
        install_plugin(src)
        m = install_plugin(src, force=True)
        assert m.name == "demo"

    def test_install_excludes_pycache(self, tmp_path):
        src = _make_plugin(tmp_path / "src")
        (src / "__pycache__").mkdir()
        (src / "__pycache__" / "x.pyc").write_bytes(b"")
        install_plugin(src)
        assert not (self.root / "demo" / "__pycache__").exists()

    def test_install_validates_before_copy(self, tmp_path):
        bad = tmp_path / "src" / "bad"
        bad.mkdir(parents=True)
        (bad / "plugin.toml").write_text("[plugin]\n", encoding="utf-8")
        with pytest.raises(PluginError):
            install_plugin(bad)
        assert not self.root.exists()

    def test_remove_existing(self, tmp_path):
        src = _make_plugin(tmp_path / "src")
        install_plugin(src)
        assert remove_plugin("demo") is True
        assert not (self.root / "demo").exists()

    def test_remove_missing_returns_false(self):
        assert remove_plugin("ghost") is False
