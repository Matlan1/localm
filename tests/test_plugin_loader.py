# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.loader - the legacy plugin manifest ("entry =")
discovery/import machinery. The live surface is parse_manifest /
discover_plugins / import_plugin_module: it is how
localm.plugins.coder.plugin_tools.register_plugin_tools() discovers and loads
third-party coder-agent tool exports."""

import sys
import textwrap
from pathlib import Path

import pytest

from localm.plugins.loader import (
    PluginError,
    discover_errors,
    discover_plugins,
    discover_warnings,
    import_plugin_module,
    parse_manifest,
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

    def test_unknown_manifest_key_warned_not_fatal(self, tmp_path):
        """A typoed [plugin] key does not fail the plugin (it still parses and
        loads) and is surfaced via discover_warnings."""
        toml = textwrap.dedent("""\
            [plugin]
            name = "demo"
            version = "1.0.0"
            descriptoin = "typo"
            entry = "demo_cli:main"
        """)
        d = _make_plugin(tmp_path, "demo", toml=toml)
        warns = []
        m = parse_manifest(d, warnings=warns)
        assert m.name == "demo"                          # parse still succeeds
        assert len(warns) == 1 and "descriptoin" in warns[0]
        assert [x.name for x in discover_plugins(tmp_path)] == ["demo"]
        assert discover_errors(tmp_path) == []           # a warning, not an error
        ws = discover_warnings(tmp_path)
        assert len(ws) == 1
        assert "unknown [plugin] key(s) ignored" in ws[0]

    def test_engine_contract_key_not_warned(self, tmp_path):
        """A key valid in the ENGINE manifest format (e.g. ``register``) raises
        no warning here; both formats share the installed dir."""
        toml = textwrap.dedent("""\
            [plugin]
            name = "demo"
            entry = "demo_cli:main"
            register = "plug"
        """)
        d = _make_plugin(tmp_path, "demo", toml=toml)
        warns = []
        parse_manifest(d, warnings=warns)
        assert warns == []

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
#  import_plugin_module (still live: backs plugin_tools.register_plugin_tools)
# ---------------------------------------------------------------------------

class TestImportPluginModule:
    def test_imports_module(self, tmp_path):
        d = _make_plugin(tmp_path)
        m = parse_manifest(d)
        mod = import_plugin_module(m)
        assert hasattr(mod, "main")

    def test_missing_entry_module(self, tmp_path):
        d = _make_plugin(tmp_path, entry_file="other.py")
        m = parse_manifest(d)
        with pytest.raises(PluginError, match="not found"):
            import_plugin_module(m)

    def test_import_error_wrapped(self, tmp_path):
        d = _make_plugin(tmp_path, entry_src="raise RuntimeError('boom')\n")
        m = parse_manifest(d)
        with pytest.raises(PluginError, match="failed to import"):
            import_plugin_module(m)
        # broken module must not linger in sys.modules
        assert not any(k.startswith("_localm_plugin_demo") and getattr(
            sys.modules.get(k), "__file__", "").startswith(str(tmp_path))
            for k in list(sys.modules))
