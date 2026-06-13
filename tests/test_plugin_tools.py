"""Plugins can export real, callable tools into the coder agent.

Regression guard for the audit finding that plugin `tool_exports` were parsed
and displayed but never wired into the agent (the README claim was vapor).
"""

import textwrap
from unittest.mock import patch

from localm.plugins.coder import tools as coder_tools
from localm.plugins.coder.plugin_tools import register_plugin_tools
from localm.plugins.coder.tools import ToolResult


_TOML = textwrap.dedent("""\
    [plugin]
    name = "issues"
    version = "0.1.0"
    description = "Issue tools"
    entry = "issues_cli:main"

    [tools]
    exports = ["tool_echo", "tool_danger", "not_a_func"]
""")

_ENTRY = textwrap.dedent('''\
    import click

    @click.command()
    def main():
        """Issues command."""

    def tool_echo(cwd, text="hi"):
        """Echo the text back to the caller."""
        return f"echo: {text}"
    tool_echo.tool_params = {"text": {"type": "string",
                                      "description": "text", "required": False}}
    tool_echo.tool_destructive = False

    def tool_danger(cwd, **args):
        from localm.plugins.coder.tools import ToolResult
        return ToolResult.success("did a thing")

    not_a_func = 42
''')


def _make_tool_plugin(root):
    d = root / "issues"
    d.mkdir(parents=True)
    (d / "plugin.toml").write_text(_TOML, encoding="utf-8")
    (d / "issues_cli.py").write_text(_ENTRY, encoding="utf-8")
    return d


class TestRegisterPluginTools:
    def _setup(self, tmp_path, monkeypatch):
        _make_tool_plugin(tmp_path)
        monkeypatch.setattr("localm.plugins.loader.plugins_dir",
                            lambda: tmp_path)

    def test_registers_exported_tool_with_metadata(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            names, _ = register_plugin_tools()
            assert "plugin_issues_tool_echo" in names
            td = coder_tools.TOOL_REGISTRY["plugin_issues_tool_echo"]
            assert td.destructive is False
            assert td.description.startswith("[plugin:issues]")
            assert "Echo the text" in td.description
            assert td.params["text"]["type"] == "string"

    def test_tool_fn_runs_and_coerces_string(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            register_plugin_tools()
            fn = coder_tools.TOOL_REGISTRY["plugin_issues_tool_echo"].fn
            result = fn(tmp_path, text="yo")
            assert isinstance(result, ToolResult)
            assert result.ok and result.output == "echo: yo"

    def test_toolresult_return_passes_through(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            register_plugin_tools()
            fn = coder_tools.TOOL_REGISTRY["plugin_issues_tool_danger"].fn
            assert fn(tmp_path).output == "did a thing"

    def test_external_tool_defaults_destructive(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            register_plugin_tools()
            # tool_danger sets no tool_destructive attr -> default True
            assert coder_tools.TOOL_REGISTRY[
                "plugin_issues_tool_danger"].destructive is True

    def test_non_callable_export_warns(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            names, warns = register_plugin_tools()
            assert "plugin_issues_not_a_func" not in names
            assert any("not_a_func" in w and "callable" in w for w in warns)

    def test_bad_arguments_contained_not_raised(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            register_plugin_tools()
            fn = coder_tools.TOOL_REGISTRY["plugin_issues_tool_echo"].fn
            r = fn(tmp_path, nope=1)        # unexpected kwarg
            assert not r.ok and "Bad arguments" in r.output

    def test_reregister_is_idempotent_no_clash(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            register_plugin_tools()
            names2, warns2 = register_plugin_tools()
            assert "plugin_issues_tool_echo" in names2
            assert not any("clash" in w for w in warns2)

    def test_plugin_without_exports_registers_nothing(self, tmp_path, monkeypatch):
        d = tmp_path / "plain"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nname = "plain"\nentry = "p:main"\n', encoding="utf-8")
        (d / "p.py").write_text(
            "import click\n@click.command()\ndef main():\n    pass\n",
            encoding="utf-8")
        monkeypatch.setattr("localm.plugins.loader.plugins_dir",
                            lambda: tmp_path)
        with patch.dict(coder_tools.TOOL_REGISTRY, {}, clear=False):
            names, _ = register_plugin_tools()
            assert names == []
