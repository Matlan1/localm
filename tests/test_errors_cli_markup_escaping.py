# SPDX-License-Identifier: AGPL-3.0-or-later
"""`run_or_die` (localm/cli/errors.py) is the shared exception-to-exit-code
helper behind plugins.py/rag.py's error reporting: on ``KeyError`` it prints
*missing_msg* (or a generic fallback) in red; on ``ValueError`` it prints the
exception text in red. Both interpolate into a Rich ``Console.print()``
f-string, where a bracketed value loses its bracketed span or has it consumed
as a style directive:

    Console().print('report[draft].txt')      -> prints "report.txt"
    Console().print('notes[bold red].md')     -> prints "notes.md"

``ValueError`` branch: ``localm plugin install <name>`` runs its target
through ``PluginManager._installed_dir`` -> ``_check_plugin_name``
(localm/plugins/engine.py) before any "not found" check, and a name that
fails ``str.isidentifier()`` raises
``ValueError(f"invalid plugin name: {name!r}")``, whose repr keeps any
bracket characters verbatim. Driven through the real CLI command.

``KeyError`` branch (``missing_msg``): the same identifier check runs before
any "no such plugin" KeyError can be raised by any current caller, so a
bracketed ``missing_msg`` cannot occur through a real command. Tested
directly against ``run_or_die`` with a synthetic click command.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from localm.cli.errors import run_or_die

# One name Rich drops outright, one it consumes as a style tag.
BRACKET_DROP_TEXT = "report[draft].txt"
BRACKET_STYLE_TEXT = "notes[bold red].md"


def _synthetic_group():
    """A click group exercising run_or_die directly, real console.print
    included."""
    @click.group()
    def g():
        pass

    @g.command("keyerror-drop")
    def keyerror_drop():
        def boom():
            raise KeyError("unused")
        run_or_die(boom, missing_msg=f"No such thing: {BRACKET_DROP_TEXT}")

    @g.command("keyerror-style")
    def keyerror_style():
        def boom():
            raise KeyError("unused")
        run_or_die(boom, missing_msg=f"No such thing: {BRACKET_STYLE_TEXT}")

    @g.command("keyerror-default")
    def keyerror_default():
        # No missing_msg: exercises the `missing_msg or 'Not found'` fallback.
        def boom():
            raise KeyError("unused")
        run_or_die(boom)

    return g


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console for every test in this module.

    rich.console.Console.size returns 80x25 outright on a dumb terminal,
    before it ever consults COLUMNS - patching is_dumb_terminal is what makes
    the COLUMNS override below actually take effect."""
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="300")


class TestRunOrDieKeyErrorMissingMsgEscaping:
    """No current caller can put a bracket in missing_msg; run_or_die escapes
    it regardless, exercised here directly."""

    def test_bracket_drop_missing_msg_survives_verbatim(self):
        r = CliRunner().invoke(_synthetic_group(), ["keyerror-drop"])
        assert r.exit_code == 1
        assert BRACKET_DROP_TEXT in r.output, (
            f"a bracketed missing_msg must survive verbatim, not be silently "
            f"dropped by Rich markup parsing: {r.output!r}")

    def test_bracket_style_missing_msg_survives_verbatim(self):
        r = CliRunner().invoke(_synthetic_group(), ["keyerror-style"])
        assert r.exit_code == 1
        assert BRACKET_STYLE_TEXT in r.output, (
            f"a missing_msg segment that looks like a style tag must be "
            f"shown as literal text, not consumed as Rich styling: "
            f"{r.output!r}")

    def test_default_missing_msg_fallback_still_works(self):
        r = CliRunner().invoke(_synthetic_group(), ["keyerror-default"])
        assert r.exit_code == 1
        assert "Not found" in r.output


class TestRunOrDieValueErrorRealCliPath:
    """End-to-end trigger: `localm plugin install <bracketed name>`, which
    raises _check_plugin_name's ValueError before any KeyError path."""

    def _cli_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

        store = tmp_path / "store"
        installed = tmp_path / "installed"
        store.mkdir()
        installed.mkdir()

        import localm.cli as climod
        from localm.plugins.engine import PluginManager
        monkeypatch.setattr(
            climod, "_engine_manager",
            lambda: PluginManager(None, store_root=store, installed_root=installed))
        return climod.main

    def test_bracket_drop_plugin_name_survives_verbatim(self, tmp_path, monkeypatch):
        main = self._cli_env(tmp_path, monkeypatch)
        r = CliRunner().invoke(main, ["plugin", "install", BRACKET_DROP_TEXT])
        assert r.exit_code == 1, r.output
        assert BRACKET_DROP_TEXT in r.output, (
            f"an invalid plugin name error must echo the exact name the "
            f"user typed, verbatim: {r.output!r}")

    def test_bracket_style_plugin_name_survives_verbatim(self, tmp_path, monkeypatch):
        main = self._cli_env(tmp_path, monkeypatch)
        r = CliRunner().invoke(main, ["plugin", "install", BRACKET_STYLE_TEXT])
        assert r.exit_code == 1, r.output
        assert BRACKET_STYLE_TEXT in r.output, (
            f"a plugin name segment that looks like a style tag must be "
            f"shown as literal text, not consumed as Rich styling: "
            f"{r.output!r}")
