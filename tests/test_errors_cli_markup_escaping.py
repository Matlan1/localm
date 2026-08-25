# SPDX-License-Identifier: AGPL-3.0-or-later
"""`run_or_die` (localm/cli/errors.py) is the shared exception-to-exit-code
helper behind plugins.py/rag.py's error reporting: on ``KeyError`` it prints
*missing_msg* (or a generic fallback) in red; on ``ValueError`` it prints the
exception text in red. Both interpolate directly into a Rich
``Console.print()`` f-string, so a bracketed value silently loses its
bracketed span or has it consumed as a bogus style directive - the same bug
``tests/test_rag_cli_markup_escaping.py`` documents for ``rag.py``:

    Console().print('report[draft].txt')      -> prints "report.txt"
    Console().print('notes[bold red].md')     -> prints "notes.md"

*missing_msg* and the ``ValueError`` text are both caller/exception-supplied
and ``run_or_die`` itself has no way to guarantee either is restricted to a
safe character class for every caller.

``ValueError`` branch, real trigger: ``localm plugin install <name>`` runs
its target through ``PluginManager._installed_dir`` -> ``_check_plugin_name``
(localm/plugins/engine.py) BEFORE any "not found" check, and a name that
fails ``str.isidentifier()`` (which a bracketed name always does) raises
``ValueError(f"invalid plugin name: {name!r}")`` - the repr preserves any
bracket characters verbatim. So this is driven through the real CLI command,
matching test_rag_cli_markup_escaping.py's own convention of exercising real
code end to end.

``KeyError`` branch (``missing_msg``): the SAME identifier check runs before
any "no such plugin" KeyError can be raised by any current caller (plugins.py
builds ``missing_msg`` from the identical CLI-typed name), so a bracketed
``missing_msg`` cannot occur through any real command today - it is
defense-in-depth only, exactly like rag.py's already-validated collection
names. Tested directly against ``run_or_die`` with a synthetic click command,
the same convention test_cli_graceful_handler.py uses for the OTHER shared,
non-subcommand piece of CLI infrastructure (_GracefulGroup) - run_or_die is
infrastructure invoked BY commands, not a command with its own real-world
"missing_msg contains a bracket" trigger to hang a test off.
"""

from __future__ import annotations

import click
from click.testing import CliRunner

from localm.cli.errors import run_or_die

# One name Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes test_rag_cli_markup_escaping.py's docstring
# describes.
BRACKET_DROP_TEXT = "report[draft].txt"
BRACKET_STYLE_TEXT = "notes[bold red].md"


def _synthetic_group():
    """A purpose-built click group exercising run_or_die directly, real
    console.print included - the same shape test_cli_graceful_handler.py
    uses for _GracefulGroup."""
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
        # No missing_msg: exercises the `missing_msg or 'Not found'` fallback
        # expression itself, so the fix must not break the plain-literal path.
        def boom():
            raise KeyError("unused")
        run_or_die(boom)

    return g


class TestRunOrDieKeyErrorMissingMsgEscaping:
    """Defense-in-depth: no real caller can put a bracket in missing_msg
    today (see module docstring), but run_or_die cannot know that for a
    future caller, so it is escaped anyway and tested directly."""

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
    """Real end-to-end trigger: `localm plugin install <bracketed name>`
    (localm/cli/plugins.py:132-133) - see module docstring for the exact
    mechanism (_check_plugin_name's ValueError, raised before any KeyError
    path, repr-preserves the raw name)."""

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
