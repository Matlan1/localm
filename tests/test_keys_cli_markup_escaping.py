# SPDX-License-Identifier: AGPL-3.0-or-later
"""A key's name, id, filesystem paths, and error/warning text shown by the
`key` CLI must survive verbatim - Rich's ``Console.print()``, and
``Table.add_row()`` too, parse ``[...]`` in ANY interpolated string as markup,
not just inside a command's own literal ``[style]`` tags:

    Console().print('report[draft].txt')       -> prints "report.txt"
    Console().print('notes[bold red].md')      -> prints "notes.md"
    Table().add_row('report[draft].txt')        -> same corruption in a cell

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently. `key create`'s confirmation, `key list`'s
table, `key show`, `key clear`/`key recover`'s failure paths, and `key rm`
can all show text that differs from the real value - sharpest for a key's
NAME (``key create <name>``), which ``create_key()`` only strips and never
restricts to a safe character class, and for a ``--rag-root`` folder path,
which ``norm_rag_roots()`` treats the same way.

Some sites are provably safe today (a generated key/id's charset, the
``--fs-access`` ``click.Choice``, a validated scope) and are escaped anyway.
Those sites are exercised here by monkeypatching the value that would normally
be safe, so the escape() wrapper at that print site is proven rather than the
upstream guarantee assumed.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from localm.cli import main

# One name Rich DROPS outright, one it consumes as a (bogus) style tag: the
# two distinct failure shapes.
BRACKET_DROP = "[draft]"
BRACKET_STYLE = "[bold red]"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console for every test in this module.

    Only ONE fixture in this file (bracketed_home_runner) ever set this, so
    every test using the plain `runner` fixture instead - including the
    whole of TestKeyListMarkupEscaping - had no width protection at all."""
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="300")


@pytest.fixture
def runner(cli_runner, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    return cli_runner


@pytest.fixture
def bracketed_home_runner(tmp_path, monkeypatch):
    """A cli_runner (conftest.py) equivalent whose LOCALM_HOME directory NAME
    itself contains a bracket pair, so any path built from home_dir()
    (key_file(), sessions_file()) exercises the bug without fabricating
    anything."""
    home = tmp_path / f"data{BRACKET_DROP}"
    home.mkdir(parents=True, exist_ok=True)
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    return CliRunner()


class TestKeyShowMarkupEscaping:
    def test_reveal_shows_bracketed_key_verbatim(self, runner, monkeypatch):
        """get_api_key() reads LOCALM_API_KEY directly: unlike `key set`, this
        path bypasses set_api_key's _KEY_CHARSET gate entirely, so it is
        reachable today rather than defense-in-depth."""
        forced = f"owner-secret-key{BRACKET_STYLE}-padding-1234567890"
        monkeypatch.setenv("LOCALM_API_KEY", forced)
        r = runner.invoke(main, ["key", "show", "--reveal"])
        assert r.exit_code == 0, r.output
        assert forced in r.output

    def test_masked_preview_shows_bracketed_prefix_verbatim(self, runner, monkeypatch):
        """_mask_key() shows only key[:4] + '...' + key[-4:], so the bracket
        pair must fit ENTIRELY inside that 4-char window to reach Rich's parser
        as a complete "[...]" tag. A longer tag like '[draft]' is truncated to
        '[dra', with no closing bracket, and passes through unchanged."""
        from localm.cli.keys import _mask_key
        forced = "[dr]restofthekeypaddingtoreach1234567890abcdef"
        monkeypatch.setenv("LOCALM_API_KEY", forced)
        expected = _mask_key(forced)
        assert "[dr]" in expected, "fixture must put a COMPLETE tag in the mask window"
        r = runner.invoke(main, ["key", "show"])
        assert r.exit_code == 0, r.output
        assert forced not in r.output               # still masked, not the full key
        assert expected in r.output, (
            f"the masked preview must survive verbatim, not be mangled by Rich "
            f"markup parsing: {r.output!r}")


class TestKeyGenerateMarkupEscaping:
    def test_generated_key_survives_verbatim_if_it_ever_contained_brackets(
            self, runner, monkeypatch):
        """generate_key()'s charset (secrets.token_urlsafe) can never produce
        '[...]', so this is defense-in-depth: forcing regenerate_key()'s return
        value proves the escape() wrapper at this print site works."""
        from localm import auth
        forced = f"forced-owner-key{BRACKET_STYLE}-padding-1234567890"
        monkeypatch.setattr(auth, "regenerate_key", lambda nbytes=32: forced)
        r = runner.invoke(main, ["key", "generate"])
        assert r.exit_code == 0, r.output
        assert forced in r.output


class TestKeySetMarkupEscaping:
    def test_masked_preview_survives_verbatim_if_charset_gate_is_bypassed(
            self, runner, monkeypatch):
        """The normal path can never reach this print with a bracketed key:
        set_api_key's _KEY_CHARSET gate raises first. Bypassing that gate here
        proves the escape() wrapper at THIS print site holds on its own."""
        from localm import auth
        from localm.cli.keys import _mask_key
        # _mask_key() shows only key[:4] + '...' + key[-4:], so the bracket
        # pair must fit ENTIRELY inside that 4-char window to reach Rich's
        # parser as a complete tag. A bracket placed mid-string is truncated
        # away by the mask itself.
        forced = "[dr]restofthekeypaddingtoreach1234567890abcdef"
        monkeypatch.setattr(auth, "set_api_key", lambda key: None)
        expected = _mask_key(forced)
        assert "[dr]" in expected, "fixture must put a COMPLETE tag in the mask window"
        r = runner.invoke(main, ["key", "set", forced])
        assert r.exit_code == 0, r.output
        assert expected in r.output


class TestKeyClearMarkupEscaping:
    def test_failed_clear_shows_bracketed_path_and_error_verbatim(
            self, bracketed_home_runner, monkeypatch):
        from pathlib import Path

        from localm import auth
        bracketed_home_runner.invoke(main, ["key", "generate"])
        assert auth.get_api_key() is not None

        real_unlink = Path.unlink
        key_name = auth.key_file().name
        err_text = f"cannot access {BRACKET_STYLE} locked file"

        def fake_unlink(self, *a, **kw):
            if self.name == key_name:
                raise PermissionError(err_text)
            return real_unlink(self, *a, **kw)
        monkeypatch.setattr(Path, "unlink", fake_unlink)

        r = bracketed_home_runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0, r.output
        assert "not fully cleared" in r.output.lower()
        assert str(auth.key_file()) in r.output, (
            f"the bracketed HOME-derived path must survive verbatim: {r.output!r}")
        assert err_text in r.output, (
            f"the forced OS error text must survive verbatim: {r.output!r}")

    def test_sessions_not_revoked_message_shows_bracketed_path_verbatim(
            self, bracketed_home_runner, monkeypatch):
        from localm import sessions
        bracketed_home_runner.invoke(main, ["key", "generate"])
        monkeypatch.setattr(sessions, "revoke_all", lambda: None)

        r = bracketed_home_runner.invoke(main, ["key", "clear", "--yes"])
        assert r.exit_code == 0, r.output
        assert "not signed out" in r.output.lower()
        assert str(sessions.sessions_file()) in r.output, (
            f"the bracketed HOME-derived sessions path must survive verbatim: "
            f"{r.output!r}")


class TestKeyRecoverMarkupEscaping:
    def test_recovered_key_survives_verbatim_if_it_ever_contained_brackets(
            self, runner, monkeypatch):
        from localm import auth
        forced = f"recovered-owner-key{BRACKET_DROP}-padding-1234567890"
        monkeypatch.setattr(auth, "regenerate_key", lambda nbytes=32: forced)
        r = runner.invoke(main, ["key", "recover"])
        assert r.exit_code == 0, r.output
        assert forced in r.output

    def test_sessions_not_revoked_message_shows_bracketed_path_verbatim(
            self, bracketed_home_runner, monkeypatch):
        from localm import sessions
        monkeypatch.setattr(sessions, "revoke_all", lambda: None)

        r = bracketed_home_runner.invoke(main, ["key", "recover"])
        assert r.exit_code == 0, r.output
        assert str(sessions.sessions_file()) in r.output, (
            f"the bracketed HOME-derived sessions path must survive verbatim: "
            f"{r.output!r}")


class TestKeyCreateMarkupEscaping:
    def test_bracket_drop_name_survives_verbatim(self, runner):
        name = f"dashboard{BRACKET_DROP}"
        r = runner.invoke(main, ["key", "create", name, "--scope", "models:read"])
        assert r.exit_code == 0, r.output
        assert name in r.output, (
            f"a key name Rich would DROP must survive verbatim: {r.output!r}")

    def test_bracket_style_name_survives_verbatim(self, runner):
        name = f"dashboard{BRACKET_STYLE}"
        r = runner.invoke(main, ["key", "create", name, "--scope", "models:read"])
        assert r.exit_code == 0, r.output
        assert name in r.output, (
            f"a key name Rich would consume as styling must survive verbatim: "
            f"{r.output!r}")

    def test_bracketed_rag_root_survives_verbatim(self, runner):
        """norm_rag_roots() only de-dupes and strips whitespace, so a
        --rag-root value is never restricted to a safe character class and this
        is reachable today, like the key name above."""
        root = f"C:/docs/{BRACKET_STYLE}"
        r = runner.invoke(main, ["key", "create", "scoped",
                                 "--scope", "rag", "--rag-root", root])
        assert r.exit_code == 0, r.output
        assert root in r.output, (
            f"a bracketed --rag-root value must survive verbatim in the "
            f"creation confirmation: {r.output!r}")

    def test_unknown_scope_error_shows_bracketed_scope_verbatim(self, runner):
        """create_key() raises ValueError('Unknown scope(s): {bad}') with the
        caller's raw --scope text, which FAILED validation and is not restricted
        to any safe character class.

        No colon in this fixture: Rich's Console also substitutes ':shortcode:'
        text as emoji, which rich.markup.escape() does not address, and
        'not:a:scope' collides with the ':a:' shortcode."""
        bad_scope = f"not-real-scope{BRACKET_DROP}"
        r = runner.invoke(main, ["key", "create", "x", "--scope", bad_scope])
        assert r.exit_code == 1, r.output
        assert bad_scope in r.output, (
            f"the unknown-scope error must echo the caller's text verbatim: "
            f"{r.output!r}")

    def test_plugin_dependency_warning_survives_verbatim(self, runner, monkeypatch):
        """scope_deps_warnings() draws from an installed plugin's own manifest
        fields (name/scope), which are not restricted to a safe character class.
        Forced directly rather than by fabricating a real installed plugin with
        a bracketed manifest."""
        from localm.plugins.engine import PluginManager
        warning = f"key grants 'models:read' but demo{BRACKET_STYLE} is not installed"
        monkeypatch.setattr(PluginManager, "scope_deps_warnings",
                            lambda self, granted: [warning])
        r = runner.invoke(main, ["key", "create", "x", "--scope", "models:read"])
        assert r.exit_code == 0, r.output
        assert warning in r.output, (
            f"a plugin dependency warning must survive verbatim: {r.output!r}")


class TestKeyListMarkupEscaping:
    def test_bracket_drop_name_survives_verbatim_in_table(self, runner):
        name = f"dashboard{BRACKET_DROP}"
        runner.invoke(main, ["key", "create", name, "--scope", "models:read"])
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0, r.output
        assert name in r.output, (
            f"a key name Table.add_row() would DROP must survive verbatim: "
            f"{r.output!r}")

    def test_bracket_style_name_survives_verbatim_in_table(self, runner):
        name = f"dashboard{BRACKET_STYLE}"
        runner.invoke(main, ["key", "create", name, "--scope", "models:read"])
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0, r.output
        assert name in r.output, (
            f"a key name Table.add_row() would consume as styling must "
            f"survive verbatim: {r.output!r}")

    def test_bracketed_rag_root_survives_verbatim_in_table(self, runner):
        root = f"C:/docs/{BRACKET_STYLE}"
        runner.invoke(main, ["key", "create", "scoped", "--scope", "rag",
                             "--rag-root", root])
        r = runner.invoke(main, ["key", "list"])
        assert r.exit_code == 0, r.output
        assert root in r.output, (
            f"a bracketed RAG-roots table cell must survive verbatim: "
            f"{r.output!r}")


class TestKeyRmMarkupEscaping:
    def test_revoked_confirmation_shows_bracketed_id_verbatim(self, runner, monkeypatch):
        """key_id is an unvalidated CLI argument, and auth.revoke_key() only
        compares it by equality against stored (always-hex) ids, so a real id
        can never contain brackets. Forced True here so the escape() wrapper on
        the success message is proven for whatever text a caller passes."""
        from localm import auth
        bracketed_id = f"totally-made-up-id{BRACKET_DROP}"
        monkeypatch.setattr(auth, "revoke_key", lambda key_id: True)
        r = runner.invoke(main, ["key", "rm", bracketed_id, "--yes"])
        assert r.exit_code == 0, r.output
        assert bracketed_id in r.output, (
            f"the revoked-confirmation message must echo the id verbatim: "
            f"{r.output!r}")

    def test_unknown_key_id_error_shows_bracketed_id_verbatim(self, runner):
        bracketed_id = f"totally-made-up-id{BRACKET_STYLE}"
        r = runner.invoke(main, ["key", "rm", bracketed_id, "--yes"])
        assert r.exit_code == 1, r.output
        assert bracketed_id in r.output, (
            f"the 'No such key' error must echo the caller's id verbatim: "
            f"{r.output!r}")
