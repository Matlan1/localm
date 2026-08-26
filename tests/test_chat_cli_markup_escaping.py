# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file path, filename, or exception message shown by `localm run`'s chat CLI
must survive verbatim - Rich's ``Console.print()`` (and ``Panel``'s renderable,
and a bare ``console.print(some_string)`` with no literal markup around it at
all) parses ``[...]`` in ANY interpolated string as markup, not just inside a
command's own literal ``[style]`` tags:

    Console().print('report[draft].txt')      -> prints "report.txt"
    Console().print('notes[bold red].md')     -> prints "notes.md"

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently. In this file specifically: the interactive
`/image` command's queue confirmation and file-not-found error, `/images`'
listing, `/save`'s success/refusal/error messages, `localm run`'s "Model not
found" message, a streamed inference error, and the interactive banner's model
display name / system prompt line can all show a user a path or message that
differs from the real one - worst for `/image` and `/save`, where the path is
exactly what the user just typed and expects to see echoed back.

Every case here uses a REAL file on disk (or a real filesystem failure) under a
name/path that exercises the bug, rather than mocking the display layer. The two
exception-message sites that an ordinary filesystem state cannot trigger
(`/save`'s "Invalid save path" and a streamed RuntimeError) force the real
underlying call to raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# One name Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes described in the module docstring above.
BRACKET_DROP_NAME = "report[draft].txt"
BRACKET_STYLE_NAME = "notes[bold red].md"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console for every test in this module.

    rich.console.Console reads COLUMNS lazily on every render, so this applies
    even though chat.py's `console` is a module-level singleton built once at
    import. Without it the non-tty width default of 80 hard-wraps mid-word
    inside a long pytest basetemp path, splitting a path across two lines and
    breaking an exact-substring assertion."""
    monkeypatch.setenv("COLUMNS", "300")


@pytest.fixture
def chat_mod():
    from localm.cli import chat as chat_mod
    return chat_mod


class TestImageCommandMarkupEscaping:
    """The interactive `/image` command (chat.py's own /attach-style flow):
    queues a local image and echoes its name/path back to the user."""

    def test_queued_message_shows_bracket_drop_name_verbatim(
            self, chat_mod, tmp_path, capsys):
        f = tmp_path / BRACKET_DROP_NAME
        f.write_bytes(b"fake image bytes")
        pending: list = []

        stop = chat_mod._handle_command(f"/image {f}", [], {}, pending, engine=None)

        assert stop is False
        out = capsys.readouterr().out
        assert BRACKET_DROP_NAME in out, (
            f"the bracketed filename must survive verbatim in the /image queue "
            f"confirmation, not be silently mangled by Rich markup: {out!r}")
        assert pending == [str(f.resolve())]

    def test_queued_message_shows_bracket_style_name_verbatim(
            self, chat_mod, tmp_path, capsys):
        f = tmp_path / BRACKET_STYLE_NAME
        f.write_bytes(b"fake image bytes")
        pending: list = []

        chat_mod._handle_command(f"/image {f}", [], {}, pending, engine=None)

        out = capsys.readouterr().out
        assert BRACKET_STYLE_NAME in out, (
            f"a filename segment that happens to look like a style tag must be "
            f"shown as literal text, not consumed as Rich styling: {out!r}")

    def test_file_not_found_shows_bracketed_arg_verbatim(
            self, chat_mod, tmp_path, capsys):
        missing = str(tmp_path / BRACKET_STYLE_NAME)

        chat_mod._handle_command(f"/image {missing}", [], {}, [], engine=None)

        out = capsys.readouterr().out
        assert missing in out, (
            f"the 'File not found' error must echo the exact path the user "
            f"typed, verbatim: {out!r}")

    def test_images_listing_shows_bracketed_path_verbatim(
            self, chat_mod, tmp_path, capsys):
        f = tmp_path / BRACKET_DROP_NAME
        f.write_bytes(b"data")
        pending = [str(f.resolve())]

        chat_mod._handle_command("/images", [], {}, pending, engine=None)

        out = capsys.readouterr().out
        assert str(f.resolve()) in out, (
            f"the /images listing must show each queued path verbatim: {out!r}")


class TestSaveCommandMarkupEscaping:
    """The interactive `/save` command: writes the conversation to JSON,
    confined to the cwd, and echoes the target path back."""

    def test_saved_confirmation_shows_bracketed_filename_verbatim(
            self, chat_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        messages = [{"role": "user", "content": "hi"}]

        chat_mod._handle_command(
            f"/save {BRACKET_DROP_NAME}", messages, {}, [], engine=None)

        out = capsys.readouterr().out
        assert BRACKET_DROP_NAME in out, (
            f"the 'Saved:' confirmation must echo the real filename verbatim: "
            f"{out!r}")
        saved = tmp_path / BRACKET_DROP_NAME
        assert saved.exists()
        assert json.loads(saved.read_text(encoding="utf-8")) == messages

    def test_refusing_outside_cwd_shows_bracketed_target_verbatim(
            self, chat_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        target = f"../{BRACKET_STYLE_NAME}"

        chat_mod._handle_command(f"/save {target}", [], {}, [], engine=None)

        out = capsys.readouterr().out
        assert target in out, (
            f"the 'Refusing to save outside...' message must echo the exact "
            f"target the user typed, verbatim: {out!r}")

    def test_invalid_save_path_exception_escaped(
            self, chat_mod, tmp_path, monkeypatch, capsys):
        """`(cwd / target).resolve()` essentially never fails on an ordinary
        path, so the real call is forced to raise."""
        monkeypatch.chdir(tmp_path)
        bad_message = "Invalid save path: drive[legacy]spec"
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **kw):
            if self.name == "trigger.json":
                raise OSError(bad_message)
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

        chat_mod._handle_command("/save trigger.json", [], {}, [], engine=None)

        out = capsys.readouterr().out
        assert bad_message in out, (
            f"an 'Invalid save path' exception message containing '[...]' must "
            f"survive verbatim, not be parsed as markup: {out!r}")

    def test_save_failed_exception_shows_bracketed_path_verbatim(
            self, chat_mod, tmp_path, monkeypatch, capsys):
        """A REAL, naturally-occurring failure (no monkeypatch): the target's
        parent directory does not exist, so open(..., "w") raises with the
        full path - which contains the bracketed segment - in its own
        message."""
        monkeypatch.chdir(tmp_path)
        target = f"missing_dir[legacy]/{BRACKET_DROP_NAME}"

        chat_mod._handle_command(f"/save {target}", [], {}, [], engine=None)

        out = capsys.readouterr().out
        assert "[legacy]" in out, (
            f"a 'Save failed' exception message containing '[...]' (here, from "
            f"the OS's own error text naming the missing directory) must "
            f"survive verbatim: {out!r}")
        assert not (tmp_path / "missing_dir[legacy]").exists()


class TestRunModelNotFoundMarkupEscaping:
    def test_model_not_found_shows_bracketed_name_verbatim(self, cli_runner):
        from localm.cli.chat import run

        bad_model = "does-not-exist[legacy]"
        result = cli_runner.invoke(run, [bad_model, "--no-server", "-p", "hi"])

        assert result.exit_code == 1, result.output
        assert bad_model in result.output, (
            f"'Model not found:' must echo the exact operator-typed name "
            f"verbatim: {result.output!r}")


class TestStreamingErrorMarkupEscaping:
    def test_runtime_error_from_stream_survives_verbatim(
            self, cli_runner, tmp_path, monkeypatch):
        """A RuntimeError from an attached server's stream (no model loaded,
        unreachable, ...), forced via a mocked Engine, since a real server error
        message is not otherwise producible from a unit test."""
        model_f = tmp_path / "some-model.gguf"
        model_f.write_bytes(b"GGUF")

        engine_instance = MagicMock(name="EngineInstance")
        engine_instance.__enter__ = MagicMock(return_value=engine_instance)
        engine_instance.__exit__ = MagicMock(return_value=False)
        # A digit-only bracket body ("[404]") is NOT recognised by Rich's own
        # tag grammar and survives even unescaped, so it cannot exercise this
        # bug; the body here is a lowercase word, like
        # BRACKET_DROP_NAME/BRACKET_STYLE_NAME.
        bad_message = "attached server error: model[legacy].bin"
        engine_instance.chat_stream = MagicMock(side_effect=RuntimeError(bad_message))
        engine_cls = MagicMock(name="Engine", return_value=engine_instance)

        monkeypatch.setattr("localm.inference.engine.Engine", engine_cls)
        monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)

        from localm.cli.chat import run
        result = cli_runner.invoke(run, [str(model_f), "--no-server", "-p", "hi"])

        assert result.exit_code == 0, result.output
        assert bad_message in result.output, (
            f"a streamed RuntimeError containing '[...]' must survive "
            f"verbatim, not be parsed as markup: {result.output!r}")


class TestInteractiveBannerMarkupEscaping:
    """`_interactive()`'s opening Panel embeds the engine's display name (a
    Rich ``Panel`` given a plain string also parses it as markup), and the
    system-prompt line embeds the operator-supplied -s/--system text.
    `_interactive()` is driven directly with `console.input` forced to raise
    EOFError immediately, so the loop exits right after printing the
    banner/system line."""

    def _run_banner_only(self, chat_mod, monkeypatch, *, display_name,
                         system_prompt):
        engine = MagicMock(name="Engine", display_name=display_name)

        def _eof(*a, **kw):
            raise EOFError()
        monkeypatch.setattr(chat_mod.console, "input", _eof)

        chat_mod._interactive(engine, system_prompt, {})

    def test_panel_shows_bracketed_display_name_verbatim(
            self, chat_mod, monkeypatch, capsys):
        self._run_banner_only(
            chat_mod, monkeypatch,
            display_name=BRACKET_DROP_NAME, system_prompt=None)

        out = capsys.readouterr().out
        assert BRACKET_DROP_NAME in out, (
            f"the interactive banner Panel must show the model's display name "
            f"verbatim, not silently drop the bracketed span: {out!r}")

    def test_system_prompt_line_shows_bracketed_text_verbatim(
            self, chat_mod, monkeypatch, capsys):
        prompt = f"Reply like {BRACKET_STYLE_NAME}"
        self._run_banner_only(
            chat_mod, monkeypatch,
            display_name="plain-model", system_prompt=prompt)

        out = capsys.readouterr().out
        assert prompt in out, (
            f"the 'system: ...' line must echo the -s/--system text verbatim, "
            f"not have a bracketed segment consumed as styling: {out!r}")
