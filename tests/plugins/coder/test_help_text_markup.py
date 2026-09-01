# SPDX-License-Identifier: AGPL-3.0-or-later
"""HELP_TEXT is printed with Rich markup enabled, so an unescaped literal
``[...]`` in an argument hint (``/diff [path]``) is parsed as a style tag
instead of shown as text: Rich silently drops it rather than raising, so the
placeholder just vanishes from the rendered /help output.
"""

from rich.console import Console

from localm.plugins.coder.display import HELP_TEXT, print_help


def _rendered(text: str) -> str:
    console = Console(record=True, width=200, highlight=False)
    console.print(text)
    return console.export_text()


class TestHelpTextArgumentPlaceholdersSurviveMarkup:
    def test_diff_path_placeholder_survives(self):
        assert "/diff [path]" in _rendered(HELP_TEXT)

    def test_export_path_placeholder_survives(self):
        assert "/export [path]" in _rendered(HELP_TEXT)

    def test_resume_id_placeholder_survives(self):
        assert "/resume [id]" in _rendered(HELP_TEXT)

    def test_scope_glob_placeholder_survives(self):
        assert "/scope [glob]" in _rendered(HELP_TEXT)

    def test_verify_cmd_auto_off_placeholder_survives(self):
        assert "/verify [cmd|auto|off]" in _rendered(HELP_TEXT)

    def test_goal_cmd_auto_off_placeholder_survives(self):
        assert "/goal [cmd|auto|off]" in _rendered(HELP_TEXT)

    def test_bold_and_dim_tags_still_style_rather_than_print_literally(self):
        # A wrong fix (e.g. disabling markup entirely) would "solve" the
        # stripped placeholders by making every tag literal, including the
        # ones that must keep working.
        out = _rendered(HELP_TEXT)
        assert "[bold]" not in out
        assert "[/bold]" not in out
        assert "[bold cyan]" not in out
        assert "[dim]" not in out
        assert "localcoder commands" in out

    def test_print_help_prints_the_fixed_help_text_on_the_real_console(self, capsys):
        print_help()
        out = capsys.readouterr().out
        assert "[path]" in out
        assert "[id]" in out
        assert "[glob]" in out
        assert "[cmd|auto|off]" in out
