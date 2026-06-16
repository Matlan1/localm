"""Static tripwire for the slash-command menu (B2).

Bug (from the field report): typing "/remember some note" then Enter discarded
the args. The slash autocomplete menu filtered on the first token only and never
closed once the user typed past the command, so Enter triggered the menu's
pick(), which overwrites the input with "/cmd " and drops the typed args instead
of sending the whole line.

Fix: attachSlashMenu's render() closes the menu as soon as a space is typed (the
command token is complete and the user is entering args), so Enter falls through
to the send path and handleSlashSubmit parses "/cmd args" correctly.

The GUI has no JS test harness, so this is a source-level guard: it fails if the
close-on-space behavior is removed. The behavioral proof is the live GUI test."""

import re
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[1]
          / "localm" / "plugins" / "gui" / "static" / "app.js")


def _func_body(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace:i + 1]
    raise AssertionError(f"unbalanced braces in function {name}")


def _strip_line_comments(src: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in src.splitlines())


def test_slash_menu_closes_once_past_the_command_token():
    """attachSlashMenu must close the menu when the input has a space after the
    command (args being typed), so Enter sends the full line instead of pick()
    discarding the args."""
    body = _strip_line_comments(_func_body(APP_JS.read_text(encoding="utf-8"),
                                           "attachSlashMenu"))
    # A guard that, when the post-"/" text contains a space, closes the menu.
    assert re.search(r'includes\("\s"\)\s*\)\s*\{\s*close\(\)', body), (
        "slash menu must close once a space is typed past the command token; "
        "without it Enter calls pick() and discards typed args (B2)")
