# SPDX-License-Identifier: AGPL-3.0-or-later
"""The user must be able to scroll up to re-read earlier text while a reply is
still streaming.

The CLI chat renderers stream append-only (plain print / rich
console.print(end="")) with only SGR styling escapes: no alternate screen, no
cursor repositioning, no scroll-region, no spinner/Live region. With nothing
repositioning the viewport, the terminal emulator's own scrollback gives the
user scroll priority.

These tests pin that property: CLI streaming must emit no viewport-control
sequence. A negative-control test proves the detector actually catches such
sequences.
"""

from __future__ import annotations

import io
import re
import sys

from localm.cli import _ThinkPrinter
from localm.plugins.coder import display


# Terminal control sequences that REPOSITION/erase the viewport. EXCLUDES SGR
# styling (sequences ending in 'm' - dim/colour/reset), which is legitimate and
# expected on a TTY, and a bare carriage return, which model text may legitimately
# contain.
_VIEWPORT_CTRL = re.compile(
    r"\x1b\[\?[0-9;]*[hl]"               # private modes: alt-screen (?1049/?47), hide-cursor (?25l) = Live/Status fingerprint
    r"|\x1b\[[0-9;]*[ABCDEFGHJKSTfsu]"   # cursor move / erase line+screen / scroll / absolute-position / save+restore
    r"|\x1b\[[0-9;]*r"                   # scroll region (DECSTBM)
    r"|\x1b[78]"                         # DECSC / DECRC cursor save/restore
)


def _has_viewport_control(text: str) -> bool:
    return _VIEWPORT_CTRL.search(text) is not None


class _FakeTTY:
    """A stdout stand-in that reports as a TTY so the renderer takes its real
    escape-emitting branch, while we capture exactly what would hit the terminal."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, s: str) -> int:
        self._parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        return "".join(self._parts)


def test_cli_think_printer_streams_without_viewport_control(monkeypatch):
    """The `localm run` / REPL token emitter (_ThinkPrinter) on a TTY: emits dim styling
    for <think> but never a viewport-repositioning sequence."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)

    printer = _ThinkPrinter()
    # On the real TTY (escape-emitting) branch, not the plain no-op branch.
    assert printer._dim == "\x1b[2m"

    for tok in ["Here is ", "line one\n", "<think>", "some reasoning\n",
                "</think>", "line two\n", "and the final line"]:
        printer.feed(tok)
    printer.flush()

    out = fake.getvalue()
    # The stream content is really there (we captured the right path)...
    assert "line one" in out and "line two" in out and "and the final line" in out
    # ...and the ONLY escapes are styling - nothing repositions the viewport.
    assert not _has_viewport_control(out), repr(out)


def test_coder_stream_renderer_streams_without_viewport_control(monkeypatch):
    """The coder REPL token emitter (display.print_streaming_token) on a forced terminal:
    rich emits styling but no viewport-repositioning sequence."""
    from rich.console import Console

    buf = io.StringIO()
    fake_console = Console(force_terminal=True, file=buf, highlight=False, width=80)
    monkeypatch.setattr(display, "console", fake_console)

    for tok in ["building ", "the answer\n", "across ", "two lines\n"]:
        display.print_streaming_token(tok)
    display.print_streaming_done()

    out = buf.getvalue()
    assert "the answer" in out and "two lines" in out
    assert not _has_viewport_control(out), repr(out)


def test_viewport_control_detector_is_not_vacuous():
    """NEGATIVE CONTROL: the detector must catch the sequences a Live region /
    alt-screen / spinner / \\r-redraw would emit, and must ignore benign SGR
    styling."""
    must_flag = [
        "\x1b[?1049h",   # enter alternate screen
        "\x1b[?1049l",   # leave alternate screen
        "\x1b[?47h",     # legacy alternate screen
        "\x1b[?25l",     # hide cursor (rich Live/Status/Progress fingerprint)
        "\x1b[H",        # cursor home
        "\x1b[10;5H",    # cursor absolute position
        "\x1b[12;3f",    # cursor absolute position (HVP form)
        "\x1b[2;20r",    # set scroll region
        "\x1b[2K",       # erase line (spinner redraw)
        "\x1b[2J",       # erase display
        "\x1b[A",        # cursor up
        "\x1b[3F",       # cursor up 3 lines, to column 0
        "\x1b[s",        # save cursor
        "\x1b7",         # DECSC save cursor
    ]
    for seq in must_flag:
        assert _has_viewport_control("text " + seq + " more"), f"missed: {seq!r}"

    must_ignore = [
        "\x1b[2mdim text\x1b[22m",      # dim on/off (what _ThinkPrinter uses)
        "\x1b[0m",                       # SGR reset
        "\x1b[31mred\x1b[0m",            # colour
        "\x1b[1;32mbold green\x1b[0m",   # bold + colour
        "plain text\nwith a newline\n",  # ordinary streamed content
        "100% complete",                 # a bare digit before a letter, no escape
    ]
    for seq in must_ignore:
        assert not _has_viewport_control(seq), f"false positive: {seq!r}"
