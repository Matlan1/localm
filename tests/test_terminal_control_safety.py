# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text localm did not author (model output, server replies, GGUF metadata,
remote search results, exception messages) reaches the terminal without its
control characters, and without being parsed as Rich markup where it is
interpolated into a markup string."""

import ast
import io
import sys
from pathlib import Path

import pytest
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

import localm.console as lc

HOSTILE = "a\x1b[2Jb\x1bcc\x9b31md\x07e\x1b]52;c;cGF5bG9hZA==\x07f\rg\x7fh"
ROOT = Path(__file__).resolve().parent.parent / "localm"


def _console():
    buf = io.StringIO()
    return lc.SafeConsole(file=buf, force_terminal=True, color_system="truecolor",
                       width=200), buf


def _plain_console():
    buf = io.StringIO()
    return lc.SafeConsole(file=buf, force_terminal=True, color_system=None,
                       highlight=False, width=200), buf


def test_strip_controls_keeps_newline_tab_and_text():
    assert lc.strip_controls(HOSTILE) == "a[2Jbcc31mde]52;c;cGF5bG9hZA==fgh"
    assert lc.strip_controls("line\n\tindent é中") == "line\n\tindent é中"


@pytest.mark.parametrize("make", [
    lambda s: escape(s),
    lambda s: Text(s),
    lambda s: Syntax(f"+{s}\n-{s}\n", "diff", theme="monokai"),
    lambda s: Panel(Text(s)),
    lambda s: (lambda t: (t.add_column("c"), t.add_row(Text(s)), t)[-1])(Table()),
], ids=["str", "Text", "Syntax", "Panel", "Table"])
def test_safe_console_renders_no_hostile_control_sequence(make):
    con, buf = _console()
    con.print(make(HOSTILE))
    out = buf.getvalue()
    for needle in ("\x1b[2J", "\x1bc", "\x9b", "\x07", "\x1b]52", "\x7f", "\r"):
        assert needle not in out, (needle, out)


def test_safe_console_keeps_its_own_styling():
    con, buf = _console()
    con.print("[bold]x[/bold]")
    assert "\x1b[1m" in buf.getvalue()


def test_a_carriage_return_line_end_still_redraws():
    con, buf = _console()
    con.print("10%", end="\r")
    con.print(escape("x\ry"), end="\r")
    out = buf.getvalue()
    assert out.count("\r") == 2 and "xy" in out


def test_markup_safe_text_is_printed_literally():
    con, buf = _plain_console()
    con.print(f"[red]Reason:[/red] {lc.markup_safe('[/] [link=http://x.example]y')}")
    assert "[/] [link=http://x.example]y" in buf.getvalue()


def test_every_console_in_localm_is_a_safe_console():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == ROOT / "console.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rich.console":
                if any(a.name == "Console" for a in node.names):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == [], offenders


def test_streamed_chat_tokens_reach_a_terminal_without_controls(monkeypatch, capsys):
    from localm.cli.chat import _ThinkPrinter
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    printer = _ThinkPrinter()
    printer.feed(HOSTILE)
    printer.flush()
    out = capsys.readouterr().out
    assert "\x1b[2J" not in out and "\x1bc" not in out and "\x9b" not in out
    assert "a[2Jb" in out


def test_streamed_chat_tokens_are_verbatim_when_piped(monkeypatch, capsys):
    from localm.cli.chat import _ThinkPrinter
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    printer = _ThinkPrinter()
    printer.feed("x\x1b[2Jy")
    printer.flush()
    assert "x\x1b[2Jy" in capsys.readouterr().out


def test_bug_report_header_survives_markup_in_the_failure(monkeypatch, tmp_path):
    from localm.bugreport import _common, consent
    con, buf = _plain_console()
    monkeypatch.setattr(_common, "console", con)
    monkeypatch.setattr(consent, "console", con)
    monkeypatch.setattr(consent._br, "save_report", lambda text: tmp_path / "r.md")
    consent.report_failure(summary="crashed on [/] \x1bc", reason="[bold]boom[/x]",
                           interactive=False)
    out = buf.getvalue()
    assert "crashed on [/] c" in out and "[bold]boom[/x]" in out
    assert "\x1bc" not in out


def test_memory_cli_lists_a_hostile_fact_without_controls(tmp_path, monkeypatch):
    from click.testing import CliRunner
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "home_dir", lambda: h)
    from localm.memory.store import MemoryRecord
    from localm.plugins.builtin.memory.cli import _store, main
    _store().add(MemoryRecord(text="user likes \x1b[2Jtea", source="user"))

    res = CliRunner().invoke(main, ["list"], catch_exceptions=False)

    assert res.exit_code == 0, res.output
    assert "user likes [2Jtea" in res.output
    assert "\x1b" not in res.output
