# SPDX-License-Identifier: AGPL-3.0-or-later
"""Output written to a file or a pipe is laid out wide enough that a table
keeps long model names and folders whole (every localm console is a
SafeConsole), while a terminal, an explicit width and COLUMNS still decide."""

from __future__ import annotations

import io

from click.testing import CliRunner
from rich.table import Table

from localm.console import PIPED_WIDTH, SafeConsole

_LONG_NAME = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL-with-a-longer-suffix"
_LONG_PATH = "D:/models/some/deeply/nested/folder/for/the/download/" + _LONG_NAME + ".gguf"


def _table() -> Table:
    table = Table("name", "type", "size", "path")
    table.add_row(_LONG_NAME, "gguf", "17.3 GB", _LONG_PATH)
    return table


def test_a_piped_console_keeps_a_long_table_row_whole(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    buf = io.StringIO()
    SafeConsole(file=buf).print(_table())
    out = buf.getvalue()
    assert _LONG_NAME in out
    assert _LONG_PATH in out
    assert "\u2026" not in out


def test_a_piped_console_is_at_least_the_piped_width(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    assert SafeConsole(file=io.StringIO()).width == PIPED_WIDTH


def test_an_explicit_width_and_columns_still_decide(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    assert SafeConsole(file=io.StringIO(), width=60).width == 60
    monkeypatch.setenv("COLUMNS", "90")
    assert SafeConsole(file=io.StringIO()).width == 90


def test_a_terminal_keeps_its_own_width(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    console = SafeConsole(file=io.StringIO(), force_terminal=True)
    assert console.is_terminal
    assert console.width < PIPED_WIDTH


def test_key_list_written_to_a_pipe_keeps_long_names(monkeypatch, tmp_path):
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    from localm import auth
    from localm.cli import main
    name = "a-device-with-a-really-long-descriptive-name-for-the-kitchen-tablet"
    auth.create_key(name, ["chat"])
    result = CliRunner().invoke(main, ["key", "list"])
    assert result.exit_code == 0, result.output
    assert name in result.output
