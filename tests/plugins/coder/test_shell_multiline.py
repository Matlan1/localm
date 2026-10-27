# SPDX-License-Identifier: AGPL-3.0-or-later
"""A run_shell command of several lines runs every line, or is refused.

Drives the real tool against the real platform shell: the files each line
writes are what is asserted, not the routing decision alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from localm.plugins.coder.background import get_registry, reset_registry
from localm.plugins.coder.tools import shell as shell_mod
from localm.plugins.coder.tools.shell import (
    CommandNotRunnable,
    _cmd_join_lines,
    _shell_argv,
    tool_check_shell_job,
    tool_run_shell,
    tool_run_shell_background,
)

_WIN = sys.platform == "win32"


def _write(name: str, text: str) -> str:
    return f"echo {text}> {name}" if _WIN else f"echo {text} > {name}"


@pytest.mark.parametrize("sep", ["\n", "\r\n"])
def test_every_line_of_a_shell_routed_command_runs(tmp_path, sep):
    cmd = _write("one.txt", "first") + sep + _write("two.txt", "second")
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert result.ok, result.output
    assert (tmp_path / "one.txt").read_text().strip() == "first"
    assert (tmp_path / "two.txt").read_text().strip() == "second"


def test_lines_without_shell_characters_run_as_separate_commands(tmp_path):
    """Two plain commands on two lines: the second line used to become
    arguments of the first."""
    cmd = "git --version\ngit --version"
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert result.ok, result.output
    assert result.output.count("git version") == 2, result.output


def test_a_failing_first_line_does_not_stop_the_second(tmp_path):
    """Like sh with separate lines: each line runs; the exit code is the last
    line's."""
    cmd = "git rev-parse --is-inside-work-tree\n" + _write("after.txt", "ran")
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert (tmp_path / "after.txt").read_text().strip() == "ran"
    assert result.ok, result.output


def test_blank_and_trailing_lines_are_ignored(tmp_path):
    cmd = "\n\n" + _write("only.txt", "x") + "\n\n"
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert result.ok, result.output
    assert (tmp_path / "only.txt").read_text().strip() == "x"


def test_a_trailing_line_break_keeps_the_argument_list_route():
    argv = _shell_argv("git status\n")
    assert isinstance(argv, list), argv
    assert argv[1:] == ["status"]


def test_a_background_command_runs_every_line(tmp_path):
    reset_registry()
    try:
        cmd = _write("bg1.txt", "a") + "\n" + _write("bg2.txt", "b")
        started = tool_run_shell_background(tmp_path, cmd)
        assert started.ok, started.output
        job_id = started.output.split("<job>")[1].split("</job>")[0]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if get_registry().get(job_id).status()["state"] != "running":
                break
            time.sleep(0.05)
        polled = tool_check_shell_job(tmp_path, job_id)
        assert "<exit_code>0</exit_code>" in polled.output, polled.output
        assert (tmp_path / "bg1.txt").read_text().strip() == "a"
        assert (tmp_path / "bg2.txt").read_text().strip() == "b"
    finally:
        reset_registry()


@pytest.mark.skipif(not _WIN, reason="cmd.exe cannot pass a quoted line break")
def test_a_quoted_line_break_through_cmd_is_refused_and_nothing_runs(tmp_path):
    cmd = _write("first.txt", "x") + '\necho "a\nb"'
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert not result.ok
    assert "line break inside double quotes" in result.output
    assert not (tmp_path / "first.txt").exists(), "a line ran before the refusal"


@pytest.mark.skipif(not _WIN, reason="cmd.exe cannot pass a quoted line break")
def test_a_background_quoted_line_break_is_refused(tmp_path):
    reset_registry()
    try:
        result = tool_run_shell_background(tmp_path, 'echo "a\nb"')
        assert not result.ok
        assert "line break inside double quotes" in result.output
        assert get_registry().ids(kind="shell") == []
    finally:
        reset_registry()


@pytest.mark.skipif(not _WIN, reason="cmd.exe block syntax")
def test_a_multi_line_cmd_block_runs(tmp_path):
    (tmp_path / "flag.txt").write_text("y", encoding="utf-8")
    cmd = ("if exist flag.txt (\n"
           "  echo inside> in.txt\n"
           ") else (\n"
           "  echo other> other.txt\n"
           ")\n"
           + _write("after.txt", "z"))
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert result.ok, result.output
    assert (tmp_path / "in.txt").read_text().strip() == "inside"
    assert not (tmp_path / "other.txt").exists()
    assert (tmp_path / "after.txt").read_text().strip() == "z"


@pytest.mark.parametrize("command, joined", [
    ("a\nb", "a & b"),
    ("a\r\n\r\nb\n", "a & b"),
    ("if x (\n echo y\n)", "if x ( echo y )"),
    (") else (\nz\n)", ") else ( z )"),
    ("a &&\nb", "a && b"),
    ("a\n|| b", "a || b"),
    ("echo one ^\ntwo", "echo one two"),
    ("echo one ^^\ntwo", "echo one ^^ & two"),
    ('echo "x"\necho y', 'echo "x" & echo y'),
    ('echo ^"\necho y', 'echo ^" & echo y'),
])
def test_cmd_join_lines(command, joined):
    assert _cmd_join_lines(command) == joined


def test_cmd_join_lines_refuses_a_quoted_line_break():
    with pytest.raises(CommandNotRunnable):
        _cmd_join_lines('echo "a\nb"')


def test_posix_routes_several_lines_to_sh_unchanged(monkeypatch):
    monkeypatch.setattr(shell_mod.sys, "platform", "linux")
    from localm.plugins.coder.tools import base
    monkeypatch.setattr(base.sys, "platform", "linux")
    cmd = "git status\ngit log -1"
    assert _shell_argv(cmd) == ["/bin/sh", "-c", cmd]


def test_a_quoted_line_break_in_an_argument_list_is_kept(tmp_path):
    """A line break inside quotes in a command with no shell characters stays
    one argument of one process (the argument-list route)."""
    script = tmp_path / "argv.py"
    script.write_text(
        "import sys, pathlib\n"
        "pathlib.Path('out.txt').write_text(repr(sys.argv[1:]))\n",
        encoding="utf-8")
    exe = Path(sys.executable).as_posix()
    cmd = f'"{exe}" argv.py "a\nb" c'
    argv = _shell_argv(cmd)
    assert isinstance(argv, list), argv
    result = tool_run_shell(tmp_path, cmd, timeout=30)
    assert result.ok, result.output
    assert (tmp_path / "out.txt").read_text() == repr(["a\nb", "c"])
