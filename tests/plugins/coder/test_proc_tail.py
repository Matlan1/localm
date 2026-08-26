# SPDX-License-Identifier: AGPL-3.0-or-later
"""StderrTail: a bounded, non-blocking capture of a child process's stderr,
shared by ManagedServer (NEW-CODER-MANAGED-SERVER-STDERR) and MCPServer
(NEW-CODER-MCP-SERVER-STDERR)."""

import subprocess
import sys
import time

from localm.plugins.coder.proc_tail import StderrTail


def _spawn(code: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )


def test_captures_lines_written_to_stderr():
    proc = _spawn("import sys; sys.stderr.write('boom: missing config\\n')")
    tail = StderrTail(proc)
    proc.wait(timeout=10)
    time.sleep(0.1)   # let the drain thread finish reading EOF
    assert "boom: missing config" in tail.tail()


def test_empty_when_the_child_writes_nothing():
    proc = _spawn("pass")
    tail = StderrTail(proc)
    proc.wait(timeout=10)
    time.sleep(0.1)
    assert tail.tail() == ""


def test_ring_keeps_only_the_last_n_lines():
    proc = _spawn(
        "import sys\n"
        "for i in range(50): sys.stderr.write(f'line{i}\\n')\n"
    )
    tail = StderrTail(proc, maxlines=5)
    proc.wait(timeout=10)
    time.sleep(0.1)
    lines = tail.tail().splitlines()
    assert lines == [f"line{i}" for i in range(45, 50)]


def test_a_chatty_child_never_deadlocks_on_an_unread_pipe():
    """The whole point of draining stderr on a thread: a child that writes
    far more than any OS pipe buffer holds must still be able to exit."""
    proc = _spawn(
        "import sys\n"
        "for i in range(20000): sys.stderr.write('x' * 200 + '\\n')\n"
    )
    StderrTail(proc)
    proc.wait(timeout=10)   # would hang here if stderr were unread
