# SPDX-License-Identifier: AGPL-3.0-or-later
"""dedup_native_stderr(): collapses consecutive identical native-stderr lines
into "line(N)" for the live terminal/GUI views, without losing anything from
the persisted debug-log record.

test_console_stream_is_captured_before_fd_redirect is a regression guard for
a real bug caught during development: _stable_console_stream() was called
AFTER fd 2 was redirected to the context's own pipe, so the "stable" console
duplicate actually pointed back at that SAME pipe (confirmed empirically -
console.fileno() equalled the pipe's own write_fd, and writes to it arrived
back on the pipe's own read end). Every emitted grouped line looped back in
as fresh input, got re-grouped, and re-emitted - a real generation request
hung indefinitely because of this (only reproduces outside pytest's own fd
capture, which substitutes a plain file for fd 2 and happens to break the
feedback chain; a live standalone script and a direct fd probe both
confirmed the mechanism). Asserting the call order directly is deterministic
and fast, unlike trying to reproduce the hang itself under pytest.
"""

import logging
import os

from localm import debuglog


def setup_function():
    debuglog.install_ring_buffer()


def test_console_stream_is_captured_before_fd_redirect(monkeypatch):
    calls = []
    real_dup2 = os.dup2
    real_stream_fn = debuglog._stable_console_stream

    def tracking_dup2(fd, fd2, *a, **kw):
        if fd2 == 2:
            calls.append("dup2")
        return real_dup2(fd, fd2, *a, **kw)

    def tracking_stream():
        calls.append("stable_console_stream")
        return real_stream_fn()

    monkeypatch.setattr(os, "dup2", tracking_dup2)
    monkeypatch.setattr(debuglog, "_stable_console_stream", tracking_stream)

    with debuglog.dedup_native_stderr():
        pass

    assert "stable_console_stream" in calls, "the console duplicate was never taken"
    assert calls.index("stable_console_stream") < calls.index("dup2"), (
        "_stable_console_stream() must run BEFORE the fd-2 dup2 redirect, "
        f"else it duplicates the pipe instead of the real stderr: {calls}"
    )


def test_consecutive_duplicates_are_grouped_with_count():
    before = len(debuglog.recent_activity())
    with debuglog.dedup_native_stderr():
        for _ in range(5):
            os.write(2, b"CUDA Graph id 51 reused\n")
        os.write(2, b"a different line\n")
        for _ in range(3):
            os.write(2, b"CUDA Graph id 54 reused\n")
    tail = debuglog.recent_activity()[before:]
    joined = "\n".join(tail)
    assert "CUDA Graph id 51 reused(5)" in joined
    assert "a different line" in joined
    # a run of exactly 1 is emitted bare, with no (N) suffix
    assert "a different line(1)" not in joined
    assert "CUDA Graph id 54 reused(3)" in joined


def test_single_line_is_not_suffixed():
    before = len(debuglog.recent_activity())
    with debuglog.dedup_native_stderr():
        os.write(2, b"only once\n")
    tail = debuglog.recent_activity()[before:]
    assert any(line.endswith("only once") for line in tail)
    assert not any("only once(" in line for line in tail)


def test_fd_2_is_restored_after_exit(capfd):
    """After the context exits, writes to fd 2 must reach the real stream
    again (not still be swallowed by the torn-down pipe)."""
    with debuglog.dedup_native_stderr():
        os.write(2, b"inside the context\n")
    os.write(2, b"after the context\n")
    captured = capfd.readouterr()
    assert "after the context" in captured.err


def test_nothing_written_is_a_clean_noop():
    before = len(debuglog.recent_activity())
    with debuglog.dedup_native_stderr():
        pass
    assert debuglog.recent_activity()[before:] == []


def test_persisted_write_failure_warns_once_and_keeps_ring_buffer(monkeypatch, caplog):
    """Honesty audit (finding 6): if the persisted debug-log write fails, the
    native line is NOT silently lost - it still reaches the ring buffer, and
    exactly ONE warning is emitted. The latch suppresses further warnings so a
    persistently-failing fd cannot spam the log (and the warning is itself
    drained back through this same reader). Guards all three regression modes:
    never-warns, warns-every-time (log spam), and line-dropped-on-failure.

    Pre-fix, os.write(debug_fd,...) was wrapped in contextlib.suppress(OSError)
    with no warning, so this test's warning-count assertion fails on old code."""
    # A sentinel debug_fd that os.write always rejects. Using a fake fd number +
    # a pass-through os.write is deterministic and platform-independent: a
    # pre-closed real fd would get REUSED by the dup/pipe fds dedup_native_stderr
    # allocates after native_stderr_target() is called, so its os.write would
    # then wrongly succeed. Real os.write is used for the console/pipe elsewhere,
    # so only the sentinel is failed; os.close(sentinel) at teardown is suppressed.
    sentinel_fd = 987654
    monkeypatch.setattr(debuglog, "native_stderr_target", lambda: sentinel_fd)
    real_os_write = os.write

    def failing_write(fd, data):
        if fd == sentinel_fd:
            raise OSError(9, "Bad file descriptor")   # simulate a dead persisted fd
        return real_os_write(fd, data)

    monkeypatch.setattr(os, "write", failing_write)

    before = len(debuglog.recent_activity())
    with caplog.at_level(logging.WARNING, logger="localm"):
        with debuglog.dedup_native_stderr():
            real_os_write(2, b"native-line-alpha\n")
            real_os_write(2, b"native-line-beta\n")   # second failure must NOT re-warn

    # both distinct lines survived to the ring buffer despite the write failures
    tail = "\n".join(debuglog.recent_activity()[before:])
    assert "native-line-alpha" in tail
    assert "native-line-beta" in tail
    # exactly ONE latched warning about the persisted-log write failure
    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "persisted debug log" in r.getMessage()]
    assert len(warns) == 1, (
        f"expected exactly one latched warning, got {len(warns)}: "
        f"{[w.getMessage() for w in warns]}")
