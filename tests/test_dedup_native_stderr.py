# SPDX-License-Identifier: AGPL-3.0-or-later
"""dedup_native_stderr(): collapses consecutive identical native-stderr lines
into "line(N)" for the live terminal/GUI views, without losing anything from
the persisted debug-log record.

test_console_stream_is_captured_before_fd_redirect pins an ORDERING invariant:
_stable_console_stream() must be called BEFORE fd 2 is redirected to the context's
own pipe. Called after, the "stable" console duplicate points back at that SAME
pipe, so every emitted grouped line loops back in as fresh input, is re-grouped and
re-emitted, and a generation request hangs indefinitely. The loop does not
reproduce under pytest's own fd capture, which substitutes a plain file for fd 2,
so the call order is asserted directly instead of the hang.
"""

import logging
import os
import time

import pytest

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


def test_two_line_cycle_is_grouped_with_count():
    """The native pattern this class exists for: two DISTINCT lines alternating,
    never repeating immediately after themselves, so a single-line lookback
    collapses neither one."""
    before = len(debuglog.recent_activity())
    with debuglog.dedup_native_stderr():
        for _ in range(20):
            os.write(2, b"ggml_backend_cuda_graph_compute: CUDA graph warmup complete\n")
            os.write(2, b"ggml_backend_cuda_graph_compute: CUDA graph warmup reset\n")
    joined = "\n".join(debuglog.recent_activity()[before:])
    assert "ggml_backend_cuda_graph_compute: CUDA graph warmup complete(20)" in joined
    assert "ggml_backend_cuda_graph_compute: CUDA graph warmup reset(20)" in joined


def test_cycle_survives_an_interleaved_changing_line():
    """The exact production shape: a 2-line repeating cycle with a THIRD line
    that changes every occurrence (mirrors "CUDA Graph id N reused", N
    varies) interleaved between repeats. The changing line can never collapse
    itself (it never matches a prior line), but it must not evict/reset the
    two lines that ARE genuinely repeating - that was the actual bug: a
    small pure lookback (or a naive fixed-order FIFO ring) gets displaced by
    every one-off arrival before the repeating pair can accumulate a count."""
    before = len(debuglog.recent_activity())
    with debuglog.dedup_native_stderr():
        for i in range(12):
            os.write(2, f"CUDA Graph id {i} reused\n".encode())
            os.write(2, b"ggml_backend_cuda_graph_compute: CUDA graph warmup complete\n")
            os.write(2, b"ggml_backend_cuda_graph_compute: CUDA graph warmup reset\n")
    joined = "\n".join(debuglog.recent_activity()[before:])
    assert "ggml_backend_cuda_graph_compute: CUDA graph warmup complete(12)" in joined
    assert "ggml_backend_cuda_graph_compute: CUDA graph warmup reset(12)" in joined
    # each changing line is distinct, so it is emitted bare, never dropped
    for i in range(12):
        assert f"CUDA Graph id {i} reused" in joined


def test_more_distinct_lines_than_capacity_still_emits_everything():
    """Bounded memory: pushing well past _MAX_PENDING distinct lines must
    still emit every one of them (via LRU eviction), never silently drop a
    line just because the pending set filled up."""
    before = len(debuglog.recent_activity())
    n = debuglog._LineGrouper._MAX_PENDING * 4
    with debuglog.dedup_native_stderr():
        for i in range(n):
            os.write(2, f"distinct line {i}\n".encode())
    joined = "\n".join(debuglog.recent_activity()[before:])
    for i in range(n):
        assert f"distinct line {i}" in joined


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
    """If the persisted debug-log write fails, the native line is NOT lost - it
    still reaches the ring buffer, and exactly ONE warning is emitted. The latch
    suppresses further warnings so a persistently-failing fd cannot spam the log
    (and the warning is itself drained back through this same reader). Covers all
    three failure modes: never-warns, warns-every-time (log spam), and
    line-dropped-on-failure."""
    # A sentinel debug_fd that os.write always rejects. A fake fd number plus a
    # pass-through os.write is deterministic and platform-independent; a pre-closed
    # real fd would be REUSED by the dup/pipe fds dedup_native_stderr allocates
    # after native_stderr_target() is called. Only the sentinel is failed;
    # os.close(sentinel) at teardown is suppressed.
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


def test_teardown_survives_a_slow_reader_thread(monkeypatch, caplog):
    """A reader thread too slow to finish within the join timeout must not lose
    data permanently - it keeps draining in the background (a daemon thread,
    never killed) and the ring buffer catches up. The timeout expiring must be
    logged, not silent, since a caller checking recent_activity() right after
    the context exits would otherwise see an incomplete view with no signal
    that anything was still in flight."""
    monkeypatch.setattr(debuglog, "_READER_JOIN_TIMEOUT", 0.05)
    real_feed = debuglog._LineGrouper.feed

    def slow_feed(self, line):
        time.sleep(0.03)
        return real_feed(self, line)

    monkeypatch.setattr(debuglog._LineGrouper, "feed", slow_feed)

    before = len(debuglog.recent_activity())
    with caplog.at_level(logging.WARNING, logger="localm"):
        with debuglog.dedup_native_stderr():
            os.write(2, b"slow-line-one\n")
            os.write(2, b"slow-line-two\n")
            os.write(2, b"slow-line-three\n")

    warns = [r for r in caplog.records
             if r.levelno >= logging.WARNING and "did not finish" in r.getMessage()]
    assert len(warns) == 1, (
        f"expected exactly one reader-timeout warning, got {len(warns)}: "
        f"{[w.getMessage() for w in warns]}")

    deadline = time.monotonic() + 5.0
    lines = ("slow-line-one", "slow-line-two", "slow-line-three")
    while time.monotonic() < deadline:
        tail = "\n".join(debuglog.recent_activity()[before:])
        if all(line in tail for line in lines):
            break
        time.sleep(0.05)
    else:
        pytest.fail("the abandoned reader thread never delivered every line "
                     f"(saw: {debuglog.recent_activity()[before:]!r})")


# --------------------------------------------------------------------------- #
#  Template grouping: a varying-integer flood
#
#  "CUDA Graph id N reused" cycles N over a bounded set, so the lines are
#  verbatim repeats spaced wider than the LRU is deep.
# --------------------------------------------------------------------------- #

def _group(lines):
    """Drive _LineGrouper directly - no fd redirection, no ring buffer."""
    out = []
    g = debuglog._LineGrouper(out.append)
    for line in lines:
        g.feed(line)
    g.flush()
    return out


def test_varying_integer_flood_collapses_to_one_counted_line():
    """A cycle LONGER than _MAX_PENDING never groups by exact match no matter how
    the LRU is sized, because every entry is evicted before it comes round."""
    cap = debuglog._LineGrouper._MAX_PENDING
    cycle = [f"CUDA Graph id {700 + 2 * i} reused" for i in range(cap * 3)]
    out = _group(cycle * 40)

    assert len(out) == 1, f"expected one counted line, got {len(out)}: {out[:4]}"
    assert out[0] == f"CUDA Graph id <N> reused (x{len(cycle) * 40}, {len(cycle)} distinct)"


def test_the_flood_no_longer_evicts_the_lines_that_DO_repeat_verbatim():
    """With more varying ids in flight than slots, a genuinely-repeating pair is
    evicted before it can accumulate a count.

    THE INTERLEAVING RATIO IS LOAD-BEARING - do not "simplify" this to one id per
    pair. With a single id between them the pair is touched every third line, so
    move_to_end keeps it most-recently-used and it is NEVER the LRU victim, and
    the test then passes with OR without the fix. The real stream emits many
    CONSECUTIVE ids between warmup lines, which is what evicts the pair."""
    cap = debuglog._LineGrouper._MAX_PENDING
    rounds = 24
    stream = []
    for _ in range(rounds):
        stream += [f"CUDA Graph id {700 + 2 * j} reused" for j in range(cap + 4)]
        stream += ["ggml: warmup complete", "ggml: warmup reset"]
    out = _group(stream)
    assert f"ggml: warmup complete({rounds})" in out, out
    assert f"ggml: warmup reset({rounds})" in out, out


def test_a_LIST_of_distinct_messages_is_never_collapsed():
    """Guards against over-reach: 28 layer-assignment lines that each appear ONCE
    are a list, not a flood, and collapsing them would lose every layer number to
    save nothing. Both conditions in _emit_one are required for this."""
    lines = [f"load_tensors: layer {i} assigned to device ROCm0" for i in range(28)]
    out = _group(lines)
    assert out == lines, "a non-repeating list must pass through verbatim"


def test_few_variants_still_group_individually_with_their_own_counts():
    """At or below _MAX_PENDING distinct variants, exact matching already works
    and is MORE informative than a placeholder - each id keeps its own count.
    Templating must not destroy that."""
    out = _group([f"CUDA Graph id {i % 2} reused" for i in range(8)])
    assert out == ["CUDA Graph id 0 reused(4)", "CUDA Graph id 1 reused(4)"], out


def test_lines_without_digits_are_completely_unaffected():
    out = _group(["no digits here"] * 5)
    assert out == ["no digits here(5)"]


def test_distinct_variant_count_is_bounded_but_still_reported():
    """The retained-variant set is capped so a never-repeating stream cannot
    make this unbounded memory; past the cap the count is reported as 'N+', never
    silently understated."""
    cap = debuglog._LineGrouper._MAX_VARIANTS
    out = _group([f"thing {i} done" for i in range(cap + 50)] * 3)
    assert len(out) == 1
    assert f"{cap}+ distinct" in out[0], out[0]
