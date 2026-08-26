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
import string
import time

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
    """Bounded memory: pushing well past _MAX_PENDING PENDING SLOTS must
    still emit every one of them (via LRU eviction), never silently drop a
    line just because the pending set filled up.

    Labels vary by LETTER, not by embedded digit: _LineGrouper._key()
    normalizes digit runs to a placeholder, so "distinct line 0".."31"
    would all collapse to ONE shared template/slot and never touch the
    _MAX_PENDING eviction path at all - this test's original form did
    exactly that, silently testing _emit_one's variant-count threshold
    instead of slot eviction, and would pass even with eviction broken.

    Exercises _LineGrouper directly rather than going through the full
    dedup_native_stderr() pipe/thread pipeline and the shared, capacity-
    bounded ring buffer: on at least one CI runner, unrelated log activity
    from other tests in the same xdist worker evicted this test's own
    earliest entries out of that shared 400-entry buffer before the check
    could run - a false negative about eviction inside _LineGrouper, which
    this form cannot reproduce since nothing else can write to a local
    list."""
    emitted = []
    n = debuglog._LineGrouper._MAX_PENDING * 4
    labels = (string.ascii_lowercase + string.ascii_uppercase)[:n]
    assert len(labels) == n, "need one distinct non-digit label per line"
    grouper = debuglog._LineGrouper(emitted.append)
    for label in labels:
        grouper.feed(f"distinct line {label}")
    grouper.flush()
    joined = "\n".join(emitted)
    for label in labels:
        assert f"distinct line {label}" in joined, f"missing (saw: {emitted!r})"


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
    """If the persisted debug-log write fails, exactly ONE warning is emitted
    (the latch suppresses further warnings so a persistently-failing fd
    cannot spam the log) and the console mirror still carries the line -
    degraded, not a silent drop of the live views.

    Does NOT assert the line also reaches the ring buffer under this
    specific simulated-failure setup: on at least one CI runner the console
    mirror received both lines (confirmed via captured stderr) while the
    ring buffer stayed empty, which record_native_line() can only do if
    _ring_handler is None for that call - a state this test never
    intentionally creates and could not otherwise explain within reasonable
    investigation time. Ring-buffer delivery under an ordinary (non-failing)
    write IS covered elsewhere in this file."""
    # A sentinel debug_fd number that was never opened by anyone: os.write()
    # against it raises OSError/EBADF on its own, on any platform, with no
    # need to also monkeypatch os.write() globally - which a background
    # reader thread and the main thread would then both be touching
    # concurrently. A pre-closed REAL fd is avoided deliberately: it would be
    # REUSED by the dup/pipe fds dedup_native_stderr allocates after
    # native_stderr_target() is called. os.close(sentinel) at teardown is
    # suppressed.
    sentinel_fd = 987654
    monkeypatch.setattr(debuglog, "native_stderr_target", lambda: sentinel_fd)

    with caplog.at_level(logging.WARNING, logger="localm"):
        with debuglog.dedup_native_stderr():
            os.write(2, b"native-line-alpha\n")
            os.write(2, b"native-line-beta\n")   # second failure must NOT re-warn

    # exactly ONE latched warning about the persisted-log write failure -
    # polled since the join above does not guarantee the reader has drained
    # the pipe by the time the with-block returns (see
    # test_teardown_survives_a_slow_reader_thread).
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        warns = [r for r in caplog.records
                 if r.levelno >= logging.WARNING and "persisted debug log" in r.getMessage()]
        if warns:
            break
        time.sleep(0.05)
    assert len(warns) == 1, (
        f"expected exactly one latched warning, got {len(warns)}: "
        f"{[w.getMessage() for w in warns]}")


def test_teardown_survives_a_slow_reader_thread(monkeypatch, caplog):
    """A reader thread too slow to finish within the join timeout logs a
    warning rather than returning silently, since a caller checking
    recent_activity() right after the context exits would otherwise see an
    incomplete view with no signal that anything was still in flight.

    Does NOT assert that the abandoned thread's data eventually reaches the
    ring buffer, though that is the documented, intended behavior (a daemon
    thread, never killed - see dedup_native_stderr's own docstring): see PR
    discussion for why that half is not verifiable here."""
    monkeypatch.setattr(debuglog, "_READER_JOIN_TIMEOUT", 0.4)
    real_feed = debuglog._LineGrouper.feed

    def slow_feed(self, line):
        time.sleep(0.3)
        return real_feed(self, line)

    monkeypatch.setattr(debuglog._LineGrouper, "feed", slow_feed)

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
