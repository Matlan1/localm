# SPDX-License-Identifier: AGPL-3.0-or-later
"""Crash-containment tests for backends/_hf_runner.py's HFRunner.

HFWorker's native calls (a "fast" tokenizer's Rust regex pre-tokenizer stage,
a torch forward pass, model.generate()) are uninterruptible from Python: a
catastrophic-backtracking tokenizer pattern or a genuinely wedged native call
hangs the calling thread forever, and nothing short of killing the process
reclaims it. localm therefore runs the whole HF backend lifecycle in an
isolated worker process; a crash or hang there kills only the worker, and the
caller gets a clean, catchable error within a bounded time while the server
itself stays up.

These tests prove the containment property with REAL, uncatchable faults (a
hard process exit, a genuine abort, and a hang) injected into the worker via
the LOCALM_HF_FAULT_FOR_TEST hook - the same code path a real hang would take.
TestExecutorPoolReclaim goes one step further and drives the runner through
loop.run_in_executor(), not just its own synchronous timeout contract.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue as _queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from localm.inference.backends import _hf_runner as runner_mod
from localm.inference.backends._hf_runner import HFRunner
from localm.inference.backends.hf import HFBackend


@pytest.fixture(autouse=True)
def _clean_fault_env():
    os.environ.pop(runner_mod._FAULT_ENV, None)
    yield
    os.environ.pop(runner_mod._FAULT_ENV, None)


_DUMMY_LOAD_PARAMS = dict(model_path="does-not-matter", device="cpu")


# --------------------------------------------------------------------------- #
# A native fault is uncatchable in-process.
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    code = (
        "import os, ctypes\n"
        "if os.name == 'nt':\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)\n"
        "try:\n"
        "    os.abort()\n"
        "except BaseException:\n"
        "    print('SURVIVED')\n"
        "else:\n"
        "    print('NO_FAULT')\n"
    )
    proc = subprocess.run([sys.executable, "-u", "-c", code],
                          capture_output=True, text=True, timeout=30)
    assert "SURVIVED" not in proc.stdout
    assert "NO_FAULT" not in proc.stdout
    assert proc.returncode != 0          # the process died from the native fault


# --------------------------------------------------------------------------- #
# Containment: a crashed/hung worker never takes the caller (this process) down.
# --------------------------------------------------------------------------- #

class TestLoadCrashContainment:
    def test_hard_exit_during_load_is_contained(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "exit")
        r = HFRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_real_native_abort_during_load_is_contained(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = HFRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_hung_load_times_out_and_is_killed(self, monkeypatch):
        # A wedged native call during load is killed at the timeout and raises a
        # clean, actionable error.
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        r = HFRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=2.0)
            assert "timed out" in str(ei.value).lower()
            assert not r.is_alive(), "the hung worker must be killed, not left running"
        finally:
            r.shutdown(grace=0)

    def test_bad_load_payload_is_a_clean_error_not_a_crash(self):
        """The HFWorker(**payload) constructor call sits INSIDE the "load"
        branch's try/except (see _hf_runner.py), so a malformed payload - a
        parent/child protocol bug, not a native fault - is a normal, clean
        error, not an uncaught crash that kills the process for no native
        reason at all."""
        r = HFRunner()
        bad_params = dict(_DUMMY_LOAD_PARAMS, this_kwarg_does_not_exist=True)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(bad_params, timeout=30.0)
            assert "unexpected keyword argument" in str(ei.value).lower()
            assert r.is_alive(), (
                "a bad payload is a protocol error, not a native fault - the "
                "worker process itself must not have been killed by it"
            )
        finally:
            r.shutdown(grace=0)


class TestRunnerLifecycle:
    def test_is_alive_false_before_spawn_and_after_shutdown(self):
        r = HFRunner()
        assert not r.is_alive()
        r.shutdown()   # must be a no-op, never raise, when nothing is running

    def test_double_shutdown_is_safe(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "exit")
        r = HFRunner()
        with pytest.raises(RuntimeError):
            r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
        r.shutdown(grace=0)
        r.shutdown(grace=0)   # must not raise the second time


class TestChatStreamCrashContainment:
    def test_dispatch_crash_during_chat_stream_is_contained(self):
        """An uncaught exception ANYWHERE in the child's dispatch loop - not
        just a real native fault - must crash only the child, never the
        caller. Reproduced here without needing a real model by sending
        chat_stream before any load (worker is None -> AttributeError inside
        _runner_main's dispatch -> uncaught -> the process dies)."""
        r = HFRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            msg = str(ei.value).lower()
            # This fault is an uncaught PYTHON exception (exit 1), so the
            # message must NOT call it a native fault.
            assert "unloaded" in msg and "reload" in msg, msg
            assert "native inference fault" not in msg, msg
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)


class TestSimpleRequestCrashContainment:
    def test_count_tokens_crash_is_contained(self):
        """A crash while handling a simple request (not just load/chat_stream)
        must also be contained and reported cleanly, never left hanging.

        Kills the worker directly rather than using the env-var fault hook:
        LOCALM_HF_FAULT_FOR_TEST is read from the CHILD's own os.environ, a
        snapshot taken at spawn time - mutating the parent's os.environ
        afterwards can never reach an already-running child."""
        r = HFRunner()
        r._spawn()
        r._proc.kill()
        r._proc.join(timeout=5)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.count_tokens("hello")
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_embed_crash_is_contained(self):
        """Same as test_count_tokens_crash_is_contained, for embed() - a real
        production call path (HFBackend.embed) that GgufRunner has no
        equivalent of, since GGUF's can_embed is a fixed False."""
        r = HFRunner()
        r._spawn()
        r._proc.kill()
        r._proc.join(timeout=5)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"])
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)


class TestCrashDiagnosticsReachDebugLog:
    """The parent's crash message points the user at the debug log, so the
    worker's actual exception detail must really reach that log, not just the
    parent's own generic message.

    Covers only the half of the problem that HAS a Python exception. The other
    half - a death by native signal, where no ``except`` clause ever runs - is
    TestNativeSignalCrashDiagnosticsReachDebugLog below."""

    def test_uncaught_dispatch_crash_is_logged_to_the_debug_log(
            self, monkeypatch, tmp_path):
        log_path = tmp_path / "worker_crash.log"
        monkeypatch.setenv("LOCALM_DEBUG", str(log_path))

        r = HFRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            msg = str(ei.value).lower()
            # This fault is an uncaught PYTHON exception (exit 1), so the
            # message must NOT call it a native fault.
            assert "unloaded" in msg and "reload" in msg, msg
            assert "native inference fault" not in msg, msg
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)

        assert log_path.is_file(), (
            "the crashed worker never even attached the shared debug log"
        )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "AttributeError" in text and "NoneType" in text, (
            "the worker crashed with no envelope (by design), but its actual "
            "exception detail never reached the debug log - the parent's own "
            "'see the debug log for the native stack trace' message is a lie "
            "if there is nothing in that log to see\n--- log content ---\n"
            + text
        )


class TestNativeSignalCrashDiagnosticsReachDebugLog:
    """The class above covers a crash that still has a PYTHON exception
    (``logger.critical(exc_info=True)`` in ``_runner_entry`` catches it). This
    one covers the OTHER half: a SIGILL/SIGSEGV/SIGABRT inside native code (a
    torch forward pass, a CUDA/ROCm kernel, a fast tokenizer's Rust stage) never
    returns to Python, so no ``except`` clause can run, and only faulthandler's
    armed trace file records anything.

    ``os.abort()`` (the ``LOCALM_HF_FAULT_FOR_TEST=abort`` hook) produces that
    class in a REAL child process: the process dies from a native signal with no
    Python exception.
    """

    def _fault_during_chat_stream(self, monkeypatch):
        """Drive a real worker to a real native abort mid-chat_stream. Returns
        ``(message, trace_path)``.

        The fault env var is set BEFORE ``_spawn()``: the child reads it from
        its OWN ``os.environ``, a snapshot taken at spawn time, so setting it
        afterwards never reaches a running child."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")

        r = HFRunner()
        r._spawn()
        trace_path = r._crash_trace_path
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            assert not r.is_alive()
            return str(ei.value), trace_path
        finally:
            r.shutdown(grace=0)

    def test_native_abort_is_reported_with_its_captured_trace(
            self, monkeypatch, caplog):
        """After a native-signal death the caller is told WHAT faulted, not
        merely that something did.

        Asserts on the TRACE CONTENT rather than on the exit code or the
        presence of the words "native inference fault": the trace text is what
        distinguishes a captured fault from an uncharacterised one."""
        with caplog.at_level(logging.ERROR, logger="localm"):
            message, _ = self._fault_during_chat_stream(monkeypatch)

        assert "native inference fault" in message.lower()
        assert "Fatal Python error" in message, (
            "a real native-signal death produced no captured trace, so the "
            "caller still cannot tell WHICH native call faulted - the reported "
            f"issue 1222 / 1223 symptom\n--- message ---\n{message}"
        )

        # The FULL multi-line trace, not just the summary line folded into the
        # message, reaches the debug log. caplog rather than a real log file, so
        # no handler is attached to the shared localm logger.
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "Fatal Python error" in logged and "_hf_runner.py" in logged, (
            "the trace never reached the localm logger, or names no Python "
            f"frame, so it cannot say where the fault happened\n{logged}"
        )

    def test_load_crash_also_reports_its_captured_trace(self, monkeypatch):
        """The load path builds its own crash message, separately from
        chat_stream's, so it carries its own captured trace."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = HFRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=60.0)
            message = str(ei.value)
            assert "crashed" in message.lower()
            assert "Fatal Python error" in message, (
                "the load-crash message carries no captured trace\n"
                f"--- message ---\n{message}")
        finally:
            r.shutdown(grace=0)

    def test_no_trace_captured_is_stated_not_implied(self, monkeypatch):
        """When nothing was captured the message SAYS so, rather than pointing
        at a debug log with nothing in it.

        Arming-independent: it drops the path the parent would read, so it
        covers the branch that exists for when arming did not work."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = HFRunner()
        r._spawn()
        # Simulate the capture having failed (an unwritable logs dir, a platform
        # where enable() no-ops) by dropping the path the parent would read.
        r._crash_trace_path = None
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            assert "no native fault trace was captured" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_hard_killed_worker_does_not_imply_a_trace(self):
        """A worker killed outright (SIGKILL / TerminateProcess) runs no handler
        at all, so faulthandler writes nothing and the simple-RPC crash message
        says a trace was not captured. Covers the third message site
        (``_simple_rpc``), which the two above never reach."""
        r = HFRunner()
        r._spawn()
        r._proc.kill()
        r._proc.join(timeout=10)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.count_tokens("hello")
            message = str(ei.value)
            assert "crashed" in message.lower()
            assert "no native fault trace was captured" in message.lower(), (
                "a hard kill leaves no trace to relay, so the message must say "
                f"so rather than pointing at a log with nothing in it\n{message}")
        finally:
            r.shutdown(grace=0)

    def test_native_crash_trace_file_is_consumed_not_left_behind(
            self, monkeypatch):
        """The per-worker trace file must not survive the crash it describes: a
        stale file would be misread as a fresh crash by the next reader.

        Asserts the trace was CAPTURED as well as gone, so the check cannot pass
        on a file that was never written."""
        message, trace_path = self._fault_during_chat_stream(monkeypatch)
        assert trace_path is not None
        assert "Fatal Python error" in message, (
            "nothing was captured, so this test would be asserting cleanup of a "
            f"file that never existed\n--- message ---\n{message}")
        assert not trace_path.exists(), (
            f"the worker crash-trace file was left behind at {trace_path}")

    def test_healthy_worker_arms_a_trace_then_reaps_it(self):
        """The capture costs one empty file per model load, and a clean shutdown
        reaps it.

        Checks the file EXISTS while the worker is alive before checking it is
        gone afterwards, so the check cannot pass on a worker that never armed.

        The existence check is POLLED, not immediate: ``_spawn()`` returns as
        soon as ``Process.start()`` does, and a spawn-context child then boots a
        fresh interpreter and runs its imports before it arms anything."""
        r = HFRunner()
        r._spawn()
        trace_path = r._crash_trace_path
        try:
            assert trace_path is not None
            deadline = time.monotonic() + 60.0
            while not trace_path.exists() and time.monotonic() < deadline:
                assert r.is_alive(), "the worker died before arming a trace file"
                time.sleep(0.05)
            assert trace_path.exists(), (
                "the worker never armed a native-fault trace file, so a native "
                "fault in it would go uncharacterised")
        finally:
            r.shutdown(grace=5)
        assert not trace_path.exists(), (
            f"a cleanly shut-down worker left {trace_path} behind")


# --------------------------------------------------------------------------- #
# The runner driven through loop.run_in_executor(): an executor-thread CALLER
# gets its pool slot back. The tests above call the runner synchronously and
# check its own timeout mechanics.
# --------------------------------------------------------------------------- #

class TestExecutorPoolReclaim:
    def test_hung_call_through_run_in_executor_frees_its_pool_slot(self, monkeypatch):
        """Drive a hung HFBackend.load() through loop.run_in_executor(pool,
        ...) exactly as production code does (routes/chat.py:99,
        http_server.py's _generate_full via HFBackend.chat_stream, etc.), on
        a deliberately tiny (2-worker) pool so exhaustion is observable
        without needing 16 concurrent hangs. Two assertions: (a) the call
        raises within its bounded timeout instead of hanging forever, and
        (b) a SUBSEQUENT, unrelated run_in_executor call on the SAME pool
        completes promptly afterward, so the slot was reclaimed rather than
        merely abandoned by this coroutine."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        # Short timeout so the test itself stays fast; only the executor-thread
        # consequence of the bound is under test here.
        monkeypatch.setattr(HFBackend, "_load_timeout_seconds", staticmethod(lambda: 2.0))

        async def scenario():
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-pool")
            loop = asyncio.get_running_loop()
            engine = HFBackend("dummy-model-for-hang-test")
            try:
                t0 = time.monotonic()
                with pytest.raises(RuntimeError, match="(?i)timed out"):
                    await loop.run_in_executor(pool, engine.load)
                elapsed = time.monotonic() - t0
                assert elapsed < 10.0, (
                    f"load() should bound at ~2s (hf_load_timeout_s override), "
                    f"took {elapsed:.1f}s - the executor thread was not freed "
                    "promptly")

                # An unrelated call on the same pool must still complete
                # promptly. Bounded by asyncio.wait_for so a never-freed slot
                # fails FAST rather than hanging the suite.
                result = await asyncio.wait_for(
                    loop.run_in_executor(pool, lambda: 1 + 1), timeout=5.0)
                assert result == 2
            finally:
                pool.shutdown(wait=False)

        asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Cooperative mid-stream cancel: HFRunner's ctrl_q/_cancel_stream_and_drain.
# Only the native decode is faked (a plain threading.Thread reading real
# req_q/ctrl_q and writing real resp_q); the real HFRunner poll/lock/drain logic
# runs over real multiprocessing.Queues, with _proc replaced by a liveness-only
# stand-in, so no real subprocess or model is needed.
# --------------------------------------------------------------------------- #

class _AliveProc:
    """Stands in for the worker process's liveness check only; the queues and
    the parent-side poll/drain loop under test are real."""

    def __init__(self):
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None

    exitcode = 0


def _make_runner():
    ctx = mp.get_context("spawn")
    r = HFRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _AliveProc()
    return r


class TestCooperativeCancel:
    def test_cancel_stream_gets_done_without_killing_the_worker(self):
        """A worker that cooperates with cancel_stream (streams a couple of
        chunks, then confirms "done" once it sees the ctrl_q signal - what
        the real control-thread + StoppingCriteria produce) leaves the process
        ALIVE: the mid-stream cancel is cooperative, not kill-based."""
        r = _make_runner()

        def _child():
            cmd = r._req_q.get(timeout=5)
            assert cmd[0] == "chat_stream"
            seq = cmd[2]   # the real HFRunner.chat_stream now sends a seq
            r._resp_q.put(("chunk", "Hel"))
            r._resp_q.put(("chunk", "lo"))
            msg = r._ctrl_q.get(timeout=5)
            assert msg == ("cancel_stream", seq), (
                "the real _cancel_stream_and_drain must echo THIS stream's "
                "own seq, not a bare ('cancel_stream',) - see "
                "_ctrl_msg_cancels_seq")
            r._resp_q.put(("done", {}))

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            gen = r.chat_stream(messages=[])
            assert next(gen) == "Hel"
            gen.close()   # what a real client disconnect does
        finally:
            child.join(2)

        assert not r._proc.terminated, (
            "a cooperative cancel must not kill the worker - it confirmed "
            "'done' without ever needing shutdown(grace=0)")
        assert r.is_alive()
        assert r._req_q is not None, "a live worker's queues must not be torn down"

    def test_drain_timeout_falls_back_to_killing_the_worker(self, monkeypatch):
        """If the child never confirms cancel_stream (a genuinely wedged native
        call), the drain does NOT assume success: it falls back to killing the
        worker.

        The fallback fires only AFTER a genuine cooperative attempt: the fake
        child records the real ctrl_q message it received (with its seq) before
        wedging, so a short-circuit straight to shutdown(grace=0) - which leaves
        the same terminated/_req_q-is-None end state - is still caught."""
        monkeypatch.setattr(runner_mod, "_CANCEL_DRAIN_TIMEOUT", 0.3)
        r = _make_runner()
        fake_proc = r._proc
        stop = threading.Event()
        received_cancel = []

        def _child():
            # The value is unused; the get() is the synchronisation point that
            # makes the child wait for the request before replying.
            r._req_q.get(timeout=5)
            r._resp_q.put(("chunk", "hi"))
            try:
                received_cancel.append(r._ctrl_q.get(timeout=2))
            except _queue.Empty:
                pass   # recorded as empty below; the outer assert catches it
            stop.wait(30)   # wedged AFTER receiving it: never confirms "done"

        child = threading.Thread(target=_child, daemon=True)
        child.start()
        try:
            gen = r.chat_stream(messages=[])
            assert next(gen) == "hi"
            gen.close()
        finally:
            stop.set(); child.join(2)

        assert received_cancel, (
            "the drain timed out without the child ever seeing a "
            "cancel_stream message - the cooperative attempt was skipped "
            "entirely, not genuinely attempted-then-timed-out")
        assert received_cancel[0][0] == "cancel_stream"
        assert fake_proc.terminated, (
            "a drain timeout must fall back to killing the worker, not "
            "silently assume the cancel succeeded")
        assert r._req_q is None, "shutdown() must have torn down the queues"


# --------------------------------------------------------------------------- #
# Direct unit coverage for _ctrl_msg_cancels_seq (the decision function that
# closes the stale-cancel race) and _CancelCriteria (the StoppingCriteria
# transformers calls). Neither needs a real subprocess or a real model, so both
# run at the fast/non-integration tier.
# --------------------------------------------------------------------------- #

from localm.inference.backends._hf_runner import _ctrl_msg_cancels_seq  # noqa: E402


class TestCtrlMsgCancelsSeq:
    """_ctrl_msg_cancels_seq is what _control_loop calls to decide whether a
    drained ctrl_q message should actually stream_cancel_event.set(). ctrl_q
    and req_q are independent queues, so a cancel meant for a stream that
    already finished can still be drained after the dispatch loop has moved on
    to a new, unrelated stream. Pure function, no threads or queues needed."""

    def test_matching_seq_cancels(self):
        assert _ctrl_msg_cancels_seq(("cancel_stream", 7), 7) is True

    def test_stale_seq_does_not_cancel(self):
        """A cancel meant for stream 7, drained after the dispatch loop has
        already moved on to stream 8, must NOT cancel stream 8."""
        assert _ctrl_msg_cancels_seq(("cancel_stream", 7), 8) is False

    def test_no_current_stream_does_not_cancel(self):
        """Before the first stream ever starts, current_seq is None - a
        cancel_stream (which always carries a real int seq from the real
        HFRunner) must never match a None current_seq."""
        assert _ctrl_msg_cancels_seq(("cancel_stream", 1), None) is False

    def test_missing_seq_on_the_message_does_not_cancel(self):
        """A malformed/legacy 1-tuple message carries no target seq at all -
        target_seq is None, and None must never match, even if
        current_seq also happens to be None (that would defeat the whole
        protection when neither side supplies a real seq)."""
        assert _ctrl_msg_cancels_seq(("cancel_stream",), None) is False
        assert _ctrl_msg_cancels_seq(("cancel_stream",), 1) is False

    def test_non_cancel_message_is_ignored(self):
        assert _ctrl_msg_cancels_seq(("something_else", 1), 1) is False

    def test_non_tuple_or_empty_message_is_ignored(self):
        assert _ctrl_msg_cancels_seq(None, 1) is False
        assert _ctrl_msg_cancels_seq((), 1) is False
        assert _ctrl_msg_cancels_seq("cancel_stream", 1) is False


class TestCancelCriteriaUnit:
    """Direct, model-free unit test of _hf_worker.py's _CancelCriteria - the
    StoppingCriteria transformers' generate() loop actually calls once per
    decode step. Needs real torch (to build a real input_ids tensor and check
    the real return dtype/shape contract StoppingCriteriaList relies on under
    its `|` reduction), but no model and no subprocess."""

    def test_cleared_event_reports_all_false(self):
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            pytest.skip("llama.cpp's native runtime is already loaded in this "
                         "process (a real compute-device probe ran earlier in "
                         "this same pytest worker) - a fresh torch import here "
                         "is the known-doomed DLL-identity conflict, not this "
                         "test's own subject")
        torch = pytest.importorskip("torch")
        from localm.inference.backends._hf_worker import _CancelCriteria

        event = threading.Event()
        criteria = _CancelCriteria(event)
        input_ids = torch.zeros((3, 5), dtype=torch.long)   # batch_size=3
        result = criteria(input_ids, scores=None)

        assert result.dtype == torch.bool
        assert tuple(result.shape) == (3,)
        assert not bool(result.any())

    def test_set_event_reports_all_true(self):
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            pytest.skip("llama.cpp's native runtime is already loaded in this "
                         "process (a real compute-device probe ran earlier in "
                         "this same pytest worker) - a fresh torch import here "
                         "is the known-doomed DLL-identity conflict, not this "
                         "test's own subject")
        torch = pytest.importorskip("torch")
        from localm.inference.backends._hf_worker import _CancelCriteria

        event = threading.Event()
        event.set()
        criteria = _CancelCriteria(event)
        input_ids = torch.zeros((2, 5), dtype=torch.long)   # batch_size=2
        result = criteria(input_ids, scores=None)

        assert result.dtype == torch.bool
        assert tuple(result.shape) == (2,)
        assert bool(result.all())

    def test_result_survives_stopping_criteria_list_or_reduction(self):
        """The actual contract that matters: StoppingCriteriaList.__call__
        does `is_done = is_done | criteria(...)` starting from an all-False
        bool tensor - reproduce that reduction directly against a real
        StoppingCriteriaList (not just _CancelCriteria in isolation), so a
        future change to the class that technically returns "a tensor" but
        breaks under `|` (e.g. wrong dtype) is still caught."""
        from localm.inference.backends.llamacpp import _loader
        if _loader.native_lib_loaded():
            pytest.skip("llama.cpp's native runtime is already loaded in this "
                         "process (a real compute-device probe ran earlier in "
                         "this same pytest worker) - a fresh torch import here "
                         "is the known-doomed DLL-identity conflict, not this "
                         "test's own subject")
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        from localm.inference.backends._hf_worker import _CancelCriteria

        event = threading.Event()
        event.set()
        criteria_list = transformers.StoppingCriteriaList([_CancelCriteria(event)])
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        is_done = torch.zeros(1, dtype=torch.bool)
        is_done = is_done | criteria_list(input_ids, scores=None)
        assert bool(is_done.all())
