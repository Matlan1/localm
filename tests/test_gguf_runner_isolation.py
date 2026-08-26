# SPDX-License-Identifier: AGPL-3.0-or-later
"""Crash-containment tests for llamacpp/_runner.py's ModelRunner.

llama_load_model_from_file (and every subsequent llama_init_from_model /
llama_decode call - context growth, token-by-token generation) can hard-abort
the whole process on a native CUDA/HIP driver failure: no Python try/except
can catch it. localm therefore runs the entire GGUF model lifecycle in an
isolated worker process; a crash or hang there kills only the worker, and the
caller gets a clean, catchable error while the server itself stays up.

These tests prove the containment property with REAL, uncatchable faults (a
hard process exit, a genuine abort, and a hang) injected into the worker via
the LOCALM_GGUF_FAULT_FOR_TEST hook - the same code path a real driver abort
would take. The premise test shows a native fault bypasses try/except
entirely.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from localm.inference.backends.base import ModelLoadCancelled
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import ModelRunner


@pytest.fixture(autouse=True)
def _clean_fault_env():
    os.environ.pop(runner_mod._FAULT_ENV, None)
    yield
    os.environ.pop(runner_mod._FAULT_ENV, None)


_DUMMY_LOAD_PARAMS = dict(
    model_path="does-not-matter.gguf", mmproj_path=None,
    n_ctx=4096, n_gpu_layers=99, n_ctx_max=None, n_ctx_grow=4096,
)


# --------------------------------------------------------------------------- #
# A native fault is uncatchable in-process.
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # A child that wraps a genuine native abort in `except BaseException` still
    # dies.
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
        # Force the worker to vanish the instant it receives the load command
        # (no Python traceback), exactly like a native abort. This process
        # (pytest) must survive with a clean, catchable error.
        monkeypatch.setenv(runner_mod._FAULT_ENV, "exit")
        r = ModelRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_real_native_abort_during_load_is_contained(self, monkeypatch):
        # A genuine uncatchable native abort (not a clean exit) during load is
        # still contained.
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = ModelRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_hung_load_times_out_and_is_killed(self, monkeypatch):
        # A wedged native call during load must not block forever: the worker
        # is killed at the timeout and a clean, actionable error is raised.
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        r = ModelRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=2.0)
            assert "timed out" in str(ei.value).lower()
            assert not r.is_alive(), "the hung worker must be killed, not left running"
        finally:
            r.shutdown(grace=0)

    def test_bad_load_payload_is_a_clean_error_not_a_crash(self):
        """The GgufWorker(**payload) constructor call sits INSIDE the "load"
        branch's try/except, so a malformed payload - a parent/child protocol
        bug, not a native fault - is a normal, clean error rather than an
        uncaught crash: the RuntimeError carries the real message ("unexpected
        keyword argument"), never the "crashed (exit code ...)" text the sibling
        crash-containment tests above assert on.

        The worker process is still reaped by spawn_and_load before this raises,
        same as every other non-"ok" load outcome (cancelled, unexpected
        envelope) - a failed load never produces a usable model, so nothing is
        left running. See TestLoadFailureReapsWorker below for the count-based
        oracle this is a special case of."""
        r = ModelRunner()
        bad_params = dict(_DUMMY_LOAD_PARAMS, this_kwarg_does_not_exist=True)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(bad_params, timeout=30.0)
            assert "unexpected keyword argument" in str(ei.value).lower()
            assert not r.is_alive(), (
                "a failed load produced no usable model - the worker must "
                "be reaped, not left orphaned for the next retry to pile "
                "another one alongside"
            )
        finally:
            r.shutdown(grace=0)


class TestLoadFailureReapsWorker:
    """engine.py's chat_stream retries GgufBackend.load() on EVERY request
    while the model is not loaded (see engine.py's auto-reload), and
    GgufBackend.load() spawns a BRAND NEW ModelRunner every attempt (see
    gguf.py's _load_native). So an unloadable model that keeps failing must not
    leave a trail of live worker processes behind it: each failed
    spawn_and_load reaps its own worker before it returns to the caller.

    Counts real, spawned multiprocessing.Process objects via is_alive(), never a
    mock: a mock's is_alive()/terminate() say nothing about the real subprocess
    boundary this property lives on."""

    def test_repeated_load_errors_never_leave_a_surviving_worker(self):
        bad_params = dict(_DUMMY_LOAD_PARAMS, this_kwarg_does_not_exist=True)
        runners = []
        try:
            for _ in range(3):
                r = ModelRunner()
                runners.append(r)
                with pytest.raises(RuntimeError, match="unexpected keyword argument"):
                    r.spawn_and_load(bad_params, timeout=30.0)
            survivors = sum(1 for r in runners if r.is_alive())
            assert survivors == 0, (
                f"{survivors} of {len(runners)} workers from a failed "
                "(ERROR) load are still alive - each failed load must reap "
                "its own worker"
            )
        finally:
            for r in runners:
                r.shutdown(grace=0)

    def test_repeated_load_cancellations_never_leave_a_surviving_worker(
            self, monkeypatch):
        # Forces a real child process to report a clean "cancelled" envelope
        # without touching the native runtime; a genuine native cancellation would
        # need a provisioned llama.cpp runtime.
        monkeypatch.setenv(runner_mod._FORCE_LOAD_CANCEL_ENV, "1")
        runners = []
        try:
            for _ in range(3):
                r = ModelRunner()
                runners.append(r)
                with pytest.raises(ModelLoadCancelled):
                    r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
            survivors = sum(1 for r in runners if r.is_alive())
            assert survivors == 0, (
                f"{survivors} of {len(runners)} workers from a "
                "CANCELLED load are still alive - each cancelled load must "
                "reap its own worker"
            )
        finally:
            for r in runners:
                r.shutdown(grace=0)


class TestRunnerLifecycle:
    def test_is_alive_false_before_spawn_and_after_shutdown(self):
        r = ModelRunner()
        assert not r.is_alive()
        r.shutdown()   # must be a no-op, never raise, when nothing is running

    def test_double_shutdown_is_safe(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "exit")
        r = ModelRunner()
        with pytest.raises(RuntimeError):
            r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
        r.shutdown(grace=0)
        r.shutdown(grace=0)   # must not raise the second time


class TestChatStreamCrashContainment:
    def test_dispatch_crash_during_chat_stream_is_contained(self):
        """An uncaught exception ANYWHERE in the child's dispatch loop - not
        just a real native abort - must crash only the child, never the
        caller: this is the exact contract chat_stream() relies on for a
        genuine mid-generation native fault (GgufWorker.chat_stream re-raises an
        unrecoverable fault uncaught - see its docstring), reproduced here
        without needing a real model or GPU by sending chat_stream before any
        load (worker is None -> AttributeError inside _runner_main's dispatch ->
        uncaught -> the process dies).

        An AttributeError exits 1, which this module's own _runner_entry
        docstring identifies as multiprocessing's signature for an uncaught
        PYTHON exception, so the message must NOT say "native inference fault":
        there is no native fault, no native trace, and the model is
        unharmed."""
        r = ModelRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            msg = str(ei.value).lower()
            # Contained and reported (the property under test)...
            assert "unloaded" in msg and "reload" in msg, msg
            # ...and NOT mislabelled as a native fault, because it was not one.
            assert "native inference fault" not in msg, msg
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)


class TestSimpleRequestCrashContainment:
    def test_count_tokens_crash_is_contained(self):
        """A crash while handling a simple request (not just load/chat_stream)
        must also be contained and reported cleanly, never left hanging.

        Kills the worker directly rather than using the env-var fault hook:
        LOCALM_GGUF_FAULT_FOR_TEST is read from the CHILD's own os.environ,
        which is a snapshot taken at spawn time - mutating the parent's
        os.environ afterwards can never reach an already-running child. A
        direct kill produces the identical observable effect (the process
        vanishes before answering) that _simple_request's is_alive() check
        must detect."""
        r = ModelRunner()
        r._spawn()
        r._proc.kill()
        r._proc.join(timeout=5)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.count_tokens("hello")
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)


class TestCrashDiagnosticsReachDebugLog:
    """The parent's crash message always says "see the debug log for the
    native stack trace" (e.g. ModelRunner.chat_stream's "Native inference
    fault" RuntimeError above), so the worker must redirect native stderr into
    that log for its WHOLE life, not only during the narrow model-load and
    generation windows (_quiet_stderr / _capture_stderr / dedup_native_stderr in
    llama.py). A fault INSIDE the generation window is not enough either:
    dedup_native_stderr's fd-2 restore (its context manager __exit__) runs as
    the exception unwinds THROUGH it, before the exception reaches
    multiprocessing's own traceback.print_exc() in BaseProcess._bootstrap, so
    the one traceback that would explain the crash lands on an already-restored
    fd 2 (closed/NUL for a GUI-launched, console-less process).

    Reuses test_dispatch_crash_during_chat_stream_is_contained's exact fault
    (chat_stream sent before any load -> worker is None -> AttributeError
    escapes _runner_main uncaught): it reproduces "an exception _runner_main
    lets escape", needs no real model or GPU, and is unaffected by the "load"
    branch's constructor call (see TestLoadCrashContainment)."""

    def test_uncaught_dispatch_crash_is_logged_to_the_debug_log(
            self, monkeypatch, tmp_path):
        log_path = tmp_path / "worker_crash.log"
        # A real path (not "1"/"true") so log_file_path() resolves it exactly like
        # a child that inherited an already-open debug log from its parent server
        # process.
        monkeypatch.setenv("LOCALM_DEBUG", str(log_path))

        r = ModelRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            # This fault is an AttributeError (exit 1), an uncaught PYTHON
            # exception, so the message must NOT call it a native fault.
            assert "native inference fault" not in str(ei.value).lower()
            # is_alive() only reports False once the OS has reported the process
            # fully exited, so any write the child's Python code did before dying is
            # already flushed and visible to this read; no wait or retry needed.
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)

        assert log_path.is_file(), (
            "the crashed worker never even attached the shared debug log"
        )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "AttributeError" in text and "NoneType" in text, (
            "the worker crashed with no envelope (by design - see "
            "GgufWorker.chat_stream's docstring), but its actual exception "
            "detail never reached the debug log - the parent's own "
            "'see the debug log for the native stack trace' message is a "
            "lie if there is nothing in that log to see\n--- log content ---\n"
            + text
        )


class TestNativeSignalCrashDiagnosticsReachDebugLog:
    """The class above closes the gap for a crash that still has a PYTHON
    exception (``logger.critical(exc_info=True)`` in ``_runner_entry`` catches
    it). It cannot close the gap for the OTHER half, and ``_runner_entry``'s own
    docstring says so: "Does NOT help a genuine native crash with no Python
    exception at all (SIGSEGV, a raw abort with nothing printed first) - Python
    never regains control there, so no ``except`` clause, including this one, can
    run."

    That residual half is ``Native inference fault (worker exit -4)``. On Linux
    ``multiprocessing`` reports ``-N`` for death by signal N, so ``-4`` is
    SIGILL: an illegal instruction inside native code, with no Python exception
    anywhere. The parent's message tells the user "See the debug log for the
    native stack trace", so something has to be written for that class.

    The two arms this test rests on:

    * a real SIGILL (an mmap'd ``0f 0b``) on Linux, and ``os.abort()`` on
      Windows, both produce a full "Fatal Python error" trace naming the Python
      frame that entered native code - but ONLY with faulthandler armed;
    * with it disarmed, the destination file is EMPTY on both platforms. That is
      the negative control, so a pass here cannot come from the trace having
      been written by something else.

    ``os.abort()`` (the ``LOCALM_GGUF_FAULT_FOR_TEST=abort`` hook) is used rather
    than a synthetic SIGILL because it is the same CLASS - the process dies from
    a native signal with no Python exception - and it is the project's existing
    way to produce that class in a REAL child process.
    """

    def _fault_during_chat_stream(self, monkeypatch):
        """Drive a real worker to a real native abort mid-chat_stream. Returns
        ``(message, trace_path)``.

        The fault env var is set BEFORE ``_spawn()``: the child reads
        it from its OWN ``os.environ``, which is a snapshot taken at spawn time, so
        setting it afterwards could never reach the running child (the same trap
        documented on test_count_tokens_crash_is_contained above)."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")

        r = ModelRunner()
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
        """The reported symptom, inverted: after a native-signal death the caller
        must be told WHAT faulted, not merely that something did.

        Asserts on the TRACE CONTENT rather than on the exit code or the
        presence of the words "native inference fault": the trace text is the
        only thing that distinguishes a captured fault from an uncharacterised
        one."""
        with caplog.at_level(logging.ERROR, logger="localm"):
            message, _ = self._fault_during_chat_stream(monkeypatch)

        assert "native inference fault" in message.lower()
        assert "Fatal Python error" in message, (
            "a real native-signal death produced no captured trace, so the "
            "caller still cannot tell WHICH native call faulted - the reported "
            f"issue 1222 / 1223 symptom\n--- message ---\n{message}"
        )

        # The FULL multi-line trace, not just the summary line folded into the
        # message, reaches the debug log. caplog rather than a real log file:
        # attaching a handler to the shared "localm" logger would leak into every
        # later test.
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "Fatal Python error" in logged and "_runner.py" in logged, (
            "the trace never reached the localm logger, or names no Python "
            f"frame, so it cannot say where the fault happened\n{logged}"
        )

    def test_no_trace_captured_is_stated_not_implied(self, monkeypatch):
        """When nothing was captured the message must SAY so, rather than
        pointing the reader at a debug-log trace that was never written."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = ModelRunner()
        r._spawn()
        # Simulate the capture having failed (an unwritable logs dir, a platform
        # where enable() no-ops) by dropping the path the parent would read.
        r._crash_trace_path = None
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            # The wording is "for this EXIT", not "for this FAULT": the no-trace
            # branch does not classify the exit.
            assert "no native fault trace was captured" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_native_crash_trace_file_is_consumed_not_left_behind(
            self, monkeypatch):
        """The per-worker trace file must not survive the crash it describes: a
        stale file would be misread as a fresh crash by the next reader.

        Asserts the trace was CAPTURED as well as gone. Without that first half
        this test passes vacuously when nothing ever wrote the file - which is
        exactly what the fires-control caught it doing: with the arming call
        removed, "the file does not exist" is trivially true and the test could
        not fail on the defect it was written for."""
        message, trace_path = self._fault_during_chat_stream(monkeypatch)
        assert trace_path is not None
        assert "Fatal Python error" in message, (
            "nothing was captured, so this test would be asserting cleanup of a "
            f"file that never existed\n--- message ---\n{message}")
        assert not trace_path.exists(), (
            f"the worker crash-trace file was left behind at {trace_path}")

    def test_healthy_worker_arms_a_trace_then_reaps_it(self):
        """The capture costs one empty file per model load, so a clean shutdown
        has to reap it or a long-running server slowly fills its own logs dir.

        Checks the file EXISTS while the worker is alive before checking it is
        gone afterwards - same reason as the test above: "absent at the end" is
        satisfied just as well by never having armed at all, so on its own it
        proves nothing about either arming or cleanup.

        The existence check is POLLED, not immediate. ``_spawn()`` returns as
        soon as ``Process.start()`` does, and a spawn-context child then has to
        boot a fresh interpreter and run its imports before it arms anything, so
        the poll waits for the file to appear while the worker stays alive."""
        r = ModelRunner()
        r._spawn()
        trace_path = r._crash_trace_path
        try:
            assert trace_path is not None
            deadline = time.monotonic() + 30.0
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


class TestSimpleRequestNativeSignalCrashDiagnosticsReachDebugLog:
    """The trace relay TestNativeSignalCrashDiagnosticsReachDebugLog pins for
    chat_stream() must also hold for _simple_request() - the code path
    count_tokens()/check_grammar() go through. A native-crash branch that
    reports the exit code without calling _crash_detail() never reads a trace
    the child DID capture: ModelRunner.shutdown()'s own comment says a trace
    surviving to teardown is "either already relayed or describes a death nobody
    is going to report" and discards it unconditionally, so a fault during a
    token count or a grammar check would leave a bare exit code and no
    trace."""

    def test_native_abort_during_count_tokens_is_reported_with_its_trace(
            self, monkeypatch, caplog):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = ModelRunner()
        r._spawn()
        try:
            with caplog.at_level(logging.ERROR, logger="localm"):
                with pytest.raises(RuntimeError) as ei:
                    r.count_tokens("hello")
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)

        message = str(ei.value)
        assert "crashed" in message.lower()
        assert "Fatal Python error" in message, (
            "a real native-signal death during count_tokens produced no "
            "captured trace in the error - the same issue 1222/1223 symptom "
            f"chat_stream's fix already closed\n--- message ---\n{message}")

        # The FULL trace reaches the debug log too, not just the summary line
        # folded into the message.
        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Fatal Python error" in logged and "_runner.py" in logged, (
            f"the trace never reached the localm logger\n{logged}")


# --------------------------------------------------------------------------- #
# An asyncio executor-thread caller (the shape routes/chat.py uses:
# run_in_executor(None, engine.load)) gets its pool slot back after a hung load.
# --------------------------------------------------------------------------- #

class TestExecutorPoolReclaim:
    def test_hung_call_through_run_in_executor_frees_its_pool_slot(
            self, monkeypatch, tmp_path):
        """Drive a hung GgufBackend.load() through loop.run_in_executor(pool,
        ...) exactly as production code does, on a tiny
        (2-worker) pool so exhaustion is observable without needing 16
        concurrent hangs. Two assertions: (a) the call raises within its
        bounded timeout instead of hanging forever, and (b) a SUBSEQUENT,
        unrelated run_in_executor call on the SAME pool completes promptly
        afterward - proof the slot was reclaimed, not merely that this
        coroutine stopped waiting on it (the maintainer's own framing: "a
        timeout at the call site stops waiting, not working" - this test
        asserts the WORKING half, not just the waiting half)."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        # Short timeout so the test itself stays fast; this test covers the
        # EXECUTOR-THREAD consequence of that bound, not the bound itself.
        monkeypatch.setattr(GgufBackend, "_load_timeout_seconds", staticmethod(lambda: 2.0))

        # A tiny REAL file (not a bare string path) so GgufBackend.load()'s
        # preflight succeeds: _model_bytes() needs stat() to work, both for the
        # normal preflight and for the except-path VRAM hint that runs on any load
        # failure. A few real bytes, never a truncate() (not sparse on NTFS).
        # n_gpu_layers=0 keeps n_gpu_layers_auto/ctx_auto (both default off) out of
        # the way, so _effective_gpu_layers/_check_vram resolve with zero VRAM
        # probing and nothing touches the abort-prone native call.
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"\0" * 4096)

        async def scenario():
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-pool")
            loop = asyncio.get_running_loop()
            engine = GgufBackend(str(model_path), n_gpu_layers=0)
            try:
                t0 = time.monotonic()
                with pytest.raises(RuntimeError, match="(?i)timed out"):
                    await loop.run_in_executor(pool, engine.load)
                elapsed = time.monotonic() - t0
                assert elapsed < 10.0, (
                    f"load() should bound at ~2s (gguf_load_timeout_s "
                    f"override), took {elapsed:.1f}s - the executor thread "
                    "was not freed promptly")

                # An unrelated call on the same pool still completes promptly,
                # bounded by asyncio.wait_for so a never-freed slot fails FAST
                # instead of hanging the suite.
                result = await asyncio.wait_for(
                    loop.run_in_executor(pool, lambda: 1 + 1), timeout=5.0)
                assert result == 2
            finally:
                pool.shutdown(wait=False)

        asyncio.run(scenario())
