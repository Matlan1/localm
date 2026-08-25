# SPDX-License-Identifier: AGPL-3.0-or-later
"""Crash-containment tests for llamacpp/_runner.py's ModelRunner."""

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
# The premise: a native fault is uncatchable in-process (so isolation is needed).
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # A child that wraps a genuine native abort in `except BaseException` still
    # dies: this is exactly why an in-process llama_load_model_from_file crash
    # takes the whole server down, and why the real fix is process isolation,
    # not a try/except (see tests/test_voice_robustness.py for the identical
    # premise proven for the STT worker).
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
        # The gold standard: a genuine uncatchable native abort (not a clean
        # exit) during load is still contained.
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
        """The GgufWorker(**payload) constructor call now sits INSIDE the 'load' branch's try/except (previously only worker.load() was guarded - see _runner.py), so a malformed payload - a parent/child protocol bug, not a native fault - is a normal, clean error, not an uncaught crash: the RuntimeError carries..."""
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
    """The regression oracle for the worker-leak fix: engine.py's chat_stream retries GgufBackend.load() on EVERY request while the model is not loaded (see engine.py's auto-reload), and GgufBackend.load() spawns a BRAND NEW ModelRunner every attempt (see gguf.py's _load_native)."""

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
        # without touching the native runtime - see _FORCE_LOAD_CANCEL_ENV's
        # own docstring in _runner.py for why this hook exists rather than
        # driving a genuine native cancellation (it would need a real,
        # provisioned llama.cpp runtime, which this suite's default
        # selection deliberately does not depend on).
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
        """An uncaught exception ANYWHERE in the child's dispatch loop - not just a real native abort - must crash only the child, never the caller: this is the exact contract chat_stream() relies on for a genuine mid-generation native fault (GgufWorker.chat_stream deliberately re-raises an unrecoverable fault..."""
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
        """A crash while handling a simple request (not just load/chat_stream) must also be contained and reported cleanly, never left hanging."""
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
    """The parent's crash message always says 'see the debug log for the native stack trace' (e.g. ModelRunner.chat_stream's 'Native inference fault' RuntimeError above) - but the worker only redirected native stderr into that log during the narrow model-load and generation windows (_quiet_stderr / _captur..."""

    def test_uncaught_dispatch_crash_is_logged_to_the_debug_log(
            self, monkeypatch, tmp_path):
        log_path = tmp_path / "worker_crash.log"
        # A real path (not "1"/"true") so log_file_path() resolves it exactly
        # like a child that inherited an already-open debug log from its
        # parent server process - see debuglog.log_file_path().
        monkeypatch.setenv("LOCALM_DEBUG", str(log_path))

        r = ModelRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            # Same correction as TestChatStreamCrashContainment above: this fault
            # is an AttributeError (exit 1), an uncaught PYTHON exception, so the
            # message must NOT call it a native fault. What this test is really
            # about is the DEBUG LOG content asserted below, not the wording.
            assert "native inference fault" not in str(ei.value).lower()
            # is_alive() only ever reports False once the OS has reported the
            # process fully exited - so any write the child's Python code did
            # before dying (including this fix's logger.critical call) is
            # already flushed and visible to this read; no wait/retry needed.
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
    """The class above closes the gap for a crash that still has a PYTHON exception (``logger.critical(exc_info=True)`` in ``_runner_entry`` catches it)."""

    def _fault_during_chat_stream(self, monkeypatch):
        """Drive a real worker to a real native abort mid-chat_stream."""
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
        """The reported symptom, inverted: after a native-signal death the caller must be told WHAT faulted, not merely that something did."""
        with caplog.at_level(logging.ERROR, logger="localm"):
            message, _ = self._fault_during_chat_stream(monkeypatch)

        assert "native inference fault" in message.lower()
        assert "Fatal Python error" in message, (
            "a real native-signal death produced no captured trace, so the "
            "caller still cannot tell WHICH native call faulted - the reported "
            f"issue 1222 / 1223 symptom\n--- message ---\n{message}"
        )

        # The FULL multi-line trace (not just the summary line folded into the
        # message) has to reach the debug log, because that is where the message
        # sends the user. caplog rather than a real log file: attaching a handler
        # to the shared "localm" logger would leak into every later test.
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "Fatal Python error" in logged and "_runner.py" in logged, (
            "the trace never reached the localm logger, or names no Python "
            f"frame, so it cannot say where the fault happened\n{logged}"
        )

    def test_no_trace_captured_is_stated_not_implied(self, monkeypatch):
        """When nothing was captured the message must SAY so."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = ModelRunner()
        r._spawn()
        # Simulate the capture having failed (an unwritable logs dir, a platform
        # where enable() no-ops) by dropping the path the parent would read.
        r._crash_trace_path = None
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            # Wording tightened deliberately: "for this EXIT", not "for this
            # FAULT". Calling it a fault in the no-trace branch pre-judges the
            # very classification this message now declines to assert.
            assert "no native fault trace was captured" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_native_crash_trace_file_is_consumed_not_left_behind(
            self, monkeypatch):
        """The per-worker trace file must not survive the crash it describes: a stale file would be misread as a fresh crash by the next reader."""
        message, trace_path = self._fault_during_chat_stream(monkeypatch)
        assert trace_path is not None
        assert "Fatal Python error" in message, (
            "nothing was captured, so this test would be asserting cleanup of a "
            f"file that never existed\n--- message ---\n{message}")
        assert not trace_path.exists(), (
            f"the worker crash-trace file was left behind at {trace_path}")

    def test_healthy_worker_arms_a_trace_then_reaps_it(self):
        """The capture costs one empty file per model load, so a clean shutdown has to reap it or a long-running server slowly fills its own logs dir."""
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
    """The trace-relay fix TestNativeSignalCrashDiagnosticsReachDebugLog proves for chat_stream() must also hold for _simple_request() - the code path count_tokens()/check_grammar() go through."""

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

        # The FULL trace has to reach the debug log too, not just the
        # summary line folded into the message - same requirement as
        # chat_stream's own version of this test above.
        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Fatal Python error" in logged and "_runner.py" in logged, (
            f"the trace never reached the localm logger\n{logged}")


# --------------------------------------------------------------------------- #
# THE regression oracle: prove the actual reported bug is fixed for
# GgufBackend too, not just the runner's own synchronous timeout contract.
# Every test above calls ModelRunner directly and checks its own bounded-wait
# mechanics; none of them prove that an asyncio executor-thread CALLER (the
# shape production code actually uses - routes/chat.py's
# run_in_executor(None, engine.load), etc.) gets its pool slot back after a
# hang. Mirrors TestExecutorPoolReclaim in test_hf_runner_isolation.py, where
# the identical gap was found and closed for HFBackend (PR #947 / dev-notes/
# decisions-2026-07-30-release-gate.md Q2).
# --------------------------------------------------------------------------- #

class TestExecutorPoolReclaim:
    def test_hung_call_through_run_in_executor_frees_its_pool_slot(
            self, monkeypatch, tmp_path):
        """Drive a hung GgufBackend.load() through loop.run_in_executor(pool, ...) exactly as production code does, on a deliberately tiny (2-worker) pool so exhaustion is observable without needing 16 concurrent hangs."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        # Short timeout so the test itself stays fast - the runner's own
        # bounded-wait mechanics are already proven by
        # test_hung_load_times_out_and_is_killed above; this test's job is
        # only to prove the EXECUTOR-THREAD consequence of that bound.
        monkeypatch.setattr(GgufBackend, "_load_timeout_seconds", staticmethod(lambda: 2.0))

        # A tiny REAL file (not a bare string path) so GgufBackend.load()'s
        # preflight succeeds: _model_bytes() needs stat() to work, both for
        # the normal preflight and for the except-path VRAM hint that runs
        # on ANY load failure (including this timeout). Same disk-safe
        # pattern as test_vram_preflight.py's _backend() helper - a few
        # real bytes, never a truncate() (not sparse on Windows/NTFS).
        # n_gpu_layers=0 keeps n_gpu_layers_auto/ctx_auto (both default off)
        # out of the way too, so _effective_gpu_layers/_check_vram resolve
        # with zero VRAM probing - nothing here touches the abort-prone
        # native call before the runner's own timeout-and-kill ends the load.
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

                # THE oracle: an unrelated call on the same pool must still
                # complete promptly. Bounded by asyncio.wait_for so a
                # regression (the slot never freed) fails FAST, not by
                # hanging the test suite itself.
                result = await asyncio.wait_for(
                    loop.run_in_executor(pool, lambda: 1 + 1), timeout=5.0)
                assert result == 2
            finally:
                pool.shutdown(wait=False)

        asyncio.run(scenario())
