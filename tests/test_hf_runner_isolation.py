# SPDX-License-Identifier: AGPL-3.0-or-later
"""Crash-containment tests for backends/_hf_runner.py's HFRunner.

HFWorker's native calls (a "fast" tokenizer's Rust regex pre-tokenizer stage,
a torch forward pass, model.generate()) are uninterruptible from Python: a
catastrophic-backtracking tokenizer pattern or a genuinely wedged native call
hangs the calling thread forever, and nothing short of killing the process
reclaims it. Before this isolation existed, that thread was one of the
server's fixed-size asyncio default-executor workers - so a single hang
permanently burned a pool slot, and 16 of them (this box's pool size)
exhausted the pool entirely, taking down embeddings/model-loads/token
counting for every OTHER model too (see dev-notes/decisions-2026-07-30-
release-gate.md, Q2). localm therefore runs the whole HF backend lifecycle in
an isolated worker process; a crash or hang there kills only the worker, and
the caller gets a clean, catchable error within a bounded time while the
server itself stays up.

These tests prove the containment property with REAL, uncatchable faults (a
hard process exit, a genuine abort, and a hang) injected into the worker via
the LOCALM_HF_FAULT_FOR_TEST hook - the same code path a real hang would
take. Modeled directly on tests/test_gguf_runner_isolation.py (the identical
property for GgufBackend/PR #606) and tests/test_voice_robustness.py (the
STT worker). The final test in this file goes one step further than either
of those: it proves the ACTUAL bug end to end, not just the runner's own
synchronous timeout contract - see TestExecutorPoolReclaim below.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
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
# The premise: a native fault is uncatchable in-process (so isolation is needed).
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # Identical premise to test_gguf_runner_isolation.py's own version of this
    # test - not HF-specific, just re-established here so this file stands
    # alone as evidence for why HF needed the same fix.
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
        # A wedged native call during load must not block forever: the worker
        # is killed at the timeout and a clean, actionable error is raised.
        # This is the runner-level contract; TestExecutorPoolReclaim below
        # proves the same fact ONE LEVEL UP, through run_in_executor.
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
        _runner_main's dispatch -> uncaught -> the process dies), exactly
        mirroring test_gguf_runner_isolation.py's identical test."""
        r = HFRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            assert "native inference fault" in str(ei.value).lower()
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
        equivalent of (GGUF's can_embed is a fixed False), so it has no
        upstream test to mirror and needs its own coverage."""
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
    """Mirrors test_gguf_runner_isolation.py's identical test: the parent's
    crash message always says "see the debug log for the native stack
    trace", so the worker's actual exception detail must really reach that
    log, not just the parent's own generic message."""

    def test_uncaught_dispatch_crash_is_logged_to_the_debug_log(
            self, monkeypatch, tmp_path):
        log_path = tmp_path / "worker_crash.log"
        monkeypatch.setenv("LOCALM_DEBUG", str(log_path))

        r = HFRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                list(r.chat_stream(messages=[{"role": "user", "content": "hi"}]))
            assert "native inference fault" in str(ei.value).lower()
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


# --------------------------------------------------------------------------- #
# THE regression oracle: prove the actual reported bug is fixed, not just the
# runner's own synchronous contract. Neither test_gguf_runner_isolation.py
# nor this file's tests above ever drive the runner through
# loop.run_in_executor() - they all call it synchronously and check its own
# timeout mechanics directly. That proves the runner bounds ITS OWN wait; it
# does not prove an asyncio executor-thread CALLER actually gets its pool
# slot back. This is the literal mechanism the bug report is about.
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
        completes promptly afterward - proof the slot was reclaimed, not
        merely that this coroutine stopped waiting on it (the maintainer's
        own framing: "a timeout at the call site stops waiting, not
        working" - this test asserts the WORKING half, not just the
        waiting half)."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        # Short timeout so the test itself stays fast - the runner's own
        # bounded-wait mechanics are already proven by
        # test_hung_load_times_out_and_is_killed above; this test's job is
        # only to prove the EXECUTOR-THREAD consequence of that bound.
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
