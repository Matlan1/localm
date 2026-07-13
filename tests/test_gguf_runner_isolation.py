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
entirely (which is why isolation, not a try/except, is the fix). Modeled
directly on tests/test_voice_robustness.py, which proves the identical
property for the STT worker (localm/voice.py).
"""

import os
import subprocess
import sys

import pytest

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
        genuine mid-generation native fault (GgufWorker.chat_stream
        deliberately re-raises an unrecoverable fault uncaught - see its
        docstring), reproduced here without needing a real model or GPU by
        sending chat_stream before any load (worker is None -> AttributeError
        inside _runner_main's dispatch -> uncaught -> the process dies)."""
        r = ModelRunner()
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
