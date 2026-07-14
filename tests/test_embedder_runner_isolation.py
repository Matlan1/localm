# SPDX-License-Identifier: AGPL-3.0-or-later
"""Crash-containment tests for _embedder_runner.py's EmbedderRunner.

llama_load_model_from_file (and llama_decode, called by every embed()) can
hard-abort the whole process on a native CUDA/HIP driver failure: no Python
try/except can catch it. The embedder therefore runs its whole lifecycle in
an isolated worker process (mirroring the chat backend's containment in
llamacpp/_runner.py, and localm/voice.py's for STT) - a crash or hang there
kills only the worker, and the caller gets a clean, catchable error while the
server itself stays up.

These tests prove the containment property with REAL, uncatchable faults (a
hard process exit, a genuine abort, and a hang) injected into the worker via
the LOCALM_EMBEDDER_FAULT_FOR_TEST hook - the same code path a real driver
abort would take. Modeled directly on tests/test_gguf_runner_isolation.py.
"""

import os

import pytest

from localm.inference import _embedder_runner as runner_mod
from localm.inference._embedder_runner import EmbedderRunner


@pytest.fixture(autouse=True)
def _clean_fault_env():
    os.environ.pop(runner_mod._FAULT_ENV, None)
    yield
    os.environ.pop(runner_mod._FAULT_ENV, None)


_DUMMY_LOAD_PARAMS = dict(
    model_path="does-not-matter.gguf", n_gpu_layers=99, n_ctx=512, pooling_type=1,
)


# --------------------------------------------------------------------------- #
# The premise: a native fault is uncatchable in-process (so isolation is needed).
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # Mirrors test_gguf_runner_isolation.py's identical premise test: a child
    # that wraps a genuine native abort in `except BaseException` still dies -
    # this is exactly why an in-process llama_load_model_from_file/llama_decode
    # crash takes the whole server down, and why the real fix is process
    # isolation, not a try/except.
    import subprocess
    import sys

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
        r = EmbedderRunner()
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
        r = EmbedderRunner()
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
        r = EmbedderRunner()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=2.0)
            assert "timed out" in str(ei.value).lower()
            assert not r.is_alive(), "the hung worker must be killed, not left running"
        finally:
            r.shutdown(grace=0)


class TestRunnerLifecycle:
    def test_is_alive_false_before_spawn_and_after_shutdown(self):
        r = EmbedderRunner()
        assert not r.is_alive()
        r.shutdown()   # must be a no-op, never raise, when nothing is running

    def test_double_shutdown_is_safe(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "exit")
        r = EmbedderRunner()
        with pytest.raises(RuntimeError):
            r.spawn_and_load(_DUMMY_LOAD_PARAMS, timeout=30.0)
        r.shutdown(grace=0)
        r.shutdown(grace=0)   # must not raise the second time


class TestEmbedCrashContainment:
    def test_real_native_abort_while_handling_embed_is_contained(self, monkeypatch):
        """A genuine native abort while the child is dispatching an 'embed'
        command is contained exactly like a load-time abort - the parent's
        detection (proc.is_alive()/exitcode) is a pure process-level check
        that does not care WHICH command the child happened to be running,
        so this proves the same mechanism covers embed() too, without needing
        a real model/GPU to reach a successful load first."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = EmbedderRunner()
        r._spawn()   # embed(), like ModelRunner.chat_stream(), assumes a prior
                      # spawn_and_load() - spawn directly so "embed" is the
                      # first (and only) command the child dispatches.
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=30.0)
            assert "crashed" in str(ei.value).lower()
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)

    def test_hung_embed_times_out_and_is_killed(self, monkeypatch):
        monkeypatch.setenv(runner_mod._FAULT_ENV, "hang")
        r = EmbedderRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=2.0)
            assert "timed out" in str(ei.value).lower()
            assert not r.is_alive()
        finally:
            r.shutdown(grace=0)

    def test_embed_on_dead_worker_reports_clean_error(self):
        """A crash discovered while handling a plain request (not just
        load/embed's own dispatch) must also be contained and reported
        cleanly. Kills the worker directly rather than via the env-var hook:
        LOCALM_EMBEDDER_FAULT_FOR_TEST is read from the CHILD's own environ,
        a snapshot taken at spawn time - mutating the parent's environ
        afterwards can never reach an already-running child. A direct kill
        produces the identical observable effect (the process vanishes before
        answering) that embed()'s is_alive() check must detect."""
        r = EmbedderRunner()
        r._spawn()
        r._proc.kill()
        r._proc.join(timeout=5)
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=10.0)
            assert "crashed" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_embed_dispatch_catches_ordinary_exceptions_without_crashing(self):
        """Unlike the chat backend's chat_stream (which deliberately lets any
        OTHER fault propagate uncaught, since generation leaves the model in
        an unknown state - see llamacpp/_runner.py), the embedder's dispatch
        loop catches ordinary Python exceptions during 'embed' and reports
        them as a clean error WITHOUT killing the worker: embedding is
        stateless per call, so one bad request should not take down a worker
        that could otherwise keep serving. Reproduced without a real model by
        sending 'embed' before any 'load' (embedder is None -> AttributeError,
        caught by the dispatch loop's own except Exception)."""
        r = EmbedderRunner()
        r._spawn()
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=10.0)
            assert "crashed" not in str(ei.value).lower()
            assert r.is_alive(), "an ordinary exception must not kill the worker"
        finally:
            r.shutdown(grace=0)
