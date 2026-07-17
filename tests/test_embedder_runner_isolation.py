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

import multiprocessing as mp
import os
import queue as _queue
import threading

import pytest

from localm.inference import _embedder_runner as runner_mod
from localm.inference._embedder_runner import EmbedderRunner
from localm.inference.embedder import IsolatedEmbedder


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


# --------------------------------------------------------------------------- #
# Concurrent callers must not receive each other's vectors (REG-643).
# --------------------------------------------------------------------------- #

def _vec_for(text: str):
    """A deterministic, input-derived vector, so a response delivered to the
    WRONG caller is detectable by both length and content."""
    return [float(len(text)), float(ord(text[0])), 0.0, 0.0]


class _AliveProc:
    """Stands in for the worker process's liveness check ONLY. _wait() polls
    proc.is_alive() whenever a resp_q poll times out (which the deliberate
    overlap window below guarantees). The correlation-free transport that
    REG-643 is about - the real mp.Queue pair and the real parent-side
    put/get - is NOT substituted."""

    def is_alive(self):
        return True

    exitcode = None


class _CountingRunner(EmbedderRunner):
    """A REAL EmbedderRunner (real queues, real embed()/_wait()) plus a probe
    recording how many RPCs are in flight at once."""

    def __init__(self):
        super().__init__()
        self._probe_lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def embed(self, texts, timeout=runner_mod._EMBED_TIMEOUT_DEFAULT):
        with self._probe_lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            return super().embed(texts, timeout=timeout)
        finally:
            with self._probe_lock:
                self.inflight -= 1


def _fifo_child(req_q, resp_q, stop):
    """Mimics _runner_main's observable protocol exactly: FIFO, one command at
    a time, ("ok", vectors) per "embed". Substitutes only the llama.cpp math
    (and the process boundary), never the parent-side code under test. The
    delay holds each request long enough that an UNSERIALIZED second caller is
    already blocked in resp_q.get() before the first response is posted - which
    is precisely when the real transport misdelivers."""
    while not stop.is_set():
        try:
            cmd = req_q.get(timeout=0.05)
        except _queue.Empty:
            continue
        if cmd[0] != "embed":
            continue
        threading.Event().wait(0.3)
        resp_q.put(("ok", [_vec_for(t) for t in cmd[1]]))


class TestCleanEmbedErrorKeepsTheWorker:
    """A clean embed error must NOT orphan a healthy worker.

    The child answers an ordinary embed failure with an ("error", msg) envelope
    and keeps serving (embedding is stateless per call - see
    test_embed_dispatch_catches_ordinary_exceptions_without_crashing above). The
    parent sees the SAME RuntimeError for that as for a real crash, so dropping
    self._runner unconditionally left a LIVE child blocked on req_q.get() with
    the model still resident in VRAM: EmbedderRunner has no __del__, GC never
    terminates an mp.Process, and close()/reset_embedder()/release_for_exit() all
    only reach the CURRENT runner - so the orphan was unreachable, survived even
    the os._exit/os.execv restart path, and the next call spawned a second worker
    beside it. One leaked worker per clean embed error (24 MB for the default
    bge-small, up to 7.49 GB for a configured Qwen3-Embedding-8B).
    """

    def test_a_clean_error_keeps_the_live_worker_instead_of_orphaning_it(self, monkeypatch):
        # A REAL worker process, spawned with no model loaded: "embed" then hits
        # embedder is None -> AttributeError -> the child's own except -> a clean
        # ("error", ...) envelope, with the child still alive. Exactly the shape
        # a real "embedding decode failed (code N)" takes.
        runner = EmbedderRunner()
        runner._spawn()
        try:
            def fake_reload(self):
                self._runner = runner
                self.dim = 4

            monkeypatch.setattr(IsolatedEmbedder, "_reload", fake_reload)
            e = IsolatedEmbedder("does-not-matter.gguf")

            with pytest.raises(RuntimeError):
                e.embed(["hello"])

            assert runner.is_alive(), "precondition: the child survives a clean error"
            assert e._runner is runner, (
                "a healthy worker was discarded after a clean embed error: it is "
                "now unreachable (no __del__, GC never terminates an mp.Process), "
                "so it keeps its model resident in VRAM forever while the next "
                "embed() spawns a second worker beside it")

            # And it must still serve the next call rather than be respawned.
            with pytest.raises(RuntimeError):
                e.embed(["again"])
            assert e._runner is runner
            assert runner.is_alive()
        finally:
            runner.shutdown(grace=0)

    def test_a_dead_worker_is_still_dropped_and_torn_down(self, monkeypatch):
        """The negative case: auto-reload after a genuine crash must still work,
        so this fix cannot be 'never drop the runner'."""
        class _DeadRunner:
            def __init__(self):
                self.shutdown_calls = []

            def is_alive(self):
                return False

            def embed(self, texts):
                raise RuntimeError("The embedding worker process crashed")

            def shutdown(self, grace=5.0):
                self.shutdown_calls.append(grace)

        dead = _DeadRunner()
        e = IsolatedEmbedder.__new__(IsolatedEmbedder)
        e.model_path = "x.gguf"
        e.active_requests = 0
        e._rpc_lock = threading.RLock()
        e._runner = dead
        # n_gpu_layers=0: this test is about the generic dead-worker/drop-and-
        # reload contract, not the GPU-crash-fallback path (covered separately
        # in test_embedder.py) - keep it CPU-configured so that branch never
        # engages here.
        e.n_gpu_layers = 0
        e.gpu_fallback_reason = None
        # is_alive() is False, so embed() reloads FIRST; keep that reload a no-op
        # so the crash arrives from the RPC itself.
        monkeypatch.setattr(IsolatedEmbedder, "_reload", lambda self: None)

        with pytest.raises(RuntimeError):
            e.embed(["hi"])
        assert e._runner is None, "a dead worker must be dropped so the next call reloads"
        assert dead.shutdown_calls == [0], (
            "a dropped runner must have its queues/handles released, not leaked")


class TestConcurrentEmbedSerialization:
    """The worker protocol has NO request-id correlation: one req_q/resp_q pair
    feeds one child, so two overlapping embed() calls are two threads blocked in
    the same resp_q.get() and the queue hands each whichever response arrives
    first - the caller can get vectors belonging to a DIFFERENT text (wrong
    length, wrong content), silently corrupting what lands in the semantic-memory
    and RAG vector stores. The pre-#643 in-process GGUFEmbedder.embed() held an
    RLock that made this impossible; IsolatedEmbedder.embed() must restore it.

    Concurrency here is not hypothetical: the singleton is shared by the memory
    inlet (event-loop thread) and background consolidation (a daemon thread),
    and by memory routes offloaded to the default multi-worker executor.
    """

    def _run_two_concurrent_embeds(self, monkeypatch):
        ctx = mp.get_context("spawn")
        req_q, resp_q = ctx.Queue(), ctx.Queue()
        runner = _CountingRunner()
        runner._req_q, runner._resp_q, runner._proc = req_q, resp_q, _AliveProc()

        stop = threading.Event()
        child = threading.Thread(
            target=_fifo_child, args=(req_q, resp_q, stop), daemon=True)
        child.start()

        # Install the runner without a real model load or a real child spawn.
        def fake_reload(self):
            self._runner = runner
            self.dim = 4

        monkeypatch.setattr(IsolatedEmbedder, "_reload", fake_reload)
        emb = IsolatedEmbedder("does-not-matter.gguf")

        results, errors = {}, {}

        def call(label, texts):
            try:
                results[label] = emb.embed(texts)
            except BaseException as e:      # noqa: BLE001 - reported below
                errors[label] = e

        # "A" -> 1 vector, "B" -> 3 vectors: distinguishable by length AND value.
        ta = threading.Thread(target=call, args=("A", ["A"]))
        tb = threading.Thread(target=call, args=("B", ["B", "B", "B"]))
        ta.start()
        tb.start()
        for t in (ta, tb):
            t.join(30)
        stop.set()
        child.join(5)
        for q in (req_q, resp_q):
            q.close()
        assert not errors, f"embed() raised: {errors}"
        return runner, results

    def test_concurrent_embed_callers_get_their_own_vectors(self, monkeypatch):
        runner, results = self._run_two_concurrent_embeds(monkeypatch)
        assert results["A"] == [_vec_for("A")], (
            "caller A received vectors belonging to another caller's text "
            f"(got {results['A']!r})")
        assert results["B"] == [_vec_for("B")] * 3, (
            "caller B received vectors belonging to another caller's text "
            f"(got {results['B']!r})")

    def test_concurrent_embed_never_overlaps_the_worker_rpc(self, monkeypatch):
        """The invariant behind the fix: because the protocol cannot correlate a
        response to its request, two RPCs must never be in flight at once."""
        runner, _ = self._run_two_concurrent_embeds(monkeypatch)
        assert runner.max_inflight == 1, (
            "two embed() RPCs overlapped on one correlation-free req/resp queue "
            f"pair (max in flight: {runner.max_inflight})")
