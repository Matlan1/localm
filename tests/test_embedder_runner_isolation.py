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
abort would take.
"""

import logging
import multiprocessing as mp
import os
import queue as _queue
import threading
import time

import pytest

from localm.inference import _embedder_runner as runner_mod
from localm.inference._embedder_runner import EmbedderRunner
from localm.inference.embedder import IsolatedEmbedder


_GPU_VISIBILITY_VARS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


@pytest.fixture(autouse=True)
def _clean_fault_env():
    # TestCpuOnlyHidesGpuDevices drives _runner_main IN-PROCESS, whose cpu_only
    # path sets the GPU-visibility vars directly on os.environ. Snapshot and
    # restore them around every test: monkeypatch.delenv(raising=False) does NOT
    # guard this, because on an ABSENT var it is a no-op that records nothing,
    # so the later direct os.environ[...]="-1" set is never undone and leaks to
    # every other test sharing the xdist worker.
    os.environ.pop(runner_mod._FAULT_ENV, None)
    saved = {k: os.environ.get(k) for k in _GPU_VISIBILITY_VARS}
    yield
    os.environ.pop(runner_mod._FAULT_ENV, None)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


_DUMMY_LOAD_PARAMS = dict(
    model_path="does-not-matter.gguf", n_gpu_layers=99, n_ctx=512, pooling_type=1,
)


# --------------------------------------------------------------------------- #
# A native fault is uncatchable in-process, so isolation is needed.
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # A child that wraps a genuine native abort in `except BaseException` still
    # dies, so an in-process llama_load_model_from_file / llama_decode crash
    # takes the whole process down.
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
        # A genuine uncatchable native abort (not a clean exit) during load is
        # still contained.
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
        detection (proc.is_alive()/exitcode) is a pure process-level check that
        does not care WHICH command the child happened to be running."""
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
        """Unlike the chat backend's chat_stream (which lets any OTHER fault
        propagate uncaught, since generation leaves the model in an unknown
        state - see llamacpp/_runner.py), the embedder's dispatch loop catches
        ordinary Python exceptions during 'embed' and reports them as a clean
        error WITHOUT killing the worker: embedding is stateless per call, so
        one bad request must not take down a worker that could otherwise keep
        serving. Reproduced without a real model by sending 'embed' before any
        'load' (embedder is None -> AttributeError, caught by the dispatch
        loop's own except Exception)."""
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
# Concurrent callers must not receive each other's vectors.
# --------------------------------------------------------------------------- #

def _vec_for(text: str):
    """A deterministic, input-derived vector, so a response delivered to the
    WRONG caller is detectable by both length and content."""
    return [float(len(text)), float(ord(text[0])), 0.0, 0.0]


class _AliveProc:
    """Stands in for the worker process's liveness check ONLY. _wait() polls
    proc.is_alive() whenever a resp_q poll times out (which the deliberate
    overlap window below guarantees). The correlation-free transport under
    test - the real mp.Queue pair and the real parent-side put/get - is NOT
    substituted."""

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
    and keeps serving, embedding being stateless per call. The parent sees the
    SAME RuntimeError for that as for a real crash, so dropping self._runner
    unconditionally leaves a LIVE child blocked on req_q.get() with the model
    still resident in VRAM: EmbedderRunner has no __del__, GC never terminates
    an mp.Process, and close()/reset_embedder()/release_for_exit() all only
    reach the CURRENT runner, so the orphan is unreachable and the next call
    spawns a second worker beside it.
    """

    def test_a_clean_error_keeps_the_live_worker_instead_of_orphaning_it(self, monkeypatch):
        # A REAL worker process, spawned with no model loaded: "embed" hits
        # embedder is None -> AttributeError -> the child's own except -> a clean
        # ("error", ...) envelope, with the child still alive.
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
        """The negative case: auto-reload after a genuine crash must still
        work."""
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
        # n_gpu_layers=0 keeps this CPU-configured, so the GPU-crash-fallback
        # branch never engages: this covers the generic dead-worker /
        # drop-and-reload contract.
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


class TestCleanLoadErrorReapsTheWorker:
    """spawn_and_load's clean ("error", msg) branch shuts the worker down
    before raising. Compare TestCleanEmbedErrorKeepsTheWorker above, which
    pins the opposite contract for embed()'s per-call error."""

    def test_a_clean_load_error_shuts_the_worker_down(self):
        # An unrecognized keyword argument makes GGUFEmbedder(**payload) raise
        # a TypeError before any native code runs; _runner_main's own except
        # Exception turns that into a clean ("error", msg) envelope.
        r = EmbedderRunner()
        params = dict(_DUMMY_LOAD_PARAMS, not_a_real_param="boom")
        try:
            with pytest.raises(RuntimeError) as ei:
                r.spawn_and_load(params, timeout=30.0)
            assert "not_a_real_param" in str(ei.value), (
                "expected the child's own exception message verbatim, got: "
                f"{ei.value!r}")
            assert not r.is_alive(), (
                "a clean load error left the worker process alive and "
                "unreachable")
        finally:
            r.shutdown(grace=0)


class TestCpuOnlyHidesGpuDevices:
    """cpu_only must hide GPU devices from the runtime BEFORE anything native
    loads - n_gpu_layers=0 alone only controls weight placement, and a large
    enough model's matmul still dispatches to a REGISTERED vendor backend
    regardless. Runs _runner_main's own dispatch loop directly (no real
    subprocess - the env-var mechanism needs no GPU hardware to verify, only
    that it engages before GGUFEmbedder is constructed and is popped before
    reaching it)."""

    def test_cpu_only_sets_env_before_construction_and_is_popped(self, monkeypatch):
        import queue as _q
        seen = {}

        class _StubGGUFEmbedder:
            def __init__(self, **kwargs):
                # Captured HERE: must reflect the env var the dispatch loop set
                # ahead of constructing this stub.
                seen["hip"] = os.environ.get("HIP_VISIBLE_DEVICES")
                seen["rocr"] = os.environ.get("ROCR_VISIBLE_DEVICES")
                seen["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                seen["kwargs"] = kwargs
                self.dim = 4
                self.declared_pooling = None
                self.pooling_type = 1
                self.n_ctx = 512

        monkeypatch.setattr("localm.inference.embedder.GGUFEmbedder", _StubGGUFEmbedder)
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising=False)

        req_q, resp_q = _q.Queue(), _q.Queue()
        req_q.put(("load", dict(model_path="x.gguf", n_gpu_layers=0, n_ctx=512,
                                pooling_type=1, cpu_only=True)))
        req_q.put(None)   # _runner_main returns after this

        runner_mod._runner_main(req_q, resp_q)

        assert resp_q.get_nowait()[0] == "ok"
        # "-1", not "": Windows' CRT putenv("VAR=") with nothing after '='
        # REMOVES the variable rather than setting it empty. "-1" is the
        # standard CUDA/HIP convention for "no valid device index".
        assert seen["hip"] == "-1"
        assert seen["rocr"] == "-1"
        assert seen["cuda"] == "-1"
        # cpu_only must never reach GGUFEmbedder's constructor (it has no such
        # parameter) - popped, not merely read.
        assert "cpu_only" not in seen["kwargs"]

    def test_cpu_only_false_leaves_env_untouched(self, monkeypatch):
        import queue as _q
        seen = {}

        class _StubGGUFEmbedder:
            def __init__(self, **kwargs):
                seen["hip"] = os.environ.get("HIP_VISIBLE_DEVICES")
                seen["kwargs"] = kwargs
                self.dim = 4
                self.declared_pooling = None
                self.pooling_type = 1
                self.n_ctx = 512

        monkeypatch.setattr("localm.inference.embedder.GGUFEmbedder", _StubGGUFEmbedder)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

        req_q, resp_q = _q.Queue(), _q.Queue()
        req_q.put(("load", dict(model_path="x.gguf", n_gpu_layers=99, n_ctx=512,
                                pooling_type=1, cpu_only=False)))
        req_q.put(None)

        runner_mod._runner_main(req_q, resp_q)

        assert resp_q.get_nowait()[0] == "ok"
        assert seen["hip"] is None            # untouched: the ordinary GPU load path
        assert "cpu_only" not in seen["kwargs"]


class TestEmbedStderrWrapping:
    """The isolated child's native EMBED-time llama_decode calls must run
    inside ONE dedup_native_stderr() scope spanning the child's whole run of
    "embed" commands, not one scope per call and not around "load" (which
    already has its own scope inside GGUFEmbedder.__init__).

    Wrapping each embed() call individually collapses nothing: a typical call
    (one RAG query, one memory fact) feeds dedup_native_stderr's grouper
    exactly one line, which flushes RAW the instant that call's own scope
    closes. The repetition worth collapsing is ACROSS separate embed() RPCs -
    many small calls in a row emitting the identical native line - so only a
    scope spanning MULTIPLE calls lets the grouper actually see the repeat.

    Drives _runner_main's own dispatch loop directly - no real subprocess
    needed, the scope-lifetime question is answered entirely by which commands
    were dispatched between enter and exit."""

    def _stub_and_spy(self, monkeypatch):
        import contextlib
        import queue as _q

        events = []

        @contextlib.contextmanager
        def spy_dedup():
            events.append("enter")
            yield
            events.append("exit")

        monkeypatch.setattr("localm.debuglog.dedup_native_stderr", spy_dedup)

        class _StubGGUFEmbedder:
            def __init__(self, **kwargs):
                self.dim = 4
                self.declared_pooling = None
                self.pooling_type = 1
                self.n_ctx = 512

            def embed(self, texts):
                events.append(f"embed:{len(texts)}")
                return [[0.0] * self.dim for _ in texts]

            def close(self):
                events.append("close")

        monkeypatch.setattr(
            "localm.inference.embedder.GGUFEmbedder", _StubGGUFEmbedder)
        req_q, resp_q = _q.Queue(), _q.Queue()
        return req_q, resp_q, events

    _LOAD_CMD = ("load", dict(
        model_path="x.gguf", n_gpu_layers=0, n_ctx=512, pooling_type=1))

    def test_one_scope_spans_every_embed_call_not_reentered(self, monkeypatch):
        req_q, resp_q, events = self._stub_and_spy(monkeypatch)
        req_q.put(self._LOAD_CMD)
        req_q.put(("embed", ["a"]))
        req_q.put(("embed", ["b", "c"]))
        req_q.put(("embed", ["d"]))
        req_q.put(("shutdown", None))

        runner_mod._runner_main(req_q, resp_q)

        # ONE enter bracketing all three embed calls (not three separate
        # enter/exit pairs), closed once at shutdown BEFORE the embedder
        # itself is closed (_close_embed_stderr_ctx runs first in the
        # "shutdown" branch) - "load" never touches this scope at all.
        assert events == [
            "enter", "embed:1", "embed:2", "embed:1", "exit", "close",
        ], events
        for _ in range(4):
            assert resp_q.get_nowait()[0] == "ok"

    def test_scope_never_entered_when_no_embed_command_arrives(self, monkeypatch):
        req_q, resp_q, events = self._stub_and_spy(monkeypatch)
        req_q.put(self._LOAD_CMD)
        req_q.put(("shutdown", None))

        runner_mod._runner_main(req_q, resp_q)

        assert events == ["close"], (
            "dedup_native_stderr was entered even though no embed() command "
            f"ever arrived: {events}")

    def test_scope_closes_on_the_none_sentinel_too(self, monkeypatch):
        """The parent-died sentinel (None on req_q) is a second, separate
        shutdown path from the explicit "shutdown" command - both must
        close an open scope, not just one of them."""
        req_q, resp_q, events = self._stub_and_spy(monkeypatch)
        req_q.put(self._LOAD_CMD)
        req_q.put(("embed", ["a"]))
        req_q.put(None)

        runner_mod._runner_main(req_q, resp_q)

        assert events == ["enter", "embed:1", "exit"], events

    def test_idle_gap_closes_the_scope_then_the_next_burst_reopens_it(self, monkeypatch):
        """Shrinks _EMBED_STDERR_IDLE_CLOSE_SECS so the test does not need a
        real 5-second sleep, then proves a genuine idle gap (a real time.sleep
        on a background feeder thread, not a pre-queued command) closes the
        scope on its own, before the next burst arrives and reopens a FRESH
        one."""
        import threading
        import time

        monkeypatch.setattr(runner_mod, "_EMBED_STDERR_IDLE_CLOSE_SECS", 0.1)
        req_q, resp_q, events = self._stub_and_spy(monkeypatch)
        req_q.put(self._LOAD_CMD)
        req_q.put(("embed", ["a"]))

        def _feed_second_burst():
            time.sleep(0.4)   # well past the 0.1s idle threshold
            req_q.put(("embed", ["b"]))
            req_q.put(("shutdown", None))

        threading.Thread(target=_feed_second_burst, daemon=True).start()
        runner_mod._runner_main(req_q, resp_q)

        assert events == [
            "enter", "embed:1", "exit",    # first burst, closed on idle
            "enter", "embed:1", "exit",    # second burst, its OWN fresh scope
            "close",
        ], events


class TestConcurrentEmbedSerialization:
    """The worker protocol has NO request-id correlation: one req_q/resp_q pair
    feeds one child, so two overlapping embed() calls are two threads blocked in
    the same resp_q.get() and the queue hands each whichever response arrives
    first - the caller can get vectors belonging to a DIFFERENT text (wrong
    length, wrong content), silently corrupting what lands in the semantic-memory
    and RAG vector stores. The in-process GGUFEmbedder.embed() holds an RLock
    that makes this impossible; IsolatedEmbedder.embed() must do the same.

    The singleton is shared by the memory inlet (event-loop thread), background
    consolidation (a daemon thread), and memory routes offloaded to the default
    multi-worker executor.
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


class TestNativeSignalCrashDiagnosticsReachDebugLog:
    """A message telling the user to see the debug log for the native stack
    trace is wrong for a death by native signal (SIGILL/SIGSEGV/SIGABRT inside
    llama.dll's own load, or the torch/ROCm conflict this worker's VRAM checks
    hit): that never returns to Python at all, so no ``except`` clause in this
    child could ever write one. The observable shape is ``worker exit -4``,
    which on Linux is SIGILL (``multiprocessing`` reports ``-N`` for death by
    signal N), with no trace in either field log.

    Every test here depends on faulthandler arming: with it armed, ``os.abort()``
    in a real spawned child writes a trace beginning "Fatal Python error:
    Aborted" plus the Python frame that entered native code; with it DISARMED
    the destination file is 0 bytes.
    """

    def _fault_during_embed(self, monkeypatch):
        """Drive a real worker to a real native abort while it dispatches an
        'embed'. Returns ``(message, trace_path)``.

        The fault env var is set BEFORE ``_spawn()``: the child reads it from
        its OWN ``os.environ``, a snapshot taken at spawn time, so setting it
        afterwards could never reach the running child."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = EmbedderRunner()
        r._spawn()   # embed() assumes a prior spawn_and_load(); spawn directly
                      # so "embed" is the first command the child dispatches.
        trace_path = r._crash_trace_path
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=60.0)
            assert not r.is_alive()
            return str(ei.value), trace_path
        finally:
            r.shutdown(grace=0)

    def test_native_abort_is_reported_with_its_captured_trace(
            self, monkeypatch, caplog):
        """After a native-signal death the caller must be told WHAT faulted,
        not merely that something did.

        Asserts on the TRACE CONTENT rather than on the exit code or the word
        "crashed", both of which are equally true of an uncharacterised
        fault."""
        with caplog.at_level(logging.ERROR, logger="localm"):
            message, _ = self._fault_during_embed(monkeypatch)

        assert "crashed" in message.lower()
        assert "Fatal Python error" in message, (
            "a real native-signal death produced no captured trace, so the "
            "caller still cannot tell WHICH native call faulted - the reported "
            f"issue 1222 / 1223 symptom\n--- message ---\n{message}")

        # The FULL multi-line trace, not just the summary line folded into the
        # message, reaches the debug log. caplog rather than a real log file:
        # attaching a handler to the shared "localm" logger would leak into
        # every later test.
        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "Fatal Python error" in logged and "_embedder_runner.py" in logged, (
            "the trace never reached the localm logger, or names no Python "
            f"frame, so it cannot say where the fault happened\n{logged}")

    def test_load_crash_also_reports_its_captured_trace(self, monkeypatch):
        """``_wait`` builds one message for both labels, but the load path is
        the one a user hits first and it reaches ``_wait`` by a different route
        (spawn_and_load), so it gets its own proof."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = EmbedderRunner()
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
        """When nothing was captured the message must SAY so.

        Arming-INDEPENDENT: it drops the path the parent would read, so it
        covers the branch for when arming did NOT work."""
        monkeypatch.setenv(runner_mod._FAULT_ENV, "abort")
        r = EmbedderRunner()
        r._spawn()
        # Simulate the capture having failed (an unwritable logs dir, a platform
        # where enable() no-ops) by dropping the path the parent would read.
        r._crash_trace_path = None
        try:
            with pytest.raises(RuntimeError) as ei:
                r.embed(["hello"], timeout=60.0)
            assert "no native stack trace was captured" in str(ei.value).lower()
        finally:
            r.shutdown(grace=0)

    def test_native_crash_trace_file_is_consumed_not_left_behind(
            self, monkeypatch):
        """The per-worker trace file must not survive the crash it describes: a
        stale file would be misread as a fresh crash by the next reader.

        Asserts the trace was CAPTURED as well as gone, so it cannot pass
        vacuously when nothing ever wrote the file."""
        message, trace_path = self._fault_during_embed(monkeypatch)
        assert trace_path is not None
        assert "Fatal Python error" in message, (
            "nothing was captured, so this test would be asserting cleanup of a "
            f"file that never existed\n--- message ---\n{message}")
        assert not trace_path.exists(), (
            f"the worker crash-trace file was left behind at {trace_path}")

    def test_healthy_worker_arms_a_trace_then_reaps_it(self):
        """The capture costs one empty file per model load, and a clean
        shutdown reaps it.

        Checks the file EXISTS while the worker is alive before checking it is
        gone afterwards: "absent at the end" is satisfied just as well by never
        having armed at all.

        The existence check is POLLED, not immediate. ``_spawn()`` returns as
        soon as ``Process.start()`` does, and a spawn-context child then has to
        boot a fresh interpreter and run its imports before it arms anything - so
        an immediate check races the child and fails on a perfectly healthy
        worker."""
        r = EmbedderRunner()
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
