# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the embedding-model lifecycle (load, embed, unload).

``llama_load_model_from_file`` and ``llama_decode`` (called by every ``embed()``)
can hard-``abort()`` the whole process on a native CUDA/HIP driver failure, and
no Python ``try/except`` can catch that.

Lighter-weight than the chat runner (``backends/llamacpp/_runner.py``): the
embedder's usage pattern is load-once-serve-many, with no streaming and no
mid-call cancellation, so this needs only a plain request/response protocol,
closer in shape to ``localm/voice.py``'s worker. Still a long-lived child, so a
small embedding model is not reloaded on every ``embed()``.

Protocol (two ``multiprocessing.Queue``s, tagged tuples):

``req_q`` (parent -> child), one command processed at a time:
    ("load", {model_path, n_gpu_layers, n_ctx, pooling_type})
    ("embed", texts)
    ("shutdown", None)

``resp_q`` (child -> parent):
    ("ok", value)      - success (a {"dim": N} dict for load, a list of
                          vectors for embed)
    ("error", message) - a clean, expected failure (e.g. a bad model path)

A native abort, or any other uncaught fault in the child's dispatch loop,
produces NO envelope - the parent detects the dead child via
``proc.is_alive()``/``exitcode``, exactly like ``ModelRunner``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
from typing import List

# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set; never set in production. Values: "abort" (a genuine
# uncatchable native abort), "exit" (a hard process exit, no Python traceback),
# "hang" (a wedged native call). Mirrors llamacpp/_runner.py's _FAULT_ENV and
# voice.py's _FAULT_ENV.
_FAULT_ENV = "LOCALM_EMBEDDER_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    os.abort()                                    # genuine uncatchable native abort


# How long the "embed" dedup_native_stderr scope (see _runner_main below) may
# sit open with nothing pending before it is closed on idle. A bursty caller
# (one RAG index pass, many embed() calls back to back with no real gap) stays
# inside ONE scope, so genuinely repeated native lines get grouped; a quiet
# server between bursts flushes within this bound instead of holding output
# until the next embed() or process shutdown.
_EMBED_STDERR_IDLE_CLOSE_SECS = 5.0


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

_crash_trace_fh = None   # child-side: kept alive so faulthandler can write to it


def _arm_native_crash_trace(path) -> None:
    """Child side: point faulthandler at *path* so a death by native SIGNAL
    leaves a trace the parent can relay into the debug log.

    THIS IS THE ONLY THING THAT CAN CAPTURE THAT CLASS. A SIGILL/SIGSEGV/
    SIGABRT inside native code (llama.dll's own load, or the torch/ROCm conflict
    this worker's VRAM checks hit - see _sizing.py) never returns to Python at
    all, so no ``except`` clause anywhere in this child can run.

    Armed as early as possible - before the native library is anywhere near
    loaded - because a fault can only be captured by a handler that was already
    installed when it happened.

    Failures are logged, never raised: losing the trace must not stop the worker
    from doing its job. ``is_enabled()`` is checked rather than trusting that
    ``enable()`` did not raise."""
    global _crash_trace_fh
    if path is None:
        return
    import faulthandler
    from localm.debuglog import logger
    try:
        _crash_trace_fh = open(path, "w", encoding="utf-8")
        faulthandler.enable(file=_crash_trace_fh, all_threads=True)
        if not faulthandler.is_enabled():
            logger.warning(
                "embedding worker: faulthandler.enable() returned without "
                "raising but is_enabled() is False - a native fault in this "
                "worker will produce no stack trace")
    except Exception as e:   # noqa: BLE001 - a diagnostic must never break the worker
        logger.warning(
            "embedding worker: could not arm the native-fault trace (%s: %s) - "
            "a native fault in this worker will produce no stack trace",
            type(e).__name__, e)


def _runner_main(req_q, resp_q, crash_trace_path=None) -> None:
    """Long-lived child: owns one GGUFEmbedder (one loaded model) for its
    whole process lifetime, dispatching one request at a time.

    *crash_trace_path* defaults to None for an in-process caller, which has no
    child to trace and must not have faulthandler repointed underneath it. A
    real spawn always passes the parent-chosen path."""
    _arm_native_crash_trace(crash_trace_path)
    from localm.debuglog import attach_child_logging, dedup_native_stderr
    attach_child_logging()   # native load-failure diagnostics land in the
                              # shared debug log from this process too.

    from localm._mp_spawn import (ignore_interrupt_signals,
                                   install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the parent even on a hard kill
                                       # (End Task / force-close); daemon=True is
                                       # atexit-gated and does not cover that, so
                                       # else this worker outlives the server
                                       # holding its embedding model in VRAM.
    ignore_interrupt_signals()        # a console Ctrl+C reaches every process on
                                       # the console; the parent alone decides when
                                       # this worker stops.
    suppress_native_error_dialogs()   # a native DLL failure here (loading llama.dll
                                       # itself, or the torch/ROCm conflict this
                                       # worker's own VRAM checks are known to hit -
                                       # see _sizing.py) must degrade to a catchable
                                       # exception, never a blocking modal dialog.

    embedder = None

    # Lazily entered on the FIRST "embed" command and held open across every
    # "embed" that follows with no real gap between them - never re-entered per
    # call, and closed on idle (see _EMBED_STDERR_IDLE_CLOSE_SECS above). Every
    # llama_decode call (embedder.py's embed()) writes a native line like
    # "decode: cannot decode batches with this context (calling encode()
    # instead)" to raw stderr, and dedup_native_stderr's grouper only collapses
    # lines seen WITHIN one open scope, so a per-call scope collapses nothing.
    # "load" is excluded: GGUFEmbedder.__init__ already wraps the model load in
    # its own dedup_native_stderr scope, and nesting two scopes would dup2 fd 2
    # onto the OUTER scope's pipe instead of the real stderr.
    embed_stderr_ctx = None

    def _close_embed_stderr_ctx() -> None:
        nonlocal embed_stderr_ctx
        if embed_stderr_ctx is not None:
            embed_stderr_ctx.__exit__(None, None, None)
            embed_stderr_ctx = None

    while True:
        # Block indefinitely while no embed scope is open (nothing pending
        # to flush, so no reason to wake up on our own). Once a scope IS
        # open, poll with the idle bound instead, so a quiet stretch closes
        # it and flushes promptly rather than waiting on the next command.
        get_timeout = _EMBED_STDERR_IDLE_CLOSE_SECS if embed_stderr_ctx is not None else None
        try:
            cmd = req_q.get(timeout=get_timeout)
        except _queue.Empty:
            _close_embed_stderr_ctx()
            continue
        if cmd is None:
            _close_embed_stderr_ctx()
            return
        name = cmd[0]
        payload = cmd[1] if len(cmd) > 1 else None

        fault = os.environ.get(_FAULT_ENV)
        if fault:
            _simulate_fault(fault)   # test-only; never returns cleanly

        if name == "shutdown":
            _close_embed_stderr_ctx()
            if embedder is not None:
                embedder.close()
            return

        if name == "load":
            payload = dict(payload)
            # cpu_only: n_gpu_layers=0 alone only controls WEIGHT placement -
            # ggml's scheduler still considers a REGISTERED GPU backend for
            # individual ops, so a large enough matmul dispatches to vendor BLAS
            # regardless of how many layers were offloaded. Clearing the
            # vendor-visible-device env vars BEFORE importing anything native
            # makes the HIP/ROCm/CUDA runtime itself report ZERO devices, so
            # ggml's backend registration finds none and the scheduler has only
            # CPU to dispatch to, independent of model size. Set in THIS child
            # process only - never touches the parent's environment or a
            # concurrently-loading chat model.
            if payload.pop("cpu_only", False):
                # "-1", not "": Windows' CRT putenv("VAR=") with nothing after
                # '=' REMOVES the variable instead of setting it empty. "-1" is
                # the standard CUDA/HIP convention for "no valid device index",
                # which the runtime cannot treat as "unset".
                os.environ["HIP_VISIBLE_DEVICES"] = "-1"
                os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            from localm.inference.embedder import GGUFEmbedder
            try:
                embedder = GGUFEmbedder(**payload)
                # The pooling facts (and, when n_ctx was None/"auto", the
                # actual window size resolved from the model's own native
                # training context) travel back with the load so the PARENT
                # can warn about a mis-pooled model / report the real window:
                # only the child ever holds the model handle either is read
                # from.
                resp_q.put(("ok", {
                    "dim": embedder.dim,
                    "declared_pooling": embedder.declared_pooling,
                    "effective_pooling": embedder.pooling_type,
                    "n_ctx": embedder.n_ctx,
                }))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during GGUFEmbedder(...) is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "embed":
            if embed_stderr_ctx is None:
                embed_stderr_ctx = dedup_native_stderr()
                embed_stderr_ctx.__enter__()
            try:
                resp_q.put(("ok", embedder.embed(payload)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # Same as above: a native abort during llama_decode propagates
            # out of this whole function, uncaught. The open
            # dedup_native_stderr scope is then never closed; the process is
            # dying, so nothing pending needs flushing.
            continue

        # An unrecognized command is a bug in this module's own parent-side
        # caller, not untrusted input. It is reported, never dropped.
        resp_q.put(("error", f"unknown embedder-runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch.
# --------------------------------------------------------------------------- #

# How often spawn_and_load/embed re-check proc.is_alive() while polling for a
# response - short, since it is purely a local poll interval, not the
# command's own deadline (see the timeouts below).
_POLL_INTERVAL = 0.2

# Default model-load timeout. Embedding models are far smaller than chat models
# (24-90 MB for the built-in bge/nomic choices), but a user-configured
# `embedding_model` can be multi-GB. Generous but bounded: a stalled load must
# raise, never hang forever.
LOAD_TIMEOUT_DEFAULT = 300.0

# Bounded wait for one embed() RPC. A large RAG re-index batch can embed
# hundreds of chunks in one call, so this is generous rather than per-token
# tight like the chat runner's stream-chunk timeout; a genuinely wedged child
# is still detected via the is_alive() poll below, well inside this bound.
_EMBED_TIMEOUT_DEFAULT = 300.0


class EmbedderRunner:
    """Parent-side handle to one isolated embedder worker process."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        # Where THIS runner's child writes its native-fault trace. Chosen by the
        # parent (see debuglog.child_crash_trace_path for why it is not
        # recomputed child-side) and set in _spawn(); None before the first spawn.
        self._crash_trace_path = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _exit_reason(self) -> str:
        """The child's exit code DECODED - "-4 (killed by signal SIGILL)" rather
        than "-4".

        Every user-facing report of a dead worker goes through this, never
        interpolating the raw code, mirroring ``ModelRunner._exit_reason``. The
        decoder itself lives in ``_mp_spawn``."""
        from localm._mp_spawn import describe_exit_code
        proc = self._proc
        return describe_exit_code(None if proc is None else proc.exitcode)

    def _native_crash_trace(self) -> str:
        """This child's captured native-fault trace, consumed and removed, or ""
        when there is none.

        Consumed rather than merely read: the file is a one-shot record of one
        death, so leaving it in place would let a later reader (or the next spawn
        of a reused runner) attribute a stale trace to a fresh crash. Fully
        guarded - a diagnostic read must never replace the real crash error with
        an IO error."""
        path = self._crash_trace_path
        if path is None:
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        finally:
            self._discard_native_crash_trace()
        return text

    def _discard_native_crash_trace(self) -> None:
        """Remove this child's trace file. Best-effort: a leftover costs one
        small file in the logs dir, never correctness."""
        path = self._crash_trace_path
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _crash_detail(self) -> str:
        """A trailing detail for a crash message: the native trace when one was
        captured, else a plain statement that none was.

        Saying "no native stack trace was captured" OUT LOUD matters as much as
        relaying one: a message that points at the debug log when nothing was
        written sends the reader looking for something that is not there."""
        trace = self._native_crash_trace()
        if not trace:
            return " No native stack trace was captured for this fault."
        from localm.debuglog import logger, native_fault_hint
        # Logged as well as returned: the trace is multi-line and belongs in the
        # debug log the message points at, not inlined into an HTTP error body.
        logger.error("embedding worker native fault trace:\n%s", trace)
        first = trace.splitlines()[0].strip()
        return f" Native fault: {first} ({native_fault_hint()})."

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        # A previous child of this runner may have left one behind (a crash whose
        # trace nothing consumed); drop it before the new child claims the name,
        # so a stale trace can never be reported against the new process.
        self._discard_native_crash_trace()
        from localm.debuglog import child_crash_trace_path, logger
        try:
            self._crash_trace_path = child_crash_trace_path("embedder-worker")
        except OSError as e:
            # An unwritable logs dir costs the trace, not the worker.
            logger.warning("could not allocate a native-fault trace file (%s); "
                           "a native fault in this worker will not be traced", e)
            self._crash_trace_path = None
        self._proc = ctx.Process(
            target=_runner_main,
            args=(self._req_q, self._resp_q, self._crash_trace_path),
            name="localm-embedder-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, timeout: float = LOAD_TIMEOUT_DEFAULT) -> dict:
        """Spawn the child and load the model. Returns ``{"dim": N}`` on
        success. Raises RuntimeError on a genuine load failure (the worker is
        shut down first), a child crash (native abort - detected via
        is_alive(), never an exception this process had to catch), or a
        timeout (the child is killed)."""
        self._spawn()
        self._req_q.put(("load", params))
        return self._wait(timeout, "load", shutdown_on_error=True)

    def embed(self, texts: List[str], timeout: float = _EMBED_TIMEOUT_DEFAULT) -> List[List[float]]:
        """Embed *texts* via the isolated worker. Raises RuntimeError on a
        clean failure, a child crash, or a timeout - the caller (embedder.py's
        IsolatedEmbedder) decides whether/how to recover.

        NOT safe to call concurrently on one runner: the protocol above carries
        no request id, so two overlapping RPCs would be two threads blocked in
        the same resp_q.get(), each free to receive the OTHER's response. The
        sole caller, IsolatedEmbedder.embed(), serializes on its _rpc_lock."""
        self._req_q.put(("embed", texts))
        return self._wait(timeout, "embed")

    def _wait(self, timeout: float, label: str, *, shutdown_on_error: bool = False):
        """Block for the next response envelope for *label*.

        With shutdown_on_error=True, a clean ("error", ...) result also
        shuts the worker down before raising. spawn_and_load passes True;
        embed leaves the default False. See
        test_a_clean_load_error_shuts_the_worker_down and
        TestCleanEmbedErrorKeepsTheWorker."""
        deadline = time.monotonic() + timeout
        result = None
        while result is None:
            try:
                result = self._resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"The embedding worker process crashed (exit code "
                        f"{self._exit_reason()}) during '{label}'. The server "
                        "stayed up." + self._crash_detail())
                if time.monotonic() > deadline:
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"Embedding worker '{label}' timed out after "
                        f"{timeout:.0f}s - the worker process may be hung "
                        "(see the debug log). The server stayed up and the "
                        "worker was stopped; retry the request.")
        kind = result[0]
        if kind == "ok":
            return result[1]
        if kind == "error":
            if shutdown_on_error:
                self.shutdown(grace=0)
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the embedding worker: {result!r}")

    def shutdown(self, grace: float = 5.0) -> None:
        """Best-effort teardown: ask the worker to close cleanly, then kill it
        if it does not exit within *grace* seconds. Safe to call more than
        once, or when nothing is running."""
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            try:
                self._req_q.put(("shutdown", None))
            except Exception:
                pass
            if grace > 0:
                proc.join(timeout=grace)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=5)
        for q in (self._req_q, self._resp_q):
            if q is not None:
                try:
                    q.close()
                    q.cancel_join_thread()   # do not let a feeder thread block exit
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None
        # A worker torn down through shutdown() has had its exit accounted for
        # by whoever called it, so any trace it left is either already relayed or
        # describes a death nobody will report. It must not outlive the process
        # it describes.
        self._discard_native_crash_trace()
        self._crash_trace_path = None
