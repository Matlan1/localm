# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the whole GGUF model lifecycle (load, generate,
tokenize, grammar-check, unload).

The model's whole lifecycle runs in a disposable child process; a native abort
there kills only that child, never the server. The runner is INSTANCE-scoped
(one ``ModelRunner`` per loaded ``GgufBackend``), not a global singleton, since
multiple GGUF models can be loaded simultaneously.

Cancellation needs no changes to llama.py:
- Load cancellation goes through ``LlamaCpp(cancel_event=...)``, polled by
  llama.cpp's own native progress callback - the child creates its OWN local
  ``threading.Event`` and a control thread ``.set()``s it on a ``cancel_load``
  signal relayed from the parent over ``ctrl_q``.
- Stream cancellation goes through plain Python generator ``.close()``
  (``GeneratorExit`` unwinds ``LlamaCpp._generate``'s lock cleanly), done
  locally in the child's own dispatch loop.

Protocol (three ``multiprocessing.Queue``s, tagged tuples):

``req_q`` (parent -> child), one command processed at a time:
    ("load", {model_path, mmproj_path, n_ctx, n_gpu_layers, n_ctx_max, n_ctx_grow,
              vram_overhead_bytes, gpu_split_ratios, n_cpu_moe})
    ("chat_stream", {messages, max_tokens, temperature, top_p, top_k,
                      repeat_penalty, grammar, grammar_lazy, grammar_triggers, seed})
    ("count_tokens", text)
    ("count_messages_tokens", messages)
    ("check_grammar", grammar)
    ("shutdown", None)

``resp_q`` (child -> parent):
    ("ok", value)                        - success (value shape depends on command)
    ("cancelled", message)                - load() aborted via cancel_load
    ("error", message[, kind])            - a clean, expected failure; kind is
                                             an optional typed-exception tag,
                                             re-raised as that type by the parent.
                                             Recognised: "InvalidGrammarError",
                                             "UnsupportedInputError",
                                             "GrammarUnsupportedError". An
                                             UNTAGGED error becomes a
                                             RuntimeError, which GgufBackend
                                             reads as "the isolated worker
                                             faulted" and answers by UNLOADING
                                             the model - so anything the CALLER
                                             can fix must carry a tag
    ("chunk", text)                       - one streamed token (chat_stream only)
    ("done", {finish_reason, grammar_unsupported, chatml_fallback_reason})
                                          - end of one chat_stream
    ("progress", payload)                 - NON-TERMINAL: the load is still
                                             running. See below.

TERMINAL vs NON-TERMINAL envelopes. Every kind above except ``progress`` ENDS
the wait that received it. ``chunk`` is non-terminal within a stream and
``done`` ends it.

``progress``'s payload is UNINTERPRETED here: this module only guarantees
delivery, and whatever decides what is worth reporting during a load owns the
payload's shape. Nothing emits it yet.

Two properties this must NOT weaken, both covered by tests:
- An UNKNOWN kind is a loud error, never ignored.
- ``progress`` does NOT extend the load deadline. The timeout bounds the WHOLE
  load.

A native abort, an unrecoverable fault deliberately left uncaught by
``GgufWorker.chat_stream``, or a genuine hang produces NO envelope - the parent
detects the dead or stuck child via ``proc.is_alive()``/``exitcode`` and a
bounded timeout.

``ctrl_q`` (parent -> child), drained by a dedicated control thread so a signal
takes effect even while the main thread is blocked in a native call:
    ("cancel_load",)
    ("cancel_stream",)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import threading
import time
from typing import Optional

from localm.inference.backends.base import (
    ContextCapacityExceededError,
    GrammarUnsupportedError, InvalidGrammarError, ModelLoadCancelled,
    UnsupportedInputError)


class RunnerBusy(Exception):
    """A best-effort, non-blocking command (a token count) declined to run
    because the runner's single response queue is already being driven by
    another command on this process (typically a live ``chat_stream``).

    This is NOT a failure: the caller has a documented fallback.
    ``count_tokens``/``count_messages_tokens`` fall back to a chars/4 heuristic;
    ``check_grammar`` defers the check to generation time, which rejects a
    malformed grammar with the same clean ``InvalidGrammarError``. ``load`` and
    ``chat_stream`` never raise this: they own their own queue drive."""


# Fault-injection hook for tests, honoured by the child only when this
# environment variable is set. Values: "abort" (an uncatchable native abort),
# "exit" (a hard process exit with no Python traceback), "hang" (a wedged
# native call). Checked at the top of every command dispatch.
_FAULT_ENV = "LOCALM_GGUF_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    os.abort()                                    # genuine uncatchable native abort


# Test-only: forces the "load" command to report a clean cancellation without
# touching the native runtime. Checked inside the "load" branch's own try.
_FORCE_LOAD_CANCEL_ENV = "LOCALM_GGUF_FORCE_LOAD_CANCEL_FOR_TEST"


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

_crash_trace_fh = None   # child-side: kept alive so faulthandler can write to it


def _arm_native_crash_trace(path) -> None:
    """Child side: point faulthandler at *path* so a death by native SIGNAL
    leaves a trace the parent can relay into the debug log.

    This is the only mechanism that captures a SIGILL/SIGSEGV/SIGABRT inside
    native code: such a fault never returns to Python, so no ``except`` clause
    can run. Armed before the native library is loaded.

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
                "gguf worker: faulthandler.enable() returned without raising "
                "but is_enabled() is False - a native fault in this worker will "
                "produce no stack trace")
    except Exception as e:   # noqa: BLE001 - a diagnostic must never break the worker
        logger.warning(
            "gguf worker: could not arm the native-fault trace (%s: %s) - a "
            "native fault in this worker will produce no stack trace",
            type(e).__name__, e)


def _runner_entry(req_q, resp_q, ctrl_q, crash_trace_path=None) -> None:
    """Process target. Wraps the whole worker body so ANY exception escaping it
    is logged via the ``logging`` module before the process dies.

    Logging here is fd-2-independent (a ``logging.FileHandler`` writes through
    its own Python-level stream, never through fd 2), so it captures the
    exception regardless of what fd 2 currently points to - the native stderr
    redirects in ``llama.py`` (``_quiet_stderr`` / ``_capture_stderr`` /
    ``dedup_native_stderr``) have already restored fd 2 by the time an escaping
    exception reaches ``multiprocessing``'s own traceback printer.

    RE-RAISES: this only ADDS a capture and never changes whether or how the
    process exits.

    Does NOT cover a native crash with no Python exception at all (SIGSEGV, a
    raw abort): Python never regains control, so no ``except`` clause can run.
    The parent's ``is_alive()``/``exitcode`` detection and the faulthandler
    trace :func:`_arm_native_crash_trace` leaves behind cover that case."""
    _arm_native_crash_trace(crash_trace_path)
    try:
        _runner_main(req_q, resp_q, ctrl_q)
    except BaseException:
        from localm.debuglog import attach_child_logging, logger
        attach_child_logging()   # idempotent - guarantees a handler exists
        logger.critical("gguf worker process crashed", exc_info=True)
        raise


def _runner_main(req_q, resp_q, ctrl_q) -> None:
    """Long-lived child: owns one GgufWorker (one loaded model) for its whole
    process lifetime, dispatching one request at a time."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # so native load-failure diagnostics captured via
                              # _quiet_stderr/_capture_stderr land in the shared
                              # debug log from this process too.

    from localm._mp_spawn import (install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the server even on a hard kill
                                       # (End Task / force-close), which daemon=True
                                       # does not cover
    suppress_native_error_dialogs()   # a native DLL failure here degrades to a
                                       # catchable exception

    from localm.inference.backends.base import InvalidGrammarError, ModelLoadCancelled
    from localm.inference.backends.llamacpp._worker import GgufWorker

    load_cancel_event = threading.Event()
    stream_cancel_event = threading.Event()

    def _control_loop() -> None:
        while True:
            msg = ctrl_q.get()
            if msg is None:
                return
            kind = msg[0] if isinstance(msg, tuple) else msg
            if kind == "cancel_load":
                load_cancel_event.set()
            elif kind == "cancel_stream":
                stream_cancel_event.set()

    threading.Thread(target=_control_loop, daemon=True, name="localm-gguf-ctrl").start()

    worker: Optional[GgufWorker] = None

    while True:
        cmd = req_q.get()
        if cmd is None:
            return
        name = cmd[0]
        payload = cmd[1] if len(cmd) > 1 else None

        fault = os.environ.get(_FAULT_ENV)
        if fault:
            _simulate_fault(fault)   # test-only; never returns cleanly

        if name == "shutdown":
            if worker is not None:
                worker.close()
            return

        if name == "load":
            # The constructor is INSIDE this try, not just worker.load(), so a
            # malformed payload produces an "error" envelope. spawn_and_load()
            # sends "load" straight after _spawn(), so `worker` is still None here.
            try:
                if os.environ.get(_FORCE_LOAD_CANCEL_ENV):
                    raise ModelLoadCancelled("forced cancellation (test-only)")
                worker = GgufWorker(cancel_event=load_cancel_event, **payload)
                meta = worker.load()
                resp_q.put(("ok", meta))
            except ModelLoadCancelled as e:
                resp_q.put(("cancelled", str(e)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during worker.load() is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "chat_stream":
            stream_cancel_event.clear()   # a stale cancel from a PRIOR stream
                                           # on this same model must not fire early
            try:
                gen = worker.chat_stream(**payload)
                for token in gen:
                    if stream_cancel_event.is_set():
                        gen.close()
                        break
                    resp_q.put(("chunk", token))
                resp_q.put(("done", {
                    "finish_reason": worker.last_finish_reason,
                    "grammar_unsupported": worker.grammar_unsupported_this_call,
                    "chatml_fallback_reason": worker.chatml_fallback_reason,
                }))
            except ContextCapacityExceededError as e:
                # An oversized prompt exceeding the configured context capacity or
                # leaving insufficient generation room. Raised in pure Python before
                # any native inference begins, so the worker keeps serving requests.
                resp_q.put(("error", str(e), "ContextCapacityExceededError"))
            except GrammarUnsupportedError as e:
                # _build_sampler REFUSES a lazy grammar it cannot apply. Raised
                # while building the sampler, before any native decode, so this
                # worker keeps serving.
                resp_q.put(("error", str(e), "GrammarUnsupportedError"))
            except InvalidGrammarError as e:
                # A malformed grammar the native parser safely rejected: an
                # ordinary Python exception, not a crash, so the worker keeps
                # serving.
                resp_q.put(("error", str(e), "InvalidGrammarError"))
            except UnsupportedInputError as e:
                # Input this model could not process, in practice a
                # VisionInputError from mtmd. A CHECKED status code from a native
                # call that RETURNED NORMALLY, so this worker keeps serving.
                # mtmd_tokenize touches no llama context; a failed
                # mtmd_helper_eval_chunks leaves the native KV populated with
                # _cached_tokens empty, which llama.py's prefill detects and wipes.
                resp_q.put(("error", str(e), "UnsupportedInputError"))
            # Any OTHER uncaught fault from the generator propagates OUT of this
            # whole function: the model is left in an unknown state, so this
            # process stops serving from it.
            continue

        if name == "count_tokens":
            try:
                resp_q.put(("ok", worker.count_tokens(payload)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            continue

        if name == "count_messages_tokens":
            try:
                resp_q.put(("ok", worker.count_messages_tokens(payload)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            continue

        if name == "check_grammar":
            try:
                worker.check_grammar(payload)
                resp_q.put(("ok", None))
            except InvalidGrammarError as e:
                resp_q.put(("error", str(e), "InvalidGrammarError"))
            except Exception as e:
                resp_q.put(("error", str(e)))
            continue

            # An unrecognized command is reported back as an error envelope.
        resp_q.put(("error", f"unknown runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch. One ModelRunner per loaded
# GgufBackend, never a module global.
# --------------------------------------------------------------------------- #

# How often spawn_and_load re-checks proc.is_alive()/cancel_event while polling
# for the load response. Not the load's own deadline (LOAD_TIMEOUT_DEFAULT).
_LOAD_POLL_INTERVAL = 0.2

# Envelope kinds the LOAD wait treats as NON-TERMINAL: they never end the wait.
# Every kind NOT in here ends it, so an unknown kind surfaces as a loud error.
_LOAD_NON_TERMINAL_KINDS = frozenset({"progress"})


def _emit_load_progress(sink, payload) -> None:
    """Hand one non-terminal load envelope to *sink*, never letting it break the
    load. A raising sink is swallowed and logged at debug with its traceback; it
    says nothing about whether the model loaded."""
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:
        from localm.debuglog import logger as _dbg
        _dbg.debug("load progress sink raised (ignored)", exc_info=True)

# Default model-load timeout. A stalled load raises rather than reporting
# "not loaded". Overridable per install via the ``gguf_load_timeout_s`` config key.
LOAD_TIMEOUT_DEFAULT = 900.0

# Per-token wait during generation. Applies once the first envelope has arrived;
# the wait for that first envelope uses FIRST_TOKEN_TIMEOUT_DEFAULT below.
_STREAM_CHUNK_TIMEOUT = 120.0

# Wait for the FIRST envelope of a stream, which covers the whole prompt
# PREFILL rather than one token's decode. Overridable per install via the
# ``gguf_first_token_timeout_s`` config key.
FIRST_TOKEN_TIMEOUT_DEFAULT = 900.0

# Bounded wait for a "done" envelope after requesting a mid-stream cancel.
_CANCEL_DRAIN_TIMEOUT = 5.0

# Bounded wait for a simple request/response command (count_tokens, etc.).
_SIMPLE_CMD_TIMEOUT = 30.0

# Coarse decode-progress heartbeat on the PARENT side. Counts "chunk" envelopes
# actually RECEIVED, not tokens the child claims to have generated.
#
# Logged at DEBUG, so it reaches the shared debug-log file only when --debug is
# on and never fills the always-on 400-record ring buffer.
_STREAM_PROGRESS_INTERVAL = 50


class _RunnerTornDown(Exception):
    """Internal: ``shutdown()`` released the child and its queues underneath an
    in-flight command on another thread.

    NOT a native fault and NOT a crash. ``shutdown()`` takes no lock (so teardown
    works while a command holds ``_q_lock``), and it CLOSES the three queues
    BEFORE it nulls them, so a command polling the response queue can land on
    either side of that window:

    * a closed queue - ``multiprocessing.Queue.get()`` after ``close()`` raises
      ``ValueError``, not ``Empty``, so it slips past the ``except _queue.Empty``
      handler;
    * ``_proc``/``_resp_q`` already None - an ``AttributeError`` on the next
      attribute access."""


class ModelRunner:
    """Parent-side handle to one isolated GGUF worker process. One instance
    per loaded ``GgufBackend`` - never a module-level singleton."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # Serialises PARENT-side use of the single response queue: every command
        # that drives the queue holds this for its whole request/response cycle,
        # so envelopes cannot interleave across threads. Acquired and released on
        # the SAME thread per command, so a plain non-reentrant Lock is correct.
        # shutdown() does NOT take it, so teardown works while a command holds it.
        self._q_lock = threading.Lock()
        # Where THIS runner's child writes its native-fault trace. Chosen by the
        # parent (debuglog.child_crash_trace_path) and set in _spawn(); None
        # before the first spawn.
        self._crash_trace_path = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _native_crash_trace(self) -> str:
        """This child's captured native-fault trace, consumed and removed, or ""
        when there is none.

        The file is a one-shot record of one death, so reading it removes it and
        a later reader cannot attribute a stale trace to a fresh crash. Fully
        guarded - a diagnostic read never replaces the real crash error with an
        IO error."""
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

    def _death_report(self):
        """``(native_evidenced, detail)`` for a dead worker.

        Reads the captured trace EXACTLY ONCE, because reading consumes it (see
        :meth:`_native_crash_trace`) and both halves need it: the trace decides
        whether this was a native fault at all, and it is also the detail worth
        relaying.

        Says "no native fault trace was captured" out loud when nothing was
        written, so a reader can tell an empty capture from their own failure to
        find one.

        The non-native branch gives an INSTRUCTION ("check the debug log") rather
        than promising a traceback: a hard ``os._exit`` produces no exception and
        therefore no traceback."""
        trace = self._native_crash_trace()
        native = self._exit_was_native_fault(trace_captured=bool(trace))
        if not trace:
            return native, " No native fault trace was captured for this exit."
        from localm.debuglog import logger
        # Logged as well as returned: the multi-line trace goes to the debug log
        # the message points at.
        logger.error("gguf worker native fault trace:\n%s", trace)
        first = trace.splitlines()[0].strip()
        return native, f" Native fault: {first} (full trace in the debug log)."

    def _crash_detail(self) -> str:
        """Just the detail half of :meth:`_death_report`, for the load and
        simple-request messages, whose opening words ("crashed") are already true
        of any worker death."""
        return self._death_report()[1]

    def _exitcode(self):
        """The child's exit code, or None once it has been released.

        Reads ``_proc`` ONCE into a local: ``is_alive()`` is False both when the
        child DIED and when ``shutdown()`` set ``_proc`` to None, so a direct
        ``self._proc.exitcode`` can AttributeError on the second case."""
        proc = self._proc
        return None if proc is None else proc.exitcode

    def _exit_reason(self) -> str:
        """The child's exit code DECODED - "-4 (killed by signal SIGILL)" rather
        than "-4".

        Every user-facing report of a dead worker goes through this rather than
        interpolating the raw code."""
        from localm._mp_spawn import describe_exit_code
        return describe_exit_code(self._exitcode())

    def _exit_was_native_fault(self, *, trace_captured: bool) -> bool:
        """Whether this worker's death is EVIDENCED as a native fault.

        The raw exit code has exactly two legitimate consumers - the decoder that
        renders it (:meth:`_exit_reason`) and this classifier - and everything
        else must go through one of those rather than touching the number."""
        from localm._mp_spawn import death_was_a_native_fault
        return death_was_a_native_fault(self._exitcode(),
                                        trace_captured=trace_captured)

    def _poll(self, timeout: float):
        """``resp_q.get()`` that turns a concurrent teardown into
        :class:`_RunnerTornDown`.

        ``_queue.Empty`` still propagates untouched - it is the normal
        keep-waiting signal every caller's loop is built around."""
        q = self._resp_q
        if q is None:
            raise _RunnerTornDown
        try:
            return q.get(timeout=timeout)
        except _queue.Empty:
            raise
        except (ValueError, OSError):
            # A closed multiprocessing.Queue raises ValueError from get(); a
            # closed underlying handle raises OSError. Neither is a native fault:
            # both mean shutdown() ran under us.
            raise _RunnerTornDown from None

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._ctrl_q = ctx.Queue()
        # A previous child of this runner may have left one behind; drop it before
        # the new child claims the name, so a stale trace is never reported
        # against the new process.
        self._discard_native_crash_trace()
        from localm.debuglog import child_crash_trace_path, logger
        try:
            self._crash_trace_path = child_crash_trace_path("gguf-worker")
        except OSError as e:
            # An unwritable logs dir costs the trace, not the worker.
            logger.warning("could not allocate a native-fault trace file (%s); "
                           "a native fault in this worker will not be traced", e)
            self._crash_trace_path = None
        self._proc = ctx.Process(
            target=_runner_entry,
            args=(self._req_q, self._resp_q, self._ctrl_q,
                  self._crash_trace_path),
            name="localm-gguf-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, cancel_event=None,
                        timeout: float = LOAD_TIMEOUT_DEFAULT,
                        on_progress=None) -> dict:
        """Spawn the child and load the model. Returns the metadata dict
        (``n_layers``/``kv_bytes_per_token``/``supports_images``) on success.

        Raises :class:`ModelLoadCancelled` if *cancel_event* fires during the
        load, or :class:`RuntimeError` on a genuine load failure, a child
        crash (native abort - detected via ``is_alive()``, never an
        exception this process had to catch), or a timeout (the child is
        killed).

        *on_progress*, if given, receives the payload of each NON-TERMINAL
        ``progress`` envelope (see this module's protocol notes). A raising sink
        is swallowed (``_emit_load_progress``) and never fails the load."""
        self._spawn()
        self._req_q.put(("load", params))
        deadline = time.monotonic() + timeout
        cancel_sent = False
        result = None
        while result is None:
            if cancel_event is not None and cancel_event.is_set() and not cancel_sent:
                self._ctrl_q.put(("cancel_load",))
                cancel_sent = True
            try:
                result = self._poll(_LOAD_POLL_INTERVAL)
            except _RunnerTornDown:
                # unload()/eviction released this runner mid-load. Reported as a
                # cancellation, the same way a superseded load is reported
                # (GgufBackend.load re-raises ModelLoadCancelled untouched).
                raise ModelLoadCancelled(
                    "the model was unloaded while it was still loading")
            except _queue.Empty:
                if self._proc is None:
                    raise ModelLoadCancelled(
                        "the model was unloaded while it was still loading")
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"The native model-loading process crashed (exit code "
                        f"{self._exit_reason()}) while loading. The server stayed up."
                        + self._crash_detail() +
                        " Retry the load, or repair the runtime with "
                        "'localm setup-llama'."
                    )
            else:
                # A NON-TERMINAL envelope reports that the load is still running,
                # so clear it and keep waiting. The isinstance guard sends a
                # NON-TUPLE down the TERMINAL path, where it fails loudly. A
                # payload-less ("progress",) counts as progress carrying None.
                if (isinstance(result, tuple) and result
                        and result[0] in _LOAD_NON_TERMINAL_KINDS):
                    _emit_load_progress(on_progress, result[1] if len(result) > 1
                                        else None)
                    result = None
            # OUTSIDE the `except _queue.Empty` branch: a progress envelope keeps
            # `get()` returning. Progress does NOT extend the deadline; it still
            # bounds the whole load.
            if result is None and time.monotonic() > deadline:
                self.shutdown(grace=0)
                raise RuntimeError(
                    f"Model load timed out after {timeout:.0f}s - the "
                    "worker process may be hung (see the debug log). The "
                    "server stayed up and the load was aborted; retry, or "
                    "raise gguf_load_timeout_s if this model genuinely "
                    "needs longer to load."
                )
        kind = result[0]
        if kind == "ok":
            return result[1]
        # No branch below produced a usable model, so reap this worker here, as
        # the LOAD-TIMEOUT branch above does.
        self.shutdown(grace=0)
        if kind == "cancelled":
            raise ModelLoadCancelled(result[1])
        if kind == "error":
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the model-loading process: {result!r}")

    def chat_stream(self, *, first_chunk_timeout: Optional[float] = None, **kwargs):
        """Yield text tokens. On the caller's ``GeneratorExit``, relays a
        ``cancel_stream`` signal to the child and drains for its confirming
        "done" before returning, so the worker is never left mid-generation when
        this backend serves its next request.

        Polls in short increments (not one big ``get(timeout=...)``) so a crashed
        child is detected within one poll interval, while still allowing up to
        ``_STREAM_CHUNK_TIMEOUT`` of genuine native decode time per token.

        The FIRST envelope gets its own, much larger budget
        (*first_chunk_timeout*, default ``FIRST_TOKEN_TIMEOUT_DEFAULT``): it
        waits for the whole prompt prefill, not for one token's decode. Keyword-
        only and popped here, so it is never forwarded to the child in *kwargs*.

        Holds ``_q_lock`` for the whole drive so no concurrent token-count RPC
        can consume this stream's envelopes off the shared response queue. The
        lock is acquired here and released when this generator is exhausted,
        errors, or is closed - all on the single producer thread that drives it,
        so the non-reentrant Lock is always released on the thread that took it.

        Emits INFO-level boundary log lines from THIS process, so they reach the
        always-on ring buffer; the one DEBUG-level line (decode progress, see
        ``_STREAM_PROGRESS_INTERVAL``) does not. The worker-died branch reports
        WHICH PHASE the child was in: no response ever received, or N chunks
        already streamed."""
        from localm.debuglog import logger
        first_budget = first_chunk_timeout or FIRST_TOKEN_TIMEOUT_DEFAULT
        awaiting_first = True
        chunks_received = 0
        _stream_t0 = time.monotonic()
        with self._q_lock:
            self._req_q.put(("chat_stream", kwargs))
            try:
                while True:
                    deadline = time.monotonic() + (
                        first_budget if awaiting_first else _STREAM_CHUNK_TIMEOUT)
                    result = None
                    while result is None:
                        try:
                            result = self._poll(_LOAD_POLL_INTERVAL)
                        except _RunnerTornDown:
                            raise RuntimeError(
                                "The model was unloaded while this reply was "
                                "being generated. It will reload on the next "
                                "request."
                            )
                        except _queue.Empty:
                            if not self.is_alive():
                                if self._proc is None:
                                    raise RuntimeError(
                                        "The model was unloaded while this reply "
                                        "was being generated. It will reload on "
                                        "the next request."
                                    )
                                # An uncaught Python exception in the worker exits
                                # 1, which is multiprocessing's signature for
                                # exactly that: no native fault, no native trace.
                                native, detail = self._death_report()
                                opening = (
                                    "Native inference fault"
                                    if native else
                                    "The model process exited unexpectedly")
                                # Records whether the worker was still prefilling
                                # or dispatching (no envelope ever arrived) or
                                # generating (N tokens already streamed back).
                                phase = (
                                    "prefill/dispatch (no response received yet)"
                                    if awaiting_first else
                                    f"decode ({chunks_received} token(s) "
                                    "already streamed)")
                                logger.error(
                                    "gguf worker: died mid-stream during %s", phase)
                                raise RuntimeError(
                                    f"{opening} (worker exit "
                                    f"{self._exit_reason()}). The model has been "
                                    "unloaded and will reload on the next "
                                    "request." + detail
                                )
                            if time.monotonic() > deadline:
                                self.shutdown(grace=0)
                                if awaiting_first:
                                    raise RuntimeError(
                                        f"Generation stalled: the model process "
                                        f"produced no output within "
                                        f"{first_budget:.0f}s of prompt processing. "
                                        "It has been unloaded and will reload on the "
                                        "next request. Raise gguf_first_token_timeout_s "
                                        "if this prompt genuinely needs longer on this "
                                        "hardware."
                                    )
                                raise RuntimeError(
                                    "Generation stalled: the model process stopped "
                                    "responding. It has been unloaded and will "
                                    "reload on the next request."
                                )
                    kind = result[0]
                    if awaiting_first:
                        logger.info(
                            "gguf worker: prefill complete, first response "
                            "after %.2fs (kind=%s)",
                            time.monotonic() - _stream_t0, kind)
                        awaiting_first = False
                    if kind == "chunk":
                        chunks_received += 1
                        # Coarse heartbeat - see _STREAM_PROGRESS_INTERVAL.
                        if chunks_received % _STREAM_PROGRESS_INTERVAL == 0:
                            logger.debug(
                                "gguf worker: decode progress, %d token(s) "
                                "received", chunks_received)
                        yield result[1]
                    elif kind == "done":
                        logger.info(
                            "gguf worker: generation complete, %d token(s), "
                            "finish_reason=%s",
                            chunks_received, result[1].get("finish_reason"))
                        self.last_done = result[1]
                        return
                    elif kind == "error":
                        # A clean, expected failure the worker did NOT let crash
                        # the process (e.g. a malformed grammar): the worker keeps
                        # running.
                        msg = result[1]
                        tag = result[2] if len(result) > 2 else ""
                        if tag == "InvalidGrammarError":
                            raise InvalidGrammarError(msg)
                        if tag == "GrammarUnsupportedError":
                            # Re-raise the TYPE, as for UnsupportedInputError
                            # below: the routes map this to a 400 naming the lazy
                            # grammar that could not be applied, while a
                            # RuntimeError out of this generator makes
                            # GgufBackend.chat_stream unload the model.
                            raise GrammarUnsupportedError(msg)
                        if tag == "UnsupportedInputError":
                            # NOT a RuntimeError: GgufBackend.chat_stream treats a
                            # RuntimeError from here as "the worker died" and
                            # unloads the model. UnsupportedInputError is a
                            # ValueError.
                            raise UnsupportedInputError(msg)
                        if tag == "ContextCapacityExceededError":
                            # An oversized prompt exceeding the configured context
                            # ceiling. NOT a RuntimeError, so GgufBackend does not
                            # unload the model. ContextCapacityExceededError is a
                            # ValueError.
                            raise ContextCapacityExceededError(msg)
                        raise RuntimeError(msg)
                    else:
                        raise RuntimeError(f"Unexpected response during generation: {result!r}")
            except GeneratorExit:
                logger.info(
                    "gguf worker: generation cancelled by caller after %d "
                    "token(s)", chunks_received)
                self._cancel_stream_and_drain()
                raise

    def _cancel_stream_and_drain(self) -> None:
        if not self.is_alive():
            return
        try:
            self._ctrl_q.put(("cancel_stream",))
        except Exception:
            self.shutdown(grace=0)
            return
        deadline = time.monotonic() + _CANCEL_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            try:
                result = self._poll(0.5)
            except _RunnerTornDown:
                return   # shutdown() got there first - nothing left to drain
            except _queue.Empty:
                if not self.is_alive():
                    return   # died on its own - nothing left to drain
                continue
            if result[0] == "done":
                return
            # A stray chunk racing the cancel is expected - keep draining.
        # Timed out waiting for "done": the child may be wedged inside a native
        # call the cancel flag cannot interrupt. Kill it, so the next request on
        # this backend spawns a fresh process.
        from localm.debuglog import logger as _dbg
        _dbg.warning("gguf runner: cancel_stream did not confirm within %.0fs; "
                     "killing the worker process", _CANCEL_DRAIN_TIMEOUT)
        self.shutdown(grace=0)

    def _simple_request(self, name: str, payload, timeout: float = _SIMPLE_CMD_TIMEOUT,
                        *, try_lock: bool = False):
        """Send one request/response command and return its value.

        Holds ``_q_lock`` for the whole exchange so its reply can never be
        stolen by (or steal from) a concurrent stream on the shared response
        queue. ``try_lock=True`` acquires the lock NON-blocking and raises
        :class:`RunnerBusy` immediately if it is held (a live stream, or another
        simple command) - used by the token counters, which have a documented
        heuristic fallback. The default blocking acquire is for commands that
        need the real answer (e.g. check_grammar)."""
        if try_lock:
            if not self._q_lock.acquire(blocking=False):
                raise RunnerBusy(name)
        else:
            self._q_lock.acquire()
        try:
            self._req_q.put((name, payload))
            deadline = time.monotonic() + timeout
            result = None
            while result is None:
                wait = max(0.01, min(0.5, deadline - time.monotonic()))
                try:
                    result = self._poll(wait)
                except _RunnerTornDown:
                    raise RuntimeError(
                        f"The model was unloaded while handling '{name}'.")
                except _queue.Empty:
                    if not self.is_alive():
                        if self._proc is None:
                            raise RuntimeError(
                                f"The model was unloaded while handling '{name}'.")
                        raise RuntimeError(
                            f"The model process crashed (exit code "
                            f"{self._exit_reason()}) while handling '{name}'."
                            + self._crash_detail())
                    if time.monotonic() > deadline:
                        self.shutdown(grace=0)
                        raise RuntimeError(f"'{name}' timed out waiting for the model process.")
            kind = result[0]
            if kind == "ok":
                return result[1]
            if kind == "error":
                msg = result[1]
                tag = result[2] if len(result) > 2 else ""
                if tag == "InvalidGrammarError":
                    raise InvalidGrammarError(msg)
                if tag == "GrammarUnsupportedError":
                    # Kept in step with the chat_stream decoder above: the two
                    # decoders read ONE protocol off ONE queue. The untagged
                    # fallback is RuntimeError, which unloads the model.
                    raise GrammarUnsupportedError(msg)
                if tag == "ContextCapacityExceededError":
                    raise ContextCapacityExceededError(msg)
                raise RuntimeError(msg)
            raise RuntimeError(f"Unexpected response for '{name}': {result!r}")
        finally:
            self._q_lock.release()

    def count_tokens(self, text: str) -> int:
        # try_lock: never block a token count behind a live generation; the
        # caller (GgufBackend) has a chars/4 fallback for RunnerBusy.
        return self._simple_request("count_tokens", text, try_lock=True)

    def count_messages_tokens(self, messages: list) -> int:
        return self._simple_request("count_messages_tokens", messages, try_lock=True)

    def check_grammar(self, grammar: str) -> None:
        # try_lock: validate_grammar (the only caller) runs SYNCHRONOUSLY on the
        # server's async event loop, so it must never block. On RunnerBusy,
        # GgufBackend.validate_grammar defers the check to generation time, which
        # still rejects a malformed grammar with the same InvalidGrammarError.
        self._simple_request("check_grammar", grammar, try_lock=True)

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
        for q in (self._req_q, self._resp_q, self._ctrl_q):
            if q is not None:
                try:
                    q.close()
                    q.cancel_join_thread()   # do not let a feeder thread block exit
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # A worker torn down through shutdown() has had its exit accounted for by
        # whoever called it, so its trace is dropped here.
        self._discard_native_crash_trace()
        self._crash_trace_path = None
