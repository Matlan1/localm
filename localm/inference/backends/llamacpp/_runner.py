# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the whole GGUF model lifecycle (load, generate,
tokenize, grammar-check, unload).

The whole lifecycle runs in a disposable child process, not just the load: a
``ctypes.c_void_p`` model/context handle is meaningless outside the process
that created it (no IPC-safe handle export exists for it, and the underlying
CUDA/HIP context is bound to its owning process), and
``LlamaCpp._prefill_fresh_context`` reaches the same abort-prone native call
class again on every context-window GROW. A native abort in the child kills
only that child, never the server.

The design mirrors ``localm/voice.py``: a long-lived
``multiprocessing.get_context("spawn")`` worker, ``Queue``s for
request/response, ``proc.is_alive()``/``exitcode`` for crash detection, and a
tagged error envelope instead of shipping native exception objects across the
boundary. The structural difference: this runner is INSTANCE-scoped (one
``ModelRunner`` per loaded ``GgufBackend``), not a global singleton, since
multiple GGUF models can be loaded simultaneously. It also streams and
supports mid-stream cancellation.

Cancellation:
- Load cancellation works via ``LlamaCpp(cancel_event=...)``, polled by
  llama.cpp's own native progress callback - the child creates its OWN local
  ``threading.Event`` for this and a control-thread ``.set()``s it on a
  ``cancel_load`` signal relayed from the parent over ``ctrl_q``.
- Stream cancellation works via plain Python generator ``.close()``
  (``GeneratorExit`` unwinds ``LlamaCpp._generate``'s lock cleanly), done
  locally by the child's own dispatch loop.

Protocol (three ``multiprocessing.Queue``s, tagged tuples, mirroring the
tagged-envelope style of ``voice.py`` rather than shipping exception objects):

``req_q`` (parent -> child), one command processed at a time:
    ("load", {model_path, mmproj_path, n_ctx, n_gpu_layers, n_ctx_max, n_ctx_grow,
              vram_overhead_bytes, gpu_split_ratios, n_cpu_moe, use_mmap,
              main_gpu?, adapters?, n_parallel?})
    ("chat_stream", {messages, max_tokens, temperature, top_p, top_k,
                      repeat_penalty, grammar, grammar_lazy, grammar_triggers, seed,
                      thinking?, min_p?, presence_penalty?, frequency_penalty?})
    ("chat_stream_mux", {sid, kwargs})   - multiplexed protocol only, see below
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
                                             "UnsupportedInputError" (and
                                             its subclasses named in
                                             _INPUT_ERROR_TYPES),
                                             "GrammarUnsupportedError", and on
                                             a load reply
                                             "PretokenizerUnusableModelError"
                                             and "AdapterLoadError". An
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
the wait that received it: ``chunk`` is non-terminal and ``done`` ends the
stream, and on the load path ``progress`` is non-terminal while everything
else ends the load.

``progress``'s payload is UNINTERPRETED here: this module only guarantees
delivery, so whatever decides what is worth reporting during a load owns the
payload's shape.

Two properties this must NOT weaken:
- An UNKNOWN kind is a loud error, never ignored. Tolerating unknown kinds
  would turn a protocol mismatch into a silent hang.
- ``progress`` does NOT extend the load deadline. The timeout bounds the WHOLE
  load, so a child emitting progress in a tight loop cannot keep a hung load
  alive.

A native abort, an unrecoverable fault left uncaught by
``GgufWorker.chat_stream``, or a genuine hang produces NO envelope - the
parent detects the dead/stuck child via ``proc.is_alive()``/``exitcode`` and a
bounded timeout.

``ctrl_q`` (parent -> child), drained by a dedicated control-thread so a
signal takes effect even while the main thread is blocked in a native call:
    ("cancel_load",)
    ("cancel_stream",)
    ("cancel_stream", sid)               - multiplexed protocol only

MULTIPLEXED PROTOCOL. A load whose reply reports ``parallel_slots`` > 1 switches
the runner to it for the rest of the child's life. Each stream is sent as
``("chat_stream_mux", {"sid": sid, "kwargs": kwargs})``; the child runs it on its
own thread, so streams and simple commands proceed side by side, and wraps every
envelope of that stream as ``("stream", sid, envelope)`` with the envelopes
above. ``("cancel_stream", sid)`` stops that stream only. On the parent a demux
thread owns ``resp_q`` and routes stream envelopes by sid and every other
envelope to the simple-command reply queue. A fault that would end the child on
the inline path still ends it: the stream thread exits the process with code 1.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue as _queue
import re
import threading
import time
from typing import Any, Callable, Optional

from localm.inference.backends.base import (
    AdapterLoadError, AudioDecodeUnavailable, AudioInputError, ContextCapacityExceededError,
    GrammarUnsupportedError, ImageDecodeUnavailable, InvalidGrammarError, ModelLoadCancelled,
    PerThread,
    PretokenizerUnsafeInputError, PretokenizerUnusableModelError,
    UnsupportedInputError, UnsupportedModelRoleError, VisionInputError,
    stream_stop_requested)

# UnsupportedInputError subclasses carried across the worker boundary by name,
# so the parent re-raises the same type. Any other subclass travels as
# "UnsupportedInputError".
_INPUT_ERROR_TYPES = {cls.__name__: cls for cls in (
    AudioDecodeUnavailable, AudioInputError, ImageDecodeUnavailable, VisionInputError)}


class RunnerBusy(Exception):
    """A best-effort, non-blocking command (a token count) declined to run
    because the runner's single response queue is already being driven by
    another command on this process (typically a live ``chat_stream``).

    This is NOT a failure: the caller has a documented fallback and uses it
    rather than blocking a request behind a whole generation, or queueing an RPC
    whose reply would race the live stream's envelopes on the shared queue.
    ``count_tokens``/``count_messages_tokens`` opt into this (they fall back to
    a chars/4 heuristic), and so does ``check_grammar``: ``validate_grammar`` is
    called SYNCHRONOUSLY on the server's async event loop, so a blocking wait
    there would freeze the whole loop for the length of a concurrent same-model
    stream - a busy check is instead DEFERRED to generation time, which rejects
    a malformed grammar with the same clean ``InvalidGrammarError``. ``load``
    and ``chat_stream`` never raise this: they own their own queue drive."""


# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set; never set in production. Values: "abort" (a genuine
# uncatchable native abort), "exit" (a hard process exit, no Python traceback),
# "hang" (a wedged native call), "abort-in-context" (an abort from a frame
# named like the context-creation entry point). Checked at the top of every
# command dispatch.
_FAULT_ENV = "LOCALM_GGUF_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    if mode == "abort-in-context":
        def llama_init_from_model() -> None:     # the frame name the trace shows
            os.abort()
        llama_init_from_model()
    os.abort()                                    # genuine uncatchable native abort


# Test-only: forces the "load" command to report a clean cancellation without
# ever touching the native runtime, so no GGUF file and no provisioned
# llama.cpp runtime are needed. Never set in production. NOT folded into
# _FAULT_ENV above, which is checked before the command name is known and
# always kills or hangs the process; this one is checked inside the "load"
# branch's own try, so a clean "cancelled" envelope still gets through.
_FORCE_LOAD_CANCEL_ENV = "LOCALM_GGUF_FORCE_LOAD_CANCEL_FOR_TEST"


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

_crash_trace_fh = None   # child-side: kept alive so faulthandler can write to it


def _arm_native_crash_trace(path) -> None:
    """Child side: point faulthandler at *path* so a death by native SIGNAL
    leaves a trace the parent can relay into the debug log.

    This is the only thing that can capture that class of death.
    ``_runner_entry``'s ``except BaseException`` below covers a crash that still
    has a Python exception; a SIGILL/SIGSEGV/SIGABRT inside native code never
    returns to Python at all, so no handler written in Python can run.
    ``worker exit -4`` is SIGILL (multiprocessing reports ``-N`` for signal N).

    Must be armed before the native library is loaded: a fault can only be
    captured by a handler that was already installed when it happened.

    Failures are logged, never raised, so losing the trace does not stop the
    worker from doing its job. ``is_enabled()`` is checked rather than trusting
    that ``enable()`` returned without raising, and a False result is warned
    about."""
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


def _runner_entry(req_q, resp_q, ctrl_q, crash_trace_path=None,
                  coder_privacy=None) -> None:
    """Process target. Wraps the whole worker body so ANY exception escaping it
    - a bug anywhere in ``_runner_main``'s own dispatch code, or the
    let-a-native-fault-kill-the-process design in the "chat_stream" branch (see
    ``GgufWorker.chat_stream``'s docstring) - is logged via the ``logging``
    module before the process dies, not left to
    ``multiprocessing.process.BaseProcess._bootstrap``'s own
    ``traceback.print_exc()`` alone.

    The native stderr redirects in ``llama.py`` (``_quiet_stderr`` /
    ``_capture_stderr`` / ``dedup_native_stderr``) do not cover this: they are
    ``@contextlib.contextmanager`` fd-2 redirects whose ``__exit__`` (restoring
    fd 2) runs as an escaping exception unwinds THROUGH them, before it ever
    reaches ``_bootstrap``. By the time multiprocessing prints its own
    traceback, fd 2 is back to whatever this process inherited from its parent
    (closed or NUL for a GUI-launched, console-less server).

    Logging here is fd-2-independent (a ``logging.FileHandler`` writes through
    its own Python-level stream, never through fd 2), so it captures the
    exception regardless of what fd 2 currently points to. RE-RAISES: this only
    ADDS a capture and must never change whether or how the process exits, since
    the parent's detection is ``is_alive()``/``exitcode``-based.

    Does NOT help a genuine native crash with no Python exception at all
    (SIGSEGV, a raw abort with nothing printed first) - Python never regains
    control there, so no ``except`` clause, including this one, can run. The
    parent's crash detection covers that, plus the faulthandler trace
    :func:`_arm_native_crash_trace` leaves behind."""
    _arm_native_crash_trace(crash_trace_path)
    if coder_privacy is not None:
        from localm.audit import adopt_shared_coder_privacy
        adopt_shared_coder_privacy(coder_privacy)
    try:
        _runner_main(req_q, resp_q, ctrl_q)
    except BaseException:
        from localm.debuglog import attach_child_logging, logger
        attach_child_logging()   # idempotent - guarantees a handler exists
                                  # even if _runner_main died before its own
                                  # (identical) call to this ever ran
        logger.critical("gguf worker process crashed", exc_info=True)
        raise


def _serve_stream(worker, payload: dict, emit: Callable, cancel_event) -> None:
    """Child side: run one ``chat_stream`` on *worker* with *payload*, sending
    its envelopes through *emit*: ``status`` and ``chunk`` while it runs, then
    ``done``, or a tagged ``error`` for a refusal that leaves the model
    unharmed. Stops early, still ending with ``done``, once *cancel_event* is
    set. Any other exception propagates: the caller must not keep serving
    from this model."""
    try:
        def _worker_status(s: str) -> None:
            emit(("status", s))
        gen = worker.chat_stream(on_status=_worker_status, **payload)
        for token in gen:
            if cancel_event.is_set():
                gen.close()
                break
            emit(("chunk", token))
        emit(("done", {
            "finish_reason": worker.last_finish_reason,
            "grammar_unsupported": worker.grammar_unsupported_this_call,
            "chatml_fallback_reason": worker.chatml_fallback_reason,
            "mtp_status": worker.mtp_status,
            "mtp_active": worker.mtp_active_this_call,
            "mtp_call_status": worker.mtp_call_status,
            "mtp_drafted": worker.mtp_drafted,
            "mtp_accepted": worker.mtp_accepted,
            "mtp_steps": worker.mtp_steps,
            "mtp_paused_steps": worker.mtp_paused_steps,
            "mtp_skipped": worker.mtp_skipped,
            "speculation": worker.spec_report,
        }))
    # Each arm below is a refusal raised before any native decode, or by a
    # native call that returned a status; the loaded model keeps serving.
    except ContextCapacityExceededError as e:
        emit(("error", str(e), "ContextCapacityExceededError"))
    except PretokenizerUnsafeInputError as e:
        emit(("error", str(e), "PretokenizerUnsafeInputError"))
    except GrammarUnsupportedError as e:
        emit(("error", str(e), "GrammarUnsupportedError"))
    except InvalidGrammarError as e:
        emit(("error", str(e), "InvalidGrammarError"))
    except UnsupportedInputError as e:
        tag = type(e).__name__
        emit(("error", str(e),
              tag if tag in _INPUT_ERROR_TYPES else "UnsupportedInputError"))


def _serve_mux_stream(worker, sid: int, payload: dict, resp_q, cancel_event,
                      cancels: dict, shutting_down=None) -> None:
    """Child side, one thread per multiplexed stream: :func:`_serve_stream`
    with every envelope wrapped as ``("stream", sid, envelope)`` and
    *cancel_event* published as the thread's stream stop check. An exception
    :func:`_serve_stream` lets through is logged and ends the whole process
    with exit code 1, as an uncaught fault on the inline path does; once
    *shutting_down* is set (the worker is closing the model) it is sent as an
    untagged ``error`` envelope instead."""
    from localm.inference.backends.base import stream_stop_check

    def emit(envelope) -> None:
        resp_q.put(("stream", sid, envelope))

    try:
        with stream_stop_check(cancel_event.is_set):
            _serve_stream(worker, payload, emit, cancel_event)
    except Exception as exc:
        if shutting_down is None or not shutting_down.is_set():
            _exit_after_stream_fault(sid)
        emit(("error", str(exc)))
    except BaseException:
        _exit_after_stream_fault(sid)
    finally:
        cancels.pop(sid, None)


def _exit_after_stream_fault(sid: int) -> None:
    """Log the exception being handled and end the process with exit code 1."""
    from localm.debuglog import attach_child_logging, logger
    attach_child_logging()
    logger.critical("gguf worker stream %d crashed", sid, exc_info=True)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    os._exit(1)


def _runner_main(req_q, resp_q, ctrl_q) -> None:
    """Long-lived child: owns one GgufWorker (one loaded model) for its whole
    process lifetime, dispatching one command at a time. A multiplexed stream
    is handed to its own thread, so the next command does not wait for it."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # so native load-failure diagnostics captured via
                              # _quiet_stderr/_capture_stderr land in the shared
                              # debug log from this process too, not just stdout.

    from localm._mp_spawn import (ignore_interrupt_signals,
                                   install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the server even on a hard kill
                                       # (End Task / force-close) - daemon=True does
                                       # not, it is atexit-gated; else this worker
                                       # outlives the server holding its model in VRAM.
    ignore_interrupt_signals()        # a console Ctrl+C reaches every process on
                                       # the console; the parent alone decides when
                                       # this worker stops.
    suppress_native_error_dialogs()   # a native DLL failure here (loading llama.dll
                                       # itself, or the torch/ROCm conflict this
                                       # worker's own VRAM checks are known to hit -
                                       # see _sizing.py) must degrade to a catchable
                                       # exception, never a blocking modal dialog.

    from localm.inference.backends.base import InvalidGrammarError, ModelLoadCancelled
    from localm.inference.backends.llamacpp._worker import GgufWorker

    load_cancel_event = threading.Event()
    stream_cancel_event = threading.Event()
    # Cancel events of the multiplexed streams in flight, by stream id, and the
    # ids cancelled before their stream was started; both under mux_lock.
    mux_cancels: dict = {}
    early_cancels: set = set()
    mux_lock = threading.Lock()
    mux_threads: list = []
    shutting_down = threading.Event()

    def _control_loop() -> None:
        while True:
            msg = ctrl_q.get()
            if msg is None:
                return
            kind = msg[0] if isinstance(msg, tuple) else msg
            if kind == "cancel_load":
                load_cancel_event.set()
            elif kind == "cancel_stream":
                if isinstance(msg, tuple) and len(msg) > 1:
                    with mux_lock:
                        event = mux_cancels.get(msg[1])
                        if event is None:
                            early_cancels.add(msg[1])
                    if event is not None:
                        event.set()
                else:
                    stream_cancel_event.set()

    threading.Thread(target=_control_loop, daemon=True, name="localm-gguf-ctrl").start()

    worker: Optional[GgufWorker] = None

    while True:
        cmd = req_q.get()
        if cmd is None:
            return
        name = cmd[0]
        payload: Any = cmd[1] if len(cmd) > 1 else None

        fault = os.environ.get(_FAULT_ENV)
        if fault:
            _simulate_fault(fault)   # test-only; never returns cleanly

        if name == "shutdown":
            shutting_down.set()
            if worker is not None:
                worker.close()
            for thread in mux_threads:
                thread.join(timeout=_SHUTDOWN_STREAM_JOIN_S)
            return

        if name == "load":
            # The constructor is INSIDE this try, not just worker.load(): a
            # malformed payload raising here is a clean, catchable Python
            # failure and gets the same "error" envelope a load() failure gets.
            # spawn_and_load() always calls self._spawn() immediately before
            # sending "load", so `worker` is still None here and there is no
            # in-place "reload" state to disturb.
            try:
                if os.environ.get(_FORCE_LOAD_CANCEL_ENV):
                    raise ModelLoadCancelled("forced cancellation (test-only)")
                worker = GgufWorker(cancel_event=load_cancel_event, **payload)
                worker.stream_cancel = stream_cancel_event
                meta = worker.load()
                resp_q.put(("ok", meta))
            except ModelLoadCancelled as e:
                resp_q.put(("cancelled", str(e)))
            except PretokenizerUnusableModelError as e:
                resp_q.put(("error", str(e), "PretokenizerUnusableModelError"))
            except AdapterLoadError as e:
                resp_q.put(("error", str(e), "AdapterLoadError"))
            except UnsupportedModelRoleError as e:
                resp_q.put(("error", str(e), "UnsupportedModelRoleError"))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during worker.load() is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "chat_stream":
            stream_cancel_event.clear()   # a stale cancel from a PRIOR stream
                                           # on this same model must not fire early
            _serve_stream(worker, payload, resp_q.put, stream_cancel_event)
            continue

        if name == "chat_stream_mux":
            sid = payload["sid"]
            event = threading.Event()
            with mux_lock:
                mux_cancels[sid] = event
                if sid in early_cancels:
                    early_cancels.discard(sid)
                    event.set()
            thread = threading.Thread(
                target=_serve_mux_stream,
                args=(worker, sid, payload["kwargs"], resp_q, event, mux_cancels,
                      shutting_down),
                name=f"localm-gguf-stream-{sid}", daemon=True)
            mux_threads[:] = [t for t in mux_threads if t.is_alive()] + [thread]
            thread.start()
            continue

        if name == "count_tokens":
            try:
                resp_q.put(("ok", worker.count_tokens(payload)))
            except PretokenizerUnsafeInputError as e:
                resp_q.put(("error", str(e), "PretokenizerUnsafeInputError"))
            except Exception as e:
                resp_q.put(("error", str(e)))
            continue

        if name == "count_messages_tokens":
            try:
                resp_q.put(("ok", worker.count_messages_tokens(payload)))
            except PretokenizerUnsafeInputError as e:
                resp_q.put(("error", str(e), "PretokenizerUnsafeInputError"))
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

        # An unrecognized command is a bug in this module's own parent-side
        # caller, not untrusted input. Reported rather than dropped.
        resp_q.put(("error", f"unknown runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch. Instance-scoped (NOT a module
# global like voice.py's singleton): multiple GGUF models can be loaded at
# once, each with its own ModelRunner held on its own GgufBackend.
# --------------------------------------------------------------------------- #

# How often spawn_and_load re-checks proc.is_alive()/cancel_event while
# polling for the load response - short, since it is purely a local poll
# interval, not the load's own deadline (see LOAD_TIMEOUT_DEFAULT below).
_LOAD_POLL_INTERVAL = 0.2

# Envelope kinds the LOAD wait treats as NON-TERMINAL: they say "still working"
# and must never end the wait. Everything NOT in here ends it, which keeps an
# unknown kind a loud error rather than a silent hang.
_LOAD_NON_TERMINAL_KINDS = frozenset({"progress"})


def _emit_load_progress(sink, payload) -> None:
    """Hand one non-terminal load envelope to *sink*, never letting it break the
    load. A raising progress callback is swallowed and logged at debug with its
    traceback; it says nothing about whether the model loaded."""
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:
        from localm.debuglog import logger as _dbg
        _dbg.debug("load progress sink raised (ignored)", exc_info=True)

# Default model-load timeout. A stalled load raises a clear error and never
# silently reports "not loaded". Generous, since a multi-GB model on a slow
# disk can legitimately take minutes; overridable per-install via the
# ``gguf_load_timeout_s`` config key (see gguf.py).
LOAD_TIMEOUT_DEFAULT = 900.0

# Per-token wait during generation. Generous (real per-token latency is
# sub-second even on CPU) but still bounded, so a genuinely wedged child is
# detected rather than blocking a request forever. Applies from the FIRST token
# onward - NOT to the first token itself, see below.
_STREAM_CHUNK_TIMEOUT = 120.0

# Wait for the FIRST envelope of a stream, a different quantity from the
# per-token ceiling above: nothing can be emitted until the whole prompt has
# been PREFILLED. On CPU (`-g 0`), under heavy partial offload, or with a
# multi-thousand-token prompt (RAG, a long document, a cold mmap cache), prefill
# can legitimately run far longer than any per-token latency, so this is sized
# like LOAD_TIMEOUT_DEFAULT rather than as a token budget: generous enough not
# to punish slow-but-working hardware, still bounded so a genuinely wedged child
# is caught. Overridable per install via the ``gguf_first_token_timeout_s``
# config key (see gguf.py).
FIRST_TOKEN_TIMEOUT_DEFAULT = 900.0

# Bounded wait for a "done" envelope after requesting a mid-stream cancel.
_CANCEL_DRAIN_TIMEOUT = 5.0

# Child side: how long a shutdown waits for each multiplexed stream thread to
# send its last envelope.
_SHUTDOWN_STREAM_JOIN_S = 2.0

# Parent side, multiplexed runner: how long a try_lock simple request waits for
# another simple request before it declines with RunnerBusy.
_MUX_SIMPLE_WAIT = 2.0

# Bounded wait for a simple request/response command (count_tokens, etc.).
# These never touch a slow native path, so it is short.
_SIMPLE_CMD_TIMEOUT = 30.0

# Coarse decode-progress heartbeat on the PARENT side, mirroring llama.py's
# _DECODE_PROGRESS_INTERVAL (an independent constant - different process, no
# shared import). Counts "chunk" envelopes actually RECEIVED rather than
# tokens the child claims to have generated, so it stays correct even if the
# two processes' native counters drift.
#
# Logged at DEBUG, not INFO: it recurs every N chunks for the life of a stream,
# and the always-on ring buffer is shared with everything else the server logs.
# At DEBUG it still reaches the shared debug-log file once --debug is on (this
# process's own FileHandler, attached whenever the server itself was launched
# with --debug).
_STREAM_PROGRESS_INTERVAL = 50


def _stream_error(result: tuple) -> Exception:
    """The exception a stream's ``error`` envelope stands for: its tagged type
    for a refusal that leaves the model loaded, else RuntimeError (the worker
    faulted)."""
    msg = result[1]
    tag = result[2] if len(result) > 2 else ""
    typed = {
        "InvalidGrammarError": InvalidGrammarError,
        "GrammarUnsupportedError": GrammarUnsupportedError,
        "UnsupportedInputError": UnsupportedInputError,
        "ContextCapacityExceededError": ContextCapacityExceededError,
        "PretokenizerUnsafeInputError": PretokenizerUnsafeInputError,
        **_INPUT_ERROR_TYPES,
    }.get(tag)
    return typed(msg) if typed is not None else RuntimeError(msg)


class _RunnerTornDown(Exception):
    """Internal: ``shutdown()`` released the child and its queues underneath an
    in-flight command on another thread.

    NOT a native fault and NOT a crash. ``shutdown()`` takes no lock (so teardown
    works while a command holds ``_q_lock``), and it CLOSES the three queues
    BEFORE it nulls them, so a command polling the response queue can land on
    either side of that window:

    * a closed queue - ``multiprocessing.Queue.get()`` after ``close()`` raises
      ``ValueError``, not ``Empty``, so it slips straight past the
      ``except _queue.Empty`` handler;
    * ``_proc``/``_resp_q`` already None - an ``AttributeError`` on the next
      attribute access.

    Both are translated into this exception so the caller reports "you unloaded
    the model while it was loading" rather than a broken native runtime."""


class NativeLoadCrashError(RuntimeError):
    """The worker process died during a load, with no envelope.

    ``phase`` names the native call that was running when it died, read from
    the captured fault trace: ``"context"`` (``llama_init_from_model``, after
    the weights loaded), ``"weights"`` (``llama_load_model_from_file``), or
    ``None`` when no trace was captured or the trace names neither."""

    def __init__(self, message: str, phase: Optional[str] = None) -> None:
        super().__init__(message)
        self.phase = phase


# The native entry points a crashed load can die inside, keyed by the function
# name faulthandler prints for its frame, mapped to the load phase they run.
_CRASH_PHASE_BY_FRAME = {
    "llama_init_from_model": "context",
    "llama_load_model_from_file": "weights",
}

_TRACE_CURRENT_THREAD_RE = re.compile(r"^Current thread\b")
_TRACE_FRAME_RE = re.compile(
    r'^\s+File "[^"\n]*", line \d+ in ([A-Za-z_][A-Za-z0-9_]*)\s*$')


def crash_phase_from_trace(trace: str) -> Optional[str]:
    """The load phase whose native call was running on the crashing thread of
    a faulthandler *trace*, or ``None`` when the trace has no "Current thread"
    block or none of its frames is a known entry point. Frames are read from
    the innermost outward and the first known entry point wins."""
    in_current = False
    for line in (trace or "").splitlines():
        if not in_current:
            in_current = bool(_TRACE_CURRENT_THREAD_RE.match(line))
            continue
        m = _TRACE_FRAME_RE.match(line)
        if m is None:
            if line.strip():
                break
            continue
        phase = _CRASH_PHASE_BY_FRAME.get(m.group(1))
        if phase is not None:
            return phase
    return None


def _load_crash_advice(phase: Optional[str]) -> str:
    """What to try after a worker died loading a model, for the phase it died in."""
    if phase == "context":
        return (" It crashed while creating the context, after the weights had "
                "loaded, so the runtime itself works. Creating the context "
                "allocates the KV cache and compute buffers on every GPU the "
                "model is split across: try a smaller context (n_ctx), fewer "
                "GPUs (gpu_split_indices) or fewer GPU layers (n_gpu_layers).")
    return " Retry the load, or repair the runtime with 'localm setup-llama'."


class ModelRunner:
    """Parent-side handle to one isolated GGUF worker process. One instance
    per loaded ``GgufBackend`` - never a module-level singleton."""

    # The "done" payload of the stream that last finished; on a multiplexed
    # runner, the one that last finished on the calling thread.
    last_done = PerThread(None, when=lambda r: getattr(r, "_mux", False))

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # Serialises PARENT-side use of the single response queue on the
        # inline protocol. The worker process is already serial (it reads req_q
        # one command at a time), but
        # two PARENT threads - a live chat_stream drive on the stream's producer
        # thread and a token-count RPC on an executor thread - would otherwise
        # both call self._resp_q.get() and STEAL each other's envelopes. Every
        # command that drives the queue holds this for its whole request/response
        # cycle, so envelopes can never interleave across threads. One lock per
        # runner, acquired and released on the SAME thread for each command (the
        # stream's whole drive + close runs on one producer thread), so a plain
        # non-reentrant Lock is correct. shutdown() does NOT take it, so teardown
        # still works while a command holds it. The multiplexed protocol does not
        # use it (see _start_mux).
        self._q_lock = threading.Lock()
        # Where THIS runner's child writes its native-fault trace. Chosen by the
        # parent via debuglog.child_crash_trace_path and set in _spawn(); None
        # before the first spawn.
        self._crash_trace_path = None
        # The load phase read from the last consumed fault trace (see
        # crash_phase_from_trace); None when none was captured.
        self._last_crash_phase = None
        # The death report of the child it was read for, as (proc, report),
        # read once under _death_lock however many streams see the death.
        self._death_cache = None
        self._death_lock = threading.Lock()
        # Multiplexed streams (a model loaded with parallel slots): a demux
        # thread owns the response queue and routes ("stream", sid, envelope)
        # to _streams[sid] and every other envelope to _simple_q.
        self._mux = False
        self._streams: dict = {}
        self._streams_lock = threading.Lock()
        self._simple_q: Optional[_queue.Queue] = None
        self._simple_lock = threading.Lock()
        self._sids = itertools.count(1)

    @property
    def multiplexed(self) -> bool:
        """True when streams run concurrently over the multiplexed protocol."""
        return self._mux

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _native_crash_trace(self) -> str:
        """This child's captured native-fault trace, consumed and removed, or ""
        when there is none.

        The file is a one-shot record of one death, so it is consumed rather
        than merely read: leaving it in place would let a later reader (or the
        next spawn of a reused runner) attribute a stale trace to a fresh crash.
        Fully guarded - a diagnostic read must never replace the real crash
        error with an IO error."""
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
        :meth:`_native_crash_trace`) and both halves need it: the trace is the
        strongest evidence of whether this was a native fault at all, and it is
        also the detail worth relaying.

        When nothing was captured, the detail says so out loud rather than
        pointing the user at a debug log that holds no trace.

        The non-native branch gives an INSTRUCTION rather than a promise ("check
        the debug log"): a Python exception escaping ``_runner_main`` is logged
        with its traceback by ``_runner_entry``, but a hard ``os._exit``
        produces no exception and therefore no traceback."""
        with self._death_lock:
            return self._death_report_locked()

    def _death_report_locked(self):
        proc = self._proc
        cached = self._death_cache
        if cached is not None and proc is not None and cached[0] is proc:
            return cached[1]
        trace = self._native_crash_trace()
        self._last_crash_phase = crash_phase_from_trace(trace)
        native = self._exit_was_native_fault(trace_captured=bool(trace))
        if not trace:
            report = (native, " No native fault trace was captured for this exit.")
        else:
            from localm.debuglog import logger, native_fault_hint
            # The full trace goes to the debug log; the report carries its first line.
            logger.error("gguf worker native fault trace:\n%s", trace)
            first = trace.splitlines()[0].strip()
            report = (native, f" Native fault: {first} ({native_fault_hint()}).")
        if proc is not None:
            self._death_cache = (proc, report)
        return report

    def _crash_detail(self) -> str:
        """Just the detail half of :meth:`_death_report`, for the load and
        simple-request messages, whose own opening words ("crashed") are already
        true of any worker death and need no native/ordinary distinction."""
        return self._death_report()[1]

    def _exitcode(self):
        """The child's exit code, or None once it has been released.

        Reads ``_proc`` ONCE into a local. ``is_alive()`` is False both when the
        child DIED and when ``shutdown()`` set ``_proc`` to None, so a caller
        reading ``self._proc.exitcode`` directly would AttributeError on the
        second case."""
        proc = self._proc
        return None if proc is None else proc.exitcode

    def _exit_reason(self) -> str:
        """The child's exit code DECODED - "-4 (killed by signal SIGILL)" rather
        than "-4".

        Every user-facing report of a dead worker goes through this rather than
        interpolating the raw code: decoding is what separates an illegal
        instruction from a segfault from an abort."""
        from localm._mp_spawn import describe_exit_code
        return describe_exit_code(self._exitcode())

    def _exit_was_native_fault(self, *, trace_captured: bool) -> bool:
        """Whether this worker's death is EVIDENCED as a native fault.

        The raw exit code has exactly two legitimate consumers: the decoder that
        renders it (:meth:`_exit_reason`) and this classifier that interprets
        it. Everything else must go through one of those rather than reading the
        number directly."""
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
            # closed underlying handle raises OSError. Neither is a native
            # fault - they mean shutdown() ran under us.
            raise _RunnerTornDown from None

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # avoid a renamed-launcher WinError 2
        self._mux = False
        self._simple_q = None
        with self._streams_lock:
            self._streams.clear()
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._ctrl_q = ctx.Queue()
        # A previous child of this runner may have left one behind (a crash whose
        # trace nothing consumed); drop it before the new child claims the name,
        # so a stale trace can never be reported against the new process.
        self._discard_native_crash_trace()
        from localm.audit import diagnostics_allowed, shared_coder_privacy_value
        coder_privacy = shared_coder_privacy_value()
        if diagnostics_allowed():
            from localm.debuglog import child_crash_trace_path, logger
            try:
                self._crash_trace_path = child_crash_trace_path("gguf-worker")
            except OSError as e:
                # An unwritable logs dir costs the trace, not the worker.
                logger.warning("could not allocate a native-fault trace file (%s); "
                               "a native fault in this worker will not be traced", e)
                self._crash_trace_path = None
        else:
            self._crash_trace_path = None
        self._proc = ctx.Process(
            target=_runner_entry,
            args=(self._req_q, self._resp_q, self._ctrl_q,
                  self._crash_trace_path, coder_privacy),
            name="localm-gguf-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, cancel_event=None,
                        timeout: float = LOAD_TIMEOUT_DEFAULT,
                        on_progress=None) -> dict:
        """Spawn the child and load the model. Returns the metadata dict
        (``n_layers``/``kv_bytes_per_token``/``supports_images``) on success.

        Raises :class:`ModelLoadCancelled` if *cancel_event* fires during the
        load, :class:`PretokenizerUnusableModelError` when the child refused
        the model's pre-tokenizer, or :class:`RuntimeError` on a genuine load failure, a child
        crash (native abort - detected via ``is_alive()``, never an
        exception this process had to catch), or a timeout (the child is
        killed; a load has no safe "unmeasurable" fallback, so this always
        raises rather than silently reporting not-loaded).

        *on_progress*, if given, receives the payload of each NON-TERMINAL
        ``progress`` envelope (see this module's protocol notes). A raising sink
        is swallowed (``_emit_load_progress``): a reporting callback must never
        fail a load that is otherwise fine."""
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
            except _RunnerTornDown as e:
                # unload()/eviction released this runner mid-load. Reported the
                # same way a superseded load is (GgufBackend.load re-raises
                # ModelLoadCancelled untouched), not as a runtime needing repair.
                raise ModelLoadCancelled(
                    "the model was unloaded while it was still loading") from e
            except _queue.Empty as e:
                if self._proc is None:
                    raise ModelLoadCancelled(
                        "the model was unloaded while it was still loading") from e
                if not self._proc.is_alive():
                    self._last_crash_phase = None
                    detail = self._crash_detail()
                    phase = self._last_crash_phase
                    raise NativeLoadCrashError(
                        f"The native model-loading process crashed (exit code "
                        f"{self._exit_reason()}) while loading. The server stayed up."
                        + detail + _load_crash_advice(phase),
                        phase=phase,
                    ) from e
            else:
                # A NON-TERMINAL envelope reports that the load is still running,
                # so clear it and keep waiting. The isinstance guard sends a
                # NON-TUPLE down the TERMINAL path instead, where it fails loudly
                # (the tail's unexpected-response error, or a TypeError on
                # something not even subscriptable); it must never be re-read as
                # progress, which would turn a broken protocol into an unbounded
                # wait. A payload-less ("progress",) counts as progress carrying
                # None, not as malformed: the payload is advisory. The deadline
                # below bounds the wait either way.
                if (isinstance(result, tuple) and result
                        and result[0] in _LOAD_NON_TERMINAL_KINDS):
                    _emit_load_progress(on_progress, result[1] if len(result) > 1
                                        else None)
                    result = None
            # The deadline check sits OUTSIDE the `except _queue.Empty` branch: a
            # progress envelope keeps `get()` returning, so an Empty-only check
            # would stop being reached once a child started emitting them, and a
            # wedged native call could run forever. Progress does NOT extend the
            # deadline; it still bounds the whole load.
            if result is None and time.monotonic() > deadline:
                self.shutdown(grace=0)
                from localm.debuglog import native_fault_hint
                raise RuntimeError(
                    f"Model load timed out after {timeout:.0f}s - the "
                    f"worker process may be hung ({native_fault_hint()}). The "
                    "server stayed up and the load was aborted; retry, or "
                    "raise gguf_load_timeout_s if this model genuinely "
                    "needs longer to load."
                )
        kind = result[0]
        if kind == "ok":
            meta = result[1]
            slots = meta.get("parallel_slots") if isinstance(meta, dict) else None
            if isinstance(slots, int) and slots > 1:
                self._start_mux()
            return meta
        # No branch below produced a usable model, so this worker holds nothing
        # worth keeping. Reaped here rather than left orphaned for the caller's
        # next load attempt to pile another one alongside it.
        self.shutdown(grace=0)
        if kind == "cancelled":
            raise ModelLoadCancelled(result[1])
        if kind == "error":
            if len(result) > 2 and result[2] == "PretokenizerUnusableModelError":
                raise PretokenizerUnusableModelError(result[1])
            if len(result) > 2 and result[2] == "AdapterLoadError":
                raise AdapterLoadError(result[1])
            if len(result) > 2 and result[2] == "UnsupportedModelRoleError":
                raise UnsupportedModelRoleError(result[1])
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the model-loading process: {result!r}")

    def chat_stream(self, *, first_chunk_timeout: Optional[float] = None,
                    on_status: Optional[Callable[[str], None]] = None,
                    stop_on_request: bool = False, **kwargs):
        """Yield text tokens. On the caller's ``GeneratorExit`` (a plain
        generator ``.close()``, which is how ``http_server.py`` cancels a
        stream), relays a ``cancel_stream`` signal to the child and drains for
        its confirming "done" before returning, so the worker is never left
        mid-generation when this backend serves its next request.

        Polls in short increments (not one big ``get(timeout=...)``) so a
        crashed child is detected promptly - within one poll interval, not
        after the full per-token ceiling - while still allowing up to
        ``_STREAM_CHUNK_TIMEOUT`` of genuine native decode time per token
        before treating it as stalled.

        With *stop_on_request* (a child that stops between the steps of a long
        call, a diffusion model), the stop check the caller published with
        :func:`~localm.inference.backends.base.stream_stop_check` is polled on
        every envelope and every poll interval; once it asks to stop, the
        generation in the child is cancelled, its confirmation awaited for up to
        ``_STREAM_CHUNK_TIMEOUT``, and this stream ends without yielding more.
        A ``KeyboardInterrupt`` raised while this waits cancels and drains the
        child the same way (``_STREAM_CHUNK_TIMEOUT`` with *stop_on_request*,
        else ``_CANCEL_DRAIN_TIMEOUT``) before propagating, so the next request
        never reads this one's envelopes.

        The FIRST envelope gets its own, much larger budget
        (*first_chunk_timeout*, default ``FIRST_TOKEN_TIMEOUT_DEFAULT``): it
        waits for the whole prompt prefill, not for one token's decode. Keyword-
        only and popped here, so it is never mistaken for a generation
        parameter forwarded to the child in *kwargs*.

        Holds ``_q_lock`` for the whole drive so no concurrent token-count RPC
        can consume this stream's envelopes off the shared response queue. The
        lock is acquired here and released when this generator is exhausted,
        errors, or is closed - all of which happen on the single producer thread
        that drives it, so the non-reentrant Lock is always released on the
        thread that took it.

        BOUNDARY LOGGING: the INFO-level lines here run in THIS process, where
        ``install_ring_buffer()`` already ran at CLI startup, so they reach the
        always-on ring buffer unconditionally. The one DEBUG-level line, decode
        progress, does not - see ``_STREAM_PROGRESS_INTERVAL``. They are built
        entirely from envelopes this method already receives, with no extra IPC.
        The worker-died branch below reports WHICH PHASE the child was in (no
        response ever received vs N chunks already streamed).

        On a multiplexed runner (a model loaded with parallel slots) the stream
        runs through :meth:`_chat_stream_mux` instead, beside other streams."""
        if self._mux:
            yield from self._chat_stream_mux(
                first_chunk_timeout=first_chunk_timeout, on_status=on_status,
                stop_on_request=stop_on_request, **kwargs)
            return
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
                        if stop_on_request and stream_stop_requested():
                            logger.info("gguf worker: generation stopped by caller "
                                        "after %d token(s)", chunks_received)
                            self._cancel_stream_and_drain(_STREAM_CHUNK_TIMEOUT)
                            return
                        try:
                            result = self._poll(_LOAD_POLL_INTERVAL)
                        except _RunnerTornDown as e:
                            raise RuntimeError(
                                "The model was unloaded while this reply was "
                                "being generated. It will reload on the next "
                                "request."
                            ) from e
                        except _queue.Empty as e:
                            if not self.is_alive():
                                if self._proc is None:
                                    raise RuntimeError(
                                        "The model was unloaded while this reply "
                                        "was being generated. It will reload on "
                                        "the next request."
                                    ) from e
                                # Do NOT call this a native fault unless the
                                # evidence says so. An uncaught Python exception
                                # in the worker exits 1, which is
                                # multiprocessing's signature for exactly that,
                                # and reporting it as a native fault would be
                                # false in every clause (no native fault, no
                                # native trace, model unharmed).
                                native, detail = self._death_report()
                                opening = (
                                    "Native inference fault"
                                    if native else
                                    "The model process exited unexpectedly")
                                # Which phase the worker died in: still
                                # prefilling/dispatching (no envelope ever
                                # arrived) or generating (N tokens already
                                # streamed back).
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
                                ) from e
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
                                    ) from e
                                raise RuntimeError(
                                    "Generation stalled: the model process stopped "
                                    "responding. It has been unloaded and will "
                                    "reload on the next request."
                                ) from e
                    kind = result[0]
                    if kind == "status":
                        status_text = result[1]
                        logger.info("gguf worker: status: %s", status_text)
                        if on_status:
                            try:
                                on_status(status_text)
                            except Exception:
                                logger.debug("chat_stream on_status callback raised (ignored)", exc_info=True)
                        continue
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
                        # the process (e.g. a malformed grammar) - the model is
                        # unharmed and the worker keeps running.
                        msg = result[1]
                        tag = result[2] if len(result) > 2 else ""
                        if tag == "InvalidGrammarError":
                            raise InvalidGrammarError(msg)
                        if tag == "GrammarUnsupportedError":
                            # Re-raise the TYPE: the routes map it to a 400
                            # naming the real problem (the lazy grammar could
                            # not be applied), while a RuntimeError out of this
                            # generator means "the worker died" and makes
                            # GgufBackend.chat_stream unload the model.
                            raise GrammarUnsupportedError(msg)
                        if tag == "UnsupportedInputError":
                            # NOT a RuntimeError: GgufBackend.chat_stream treats
                            # RuntimeError from here as "the worker died" and
                            # unloads the model. This is a per-request refusal by
                            # a healthy worker, so it must not evict a loaded
                            # model. UnsupportedInputError is a ValueError.
                            raise UnsupportedInputError(msg)
                        if tag in _INPUT_ERROR_TYPES:
                            # A typed per-request refusal: same handling as above,
                            # keeping the exact subclass.
                            raise _INPUT_ERROR_TYPES[tag](msg)
                        if tag == "ContextCapacityExceededError":
                            # An oversized prompt exceeding the configured context ceiling.
                            # NOT a RuntimeError, so GgufBackend does not unload
                            # the model. ContextCapacityExceededError is a ValueError.
                            raise ContextCapacityExceededError(msg)
                        if tag == "PretokenizerUnsafeInputError":
                            # Refused before any native call. NOT a RuntimeError,
                            # so GgufBackend does not unload the model.
                            # PretokenizerUnsafeInputError is a ValueError.
                            raise PretokenizerUnsafeInputError(msg)
                        raise RuntimeError(msg)
                    else:
                        raise RuntimeError(f"Unexpected response during generation: {result!r}")
            except GeneratorExit:
                logger.info(
                    "gguf worker: generation cancelled by caller after %d "
                    "token(s)", chunks_received)
                self._cancel_stream_and_drain()
                raise
            except KeyboardInterrupt:
                logger.info(
                    "gguf worker: generation interrupted after %d token(s)",
                    chunks_received)
                self._cancel_stream_and_drain(
                    _STREAM_CHUNK_TIMEOUT if stop_on_request else _CANCEL_DRAIN_TIMEOUT)
                raise

    def _cancel_stream_and_drain(self, timeout: float = _CANCEL_DRAIN_TIMEOUT) -> None:
        """Ask the child to stop the running stream and consume its envelopes
        until its "done" (stored as ``last_done``), for at most *timeout*
        seconds; past that the child is killed."""
        if not self.is_alive():
            return
        try:
            self._ctrl_q.put(("cancel_stream",))
        except Exception:
            self.shutdown(grace=0)
            return
        deadline = time.monotonic() + timeout
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
                if len(result) > 1 and isinstance(result[1], dict):
                    self.last_done = result[1]
                return
            # A stray chunk racing the cancel is expected - keep draining.
        # Timed out waiting for "done": the child may be wedged inside a native
        # call the cancel flag cannot interrupt. Warn and kill it, so the next
        # request on this backend spawns a known-good process instead of reusing
        # one that never confirmed it stopped.
        from localm.debuglog import logger as _dbg
        _dbg.warning("gguf runner: cancel_stream did not confirm within %.0fs; "
                     "killing the worker process", timeout)
        self.shutdown(grace=0)

    # ------------------------------------------------------------------ #
    #  Multiplexed streams (a model loaded with parallel slots)            #
    # ------------------------------------------------------------------ #

    def _start_mux(self) -> None:
        """Switch to the multiplexed protocol: start the demux thread that owns
        the response queue from now on."""
        self._simple_q = _queue.Queue()
        self._mux = True
        threading.Thread(target=self._demux, args=(self._resp_q, self._simple_q),
                         name="localm-gguf-demux", daemon=True).start()

    def _demux(self, resp_q, simple_q) -> None:
        """Route the child's envelopes: ``("stream", sid, envelope)`` to that
        stream's queue (dropped when the stream already ended), anything else to
        *simple_q*. Returns once the queue is torn down or the child is gone."""
        from localm.debuglog import logger
        while True:
            try:
                item = resp_q.get(timeout=_LOAD_POLL_INTERVAL)
            except _queue.Empty:
                if self._resp_q is not resp_q or not self.is_alive():
                    return
                continue
            except (ValueError, OSError, EOFError):
                return
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "stream":
                with self._streams_lock:
                    target = self._streams.get(item[1])
                if target is not None:
                    target.put(item[2])
                else:
                    logger.debug("gguf runner: envelope for ended stream %r dropped",
                                 item[1])
                continue
            simple_q.put(item)

    def _mux_unloaded_error(self) -> RuntimeError:
        return RuntimeError(
            "The model was unloaded while this reply was being generated. It "
            "will reload on the next request.")

    def _chat_stream_mux(self, *, first_chunk_timeout: Optional[float] = None,
                         on_status: Optional[Callable[[str], None]] = None,
                         stop_on_request: bool = False, **kwargs):
        """:meth:`chat_stream` on a multiplexed runner: same envelopes, errors,
        timeouts and cancel contract, for one stream among several.

        A status equal to the previous one is not passed to *on_status* again;
        every status still restarts the wait for the next envelope. A stop
        published with ``stream_stop_check`` on this thread cancels the stream
        in the child and ends this generator once the child confirms."""
        from localm.debuglog import logger
        first_budget = first_chunk_timeout or FIRST_TOKEN_TIMEOUT_DEFAULT
        awaiting_first = True
        chunks_received = 0
        last_status = None
        _stream_t0 = time.monotonic()
        self.last_done = None
        sid = next(self._sids)
        stream_q: _queue.Queue = _queue.Queue()
        with self._streams_lock:
            self._streams[sid] = stream_q
        try:
            req_q = self._req_q
            if req_q is None:
                raise self._mux_unloaded_error()
            req_q.put(("chat_stream_mux", {"sid": sid, "kwargs": kwargs}))
            while True:
                deadline = time.monotonic() + (
                    first_budget if awaiting_first else _STREAM_CHUNK_TIMEOUT)
                result = None
                while result is None:
                    if stream_stop_requested():
                        logger.info("gguf worker: stream %d stopped by caller after "
                                    "%d token(s)", sid, chunks_received)
                        self._cancel_mux_stream_and_drain(
                            sid, stream_q,
                            _STREAM_CHUNK_TIMEOUT if stop_on_request else _CANCEL_DRAIN_TIMEOUT)
                        return
                    try:
                        result = stream_q.get(timeout=_LOAD_POLL_INTERVAL)
                    except _queue.Empty as e:
                        if self._proc is None:
                            raise self._mux_unloaded_error() from e
                        if not self.is_alive():
                            native, detail = self._death_report()
                            opening = ("Native inference fault" if native
                                       else "The model process exited unexpectedly")
                            phase = ("prefill/dispatch (no response received yet)"
                                     if awaiting_first else
                                     f"decode ({chunks_received} token(s) already streamed)")
                            logger.error("gguf worker: died mid-stream during %s", phase)
                            raise RuntimeError(
                                f"{opening} (worker exit {self._exit_reason()}). The "
                                "model has been unloaded and will reload on the next "
                                "request." + detail) from e
                        if time.monotonic() > deadline:
                            self.shutdown(grace=0)
                            if awaiting_first:
                                raise RuntimeError(
                                    f"Generation stalled: the model process produced "
                                    f"no output within {first_budget:.0f}s of prompt "
                                    "processing. It has been unloaded and will reload "
                                    "on the next request. Raise "
                                    "gguf_first_token_timeout_s if this prompt "
                                    "genuinely needs longer on this hardware.") from e
                            raise RuntimeError(
                                "Generation stalled: the model process stopped "
                                "responding. It has been unloaded and will reload on "
                                "the next request.") from e
                kind = result[0]
                if kind == "status":
                    status_text = result[1]
                    if status_text != last_status:
                        last_status = status_text
                        logger.info("gguf worker: stream %d status: %s", sid, status_text)
                        if on_status:
                            try:
                                on_status(status_text)
                            except Exception:
                                logger.debug("chat_stream on_status callback raised "
                                             "(ignored)", exc_info=True)
                    continue
                if awaiting_first:
                    logger.info("gguf worker: stream %d prefill complete, first "
                                "response after %.2fs (kind=%s)", sid,
                                time.monotonic() - _stream_t0, kind)
                    awaiting_first = False
                if kind == "chunk":
                    chunks_received += 1
                    if chunks_received % _STREAM_PROGRESS_INTERVAL == 0:
                        logger.debug("gguf worker: stream %d decode progress, %d "
                                     "token(s) received", sid, chunks_received)
                    yield result[1]
                elif kind == "done":
                    logger.info("gguf worker: stream %d complete, %d token(s), "
                                "finish_reason=%s", sid, chunks_received,
                                result[1].get("finish_reason"))
                    self.last_done = result[1]
                    return
                elif kind == "error":
                    raise _stream_error(result)
                else:
                    raise RuntimeError(f"Unexpected response during generation: {result!r}")
        except GeneratorExit:
            logger.info("gguf worker: stream %d cancelled by caller after %d token(s)",
                        sid, chunks_received)
            self._cancel_mux_stream_and_drain(sid, stream_q)
            raise
        except KeyboardInterrupt:
            logger.info("gguf worker: stream %d interrupted after %d token(s)",
                        sid, chunks_received)
            self._cancel_mux_stream_and_drain(
                sid, stream_q,
                _STREAM_CHUNK_TIMEOUT if stop_on_request else _CANCEL_DRAIN_TIMEOUT)
            raise
        finally:
            with self._streams_lock:
                self._streams.pop(sid, None)

    def _cancel_mux_stream_and_drain(self, sid: int, stream_q: _queue.Queue,
                                     timeout: float = _CANCEL_DRAIN_TIMEOUT) -> None:
        """Ask the child to stop stream *sid* and consume its envelopes until its
        "done" or "error", for at most *timeout* seconds; past that the child is
        killed, ending every stream on it."""
        ctrl_q = self._ctrl_q
        if not self.is_alive() or ctrl_q is None:
            return
        try:
            ctrl_q.put(("cancel_stream", sid))
        except Exception:
            self.shutdown(grace=0)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = stream_q.get(timeout=0.5)
            except _queue.Empty:
                if not self.is_alive():
                    return
                continue
            if result[0] == "done":
                if len(result) > 1 and isinstance(result[1], dict):
                    self.last_done = result[1]
                return
            if result[0] == "error":
                return
        from localm.debuglog import logger as _dbg
        _dbg.warning("gguf runner: cancel of stream %d did not confirm within %.0fs; "
                     "killing the worker process", sid, timeout)
        self.shutdown(grace=0)

    def _simple_request(self, name: str, payload, timeout: float = _SIMPLE_CMD_TIMEOUT,
                        *, try_lock: bool = False):
        """Send one request/response command and return its value.

        Holds ``_q_lock`` for the whole exchange so its reply can never be
        stolen by (or steal from) a concurrent stream on the shared response
        queue. ``try_lock=True`` acquires the lock NON-blocking and
        raises :class:`RunnerBusy` immediately if it is held (a live stream, or
        another simple command) - used by the token counters, which have a
        documented heuristic fallback and must not queue a 30s-timeout RPC
        behind a whole generation. The default blocking acquire is for commands
        that genuinely need the real answer (e.g. check_grammar).

        On a multiplexed runner streams do not hold the queue: the request takes
        ``_simple_lock`` instead, waiting up to ``_MUX_SIMPLE_WAIT`` seconds for
        another simple request with *try_lock*, and reads its reply from the
        demux thread's queue."""
        mux = self._mux
        lock = self._simple_lock if mux else self._q_lock
        if try_lock:
            acquired = (lock.acquire(timeout=_MUX_SIMPLE_WAIT) if mux
                        else lock.acquire(blocking=False))
            if not acquired:
                raise RunnerBusy(name)
        else:
            lock.acquire()
        try:
            self._req_q.put((name, payload))
            deadline = time.monotonic() + timeout
            result = None
            while result is None:
                wait = max(0.01, min(0.5, deadline - time.monotonic()))
                try:
                    result = self._poll_simple(wait) if mux else self._poll(wait)
                except _RunnerTornDown as e:
                    raise RuntimeError(
                        f"The model was unloaded while handling '{name}'.") from e
                except _queue.Empty as e:
                    if not self.is_alive():
                        if self._proc is None:
                            raise RuntimeError(
                                f"The model was unloaded while handling '{name}'.") from e
                        raise RuntimeError(
                            f"The model process crashed (exit code "
                            f"{self._exit_reason()}) while handling '{name}'."
                            + self._crash_detail()) from e
                    if time.monotonic() > deadline:
                        self.shutdown(grace=0)
                        raise RuntimeError(f"'{name}' timed out waiting for the model process.") from e
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
                    # decoders read ONE protocol off ONE queue, so a tag honoured
                    # by one and not the other would make the same envelope mean
                    # different things depending on which command was in flight,
                    # and the untagged fallback is RuntimeError, which reads as
                    # "the worker faulted" and unloads the model.
                    raise GrammarUnsupportedError(msg)
                if tag == "ContextCapacityExceededError":
                    raise ContextCapacityExceededError(msg)
                if tag == "PretokenizerUnsafeInputError":
                    raise PretokenizerUnsafeInputError(msg)
                raise RuntimeError(msg)
            raise RuntimeError(f"Unexpected response for '{name}': {result!r}")
        finally:
            lock.release()

    def _poll_simple(self, timeout: float):
        """:meth:`_poll` for a simple request's reply on a multiplexed runner."""
        q = self._simple_q
        if q is None or self._resp_q is None:
            raise _RunnerTornDown
        return q.get(timeout=timeout)

    def count_tokens(self, text: str) -> int:
        # try_lock: never block a token count behind a live generation; the
        # caller (GgufBackend) has a chars/4 fallback for RunnerBusy.
        return self._simple_request("count_tokens", text, try_lock=True)

    def count_messages_tokens(self, messages: list) -> int:
        return self._simple_request("count_messages_tokens", messages, try_lock=True)

    def check_grammar(self, grammar: str) -> None:
        # try_lock: validate_grammar (the only caller) runs SYNCHRONOUSLY on the
        # server's async event loop, so a blocking wait here would freeze the
        # whole loop for the full duration of a concurrent same-model stream that
        # holds the queue. Raise RunnerBusy instead; GgufBackend.validate_grammar
        # defers the check to generation time, which still rejects a malformed
        # grammar with the same clean InvalidGrammarError (llama.py raises it on
        # the native NULL return, before any token) - never a native fault.
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
        # whoever called it, so any trace it left is either already relayed or
        # describes a death nobody will report. It must not outlive the process
        # it describes, or the logs dir grows one file per model load.
        self._discard_native_crash_trace()
        self._crash_trace_path = None
