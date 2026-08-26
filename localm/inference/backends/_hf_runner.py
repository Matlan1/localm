# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the whole HuggingFace-transformers backend
lifecycle (load, tokenize, embed, generate, unload).

None of ``HFWorker.count_tokens()`` (a Rust fast tokenizer, whose
pre-tokenizer regex stage carries the same native-regex-hang risk a
catastrophic pattern can hit), ``embed()`` (a real torch forward pass) or
``chat_stream()`` (``model.generate()``) is cancellable once started: a Python
``threading`` timeout only stops the CALLER waiting, never the underlying
native call. Running the backend in its own disposable child process makes a
hang killable without taking the server down.

Mid-stream cancellation is COOPERATIVE. ``generate()``'s decode loop calls
every ``StoppingCriteria`` in ``stopping_criteria=`` once per generated token,
so a ``StoppingCriteria`` that polls a ``threading.Event`` is a per-token
cancel hook. A disconnected stream sends ``("cancel_stream", seq)`` over
``ctrl_q``; the child's control thread ``.set()``s the ``threading.Event`` the
worker passes into ``model.generate()`` (``_hf_worker.py``'s
``_CancelCriteria``); and the SAME worker process keeps serving the next
request.

``seq`` is a monotonically increasing id the parent assigns per stream, echoed
on both the ``"chat_stream"`` command and its matching ``"cancel_stream"``.
``ctrl_q`` and ``req_q`` are independent queues with no ordering relationship,
so a cancel for a stream that has already finished can still be sitting on
``ctrl_q`` when a later, unrelated stream starts; ``_ctrl_msg_cancels_seq``
drops it as stale.

Cancellation cannot interrupt an in-flight forward pass or the prompt prefill:
the check runs between decode steps, so a cancelled stream finishes its current
token before stopping. If the child never confirms within
``_CANCEL_DRAIN_TIMEOUT``, ``_cancel_stream_and_drain`` falls back to
``shutdown(grace=0)``.

Protocol (three ``multiprocessing.Queue``s, tagged tuples; ``req_q``/
``resp_q`` process one command at a time, ``ctrl_q`` is drained by a
dedicated control-thread so a cancel signal takes effect even while the main
dispatch thread is blocked on ``req_q.get()`` or forwarding chunks):

``req_q`` (parent -> child):
    ("load", {model_path, device})
    ("chat_stream", {messages, max_tokens, temperature, top_p, top_k,
                      repeat_penalty, grammar, grammar_lazy,
                      grammar_triggers, seed}, seq)
    ("count_tokens", text)
    ("count_messages_tokens", messages)
    ("embed", texts)
    ("shutdown", None)

``ctrl_q`` (parent -> child):
    ("cancel_stream", seq)   - seq must match the CURRENTLY active stream's
                                seq (see _ctrl_msg_cancels_seq) or it is
                                silently dropped as stale.

``resp_q`` (child -> parent):
    ("ok", value)              - success (value shape depends on command;
                                  "load" returns {supports_images, can_embed})
    ("error", message[, kind]) - a clean, expected failure; kind is an
                                  optional typed-exception tag, re-raised as that
                                  type by the parent. Recognised tags:
                                  "UnsupportedInputError",
                                  "GrammarUnsupportedError",
                                  "InvalidGrammarError". An UNTAGGED error becomes
                                  a RuntimeError, which callers read as "the
                                  isolated worker faulted" (503), so anything the
                                  CALLER can fix needs a tag
    ("chunk", text)            - one streamed token (chat_stream only)
    ("done", {"finish_reason": "stop"|"length"}) - end of one chat_stream,
                                  whether it ran to completion, hit a genuine
                                  end-of-sequence token, was cut off by
                                  max_tokens, or was stopped by a cooperative
                                  cancel (reported as "stop", the same value a
                                  normal EOS gets). finish_reason is
                                  ``HFWorker.last_finish_reason``: "stop" when
                                  the model produced its own end-of-sequence
                                  token or was cancelled, "length" when the
                                  max_tokens budget ran out first with no EOS
                                  ever produced.

A native abort, or any other uncaught fault in the child's dispatch loop,
produces NO envelope - the parent detects the dead or stuck child via
``proc.is_alive()``/``exitcode`` and a bounded timeout.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import threading
import time
from typing import List, Optional


class RunnerBusy(Exception):
    """A best-effort, non-blocking command (a token count) declined to run
    because the runner's single response queue is already being driven by
    another command on this process (typically a live ``chat_stream``).

    This is NOT a failure: the caller (``HFBackend``) has a documented
    chars/4 heuristic fallback. The protocol carries no per-request
    correlation id, so an RPC queued behind a live stream would race that
    stream's envelopes on the shared queue."""


# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set; never set in production. Values: "abort" (an uncatchable
# native abort), "exit" (a hard process exit with no Python traceback), "hang"
# (a wedged native call).
_FAULT_ENV = "LOCALM_HF_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    os.abort()                                    # genuine uncatchable native abort


def _ctrl_msg_cancels_seq(msg, current_seq) -> bool:
    """True when *msg* (a ctrl_q message) is a ``cancel_stream`` targeting
    *current_seq* - the currently active stream's sequence number, as known
    at the moment this is evaluated.

    ``ctrl_q`` and ``req_q`` are independent ``multiprocessing.Queue``s with
    no ordering relationship between them, so a cancel the parent sent for
    stream N can still be sitting on ``ctrl_q`` when the dispatch thread has
    already started stream N+1. Without this per-stream identity check the
    control thread would set ``stream_cancel_event`` for N+1, truncating an
    unrelated request.

    Every ``("chat_stream", payload, seq)`` the parent sends and every
    ``("cancel_stream", seq)`` it later sends for that same request carry
    the SAME parent-assigned seq (see ``HFRunner.chat_stream`` /
    ``_cancel_stream_and_drain``), so a cancel only takes effect if its
    target seq still matches whatever stream is actually current when the
    control thread gets to it; a stale one is silently dropped.
    ``target_seq is not None`` guards against an accidental match when
    neither side supplies a real seq (current_seq is None before the first
    stream starts).

    Pure and side-effect-free."""
    if not isinstance(msg, tuple) or not msg or msg[0] != "cancel_stream":
        return False
    target_seq = msg[1] if len(msg) > 1 else None
    return target_seq is not None and target_seq == current_seq


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

_crash_trace_fh = None   # child-side: kept alive so faulthandler can write to it


def _arm_native_crash_trace(path) -> None:
    """Child side: point faulthandler at *path* so a death by native SIGNAL
    leaves a trace the parent can relay into the debug log.

    This is the only mechanism that captures a SIGILL/SIGSEGV/SIGABRT inside
    native code (a torch forward pass, a CUDA/ROCm kernel, a fast tokenizer's
    Rust stage): such a fault never returns to Python, so no ``except`` clause
    can run. Armed before torch or any native library is loaded.

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
                "hf worker: faulthandler.enable() returned without raising "
                "but is_enabled() is False - a native fault in this worker will "
                "produce no stack trace")
    except Exception as e:   # noqa: BLE001 - a diagnostic must never break the worker
        logger.warning(
            "hf worker: could not arm the native-fault trace (%s: %s) - a "
            "native fault in this worker will produce no stack trace",
            type(e).__name__, e)


def _runner_entry(req_q, resp_q, ctrl_q, crash_trace_path=None) -> None:
    """Process target. Wraps ``_runner_main`` so any exception escaping it is
    logged via the ``logging`` module before the process dies. A
    ``logging.FileHandler`` writes through its own Python-level stream,
    independent of fd 2, so it survives the unwind that restores fd 2 out from
    under multiprocessing's own traceback printer. RE-RAISES: this only ADDS a
    capture, never changes how or whether the process exits.

    Does NOT cover a genuine native crash with no Python exception at all
    (SIGSEGV, a raw abort): Python never regains control there, so no ``except``
    clause can run. The faulthandler trace :func:`_arm_native_crash_trace`
    leaves behind covers that half."""
    _arm_native_crash_trace(crash_trace_path)
    try:
        _runner_main(req_q, resp_q, ctrl_q)
    except BaseException:
        from localm.debuglog import attach_child_logging, logger
        attach_child_logging()
        logger.critical("hf worker process crashed", exc_info=True)
        raise


def _runner_main(req_q, resp_q, ctrl_q) -> None:
    """Long-lived child: owns one HFWorker (one loaded model) for its whole
    process lifetime, dispatching one request at a time on ``req_q``/
    ``resp_q``. A dedicated control thread drains ``ctrl_q`` for a
    mid-stream cancel signal and sets ``stream_cancel_event``, which the
    active ``chat_stream``'s ``StoppingCriteria`` polls (see
    ``_hf_worker.py``'s ``_CancelCriteria``). There is no load-cancel message:
    ``spawn_and_load`` below takes no ``cancel_event``."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # native/tokenizer failure diagnostics land in
                              # the shared debug log from this process too.

    from localm._mp_spawn import (install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the parent even on a hard kill
                                       # (End Task / force-close), which
                                       # daemon=True does not cover.
    suppress_native_error_dialogs()   # a native DLL failure here (torch/CUDA/
                                       # ROCm init) must degrade to a catchable
                                       # exception, never a blocking modal dialog.

    from localm.inference.backends._hf_worker import HFWorker
    from localm.inference.backends.base import (
        ContextCapacityExceededError,
        GrammarUnsupportedError,
        InvalidGrammarError,
        UnsupportedInputError,
    )

    stream_cancel_event = threading.Event()
    # The seq of the stream currently being dispatched, as told to us by the
    # parent in its chat_stream command. Written ONLY by the dispatch loop below
    # and read ONLY by _control_loop, a single-int handoff that needs no extra
    # Lock. None before the first stream starts.
    current_seq = [None]

    def _control_loop() -> None:
        while True:
            msg = ctrl_q.get()
            if msg is None:
                return
            if _ctrl_msg_cancels_seq(msg, current_seq[0]):
                stream_cancel_event.set()

    threading.Thread(target=_control_loop, daemon=True, name="localm-hf-ctrl").start()

    worker: Optional[HFWorker] = None

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
            # Explicit teardown before the process exits.
            if worker is not None:
                worker.unload()
            return

        if name == "load":
            # The constructor and load() are BOTH inside this try: a malformed
            # payload or a load failure are clean, catchable Python failures, not
            # native faults, and get the same error envelope. worker is always
            # still None here.
            try:
                worker = HFWorker(**payload)
                worker.load()
                # Computed once here, right after load, and cached on the parent
                # proxy: the model lives in the child and cannot be re-read live.
                resp_q.put(("ok", {
                    "supports_images": worker.supports_images,
                    "can_embed": worker.can_embed,
                    "device": worker.resolved_device,
                    "context_capacity": getattr(worker, "context_capacity", None),
                }))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during worker.load() is NOT caught here: the
            # process dies and the parent detects that via is_alive().
            continue

        if name == "chat_stream":
            # cmd[2] is the parent-assigned seq for THIS stream, recorded before
            # clearing the event so a cancel_stream for this exact seq is always
            # honoured, and one for a stale seq is rejected by
            # _ctrl_msg_cancels_seq regardless of clear() timing.
            current_seq[0] = cmd[2] if len(cmd) > 2 else None
            stream_cancel_event.clear()   # a stale cancel from a PRIOR stream
                                           # on this same model must not fire early
            try:
                gen = worker.chat_stream(cancel_event=stream_cancel_event, **payload)
                for token in gen:
                    resp_q.put(("chunk", token))
                resp_q.put(("done", {"finish_reason": worker.last_finish_reason}))
            except ContextCapacityExceededError as e:
                # An oversized prompt exceeding the model's context capacity,
                # raised in pure Python before native generation: the loaded model
                # is unharmed and the worker keeps running.
                resp_q.put(("error", str(e), "ContextCapacityExceededError"))
            except UnsupportedInputError as e:
                # A clean, expected refusal (an image against a text-only
                # checkpoint). The loaded model is unharmed, so report it and keep
                # serving.
                resp_q.put(("error", str(e), "UnsupportedInputError"))
            except GrammarUnsupportedError as e:
                # _grammar_processor refuses a grammar it cannot apply rather than
                # generating unconstrained text. Raised during setup, before a
                # single token, so the model is untouched and this must not fall
                # through to the worker-killing arm below.
                resp_q.put(("error", str(e), "GrammarUnsupportedError"))
            except InvalidGrammarError as e:
                # A grammar xgrammar could not compile. The caller's input is the
                # problem, so keep serving and let the parent re-raise the typed
                # error the routes map to a 400.
                resp_q.put(("error", str(e), "InvalidGrammarError"))
            # Any OTHER uncaught fault (a torch/CUDA crash inside
            # model.generate(), a tokenizer failure mid-stream) propagates OUT of
            # this whole function, uncaught: the model is left in an unknown
            # state, so this process must not keep serving from it.
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

        if name == "embed":
            try:
                resp_q.put(("ok", worker.embed(payload)))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # An uncaught native fault here (a torch forward-pass crash)
            # propagates uncaught, like chat_stream and load.
            continue

        # An unrecognized command is reported back as an error envelope.
        resp_q.put(("error", f"unknown hf-runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch. One HFRunner per loaded HFBackend,
# never a module-level singleton.
# --------------------------------------------------------------------------- #

# How often the waits below re-check proc.is_alive() while polling for a
# response. A local poll interval, not the command's own deadline.
_POLL_INTERVAL = 0.2

# Default model-load timeout. A stalled load raises rather than hanging forever.
# Overridable via the hf_load_timeout_s config key.
LOAD_TIMEOUT_DEFAULT = 900.0

# Wait for the FIRST envelope of a stream: covers the whole prompt PREFILL, not
# one token's decode. Overridable via hf_first_token_timeout_s.
FIRST_TOKEN_TIMEOUT_DEFAULT = 900.0

# Per-token wait during generation, once streaming has started.
_STREAM_CHUNK_TIMEOUT = 120.0

# Bounded wait for a done envelope after requesting a mid-stream cancel. A
# wedged native call never confirms, so this is the fallback-to-kill bound, not
# an expected steady-state wait.
_CANCEL_DRAIN_TIMEOUT = 5.0

# Bounded wait for a simple request/response command (count_tokens,
# count_messages_tokens).
_SIMPLE_CMD_TIMEOUT = 30.0

# Bounded wait for one embed() RPC, sized independently of the simple-command
# timeout above: HFWorker.embed() loops over texts one at a time with no
# batching, and every HF load is full bf16/fp32. Overridable via
# hf_embed_timeout_s.
EMBED_TIMEOUT_DEFAULT = 600.0

# Per-request caps enforced by HFBackend.embed() before a batch crosses the
# process boundary into the isolated worker. Two independent axes: text COUNT
# (each text is its own forward pass) and total CHARACTERS (a huge text is slow
# to tokenize, and the sentence-transformer encode() path applies no truncation).
# Overridable via hf_embed_max_texts / hf_embed_max_chars.
EMBED_MAX_TEXTS_DEFAULT = 256
EMBED_MAX_CHARS_DEFAULT = 200_000


class HFRunner:
    """Parent-side handle to one isolated HF worker process."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # Serialises PARENT-side use of the single response queue. The worker is
        # already serial, but two parent threads (a live chat_stream's producer
        # and a token-count RPC on an executor thread) could otherwise both call
        # self._resp_q.get() and steal each other's envelopes, since the protocol
        # carries no per-request correlation id. Every command that drives the
        # queue holds this for its whole request/response cycle.
        self._q_lock = threading.Lock()
        self.last_done: dict = {}
        # Monotonically increasing per-stream id, sent with every chat_stream
        # command and echoed back on the matching cancel_stream, so the child's
        # control thread can tell a still-current cancel from a stale one left
        # over by a finished stream. Parent-assigned: the child has no
        # independent notion of which request this is.
        self._stream_seq = 0
        # Where THIS runner's child writes its native-fault trace. Chosen by the
        # parent and set in _spawn(); None before the first spawn.
        self._crash_trace_path = None
        self._shutdown_requested = False

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _exit_reason(self) -> str:
        """The child's exit code DECODED - "-4 (killed by signal SIGILL)" rather
        than "-4".

        Every user-facing report of a dead worker goes through this rather than
        interpolating the raw code. The decoder lives in ``_mp_spawn`` and is
        shared with ``ModelRunner``."""
        from localm._mp_spawn import describe_exit_code
        proc = self._proc
        return describe_exit_code(None if proc is None else proc.exitcode)

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

    def _exit_was_native_fault(self, *, trace_captured: bool) -> bool:
        """Whether this worker's death is EVIDENCED as a native fault.

        The raw exit code has exactly two legitimate consumers - the decoder that
        renders it (:meth:`_exit_reason`) and this classifier - and everything
        else goes through one of those. The predicate lives in ``_mp_spawn`` and
        is shared with ``ModelRunner``."""
        from localm._mp_spawn import death_was_a_native_fault
        proc = self._proc
        return death_was_a_native_fault(None if proc is None else proc.exitcode,
                                        trace_captured=trace_captured)

    def _death_report(self):
        """``(native_evidenced, detail)`` for a dead worker.

        Reads the captured trace EXACTLY ONCE, because reading consumes it and
        both halves need it: the trace decides whether this was a native fault at
        all, and it is also the detail worth relaying.

        Says "no native fault trace was captured" out loud when nothing was
        written, so a reader can tell an empty capture from their own failure to
        find one.

        The non-native branch gives an INSTRUCTION rather than promising a
        traceback: a hard ``os._exit`` produces no exception and therefore no
        traceback."""
        trace = self._native_crash_trace()
        native = self._exit_was_native_fault(trace_captured=bool(trace))
        if not trace:
            return native, " No native fault trace was captured for this exit."
        from localm.debuglog import logger
        # Logged as well as returned: the multi-line trace goes to the debug log
        # the message points at.
        logger.error("hf worker native fault trace:\n%s", trace)
        first = trace.splitlines()[0].strip()
        return native, f" Native fault: {first} (full trace in the debug log)."

    def _crash_detail(self) -> str:
        """Just the detail half of :meth:`_death_report`, for messages whose own
        opening words ("crashed") are already true of any worker death and need no
        native/ordinary distinction."""
        return self._death_report()[1]

    def _spawn(self) -> None:
        self._shutdown_requested = False
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
            self._crash_trace_path = child_crash_trace_path("hf-worker")
        except OSError as e:
            # An unwritable logs dir costs the trace, not the worker.
            logger.warning("could not allocate a native-fault trace file (%s); "
                           "a native fault in this worker will not be traced", e)
            self._crash_trace_path = None
        self._proc = ctx.Process(
            target=_runner_entry,
            args=(self._req_q, self._resp_q, self._ctrl_q,
                  self._crash_trace_path),
            name="localm-hf-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, timeout: float = LOAD_TIMEOUT_DEFAULT) -> dict:
        """Spawn the child and load the model. Returns ``{supports_images,
        can_embed}`` on success. Raises RuntimeError on a genuine load
        failure, a child crash (native abort - detected via is_alive(),
        never an exception this process had to catch), or a timeout (the
        child is killed).

        Takes no ``cancel_event``, unlike ``ModelRunner.spawn_and_load``: this
        backend does not support preemptive load cancellation
        (``BaseBackend.set_load_cancel``'s default no-op is never overridden),
        so there is nothing to relay."""
        self._spawn()
        self._req_q.put(("load", params))
        deadline = time.monotonic() + timeout
        result = None
        while result is None:
            try:
                result = self._resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"The HuggingFace model-loading process crashed (exit "
                        f"code {self._exit_reason()}) while loading. The server "
                        "stayed up." + self._crash_detail())
                if time.monotonic() > deadline:
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"HuggingFace model load timed out after {timeout:.0f}s "
                        "- the worker process may be hung (see the debug log). "
                        "The server stayed up and the load was aborted; retry, "
                        "or raise hf_load_timeout_s if this model genuinely "
                        "needs longer to load.")
        kind = result[0]
        if kind == "ok":
            return result[1]
        if kind == "error":
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the HF model-loading process: {result!r}")

    def chat_stream(self, *, first_chunk_timeout: Optional[float] = None, **kwargs):
        """Yield text tokens. On the caller's ``GeneratorExit`` (a client
        disconnect or a superseding request), requests a cooperative cancel
        and drains for its confirmation - see ``_cancel_stream_and_drain``
        and the module docstring for the mechanism and its fallback to a
        kill.

        Holds ``_q_lock`` for the whole drive so no concurrent token-count
        RPC can consume this stream's envelopes off the shared response
        queue. Released when this generator is exhausted, errors, or is
        closed - all on the single producer thread that drives it, so the
        non-reentrant Lock is always released on the thread that took it."""
        from localm.debuglog import logger
        first_budget = first_chunk_timeout or FIRST_TOKEN_TIMEOUT_DEFAULT
        awaiting_first = True
        with self._q_lock:
            # Incremented under _q_lock: += 1 is a read then a write, not atomic
            # under the GIL, so two callers racing chat_stream() could otherwise
            # be handed the same seq.
            self._stream_seq += 1
            my_seq = self._stream_seq
            self._req_q.put(("chat_stream", kwargs, my_seq))
            try:
                while True:
                    deadline = time.monotonic() + (
                        first_budget if awaiting_first else _STREAM_CHUNK_TIMEOUT)
                    result = None
                    while result is None:
                        if self._shutdown_requested:
                            logger.debug("hf: shutdown requested, ending chat_stream")
                            return
                        try:
                            result = self._resp_q.get(timeout=_POLL_INTERVAL)
                        except (ValueError, _queue.Empty):
                            if self._shutdown_requested:
                                logger.debug("hf: shutdown requested during queue get, ending chat_stream")
                                return
                            if not self.is_alive():
                                # Only call this a native fault when the evidence
                                # says so: an uncaught Python exception in the
                                # worker exits 1, which leaves no native trace and
                                # no damaged model.
                                native, detail = self._death_report()
                                opening = (
                                    "Native inference fault"
                                    if native else
                                    "The model process exited unexpectedly")
                                raise RuntimeError(
                                    f"{opening} (worker exit "
                                    f"{self._exit_reason()}). The model has been "
                                    "unloaded and will reload on the next "
                                    "request." + detail)
                            if time.monotonic() > deadline:
                                self.shutdown(grace=0)
                                if awaiting_first:
                                    raise RuntimeError(
                                        f"Generation stalled: the model process "
                                        f"produced no output within "
                                        f"{first_budget:.0f}s of prompt "
                                        "processing. It has been unloaded and "
                                        "will reload on the next request. Raise "
                                        "hf_first_token_timeout_s if this prompt "
                                        "genuinely needs longer on this hardware.")
                                raise RuntimeError(
                                    "Generation stalled: the model process "
                                    "stopped responding. It has been unloaded "
                                    "and will reload on the next request.")
                    awaiting_first = False
                    kind = result[0]
                    if kind == "chunk":
                        yield result[1]
                    elif kind == "done":
                        self.last_done = result[1]
                        return
                    elif kind == "error":
                        msg = result[1]
                        tag = result[2] if len(result) > 2 else ""
                        if tag == "UnsupportedInputError":
                            from localm.inference.backends.base import UnsupportedInputError
                            raise UnsupportedInputError(msg)
                        # Re-raise the TYPE, not a bare RuntimeError: the routes
                        # map GrammarUnsupportedError and InvalidGrammarError to a
                        # 400 naming the real problem, while a RuntimeError from
                        # this generator means the isolated worker faulted and is
                        # reported as a 503.
                        if tag == "GrammarUnsupportedError":
                            from localm.inference.backends.base import GrammarUnsupportedError
                            raise GrammarUnsupportedError(msg)
                        if tag == "InvalidGrammarError":
                            from localm.inference.backends.base import InvalidGrammarError
                            raise InvalidGrammarError(msg)
                        if tag == "ContextCapacityExceededError":
                            from localm.inference.backends.base import ContextCapacityExceededError
                            raise ContextCapacityExceededError(msg)
                        raise RuntimeError(msg)
                    else:
                        raise RuntimeError(f"Unexpected response during generation: {result!r}")
            except GeneratorExit:
                self._cancel_stream_and_drain(my_seq)
                raise

    def _cancel_stream_and_drain(self, seq) -> None:
        """Ask the child to stop the live generation cooperatively and wait
        for its confirmation. Does NOT cache the drained "done" envelope onto
        ``self.last_done``: a cancelled stream's caller never reaches the code
        that would read it, since ``GeneratorExit`` unwinds straight past it
        (see ``hf.py``'s ``chat_stream``). Falls back to a kill if the child
        never confirms within ``_CANCEL_DRAIN_TIMEOUT``, and never assumes
        cancellation succeeded without seeing it.

        *seq* is this stream's id (see ``HFRunner.__init__``), echoed to the
        child so its control thread can tell this genuine, still-current cancel
        apart from a stale one left over from an already-finished stream - see
        ``_ctrl_msg_cancels_seq``."""
        if self._shutdown_requested or not self.is_alive():
            return
        try:
            self._ctrl_q.put(("cancel_stream", seq))
        except Exception:
            self.shutdown(grace=0)
            return
        deadline = time.monotonic() + _CANCEL_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            try:
                result = self._resp_q.get(timeout=0.5)
            except _queue.Empty:
                if not self.is_alive():
                    return   # died on its own - nothing left to drain
                continue
            if result[0] == "done":
                return
            # A stray chunk racing the cancel is expected - keep draining.
        # Timed out waiting for done: the child may be wedged inside a native call
        # the cancel flag cannot interrupt. Kill it, so the next request on this
        # backend spawns a known-good process instead of reusing one that never
        # confirmed it stopped.
        from localm.debuglog import logger as _dbg
        _dbg.warning("hf runner: cancel_stream did not confirm within %.0fs; "
                     "killing the worker process", _CANCEL_DRAIN_TIMEOUT)
        self.shutdown(grace=0)

    def _simple_request(self, name: str, payload, timeout: float = _SIMPLE_CMD_TIMEOUT,
                        *, try_lock: bool = False):
        """Send one request/response command and return its value.

        ``try_lock=True`` acquires ``_q_lock`` NON-blocking and raises
        :class:`RunnerBusy` immediately if it is held (a live stream, or
        another simple command) - used by the token counters, which have a
        documented heuristic fallback and must not queue a timeout-bound RPC
        behind a whole generation. The default blocking acquire is for
        commands with no honest fallback value (``embed``)."""
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
                    result = self._resp_q.get(timeout=wait)
                except _queue.Empty:
                    if not self.is_alive():
                        raise RuntimeError(
                            f"The HF model process crashed (exit code "
                            f"{self._exit_reason()}) while handling '{name}'."
                            + self._crash_detail())
                    if time.monotonic() > deadline:
                        self.shutdown(grace=0)
                        raise RuntimeError(f"'{name}' timed out waiting for the HF model process.")
            kind = result[0]
            if kind == "ok":
                return result[1]
            if kind == "error":
                raise RuntimeError(result[1])
            raise RuntimeError(f"Unexpected response for '{name}': {result!r}")
        finally:
            self._q_lock.release()

    def count_tokens(self, text: str) -> int:
        # try_lock: never block a token count behind a live generation; the
        # caller has a chars/4 fallback for RunnerBusy.
        return self._simple_request("count_tokens", text, try_lock=True)

    def count_messages_tokens(self, messages: list) -> int:
        return self._simple_request("count_messages_tokens", messages, try_lock=True)

    def embed(self, texts: List[str], timeout: float = EMBED_TIMEOUT_DEFAULT) -> List[List[float]]:
        # NOT try_lock: embedding has no honest fallback value, so a caller that
        # needs it must wait.
        return self._simple_request("embed", texts, timeout=timeout, try_lock=False)

    def shutdown(self, grace: float = 5.0) -> None:
        """Best-effort teardown: ask the worker to close cleanly, then kill it
        if it does not exit within *grace* seconds. Safe to call more than
        once, or when nothing is running."""
        self._shutdown_requested = True
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
