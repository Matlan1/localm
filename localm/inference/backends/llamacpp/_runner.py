# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the whole GGUF model lifecycle (load, generate,
tokenize, grammar-check, unload) - the fix for a confirmed native-abort crash
in ``llama_load_model_from_file``.

WHY the whole lifecycle, not just the load: a ``ctypes.c_void_p`` model/context
handle is meaningless outside the process that created it (no IPC-safe handle
export exists for it, and the underlying CUDA/HIP context is bound to its
owning process), and ``LlamaCpp._prefill_fresh_context`` calls the same
abort-prone native call class again on every context-window GROW (a common
event, not an edge case) - so isolating only the initial load would leave the
identical crash reachable on the very next grow of a model that "loaded
successfully". The model's whole lifecycle therefore runs in a disposable
child process; a native abort there kills only that child, never the server.

This mirrors ``localm/voice.py``'s proven design for the identical class of
problem (an uncatchable native abort in faster-whisper/ctranslate2) - a
long-lived ``multiprocessing.get_context("spawn")`` worker, ``Queue``s for
request/response, ``proc.is_alive()``/``exitcode`` for crash detection, and a
tagged error envelope instead of shipping native exception objects across the
boundary - proven on Windows (this project's primary platform) with its own
crash-containment test suite. The one structural difference: this runner is
INSTANCE-scoped (one ``ModelRunner`` per loaded ``GgufBackend``), not a global
singleton, since multiple GGUF models can be loaded simultaneously (the
existing VRAM-based eviction/LRU in http_server.py). It also needs to stream
(voice.py is one-shot request/response) and support mid-stream cancellation.

Cancellation needs NO changes to llama.py (verified during design):
- Load cancellation already works via ``LlamaCpp(cancel_event=...)``, polled
  by llama.cpp's own native progress callback - the child just creates its
  OWN local ``threading.Event`` for this and a control-thread ``.set()``s it
  on a ``cancel_load`` signal relayed from the parent over ``ctrl_q``.
- Stream cancellation already works via plain Python generator ``.close()``
  (``GeneratorExit`` unwinds ``LlamaCpp._generate``'s lock cleanly) - the
  child's own dispatch loop does this locally now that the generator lives
  there instead of in the parent.

Protocol (three ``multiprocessing.Queue``s, tagged tuples, mirroring the
tagged-envelope style of ``voice.py`` rather than shipping exception objects):

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
                                             an optional typed-exception tag
                                             (e.g. "InvalidGrammarError")
    ("chunk", text)                       - one streamed token (chat_stream only)
    ("done", {finish_reason, grammar_unsupported})  - end of one chat_stream

A native abort, an unrecoverable fault deliberately left uncaught by
``GgufWorker.chat_stream``, or a genuine hang produces NO envelope - the
parent detects the dead/stuck child via ``proc.is_alive()``/``exitcode`` and a
bounded timeout, exactly like ``voice.py``.

``ctrl_q`` (parent -> child), drained by a dedicated control-thread so a
signal takes effect even while the main thread is blocked in a native call:
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

from localm.inference.backends.base import InvalidGrammarError, ModelLoadCancelled


class RunnerBusy(Exception):
    """A best-effort, non-blocking command (a token count) declined to run
    because the runner's single response queue is already being driven by
    another command on this process (typically a live ``chat_stream``).

    This is NOT a failure: the caller has a documented fallback and should use
    it rather than block a request behind a whole generation - or, worse, queue
    an RPC whose reply would race the live stream's envelopes on the shared queue
    (HON-02). ``count_tokens``/``count_messages_tokens`` opt into this (they fall
    back to a chars/4 heuristic), and so does ``check_grammar``:
    ``validate_grammar`` is called SYNCHRONOUSLY on the server's async event
    loop, so a blocking wait there would freeze the whole loop for the length of
    a concurrent same-model stream - a busy check is instead DEFERRED to
    generation time, which rejects a malformed grammar with the same clean
    ``InvalidGrammarError``. ``load`` and ``chat_stream`` never raise this: they
    own their own queue drive."""


# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set. Exists exclusively so the test suite can prove the
# crash-containment property with a REAL uncatchable fault (the same code
# path a genuine native abort would take); never set in production. Values:
# "abort" (a genuine uncatchable native abort), "exit" (a hard process exit,
# no Python traceback), "hang" (a wedged native call). Checked at the top of
# every command dispatch, mirroring localm/voice.py's _FAULT_ENV.
_FAULT_ENV = "LOCALM_GGUF_FAULT_FOR_TEST"


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    os.abort()                                    # genuine uncatchable native abort


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

def _runner_main(req_q, resp_q, ctrl_q) -> None:
    """Long-lived child: owns one GgufWorker (one loaded model) for its whole
    process lifetime, dispatching one request at a time."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # so native load-failure diagnostics captured via
                              # _quiet_stderr/_capture_stderr land in the shared
                              # debug log from this process too, not just stdout.

    from localm._mp_spawn import (install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the server even on a hard kill
                                       # (End Task / force-close) - daemon=True does
                                       # not, it is atexit-gated; else this worker
                                       # outlives the server holding its model in VRAM.
    suppress_native_error_dialogs()   # a native DLL failure here (loading llama.dll
                                       # itself, or the torch/ROCm conflict this
                                       # worker's own VRAM checks are known to hit -
                                       # see _sizing.py) must degrade to a catchable
                                       # exception, never a blocking modal dialog.

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
            worker = GgufWorker(cancel_event=load_cancel_event, **payload)
            try:
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
                }))
            except InvalidGrammarError as e:
                # A malformed grammar the native parser safely rejected (a
                # checked, ordinary Python exception, not a crash) - the
                # loaded model is completely unharmed. Report it cleanly and
                # keep serving; regression pin for a real bug where this used
                # to latch a permanent "grammar unsupported" degrade (see
                # tests/test_grammar_sampling.py). Deliberately NOT caught
                # alongside a genuine native fault (any OTHER exception here
                # propagates uncaught, on purpose - see below).
                resp_q.put(("error", str(e), "InvalidGrammarError"))
            # Any OTHER uncaught fault from the generator (a non-grammar
            # native fault, deliberately re-raised by GgufWorker.chat_stream)
            # propagates OUT of this whole function, uncaught, on purpose:
            # the model is left in an unknown state, so this process should
            # not keep serving from it - see GgufWorker.chat_stream's docstring.
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

        # An unrecognized command is a bug in this module's own parent-side
        # caller, not untrusted input - but rule 5 says never silently drop
        # it either way.
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

# Default model-load timeout. Unlike the VRAM-probe daemon's short bounded
# wait (which has a safe "unmeasurable, skip" fallback), a model load has NO
# safe default when it stalls - it must raise a clear, actionable error, never
# silently report "not loaded". Generous because a multi-GB model on a slow
# disk can legitimately take minutes; overridable per-install via the
# ``gguf_load_timeout_s`` config key (see gguf.py).
LOAD_TIMEOUT_DEFAULT = 900.0

# Per-token wait during generation. Generous (real per-token latency is
# sub-second even on CPU) but still bounded, so a genuinely wedged child is
# detected rather than blocking a request forever. Applies from the FIRST token
# onward - NOT to the first token itself, see below.
_STREAM_CHUNK_TIMEOUT = 120.0

# Wait for the FIRST envelope of a stream, which is a different quantity from
# the per-token ceiling above: nothing can be emitted until the whole prompt has
# been PREFILLED. On CPU (`-g 0`), under heavy partial offload (#549), or with a
# multi-thousand-token prompt (RAG, a long document, a cold mmap cache), prefill
# can legitimately run far longer than any per-token latency - and holding it to
# the per-token ceiling killed the worker and reported a false "stalled", on a
# prompt that would simply never fit under that ceiling no matter how often it
# was retried (REG-606). Sized like LOAD_TIMEOUT_DEFAULT rather than a token
# budget, for the same reason: generous enough to never punish slow-but-working
# hardware, still bounded so a genuinely wedged child is caught. Overridable per
# install via the ``gguf_first_token_timeout_s`` config key (see gguf.py),
# exactly like ``gguf_load_timeout_s``, since this varies far more by install
# than a fixed constant could cover.
FIRST_TOKEN_TIMEOUT_DEFAULT = 900.0

# Bounded wait for a "done" envelope after requesting a mid-stream cancel.
_CANCEL_DRAIN_TIMEOUT = 5.0

# Bounded wait for a simple request/response command (count_tokens, etc.) -
# these never touch a slow native path, so this is intentionally short.
_SIMPLE_CMD_TIMEOUT = 30.0


class ModelRunner:
    """Parent-side handle to one isolated GGUF worker process. One instance
    per loaded ``GgufBackend`` - never a module-level singleton."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # Serialises PARENT-side use of the single response queue. The worker
        # process is already serial (it reads req_q one command at a time), but
        # nothing stopped two PARENT threads - a live chat_stream drive on the
        # stream's producer thread and a token-count RPC on an executor thread -
        # from both calling self._resp_q.get() and STEALING each other's
        # envelopes (HON-02: a stolen chunk drops a token; a stolen "done"
        # spins the stream to its timeout; a delayed reply trips a simple
        # command's timeout and kills the worker mid-generation). Every command
        # that drives the queue holds this for its whole request/response cycle,
        # so envelopes can never interleave across threads. One lock per runner;
        # it is acquired and released on the SAME thread for each command
        # (the stream's whole drive + close runs on one producer thread), so a
        # plain non-reentrant Lock is correct. shutdown() deliberately does NOT
        # take it, so teardown still works while a command holds it.
        self._q_lock = threading.Lock()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # #617: avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._ctrl_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_runner_main, args=(self._req_q, self._resp_q, self._ctrl_q),
            name="localm-gguf-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, cancel_event=None,
                        timeout: float = LOAD_TIMEOUT_DEFAULT) -> dict:
        """Spawn the child and load the model. Returns the metadata dict
        (``n_layers``/``kv_bytes_per_token``/``supports_images``) on success.

        Raises :class:`ModelLoadCancelled` if *cancel_event* fires during the
        load, or :class:`RuntimeError` on a genuine load failure, a child
        crash (native abort - detected via ``is_alive()``, never an
        exception this process had to catch), or a timeout (the child is
        killed; a load has no safe "unmeasurable" fallback, so this always
        raises rather than silently reporting not-loaded)."""
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
                result = self._resp_q.get(timeout=_LOAD_POLL_INTERVAL)
            except _queue.Empty:
                if not self._proc.is_alive():
                    code = self._proc.exitcode
                    raise RuntimeError(
                        f"The native model-loading process crashed (exit code "
                        f"{code}) while loading. The server stayed up; see the "
                        "debug log for the native stack trace. Retry the load, "
                        "or repair the runtime with 'localm setup-llama'."
                    )
                if time.monotonic() > deadline:
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
        if kind == "cancelled":
            raise ModelLoadCancelled(result[1])
        if kind == "error":
            raise RuntimeError(result[1])
        raise RuntimeError(f"Unexpected response from the model-loading process: {result!r}")

    def chat_stream(self, *, first_chunk_timeout: Optional[float] = None, **kwargs):
        """Yield text tokens. On the caller's ``GeneratorExit`` (mirroring how
        ``http_server.py`` cancels a stream today - a plain generator
        ``.close()``), relays a ``cancel_stream`` signal to the child and
        drains for its confirming "done" before returning, so the worker is
        never left mid-generation when this backend serves its next request.

        Polls in short increments (not one big ``get(timeout=...)``) so a
        crashed child is detected promptly - within one poll interval, not
        after the full per-token ceiling - while still allowing up to
        ``_STREAM_CHUNK_TIMEOUT`` of genuine native decode time per token
        before treating it as stalled.

        The FIRST envelope gets its own, much larger budget
        (*first_chunk_timeout*, default ``FIRST_TOKEN_TIMEOUT_DEFAULT``): it
        waits for the whole prompt prefill, not for one token's decode. Keyword-
        only and popped here, so it is never mistaken for a generation
        parameter forwarded to the child in *kwargs*.

        Holds ``_q_lock`` for the whole drive so no concurrent token-count RPC
        can consume this stream's envelopes off the shared response queue
        (HON-02). The lock is acquired here and released when this generator is
        exhausted, errors, or is closed - all of which happen on the single
        producer thread that drives it, so the non-reentrant Lock is always
        released on the thread that took it."""
        first_budget = first_chunk_timeout or FIRST_TOKEN_TIMEOUT_DEFAULT
        awaiting_first = True
        with self._q_lock:
            self._req_q.put(("chat_stream", kwargs))
            try:
                while True:
                    deadline = time.monotonic() + (
                        first_budget if awaiting_first else _STREAM_CHUNK_TIMEOUT)
                    result = None
                    while result is None:
                        try:
                            result = self._resp_q.get(timeout=_LOAD_POLL_INTERVAL)
                        except _queue.Empty:
                            if not self.is_alive():
                                raise RuntimeError(
                                    f"Native inference fault (worker exit "
                                    f"{self._proc.exitcode}). The model has been "
                                    "unloaded and will reload on the next request. "
                                    "See the debug log for the native stack trace."
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
                    awaiting_first = False
                    kind = result[0]
                    if kind == "chunk":
                        yield result[1]
                    elif kind == "done":
                        self.last_done = result[1]
                        return
                    elif kind == "error":
                        # A clean, expected failure the worker deliberately did
                        # NOT let crash the process (e.g. a malformed grammar) -
                        # the model is unharmed and the worker keeps running.
                        msg = result[1]
                        tag = result[2] if len(result) > 2 else ""
                        if tag == "InvalidGrammarError":
                            raise InvalidGrammarError(msg)
                        raise RuntimeError(msg)
                    else:
                        raise RuntimeError(f"Unexpected response during generation: {result!r}")
            except GeneratorExit:
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
                result = self._resp_q.get(timeout=0.5)
            except _queue.Empty:
                if not self.is_alive():
                    return   # died on its own - nothing left to drain
                continue
            if result[0] == "done":
                return
            # A stray chunk racing the cancel is expected - keep draining.
        # Timed out waiting for "done": the child may be wedged inside a
        # native call the cancel flag cannot interrupt. Do NOT silently act as
        # if cancellation succeeded (rule 5) - kill it so the next request on
        # this backend spawns a known-good process instead of reusing one
        # that never confirmed it stopped.
        from localm.debuglog import logger as _dbg
        _dbg.warning("gguf runner: cancel_stream did not confirm within %.0fs; "
                     "killing the worker process", _CANCEL_DRAIN_TIMEOUT)
        self.shutdown(grace=0)

    def _simple_request(self, name: str, payload, timeout: float = _SIMPLE_CMD_TIMEOUT,
                        *, try_lock: bool = False):
        """Send one request/response command and return its value.

        Holds ``_q_lock`` for the whole exchange so its reply can never be
        stolen by (or steal from) a concurrent stream on the shared response
        queue (HON-02). ``try_lock=True`` acquires the lock NON-blocking and
        raises :class:`RunnerBusy` immediately if it is held (a live stream, or
        another simple command) - used by the token counters, which have a
        documented heuristic fallback and must not queue a 30s-timeout RPC
        behind a whole generation. The default blocking acquire is for commands
        that genuinely need the real answer (e.g. check_grammar)."""
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
                            f"The model process crashed (exit code {self._proc.exitcode}) "
                            f"while handling '{name}'.")
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
