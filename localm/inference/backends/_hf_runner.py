# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for the whole HuggingFace-transformers backend
lifecycle (load, tokenize, embed, generate, unload) - the fix for the
"thread cannot be interrupted" class of bug: ``HFWorker.count_tokens()``
calls a Rust "fast" tokenizer directly (its pre-tokenizer regex stage is the
same Oniguruma-class native-regex-hang risk a catastrophic pattern can hit),
``embed()`` runs a real torch forward pass, and ``chat_stream()`` drives
``model.generate()``. None of these are cancellable once started - a Python
``threading`` timeout only stops the CALLER waiting, never the underlying
native call - so a hang used to burn a slot in the server's shared
``asyncio`` default thread pool PERMANENTLY (see
``dev-notes/decisions-2026-07-30-release-gate.md``, Q2). Isolating this
backend in its own disposable child process is what makes a hang killable
without taking the server down with it.

Mirrors ``backends/llamacpp/_runner.py``'s ``ModelRunner`` (PR #606) and
``_embedder_runner.py``'s ``EmbedderRunner`` closely: three
``multiprocessing.Queue``s, tagged-tuple commands/responses, ``proc.
is_alive()``/``exitcode`` for crash detection, a bounded RPC timeout per
command that kills the child and raises on expiry rather than waiting
forever.

Mid-stream cancellation is COOPERATIVE, the same shape as ``ModelRunner``'s
``ctrl_q``/``_cancel_stream_and_drain`` design, though built on a different
hook: transformers exposes nothing shaped like llama.cpp's native
progress-callback, but ``generate()``'s own decode loop calls every
``StoppingCriteria`` in ``stopping_criteria=`` once per generated token
(verified directly against transformers' own ``generation/utils.py``:
``unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids,
scores)``, checked right after each token is pushed onto the streamer) - so a
``StoppingCriteria`` that polls a ``threading.Event`` is a real per-token
cancel hook. A disconnected stream now sends ``("cancel_stream", seq)`` over
``ctrl_q``; the child's control-thread ``.set()``s a ``threading.Event`` the
worker passes into ``model.generate()`` as a ``StoppingCriteria``
(``_hf_worker.py``'s ``_CancelCriteria``); and the SAME worker process keeps
serving the next request instead of respawning. ``seq`` (a monotonically
increasing id the parent assigns per stream, echoed on both the
``"chat_stream"`` command and its matching ``"cancel_stream"``) exists
because ``ctrl_q`` and ``req_q`` are independent queues with no ordering
relationship: if a stream finishes naturally right as it is also being
cancelled, the cancel can still be sitting on ``ctrl_q`` when the dispatch
loop has already moved on to a later, unrelated stream - without the seq
check (``_ctrl_msg_cancels_seq``), that stale message would wrongly cancel
the new stream instead of being silently dropped. One residual limit, shared
with GGUF: cancellation still cannot interrupt an in-flight forward pass or
the prompt prefill - the check only runs between decode steps, so a
cancelled stream still finishes its current token (at most one extra token
past the signal) before stopping. If the child never confirms within
``_CANCEL_DRAIN_TIMEOUT`` (a genuinely wedged native call - the same
uninterruptible-from-Python risk this whole module exists to contain),
``_cancel_stream_and_drain`` falls back to ``shutdown(grace=0)`` exactly as
before: kill is now the timeout fallback, not the primary path. The
previously accepted cost (a disconnect-heavy client serializing its later
requests behind a full reload) no longer applies to the common case; see
``hf.py`` for what changed on the parent-proxy side (nothing - it already
propagates ``GeneratorExit`` through untouched and already reads worker
liveness live).

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
                                  cancel (see ``_CancelCriteria`` above -
                                  reported as "stop", the same value a normal
                                  EOS gets, since a cancel is not a length
                                  cutoff). finish_reason is
                                  ``HFWorker.last_finish_reason``, computed for
                                  real by ``HFWorker.chat_stream`` (see
                                  ``_hf_worker.py``'s ``_FinishReasonObserver``)
                                  - "stop" when the model produced its own
                                  end-of-sequence token (or was cancelled),
                                  "length" when the max_tokens budget ran out
                                  first with no EOS ever produced. Mirrors
                                  GgufBackend/ModelRunner's identical "done"
                                  envelope shape (llamacpp/_runner.py).

A native abort, or any other uncaught fault in the child's dispatch loop,
produces NO envelope - the parent detects the dead/stuck child via
``proc.is_alive()``/``exitcode`` and a bounded timeout, exactly like
``ModelRunner``/``EmbedderRunner``.
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
    chars/4 heuristic fallback and should use it rather than block a request
    behind a whole generation - or, worse, queue an RPC whose reply would
    race the live stream's envelopes on the shared queue (the protocol
    carries no per-request correlation id). Mirrors
    ``llamacpp._runner.RunnerBusy`` exactly in spirit; kept as its own class
    here rather than imported, matching how every other piece of this
    isolation layer mirrors rather than couples to the GGUF one - the two
    runners share zero imports beyond the common ``_mp_spawn.py`` leaf."""


# Fault-injection hook, honoured by the child ONLY when this environment
# variable is set. Exists exclusively so the test suite can prove the
# crash-containment property with a REAL uncatchable fault (the same code
# path a genuine native abort would take); never set in production. Values:
# "abort" (a genuine uncatchable native abort), "exit" (a hard process exit,
# no Python traceback), "hang" (a wedged native call). Mirrors
# llamacpp/_runner.py's _FAULT_ENV / embedder's / voice.py's.
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
    no ordering relationship between them: the parent sending a cancel
    before starting a new "chat_stream" does NOT guarantee the child's
    control-thread drains that cancel before the child's dispatch thread
    moves on. If stream N finishes naturally (its own "done" already sent)
    right as the parent decides to cancel it, the parent's cancel can still
    be sitting on ``ctrl_q`` when the dispatch thread starts stream N+1 -
    without a per-stream identity check, the control-thread would set
    ``stream_cancel_event`` for N+1 once it finally drains N's stale
    message, silently truncating an unrelated request to ~1 token.

    Every ``("chat_stream", payload, seq)`` the parent sends and every
    ``("cancel_stream", seq)`` it later sends for that same request carry
    the SAME parent-assigned seq (see ``HFRunner.chat_stream`` /
    ``_cancel_stream_and_drain``), so a cancel only takes effect if its
    target seq still matches whatever stream is actually current when the
    control-thread gets to it - a stale one is silently, correctly dropped.
    ``target_seq is not None`` guards against an accidental match when
    neither side supplies a real seq (current_seq defaults to None before
    the first stream starts).

    Pure and side-effect-free so it is unit-testable without a real
    subprocess or model."""
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

    THIS IS THE ONLY THING THAT CAN CAPTURE THAT CLASS. ``_runner_entry``'s
    ``except BaseException`` below covers a crash that still has a Python
    exception; a SIGILL/SIGSEGV/SIGABRT inside native code (a torch forward
    pass, a CUDA/ROCm kernel, a fast tokenizer's Rust stage) never returns to
    Python at all, so no handler written in Python can run and this runner's
    "see the debug log for the native stack trace" had nothing behind it.

    Ported from ``llamacpp/_runner.py``, where the same gap was closed first;
    issues 1222 / 1223 are that shape (``worker exit -4`` is SIGILL, since
    multiprocessing reports ``-N`` for signal N) and neither field log contains
    any trace.

    Armed as early as possible - before torch or any native library is anywhere
    near loaded - because a fault can only be captured by a handler that was
    already installed when it happened.

    Failures are logged, never raised: losing the trace must not stop the worker
    from doing its job. But it is NOT silenced (AGENTS.md rule 5) and
    ``is_enabled()`` is checked rather than trusting "enable() did not raise" -
    that exact silent-no-op is on record in bugreport.arm_crash_guard, where
    every native-trace file on the maintainer's box came out 0 bytes with no
    clue why."""
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
    logged via the ``logging`` module before the process dies, not left to
    multiprocessing's own ``traceback.print_exc()`` alone - mirrors
    ``llamacpp._runner._runner_entry``'s reasoning exactly (native stderr
    redirects installed during a load/generate can restore fd 2 as an
    escaping exception unwinds through them, before ``_bootstrap`` ever gets
    to print anything; a ``logging.FileHandler`` writes through its own
    Python-level stream, independent of fd 2, so it survives that unwind).
    Deliberately RE-RAISES: this only ADDS a capture, never changes how or
    whether the process exits.

    Does NOT help a genuine native crash with no Python exception at all
    (SIGSEGV, a raw abort): Python never regains control there, so no ``except``
    clause, including this one, can run. That residual half is covered by the
    faulthandler trace :func:`_arm_native_crash_trace` leaves behind, which is
    the one mechanism that CAN say where such a fault happened."""
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
    ``resp_q``. A dedicated control-thread drains ``ctrl_q`` for a
    mid-stream cancel signal and sets ``stream_cancel_event``, which the
    active ``chat_stream``'s ``StoppingCriteria`` polls (see
    ``_hf_worker.py``'s ``_CancelCriteria``) - mirrors
    ``llamacpp/_runner.py``'s ``_control_loop``, minus the load-cancel
    message HF never supported (``spawn_and_load`` below still takes no
    ``cancel_event`` - see its docstring for why)."""
    from localm.debuglog import attach_child_logging
    attach_child_logging()   # native/tokenizer failure diagnostics land in
                              # the shared debug log from this process too.

    from localm._mp_spawn import (install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()   # die with the parent even on a hard kill
                                       # (End Task / force-close); daemon=True is
                                       # atexit-gated and does not cover that, so
                                       # else this worker outlives the server
                                       # holding its model in VRAM.
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
    # parent in its "chat_stream" command - written ONLY by the dispatch
    # loop below, read ONLY by _control_loop, a single-int handoff safe
    # under CPython's GIL without an extra Lock (the same assumption
    # threading.Event's own internal state already relies on). None before
    # the first stream ever starts.
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
            # Mirrors _runner.py's identical branch (worker.close() there):
            # explicit teardown before the process exits, not a bare return
            # relying on "the OS reclaims it anyway" - that claim was never
            # verified for torch/CUDA/ROCm specifically, and the established
            # precedent this module mirrors everywhere else does the explicit
            # unload rather than assume it is unnecessary.
            if worker is not None:
                worker.unload()
            return

        if name == "load":
            # The constructor + load() are BOTH inside this try: a malformed
            # payload (a parent/child protocol bug) or a load failure are both
            # clean, catchable Python failures, not native faults - they
            # deserve the same "error" envelope. spawn_and_load() always
            # calls self._spawn() immediately before sending "load", so
            # `worker` is always still None here.
            try:
                worker = HFWorker(**payload)
                worker.load()
                # Computed ONCE here, right after load, and cached on the
                # parent proxy - the old in-process HFBackend re-read these
                # live on every access (self._is_multimodal / self._model),
                # which is no longer possible once the model lives in a
                # child. See hf.py for the liveness-gated caching.
                resp_q.put(("ok", {
                    "supports_images": worker.supports_images,
                    "can_embed": worker.can_embed,
                    "device": worker.resolved_device,
                    "context_capacity": getattr(worker, "context_capacity", None),
                }))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during worker.load() is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "chat_stream":
            # cmd[2] is the parent-assigned seq for THIS stream (see
            # HFRunner.chat_stream) - recorded before clearing the event so
            # a cancel_stream that arrives for this exact seq is always
            # honoured, and one that arrives for a DIFFERENT (stale) seq is
            # rejected by _ctrl_msg_cancels_seq regardless of clear() timing.
            current_seq[0] = cmd[2] if len(cmd) > 2 else None
            stream_cancel_event.clear()   # a stale cancel from a PRIOR stream
                                           # on this same model must not fire early
            try:
                gen = worker.chat_stream(cancel_event=stream_cancel_event, **payload)
                for token in gen:
                    resp_q.put(("chunk", token))
                resp_q.put(("done", {"finish_reason": worker.last_finish_reason}))
            except ContextCapacityExceededError as e:
                # An oversized prompt exceeding the model's context capacity.
                # Raised in pure Python before native generation - the loaded model
                # is unharmed and the worker keeps running.
                resp_q.put(("error", str(e), "ContextCapacityExceededError"))
            except UnsupportedInputError as e:
                # A clean, expected refusal (e.g. an image against a
                # text-only checkpoint) - the loaded model is unharmed.
                # Report it cleanly and keep serving, mirroring how
                # GgufWorker's chat_stream distinguishes InvalidGrammarError
                # from a genuine native fault.
                resp_q.put(("error", str(e), "UnsupportedInputError"))
            except GrammarUnsupportedError as e:
                # Same shape: _grammar_processor now REFUSES a grammar it cannot
                # apply instead of returning None and generating unconstrained
                # text (NEW-LAZY-GRAMMAR-SILENT-UNCONSTRAINED). Raised during
                # setup, before a single token, so the model is untouched - it
                # must not fall through to the worker-killing arm below.
                resp_q.put(("error", str(e), "GrammarUnsupportedError"))
            except InvalidGrammarError as e:
                # A grammar xgrammar could not compile. The caller's input is the
                # problem, not this worker, so keep serving and let the parent
                # re-raise the typed error the routes already map to a 400.
                resp_q.put(("error", str(e), "InvalidGrammarError"))
            # Any OTHER uncaught fault (a torch/CUDA crash inside
            # model.generate(), a tokenizer failure mid-stream) propagates
            # OUT of this whole function, uncaught, on purpose: the model is
            # left in an unknown state, so this process should not keep
            # serving from it.
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
            # Same as chat_stream/load: an uncaught native fault here (a
            # torch forward-pass crash) propagates uncaught, on purpose.
            continue

        # An unrecognized command is a bug in this module's own parent-side
        # caller, not untrusted input - but rule 5 says never silently drop
        # it either way.
        resp_q.put(("error", f"unknown hf-runner command: {name!r}"))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch. Instance-scoped (one HFRunner
# per loaded HFBackend), mirroring ModelRunner - never a module-level
# singleton, since multiple HF models can be loaded simultaneously (the
# existing VRAM-based eviction/LRU in http_server.py).
# --------------------------------------------------------------------------- #

# How often the various waits below re-check proc.is_alive() while polling
# for a response - short, since it is purely a local poll interval, not the
# command's own deadline.
_POLL_INTERVAL = 0.2

# Default model-load timeout. HF loads read full-precision safetensors from
# disk (no quantized-mmap fast path the way GGUF has), so this mirrors
# LOAD_TIMEOUT_DEFAULT from llamacpp/_runner.py rather than assuming HF is
# faster - a stalled load has no safe "unmeasurable" fallback, it must raise
# rather than hang forever. Overridable via the ``hf_load_timeout_s`` config
# key (see hf.py).
LOAD_TIMEOUT_DEFAULT = 900.0

# Wait for the FIRST envelope of a stream - covers the whole prompt PREFILL,
# not one token's decode, so it is sized like the load timeout rather than a
# tight per-token ceiling (mirrors llamacpp/_runner.py's
# FIRST_TOKEN_TIMEOUT_DEFAULT exactly). Overridable via
# ``hf_first_token_timeout_s``.
FIRST_TOKEN_TIMEOUT_DEFAULT = 900.0

# Per-token wait during generation, once streaming has started. Mirrors
# llamacpp/_runner.py's _STREAM_CHUNK_TIMEOUT verbatim as a STARTING default;
# flagged in the plan as an unvalidated assumption for HF specifically - that
# constant's own justification assumes llama.cpp's quantized ggml kernels,
# which HF's always-full-precision CPU path does not share. Not preemptively
# widened without a measurement to justify a different number; if a slow-CPU
# HF report surfaces, this is the first constant to revisit.
_STREAM_CHUNK_TIMEOUT = 120.0

# Bounded wait for a "done" envelope after requesting a mid-stream cancel.
# Mirrors llamacpp/_runner.py's identical constant - a genuinely wedged
# native call (the same uninterruptible-from-Python risk this whole module
# exists to contain) never confirms, so this is the fallback-to-kill bound,
# not an expected steady-state wait.
_CANCEL_DRAIN_TIMEOUT = 5.0

# Bounded wait for a simple request/response command (count_tokens,
# count_messages_tokens). Mirrors llamacpp/_runner.py's
# _SIMPLE_CMD_TIMEOUT - HF's fast tokenizer path is the same "never touch a
# slow native path" assumption as GGUF's RPCs, so the same bound applies
# (the slow pure-Python AutoTokenizer fallback hf.py can hit on a broken
# processor load is a lower-probability, still-bounded-by-this-timeout case,
# not a reason to widen the common case).
_SIMPLE_CMD_TIMEOUT = 30.0

# Bounded wait for one embed() RPC. Deliberately NOT the same bound as the
# simple-command timeout above, and NOT reused unmodified from the dedicated
# GGUF-based embedder's own 300s: HFWorker.embed() loops over texts one at a
# time with no batching, and HFWorker.load() never sets any quantization
# config - every HF load is full bf16/fp32, unlike the embedder's
# small-purpose-built-model assumption. A large unchunked batch against a
# full-precision CPU-fallback model can plausibly run longer than the
# dedicated embedder ever needs to - HFBackend.embed() rejects an oversized
# one outright (see EMBED_MAX_TEXTS_DEFAULT/EMBED_MAX_CHARS_DEFAULT below),
# but this timeout still bounds whatever is allowed through.
# Overridable via ``hf_embed_timeout_s``.
EMBED_TIMEOUT_DEFAULT = 600.0

# Per-request caps enforced by HFBackend.embed() before a batch is ever
# handed to this module's IPC (i.e. before it crosses the process boundary
# into the isolated worker) - see that method for the enforcement and
# hf.py's _embed_max_texts()/_embed_max_chars() for the config-read pattern.
# Two independent axes because either alone is a distinct hang vector: many
# texts means many one-at-a-time forward passes (the AutoModel path, see
# HFWorker.embed), while a huge individual or aggregate text can be slow to
# even tokenize, and the sentence-transformer `.encode()` path applies no
# truncation at all. Overridable via ``hf_embed_max_texts`` /
# ``hf_embed_max_chars``.
EMBED_MAX_TEXTS_DEFAULT = 256
EMBED_MAX_CHARS_DEFAULT = 200_000


class HFRunner:
    """Parent-side handle to one isolated HF worker process."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._ctrl_q = None
        # Serialises PARENT-side use of the single response queue - the
        # worker itself is already serial (reads req_q one command at a
        # time), but nothing stops two PARENT threads (a live chat_stream's
        # producer thread and a token-count RPC on an executor thread) both
        # calling self._resp_q.get() and stealing each other's envelopes,
        # since this protocol carries no per-request correlation id. Every
        # command that drives the queue holds this for its whole
        # request/response cycle. Mirrors ModelRunner._q_lock exactly.
        self._q_lock = threading.Lock()
        self.last_done: dict = {}
        # Monotonically increasing per-stream id, sent with every
        # "chat_stream" command and echoed back on the matching
        # "cancel_stream" - lets the child's control-thread tell a genuine,
        # still-current cancel apart from a stale one left over from a
        # stream that already finished (see _ctrl_msg_cancels_seq's
        # docstring for the race this closes). Parent-assigned rather than
        # child-assigned: the child has no independent notion of "which
        # request is this" beyond what the parent tells it.
        self._stream_seq = 0
        # Where THIS runner's child writes its native-fault trace. Chosen by the
        # parent (see debuglog.child_crash_trace_path for why it is not
        # recomputed child-side) and set in _spawn(); None before the first spawn.
        self._crash_trace_path = None
        self._shutdown_requested = False

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _exit_reason(self) -> str:
        """The child's exit code DECODED - "-4 (killed by signal SIGILL)" rather
        than "-4".

        Every user-facing report of a dead worker goes through this rather than
        interpolating the raw code, mirroring ``ModelRunner._exit_reason``. The
        decoder lives in ``_mp_spawn`` precisely so this runner reuses it
        instead of growing a second version."""
        from localm._mp_spawn import describe_exit_code
        proc = self._proc
        return describe_exit_code(None if proc is None else proc.exitcode)

    def _native_crash_trace(self) -> str:
        """This child's captured native-fault trace, consumed and removed, or ""
        when there is none.

        Consuming rather than merely reading is deliberate: the file is a
        one-shot record of one death, so leaving it in place would let a later
        reader (or the next spawn of a reused runner) attribute a stale trace to
        a fresh crash. Fully guarded - a diagnostic read must never replace the
        real crash error with an IO error."""
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

        A named accessor per concern, exactly like :meth:`_exit_reason` above, and
        mirroring ``ModelRunner._exit_was_native_fault``: the raw exit code has
        precisely two legitimate consumers - the decoder that renders it and the
        classifier that interprets it - and everything else goes through one of
        those. The predicate lives in ``_mp_spawn`` so this runner reuses it
        rather than growing a second version."""
        from localm._mp_spawn import death_was_a_native_fault
        proc = self._proc
        return death_was_a_native_fault(None if proc is None else proc.exitcode,
                                        trace_captured=trace_captured)

    def _death_report(self):
        """``(native_evidenced, detail)`` for a dead worker.

        Reads the captured trace EXACTLY ONCE, because reading consumes it and
        both halves need it: the trace is the strongest evidence of whether this
        was a native fault at all, and it is also the detail worth relaying.

        Saying "no native fault trace was captured" OUT LOUD matters as much as
        relaying one (AGENTS.md rule 5): the message used to claim a trace was in
        the debug log whether or not anything had written one, so a user following
        that instruction found nothing and could not tell an empty capture from
        their own failure to find it.

        The non-native branch gives an INSTRUCTION rather than a promise - a
        Python exception escaping the worker body is logged with its traceback,
        but a hard ``os._exit`` produces no exception and therefore no traceback,
        and promising one for that case would repeat the very defect this fixes."""
        trace = self._native_crash_trace()
        native = self._exit_was_native_fault(trace_captured=bool(trace))
        if not trace:
            return native, " No native fault trace was captured for this exit."
        from localm.debuglog import logger, native_fault_hint
        # Logged as well as returned: the trace is multi-line and belongs in the
        # debug log the message points at, not inlined into an HTTP error body.
        logger.error("hf worker native fault trace:\n%s", trace)
        first = trace.splitlines()[0].strip()
        return native, f" Native fault: {first} ({native_fault_hint()})."

    def _crash_detail(self) -> str:
        """Just the detail half of :meth:`_death_report`, for messages whose own
        opening words ("crashed") are already true of any worker death and need no
        native/ordinary distinction."""
        return self._death_report()[1]

    def _spawn(self) -> None:
        self._shutdown_requested = False
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # #617: avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._ctrl_q = ctx.Queue()
        # A previous child of this runner may have left one behind (a crash whose
        # trace nothing consumed); drop it before the new child claims the name,
        # so a stale trace can never be reported against the new process.
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

        No ``cancel_event`` parameter, unlike ``ModelRunner.spawn_and_load``:
        the in-process ``HFBackend`` this replaces never supported
        preemptive load cancellation either (``BaseBackend.set_load_cancel``'s
        default no-op was never overridden), so there is nothing to relay -
        adding cancellation here would be new behavior, not a preserved
        contract."""
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
            # Incremented under _q_lock, not before it: two Python-level
            # callers racing chat_stream() before either acquires the lock
            # could otherwise interleave the += 1 (a read then a write, not
            # atomic under the GIL as a compound op) and hand out the same
            # seq to two different streams.
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
                                # Do NOT call this a native fault unless the
                                # evidence says so. An uncaught Python exception
                                # in the worker exits 1 - multiprocessing's own
                                # signature for that - and reporting it as a
                                # native fault is false in every clause (no
                                # native fault, no native trace, model
                                # unharmed). See
                                # tests/test_image_decode_without_pillow.py,
                                # where exactly that wrong message is the
                                # subject: a missing Pillow reported as "Native
                                # inference fault (worker exit 1)". Fixed for
                                # Pillow specifically; the misclassification
                                # lives HERE, at the site that words it.
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
                        # this generator means "the isolated worker faulted" and
                        # is reported as a 503. Collapsing a caller's bad grammar
                        # into the worker-fault arm would tell them to fix the
                        # wrong thing - the same mislabelling routes/chat.py's own
                        # worker-fault comment was written to correct.
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
        for its confirmation, mirroring ``ModelRunner._cancel_stream_and_drain``
        exactly (including NOT caching the drained "done" envelope onto
        ``self.last_done`` - a cancelled stream's caller never reaches the
        code that would read it, since ``GeneratorExit`` unwinds straight
        past it; see ``hf.py``'s ``chat_stream``). Falls back to a kill if
        the child never confirms within ``_CANCEL_DRAIN_TIMEOUT`` - never
        assumes cancellation succeeded without seeing it (rule 5).

        *seq* is this stream's id (see ``HFRunner.__init__``), echoed to the
        child so its control-thread can tell this genuine, still-current
        cancel apart from a stale one left over from an already-finished
        stream - see ``_ctrl_msg_cancels_seq``."""
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
        # Timed out waiting for "done": the child may be wedged inside a
        # native call the cancel flag cannot interrupt. Do NOT silently act
        # as if cancellation succeeded (rule 5) - kill it so the next
        # request on this backend spawns a known-good process instead of
        # reusing one that never confirmed it stopped.
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
        # caller (HFBackend) has a chars/4 fallback for RunnerBusy.
        return self._simple_request("count_tokens", text, try_lock=True)

    def count_messages_tokens(self, messages: list) -> int:
        return self._simple_request("count_messages_tokens", messages, try_lock=True)

    def embed(self, texts: List[str], timeout: float = EMBED_TIMEOUT_DEFAULT) -> List[List[float]]:
        # NOT try_lock: unlike a token count, embedding has no honest
        # fallback value - a caller that needs it must wait, mirroring
        # IsolatedEmbedder.embed()'s plain blocking _rpc_lock.
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
        # whoever called it, so any trace it left is either already relayed or
        # describes a death nobody is going to report. Either way it must not
        # outlive the process it describes, or the logs dir grows one file per
        # model load for the life of the server.
        self._discard_native_crash_trace()
        self._crash_trace_path = None
