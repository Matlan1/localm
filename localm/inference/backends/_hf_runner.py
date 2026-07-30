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
``_embedder_runner.py``'s ``EmbedderRunner`` almost exactly: two
``multiprocessing.Queue``s, tagged-tuple commands/responses, ``proc.
is_alive()``/``exitcode`` for crash detection, a bounded RPC timeout per
command that kills the child and raises on expiry rather than waiting
forever. ONE deliberate structural difference from ``ModelRunner``: NO
``ctrl_q``. GGUF's mid-stream cancel is graceful (a ``ctrl_q`` signal the
child's control-thread relays into llama.cpp's own progress-callback hook,
which the native sampler loop actually checks between tokens) because
llama.cpp exposes that hook. Transformers exposes nothing equivalent -
``model.generate()`` runs to completion or crash on its own background
thread with no way to ask it to stop early - so trying a cooperative signal
first, the way ``ModelRunner._cancel_stream_and_drain`` does, would be a
pure, structural wait for zero benefit on every single HF cancel. HF's
cancel is KILL-BASED ONLY: a disconnected stream calls ``shutdown(grace=0)``
directly, and the next request to this backend transparently respawns and
reloads. This is still a strict improvement over the in-process code this
replaces, which could not reclaim the leaked ``model.generate()`` thread at
all (killing an expendable child does; killing the server's own thread does
not exist as an option). The accepted cost: a disconnect-heavy client
against a large/slow-loading model now serializes its later requests behind
a full reload rather than a graceful resume - see ``hf.py`` for the fuller
trade-off writeup.

Protocol (two ``multiprocessing.Queue``s, tagged tuples, one command
processed at a time):

``req_q`` (parent -> child):
    ("load", {model_path, device})
    ("chat_stream", {messages, max_tokens, temperature, top_p, top_k,
                      repeat_penalty, grammar, grammar_lazy,
                      grammar_triggers, seed})
    ("count_tokens", text)
    ("count_messages_tokens", messages)
    ("embed", texts)
    ("shutdown", None)

``resp_q`` (child -> parent):
    ("ok", value)              - success (value shape depends on command;
                                  "load" returns {supports_images, can_embed})
    ("error", message[, kind]) - a clean, expected failure; kind is an
                                  optional typed-exception tag (e.g.
                                  "UnsupportedInputError")
    ("chunk", text)            - one streamed token (chat_stream only)
    ("done", {})                - end of one chat_stream. Deliberately EMPTY:
                                  the in-process HFBackend this replaces never
                                  set a real ``last_finish_reason`` either (no
                                  attribute existed at all, so
                                  ``Engine.last_finish_reason``'s
                                  ``getattr(..., "stop")`` fallback always
                                  fired) - this preserves that EXACT behavior
                                  rather than silently fixing or silently
                                  perpetuating it disguised as now-handled.
                                  See hf.py and the tracked follow-up task.

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


# --------------------------------------------------------------------------- #
# Child side - runs ONLY inside the isolated worker process.
# --------------------------------------------------------------------------- #

def _runner_entry(req_q, resp_q) -> None:
    """Process target. Wraps ``_runner_main`` so any exception escaping it is
    logged via the ``logging`` module before the process dies, not left to
    multiprocessing's own ``traceback.print_exc()`` alone - mirrors
    ``llamacpp._runner._runner_entry``'s reasoning exactly (native stderr
    redirects installed during a load/generate can restore fd 2 as an
    escaping exception unwinds through them, before ``_bootstrap`` ever gets
    to print anything; a ``logging.FileHandler`` writes through its own
    Python-level stream, independent of fd 2, so it survives that unwind).
    Deliberately RE-RAISES: this only ADDS a capture, never changes how or
    whether the process exits."""
    try:
        _runner_main(req_q, resp_q)
    except BaseException:
        from localm.debuglog import attach_child_logging, logger
        attach_child_logging()
        logger.critical("hf worker process crashed", exc_info=True)
        raise


def _runner_main(req_q, resp_q) -> None:
    """Long-lived child: owns one HFWorker (one loaded model) for its whole
    process lifetime, dispatching one request at a time. Fully sequential,
    single-threaded dispatch (no separate control thread) - there is no
    cooperative cancel signal to relay (see module docstring), so unlike
    ``ModelRunner`` there is nothing a second thread would ever need to do
    here."""
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
    from localm.inference.backends.base import UnsupportedInputError

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
                }))
            except Exception as e:
                resp_q.put(("error", str(e)))
            # A hard native abort during worker.load() is NOT caught here -
            # the process dies, and the parent detects that via is_alive().
            continue

        if name == "chat_stream":
            try:
                gen = worker.chat_stream(**payload)
                for token in gen:
                    resp_q.put(("chunk", token))
                resp_q.put(("done", {}))
            except UnsupportedInputError as e:
                # A clean, expected refusal (e.g. an image against a
                # text-only checkpoint) - the loaded model is unharmed.
                # Report it cleanly and keep serving, mirroring how
                # GgufWorker's chat_stream distinguishes InvalidGrammarError
                # from a genuine native fault.
                resp_q.put(("error", str(e), "UnsupportedInputError"))
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
# time with no batching, /v1/embeddings passes its input through with no
# size cap, and HFWorker.load() never sets any quantization config - every
# HF load is full bf16/fp32, unlike the embedder's small-purpose-built-model
# assumption. A large unchunked batch against a full-precision CPU-fallback
# model can plausibly run longer than the dedicated embedder ever needs to.
# Overridable via ``hf_embed_timeout_s``.
EMBED_TIMEOUT_DEFAULT = 600.0


class HFRunner:
    """Parent-side handle to one isolated HF worker process."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
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

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()   # #617: avoid a renamed-launcher WinError 2
        ctx = mp.get_context("spawn")   # explicit: identical on every OS
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_runner_entry, args=(self._req_q, self._resp_q),
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
                    code = self._proc.exitcode
                    raise RuntimeError(
                        f"The HuggingFace model-loading process crashed (exit "
                        f"code {code}) while loading. The server stayed up; "
                        "see the debug log for the native stack trace.")
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
        disconnect or a superseding request), kills the worker directly -
        see the module docstring for why this is kill-based rather than
        graceful like GGUF's.

        Holds ``_q_lock`` for the whole drive so no concurrent token-count
        RPC can consume this stream's envelopes off the shared response
        queue. Released when this generator is exhausted, errors, or is
        closed - all on the single producer thread that drives it, so the
        non-reentrant Lock is always released on the thread that took it."""
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
                            result = self._resp_q.get(timeout=_POLL_INTERVAL)
                        except _queue.Empty:
                            if not self.is_alive():
                                raise RuntimeError(
                                    f"Native inference fault (worker exit "
                                    f"{self._proc.exitcode}). The model has been "
                                    "unloaded and will reload on the next request. "
                                    "See the debug log for the native stack trace.")
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
                        raise RuntimeError(msg)
                    else:
                        raise RuntimeError(f"Unexpected response during generation: {result!r}")
            except GeneratorExit:
                # Kill-based only - see module docstring for why no
                # cooperative drain is attempted first.
                self.shutdown(grace=0)
                raise

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
                            f"{self._proc.exitcode}) while handling '{name}'.")
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
