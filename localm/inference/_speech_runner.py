# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subprocess isolation for speech synthesis with a text-to-speech GGUF.

The model load and every generation stage run native llama.cpp / libmtmd code
that can ``abort()`` the whole process on a driver failure, so they run in a
long-lived spawn child that owns one :class:`SpeechSynthesizer`. A crash or a
hang costs the worker, never the server.

Protocol (two ``multiprocessing.Queue``s plus one ``multiprocessing.Event``):

``req_q`` (parent -> child), one command at a time:
    ("load", {model_path, mmproj_path, n_gpu_layers, n_ctx, n_threads, main_gpu})
    ("speak", {text, language, reference, seed}); reference is mono float32
                                                   little-endian samples or None
    ("shutdown", None)

``resp_q`` (child -> parent):
    ("progress", frames)     - during "speak", at most every PROGRESS_INTERVAL s
    ("ok", value)            - load: {pipeline, sample_rate, encoder_sample_rate,
                               projector_on_gpu}; speak: {wav, sample_rate,
                               n_samples, frames, seed}
    ("error", message, tag)  - a clean failure; *tag* names the exception class
                               the parent re-raises

``cancel`` (an Event): set by the parent to stop the "speak" in progress at the
next frame boundary; the child answers ("error", ..., "SpeechCancelled"), keeps
its model loaded and serves the next command. The parent clears it before each
"speak".
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
from typing import Callable, Optional

_FAULT_ENV = "LOCALM_SPEECH_FAULT_FOR_TEST"

PROGRESS_INTERVAL = 0.25

# Exception classes whose name travels as an error tag and is re-raised by type.
_TAGGED = ("SpeechInputError", "SpeechCancelled", "SpeechBudgetExceeded",
           "SpeechUnavailable", "SpeechStageFailed", "PretokenizerUnsafeInputError")


def _simulate_fault(mode: str) -> None:
    if mode == "hang":
        while True:
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)
    os.abort()


_crash_trace_fh = None


def _arm_native_crash_trace(path) -> None:
    """Child side: point faulthandler at *path* so a death by a native signal
    leaves a trace the parent relays. Failures are logged, never raised."""
    global _crash_trace_fh
    if path is None:
        return
    import faulthandler

    from localm.debuglog import logger
    try:
        _crash_trace_fh = open(path, "w", encoding="utf-8")
        faulthandler.enable(file=_crash_trace_fh, all_threads=True)
        if not faulthandler.is_enabled():
            logger.warning("speech worker: faulthandler is not enabled; a native "
                           "fault in this worker will produce no stack trace")
    except Exception as e:   # noqa: BLE001 - a diagnostic must never break the worker
        logger.warning("speech worker: could not arm the native-fault trace (%s: %s); "
                       "a native fault in this worker will produce no stack trace",
                       type(e).__name__, e)


def _runner_main(req_q, resp_q, cancel, crash_trace_path=None) -> None:
    """Child: owns one SpeechSynthesizer for its lifetime and serves commands in
    order."""
    _arm_native_crash_trace(crash_trace_path)
    from localm.debuglog import attach_child_logging
    attach_child_logging()
    from localm._mp_spawn import (ignore_interrupt_signals,
                                   install_parent_death_watchdog,
                                   suppress_native_error_dialogs)
    install_parent_death_watchdog()
    ignore_interrupt_signals()
    suppress_native_error_dialogs()

    synth = None
    while True:
        cmd = req_q.get()
        if cmd is None:
            return
        name = cmd[0]
        payload = cmd[1] if len(cmd) > 1 else None

        fault = os.environ.get(_FAULT_ENV)
        if fault and (name != "speak" or fault.startswith("speak-")):
            _simulate_fault(fault.removeprefix("speak-"))

        if name == "shutdown":
            if synth is not None:
                synth.close()
            return

        if name == "load":
            params = dict(payload or {})
            if params.pop("cpu_only", False):
                os.environ["HIP_VISIBLE_DEVICES"] = "-1"
                os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                from localm.inference.backends.llamacpp.mtmd_gen import SpeechSynthesizer
                synth = SpeechSynthesizer(**params)
                resp_q.put(("ok", {
                    "pipeline": synth.pipeline,
                    "sample_rate": synth.sample_rate,
                    "encoder_sample_rate": synth.encoder_sample_rate,
                    "projector_on_gpu": synth.projector_on_gpu,
                }))
            except Exception as e:
                resp_q.put(("error", str(e), _tag(e)))
            continue

        if name == "speak":
            if synth is None:
                resp_q.put(("error", "the speech worker got 'speak' before a model "
                                     "was loaded", None))
                continue
            try:
                result = synth.synthesize(
                    payload["text"], language=payload.get("language"),
                    reference=payload.get("reference"), seed=payload.get("seed"),
                    on_progress=_ProgressSender(resp_q), should_stop=cancel.is_set)
                resp_q.put(("ok", {
                    "wav": result.wav, "sample_rate": result.sample_rate,
                    "n_samples": result.n_samples, "frames": result.frames,
                    "seed": result.seed}))
            except Exception as e:
                resp_q.put(("error", str(e), _tag(e)))
            continue

        resp_q.put(("error", f"unknown speech-runner command: {name!r}", None))


class _ProgressSender:
    """Sends ("progress", frames) for the first frame and then at most every
    PROGRESS_INTERVAL seconds."""

    def __init__(self, resp_q) -> None:
        self._resp_q = resp_q
        self._last = 0.0

    def __call__(self, frames: int) -> None:
        now = time.monotonic()
        if frames == 1 or now - self._last >= PROGRESS_INTERVAL:
            self._last = now
            self._resp_q.put(("progress", frames))


def _tag(e: BaseException) -> Optional[str]:
    name = type(e).__name__
    return name if name in _TAGGED else None


def _raise_tagged(message: str, tag: Optional[str]) -> None:
    from localm.inference.backends.base import PretokenizerUnsafeInputError
    from localm.inference.backends.llamacpp import mtmd_gen as g
    classes = {
        "SpeechInputError": g.SpeechInputError,
        "SpeechCancelled": g.SpeechCancelled,
        "SpeechBudgetExceeded": g.SpeechBudgetExceeded,
        "SpeechUnavailable": g.SpeechUnavailable,
        "SpeechStageFailed": g.SpeechStageFailed,
        "PretokenizerUnsafeInputError": PretokenizerUnsafeInputError,
    }
    cls = classes.get(tag or "")
    if cls is g.SpeechCancelled:
        raise g.SpeechCancelled(message)
    if cls is not None:
        raise cls(message)
    raise RuntimeError(message)


class SpeechWorkerHung(RuntimeError):
    """The speech worker stopped making progress and was killed."""


_POLL_INTERVAL = 0.2
LOAD_TIMEOUT_DEFAULT = 600.0
# Longest a "speak" may go without any message from the worker before it is
# treated as hung. Progress arrives at least every frame, so this bounds the
# prompt and the slowest single frame, not the whole synthesis.
SPEAK_STALL_TIMEOUT = 300.0
# How long a cancelled "speak" may take to stop before the worker is killed.
CANCEL_GRACE = 30.0


class SpeechRunner:
    """Parent-side handle to one isolated speech worker process. One RPC at a
    time: the caller serialises :meth:`speak` calls."""

    def __init__(self) -> None:
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._cancel = None
        self._crash_trace_path = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _exit_reason(self) -> str:
        from localm._mp_spawn import describe_exit_code
        proc = self._proc
        return describe_exit_code(None if proc is None else proc.exitcode)

    def _crash_detail(self) -> str:
        path = self._crash_trace_path
        trace = ""
        if path is not None:
            try:
                trace = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                trace = ""
            finally:
                self._discard_trace()
        if not trace:
            return " No native stack trace was captured for this fault."
        from localm.debuglog import logger, native_fault_hint
        logger.error("speech worker native fault trace:\n%s", trace)
        return f" Native fault: {trace.splitlines()[0].strip()} ({native_fault_hint()})."

    def _discard_trace(self) -> None:
        path = self._crash_trace_path
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _spawn(self) -> None:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._cancel = ctx.Event()
        self._discard_trace()
        from localm.audit import diagnostics_allowed
        self._crash_trace_path = None
        if diagnostics_allowed():
            from localm.debuglog import child_crash_trace_path, logger
            try:
                self._crash_trace_path = child_crash_trace_path("speech-worker")
            except OSError as e:
                logger.warning("could not allocate a native-fault trace file (%s); "
                               "a native fault in the speech worker will not be "
                               "traced", e)
        self._proc = ctx.Process(
            target=_runner_main,
            args=(self._req_q, self._resp_q, self._cancel, self._crash_trace_path),
            name="localm-speech-worker", daemon=True)
        self._proc.start()

    def spawn_and_load(self, params: dict, timeout: float = LOAD_TIMEOUT_DEFAULT) -> dict:
        """Spawn the worker and load the model. Returns the load facts. Raises
        the tagged speech error or RuntimeError (the worker is stopped first)."""
        self._spawn()
        self._req_q.put(("load", params))
        deadline = time.monotonic() + timeout
        while True:
            try:
                msg = self._resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty as e:
                if not self._proc.is_alive():
                    detail = self._crash_detail()
                    reason = self._exit_reason()
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"The speech worker crashed (exit code {reason}) while "
                        f"loading the model. The server stayed up.{detail}") from e
                if time.monotonic() > deadline:
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"Loading the speech model timed out after {timeout:.0f}s; "
                        "the worker was stopped.") from e
                continue
            if msg[0] == "ok":
                return msg[1]
            if msg[0] == "error":
                self.shutdown(grace=0)
                _raise_tagged(msg[1], msg[2] if len(msg) > 2 else None)

    def speak(self, payload: dict, *, on_progress: Optional[Callable[[int], None]] = None,
              should_cancel: Optional[Callable[[], bool]] = None,
              stall_timeout: float = SPEAK_STALL_TIMEOUT) -> dict:
        """Synthesize one request. *on_progress* receives frame counts;
        *should_cancel* is polled and, once True, stops the worker's synthesis
        (``SpeechCancelled`` is raised; the worker stays loaded unless it fails
        to stop within ``CANCEL_GRACE`` seconds, when it is killed).

        Raises the tagged speech errors for a clean failure, RuntimeError for a
        crash or a hang (the worker is gone afterwards)."""
        if self._req_q is None or not self.is_alive():
            raise RuntimeError("The speech worker is not running.")
        self._cancel.clear()
        self._req_q.put(("speak", payload))
        last_message = time.monotonic()
        cancel_deadline = None
        while True:
            try:
                msg = self._resp_q.get(timeout=_POLL_INTERVAL)
            except _queue.Empty as e:
                now = time.monotonic()
                if not self._proc.is_alive():
                    detail = self._crash_detail()
                    reason = self._exit_reason()
                    self.shutdown(grace=0)
                    raise RuntimeError(
                        f"The speech worker crashed (exit code {reason}) while "
                        f"speaking. The server stayed up.{detail}") from e
                if cancel_deadline is None and should_cancel is not None and should_cancel():
                    self._cancel.set()
                    cancel_deadline = now + CANCEL_GRACE
                if cancel_deadline is not None and now > cancel_deadline:
                    self.shutdown(grace=0)
                    from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled
                    raise SpeechCancelled(
                        "The speech synthesis was cancelled; the worker did not "
                        "stop in time and was restarted.") from e
                if cancel_deadline is None and now - last_message > stall_timeout:
                    self.shutdown(grace=0)
                    raise SpeechWorkerHung(
                        f"The speech worker made no progress for {stall_timeout:.0f}s "
                        "and was stopped; retry the request.") from e
                continue
            last_message = time.monotonic()
            kind = msg[0]
            if kind == "progress":
                if on_progress is not None:
                    on_progress(int(msg[1]))
                if cancel_deadline is None and should_cancel is not None and should_cancel():
                    self._cancel.set()
                    cancel_deadline = last_message + CANCEL_GRACE
                continue
            if kind == "ok":
                return msg[1]
            if kind == "error":
                _raise_tagged(msg[1], msg[2] if len(msg) > 2 else None)
            raise RuntimeError(f"Unexpected response from the speech worker: {kind!r}")

    def shutdown(self, grace: float = 5.0) -> None:
        """Stop the worker: politely within *grace* seconds, then by force. Safe
        to call more than once."""
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            if self._cancel is not None:
                self._cancel.set()
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
                    q.cancel_join_thread()
                except Exception:
                    pass
        self._proc = None
        self._req_q = None
        self._resp_q = None
        self._cancel = None
        self._discard_trace()
        self._crash_trace_path = None
