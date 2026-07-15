# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speech-to-text for the GUI chat: Whisper via faster-whisper (CPU int8).

faster-whisper was chosen over openai-whisper/torch because it runs well on
CPU with int8 quantization and ships as plain wheels - no torch required, so
it works on the GGUF-only base install. It is still optional:

    pip install "localm[voice]"

The model (config ``voice_stt_model``, default "base") is downloaded from
HuggingFace on FIRST use into localm's OWN data dir (``stt_cache_dir()``), not
the global HuggingFace cache in the user's home profile - that one download is
the only network access; transcription itself is fully local and offline.

Text-to-speech needs no backend at all: the GUI uses Kokoro in the browser
(see the ``tts`` plugin) or the browser's built-in speechSynthesis.

VOICE-1 - why transcription runs in a SEPARATE PROCESS
------------------------------------------------------
faster-whisper's native engine (ctranslate2) and the PyAV decoder are C/C++
libraries. On some inputs and some builds they do not raise a Python exception
- they fault at the C level (an abort()/access-violation). A native fault
cannot be caught by ``try/except``: it terminates the whole interpreter, which
took the entire localm server down with it (the reported "speech to text
crashed the server"). An empty/undecodable-blob guard does not help, because
the fault is in the real transcription path, not in obviously-bad input.

The only robust containment for an uncatchable native fault is process
isolation. So the entire native pipeline (decode -> load -> transcribe) runs in
a dedicated worker process. If it crashes or hangs, only the worker dies; the
server process detects the dead/stuck worker, returns a clean ``VoiceError``,
and respawns the worker for the next request. STT can therefore never take the
server down.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import queue as _queue
import threading
import time
from typing import Optional

import multiprocessing as mp

# A single transcription that exceeds this many seconds is treated as a hung
# native call: the worker is killed and a clean VoiceError is returned, so a
# wedged native library can never block STT forever. Overridable per install via
# the ``voice_stt_timeout_s`` config key.
_WORKER_TIMEOUT = 120.0

# Fault-injection hook, honoured by the worker ONLY when this environment
# variable is set. It exists exclusively so the test-suite can prove the
# crash-containment property with a REAL native fault; it is never set in
# production. Values: "abort" (a genuine uncatchable native abort, the default),
# "exit" (a hard process exit), or "hang" (a wedged native call).
_FAULT_ENV = "LOCALM_VOICE_FAULT_FOR_TEST"


class VoiceError(Exception):
    """Transcription failed; the message says why and what to install.

    ``code`` carries the failure CLASS as a stable machine-readable tag (the
    worker's own structured tags - "needs-faster-whisper", "decode", "empty",
    "load", "transcribe" - plus the manager-side "spawn", "timeout", "crash",
    "no-speech"), so an HTTP endpoint can pick a status by class instead of
    substring-matching the human message: rewording a message must never flip
    a status code."""

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def stt_available() -> tuple[bool, str]:
    """(available, reason) - lets the GUI grey out the mic button up front
    instead of letting the user record and only then failing.

    Uses ``find_spec`` (does NOT import the package), so a status check never
    loads the native STT stack into the server process. That stack initialises
    OpenMP, which can abort the whole process on the OMP-runtime collision
    described in the module docstring; it must only ever load in the isolated
    worker. Keeping this import-free also makes the status probe instant."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return True, ""
    return False, (
        "Speech-to-text needs the faster-whisper package. Install it "
        "with: pip install \"localm[voice]\"  (then restart the server)")


def stt_cache_dir():
    """Where localm keeps the Whisper STT model: inside the data dir (rule 4).

    THE single source of truth for the model's location, used by BOTH the
    ``stt_model_cached`` probe below and the worker's actual download
    (``WhisperModel(download_root=...)``). One function, so the probe cannot drift
    from where the bytes really land - a disagreement would show as a consent prompt
    for a model that is already there, or vice versa.

    Left to itself, faster-whisper downloads into the GLOBAL HuggingFace hub cache
    (``~/.cache/huggingface/hub``, i.e. the user's home profile - up to ~1.5 GB for the
    larger models). The user consents to the DOWNLOAD; they never consented to that
    LOCATION. faster-whisper passes ``download_root`` straight through as
    huggingface_hub's ``cache_dir``, so the on-disk layout is the standard
    ``models--<org>--<repo>/snapshots/<rev>/`` either way - only the root moves."""
    from localm.config import cache_dir
    return cache_dir() / "whisper"


def stt_model_cached() -> tuple[bool, str]:
    """(cached, model_name) - is the configured Whisper model already in localm's
    own model cache (``stt_cache_dir()``)? First use otherwise downloads it; the GUI
    asks for consent before triggering that one network access.

    Resolved WITHOUT importing huggingface_hub (or faster-whisper): we read the
    documented hub-cache layout directly. The earlier version imported
    ``huggingface_hub`` here, whose heavy transitive import chain (requests,
    fsspec, filelock, tqdm, ...) could take tens of seconds on a cold start and,
    because this runs inside the ``async def`` /api/voice/status handler, froze
    the event loop and stalled the whole first /api/* batch (R24). A direct path
    probe keeps it genuinely instant, matching ``stt_available``."""
    from pathlib import Path

    from localm.config import load_config
    name = str(load_config().get("voice_stt_model", "base"))
    if Path(name).expanduser().is_dir():
        return True, name                       # local model directory
    # Standard faster-whisper repos follow the Systran/faster-whisper-<name>
    # convention; an explicit "<org>/<repo>" id is used as-is. A few exotic ids
    # (distil/turbo) map elsewhere and would simply show as not-cached - one
    # harmless extra consent prompt, never a crash. The worker is the source of
    # truth for whether the model actually loads.
    repo = name if "/" in name else f"Systran/faster-whisper-{name}"
    # Hub cache layout: <cache>/models--<org>--<repo>/snapshots/<rev>/model.bin
    # (the snapshot entry is a symlink or, in no-symlink mode, a real file -
    # glob matches both). A false negative only costs one harmless consent
    # prompt; the worker downloads on demand regardless.
    try:
        repo_dir = stt_cache_dir() / ("models--" + repo.replace("/", "--"))
        snaps = repo_dir / "snapshots"
        cached = snaps.is_dir() and any(snaps.glob("*/model.bin"))
        return bool(cached), name
    except Exception:
        return False, name


# --------------------------------------------------------------------------- #
# Worker side - everything below runs in the isolated child process.
# --------------------------------------------------------------------------- #

def _silence_native_crash_dialogs() -> None:
    """Windows: stop a native fault in this worker from popping a Windows Error
    Reporting dialog. Without this, a real ctranslate2/PyAV access-violation
    could block on a "program stopped working" dialog instead of dying fast; we
    want the worker to terminate immediately so the parent detects it and
    recovers. No-op off Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        # SEM_FAILCRITICALERRORS(0x1) | SEM_NOGPFAULTERRORBOX(0x2)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    except Exception:
        pass


def _simulate_fault(mode: str) -> None:
    """Test-only: reproduce a genuine uncatchable native fault on demand.

    Used solely by the crash-containment test (gated behind ``_FAULT_ENV``,
    never set in production) to prove a real native fault in the STT worker is
    contained instead of taking the server down."""
    if mode == "hang":
        while True:                              # a wedged native call
            time.sleep(3600)
    if mode == "exit":
        os._exit(134)                            # vanish with no Python traceback
    # default ("abort"): a genuine uncatchable native abort - the same shape as
    # a ctranslate2 std::terminate. No Python ``except`` can intercept it; the
    # process dies (worker WER/abort dialogs are already suppressed at startup).
    os.abort()


def _worker_main(req_q, resp_q) -> None:
    """Long-lived child: decode + load + transcribe, one request at a time.

    Returns a structured envelope on the response queue so we never depend on
    pickling native exception objects across the process boundary:
        ("ok", text)            - success
        ("error", tag, detail)  - a clean, expected failure (tags below)
    A native crash/hang produces NO envelope; the parent detects the dead/stuck
    worker instead."""
    # ROOT CAUSE of the reported crash: faster-whisper pulls in two OpenMP
    # runtimes (ctranslate2 -> Intel libiomp5md, onnxruntime -> LLVM libomp140).
    # When both initialise in one process, OpenMP raises "OMP: Error #15" and
    # calls abort() - an uncatchable native fault that, in the old in-process
    # design, took the whole server down. KMP_DUPLICATE_LIB_OK lets the second
    # runtime load instead of aborting. It is the documented workaround and is
    # safe to scope to this worker, whose only job is transcription; isolation
    # remains the safety net for any OTHER native fault. Set before any
    # faster-whisper import so OpenMP reads it at init.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _silence_native_crash_dialogs()
    model = None
    model_name = None
    while True:
        msg = req_q.get()
        if msg is None:                          # shutdown sentinel
            return
        data, name, language, download_root = msg

        fault = os.environ.get(_FAULT_ENV)
        if fault:
            _simulate_fault(fault)               # test-only; never returns clean

        try:
            from faster_whisper import WhisperModel
            from faster_whisper.audio import decode_audio
        except Exception as e:                   # native lib missing / failed to load
            resp_q.put(("error", "needs-faster-whisper", str(e)))
            continue

        import io
        try:
            audio = decode_audio(io.BytesIO(data))
        except Exception as e:
            resp_q.put(("error", "decode", str(e)))
            continue
        if audio is None or len(audio) == 0:
            resp_q.put(("error", "empty", ""))
            continue

        try:
            if model is None or model_name != name:
                # download_root keeps the model inside localm's data dir instead of the
                # global HF cache in the user's home profile (rule 4). Resolved by the
                # PARENT and sent with the request, not recomputed here: the parent's
                # stt_model_cached() probe and this download then read the same value by
                # construction, with no dependence on env surviving the spawn.
                model = WhisperModel(name, device="cpu", compute_type="int8",
                                     download_root=download_root)
                model_name = name
        except Exception as e:
            model = None
            resp_q.put(("error", "load", str(e)))
            continue

        try:
            segments, _info = model.transcribe(audio, language=language)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            resp_q.put(("error", "transcribe", str(e)))
            continue

        resp_q.put(("ok", text))


# --------------------------------------------------------------------------- #
# Parent side - worker lifecycle + dispatch.
# --------------------------------------------------------------------------- #

_mgr_lock = threading.Lock()
_proc: Optional[mp.process.BaseProcess] = None
_req_q = None
_resp_q = None


def _spawn_worker() -> None:
    """Start a fresh worker. Caller holds ``_mgr_lock``."""
    global _proc, _req_q, _resp_q
    from localm._mp_spawn import ensure_spawn_uses_venv_python
    ensure_spawn_uses_venv_python()   # #617: avoid a renamed-launcher WinError 2
    ctx = mp.get_context("spawn")               # explicit: identical on every OS
    _req_q = ctx.Queue()
    _resp_q = ctx.Queue()
    _proc = ctx.Process(
        target=_worker_main, args=(_req_q, _resp_q),
        name="localm-stt-worker", daemon=True)
    _proc.start()


def _kill_worker() -> None:
    """Terminate the worker and drop its queues. Caller holds ``_mgr_lock``."""
    global _proc, _req_q, _resp_q
    p = _proc
    if p is not None:
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.join(timeout=5)
        except Exception:
            pass
    for q in (_req_q, _resp_q):
        if q is not None:
            try:
                q.close()
                q.cancel_join_thread()           # do not let a feeder thread block exit
            except Exception:
                pass
    _proc = None
    _req_q = None
    _resp_q = None


def _ensure_worker() -> None:
    """Spawn the worker if it is absent or has died. Caller holds ``_mgr_lock``."""
    if _proc is None or not _proc.is_alive():
        if _proc is not None:                    # reap a dead handle + its queues
            _kill_worker()
        _spawn_worker()


def _run_in_worker(data: bytes, name: str, language, timeout: float) -> str:
    """Send one transcription to the isolated worker and wait for its result.

    Raises ``VoiceError`` for every failure mode - a clean worker error, a
    native crash (worker died), or a hang (timeout) - and always leaves a
    healthy worker (or no worker) behind for the next call."""
    with _mgr_lock:
        try:
            _ensure_worker()                     # spawn failures must not escape raw
        except Exception as e:
            _kill_worker()
            raise VoiceError(f"Could not start the speech-to-text worker: {e}",
                             code="spawn")
        proc, req_q, resp_q = _proc, _req_q, _resp_q
        try:
            req_q.put((data, name, language, str(stt_cache_dir())))
        except Exception as e:
            _kill_worker()
            raise VoiceError(f"Could not dispatch transcription to the STT worker: {e}",
                             code="spawn")

        deadline = time.monotonic() + timeout
        result = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_worker()                   # kill the hung worker
                raise VoiceError(
                    "Transcription timed out: the speech-to-text engine stopped "
                    "responding and was restarted. The server stayed up - please "
                    "try again, or use a smaller model (localm config "
                    "voice_stt_model tiny).", code="timeout")
            try:
                result = resp_q.get(timeout=min(0.5, remaining))
                break
            except _queue.Empty:
                if not proc.is_alive():          # native crash: worker vanished
                    code = proc.exitcode
                    _kill_worker()
                    raise VoiceError(
                        f"The speech-to-text engine crashed (worker exit {code}) "
                        "on this recording. The server stayed up and STT was "
                        "restarted - please try again.", code="crash")
                continue

    kind = result[0]
    if kind == "ok":
        text = result[1]
        if not text:
            raise VoiceError("No speech detected in the recording", code="no-speech")
        return text

    tag = result[1]
    detail = result[2] if len(result) > 2 else ""
    if tag == "needs-faster-whisper":
        raise VoiceError(
            "Speech-to-text needs the faster-whisper package. Install it with: "
            f"pip install \"localm[voice]\"  (worker import failed: {detail})",
            code=tag)
    if tag == "decode":
        raise VoiceError(
            f"Could not decode the recording (corrupt or unsupported audio): {detail}",
            code=tag)
    if tag == "empty":
        raise VoiceError("No audio in the recording (it was empty or zero-length).",
                         code=tag)
    if tag == "load":
        raise VoiceError(
            f"Could not load Whisper model '{name}': {detail}. The first use "
            "downloads it from HuggingFace - check the network, or set a "
            "different model: localm config voice_stt_model tiny", code=tag)
    if tag == "transcribe":
        raise VoiceError(f"Transcription failed: {detail}", code=tag)
    raise VoiceError("Transcription failed.", code="transcribe")


def transcribe_bytes(data: bytes, language: Optional[str] = None) -> str:
    """Transcribe an audio blob (webm/ogg/wav/mp3 - anything PyAV decodes).

    The native pipeline (decode -> load -> transcribe) runs in an isolated
    worker process; see the module docstring. The only work done in the server
    process is the pure-Python guards below, so no native code path can fault
    the server."""
    # Reject an empty recording up front (pure Python, no native, no worker
    # spawn): a 0-byte blob (a mic that captured nothing) is a clean client error.
    if not data:
        raise VoiceError("Empty recording (no audio was captured).", code="empty")

    # Cheap availability check without importing the native lib (find_spec does
    # not load ctranslate2), so a missing dependency is reported instantly
    # rather than after a worker spawn.
    if importlib.util.find_spec("faster_whisper") is None:
        raise VoiceError(
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"", code="needs-faster-whisper")

    from localm.config import load_config
    cfg = load_config()
    name = str(cfg.get("voice_stt_model", "base"))
    lang = language or cfg.get("voice_stt_language") or None
    try:
        timeout = float(cfg.get("voice_stt_timeout_s") or _WORKER_TIMEOUT)
    except (TypeError, ValueError):
        timeout = _WORKER_TIMEOUT

    return _run_in_worker(data, name, lang, timeout)


def shutdown_stt() -> None:
    """Stop the worker process (best effort). Safe to call when none is running."""
    with _mgr_lock:
        _kill_worker()


# A daemon worker already dies with the parent, but tear it down explicitly at
# interpreter exit so its queue feeder threads never delay shutdown.
atexit.register(shutdown_stt)
