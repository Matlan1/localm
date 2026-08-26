# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speech-to-text for the GUI chat: Whisper via faster-whisper (CPU int8).

faster-whisper runs on CPU with int8 quantization and ships as plain wheels, so
no torch is required and it works on the GGUF-only base install. It is
optional:

    pip install "localm[voice]"

The model (config ``voice_stt_model``, default "base") is downloaded from
HuggingFace on FIRST use into localm's OWN data dir (``stt_cache_dir()``), not
the global HuggingFace cache in the user's home profile - that one download is
the only network access; transcription itself is fully local and offline.

That one download is gated by the network policy (netpolicy.network_mode), the
same bypass-ask-respect-off rule the embedder's ``_download_known`` applies:
under ``net_mode=allow`` the first use fetches automatically; under ``ask`` the
fetch needs an explicit one-time authorization (``prefetch_stt_model``); under
``off`` nothing fetches, unless ``net_allow_model_downloads`` exempts an
explicit fetch - off is still the default kill switch. The worker enforces
this structurally: it is dispatched with ``local_files_only`` so it CANNOT
download unless the policy decision in this (parent) process said so - and a
cached model always loads with ``local_files_only=True``, so a routine load
never touches the network at all.

Text-to-speech needs no backend at all: the GUI uses Kokoro in the browser
(see the ``tts`` plugin) or the browser's built-in speechSynthesis.

TRANSCRIPTION RUNS IN A SEPARATE PROCESS
----------------------------------------
faster-whisper's native engine (ctranslate2) and the PyAV decoder are C/C++
libraries that, on some inputs and some builds, fault at the C level (an
abort()/access-violation) rather than raising a Python exception. A native
fault cannot be caught by ``try/except``: it terminates the whole interpreter.

So the entire native pipeline (decode -> load -> transcribe) runs in a
dedicated worker process. If it crashes or hangs, only the worker dies; the
server process detects the dead/stuck worker, returns a clean ``VoiceError``,
and respawns the worker for the next request.
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

# A transcription exceeding this many seconds is treated as a hung native call:
# the worker is killed and a VoiceError returned. Overridable per install via the
# ``voice_stt_timeout_s`` config key.
_WORKER_TIMEOUT = 120.0

# Fault-injection hook, honoured by the worker only when this environment
# variable is set. Values: "abort" (the default), "exit" (a hard process exit),
# or "hang" (a wedged native call).
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

    Three gates, each with its own honest reason (a blocked prerequisite is
    NEVER silent): the faster-whisper package must be importable, and when the
    configured model is not cached yet the network policy must permit the
    one-time download - automatic only under ``net_mode=allow``; ``ask`` needs
    the explicit download action (``prefetch_stt_model`` via
    POST /api/voice/model/download) and ``off`` blocks absolutely.

    Uses ``find_spec`` (does NOT import the package), so a status check never
    loads the native STT stack into the server process. That stack initialises
    OpenMP, which can abort the whole process on the OMP-runtime collision
    described in the module docstring; it must only ever load in the isolated
    worker. Keeping this import-free (netpolicy/config are stdlib-light; no
    faster-whisper, no huggingface_hub) also keeps the status probe instant."""
    if importlib.util.find_spec("faster_whisper") is None:
        return False, (
            "Speech-to-text needs the faster-whisper package. Install it "
            "with: pip install \"localm[voice]\"  (then restart the server)")
    cached, name = stt_model_cached()
    if cached:
        return True, ""
    from localm.netpolicy import network_mode
    mode = network_mode()
    if mode == "allow":
        return True, ""                          # first use downloads automatically
    return False, _stt_download_blocked_reason(name, mode)


def _stt_download_blocked_reason(name: str, mode: str) -> str:
    """Why the one-time Whisper download cannot happen right now, stated for
    the user. ONE helper shared by ``stt_available``, the transcribe path and
    ``prefetch_stt_model``, so every surface reports the same reason."""
    if mode == "off":
        return (
            f"The Whisper '{name}' speech model is not downloaded, and network "
            "access is disabled (net_mode=off). Enable network access to fetch "
            "it once; transcription itself runs fully offline.")
    return (
        f"The Whisper '{name}' speech model is not downloaded yet, and "
        f"net_mode={mode} does not download automatically. Use the one-time "
        "download action (or set net_mode=allow); transcription itself runs "
        "fully offline afterwards.")


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


def _stt_repo_for(name: str) -> str:
    """The HuggingFace repo a ``voice_stt_model`` value maps to. ONE helper
    shared by the ``stt_model_cached`` probe and ``prefetch_stt_model``'s
    download, so the two cannot drift: the prefetch fetches exactly the repo
    the probe checks, and a successful prefetch therefore always flips the
    probe to cached (the same one-source-of-truth argument as ``stt_cache_dir``).

    Standard faster-whisper repos follow the Systran/faster-whisper-<name>
    convention - exact for every value the ``voice_stt_model`` SELECT can hold
    (tiny/base/small/medium; the settings schema validates SELECT values against
    their options) - and an explicit "<org>/<repo>" id is used as-is. A few
    exotic aliases (distil-*/turbo), reachable only by hand-editing config.json
    past the schema, map elsewhere upstream and would show as not-cached here;
    the worker stays the source of truth for whether the model actually loads."""
    return name if "/" in name else f"Systran/faster-whisper-{name}"


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
    repo = _stt_repo_for(name)
    # Hub cache layout: <cache>/models--<org>--<repo>/snapshots/<rev>/model.bin.
    # The snapshot entry may be a symlink or a real file; glob matches both.
    try:
        repo_dir = stt_cache_dir() / ("models--" + repo.replace("/", "--"))
        snaps = repo_dir / "snapshots"
        cached = snaps.is_dir() and any(snaps.glob("*/model.bin"))
        return bool(cached), name
    except Exception:
        return False, name


# Name of the background thread the voice plugin's on_install hook runs
# prefetch_stt_model on, so a caller can find and join that exact thread.
PREFETCH_THREAD_NAME = "localm-voice-prefetch"

# The file set the explicit prefetch pulls, matching faster_whisper's own
# download_model. The worker's lazy download under net_mode=allow still fetches
# anything else it needs.
_STT_ALLOW_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin",
                       "tokenizer.json", "vocabulary.*")


def prefetch_stt_model(allow_download: Optional[bool] = None) -> tuple[bool, str]:
    """Fetch the configured Whisper model into localm's own cache, gated by the
    network policy. Returns ``(ok, reason)``: ok True when the model is present
    afterwards (already cached counts), reason says why when it is not.

    The policy rule is the bypass-ask-respect-off one the embedder's
    ``_download_known`` applies, with the same parameter shape:

    * ``allow_download=None`` follows net_mode - automatic fetch only under
      "allow";
    * ``allow_download=True`` is an EXPLICIT, ONE-CALL authorization that
      bypasses net_mode="ask". The caller has collected the user's consent (the
      GUI's download action, gated on config:write, or installing the voice
      plugin - itself an explicit user action);
    * net_mode="off" refuses, even with ``allow_download=True``, UNLESS
      ``netpolicy.downloads_allowed_when_off()`` says the owner exempted
      explicit downloads (settings_schema.py's ``net_allow_model_downloads``,
      admin_only, default False - so off is still the default kill switch).

    The authorization is nothing but this function argument - not a config key,
    not module state, not an env var - so a one-time grant dies with this call
    and cannot persist into any other net_mode-gated path.

    Downloads via huggingface_hub directly rather than through the STT worker,
    so it needs no native code and works at plugin-install time BEFORE the
    faster-whisper pip extra is installed, and never holds the worker lock for
    the length of a download. The repo and cache root come from the same helpers
    the cached-probe reads (``_stt_repo_for`` / ``stt_cache_dir``), so a
    successful prefetch flips ``stt_model_cached()`` to True.

    NOT import-free (huggingface_hub is a heavy import): call from a job or a
    background thread, never from an async status handler."""
    from localm.debuglog import logger
    cached, name = stt_model_cached()
    if cached:
        return True, ""
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if allow_download is None:
        allow_download = network_mode() == "allow"
    if not allow_download:
        # Expected states (an unset policy, an offline box), not defects: INFO.
        reason = _stt_download_blocked_reason(name, network_mode())
        logger.info(reason)
        return False, reason
    if network_mode() == "off" and not downloads_allowed_when_off():
        reason = _stt_download_blocked_reason(name, "off")
        logger.info(reason)
        return False, reason
    repo = _stt_repo_for(name)
    root = stt_cache_dir()
    try:
        from huggingface_hub import snapshot_download
        root.mkdir(parents=True, exist_ok=True)
        logger.info("downloading Whisper STT model %s (one-time)...", repo)
        snapshot_download(repo, cache_dir=str(root),
                          allow_patterns=list(_STT_ALLOW_PATTERNS))
    except Exception as e:
        reason = f"Whisper model '{name}' download failed ({e})"
        logger.warning(reason)
        return False, reason
    # Verify the fetch produced a loadable snapshot: a repo that is not a
    # faster-whisper (CTranslate2) conversion has no model.bin, and
    # snapshot_download still succeeds.
    cached, _ = stt_model_cached()
    if not cached:
        reason = (f"Whisper model '{name}' was fetched but no model.bin "
                  "snapshot is present - it does not look like a faster-whisper "
                  "(CTranslate2) model repo.")
        logger.warning(reason)
        return False, reason
    logger.info("Whisper STT model '%s' is cached; transcription now runs "
                "fully offline.", name)
    return True, ""


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
    # default ("abort"): an uncatchable native abort, the same shape as a
    # ctranslate2 std::terminate. The process dies; no Python except intercepts it.
    os.abort()


def _worker_main(req_q, resp_q) -> None:
    """Long-lived child: decode + load + transcribe, one request at a time.

    Returns a structured envelope on the response queue so we never depend on
    pickling native exception objects across the process boundary:
        ("ok", text)            - success
        ("error", tag, detail)  - a clean, expected failure (tags below)
    A native crash/hang produces NO envelope; the parent detects the dead/stuck
    worker instead."""
    # faster-whisper pulls in two OpenMP runtimes (ctranslate2 -> Intel
    # libiomp5md, onnxruntime -> LLVM libomp140), and both initialising in one
    # process makes OpenMP abort(). KMP_DUPLICATE_LIB_OK lets the second load.
    # Set before any faster-whisper import so OpenMP reads it at init.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _silence_native_crash_dialogs()

    from localm._mp_spawn import (ignore_interrupt_signals,
                                   install_parent_death_watchdog)
    install_parent_death_watchdog()   # die with the parent even on a hard kill
                                       # (End Task / force-close); daemon=True is
                                       # atexit-gated and does not cover that, so
                                       # else this STT worker outlives the server
                                       # holding its model in VRAM.
    ignore_interrupt_signals()        # a console Ctrl+C reaches every process on
                                       # the console; the parent alone decides when
                                       # this worker stops.

    model = None
    model_name = None
    while True:
        msg = req_q.get()
        if msg is None:                          # shutdown sentinel
            return
        data, name, language, download_root, local_files_only = msg

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
                # download_root keeps the model inside localm's data dir rather
                # than the global HF cache. Both it and local_files_only are
                # resolved by the parent and sent with the request, so the child
                # cannot download past the parent's network-policy decision.
                model = WhisperModel(name, device="cpu", compute_type="int8",
                                     download_root=download_root,
                                     local_files_only=local_files_only)
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
    ensure_spawn_uses_venv_python()   # a renamed launcher would spawn WinError 2
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


def _run_in_worker(data: bytes, name: str, language, timeout: float, *,
                   local_files_only: bool = True,
                   blocked_reason: Optional[str] = None) -> str:
    """Send one transcription to the isolated worker and wait for its result.

    ``local_files_only`` is the caller's network-policy decision (default True:
    never download unless something explicitly said so - the fail-safe
    direction). ``blocked_reason`` carries the policy explanation to attach
    when a download WOULD have been needed: the worker's offline load then
    fails, and that failure is reported as the policy block it really is
    (code "download-blocked"), not as a mysterious load error.

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
            req_q.put((data, name, language, str(stt_cache_dir()),
                       bool(local_files_only)))
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
        if blocked_reason:
            # The load ran offline because the network policy refused the
            # download, so the policy is reported as the cause with the loader
            # detail kept.
            raise VoiceError(f"{blocked_reason} (offline load failed: {detail})",
                             code="download-blocked")
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

    # Availability check without importing the native lib, so a missing
    # dependency is reported before a worker spawn.
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

    # Network-policy gate for the one-time model download, decided in the parent;
    # the worker only executes the decision via local_files_only. A cached model
    # loads offline; a missing one downloads only under net_mode=allow.
    cached, _ = stt_model_cached()
    local_files_only = True
    blocked_reason = None
    if not cached:
        from localm.netpolicy import network_mode
        mode = network_mode()
        if mode == "allow":
            local_files_only = False
        else:
            blocked_reason = _stt_download_blocked_reason(name, mode)

    return _run_in_worker(data, name, lang, timeout,
                          local_files_only=local_files_only,
                          blocked_reason=blocked_reason)


def shutdown_stt() -> None:
    """Stop the worker process (best effort). Safe to call when none is running."""
    with _mgr_lock:
        _kill_worker()


# A daemon worker already dies with the parent, but tear it down explicitly at
# interpreter exit so its queue feeder threads never delay shutdown.
atexit.register(shutdown_stt)
