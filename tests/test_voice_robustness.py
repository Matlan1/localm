# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speech-to-text must never take the whole server down.

faster-whisper's native engine (ctranslate2) and PyAV can fault at the C level
on some inputs/builds - an abort()/access-violation that no Python ``try/except``
can catch and that terminates the whole interpreter. localm therefore runs the
native pipeline (decode -> load -> transcribe) in an isolated worker process; a
crash or hang there kills only the worker, and the server returns a clean
``VoiceError`` and respawns.

These tests prove the containment property with REAL, uncatchable faults (a hard
process exit, a genuine segfault, and a hang) injected into the worker via the
``LOCALM_VOICE_FAULT_FOR_TEST`` hook - the same code path a real ctranslate2
crash would take. The premise test below shows that, without isolation, such a
fault bypasses ``try/except`` entirely."""

import io
import math
import os
import struct
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

from localm import voice


def _has_faster_whisper() -> bool:
    try:
        import faster_whisper.audio  # noqa: F401
        return True
    except (ImportError, OSError):
        # The native lib can fail to load on Windows under load (documented
        # WinError 127 flake); skip rather than report that flake as a failure.
        return False


@pytest.fixture(autouse=True)
def _clean_worker():
    """Each test starts and ends with no STT worker and no fault env, so the
    next worker spawns fresh (and inherits only the env the test sets)."""
    os.environ.pop(voice._FAULT_ENV, None)
    voice.shutdown_stt()
    yield
    os.environ.pop(voice._FAULT_ENV, None)
    voice.shutdown_stt()


# --------------------------------------------------------------------------- #
# Pure-Python guards (hold even when faster-whisper is absent).
# --------------------------------------------------------------------------- #

def test_empty_audio_raises_clean_voiceerror():
    # Rejected up front, before any worker spawn or native call.
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"")
    assert "empty" in str(ei.value).lower()


def test_status_checks_do_not_load_native_stack():
    # The server process must never import faster-whisper/ctranslate2 just to
    # answer a status probe: that stack initialises OpenMP and can abort the whole
    # process. It must ALSO not import huggingface_hub: that heavy transitive
    # import (requests, fsspec, filelock, tqdm, ...) can take tens of seconds on a
    # cold start and, because /api/voice/status is an ``async def``, would freeze
    # the event loop and stall the whole first /api/* batch. Run in a clean
    # subprocess (so other tests that imported the stack do not pollute the check)
    # and load this worktree's voice.py by file path (so an editable install of
    # localm cannot shadow it with a different copy).
    from pathlib import Path
    worktree = Path(__file__).resolve().parents[1].as_posix()
    vpath = (Path(__file__).resolve().parents[1] / "localm" / "voice.py").as_posix()
    code = (
        "import sys, importlib.util as iu\n"
        f"sys.path.insert(0, {worktree!r})\n"
        f"spec = iu.spec_from_file_location('lvoice', {vpath!r})\n"
        "m = iu.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.stt_available(); m.stt_model_cached()\n"
        "bad = [x for x in ('faster_whisper', 'ctranslate2', 'huggingface_hub')"
        " if x in sys.modules]\n"
        "print('BAD=' + ','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("BAD=")), None)
    assert line is not None, f"probe did not run:\n{proc.stdout}\n{proc.stderr[-600:]}"
    assert line == "BAD=", f"status probe loaded a heavy/native module: {line}"


def test_stt_model_cached_resolves_hub_path_without_import(monkeypatch, tmp_path):
    # stt_model_cached reads the documented hub-cache layout directly (the same
    # models--<org>--<repo>/snapshots/<rev>/ layout download_root produces, since
    # faster-whisper passes it through as huggingface_hub's cache_dir). Lay down the
    # snapshot file the resolver expects inside localm's OWN cache dir; it must
    # report cached=True without importing huggingface_hub, and False when absent.
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "load_config", lambda: {"voice_stt_model": "base"})

    cached, name = voice.stt_model_cached()
    assert (cached, name) == (False, "base")     # empty cache -> not cached

    snap = (voice.stt_cache_dir() / "models--Systran--faster-whisper-base"
            / "snapshots" / "abc123")
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"\x00")
    cached, name = voice.stt_model_cached()
    assert (cached, name) == (True, "base")      # snapshot present -> cached


def test_stt_cache_dir_is_contained_in_the_data_dir(monkeypatch, tmp_path):
    """The Whisper model lands inside LOCALM_HOME, never the user's home profile.

    Left to itself faster-whisper caches into the global HF hub cache
    (~/.cache/huggingface/hub, up to ~1.5 GB) outside the data dir. An ambient
    HF_* env var must NOT be able to pull it back out. LOCALM_HOME is the knob,
    so the cache follows the data dir."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(k, str(tmp_path / "somewhere-else"))

    root = voice.stt_cache_dir()
    assert root == tmp_path / "cache" / "whisper"
    assert tmp_path in root.parents                      # inside the data dir
    if Path.home() not in tmp_path.parents:
        assert Path.home() not in root.parents               # NOT the home profile


def test_stt_request_carries_the_contained_download_root(monkeypatch, tmp_path):
    """The parent sends the resolved cache dir WITH each request, and the worker
    hands it to WhisperModel as download_root.

    These are two processes: a worker that recomputed the path itself could
    disagree with the parent's stt_model_cached() probe (a consent prompt for a
    model already on disk, or a re-download into a second location), so the value
    crosses the queue. Uses a fake queue, so no worker is spawned."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    sent = []

    class _FakeQ:
        def put(self, msg):
            sent.append(msg)

        def get(self, timeout=None):
            return ("ok", "hello")

    monkeypatch.setattr(voice, "_ensure_worker", lambda: None)
    monkeypatch.setattr(voice, "_proc", types.SimpleNamespace(is_alive=lambda: True))
    monkeypatch.setattr(voice, "_req_q", _FakeQ())
    monkeypatch.setattr(voice, "_resp_q", _FakeQ())

    assert voice._run_in_worker(b"audio", "base", None, 5.0) == "hello"
    assert len(sent) == 1
    data, name, language, download_root, local_files_only = sent[0]
    assert (data, name, language) == (b"audio", "base", None)
    assert download_root == str(tmp_path / "cache" / "whisper")
    # The parent's network-policy decision crosses the queue too, and the
    # DEFAULT is the fail-safe direction: no download unless a caller
    # explicitly decided otherwise.
    assert local_files_only is True


@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper native lib unavailable")
def test_garbage_audio_raises_voiceerror_not_crash():
    # Undecodable bytes reach the worker, fail to decode there, and come back as
    # a clean VoiceError - never the native transcription path, never a crash.
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"this is not an audio container at all" * 64)
    msg = str(ei.value).lower()
    assert "decode" in msg or "no audio" in msg


# --------------------------------------------------------------------------- #
# The premise: a native fault is uncatchable in-process (so isolation is needed).
# --------------------------------------------------------------------------- #

def test_native_fault_bypasses_try_except():
    # A child that wraps a genuine native abort in `except BaseException` still
    # dies, so an in-process try/except cannot contain it - hence the process
    # isolation.
    code = (
        "import os, ctypes\n"
        "if os.name == 'nt':\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)\n"
        "try:\n"
        "    os.abort()\n"
        "except BaseException:\n"
        "    print('SURVIVED')\n"
        "else:\n"
        "    print('NO_FAULT')\n"
    )
    proc = subprocess.run([sys.executable, "-u", "-c", code],
                          capture_output=True, text=True, timeout=30)
    assert "SURVIVED" not in proc.stdout
    assert "NO_FAULT" not in proc.stdout
    assert proc.returncode != 0          # the process died from the native fault


# --------------------------------------------------------------------------- #
# Containment: a crashed / hung worker never takes the server (this process) down.
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper native lib unavailable")
def test_worker_hard_exit_is_contained_and_recovers(monkeypatch):
    # Force the next worker to vanish mid-request (no Python traceback), exactly
    # like a native abort. The server (this process) must survive with a clean
    # error, then recover on the next call.
    monkeypatch.setenv(voice._FAULT_ENV, "exit")
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"\x00" * 4096)
    assert "crash" in str(ei.value).lower()

    # Recovery: env cleared, the worker respawns and processes a request again.
    monkeypatch.delenv(voice._FAULT_ENV, raising=False)
    with pytest.raises(voice.VoiceError) as ei2:
        voice.transcribe_bytes(b"still not audio" * 64)
    assert "decode" in str(ei2.value).lower()


@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper native lib unavailable")
def test_real_native_abort_in_worker_is_contained(monkeypatch):
    # The gold standard: a genuine uncatchable native abort (not a clean exit) in
    # the worker is still contained. Worker WER/abort dialogs are suppressed so it
    # dies fast instead of blocking.
    monkeypatch.setenv(voice._FAULT_ENV, "abort")
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"\x00" * 4096)
    assert "crash" in str(ei.value).lower()


@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper native lib unavailable")
def test_hung_worker_times_out_and_recovers(monkeypatch):
    # A wedged native call must not block STT forever: the worker is killed at the
    # timeout and a clean error is returned, then STT recovers.
    monkeypatch.setattr(voice, "_WORKER_TIMEOUT", 2.0)
    monkeypatch.setenv(voice._FAULT_ENV, "hang")
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"\x00" * 4096)
    assert "timed out" in str(ei.value).lower()

    # Recovery: clear the hang and give the cold worker a normal timeout (a fresh
    # spawn + faster-whisper import easily exceeds the 2s used above).
    monkeypatch.delenv(voice._FAULT_ENV, raising=False)
    monkeypatch.setattr(voice, "_WORKER_TIMEOUT", 120.0)
    with pytest.raises(voice.VoiceError) as ei2:
        voice.transcribe_bytes(b"not audio either" * 64)
    assert "decode" in str(ei2.value).lower()


# --------------------------------------------------------------------------- #
# Happy path through the real worker + real model (network + model download).
# --------------------------------------------------------------------------- #

def _make_wav(seconds: float = 1.0, freq: float = 220.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            frames += struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / rate)))
        w.writeframes(frames)
    return buf.getvalue()


@pytest.mark.integration
@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper native lib unavailable")
def test_real_transcription_runs_in_worker(monkeypatch):
    # End-to-end through the isolated worker with the real (tiny) model. A pure
    # tone has no speech, so "no speech detected" is a valid, non-crashing
    # outcome that still proves decode->load->transcribe ran in the worker.
    import localm.config as _cfg
    base = dict(_cfg.load_config())
    base["voice_stt_model"] = "tiny"
    # The one-time model download is policy-gated now: this integration test
    # legitimately downloads (that is what it tests), so run it under
    # net_mode=allow rather than whatever the box's config says.
    base["net_mode"] = "allow"
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(_cfg, "load_config", lambda: base)
    try:
        text = voice.transcribe_bytes(_make_wav())
        assert isinstance(text, str)
    except voice.VoiceError as e:
        assert "no speech" in str(e).lower()
