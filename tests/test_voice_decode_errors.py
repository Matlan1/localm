# SPDX-License-Identifier: AGPL-3.0-or-later
"""A decoder library fault is reported as such, never as a corrupt recording."""
from __future__ import annotations

import importlib.util
import io
import queue
import re
import struct
import wave
from pathlib import Path

import pytest

from localm import voice
from localm.plugins.builtin.voice.plug import _voice_error_status

_KWARG_FAULT = "open() got an unexpected keyword argument 'metadata_errors'"


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _raiser(exc):
    def decode_audio(_fh):
        raise exc
    return decode_audio


@pytest.mark.skipif(not _has("av"), reason="PyAV not installed")
def test_media_error_classified_as_decode():
    from av.error import InvalidDataError
    exc = InvalidDataError(1094995529, "Invalid data found when processing input")
    assert voice._is_media_error(exc) is True
    audio, err = voice._decode_or_error(b"x", _raiser(exc))
    assert audio is None
    assert err[0] == "decode"


@pytest.mark.parametrize("exc", [
    TypeError(_KWARG_FAULT),
    AttributeError("module 'av' has no attribute 'x'"),
    ImportError("DLL load failed"),
])
def test_library_fault_classified_as_decoder_fault(exc):
    assert voice._is_media_error(exc) is False
    audio, err = voice._decode_or_error(b"x", _raiser(exc))
    assert audio is None
    assert err[0] == "decoder-fault"
    assert type(exc).__name__ in err[1]


@pytest.mark.parametrize("exc", [OSError("truncated"), ValueError("bad"), EOFError()])
def test_odd_stream_errors_classified_as_decode(exc):
    assert voice._decode_or_error(b"x", _raiser(exc))[1][0] == "decode"


class _AliveProc:
    exitcode = None

    def is_alive(self):
        return True


def _run_with_worker_reply(monkeypatch, reply):
    resp_q = queue.Queue()
    resp_q.put(reply)
    monkeypatch.setattr(voice, "_ensure_worker", lambda: None)
    monkeypatch.setattr(voice, "_proc", _AliveProc())
    monkeypatch.setattr(voice, "_req_q", queue.Queue())
    monkeypatch.setattr(voice, "_resp_q", resp_q)
    with pytest.raises(voice.VoiceError) as ei:
        voice._run_in_worker(b"x", "tiny", None, 5.0)
    return ei.value


def test_decode_tag_is_422_corrupt_audio(monkeypatch):
    e = _run_with_worker_reply(monkeypatch, ("error", "decode", "InvalidDataError: bad"))
    assert e.code == "decode"
    assert "corrupt or unsupported audio" in str(e)
    assert _voice_error_status(e)[0] == 422


def test_decoder_fault_tag_is_500_and_not_blamed_on_audio(monkeypatch):
    e = _run_with_worker_reply(
        monkeypatch, ("error", "decoder-fault", f"TypeError: {_KWARG_FAULT}"))
    assert e.code == "decoder-fault"
    assert "corrupt" not in str(e).lower()
    assert "unsupported audio" not in str(e).lower()
    assert _KWARG_FAULT in str(e)
    assert _voice_error_status(e)[0] == 500


def test_voice_extra_bounds_av_below_19():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8")
    block = re.search(r"^voice\s*=\s*\[(.*?)\]", text, re.S | re.M).group(1)
    reqs = re.findall(r'"([^"]+)"', block)
    from packaging.requirements import Requirement
    av = [Requirement(r) for r in reqs if Requirement(r).name == "av"]
    assert len(av) == 1
    assert av[0].specifier.contains("18.1.0")
    assert not av[0].specifier.contains("19.0.0")
    assert not av[0].specifier.contains("19.0.1")


@pytest.mark.skipif(not (_has("faster_whisper") and _has("av")),
                    reason="faster-whisper / PyAV not installed")
def test_installed_decoder_pair_decodes_a_wav():
    from faster_whisper.audio import decode_audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<16000h", *([0] * 16000)))
    audio, err = voice._decode_or_error(buf.getvalue(), decode_audio)
    assert err is None
    assert len(audio) == 16000
