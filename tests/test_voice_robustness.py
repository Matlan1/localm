# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-1: voice transcription must reject empty / undecodable audio with a clean
VoiceError BEFORE it reaches the native (ctranslate2/PyAV) path, so a malformed
recording can never fault the whole server. These guard the decode-first ordering:
the Whisper model must NOT be loaded for input that cannot be decoded."""

import pytest

from localm import voice


def _fail_if_model_loaded(monkeypatch):
    """Make WhisperModel construction an instant failure, so any test that reaches
    model load fails loudly - proving the guard rejected bad input before that."""
    def _boom(*a, **k):
        raise AssertionError(
            "WhisperModel must not be constructed for empty/undecodable audio")
    monkeypatch.setattr("faster_whisper.WhisperModel", _boom, raising=False)


def test_empty_audio_raises_clean_voiceerror(monkeypatch):
    _fail_if_model_loaded(monkeypatch)
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"")
    assert "empty" in str(ei.value).lower()


def test_garbage_audio_raises_voiceerror_not_crash(monkeypatch):
    pytest.importorskip("faster_whisper")
    _fail_if_model_loaded(monkeypatch)
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"this is not an audio container at all" * 64)
    msg = str(ei.value).lower()
    assert "decode" in msg or "no audio" in msg
