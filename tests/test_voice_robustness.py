# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-1: voice transcription must reject empty / undecodable audio with a clean
VoiceError BEFORE it reaches the native (ctranslate2/PyAV) path, so a malformed
recording can never fault the whole server.

faster-whisper's native lib (ctranslate2) can fail to load on Windows under load
(OSError WinError 127 - a documented pre-existing flake; CI Linux is green). The
empty-input guard runs before any import, so that test needs nothing; the decode
test skips if the native lib is unavailable rather than reporting that flake."""

import pytest

from localm import voice


def test_empty_audio_raises_clean_voiceerror():
    # Rejected up front, before any import or native call - so this holds even
    # when faster-whisper is not installed / cannot load.
    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"")
    assert "empty" in str(ei.value).lower()


def test_garbage_audio_raises_voiceerror_not_crash():
    try:
        import faster_whisper.audio  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"faster-whisper native lib unavailable: {e}")

    with pytest.raises(voice.VoiceError) as ei:
        voice.transcribe_bytes(b"this is not an audio container at all" * 64)
    msg = str(ei.value).lower()
    # "could not decode" comes only from the decode step, which runs BEFORE the
    # model is loaded - a clean VoiceError here proves the bad blob never reached
    # the native transcription path (which is where a crash would have happened).
    assert "decode" in msg or "no audio" in msg
