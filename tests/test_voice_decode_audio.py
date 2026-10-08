# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speech-to-text decodes recordings with localm's own PyAV decoder.

faster-whisper 1.2.1's decoder passes a keyword that PyAV 19 removed from
``av.open``, which made every recording fail on installs that had PyAV 19. These
tests run real PyAV on real encoded audio (no fake decoder), so they fail on any
PyAV that cannot decode what a browser microphone produces.
"""
from __future__ import annotations

import importlib.util
import io
import struct
import wave
from pathlib import Path

import pytest

from localm import voice

pytestmark = pytest.mark.skipif(importlib.util.find_spec("av") is None,
                                reason="PyAV not installed")

_TONE_HZ = 440
_RATE_OUT = 16000


def _tone(rate: int, seconds: float):
    import numpy as np
    t = np.arange(int(rate * seconds))
    return (0.3 * np.sin(2 * np.pi * _TONE_HZ * t / rate) * 32767).astype(np.int16)


def _wav(rate: int, channels: int, seconds: float, extra_chunks: bytes = b"") -> bytes:
    import numpy as np
    mono = _tone(rate, seconds)
    pcm = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    raw = buf.getvalue()
    if extra_chunks:
        raw = raw[:4] + struct.pack("<I", len(raw) - 8 + len(extra_chunks)) + raw[8:] + extra_chunks
    return raw


def _encoded(container_format: str, codec: str, rate: int, seconds: float = 1.0) -> bytes:
    """*seconds* of a 440 Hz mono tone encoded with real PyAV, the way a browser
    recorder would write it. Skips the test when the codec is not in this build."""
    import av
    buf = io.BytesIO()
    try:
        with av.open(buf, "w", format=container_format) as out:
            stream = out.add_stream(codec, rate=rate)
            stream.layout = "mono"
            resampler = av.audio.resampler.AudioResampler(
                format=stream.format.name, layout="mono", rate=rate)
            frame = av.AudioFrame.from_ndarray(
                _tone(rate, seconds).reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = rate
            for piece in list(resampler.resample(frame)) + list(resampler.resample(None)):
                for packet in stream.encode(piece):
                    out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
    except av.error.FFmpegError as e:
        pytest.skip(f"this PyAV build cannot encode {codec}: {e}")
    return buf.getvalue()


def _peak_hz(audio) -> float:
    import numpy as np
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.argmax(spectrum) * _RATE_OUT / len(audio))


def _assert_is_the_tone(audio, expected_len: int, slack: int = 0) -> None:
    assert audio.dtype.name == "float32"
    assert abs(len(audio) - expected_len) <= slack
    assert float(abs(audio).max()) <= 1.0
    assert float(abs(audio).max()) > 0.1
    assert abs(_peak_hz(audio) - _TONE_HZ) < 8


def _decode(data: bytes):
    return voice.decode_audio(io.BytesIO(data))


def test_a_16k_mono_wav_decodes_to_the_tone():
    _assert_is_the_tone(_decode(_wav(16000, 1, 1.0)), 16000)


def test_a_44k_stereo_wav_is_downmixed_and_resampled_to_16k():
    _assert_is_the_tone(_decode(_wav(44100, 2, 2.0)), 32000)


def test_a_browser_shaped_webm_opus_recording_decodes():
    _assert_is_the_tone(_decode(_encoded("webm", "libopus", 48000)), 16000, slack=1600)


def test_an_mp3_decodes():
    _assert_is_the_tone(_decode(_encoded("mp3", "libmp3lame", 44100)), 16000, slack=3200)


def test_a_wav_with_metadata_that_is_not_valid_text_still_decodes():
    bad = b"INAM" + struct.pack("<I", 6) + b"\xff\xfe\xfa\xf0ab"
    chunks = b"LIST" + struct.pack("<I", 4 + len(bad)) + b"INFO" + bad
    _assert_is_the_tone(_decode(_wav(16000, 1, 1.0, extra_chunks=chunks)), 16000)


def test_a_truncated_recording_yields_the_audio_it_has():
    data = _encoded("webm", "libopus", 48000, seconds=2.0)
    audio = _decode(data[: int(len(data) * 0.5)])
    assert 0 < len(audio) < 2 * _RATE_OUT


def test_garbage_is_a_media_error_not_a_library_fault():
    from av.error import InvalidDataError
    with pytest.raises(InvalidDataError) as caught:
        _decode(bytes(range(256)) * 16)
    assert voice._is_media_error(caught.value) is True


def test_a_wav_with_no_samples_decodes_to_nothing():
    header = (b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
              + b"data" + struct.pack("<I", 0))
    assert len(_decode(header)) == 0


_REMOVED = "open() got an unexpected keyword argument 'metadata_errors'"


@pytest.fixture
def pyav_without_metadata_errors(monkeypatch):
    """Real PyAV, but ``av.open`` refuses ``metadata_errors`` exactly as PyAV 19
    does."""
    import av
    real_open = av.open

    def open_like_pyav_19(*args, **kwargs):
        if "metadata_errors" in kwargs:
            raise TypeError(_REMOVED)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", open_like_pyav_19)


def test_decoding_works_on_a_pyav_that_dropped_metadata_errors(pyav_without_metadata_errors):
    _assert_is_the_tone(_decode(_wav(16000, 1, 1.0)), 16000)
    _assert_is_the_tone(_decode(_encoded("webm", "libopus", 48000)), 16000, slack=1600)


def test_the_simulated_pyav_reproduces_the_failure_faster_whispers_decoder_hits(
        pyav_without_metadata_errors):
    """Fires-control for the fixture above: it must break the decoder that broke
    in the field, so the previous test passing means something."""
    pytest.importorskip("faster_whisper")
    from faster_whisper.audio import decode_audio
    with pytest.raises(TypeError, match="metadata_errors"):
        decode_audio(io.BytesIO(_wav(16000, 1, 1.0)))


def test_an_unrelated_type_error_from_pyav_is_not_swallowed(monkeypatch):
    import av

    def broken_open(*_a, **_k):
        raise TypeError("something else entirely")

    monkeypatch.setattr(av, "open", broken_open)
    with pytest.raises(TypeError, match="something else entirely"):
        _decode(_wav(16000, 1, 1.0))


def test_the_installed_decoder_runs_a_real_recording_through_the_worker_path():
    audio, err = voice._decode_or_error(_encoded("webm", "libopus", 48000),
                                        voice.decode_audio)
    assert err is None
    _assert_is_the_tone(audio, 16000, slack=1600)


def test_the_worker_does_not_use_faster_whispers_decoder():
    text = Path(voice.__file__).read_text(encoding="utf-8")
    assert "faster_whisper.audio" not in text


@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is None,
                    reason="faster-whisper not installed")
def test_the_self_test_decodes_with_the_installed_pyav():
    state, detail = voice.decode_self_test()
    assert state == "ok", detail
    assert "PyAV" in detail


def test_the_self_test_child_reports_a_decoder_failure(monkeypatch, capsys):
    def broken(_source, sampling_rate=16000):
        raise TypeError(_REMOVED)

    monkeypatch.setattr(voice, "decode_audio", broken)
    assert voice._self_test_child() == 1
    assert _REMOVED in capsys.readouterr().out


def test_the_self_test_is_absent_without_the_voice_extra(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "faster_whisper"
                        else real_find_spec(name, *a, **k))
    state, _detail = voice.decode_self_test()
    assert state == "absent"
