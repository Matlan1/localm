# SPDX-License-Identifier: AGPL-3.0-or-later
"""decode_audio_clip: the audio decoder behind ``input_audio`` on GGUF models.

WAV payloads are built byte by byte here, so every encoding the decoder claims
to read is exercised against values computed independently of it.
"""
import base64
import builtins
import math
import struct
from array import array

import pytest

from localm.inference import media
from localm.inference.backends.base import AudioInputError, UnsupportedInputError
from localm.inference.media import decode_audio_clip


def _wav(frames: bytes, *, rate=16000, channels=1, bits=16, tag=1,
         extensible=False, extra_chunks=b"") -> bytes:
    """A RIFF/WAVE file holding *frames* with the given format header."""
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * block, block, bits)
        fmt += struct.pack("<HHI", 22, bits, 0)
        fmt += struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra_chunks
            + b"data" + struct.pack("<I", len(frames)) + frames)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _pcm16(values) -> bytes:
    return struct.pack("<%dh" % len(values), *values)


def _tone(n, rate, freq=440.0, amp=0.5):
    return [amp * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]


class TestWavEncodings:
    def test_pcm16_mono_at_the_target_rate_is_read_exactly(self):
        values = [0, 1000, -1000, 32767, -32768] * 400
        out = decode_audio_clip(_b64(_wav(_pcm16(values))), "wav", 16000)
        assert isinstance(out, array) and out.typecode == "f"
        assert list(out) == [v / 32768.0 for v in values]

    def test_pcm8_is_unsigned_around_128(self):
        raw = bytes([128, 255, 0, 192] * 500)
        out = decode_audio_clip(_b64(_wav(raw, bits=8)), "wav", 16000)
        assert list(out[:4]) == [0.0, 127 / 128, -1.0, 0.5]

    def test_pcm24_is_sign_extended(self):
        samples = [0, 1, -1, 0x7FFFFF, -0x800000] * 400
        raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in samples)
        out = decode_audio_clip(_b64(_wav(raw, bits=24)), "wav", 16000)
        assert list(out[:5]) == pytest.approx(
            [0.0, 1 / 2**23, -1 / 2**23, 0x7FFFFF / 2**23, -1.0])

    def test_pcm32(self):
        samples = [0, 2**30, -2**31] * 600
        raw = struct.pack("<%di" % len(samples), *samples)
        out = decode_audio_clip(_b64(_wav(raw, bits=32)), "wav", 16000)
        assert list(out[:3]) == [0.0, 0.5, -1.0]

    @pytest.mark.parametrize("bits,code", [(32, "f"), (64, "d")])
    def test_ieee_float(self, bits, code):
        samples = [0.25, -0.5, 0.75] * 600
        raw = array(code, samples).tobytes()
        out = decode_audio_clip(_b64(_wav(raw, bits=bits, tag=3)), "wav", 16000)
        assert list(out[:3]) == [0.25, -0.5, 0.75]

    def test_extensible_header_resolves_its_subformat(self):
        values = [100, -100] * 1000
        out = decode_audio_clip(_b64(_wav(_pcm16(values), extensible=True)), "wav", 16000)
        assert list(out[:2]) == [100 / 32768.0, -100 / 32768.0]

    def test_stereo_is_averaged_to_mono(self):
        frames = []
        for _ in range(2000):
            frames += [16384, -8192]
        out = decode_audio_clip(_b64(_wav(_pcm16(frames), channels=2)), "wav", 16000)
        assert len(out) == 2000
        assert out[0] == pytest.approx((16384 - 8192) / 2 / 32768.0)

    def test_odd_sized_chunk_before_data_is_skipped_with_its_pad_byte(self):
        values = [500] * 2000
        extra = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\x00"
        out = decode_audio_clip(_b64(_wav(_pcm16(values), extra_chunks=extra)), "wav", 16000)
        assert len(out) == 2000 and out[0] == 500 / 32768.0

    def test_declared_data_size_past_the_end_reads_what_is_there(self):
        raw = bytearray(_wav(_pcm16([7] * 2000)))
        data_size_at = raw.index(b"data") + 4
        raw[data_size_at:data_size_at + 4] = struct.pack("<I", 0xFFFFFFFF)
        out = decode_audio_clip(_b64(bytes(raw)), "wav", 16000)
        assert len(out) == 2000


class TestResampling:
    def test_builtin_resampler_halves_the_length_and_keeps_a_constant(self, monkeypatch):
        monkeypatch.setattr(media, "_resample_with_av", lambda *a: None)
        out = decode_audio_clip(_b64(_wav(_pcm16([8192] * 32000), rate=32000)), "wav", 16000)
        assert len(out) == 16000
        assert all(v == pytest.approx(0.25) for v in out)

    def test_builtin_resampler_raises_the_rate_by_interpolation(self, monkeypatch):
        monkeypatch.setattr(media, "_resample_with_av", lambda *a: None)
        values = [0, 16384] * 4000
        out = decode_audio_clip(_b64(_wav(_pcm16(values), rate=8000)), "wav", 16000)
        assert len(out) == 16000
        assert list(out[:4]) == pytest.approx([0.0, 0.25, 0.5, 0.25])

    def test_builtin_resampler_attenuates_a_tone_above_the_new_nyquist(self, monkeypatch):
        monkeypatch.setattr(media, "_resample_with_av", lambda *a: None)
        tone = _tone(44100, 44100, freq=15000)
        raw = array("f", tone).tobytes()
        out = decode_audio_clip(_b64(_wav(raw, rate=44100, bits=32, tag=3)), "wav", 16000)
        rms = math.sqrt(sum(v * v for v in out) / len(out))
        assert rms < 0.05, f"a 15 kHz tone must not alias into a 16 kHz clip at {rms}"

    def test_ffmpeg_resampler_is_used_when_installed(self):
        pytest.importorskip("av")
        tone = _tone(44100, 44100, freq=1000)
        raw = array("f", tone).tobytes()
        out = decode_audio_clip(_b64(_wav(raw, rate=44100, bits=32, tag=3)), "wav", 16000)
        assert abs(len(out) - 16000) <= 1
        mid = out[2000:14000]
        rms = math.sqrt(sum(v * v for v in mid) / len(mid))
        assert rms == pytest.approx(0.5 / math.sqrt(2), rel=0.02)


class TestOtherFormats:
    def test_non_wav_without_pyav_names_the_extra(self, monkeypatch):
        real_import = builtins.__import__

        def no_av(name, *args, **kwargs):
            if name == "av":
                raise ImportError("no av")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_av)
        with pytest.raises(AudioInputError, match="voice extra"):
            decode_audio_clip(_b64(b"ID3" + b"\x00" * 64), "mp3", 16000)

    def test_flac_decodes_through_pyav(self):
        pytest.importorskip("av")
        sf = pytest.importorskip("soundfile")
        import io
        buf = io.BytesIO()
        sf.write(buf, _tone(16000, 16000), 16000, format="FLAC", subtype="PCM_16")
        out = decode_audio_clip(_b64(buf.getvalue()), "flac", 16000)
        assert len(out) == 16000
        assert out[400] == pytest.approx(_tone(16000, 16000)[400], abs=1e-3)

    def test_garbage_that_pyav_cannot_read_is_an_audio_input_error(self):
        pytest.importorskip("av")
        with pytest.raises(AudioInputError, match="could not be decoded"):
            decode_audio_clip(_b64(b"not audio at all " * 20), "mp3", 16000)


class TestRefusals:
    def test_audio_input_error_is_an_unsupported_input_error(self):
        assert issubclass(AudioInputError, UnsupportedInputError)

    @pytest.mark.parametrize("payload,match", [
        ("", "empty"),
        ("!!!not base64", "not valid base64"),
    ])
    def test_unusable_payloads(self, payload, match):
        with pytest.raises(AudioInputError, match=match):
            decode_audio_clip(payload, "wav", 16000)

    def test_declared_wav_that_is_not_wav(self):
        with pytest.raises(AudioInputError, match="declared as WAV"):
            decode_audio_clip(_b64(b"OggS" + b"\x00" * 64), "wav", 16000)

    def test_riff_without_fmt_or_data(self):
        with pytest.raises(AudioInputError, match="no format or data"):
            decode_audio_clip(_b64(b"RIFF\x04\x00\x00\x00WAVE"), "wav", 16000)

    def test_unsupported_wav_encoding(self):
        with pytest.raises(AudioInputError, match="cannot read"):
            decode_audio_clip(_b64(_wav(b"\x00" * 4000, tag=2)), "wav", 16000)

    def test_inconsistent_block_align(self):
        raw = bytearray(_wav(_pcm16([0] * 2000)))
        fmt_at = raw.index(b"fmt ") + 8
        raw[fmt_at + 12:fmt_at + 14] = struct.pack("<H", 3)
        with pytest.raises(AudioInputError, match="inconsistent"):
            decode_audio_clip(_b64(bytes(raw)), "wav", 16000)

    def test_non_finite_float_samples(self):
        raw = array("f", [float("nan")] * 2000).tobytes()
        with pytest.raises(AudioInputError, match="non-finite"):
            decode_audio_clip(_b64(_wav(raw, bits=32, tag=3)), "wav", 16000)

    def test_too_short(self):
        with pytest.raises(AudioInputError, match="shorter than"):
            decode_audio_clip(_b64(_wav(_pcm16([1] * 100))), "wav", 16000)

    def test_too_long(self, monkeypatch):
        monkeypatch.setattr(media, "AUDIO_MAX_SECONDS", 0.5)
        with pytest.raises(AudioInputError, match="longer than"):
            decode_audio_clip(_b64(_wav(_pcm16([1] * 16000))), "wav", 16000)

    def test_too_big(self, monkeypatch):
        monkeypatch.setattr(media, "AUDIO_MAX_BYTES", 1000)
        with pytest.raises(AudioInputError, match="larger than"):
            decode_audio_clip(_b64(_wav(_pcm16([1] * 16000))), "wav", 16000)

    def test_an_unexpected_parse_error_stays_a_per_request_error(self, monkeypatch):
        def boom(raw):
            raise struct.error("unpack requires a buffer of 4 bytes")
        monkeypatch.setattr(media, "_wav_frames", boom)
        with pytest.raises(AudioInputError, match="malformed"):
            decode_audio_clip(_b64(_wav(_pcm16([1] * 4000))), "wav", 16000)

    def test_a_resampler_failure_stays_a_per_request_error(self, monkeypatch):
        def boom(samples, rate, target):
            raise RuntimeError("swr_init failed")
        monkeypatch.setattr(media, "_resample_with_av", boom)
        with pytest.raises(AudioInputError, match="could not be resampled from 32000 Hz"):
            decode_audio_clip(_b64(_wav(_pcm16([1] * 32000), rate=32000)), "wav", 16000)

    def test_a_missing_decoder_is_its_own_type(self, monkeypatch):
        from localm.inference.backends.base import AudioDecodeUnavailable
        real_import = builtins.__import__

        def no_av(name, *args, **kwargs):
            if name == "av":
                raise ImportError("no av")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_av)
        with pytest.raises(AudioDecodeUnavailable):
            decode_audio_clip(_b64(b"ID3" + b"\x00" * 64), "mp3", 16000)

    def test_no_refusal_message_carries_the_payload(self):
        secret = b"SECRET-AUDIO-CONTENT-" * 8
        cases = [
            (_b64(secret), "wav"),
            (_b64(b"RIFF\x00\x00\x00\x00WAVE" + secret), "wav"),
            (_b64(_wav(secret[:40])), "wav"),
        ]
        for payload, fmt in cases:
            with pytest.raises(AudioInputError) as info:
                decode_audio_clip(payload, fmt, 16000)
            text = str(info.value)
            assert "SECRET" not in text and payload[:24] not in text, text
