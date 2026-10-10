# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading a WAV recording into mono float32 samples at a chosen rate."""

import struct
from array import array

import pytest

from localm.wav_audio import WavError, read_info, to_mono_float32


def wav(payload: bytes, *, rate=24000, channels=1, bits=16, tag=1,
        extensible_subformat=None, extra_chunks=b"") -> bytes:
    block = channels * bits // 8
    if extensible_subformat is not None:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * block, block, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", extensible_subformat) + b"\x00" * 14
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = (b"WAVE" + extra_chunks + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(payload)) + payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def floats(data: bytes) -> list:
    out = array("f")
    out.frombytes(data)
    return list(out)


class TestDecoding:
    def test_16_bit_mono_at_the_target_rate_is_s_over_32768(self):
        pcm = struct.pack("<4h", 0, 16384, -32768, 32767)
        assert floats(to_mono_float32(wav(pcm), 24000, max_seconds=30)) == [
            0.0, 0.5, -1.0, 32767 / 32768]

    def test_stereo_is_averaged_to_mono(self):
        pcm = struct.pack("<4h", 16384, 0, -16384, -16384)
        assert floats(to_mono_float32(wav(pcm, channels=2), 24000, max_seconds=30)) == [
            0.25, -0.5]

    def test_float32_is_taken_as_is(self):
        pcm = struct.pack("<3f", 0.25, -0.75, 1.0)
        got = floats(to_mono_float32(wav(pcm, bits=32, tag=3), 24000, max_seconds=30))
        assert got == [0.25, -0.75, 1.0]

    def test_8_bit_is_unsigned_around_128(self):
        assert floats(to_mono_float32(wav(bytes([128, 192, 0]), bits=8), 24000,
                                      max_seconds=30)) == [0.0, 0.5, -1.0]

    def test_24_bit_is_signed_little_endian(self):
        pcm = (4194304).to_bytes(3, "little", signed=True) + (-8388608).to_bytes(3, "little", signed=True)
        assert floats(to_mono_float32(wav(pcm, bits=24), 24000, max_seconds=30)) == [0.5, -1.0]

    def test_extensible_pcm_is_read(self):
        pcm = struct.pack("<2h", 16384, -16384)
        got = floats(to_mono_float32(wav(pcm, extensible_subformat=1), 24000, max_seconds=30))
        assert got == [0.5, -0.5]

    def test_unknown_chunks_before_fmt_are_skipped(self):
        pcm = struct.pack("<2h", 16384, 0)
        extra = b"LIST" + struct.pack("<I", 3) + b"abc\x00"
        got = floats(to_mono_float32(wav(pcm, extra_chunks=extra), 24000, max_seconds=30))
        assert got == [0.5, 0.0]

    def test_resampling_changes_the_length_by_the_rate_ratio(self):
        pcm = struct.pack("<16000h", *([1000] * 16000))
        out = floats(to_mono_float32(wav(pcm, rate=16000), 24000, max_seconds=30))
        assert len(out) == 24000
        assert all(abs(x - 1000 / 32768) < 1e-6 for x in out)

    def test_resampling_interpolates_linearly(self):
        pcm = struct.pack("<2h", 0, 16384)
        out = floats(to_mono_float32(wav(pcm, rate=12000), 24000, max_seconds=30))
        assert out[:3] == [0.0, 0.25, 0.5]

    def test_info_reports_the_layout(self):
        info = read_info(wav(struct.pack("<4h", 0, 0, 0, 0), rate=8000, channels=2))
        assert (info.sample_rate, info.channels, info.bits, info.n_frames) == (8000, 2, 16, 2)


class TestRefusals:
    @pytest.mark.parametrize("data,match", [
        (b"", "not a WAV"),
        (b"RIFF\x00\x00\x00\x00AVI LIST", "not a WAV"),
        (b"RIFF" + struct.pack("<I", 4) + b"WAVE", "no format or no data"),
    ])
    def test_not_a_wav(self, data, match):
        with pytest.raises(WavError, match=match):
            to_mono_float32(data, 24000, max_seconds=30)

    def test_compressed_audio_is_refused(self):
        with pytest.raises(WavError, match="compressed"):
            to_mono_float32(wav(b"\x00" * 8, tag=2), 24000, max_seconds=30)

    def test_an_empty_recording_is_refused(self):
        with pytest.raises(WavError, match="no audio"):
            to_mono_float32(wav(b""), 24000, max_seconds=30)

    def test_a_long_recording_is_refused(self):
        pcm = b"\x00\x00" * 8000 * 3
        with pytest.raises(WavError, match="at most 2 s"):
            to_mono_float32(wav(pcm, rate=8000), 24000, max_seconds=2)

    def test_a_block_alignment_that_does_not_match_is_refused(self):
        bad = bytearray(wav(b"\x00" * 8))
        struct.pack_into("<H", bad, 32, 3)
        with pytest.raises(WavError, match="block alignment"):
            to_mono_float32(bytes(bad), 24000, max_seconds=30)

    def test_too_many_channels_is_refused(self):
        with pytest.raises(WavError, match="channel count"):
            to_mono_float32(wav(b"\x00" * 18, channels=9), 24000, max_seconds=30)
