# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading WAV files into mono float samples at a chosen sample rate.

Handles RIFF/WAVE with integer PCM (8, 16, 24 or 32 bit), IEEE float (32 or
64 bit) and WAVE_FORMAT_EXTENSIBLE carrying either. Channels are averaged to
mono and the result is resampled by linear interpolation. Integer samples are
scaled by 1 / 2**(bits-1), so a 16-bit sample ``s`` becomes ``s / 32768``.

Imports nothing from localm.
"""

from __future__ import annotations

import struct
from array import array
from dataclasses import dataclass

_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE
_MAX_CHANNELS = 8


class WavError(ValueError):
    """The bytes are not a WAV file this module can read."""


@dataclass(frozen=True)
class WavInfo:
    """The layout of a WAV file's sample data."""
    sample_rate: int
    channels: int
    bits: int
    is_float: bool
    n_frames: int

    @property
    def seconds(self) -> float:
        return self.n_frames / self.sample_rate if self.sample_rate else 0.0


def _chunks(data: bytes):
    off = 12
    while off + 8 <= len(data):
        cid = data[off:off + 4]
        (size,) = struct.unpack_from("<I", data, off + 4)
        body = off + 8
        yield cid, body, min(size, len(data) - body)
        off = body + size + (size & 1)


def _parse(data: bytes) -> tuple[WavInfo, memoryview]:
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a WAV file (no RIFF/WAVE header)")
    fmt = None
    payload = None
    for cid, body, size in _chunks(data):
        if cid == b"fmt " and fmt is None:
            if size < 16:
                raise WavError("the WAV format chunk is truncated")
            fmt = data[body:body + size]
        elif cid == b"data" and payload is None:
            payload = memoryview(data)[body:body + size]
    if fmt is None or payload is None:
        raise WavError("the WAV file has no format or no data chunk")
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == _FORMAT_EXTENSIBLE:
        if len(fmt) < 26:
            raise WavError("the WAV extensible format chunk is truncated")
        (tag,) = struct.unpack_from("<H", fmt, 24)
    if tag not in (_FORMAT_PCM, _FORMAT_FLOAT):
        raise WavError("the WAV file is compressed; send uncompressed PCM or float WAV")
    is_float = tag == _FORMAT_FLOAT
    if is_float and bits not in (32, 64):
        raise WavError(f"unsupported float WAV sample size: {bits} bits")
    if not is_float and bits not in (8, 16, 24, 32):
        raise WavError(f"unsupported PCM WAV sample size: {bits} bits")
    if not 1 <= channels <= _MAX_CHANNELS:
        raise WavError(f"unsupported WAV channel count: {channels}")
    if rate <= 0:
        raise WavError("the WAV file declares no sample rate")
    width = bits // 8
    if block_align != width * channels:
        raise WavError("the WAV block alignment does not match its sample format")
    n_frames = len(payload) // block_align
    return WavInfo(rate, channels, bits, is_float, n_frames), payload[:n_frames * block_align]


def read_info(data: bytes) -> WavInfo:
    """The layout of WAV file *data*. Raises :class:`WavError`."""
    return _parse(data)[0]


def _samples(info: WavInfo, payload: memoryview) -> array:
    """Interleaved samples of *payload* as floats in [-1, 1]."""
    if info.is_float:
        out = array("f" if info.bits == 32 else "d")
        out.frombytes(payload)
        return out if info.bits == 32 else array("f", out)
    if info.bits == 8:
        return array("f", ((b - 128) / 128.0 for b in payload))
    if info.bits == 16:
        ints = array("h")
        ints.frombytes(payload)
        return array("f", (s / 32768.0 for s in ints))
    if info.bits == 32:
        ints = array("i")
        ints.frombytes(payload)
        return array("f", (s / 2147483648.0 for s in ints))
    raw = bytes(payload)
    return array("f", (int.from_bytes(raw[i:i + 3], "little", signed=True) / 8388608.0
                       for i in range(0, len(raw), 3)))


def _mono(samples: array, channels: int) -> array:
    if channels == 1:
        return samples
    return array("f", (sum(samples[i:i + channels]) / channels
                       for i in range(0, len(samples), channels)))


def _resample(samples: array, src_rate: int, dst_rate: int) -> array:
    if src_rate == dst_rate or len(samples) < 2:
        return samples
    n_out = max(1, int(len(samples) * dst_rate / src_rate))
    step = src_rate / dst_rate
    last = len(samples) - 1
    out = array("f", bytes(4 * n_out))
    for j in range(n_out):
        pos = j * step
        i = int(pos)
        if i >= last:
            out[j] = samples[last]
            continue
        frac = pos - i
        out[j] = samples[i] + (samples[i + 1] - samples[i]) * frac
    return out


def to_mono_float32(data: bytes, target_rate: int, *, max_seconds: float) -> bytes:
    """WAV file *data* as mono float32 little-endian samples at *target_rate*.

    Raises :class:`WavError` for a file that cannot be read, is empty, or is
    longer than *max_seconds*."""
    info, payload = _parse(data)
    if info.n_frames == 0:
        raise WavError("the WAV file holds no audio")
    if info.seconds > max_seconds:
        raise WavError(f"the recording is {info.seconds:.1f} s long; at most "
                       f"{max_seconds:.0f} s is accepted")
    mono = _resample(_mono(_samples(info, payload), info.channels),
                     info.sample_rate, target_rate)
    out = mono.tobytes()
    if struct.pack("=H", 1) != struct.pack("<H", 1):
        swapped = array("f", mono)
        swapped.byteswap()
        out = swapped.tobytes()
    return out
