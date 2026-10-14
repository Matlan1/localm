# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode base64-encoded or URL-fetched media into PIL / numpy objects."""

from __future__ import annotations

import base64
import io
import re
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    # numpy is referenced ONLY by decode_audio's return annotation, and this
    # module has `from __future__ import annotations`, so that annotation is
    # never evaluated at runtime. A module-scope import would make the whole
    # module unimportable without numpy - including decode_image_url, which needs
    # only Pillow - and numpy is not a core dependency (it arrives transitively
    # via the voice extra).
    import numpy as np


# A vision image can be several MB; cap the network fetch generously but
# bounded.
_IMAGE_MAX_BYTES = 25_000_000

# Largest image accepted, in pixels, checked from the image header before it is
# decoded. Above a 50-megapixel camera frame (8160x6120) and 8K UHD (7680x4320);
# about 143 MiB once decoded to RGB.
MAX_IMAGE_PIXELS = 50_000_000
# Most images, and most pixels summed over them, one request may decode. A chat
# client resends every earlier image each turn, so these count the whole
# conversation: 64 twelve-megapixel photos (4032x3024) fit in the pixel total.
MAX_REQUEST_IMAGES = 64
MAX_REQUEST_IMAGE_PIXELS = 800_000_000
# Formats decoded. Each one's open reads only the header, so the size check
# runs before any pixel data is decoded.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF", "AVIF")


class ImageBudget:
    """The images one request has decoded so far, against
    ``MAX_REQUEST_IMAGES`` and ``MAX_REQUEST_IMAGE_PIXELS``. Pass one instance
    to every ``decode_image_url`` call of a request. Not thread-safe."""

    def __init__(self) -> None:
        self.images = 0
        self.pixels = 0

    def take_image(self) -> None:
        """Count one more image. Raises ImageLimitError past
        ``MAX_REQUEST_IMAGES``."""
        self.images += 1
        if self.images > MAX_REQUEST_IMAGES:
            from localm.inference.backends.base import ImageLimitError
            raise ImageLimitError(
                f"The request attaches more than {MAX_REQUEST_IMAGES} images; "
                f"send at most {MAX_REQUEST_IMAGES}.")

    def take_pixels(self, pixels: int) -> None:
        """Count *pixels* more decoded pixels. Raises ImageLimitError past
        ``MAX_REQUEST_IMAGE_PIXELS``."""
        self.pixels += pixels
        if self.pixels > MAX_REQUEST_IMAGE_PIXELS:
            from localm.inference.backends.base import ImageLimitError
            raise ImageLimitError(
                f"The attached images total more than {MAX_REQUEST_IMAGE_PIXELS} "
                "pixels; send fewer or smaller images.")


def _other_known_format(Image, raw: bytes) -> bool:
    """True when a Pillow format outside ``IMAGE_FORMATS`` recognises the first
    bytes of *raw*. Reads only that prefix; opens nothing."""
    Image.init()
    prefix = raw[:16]
    for fmt, (_factory, accept) in Image.OPEN.items():
        if fmt not in IMAGE_FORMATS and accept is not None and accept(prefix):
            return True
    return False


def _open_bounded(Image, raw: bytes, budget: ImageBudget | None):
    """*raw* decoded by the ``PIL.Image`` module *Image* to an RGB image, after
    its declared size has been checked against ``MAX_IMAGE_PIXELS`` and
    *budget*. Raises ImageLimitError for an image over either limit or in a
    format outside ``IMAGE_FORMATS``, before any pixel data is decoded."""
    from localm.inference.backends.base import ImageLimitError
    Image.init()
    available = [f for f in IMAGE_FORMATS if f in Image.OPEN]
    try:
        img = Image.open(io.BytesIO(raw), formats=available)
    except Image.DecompressionBombError as e:
        raise ImageLimitError(
            f"The attached image is too large to decode: {e}") from e
    except Image.UnidentifiedImageError as e:
        if not _other_known_format(Image, raw):
            raise
        raise ImageLimitError(
            "The attached image is not in a supported format "
            f"({', '.join(IMAGE_FORMATS)}).") from e
    width, height = img.size
    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ImageLimitError(
            f"The attached image is {width}x{height} ({pixels} pixels), over the "
            f"{MAX_IMAGE_PIXELS}-pixel limit; scale it down and try again.")
    if budget is not None:
        budget.take_pixels(pixels)
    return img.convert("RGB")


def decode_image_url(url: str, budget: ImageBudget | None = None):
    """Return a PIL.Image.Image from a data-URI or http(s) URL.

    Raises ImageLimitError for an image over ``MAX_IMAGE_PIXELS`` or in a
    format outside ``IMAGE_FORMATS``, and, when *budget* is given, for one that
    takes the request past its image count or pixel limits. All are checked
    before the image is decoded.

    A remote (http/https) URL is fetched through localm.netpolicy so a chat
    'image_url' content part cannot turn the server into an SSRF proxy: the
    private-address guard, net_mode, and net_allow/net_deny all apply, every
    redirect hop is re-validated, and the body is size-capped. (Chat is the
    baseline capability - any key can send an image_url - so this fetch must be
    policy-checked like every other model-triggered request.)
    """
    try:
        from PIL import Image
    except ImportError as e:
        # Pillow is a core dependency, so this is reachable only on a build where
        # it failed to install or was removed. Report THAT, rather than letting an
        # ImportError escape: on the GGUF path this call runs inside the worker
        # process, whose dispatch loop treats any escaping exception as a native
        # fault and kills the process, evicting the model.
        # ImageDecodeUnavailable is an UnsupportedInputError, so the worker
        # reports it per-request and keeps serving (_runner.py).
        from localm.inference.backends.base import ImageDecodeUnavailable
        raise ImageDecodeUnavailable(
            "Cannot decode the attached image: the Pillow imaging library is not "
            "installed in this localm environment. Install it into the same "
            "environment (uv pip install pillow) and try again."
        ) from e

    if budget is not None:
        budget.take_image()
    if url.startswith("data:"):
        # data:image/jpeg;base64,<bytes>
        match = re.match(r"data:[^;]+;base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError(f"Malformed data URI: {url[:60]}")
        raw = base64.b64decode(match.group(1))
        return _open_bounded(Image, raw, budget)

    from localm.netpolicy import safe_fetch_bytes
    _final_url, _content_type, raw = safe_fetch_bytes(
        url, max_bytes=_IMAGE_MAX_BYTES)
    return _open_bounded(Image, raw, budget)


def decode_audio(b64: str, fmt: str) -> tuple[np.ndarray, int]:
    """Return (samples_float32, sample_rate) from a base64-encoded audio blob."""
    import soundfile as sf

    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    samples, sr = sf.read(buf, dtype="float32")
    return samples, sr


# Largest encoded audio clip accepted, in bytes after base64 decoding.
AUDIO_MAX_BYTES = 50_000_000
# Longest audio clip accepted, in seconds.
AUDIO_MAX_SECONDS = 600.0
# Shortest audio clip accepted, in seconds.
AUDIO_MIN_SECONDS = 0.1

# Samples a decode or resample step handles between two stop checks.
_DECODE_CHUNK = 1 << 20

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _audio_error(message: str):
    from localm.inference.backends.base import AudioInputError
    return AudioInputError(message)


class AudioDecodeCancelled(Exception):
    """An audio decode stopped because its stop check returned True."""


def _check_stop(should_stop: Optional[Callable[[], bool]]) -> None:
    if should_stop is not None and should_stop():
        raise AudioDecodeCancelled()


def _wav_fmt(raw: bytes, off: int, size: int) -> tuple[int, int, int, int]:
    """``(encoding, channels, sample_rate, bits)`` from a WAV ``fmt `` chunk body
    at *off*. *encoding* is the PCM or IEEE-float format tag, with an
    extensible header resolved to its sub-format."""
    import struct
    if size < 16:
        raise _audio_error("The attached WAV file has a truncated format header.")
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", raw, off)
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        if size < 40:
            raise _audio_error("The attached WAV file has a truncated format header.")
        (tag,) = struct.unpack_from("<H", raw, off + 24)
    if tag not in (_WAVE_FORMAT_PCM, _WAVE_FORMAT_IEEE_FLOAT):
        raise _audio_error(
            f"The attached WAV file uses an encoding localm cannot read (format "
            f"tag {tag:#06x}). Send 8, 16, 24 or 32-bit PCM or 32/64-bit float WAV.")
    widths = (8, 16, 24, 32) if tag == _WAVE_FORMAT_PCM else (32, 64)
    if bits not in widths or not 1 <= channels <= 32 or not 1 <= rate <= 768_000:
        raise _audio_error(
            f"The attached WAV file declares an unsupported layout ({channels} "
            f"channel(s), {rate} Hz, {bits}-bit).")
    if block_align != channels * bits // 8:
        raise _audio_error("The attached WAV file has an inconsistent format header.")
    return tag, channels, rate, bits


def _wav_frames(raw: bytes, should_stop: Optional[Callable[[], bool]] = None):
    """``(mono_samples, sample_rate)`` from the RIFF/WAVE bytes *raw*: every
    channel averaged into one, scaled to [-1, 1], as an ``array('f')``. Raises
    ``AudioInputError`` on a malformed or unsupported file, and
    ``AudioDecodeCancelled`` once *should_stop* returns True (polled every
    ``_DECODE_CHUNK`` samples)."""
    import struct
    import sys
    from array import array

    fmt = None
    data = None
    off = 12
    while off + 8 <= len(raw):
        cid = raw[off:off + 4]
        (size,) = struct.unpack_from("<I", raw, off + 4)
        body = off + 8
        if cid == b"fmt ":
            fmt = _wav_fmt(raw, body, min(size, len(raw) - body))
        elif cid == b"data":
            data = raw[body:body + size]
            break
        off = body + size + (size & 1)
    if fmt is None or data is None:
        raise _audio_error("The attached WAV file has no format or data section.")
    tag, channels, rate, bits = fmt
    width = bits // 8
    frame = width * channels
    n_frames = len(data) // frame
    if n_frames > AUDIO_MAX_SECONDS * rate:
        raise _audio_error(
            f"The attached audio is longer than {AUDIO_MAX_SECONDS:.0f} "
            "seconds, the most localm accepts in one clip.")
    data = data[:n_frames * frame]
    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        values = array("f" if bits == 32 else "d")
        scale = 1.0
    elif bits == 8:
        values = array("B")
        scale = 1.0 / 128.0
    elif bits == 16:
        values = array("h")
        scale = 1.0 / 32768.0
    else:
        values = array("i")
        scale = 1.0 / 2147483648.0
        if bits == 24:
            widened = bytearray(len(data) // 3 * 4)
            widened[1::4] = data[0::3]
            widened[2::4] = data[1::3]
            widened[3::4] = data[2::3]
            data = bytes(widened)
    if values.itemsize != (4 if bits == 24 else width):
        raise _audio_error("This platform cannot decode the attached WAV sample width.")
    values.frombytes(data)
    if sys.byteorder != "little" and values.itemsize > 1:
        values.byteswap()
    offset = 128 if bits == 8 else 0
    step = max(1, _DECODE_CHUNK // channels) * channels
    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        for i in range(0, len(values), step):
            _check_stop(should_stop)
            if not all(-1e30 < v < 1e30 for v in values[i:i + step]):
                raise _audio_error(
                    "The attached WAV file contains non-finite or out-of-range samples.")
    mono = array("f")
    mix = scale / channels
    bias = offset * channels
    for i in range(0, len(values), step):
        _check_stop(should_stop)
        chunk = values[i:i + step]
        if channels == 1:
            mono.extend((v - offset) * scale for v in chunk)
        else:
            lanes = [chunk[c::channels] for c in range(channels)]
            mono.extend((sum(group) - bias) * mix
                        for group in zip(*lanes, strict=True))
    return mono, rate


def _resample_with_av(samples, rate: int, target: int):
    """The ``array('f')`` *samples* at *rate* Hz resampled to *target* Hz by
    FFmpeg's resampler, as an ``array('f')``, or None when PyAV is not
    installed."""
    try:
        import av
    except ImportError:
        return None
    from array import array
    frame = av.AudioFrame(format="flt", layout="mono", samples=len(samples))
    frame.planes[0].update(samples.tobytes())
    frame.sample_rate = rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target)
    out = array("f")
    for fr in resampler.resample(frame) + resampler.resample(None):
        out.frombytes(bytes(fr.planes[0])[:fr.samples * 4])
    return out


def _resample(samples, rate: int, target: int,
              should_stop: Optional[Callable[[], bool]] = None):
    """The ``array('f')`` *samples* at *rate* Hz resampled to *target* Hz, as an
    ``array('f')``: with FFmpeg's resampler when PyAV is installed; otherwise
    box-filtered averaging over each output sample's span when lowering the
    rate, linear interpolation when raising it. Without PyAV, raises
    ``AudioDecodeCancelled`` once *should_stop* returns True (polled every
    ``_DECODE_CHUNK`` samples) and holds the running sums of at most
    ``_DECODE_CHUNK`` input samples at a time."""
    from array import array
    if rate == target or not samples:
        return samples
    resampled = _resample_with_av(samples, rate, target)
    if resampled is not None:
        return resampled
    n = len(samples)
    m = n * target // rate
    step = rate / target
    out = array("f")
    if rate < target:
        last = n - 1

        def lerp(i: int) -> float:
            t = i * step
            k = int(t)
            if k >= last:
                return samples[last]
            return samples[k] + (samples[k + 1] - samples[k]) * (t - k)

        for i0 in range(0, m, _DECODE_CHUNK):
            _check_stop(should_stop)
            out.extend(lerp(i) for i in range(i0, min(m, i0 + _DECODE_CHUNK)))
        return out
    from itertools import accumulate
    i = 0
    prev = 0.0
    total = 0.0
    for base in range(0, n, _DECODE_CHUNK):
        _check_stop(should_stop)
        chunk = samples[base:base + _DECODE_CHUNK]
        prefix = array("d", accumulate(chunk, initial=total))
        end = base + len(chunk)
        while i < m:
            t = (i + 1) * step
            k = int(t)
            if k >= end:
                break
            here = prefix[k - base] + (t - k) * chunk[k - base]
            out.append((here - prev) / step)
            prev = here
            i += 1
        total = prefix[-1]
    while i < m:
        out.append((total - prev) / step)
        prev = total
        i += 1
    return out


def _decode_with_av(raw: bytes, target: int, max_seconds: float,
                    should_stop: Optional[Callable[[], bool]] = None):
    """Mono float samples at *target* Hz, as an ``array('f')``, decoded from
    *raw* with PyAV, which reads any container and codec FFmpeg knows. Raises
    ``AudioInputError`` when PyAV is not installed or cannot decode *raw*, and
    ``AudioDecodeCancelled`` once *should_stop* returns True (polled once per
    decoded frame)."""
    try:
        import av
    except ImportError as e:
        from localm.inference.backends.base import AudioDecodeUnavailable
        raise AudioDecodeUnavailable(
            "The attached audio is not a WAV file, and reading other audio "
            "formats needs the voice extra (pip install 'localm[voice]'). Send "
            "the clip as WAV, or install the extra.") from e
    from array import array

    from localm.inference.backends.base import AudioInputError
    out = array("f")
    limit = int(max_seconds * target) + 1
    import av.error
    decode_errors = (av.error.FFmpegError, ValueError, OSError, EOFError)
    try:
        with av.open(io.BytesIO(raw), mode="r") as container:
            if not container.streams.audio:
                raise _audio_error("The attached file contains no audio stream.")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=target)
            stream = container.streams.audio[0]

            def take(frames) -> None:
                for fr in frames:
                    out.frombytes(bytes(fr.planes[0])[:fr.samples * 4])
                if len(out) > limit:
                    raise _audio_error(
                        f"The attached audio is longer than {max_seconds:.0f} "
                        "seconds, the most localm accepts in one clip.")

            for frame in container.decode(stream):
                _check_stop(should_stop)
                take(resampler.resample(frame))
            take(resampler.resample(None))
    except AudioInputError:
        raise
    except decode_errors as e:
        raise _audio_error(
            f"The attached audio could not be decoded ({type(e).__name__}).") from e
    return out


def decode_audio_clip(b64: str, fmt: str, target_rate: int,
                      should_stop: Optional[Callable[[], bool]] = None):
    """Decode the base64 audio *b64* to mono 32-bit float samples at
    *target_rate* Hz, as an ``array('f')``.

    A RIFF/WAVE payload is decoded by localm itself (8/16/24/32-bit PCM and
    32/64-bit float, any channel count, averaged to mono, resampled when its
    rate differs). Any other payload is decoded with PyAV when it is installed.
    *fmt* is the client's declared format: ``"wav"`` on a payload that is not
    WAV is refused.

    Raises ``AudioInputError`` (an ``UnsupportedInputError``) when the payload
    is not valid base64, larger than :data:`AUDIO_MAX_BYTES`, undecodable,
    shorter than :data:`AUDIO_MIN_SECONDS` or longer than
    :data:`AUDIO_MAX_SECONDS`. No message includes the payload's bytes.
    Raises ``AudioDecodeCancelled`` once *should_stop* returns True; it is
    polled while the samples are decoded and resampled."""
    import binascii

    if not isinstance(b64, str) or not b64:
        raise _audio_error("The attached audio clip is empty.")
    if len(b64) > AUDIO_MAX_BYTES * 4 // 3 + 4:
        raise _audio_error(
            f"The attached audio is larger than {AUDIO_MAX_BYTES // 1_000_000} MB, "
            "the most localm accepts in one clip.")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise _audio_error("The attached audio is not valid base64 data.") from e
    if not raw:
        raise _audio_error("The attached audio clip is empty.")
    is_wav = len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    declared = (fmt or "").strip().lower()
    if declared == "wav" and not is_wav:
        raise _audio_error(
            "The attached audio is declared as WAV but is not a WAV file.")
    if is_wav:
        import struct

        from localm.inference.backends.base import AudioInputError
        try:
            samples, rate = _wav_frames(raw, should_stop=should_stop)
        except AudioInputError:
            raise
        except (struct.error, ValueError, IndexError) as e:
            raise _audio_error(
                f"The attached WAV file is malformed ({type(e).__name__}).") from e
        except MemoryError as e:
            raise _audio_error("The attached audio is too large to decode.") from e
        try:
            samples = _resample(samples, rate, target_rate, should_stop)
        except AudioDecodeCancelled:
            raise
        except MemoryError as e:
            raise _audio_error("The attached audio is too large to resample.") from e
        except Exception as e:  # noqa: BLE001 - any resampler failure is this request's
            raise _audio_error(
                f"The attached audio could not be resampled from {rate} Hz to "
                f"{target_rate} Hz ({type(e).__name__}).") from e
    else:
        samples = _decode_with_av(raw, target_rate, AUDIO_MAX_SECONDS,
                                  should_stop=should_stop)
    if len(samples) < AUDIO_MIN_SECONDS * target_rate:
        raise _audio_error(
            f"The attached audio is shorter than {AUDIO_MIN_SECONDS:g} seconds, "
            "too short to process.")
    return samples
