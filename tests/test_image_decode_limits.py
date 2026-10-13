# SPDX-License-Identifier: AGPL-3.0-or-later
"""decode_image_url checks an image's declared size before decoding it, and a
request's ImageBudget caps how many images, and how many pixels in total, one
request may decode, on both the GGUF and the HF decode paths."""

from __future__ import annotations

import base64
import io
import struct
import zlib

import pytest
from PIL import Image

from localm.inference import media
from localm.inference.backends.base import UnsupportedInputError, VisionInputError
from localm.inference.media import ImageBudget, decode_image_url


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _uniform_grey_png(width: int, height: int) -> bytes:
    """A valid 8-bit greyscale PNG of *width* x *height*, compressed row by row
    so the full raster is never held in memory."""
    comp = zlib.compressobj(9)
    row = b"\x00" * (width + 1)
    idat = b"".join(comp.compress(row) for _ in range(height)) + comp.flush()
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b""))


def _data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _small_png(w: int = 10, h: int = 10) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return _data_uri(buf.getvalue())


def test_compressed_image_over_the_pixel_limit_is_refused_before_decode():
    width, height = 8000, 6500
    assert width * height > media.MAX_IMAGE_PIXELS
    assert width * height < Image.MAX_IMAGE_PIXELS
    raw = _uniform_grey_png(width, height)
    assert len(raw) < 200_000
    with pytest.raises(VisionInputError, match="pixel limit") as ei:
        decode_image_url(_data_uri(raw))
    assert isinstance(ei.value, UnsupportedInputError)


def test_pixel_limit_boundary(monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 100)
    img = decode_image_url(_small_png(10, 10))
    assert img.size == (10, 10) and img.mode == "RGB"
    with pytest.raises(VisionInputError, match="pixel limit"):
        decode_image_url(_small_png(10, 11))


def test_pillow_decompression_bomb_is_a_vision_input_error(monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 10 ** 9)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(VisionInputError, match="too large to decode"):
        decode_image_url(_small_png(10, 10))


def test_budget_refuses_the_image_past_the_count_limit(monkeypatch):
    monkeypatch.setattr(media, "MAX_REQUEST_IMAGES", 3)
    budget = ImageBudget()
    for _ in range(3):
        decode_image_url(_small_png(), budget=budget)
    with pytest.raises(VisionInputError, match="more than 3 images"):
        decode_image_url(_small_png(), budget=budget)


def test_budget_refuses_the_image_past_the_request_pixel_total(monkeypatch):
    monkeypatch.setattr(media, "MAX_REQUEST_IMAGE_PIXELS", 250)
    budget = ImageBudget()
    decode_image_url(_small_png(), budget=budget)
    decode_image_url(_small_png(), budget=budget)
    with pytest.raises(VisionInputError, match="total more than 250"):
        decode_image_url(_small_png(), budget=budget)


def test_request_limits_hold_the_documented_values():
    assert media.MAX_IMAGE_PIXELS == 50_000_000
    assert media.MAX_REQUEST_IMAGE_PIXELS >= media.MAX_IMAGE_PIXELS
    assert media.MAX_IMAGE_PIXELS < Image.MAX_IMAGE_PIXELS


def test_gguf_marker_split_shares_one_budget_across_the_request(monkeypatch):
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    monkeypatch.setattr(media, "MAX_REQUEST_IMAGES", 2)
    part = {"type": "image_url", "image_url": {"url": _small_png()}}
    messages = [{"role": "user", "content": [part, part]},
                {"role": "user", "content": [part]}]
    with pytest.raises(VisionInputError, match="more than 2 images"):
        LlamaCpp._messages_with_markers(messages, "<__media__>", 0)
    _, decoded = LlamaCpp._messages_with_markers(messages[:1], "<__media__>", 0)
    assert len(decoded) == 2

