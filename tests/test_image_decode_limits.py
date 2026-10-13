# SPDX-License-Identifier: AGPL-3.0-or-later
"""decode_image_url checks an image's format and declared size before decoding
it, and a request's ImageBudget caps how many images, and how many pixels in
total, one request may decode. Every refusal is an ImageLimitError, which the
routes answer with 413 and which crosses both model-worker boundaries."""

from __future__ import annotations

import base64
import io
import os
import struct
import zlib
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageFile

from localm.inference import media
from localm.inference.backends.base import (
    ImageLimitError,
    UnsupportedInputError,
    VisionInputError,
)
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


def _data_uri(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _encoded(w: int = 10, h: int = 10, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format=fmt)
    return buf.getvalue()


def _small_png(w: int = 10, h: int = 10) -> str:
    return _data_uri(_encoded(w, h))


@pytest.fixture
def load_calls(monkeypatch):
    calls = {"n": 0}
    real = ImageFile.ImageFile.load

    def counting(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", counting)
    return calls


def test_compressed_image_over_the_pixel_limit_is_refused_before_decode(load_calls):
    width, height = 8000, 6500
    assert width * height > media.MAX_IMAGE_PIXELS
    assert width * height < Image.MAX_IMAGE_PIXELS
    raw = _uniform_grey_png(width, height)
    assert len(raw) < 200_000
    with pytest.raises(ImageLimitError, match="pixel limit") as ei:
        decode_image_url(_data_uri(raw))
    assert isinstance(ei.value, VisionInputError)
    assert isinstance(ei.value, UnsupportedInputError)
    assert load_calls["n"] == 0


def test_accepted_image_is_decoded(load_calls):
    img = decode_image_url(_small_png(10, 10))
    assert img.size == (10, 10) and img.mode == "RGB"
    assert load_calls["n"] >= 1


def test_pixel_limit_boundary(monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 100)
    img = decode_image_url(_small_png(10, 10))
    assert img.size == (10, 10) and img.mode == "RGB"
    with pytest.raises(ImageLimitError, match="pixel limit"):
        decode_image_url(_small_png(10, 11))


@pytest.mark.parametrize("fmt,mime", [
    ("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("GIF", "image/gif"),
    ("BMP", "image/bmp"), ("TIFF", "image/tiff"),
])
def test_allowed_formats_decode(fmt, mime):
    img = decode_image_url(_data_uri(_encoded(12, 8, fmt), mime))
    assert img.size == (12, 8) and img.mode == "RGB"


def test_ico_is_refused_without_decoding_its_pixels(load_calls):
    raw = _encoded(32, 32, "ICO")
    assert Image.open(io.BytesIO(raw)).format == "ICO"
    load_calls["n"] = 0
    with pytest.raises(ImageLimitError, match="not in a supported format"):
        decode_image_url(_data_uri(raw, "image/x-icon"))
    assert load_calls["n"] == 0


def test_pillow_decompression_bomb_is_an_image_limit_error(monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 10 ** 9)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ImageLimitError, match="too large to decode"):
        decode_image_url(_small_png(10, 10))


def test_budget_refuses_the_image_past_the_count_limit(monkeypatch):
    monkeypatch.setattr(media, "MAX_REQUEST_IMAGES", 3)
    budget = ImageBudget()
    for _ in range(3):
        decode_image_url(_small_png(), budget=budget)
    with pytest.raises(ImageLimitError, match="more than 3 images"):
        decode_image_url(_small_png(), budget=budget)


def test_budget_refuses_the_image_past_the_request_pixel_total(monkeypatch):
    monkeypatch.setattr(media, "MAX_REQUEST_IMAGE_PIXELS", 250)
    budget = ImageBudget()
    decode_image_url(_small_png(), budget=budget)
    decode_image_url(_small_png(), budget=budget)
    with pytest.raises(ImageLimitError, match="total more than 250"):
        decode_image_url(_small_png(), budget=budget)


def test_limits_fit_a_resent_conversation_of_phone_photos():
    twelve_mp = 4032 * 3024
    assert media.MAX_REQUEST_IMAGES * twelve_mp <= media.MAX_REQUEST_IMAGE_PIXELS
    assert 8160 * 6120 <= media.MAX_IMAGE_PIXELS
    assert media.MAX_IMAGE_PIXELS < Image.MAX_IMAGE_PIXELS


def test_gguf_marker_split_shares_one_budget_across_the_request(monkeypatch):
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    monkeypatch.setattr(media, "MAX_REQUEST_IMAGES", 2)
    part = {"type": "image_url", "image_url": {"url": _small_png()}}
    messages = [{"role": "user", "content": [part, part]},
                {"role": "user", "content": [part]}]
    with pytest.raises(ImageLimitError, match="more than 2 images"):
        LlamaCpp._messages_with_markers(messages, "<__media__>", 0)
    _, decoded = LlamaCpp._messages_with_markers(messages[:1], "<__media__>", 0)
    assert len(decoded) == 2


def test_status_table_answers_413_for_a_limit_and_400_for_other_vision_errors():
    from localm.inference import http_server as hs

    assert hs.backend_error_status(ImageLimitError("x")) == 413
    assert hs.backend_error_status(VisionInputError("x")) == 400


def test_route_answers_413_when_the_backend_refuses_an_image_over_a_limit():
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "test-model"
    type(engine).loaded = property(lambda self: True)
    engine.active_requests = 0
    engine.supports_images = True
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 2

    def refusing_stream(*a, **kw):
        raise ImageLimitError("The request attaches more than 64 images; send at most 64.")
        yield  # pragma: no cover

    engine.chat_stream.side_effect = refusing_stream
    client = TestClient(create_app(engine), raise_server_exceptions=True)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": _small_png()}}]}],
        "max_tokens": 4,
        "stream": False,
    })
    assert r.status_code == 413, (r.status_code, r.text)
    assert "more than 64 images" in r.text

