# SPDX-License-Identifier: AGPL-3.0-or-later
"""A refusal of an attached image or audio clip names the attachment it was
about (``X-Localm-Refusal`` on an HTTP error, ``refusal`` on a streamed
``localm_error``), and an image no decoder recognises is answered 413, not 400."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    AudioInputError, ContextCapacityExceededError, ImageDecodeUnavailable,
    ImageLimitError, InvalidGrammarError, UnsupportedInputError, VisionInputError)
from localm.inference.http_server import (
    MEDIA_REFUSAL_HEADER, create_app, media_refusal_kind)

_PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg==")
_WAV = "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode("ascii")


def _engine(*, images=False, audio=False, stream_exc=None):
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = images
    engine.supports_audio = audio
    engine.can_be_multimodal = False
    engine.supports_grammar = True
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    type(engine).loaded = property(lambda self: True)

    def _stream(messages, **kw):
        if stream_exc is not None:
            raise stream_exc
        yield "ok"

    engine.chat_stream.side_effect = _stream
    return engine


def _post(engine, content):
    with TestClient(create_app(engine), raise_server_exceptions=False) as c:
        return c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": False, "max_tokens": 4,
            "messages": [{"role": "user", "content": content}]})


_IMAGE = [{"type": "text", "text": "what is this"},
          {"type": "image_url", "image_url": {"url": _PNG}}]
_AUDIO = [{"type": "text", "text": "transcribe"},
          {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]


@pytest.mark.parametrize("exc,kind", [
    (AudioInputError("x"), "audio"),
    (VisionInputError("x"), "image"),
    (ImageLimitError("x"), "image"),
    (ImageDecodeUnavailable("x"), "image"),
    (UnsupportedInputError("x"), None),
    (ContextCapacityExceededError("x"), None),
    (InvalidGrammarError("x"), None),
    (RuntimeError("x"), None),
])
def test_refusal_kind_follows_the_attachment_the_error_is_about(exc, kind):
    assert media_refusal_kind(exc) == kind


def test_an_image_sent_to_a_model_that_cannot_read_images_is_marked_image():
    r = _post(_engine(images=False), _IMAGE)
    assert r.status_code == 400, r.text
    assert r.headers.get(MEDIA_REFUSAL_HEADER) == "image"


def test_audio_sent_to_a_model_that_cannot_hear_is_marked_audio():
    r = _post(_engine(audio=False), _AUDIO)
    assert r.status_code == 400, r.text
    assert r.headers.get(MEDIA_REFUSAL_HEADER) == "audio"


@pytest.mark.parametrize("exc,status,kind", [
    (AudioInputError("The audio could not be decoded"), 400, "audio"),
    (VisionInputError("cannot read this image"), 400, "image"),
    (ImageLimitError("too many images"), 413, "image"),
])
def test_a_backend_media_refusal_on_the_plain_path_is_marked(exc, status, kind):
    r = _post(_engine(images=True, audio=True, stream_exc=exc), _IMAGE)
    assert r.status_code == status, r.text
    assert r.headers.get(MEDIA_REFUSAL_HEADER) == kind


@pytest.mark.parametrize("exc,status", [
    (ContextCapacityExceededError("The conversation is too long"), 413),
    (InvalidGrammarError("grammar failed to parse"), 400),
])
def test_a_refusal_with_another_cause_is_not_marked_as_media(exc, status):
    r = _post(_engine(images=True, audio=True, stream_exc=exc), _IMAGE)
    assert r.status_code == status, r.text
    assert MEDIA_REFUSAL_HEADER not in r.headers


def test_an_image_no_decoder_recognises_is_an_image_limit_error():
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    garbage = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode("ascii")
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": garbage}}]}]
    with pytest.raises(ImageLimitError, match="could not be read") as ei:
        LlamaCpp._messages_with_markers(messages, "<__media__>", 0)
    assert type(ei.value.__cause__).__name__ == "UnidentifiedImageError"

