# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the ``/v1/*`` request bodies through the real ASGI app.

A request body is the one input a network client controls end to end. The
contract: a body the server cannot use is answered with a 4xx and a JSON
error, never a 5xx, never a hang, whatever the JSON shape, encoding or depth.
The engine behind the app is a stub, so this exercises the routes' parsing and
validation, which is where the body is first trusted."""
from __future__ import annotations

import base64
import json
import struct
import zlib
from unittest.mock import MagicMock

import pytest

pytest.importorskip("hypothesis")

from fastapi.testclient import TestClient  # noqa: E402
from hypothesis import example, given, strategies as st  # noqa: E402

from localm import textguard  # noqa: E402
from localm.inference.http_server import create_app  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402

_leaf = (st.none() | st.booleans() | st.integers(-(2 ** 70), 2 ** 70)
         | st.floats(allow_nan=True, allow_infinity=True) | st.text(max_size=12)
         | st.sampled_from(["", "localm", "yes", "x" * 3000, "\x00", "9" * 400]))
_json = st.recursive(
    _leaf,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=10), inner,
                                                                  max_size=4),
    max_leaves=25)

# An embeddings `input` list longer than any list _json draws.
_BATCH = st.integers(5, 600).flatmap(
    lambda n: st.lists(st.text(max_size=8), min_size=n, max_size=n))

_MESSAGE = st.fixed_dictionaries(
    {"role": st.sampled_from(["user", "system", "assistant", "tool", "bogus", 5, None]),
     "content": st.one_of(_leaf, st.lists(st.fixed_dictionaries(
         {"type": st.sampled_from(["text", "image_url", "input_audio", "other"]),
          "text": _json, "image_url": _json, "input_audio": _json}), max_size=3))},
    optional={"untrusted_spans": _json, "origin": _json, "reasoning_content": _json})

_CHAT_FIELDS = {
    "model": _json, "messages": st.lists(_MESSAGE | _json, max_size=3), "stream": _json,
    "max_tokens": _json, "temperature": _json, "top_p": _json, "top_k": _json,
    "repeat_penalty": _json, "grammar": _json, "grammar_lazy": _json,
    "grammar_triggers": _json, "seed": _json, "required_capabilities": _json,
    "pin_model": _json, "min_context": _json, "chat_template_kwargs": _json,
}


_VALID_CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
_VALID_COMPLETION = {"model": "test-model", "prompt": "hi"}
_VALID_EMBEDDING = {"model": "test-model", "input": "hi"}


def _bodies(fields: dict, base: dict):
    """Request bodies: a valid *base* with some fields replaced by junk (so the
    request gets past the model check into the handler), a bare dict of junk
    fields, any JSON value, and raw bytes."""
    overrides = st.dictionaries(st.sampled_from(sorted(fields)), st.one_of(*fields.values()),
                                max_size=6)
    return st.one_of(
        overrides.map(lambda o: json.dumps({**base, **o}, allow_nan=True).encode("utf-8")),
        overrides.map(lambda o: json.dumps({**base, **o}, allow_nan=True).encode("utf-8")),
        overrides.map(lambda o: json.dumps(o, allow_nan=True).encode("utf-8")),
        _json.map(lambda v: json.dumps(v, allow_nan=True).encode("utf-8")),
        st.binary(max_size=80),
        st.sampled_from([b"", b"{", b"[" * 20000, b'{"a":' * 20000, b"\xff\xfe{}",
                         b"NaN", b'{"messages": [{"role":"user","content":"' + b"a" * 200000 + b'"}]}']),
    )


def _engine(images: bool = False) -> MagicMock:
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = images
    engine.can_be_multimodal = images
    engine.supports_grammar = True
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: _pieces()
    engine.complete_stream.side_effect = lambda prompt, **kw: _pieces()
    return engine


def _pieces():
    yield "ok"


def _decoding_pieces(messages):
    """What a vision backend does with the request: decode every attached image
    and report one it cannot read as an unsupported input."""
    from localm.inference import media
    from localm.inference.backends.base import UnsupportedInputError, VisionInputError

    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            spec = part.get("image_url")
            url = spec.get("url", "") if isinstance(spec, dict) else ""
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            try:
                media.decode_image_url(url).convert("RGB")
            except UnsupportedInputError:
                raise
            except Exception as exc:
                raise VisionInputError(
                    f"The attached image could not be read ({type(exc).__name__}).") from exc
    yield "ok"


def _vision_engine() -> MagicMock:
    engine = _engine(images=True)
    engine.chat_stream.side_effect = lambda messages, **kw: _decoding_pieces(messages)
    return engine


_client_cache: dict = {}


def _client() -> TestClient:
    if "c" not in _client_cache:
        _client_cache["c"] = TestClient(create_app(_engine()), raise_server_exceptions=False)
    return _client_cache["c"]


def _new_client(engine) -> TestClient:
    """A client for a fresh app. The app serves through a module-level engine
    that the most recent create_app call sets, so the cached default client is
    dropped and rebuilt on its next use."""
    _client_cache.pop("c", None)
    return TestClient(create_app(engine), raise_server_exceptions=False)


@pytest.fixture
def vision_client():
    yield _new_client(_vision_engine())
    _client_cache.pop("c", None)


def _post(method: str, path: str, body: bytes, client=None):
    return _bounds.returns_within(
        (client or _client()).request, method, path, content=body,
        headers={"content-type": "application/json"}, seconds=30)


@pytest.mark.parametrize("path,fields,base", [
    ("/v1/chat/completions", _CHAT_FIELDS, _VALID_CHAT),
    ("/v1/completions", {k: _json for k in (
        "model", "prompt", "stream", "max_tokens", "temperature", "top_p", "top_k",
        "repeat_penalty", "grammar", "grammar_lazy", "grammar_triggers", "seed")},
     _VALID_COMPLETION),
    ("/v1/embeddings", {"model": _json, "input": _json | _BATCH, "encoding_format": _json},
     _VALID_EMBEDDING),
])
@given(data=st.data())
def test_inference_bodies_never_answer_5xx(path, fields, base, data):
    body = data.draw(_bodies(fields, base))
    resp = _post("POST", path, body)
    assert resp.status_code < 500, (resp.status_code, resp.text[:300], body[:300])


@pytest.mark.parametrize("method,path", [
    ("PATCH", "/v1/config"),
    ("POST", "/v1/media/config/image"),
    ("POST", "/v1/media/config/bogus"),
    ("POST", "/v1/tts/config"),
    ("POST", "/v1/plugins/chat/settings"),
    ("POST", "/v1/keys"),
    ("POST", "/v1/models/rename"),
    ("POST", "/v1/models/unload"),
])
@given(data=st.data())
def test_admin_bodies_never_answer_5xx(method, path, data):
    body = data.draw(_bodies({"name": _json, "scopes": _json, "model": _json, "key": _json,
                              "value": _json, "path": _json, "new_name": _json,
                              "old_name": _json, "label": _json}, {"name": "x"}))
    resp = _post(method, path, body)
    assert resp.status_code < 500, (resp.status_code, resp.text[:300], body[:300])


@pytest.mark.parametrize("path,base", [
    ("/v1/chat/completions", _VALID_CHAT),
    ("/v1/completions", _VALID_COMPLETION),
    ("/v1/embeddings", _VALID_EMBEDDING),
])
def test_a_valid_body_gets_past_validation_into_the_handler(path, base):
    resp = _post("POST", path, json.dumps(base).encode())
    assert resp.status_code not in (400, 401, 403, 404, 405, 415, 422), resp.text[:300]


def test_the_harness_reports_a_server_error_when_one_happens(monkeypatch):
    from localm.inference import http_server

    def boom(_messages):
        raise RuntimeError("boom")

    monkeypatch.setattr(http_server, "_protocol_messages_to_dicts", boom)
    client = TestClient(create_app(_engine()), raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", content=json.dumps(_VALID_CHAT).encode(),
                       headers={"content-type": "application/json"})
    assert resp.status_code >= 500


def test_a_malformed_body_is_refused_not_served():
    statuses = {_post("POST", "/v1/chat/completions", body).status_code
                for body in (b"{}", b"[", json.dumps({"model": "test-model",
                                                      "messages": "x"}).encode(),
                             json.dumps({**_VALID_CHAT, "temperature": float("nan")}).encode())}
    assert statuses <= {400, 422}


_spans = st.lists(st.lists(st.integers(-(2 ** 70), 2 ** 70), max_size=4), max_size=6)


@given(text=st.text(max_size=60), spans=_spans)
@example(text="abc", spans=[[2 ** 65, -(2 ** 65)]])
def test_guarded_text_spans_are_clamped_ordered_and_disjoint(text, spans):
    pairs = [tuple(s[:2]) for s in spans if len(s) >= 2]
    g = textguard.GuardedText(text, pairs)
    last = 0
    for a, b in g.untrusted_spans:
        assert last <= a < b <= len(text)
        last = b
    sliced = textguard.slice_guarded(g, -5, 2 ** 70)
    assert sliced == text[:]
    textguard.split_by_trust(text, pairs)


@given(text=st.text(max_size=400))
def test_neutralise_is_idempotent(text):
    once = textguard.neutralise(text)
    assert textguard.neutralise(once) == once


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _png(width: int, height: int, rows: int) -> bytes:
    """A PNG whose header declares *width* x *height* and whose pixel data holds
    *rows* rows of mid-grey."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = (b"\x00" + b"\x7f" * (3 * min(width, 4096))) * rows
    return (_PNG_MAGIC + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def _uri(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


_image_url = st.one_of(
    st.builds(lambda w, h: _uri(_png(w, h, h)), st.integers(1, 8), st.integers(1, 8)),
    st.builds(lambda w, h, rows: _uri(_png(w, h, rows)),
              st.sampled_from([30_000, 2 ** 20, 2 ** 31 - 1]),
              st.sampled_from([30_000, 2 ** 20, 2 ** 31 - 1]), st.integers(0, 2)),
    st.builds(lambda cut: _uri(_png(8, 8, 8)[:cut]), st.integers(0, 80)),
    st.builds(lambda magic, junk: _uri(magic + junk),
              st.sampled_from([b"\xff\xd8\xff", b"GIF89a", b"BM", _PNG_MAGIC, b""]),
              st.binary(max_size=200)),
    st.builds(lambda tail: "data:image/png;base64," + tail, st.text(max_size=40)),
    st.sampled_from(["data:", "data:image/png;base64,", "data:image/png,abc", "data:;base64,AAAA",
                     "data:image/png;base64," + "A" * 4097, "data:image/png;base64,A"]),
)

_image_message = st.lists(
    st.fixed_dictionaries({"type": st.just("image_url"),
                           "image_url": st.fixed_dictionaries({"url": _image_url})}),
    min_size=1, max_size=3).map(
        lambda parts: {"role": "user", "content": [{"type": "text", "text": "what is this"}, *parts]})


@given(messages=st.lists(_image_message, min_size=1, max_size=2),
       extra=st.dictionaries(st.sampled_from(["max_tokens", "temperature", "stream"]), _json,
                             max_size=2))
def test_image_bodies_never_answer_5xx(messages, extra, vision_client):
    body = json.dumps({**_VALID_CHAT, "messages": messages, **extra}).encode()
    resp = _post("POST", "/v1/chat/completions", body, vision_client)
    assert resp.status_code < 500, (resp.status_code, resp.text[:300], body[:300])


def test_an_image_body_reaches_the_decoder_only_through_a_decoding_engine(monkeypatch):
    from localm.inference import media

    calls = []
    real = media.decode_image_url
    monkeypatch.setattr(media, "decode_image_url", lambda url: calls.append(url) or real(url))
    body = json.dumps({**_VALID_CHAT, "messages": [{
        "role": "user", "content": [{"type": "image_url",
                                     "image_url": {"url": _uri(_png(2, 2, 2))}}]}]}).encode()
    try:
        _post("POST", "/v1/chat/completions", body, _new_client(_engine()))
        assert calls == []
        vision = _post("POST", "/v1/chat/completions", body, _new_client(_vision_engine()))
        assert vision.status_code < 400 and len(calls) == 1
    finally:
        _client_cache.pop("c", None)


def test_a_pixel_bomb_is_refused_cleanly_by_a_vision_engine(vision_client):
    body = json.dumps({**_VALID_CHAT, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _uri(_png(30_000, 30_000, 1))}}]}]}).encode()
    resp = _post("POST", "/v1/chat/completions", body, vision_client)
    assert 400 <= resp.status_code < 500, (resp.status_code, resp.text[:300])


def test_the_embeddings_strategy_draws_batches_the_json_strategy_cannot():
    from hypothesis import find, settings
    from hypothesis.errors import NoSuchExample

    find(_BATCH, lambda v: len(v) > 100, settings=settings(max_examples=500, database=None))
    with pytest.raises(NoSuchExample):
        find(_json, lambda v: isinstance(v, list) and len(v) > 4,
             settings=settings(max_examples=500, database=None))
