# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the ``/v1/*`` request bodies through the real ASGI app.

A request body is the one input a network client controls end to end. The
contract: a body the server cannot use is answered with a 4xx and a JSON
error, never a 5xx, never a hang, whatever the JSON shape, encoding or depth.
The engine behind the app is a stub, so this exercises the routes' parsing and
validation, which is where the body is first trusted."""
from __future__ import annotations

import json
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


def _engine() -> MagicMock:
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
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


_client_cache: dict = {}


def _client() -> TestClient:
    from localm.inference import http_server
    if "c" not in _client_cache:
        engine = _engine()
        _client_cache["c"] = TestClient(create_app(engine), raise_server_exceptions=False)
        _client_cache["engine"] = engine
    http_server._engine = _client_cache["engine"]
    return _client_cache["c"]


def _post(method: str, path: str, body: bytes):
    return _bounds.returns_within(
        _client().request, method, path, content=body,
        headers={"content-type": "application/json"}, seconds=30)


@pytest.mark.parametrize("path,fields,base", [
    ("/v1/chat/completions", _CHAT_FIELDS, _VALID_CHAT),
    ("/v1/completions", {k: _json for k in (
        "model", "prompt", "stream", "max_tokens", "temperature", "top_p", "top_k",
        "repeat_penalty", "grammar", "grammar_lazy", "grammar_triggers", "seed")},
     _VALID_COMPLETION),
    ("/v1/embeddings", {k: _json for k in ("model", "input", "encoding_format")},
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
