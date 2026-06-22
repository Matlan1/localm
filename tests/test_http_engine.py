# SPDX-License-Identifier: AGPL-3.0-or-later
"""HttpEngine (H3): the Engine-compatible client `localm run` uses to ATTACH to a
running localm server's /v1 API instead of loading a second in-process model copy."""

import json
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.base import UnsupportedInputError
from localm.inference.http_engine import HttpEngine, remote_active_model


def _sse(*chunks):
    """A fake streaming response yielding OpenAI-style SSE content deltas."""
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
             for c in chunks]
    lines.append("data: [DONE]")
    r = MagicMock()
    r.status_code = 200
    r.iter_lines = lambda decode_unicode=False: iter(lines)
    return r


def test_chat_stream_yields_tokens(monkeypatch):
    eng = HttpEngine("http://x/v1", token="t", model="m")
    captured = {}

    def fake_post(url, json=None, headers=None, stream=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _sse("Hel", "lo", " world")

    monkeypatch.setattr("requests.post", fake_post)
    out = list(eng.chat_stream([{"role": "user", "content": "hi"}],
                               max_tokens=16, temperature=0.5))
    assert out == ["Hel", "lo", " world"]
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["stream"] is True
    assert captured["json"]["model"] == "m"
    assert captured["json"]["max_tokens"] == 16 and captured["json"]["temperature"] == 0.5
    assert "top_k" not in captured["json"]              # unset params are omitted
    assert captured["headers"]["Authorization"] == "Bearer t"


def test_chat_stream_ignores_non_sse(monkeypatch):
    # NEGATIVE: a non-SSE / non-JSON body yields no tokens and does not crash.
    r = MagicMock()
    r.status_code = 200
    r.iter_lines = lambda decode_unicode=False: iter(["<html>no</html>", "", "garbage"])
    monkeypatch.setattr("requests.post", lambda *a, **k: r)
    assert list(HttpEngine("http://x/v1").chat_stream(
        [{"role": "user", "content": "hi"}])) == []


def test_image_400_raises_unsupported(monkeypatch):
    r = MagicMock()
    r.status_code = 400
    r.json = lambda: {"detail": "This model cannot accept image input (text-only)."}
    monkeypatch.setattr("requests.post", lambda *a, **k: r)
    with pytest.raises(UnsupportedInputError):
        list(HttpEngine("http://x/v1").chat_stream([{"role": "user", "content": "x"}]))


def test_server_error_raises_runtime(monkeypatch):
    r = MagicMock()
    r.status_code = 503
    r.json = lambda: {"detail": "no model loaded"}
    monkeypatch.setattr("requests.post", lambda *a, **k: r)
    with pytest.raises(RuntimeError) as ei:
        list(HttpEngine("http://x/v1").chat_stream([{"role": "user", "content": "x"}]))
    assert "no model loaded" in str(ei.value)


def test_unreachable_raises_runtime(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.RequestException("refused")

    monkeypatch.setattr("requests.post", boom)
    with pytest.raises(RuntimeError) as ei:
        list(HttpEngine("http://x/v1").chat_stream([{"role": "user", "content": "x"}]))
    assert "Could not reach" in str(ei.value)


def test_engine_surface_for_the_repl():
    eng = HttpEngine("http://x/v1", model="m", display_name="my-model")
    assert eng.display_name == "my-model"
    assert eng.loaded is True
    eng.load()
    eng.unload()                                 # no-ops, must not raise
    with eng as e:
        assert e is eng                          # context manager
    assert eng.count_tokens("abcdabcdabcd") >= 1
    assert eng.count_tokens("") >= 1             # never zero/negative


def test_remote_active_model(monkeypatch):
    ok = MagicMock()
    ok.status_code = 200
    ok.json = lambda: {"object": "list", "data": [{"id": "gemma-4", "loaded": True}]}
    monkeypatch.setattr("requests.get", lambda *a, **k: ok)
    assert remote_active_model("http://x/v1", "tok") == "gemma-4"

    empty = MagicMock()
    empty.status_code = 200
    empty.json = lambda: {"data": []}
    monkeypatch.setattr("requests.get", lambda *a, **k: empty)
    assert remote_active_model("http://x/v1") is None              # no model loaded

    def boom(*a, **k):
        raise Exception("down")

    monkeypatch.setattr("requests.get", boom)
    assert remote_active_model("http://x/v1") is None              # unreachable
