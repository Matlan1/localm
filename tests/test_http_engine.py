# SPDX-License-Identifier: AGPL-3.0-or-later
"""HttpEngine (H3): the Engine-compatible client `localm run` uses to ATTACH to a
running localm server's /v1 API instead of loading a second in-process model copy."""

import json
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.base import UnsupportedInputError
from localm.inference.http_engine import (
    HttpEngine, remote_active_model, remote_model_status)


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


def _sse_reasoning(*deltas):
    """A fake streaming response yielding raw delta dicts verbatim, so tests can
    mix `content` and `reasoning_content` fields per chunk like the real server's
    H4 reasoning/content split does."""
    lines = ["data: " + json.dumps({"choices": [{"delta": d}]}) for d in deltas]
    lines.append("data: [DONE]")
    r = MagicMock()
    r.status_code = 200
    r.iter_lines = lambda decode_unicode=False: iter(lines)
    return r


def test_chat_stream_surfaces_reasoning_content(monkeypatch):
    """The server splits <think> reasoning into its own `reasoning_content` SSE
    field; HttpEngine must re-wrap it in inline <think>...</think> markers (like
    the in-process Engine's raw stream) so cli/chat.py's ThinkSplitter and
    _ThinkPrinter dim it instead of dropping it in localm run's default attach
    mode."""
    deltas = [
        {"reasoning_content": "because "},
        {"reasoning_content": "reasons"},
        {"content": "The "},
        {"content": "answer."},
    ]
    monkeypatch.setattr("requests.post", lambda *a, **k: _sse_reasoning(*deltas))
    out = list(HttpEngine("http://x/v1").chat_stream(
        [{"role": "user", "content": "hi"}]))
    full = "".join(out)
    assert full == "<think>because </think><think>reasons</think>The answer."
    from localm.textnorm import ThinkSplitter
    splitter = ThinkSplitter()
    content, reasoning = "", ""
    for piece in out:
        c, r = splitter.feed(piece)
        content += c
        reasoning += r
    fc, fr = splitter.flush()
    content += fc
    reasoning += fr
    assert content == "The answer."
    assert reasoning == "because reasons"


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


def test_remote_model_status_distinguishes_no_model_from_cannot_read(monkeypatch):
    """`localm run` attaching to a live, model-loaded server must not print 'no
    model loaded': /v1/models needs the models scope and a chat-scoped attach
    token gets a 403, which returning None cannot tell apart from a genuinely
    model-less server. remote_model_status must distinguish 'empty' (server says
    none) from 'unknown' (it could not be read)."""
    def _resp(status, payload):
        m = MagicMock()
        m.status_code = status
        m.json = lambda: payload
        return m

    # A model is loaded -> ("loaded", id)
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: _resp(200, {"data": [{"id": "gemma-4"}]}))
    assert remote_model_status("http://x/v1", "tok") == ("loaded", "gemma-4")

    # Server ANSWERS but has no model -> ("empty", None): the honest "no model" case.
    monkeypatch.setattr("requests.get", lambda *a, **k: _resp(200, {"data": []}))
    assert remote_model_status("http://x/v1") == ("empty", None)

    # 403 (attach token lacks the models scope) -> ("unknown", None): NOT "no model".
    monkeypatch.setattr("requests.get", lambda *a, **k: _resp(403, {"detail": "scope"}))
    assert remote_model_status("http://x/v1", "chat-scoped") == ("unknown", None)

    # Unreachable -> ("unknown", None), never a false "no model".
    def boom(*a, **k):
        raise Exception("down")

    monkeypatch.setattr("requests.get", boom)
    assert remote_model_status("http://x/v1") == ("unknown", None)


def test_context_capacity_fetches_and_caches(monkeypatch):
    eng = HttpEngine("http://x/v1", token="t", model="m")
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        r = MagicMock()
        r.status_code = 200
        r.json = lambda: {"effective_ctx_max": 8192}
        return r

    monkeypatch.setattr("requests.get", fake_get)
    # First fetch: calls requests.get
    assert eng.context_capacity() == 8192
    assert len(calls) == 1
    assert calls[0][0] == "http://x/v1/config"
    assert calls[0][1]["Authorization"] == "Bearer t"

    # Second fetch: uses cached value
    assert eng.context_capacity() == 8192
    assert len(calls) == 1


def test_context_capacity_degrades_gracefully_on_error(monkeypatch):
    eng = HttpEngine("http://x/v1", token="t", model="m")

    # Server error
    r = MagicMock()
    r.status_code = 500
    monkeypatch.setattr("requests.get", lambda *a, **k: r)
    assert eng.context_capacity() is None

    # Reset cache
    eng._ctx_capacity_cached = False
    # Network exception
    def boom(*a, **k):
        raise Exception("refused")
    monkeypatch.setattr("requests.get", boom)
    assert eng.context_capacity() is None

