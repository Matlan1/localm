# SPDX-License-Identifier: AGPL-3.0-or-later
"""``stop`` on /v1/chat/completions and /v1/completions, streaming and not:
the reply is cut before the first match, the generation ends there instead of
running to its budget, and the exchange is recorded as a normal completion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

MODEL = "stop-model"


class _Source:
    """An engine whose token stream is slow enough that stopping early is
    observable: it records how many tokens were produced and whether the
    generator was closed."""

    def __init__(self, tokens, delay=0.05):
        self.tokens = list(tokens)
        self.delay = delay
        self.produced = 0
        self.closed = False
        self.calls = []

    def chat_stream(self, messages, **kwargs):
        self.calls.append(kwargs)
        try:
            for token in self.tokens:
                self.produced += 1
                yield token
                time.sleep(self.delay)
        finally:
            self.closed = True


def _engine(source):
    engine = MagicMock()
    engine.chat_stream.side_effect = source.chat_stream
    engine.display_name = MODEL
    engine.model_path = ""
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.gpu_placement = None
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    engine.loaded = True
    return engine


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".localm"
    root.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(root))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", root)
    monkeypatch.setattr(_cfg, "MODELS_DIR", root / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", root / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", root / "registry.json")
    return root


def _client(source, home):
    app = create_app(_engine(source))
    return TestClient(app)


LONG = ["Hello", " wor", "ld", " and", " more"] + [" tail"] * 40


def _chat(client, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/chat/completions", json=body)


def _complete(client, **extra):
    body = {"model": MODEL, "prompt": "hi", **extra}
    return client.post("/v1/completions", json=body)


def _sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def _chat_text(chunks, field="content"):
    return "".join(c["choices"][0]["delta"].get(field) or "" for c in chunks if c.get("choices"))


def _finish(chunks):
    return [c["choices"][0]["finish_reason"] for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason")]


# ------------------------------------------------------------------ chat, streaming


def test_stream_cuts_across_tokens_ends_the_generation_and_reports_stop(home):
    source = _Source(LONG)
    with _client(source, home) as c:
        chunks = _sse(_chat(c, stream=True, stop="world"))
    assert _chat_text(chunks) == "Hello "
    assert _finish(chunks) == ["stop"]
    assert source.closed is True
    assert source.produced < 12, "the generation must stop near the stop sequence"
    usage = [c["usage"] for c in chunks if c.get("usage")][0]
    assert usage["completion_tokens"] > 0


def test_stream_stop_is_not_passed_to_the_engine(home):
    source = _Source(["a", "b"], delay=0)
    with _client(source, home) as c:
        _chat(c, stream=True, stop=["zzz"])
    assert source.calls and "stop" not in source.calls[0]


def test_stream_without_a_match_streams_everything_and_releases_the_held_tail(home):
    source = _Source(["ab<", "/s"], delay=0)
    with _client(source, home) as c:
        chunks = _sse(_chat(c, stream=True, stop="</s>x"))
    assert _chat_text(chunks) == "ab</s"
    assert _finish(chunks) == ["stop"]


def test_stream_picks_the_earliest_of_several_stops(home):
    source = _Source(["one two three"], delay=0)
    with _client(source, home) as c:
        chunks = _sse(_chat(c, stream=True, stop=["three", "two"]))
    assert _chat_text(chunks) == "one "


def test_a_stop_inside_the_reasoning_block_is_not_applied(home):
    source = _Source(["<think>say STOP here</think>", "answer STOP more"], delay=0)
    with _client(source, home) as c:
        chunks = _sse(_chat(c, stream=True, stop="STOP"))
    assert _chat_text(chunks, "reasoning_content") == "say STOP here"
    assert _chat_text(chunks) == "answer "


def test_a_stream_cut_by_stop_is_recorded_as_a_normal_completion(home):
    source = _Source(LONG)
    with patch("localm.inference.http_server._audit_exchange") as audit, \
            _client(source, home) as c:
        assert _chat(c, stream=True, stop="world").status_code == 200
    audit.assert_called_once()
    args, kwargs = audit.call_args
    assert args[3] == "Hello "
    assert kwargs["outcome"] == "success"


# ------------------------------------------------------------------ chat, not streaming


def test_non_stream_cuts_the_reply_and_ends_the_generation(home):
    source = _Source(LONG)
    with _client(source, home) as c:
        body = _chat(c, stop=["world"]).json()
    choice = body["choices"][0]
    assert choice["message"]["content"] == "Hello "
    assert choice["finish_reason"] == "stop"
    assert source.closed is True
    assert source.produced < 12
    assert body["usage"]["completion_tokens"] > 0


def test_non_stream_keeps_reasoning_and_cuts_only_the_answer(home):
    source = _Source(["<think>say STOP here</think>", "answer STOP more"], delay=0)
    with _client(source, home) as c:
        message = _chat(c, stop="STOP").json()["choices"][0]["message"]
    assert message["reasoning_content"] == "say STOP here"
    assert message["content"] == "answer "


def test_non_stream_without_a_match_is_untouched(home):
    source = _Source(["Hello", " there"], delay=0)
    with _client(source, home) as c:
        body = _chat(c, stop="zzz").json()
    assert body["choices"][0]["message"]["content"] == "Hello there"


def test_non_stream_cut_is_recorded_as_a_normal_completion(home):
    source = _Source(LONG)
    with patch("localm.inference.http_server._audit_exchange") as audit, \
            _client(source, home) as c:
        _chat(c, stop="world")
    args, kwargs = audit.call_args
    assert args[3] == "Hello " and kwargs["outcome"] == "success"


@pytest.mark.parametrize("stream", [True, False])
def test_a_stop_hit_reports_stop_and_a_miss_keeps_the_engines_reason(home, stream):
    def finish_of(stop):
        engine = _engine(_Source(LONG, delay=0))
        engine.last_finish_reason = "length"
        with TestClient(create_app(engine)) as c:
            r = _chat(c, stream=stream, stop=stop)
        return _finish(_sse(r)) if stream else [r.json()["choices"][0]["finish_reason"]]

    assert finish_of("world") == ["stop"]
    assert finish_of("zzz") == ["length"]


def test_a_bad_stop_is_a_422(home):
    source = _Source(["a"], delay=0)
    with _client(source, home) as c:
        assert _chat(c, stop=5).status_code == 422
        assert _chat(c, stop=["x"] * 17).status_code == 422
        assert _chat(c, stop=[1]).status_code == 422


# ------------------------------------------------------------------ completions


def test_completions_stream_cut(home):
    source = _Source(LONG)
    with _client(source, home) as c:
        chunks = _sse(_complete(c, stream=True, stop="world"))
    text = "".join(ch["choices"][0]["text"] for ch in chunks)
    assert text == "Hello "
    assert source.closed is True and source.produced < 12
    assert "stop" not in source.calls[0]


def test_completions_stream_releases_a_held_tail(home):
    source = _Source(["ab<"], delay=0)
    with _client(source, home) as c:
        chunks = _sse(_complete(c, stream=True, stop="</s>"))
    assert "".join(ch["choices"][0]["text"] for ch in chunks) == "ab<"


def test_completions_non_stream_cut(home):
    source = _Source(LONG)
    with _client(source, home) as c:
        body = _complete(c, stop=["world"]).json()
    assert body["choices"][0]["text"] == "Hello "
    assert source.closed is True and source.produced < 12
