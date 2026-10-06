# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-side conversation compaction: the summariser runs with the reasoning
channel off, a reasoning-only reply never drops the history, the forwarded
history alternates roles and keeps the latest user turn, and a client
disconnect stops the summariser."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

import localm.inference.http_server as hs
from localm.inference.backends.base import no_think_prompt


_THINK_ONLY = "<think>\nThinking Process: drafting a summary of the"

_THREAD = [
    {"role": "user", "content": "tell me about tidal locking"},
    {"role": "assistant", "content": "Tidal locking is when a body always shows one face."},
    {"role": "user", "content": "and the moon?"},
    {"role": "assistant", "content": "The moon is tidally locked to Earth."},
    {"role": "user", "content": "expand on that"},
    {"role": "assistant", "content": "I think there's depth here worth unpacking."},
    {"role": "user", "content": "what were we discussing?"},
]


# --------------------------------------------------------------------------- #
#  no_think_prompt                                                             #
# --------------------------------------------------------------------------- #

class TestNoThinkPrompt:
    def test_appends_an_empty_block_for_a_think_template(self):
        p = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        out = no_think_prompt(p, "{% if x %}<think>{% endif %}")
        assert out == p + "<think>\n\n</think>\n\n"

    def test_closes_an_open_think_generation_prefix(self):
        p = "<|User|>hi<|Assistant|><think>\n"
        assert no_think_prompt(p, None) == "<|User|>hi<|Assistant|><think>\n\n</think>\n\n"

    def test_leaves_an_already_disabled_prompt_alone(self):
        p = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        assert no_think_prompt(p, "<think>") == p

    def test_leaves_a_model_without_a_think_convention_alone(self):
        p = "<start_of_turn>model\n"
        assert no_think_prompt(p, "{{ messages }}") == p
        assert no_think_prompt(p, None) == p


# --------------------------------------------------------------------------- #
#  Backend plumbing                                                            #
# --------------------------------------------------------------------------- #

def _llama_with_template(monkeypatch, rendered: str, template: str):
    from localm.inference.backends.llamacpp import llama as llama_mod
    monkeypatch.setattr(llama_mod, "_apply_model_template",
                        lambda ptr, msgs: (rendered, None))
    monkeypatch.setattr(llama_mod, "_untrusted_prompt_ranges",
                        lambda *a, **k: ())
    monkeypatch.setattr(llama_mod.api, "llama_model_chat_template",
                        lambda ptr: template)
    llm = object.__new__(llama_mod.LlamaCpp)
    llm._model_ptr = 1
    llm._mtmd = None
    encoded = []

    class _Tok:
        def encode(self, prompt, add_bos=True, untrusted_ranges=()):
            encoded.append(prompt)
            return [1, 2, 3]

    llm._tokenizer = _Tok()
    llm._generate = lambda tokens, **kw: iter(())
    return llm, encoded


class TestGgufPrefill:
    RENDERED = "<|im_start|>user\nsummarise<|im_end|>\n<|im_start|>assistant\n"

    def test_thinking_false_prefills_an_empty_think_block(self, monkeypatch):
        llm, encoded = _llama_with_template(monkeypatch, self.RENDERED, "...<think>...")
        llm.create_chat_completion([{"role": "user", "content": "summarise"}],
                                   stream=True, thinking=False)
        assert encoded == [self.RENDERED + "<think>\n\n</think>\n\n"]

    def test_default_prompt_is_unchanged(self, monkeypatch):
        llm, encoded = _llama_with_template(monkeypatch, self.RENDERED, "...<think>...")
        llm.create_chat_completion([{"role": "user", "content": "summarise"}],
                                   stream=True)
        assert encoded == [self.RENDERED]

    def test_worker_forwards_thinking_to_the_completion_call(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        seen = []

        class _Llm:
            def create_chat_completion(self, **kw):
                seen.append(kw)
                yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

        w = object.__new__(GgufWorker)
        w._llm = _Llm()
        assert list(w.chat_stream([{"role": "user", "content": "x"}], thinking=False)) == ["ok"]
        assert seen[0]["thinking"] is False
        list(w.chat_stream([{"role": "user", "content": "x"}]))
        assert "thinking" not in seen[1]

    def test_gguf_backend_sends_thinking_to_the_worker(self):
        from localm.inference.backends.gguf import GgufBackend
        seen = []

        class _Runner:
            last_done = {"finish_reason": "stop"}

            def chat_stream(self, **kw):
                seen.append(kw)
                yield "ok"

        b = GgufBackend("does-not-exist.gguf", n_ctx=512)
        b._loaded = True
        b._runner = _Runner()
        list(b.chat_stream([{"role": "user", "content": "x"}], thinking=False))
        list(b.chat_stream([{"role": "user", "content": "x"}]))
        assert seen[0]["thinking"] is False
        assert "thinking" not in seen[1]


class TestEngineForwarding:
    def test_engine_forwards_thinking_only_when_set(self):
        from localm.inference.engine import Engine
        seen = []

        class _Backend:
            loaded = True

            def chat_stream(self, messages, **kw):
                seen.append(kw)
                yield "ok"

        eng = object.__new__(Engine)
        eng._backend = _Backend()
        list(eng.chat_stream([{"role": "user", "content": "x"}], thinking=False))
        list(eng.chat_stream([{"role": "user", "content": "x"}]))
        assert seen[0]["thinking"] is False
        assert "thinking" not in seen[1]


class TestHfPrompt:
    class _Tok:
        def __init__(self):
            self.renders = []
            self.calls = []

        def apply_chat_template(self, msgs, tokenize=False,
                                add_generation_prompt=True, **kw):
            self.renders.append(kw)
            body = "".join(f"<u>{m['content']}</u>" for m in msgs)
            tail = "<a><think></think>" if kw.get("enable_thinking") is False else "<a>"
            return body + tail

        def __call__(self, text, **kw):
            self.calls.append((text, kw))

            class _Enc(dict):
                def to(self, device):
                    return self
            return _Enc(input_ids=[ord(c) for c in text])

    def test_spans_are_located_with_the_thinking_kwarg_and_the_suffix_is_trusted(self):
        pytest.importorskip("torch")
        from localm.inference.backends import _hf_worker
        from localm.textguard import compose, untrusted_span
        tok = self._Tok()
        msgs = [{"role": "user", "content": compose("x ", untrusted_span("web"))}]
        text = tok.apply_chat_template(msgs, enable_thinking=False)
        _hf_worker._tokenize_prompt(tok, msgs, text, "cpu",
                                    template_kwargs={"enable_thinking": False},
                                    suffix="SUFFIX")
        assert all(r.get("enable_thinking") is False for r in tok.renders)
        segments = [(t, kw.get("split_special_tokens")) for t, kw in tok.calls]
        assert ("web", True) in segments, segments
        assert "".join(t for t, _ in segments) == text + "SUFFIX"
        assert segments[-1][1] is False

    def test_plain_prompt_tokenises_text_and_suffix(self):
        from localm.inference.backends import _hf_worker
        tok = self._Tok()
        _hf_worker._tokenize_prompt(tok, [{"role": "user", "content": "p"}],
                                    "PROMPT", "cpu", suffix="SUFFIX")
        assert tok.calls[-1][0] == "PROMPTSUFFIX"


# --------------------------------------------------------------------------- #
#  The server compaction helper                                                #
# --------------------------------------------------------------------------- #

class _RecordingEngine:
    display_name = "rec-model"

    def __init__(self, reply=_THINK_ONLY):
        self.calls = []
        self.reply = reply

    def chat_stream(self, messages, **kw):
        self.calls.append((messages, kw))
        yield self.reply


def test_the_summariser_runs_with_thinking_off():
    eng = _RecordingEngine(reply="We discussed tidal locking.")
    out, changed, gone = asyncio.run(hs._compact_for_capacity(eng, list(_THREAD)))
    assert changed is True and gone is False
    (_msgs, kw), = eng.calls
    assert kw["thinking"] is False
    assert kw["max_tokens"] == 1024
    assert out[0]["content"] == "[Conversation summary]\nWe discussed tidal locking."


class _EndlessEngine:
    """A summariser that runs for up to ~6 s unless closed, holding a lock
    while generating."""
    display_name = "endless-model"

    def __init__(self):
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.closed = threading.Event()

    def chat_stream(self, messages, **kw):
        def _gen():
            with self.lock:
                self.entered.set()
                try:
                    for _ in range(600):
                        time.sleep(0.01)
                        yield "t"
                finally:
                    self.closed.set()
        return _gen()


class _Request:
    """A request whose disconnect poll reports gone once *gone* is set."""

    def __init__(self, gone: threading.Event):
        self._gone = gone
        self.scope = {hs._DISCONNECT_POLL_KEY: self._poll}

    async def _poll(self):
        return self._gone.is_set()


def test_a_disconnect_stops_the_summariser():
    async def scenario():
        eng = _EndlessEngine()
        gone = threading.Event()
        task = asyncio.ensure_future(
            hs._compact_for_capacity(eng, list(_THREAD), _Request(gone)))
        for _ in range(300):
            if eng.entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert eng.entered.is_set(), "the summariser never started"
        assert eng.lock.locked()
        gone.set()
        started = time.monotonic()
        out, changed, disconnected = await task
        return eng, out, changed, disconnected, time.monotonic() - started

    eng, out, changed, disconnected, elapsed = asyncio.run(scenario())
    assert disconnected is True
    assert elapsed < 2.0, f"the summariser ran on for {elapsed:.1f}s after the disconnect"
    assert eng.closed.is_set(), "the summariser generation was not closed"
    assert not eng.lock.locked(), "the summariser still holds its lock"


def test_without_a_disconnect_the_summariser_is_not_cancelled():
    eng = _RecordingEngine(reply="Summary.")
    gone = threading.Event()
    out, changed, disconnected = asyncio.run(
        hs._compact_for_capacity(eng, list(_THREAD), _Request(gone)))
    assert disconnected is False
    assert out[0]["content"] == "[Conversation summary]\nSummary."


# --------------------------------------------------------------------------- #
#  Through the chat route                                                      #
# --------------------------------------------------------------------------- #

def _engine(summary_reply: str):
    """An engine whose context is nearly full on the first count, so the
    route compacts; the summariser call is the one with thinking=False."""
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.supports_grammar = True
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    counts = iter([3000])
    engine.count_messages_tokens.side_effect = lambda ms: next(counts, 100)
    engine.count_tokens.return_value = 2
    type(engine).loaded = property(lambda self: True)
    engine.calls = []

    def _chat(messages, **kw):
        engine.calls.append((list(messages), kw))
        reply = summary_reply if kw.get("thinking") is False else "answer"

        def _gen():
            yield reply
        return _gen()

    engine.chat_stream.side_effect = _chat
    return engine


def _post(engine, payload):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app
    with TestClient(create_app(engine), raise_server_exceptions=False) as client:
        return client.post("/v1/chat/completions", json=payload)


@pytest.mark.parametrize("stream", [False, True])
def test_route_reasoning_only_summary_keeps_history_and_alternation(stream):
    engine = _engine(_THINK_ONLY)
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": stream})
    summariser = [c for c in engine.calls if c[1].get("thinking") is False]
    answers = [c for c in engine.calls if c[1].get("thinking") is not False]
    assert len(summariser) == 1
    assert len(answers) == 1
    forwarded = [m for m in answers[0][0] if m["role"] != "system"]
    bridge = str(forwarded[0]["content"])
    assert "condensed to fit the context window" in bridge
    assert "tell me about tidal locking" in bridge
    assert forwarded[-1]["content"] == "what were we discussing?"
    assert {"role": "user", "content": "expand on that"} in [
        {"role": m["role"], "content": str(m["content"])} for m in forwarded]
    roles = [m["role"] for m in forwarded]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    assert r.status_code == 200
    assert r.headers.get("X-Localm-Context-Compacted") == "1"


def test_route_a_disconnect_during_compaction_skips_generation(monkeypatch):
    engine = _engine("unused")

    async def _gone(engine_, messages, request=None):
        return messages, False, True

    monkeypatch.setattr(hs, "_compact_for_capacity", _gone)
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": False})
    assert engine.calls == []
    assert r.status_code == 499


def test_route_passes_enable_thinking_false_to_the_generation():
    engine = _engine("unused")
    engine.context_capacity.return_value = None
    r = _post(engine, {"model": "test-model", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}],
                       "chat_template_kwargs": {"enable_thinking": False}})
    assert engine.calls[-1][1]["thinking"] is False
    assert r.status_code == 200


@pytest.mark.parametrize("kwargs", [{"foo": 1}, {"enable_thinking": "no"}])
def test_route_rejects_an_unsupported_template_kwarg(kwargs):
    engine = _engine("unused")
    r = _post(engine, {"model": "test-model", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}],
                       "chat_template_kwargs": kwargs})
    assert engine.calls == []
    assert r.status_code == 422
