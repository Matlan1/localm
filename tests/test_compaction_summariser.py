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

    HARMONY = ("<|start|>user<|message|>hi<|end|>"
               "<|start|>assistant")

    @pytest.mark.parametrize("template", [None, "<|start|>...<|channel|>"])
    def test_harmony_prompt_goes_straight_to_the_final_channel(self, template):
        assert (no_think_prompt(self.HARMONY, template)
                == self.HARMONY + "<|channel|>final<|message|>")

    def test_harmony_prompt_already_in_the_final_channel_is_unchanged(self):
        p = self.HARMONY + "<|channel|>final<|message|>"
        assert no_think_prompt(p, None) == p

    def test_harmony_prompt_mid_conversation_is_not_prefilled(self):
        p = self.HARMONY + "<|channel|>final<|message|>hello<|end|>"
        assert no_think_prompt(p, None) == p

    GEMMA4 = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
    GEMMA4_TEMPLATE = "{{- '<|channel>thought\\n' -}}"

    def test_gemma4_open_thought_channel_is_closed_empty(self):
        p = self.GEMMA4 + "<|channel>thought\n"
        assert no_think_prompt(p, None) == p + "<channel|>"

    def test_gemma4_prompt_with_an_empty_thought_is_unchanged(self):
        p = self.GEMMA4 + "<|channel>thought\n<channel|>"
        assert no_think_prompt(p, self.GEMMA4_TEMPLATE) == p

    def test_gemma4_model_turn_is_not_prefilled(self):
        assert no_think_prompt(self.GEMMA4, self.GEMMA4_TEMPLATE) == self.GEMMA4
        assert no_think_prompt(self.GEMMA4, None) == self.GEMMA4


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

    def test_thinking_false_sends_a_harmony_prompt_to_the_final_channel(
            self, monkeypatch):
        rendered = "<|start|>user<|message|>summarise<|end|><|start|>assistant"
        llm, encoded = _llama_with_template(
            monkeypatch, rendered, "<|start|>{{ x }}<|channel|>")
        llm.create_chat_completion([{"role": "user", "content": "summarise"}],
                                   stream=True, thinking=False)
        assert encoded == [rendered + "<|channel|>final<|message|>"]

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
                                    prompt=text + "SUFFIX")
        assert all(r.get("enable_thinking") is False for r in tok.renders)
        segments = [(t, kw.get("split_special_tokens")) for t, kw in tok.calls]
        assert ("web", True) in segments, segments
        assert "".join(t for t, _ in segments) == text + "SUFFIX"
        assert segments[-1][1] is False

    def test_plain_prompt_tokenises_the_final_prompt(self):
        from localm.inference.backends import _hf_worker
        tok = self._Tok()
        _hf_worker._tokenize_prompt(tok, [{"role": "user", "content": "p"}],
                                    "PROMPT", "cpu", prompt="PROMPTSUFFIX")
        assert tok.calls[-1][0] == "PROMPTSUFFIX"

    def test_fallback_after_a_span_tokenisation_failure_adds_the_suffix_once(self):
        pytest.importorskip("torch")
        from localm.inference.backends import _hf_worker
        from localm.textguard import compose, untrusted_span

        class _Failing(self._Tok):
            def __call__(self, text, **kw):
                if "split_special_tokens" in kw:
                    raise ValueError("knob unsupported")
                return super().__call__(text, **kw)

        tok = _Failing()
        msgs = [{"role": "user", "content": compose("x ", untrusted_span("web"))}]
        text = tok.apply_chat_template(msgs, enable_thinking=False)
        _hf_worker._tokenize_prompt(tok, msgs, text, "cpu",
                                    template_kwargs={"enable_thinking": False},
                                    prompt=text + "SUFFIX")
        assert tok.calls[-1][0] == text + "SUFFIX"


# --------------------------------------------------------------------------- #
#  The server compaction helper                                                #
# --------------------------------------------------------------------------- #

class _RecordingEngine:
    display_name = "rec-model"

    def __init__(self, reply=_THINK_ONLY):
        self.calls = []
        self.reply = reply

    def context_capacity(self):
        return None

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def chat_stream(self, messages, **kw):
        self.calls.append((messages, kw))
        yield self.reply


def test_the_summariser_runs_with_thinking_off():
    eng = _RecordingEngine(reply="We discussed tidal locking.")
    out, changed, gone = asyncio.run(hs._compact_for_capacity(eng, list(_THREAD)))
    assert changed is True and gone is False
    (_msgs, kw), = eng.calls
    assert kw["thinking"] is False
    assert kw["max_tokens"] == 512
    assert out[0]["content"] == "[Conversation summary]\nWe discussed tidal locking."


_TOMATO = ("my tomato variety is called Purple Cherokee and I bought the seeds "
           "in Lisbon")


class _WindowedEngine:
    """An engine with a *capacity*-token window whose summariser refuses a
    prompt that, with max_tokens, does not fit it (one token per three
    characters), and otherwise replies with the facts it can see."""
    display_name = "windowed-model"

    def __init__(self, capacity):
        self.capacity = capacity
        self.calls = []

    def context_capacity(self):
        return self.capacity

    def count_tokens(self, text):
        return len(text) // 3

    def chat_stream(self, messages, **kw):
        prompt = messages[0]["content"]
        self.calls.append((prompt, kw))
        used = len(prompt) // 3
        if used + kw["max_tokens"] > self.capacity:
            raise RuntimeError(f"Prompt ({used} tokens) exceeds the model's maximum "
                               f"context capacity ({self.capacity} tokens)")
        facts = _TOMATO if _TOMATO in prompt else "none"
        yield f"Pass {len(self.calls)}. Facts: {facts}."


def _long_thread(n=30, chars=600):
    msgs = []
    for i in range(n):
        text = f"Turn {i}: {_TOMATO}. " if i == 2 else f"Turn {i}: "
        w = 0
        while len(text) < chars:
            text += f"filler{i}x{w} "
            w += 1
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return msgs


def test_the_server_summariser_fits_the_engine_window_in_rolling_passes():
    eng = _WindowedEngine(2048)
    out, changed, gone = asyncio.run(hs._compact_for_capacity(eng, _long_thread()))
    assert changed is True and gone is False
    assert len(eng.calls) > 1
    for prompt, kw in eng.calls:
        assert len(prompt) // 3 + kw["max_tokens"] <= 2048 - 64
        assert kw["thinking"] is False
    bridge = str(out[0]["content"])
    assert bridge.startswith("[Conversation summary]\n"), bridge[:80]
    assert _TOMATO in bridge


def test_a_refused_count_is_budgeted_with_the_estimate():
    from localm.inference.backends.base import PretokenizerUnsafeInputError

    class _Refusing(_WindowedEngine):
        def count_tokens(self, text):
            raise PretokenizerUnsafeInputError("refused")

    eng = _Refusing(2048)
    out, changed, _gone = asyncio.run(hs._compact_for_capacity(eng, _long_thread()))
    assert changed is True
    assert len(eng.calls) > 1
    for prompt, kw in eng.calls:
        assert len(prompt) // 3 + kw["max_tokens"] <= 2048 - 64
    assert str(out[0]["content"]).startswith("[Conversation summary]\n")
    assert _TOMATO in str(out[0]["content"])


def test_the_server_summariser_makes_one_pass_in_a_large_window():
    eng = _WindowedEngine(32768)
    out, changed, _gone = asyncio.run(hs._compact_for_capacity(eng, _long_thread()))
    assert changed is True
    assert len(eng.calls) == 1
    assert eng.calls[0][1]["max_tokens"] == 512


class _EndlessEngine:
    """A summariser that runs for up to ~6 s unless closed, holding a lock
    while generating."""
    display_name = "endless-model"

    def __init__(self):
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.closed = threading.Event()

    def context_capacity(self):
        return None

    def count_tokens(self, text):
        return max(1, len(text) // 4)

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
def test_route_reasoning_only_summary_keeps_history_and_alternation(stream, monkeypatch):
    monkeypatch.setattr(hs, "PREP_STATUS_GRACE_S", 30.0)
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
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles
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


@pytest.mark.parametrize("value", ["no", 1, [False]])
def test_route_rejects_a_non_boolean_enable_thinking(value):
    engine = _engine("unused")
    r = _post(engine, {"model": "test-model", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}],
                       "chat_template_kwargs": {"enable_thinking": value}})
    assert engine.calls == []
    assert r.status_code == 422


@pytest.mark.parametrize("kwargs", [{"reasoning_effort": "low"},
                                    {"enable_thinking": None, "foo": 1}])
def test_route_ignores_other_template_kwargs(kwargs):
    engine = _engine("unused")
    engine.context_capacity.return_value = None
    r = _post(engine, {"model": "test-model", "stream": False,
                       "messages": [{"role": "user", "content": "hi"}],
                       "chat_template_kwargs": kwargs})
    assert "thinking" not in engine.calls[-1][1]
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
#  The stream says it is compacting                                            #
# --------------------------------------------------------------------------- #

def _sse_deltas(text: str) -> list:
    import json
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            obj = json.loads(line[len("data: "):])
            for ch in obj.get("choices", []):
                out.append((ch.get("delta") or {}, ch.get("finish_reason")))
    return out


def test_stream_reports_compacting_before_the_reply(monkeypatch):
    monkeypatch.setattr(hs, "PREP_STATUS_GRACE_S", 30.0)
    engine = _engine("We discussed tidal locking.")
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": True})
    deltas = _sse_deltas(r.text)
    codes = [d.get("status_code") for d, _ in deltas if d.get("status")]
    first_content = next(i for i, (d, _) in enumerate(deltas) if d.get("content"))
    compacting_at = next(i for i, (d, _) in enumerate(deltas)
                         if d.get("status_code") == "compacting")
    assert compacting_at < first_content
    assert codes[0] == "compacting", codes
    assert deltas[compacting_at][0]["status"] == "Compacting conversation..."
    summariser = [c for c in engine.calls if c[1].get("thinking") is False]
    assert len(summariser) == 1
    assert r.headers.get("X-Localm-Context-Compacted") == "1"
    assert r.status_code == 200


def test_stream_without_compaction_has_no_compacting_status():
    engine = _engine("unused")
    engine.context_capacity.return_value = 100000
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": True})
    codes = [d.get("status_code") for d, _ in _sse_deltas(r.text) if d.get("status")]
    assert "compacting" not in codes
    assert "X-Localm-Context-Compacted" not in r.headers


def test_stream_overflow_after_compaction_is_reported_in_the_stream():
    engine = _engine("Summary.")
    engine.count_messages_tokens.side_effect = lambda ms: 5000
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": True})
    deltas = _sse_deltas(r.text)
    text = "".join(d.get("content") or "" for d, _ in deltas)
    answers = [c for c in engine.calls if c[1].get("thinking") is not False]
    assert answers == []
    assert "exceeds the model's maximum context capacity (4096 tokens)" in text
    assert deltas[-1][1] == "error"
    assert r.status_code == 200


def test_stream_recount_refusal_after_compaction_is_reported_in_the_stream():
    from localm.inference.backends.base import PretokenizerUnsafeInputError
    engine = _engine("Summary.")
    counts = iter([3000])

    def _count(ms):
        try:
            return next(counts)
        except StopIteration as e:
            raise PretokenizerUnsafeInputError("unbroken run of 9000 characters") from e

    engine.count_messages_tokens.side_effect = _count
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": True})
    deltas = _sse_deltas(r.text)
    text = "".join(d.get("content") or "" for d, _ in deltas)
    assert "unbroken run" in text
    assert deltas[-1][1] == "error"
    assert [c for c in engine.calls if c[1].get("thinking") is not False] == []


def test_non_stream_overflow_is_still_413():
    engine = _engine("Summary.")
    engine.count_messages_tokens.side_effect = lambda ms: 5000
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": False})
    assert "exceeds the model's maximum context capacity" in r.json()["detail"]
    assert r.status_code == 413


def test_maybe_compact_announces_only_a_real_compaction():
    from localm.inference.compact import maybe_compact
    calls = []
    long = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400}
            for i in range(12)]
    out, changed = maybe_compact(long, limit_tokens=1000,
                                 generate=lambda m, t: "S",
                                 on_compact=lambda: calls.append(1))
    assert changed is True and calls == [1]
    calls.clear()
    maybe_compact(long, limit_tokens=100000, generate=lambda m, t: "S",
                  on_compact=lambda: calls.append(1))
    assert calls == []
    huge_but_short = [{"role": "user", "content": "x" * 40000}]
    out, changed = maybe_compact(huge_but_short, limit_tokens=1000,
                                 generate=lambda m, t: "S",
                                 on_compact=lambda: calls.append(1))
    assert changed is False
    assert calls == []


def test_stream_recount_fault_after_compaction_ends_the_stream_with_an_error():
    engine = _engine("Summary.")
    counts = iter([3000])

    def _count(ms):
        try:
            return next(counts)
        except StopIteration as e:
            raise RuntimeError("worker died during the token count") from e

    engine.count_messages_tokens.side_effect = _count
    r = _post(engine, {"model": "test-model", "messages": _THREAD, "stream": True})
    deltas = _sse_deltas(r.text)
    text = "".join(d.get("content") or "" for d, _ in deltas)
    assert "worker died during the token count" in text
    assert deltas[-1][1] == "error"
    assert r.text.rstrip().endswith("data: [DONE]")
    assert [c for c in engine.calls if c[1].get("thinking") is not False] == []
