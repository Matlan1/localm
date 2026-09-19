# SPDX-License-Identifier: AGPL-3.0-or-later
"""The terminal outcome of a chat turn reaches the hook pipeline and the audit
record instead of being flattened into "a completed turn".

Every path (chat stream, raw completions stream, non-streaming) sets
ChatHookContext.outcome before the outlet phase; an "error" turn skips the
outlet entirely; a client disconnect leaves "abort" on the context; success and
length both run the outlet exactly once, distinguishably. The memory outlet
refuses to schedule consolidation for a failed turn, and the audit/transcript
record carries the non-success outcome.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.chat_pipeline import ChatHookContext
from localm.inference.http_server import _audit_exchange, _stream_sse, create_app


def _engine(tokens=("part", "ial"), reason="stop", raise_after=False):
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        for t in tokens:
            yield t
        if raise_after:
            raise RuntimeError("decode failed mid-turn")

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = len(tokens)
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = reason
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: True)
    return engine


def _hooked_app(engine):
    """create_app plus recording inlet/outlet hooks on the real pipeline."""
    app = create_app(engine)
    seen = {"ctx": [], "outlet": []}

    def _inlet(messages, ctx):
        seen["ctx"].append(ctx)
        return messages

    def _outlet(text, messages, ctx):
        seen["outlet"].append((text, ctx.outcome))
        return text

    app.state.chat_pipeline.add_hook("inlet", _inlet, plugin="t")
    app.state.chat_pipeline.add_hook("outlet", _outlet, plugin="t")
    return app, seen


def _terminal_reason(sse_text):
    reasons = []
    for line in sse_text.splitlines():
        payload = line.strip()[len("data:"):].strip() if line.strip().startswith("data:") else ""
        if not payload or payload == "[DONE]":
            continue
        for ch in json.loads(payload).get("choices", []):
            if ch.get("finish_reason") is not None:
                reasons.append(ch["finish_reason"])
    return reasons[-1] if reasons else None


# --------------------------------------------------------------------------- #
#  chat streaming path                                                        #
# --------------------------------------------------------------------------- #

def test_chat_stream_error_skips_outlet_and_marks_ctx_error():
    app, seen = _hooked_app(_engine(raise_after=True))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "[inference error" in r.text
    assert _terminal_reason(r.text) == "error"
    assert seen["ctx"][0].outcome == "error"
    assert seen["outlet"] == [], "the outlet ran for a failed generation"


def test_chat_stream_success_runs_outlet_once_with_success():
    app, seen = _hooked_app(_engine())
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    assert _terminal_reason(r.text) == "stop"
    assert seen["outlet"] == [("partial", "success")]
    assert seen["ctx"][0].outcome == "success"


def test_chat_stream_length_runs_outlet_with_length():
    app, seen = _hooked_app(_engine(reason="length"))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    assert _terminal_reason(r.text) == "length"
    assert seen["outlet"] == [("partial", "length")]


def test_chat_stream_client_abort_marks_ctx_abort_and_skips_outlet():
    engine = _engine(tokens=("a", "b", "c"))
    outlet_calls = []
    from localm.inference.chat_pipeline import ChatPipeline
    pipeline = ChatPipeline()
    pipeline.add_hook("outlet", lambda t, m, c: outlet_calls.append(c.outcome) or t)
    ctx = ChatHookContext(model_id="m", stream=True, request_id="r")

    async def _drive():
        gen = _stream_sse(engine, [{"role": "user", "content": "hi"}], "m",
                          asyncio.Semaphore(1), pipeline=pipeline, ctx=ctx,
                          prompt_tokens=1)
        # Consume the role chunk, the status chunk, then one content chunk, then disconnect.
        await gen.__anext__()   # role chunk
        await gen.__anext__()   # status chunk ("Processing prompt...")
        await gen.__anext__()   # first content chunk
        await gen.aclose()

    asyncio.run(_drive())
    assert ctx.outcome == "abort"
    assert outlet_calls == [], "the outlet ran for a turn the client abandoned"


# --------------------------------------------------------------------------- #
#  raw completions streaming path                                             #
# --------------------------------------------------------------------------- #

def test_completions_stream_error_skips_outlet_and_marks_ctx_error():
    app, seen = _hooked_app(_engine(raise_after=True))
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": True})
    assert r.status_code == 200
    assert "[inference error" in r.text
    assert _terminal_reason(r.text) == "error"
    assert seen["ctx"][0].outcome == "error"
    assert seen["outlet"] == []


def test_completions_stream_success_runs_outlet_once():
    app, seen = _hooked_app(_engine())
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": True})
    assert _terminal_reason(r.text) == "stop"
    assert seen["outlet"] == [("partial", "success")]


def test_completions_nonstream_error_skips_outlet_and_marks_ctx_error():
    app, seen = _hooked_app(_engine(raise_after=True))
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": False})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "error"
    assert "[inference error" in choice["text"]
    assert seen["ctx"][0].outcome == "error"
    assert seen["outlet"] == []


def test_completions_nonstream_success_runs_outlet_once():
    app, seen = _hooked_app(_engine())
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": False})
    assert r.json()["choices"][0]["finish_reason"] == "stop"
    assert seen["outlet"] == [("partial", "success")]


# --------------------------------------------------------------------------- #
#  non-streaming path                                                         #
# --------------------------------------------------------------------------- #

def test_nonstream_error_skips_outlet_and_marks_ctx_error():
    app, seen = _hooked_app(_engine(raise_after=True))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": False,
            "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "error"
    assert "[inference error" in choice["message"]["content"]
    assert seen["ctx"][0].outcome == "error"
    assert seen["outlet"] == []


def test_nonstream_length_marks_ctx_length_and_runs_outlet():
    app, seen = _hooked_app(_engine(reason="length"))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": False,
            "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["finish_reason"] == "length"
    assert seen["outlet"] == [("partial", "length")]


# --------------------------------------------------------------------------- #
#  audit / transcript record                                                  #
# --------------------------------------------------------------------------- #

def test_audit_records_non_success_outcome():
    audit = MagicMock()
    transcript = MagicMock()
    msgs = [{"role": "user", "content": "hi"}]
    _audit_exchange(audit, transcript, msgs, "part[inference error: x]", outcome="error")
    audit.llm.assert_called_once_with("part[inference error: x]")
    audit.notice.assert_called_once_with("finish_reason", "error")
    transcript.exchange.assert_called_once_with(
        "hi", "part[inference error: x]\n\n[finish_reason: error]")


def test_audit_success_writes_no_notice_and_plain_transcript():
    audit = MagicMock()
    transcript = MagicMock()
    msgs = [{"role": "user", "content": "hi"}]
    _audit_exchange(audit, transcript, msgs, "fine", outcome="success")
    audit.notice.assert_not_called()
    transcript.exchange.assert_called_once_with("hi", "fine")


def test_stream_error_audit_carries_visible_error_text(monkeypatch):
    """The recorded reply is what the client saw: partial text plus the error
    chunk, and the outcome notice sits beside it."""
    import localm.inference.http_server as hs
    records = []
    monkeypatch.setattr(hs, "_audit_exchange",
                        lambda a, t, m, reply, outcome="success": records.append((reply, outcome)))
    app = create_app(_engine(raise_after=True))
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    assert len(records) == 1
    reply, outcome = records[0]
    assert outcome == "error"
    assert reply.startswith("partial")
    assert "[inference error" in reply


# --------------------------------------------------------------------------- #
#  memory outlet consults the outcome                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("outcome,expected_calls", [
    ("success", 1), ("length", 1), ("error", 0), ("abort", 0),
])
def test_memory_outlet_schedules_consolidation_only_for_completed_turns(monkeypatch, outcome, expected_calls):
    from localm.plugins.builtin.memory import plug
    consolidate = MagicMock()
    sweep = MagicMock()
    monkeypatch.setattr(plug, "_maybe_auto_consolidate", consolidate)
    monkeypatch.setattr(plug, "_maybe_sweep_backfill", sweep)
    ctx = ChatHookContext(model_id="m", stream=True, request_id="r", outcome=outcome)
    assert plug._memory_outlet("reply", [], ctx) == "reply"
    assert consolidate.call_count == expected_calls
    assert sweep.call_count == expected_calls


def test_memory_outlet_treats_a_ctx_without_outcome_as_completed(monkeypatch):
    from localm.plugins.builtin.memory import plug
    consolidate = MagicMock()
    monkeypatch.setattr(plug, "_maybe_auto_consolidate", consolidate)
    monkeypatch.setattr(plug, "_maybe_sweep_backfill", MagicMock())
    plug._memory_outlet("reply", [], {})
    assert consolidate.call_count == 1
