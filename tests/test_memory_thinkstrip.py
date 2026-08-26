# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thinking-model reasoning output must never reach any memory-forming store, and
its presence must never silently zero a pipeline.

Every test drives the actual production code path with a deterministic fake
``complete``/engine emitting the shapes a thinking model produces: scratchpad
before JSON, braces inside the scratchpad, all-reasoning truncated replies."""

from __future__ import annotations

import json

import pytest

from localm.textnorm import strip_think
from localm.memory import MemoryRecord, MemoryStore, run_consolidation
from localm.memory.consolidate import (_parse_json_object, extract,
                                       summarize_session)

THINK_JSON = (
    "<think>We need to produce JSON. The user {Ada} likes {metric}. Let me "
    "reason about braces {{{ ... </think>\n"
    '{"facts": [{"fact": "User prefers metric units", "confidence": 0.9}]}'
)
THINK_ONLY = "<think>We need to produce a one-sentence summary of the conv"
SESSION = "User: my name is Ada and I prefer metric units\nAssistant: Noted!"


# --------------------------------------------------------------- strip_think #

def test_strip_think_closed_block():
    assert strip_think("<think>secret</think>answer") == "answer"


def test_strip_think_unclosed_block_truncation():
    # A truncated thinking reply is ALL scratchpad; nothing may leak.
    assert strip_think(THINK_ONLY) == ""


def test_strip_think_dialect_normalisation():
    # Harmony-style channel tags are scrubbed to canonical tags first.
    raw = "<|channel|>analysis<|message|>hidden<|channel|>final<|message|>visible"
    assert strip_think(raw).strip() == "visible"


def test_strip_think_idempotent():
    once = strip_think(THINK_JSON)
    assert strip_think(once) == once
    assert "<think" not in once


# ------------------------------------------------------- extraction parsing #

def test_extract_parses_json_behind_think_block():
    # Braces inside <think> break the first-{-to-last-} scavenge and extraction
    # returns zero facts.
    facts = extract(lambda p: THINK_JSON, SESSION)
    assert [f["text"] for f in facts] == ["User prefers metric units"]


def test_extract_all_think_reply_yields_nothing():
    assert extract(lambda p: THINK_ONLY, SESSION) == []


def test_parse_json_object_balanced_span_fallback():
    # Prose with stray braces around the object: first-{-to-last-} fails, the
    # balanced-span fallback must still find the real JSON (last span wins).
    raw = 'Note {this} is chatter. {"decision": "NO_OP", "confidence": 0.9} ok {'
    assert _parse_json_object(raw)["decision"] == "NO_OP"


def test_parse_json_object_strips_think_itself():
    # Defense in depth: even an unstripped caller cannot poison parsing.
    assert _parse_json_object(THINK_JSON)["facts"][0]["confidence"] == 0.9


# ------------------------------------------------------------ episodic path #

def test_summarize_session_skips_think_opener_line():
    # The first raw line is the think opener; it must not be stored verbatim as a
    # durable episodic record.
    raw = "<think>We need to produce a one-sentence summary...</think>\n" \
          "Discussed the greenhouse controller project."
    assert summarize_session(lambda p: raw, SESSION) == \
        "Discussed the greenhouse controller project."


def test_summarize_session_all_think_returns_empty():
    assert summarize_session(lambda p: THINK_ONLY, SESSION) == ""


# ------------------------------------------------- store-level no-poisoning #

def test_consolidation_never_stores_think_text(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")   # writes gated off in privacy
    store = MemoryStore("owner", "chat", root=tmp_path)

    def complete(prompt: str) -> str:
        return THINK_JSON if '"facts"' in prompt or "durable" in prompt \
            else "<think>hm</think>" + json.dumps(
                {"decision": "ADD", "confidence": 0.9})

    res = run_consolidation(store, SESSION, complete)
    assert res["added"] == 1
    for rec in store.all():
        assert "<think" not in rec.text


def test_store_regression_no_think_substring_via_episode(tmp_path):
    # An episodic record beginning "<think>We need to produce..." must be
    # impossible.
    store = MemoryStore("owner", "chat", root=tmp_path)
    summ = summarize_session(lambda p: THINK_ONLY, SESSION)
    if summ:                       # must not happen; guard the assert below
        store.add(MemoryRecord(text=summ, kind="episodic"))
    assert all("<think" not in r.text for r in store.all())
    assert summ == ""


# ------------------------------------------------------------- job results #

def test_webtool_final_answer_strips_think():
    from localm.plugins.builtin.jobs.webtool import _final_answer
    assert _final_answer("<think>plan</think>PONG") == "PONG"


def test_webtool_final_answer_all_think_is_honest():
    from localm.plugins.builtin.jobs.webtool import _final_answer
    out = _final_answer(THINK_ONLY)
    assert "reasoning" in out and "truncated" in out


def test_run_memory_reports_reasoning_only_replies(monkeypatch):
    # When extraction finds nothing AND replies were empty after the strip,
    # the job result must carry the caveat, not read as a clean no-op.
    from localm.plugins.builtin.jobs import runner

    def fake_synthesize(complete):
        complete("extract prompt")           # exercises the wrapper's counter
        return {"status": "ok", "added": 0, "facts": []}

    import localm.plugins.builtin.memory.plug as chat_plug
    monkeypatch.setattr(chat_plug, "synthesize_memory", fake_synthesize)

    class _Eng:
        def chat_stream(self, messages, **kw):
            yield THINK_ONLY

    job = type("J", (), {"task_kind": "memory", "prompt": "", "model": None})()
    out = runner._run_memory(job, engine=_Eng())
    assert "only reasoning output" in out


# ------------------------------------------------------ compaction summary #

def test_compaction_summary_never_contains_think():
    from localm.inference.compact import compact_messages
    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(12)
    ]
    out, changed = compact_messages(
        msgs, lambda m, mt: "<think>reasoning</think>Talked about turns.")
    assert changed
    joined = json.dumps(out)
    assert "<think" not in joined
    assert "Talked about turns." in joined


def test_compaction_all_think_summary_falls_back_to_trim():
    from localm.inference.compact import compact_messages
    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(12)
    ]
    out, changed = compact_messages(msgs, lambda m, mt: THINK_ONLY)
    assert changed
    joined = json.dumps(out)
    assert "<think" not in joined
    assert "removed to fit the context window" in joined


# -------------------------------------------------------- coder reflection #

def test_reflect_and_store_strips_think(tmp_path):
    from localm.plugins.coder.episodes import EpisodeStore, reflect_and_store
    store = EpisodeStore(tmp_path / "proj", root=tmp_path / "eps")
    raw = "<think>let me reflect {a}</think>" + json.dumps({
        "summary": "Added hello.txt", "what_worked": "write_file",
        "what_failed": "", "lesson": "Create the file directly."})
    ep = reflect_and_store(
        store, task="create hello.txt", diff="+hi", outcome="ok",
        files=["hello.txt"], turns=1, complete=lambda p: raw)
    assert ep.lesson == "Create the file directly."
    stored = store.all()
    assert len(stored) == 1
    assert "<think" not in json.dumps(stored[0].to_dict())


def test_reflect_and_store_empty_after_think_warns_and_skips(tmp_path, caplog):
    import logging

    from localm.plugins.coder.episodes import EpisodeStore, reflect_and_store
    store = EpisodeStore(tmp_path / "proj", root=tmp_path / "eps")
    with caplog.at_level(logging.WARNING, logger="localm"):
        reflect_and_store(
            store, task="t", diff="", outcome="ok", files=[], turns=1,
            complete=lambda p: THINK_ONLY)
    assert store.all() == []
    assert any("reflection produced no usable lesson" in r.message
               for r in caplog.records)


# --------------------------------------------------- session-text hygiene #

def test_recent_sessions_text_strips_assistant_think(tmp_path, monkeypatch):
    import localm.plugins.builtin.memory.plug as chat_plug
    monkeypatch.setattr(chat_plug, "_home", lambda: tmp_path)
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [
        {"type": "user", "data": {"content": "my name is Ada"}},
        {"type": "llm", "data": {"content": "<think>who?</think>Hi Ada!"}},
        {"type": "llm", "data": {"content": THINK_ONLY}},
    ]
    (sdir / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    text = chat_plug._recent_sessions_text()
    assert "my name is Ada" in text
    assert "Hi Ada!" in text
    assert "<think" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# -------------------------------------------------------- episodic quality gate #

def test_summarize_session_rejects_prompt_echo():
    # A verbatim prompt echo stored as an episode; the gate must drop it.
    from localm.memory.consolidate import summarize_session
    echo = "Summarise in ONE short sentence what the user and the assistant did"
    assert summarize_session(lambda p: echo, SESSION) == ""


def test_summarize_session_rejects_self_narration():
    from localm.memory.consolidate import summarize_session
    assert summarize_session(
        lambda p: "As an AI language model, I discussed things.", SESSION) == ""


def test_summarize_session_rejects_few_shot_parrot():
    from localm.memory.consolidate import summarize_session
    parrot = "Discussed migrating the database to Postgres 16"
    assert summarize_session(lambda p: parrot, SESSION) == ""


def test_summarize_session_accepts_real_summary():
    from localm.memory.consolidate import summarize_session
    good = "Planned the greenhouse controller's sensor polling loop."
    assert summarize_session(lambda p: good, SESSION) == good


def test_summarize_session_skips_bad_line_takes_good_one():
    from localm.memory.consolidate import summarize_session
    raw = "Sure, here is your summary:\nBuilt a CSV export for the reports page."
    assert summarize_session(lambda p: raw, SESSION) == \
        "Built a CSV export for the reports page."


def test_summarize_session_accepts_natural_the_conversation_opener():
    # "The conversation focused on..." is a legitimate summary opener and must NOT
    # be dropped (only the echo-specific "the conversation below/above/is" forms
    # are rejected).
    from localm.memory.consolidate import summarize_session
    good = "The conversation focused on the sensor polling loop design."
    assert summarize_session(lambda p: good, SESSION) == good


def test_summarize_session_still_rejects_conversation_echo():
    from localm.memory.consolidate import summarize_session
    echo = "The conversation below is data to summarise."
    assert summarize_session(lambda p: echo, SESSION) == ""


def test_summarize_session_rejects_json_blob():
    # A model that returns JSON for the episode prompt must not have that blob
    # stored as a durable episodic record.
    from localm.memory.consolidate import summarize_session
    blob = '{"facts": [{"fact": "x", "confidence": 0.9}]}'
    assert summarize_session(lambda p: blob, SESSION) == ""
