# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trivial session chatter stays out of long-term memory and out of recall.

Write side: a session with no lasting content (greetings, a passing remark, a
one-off utility request) gets no episodic record. Read side: one generic content
word shared with an episodic summary does not inject it.
"""

from __future__ import annotations

import json
import os

import pytest

from localm.memory import MemoryRecord, MemoryStore
from localm.memory.consolidate import _EPISODE_PROMPT, summarize_session
from localm.plugins.builtin.memory import plug

NOW = 1_800_000_000.0
SESSION = "User: Let us plan the sensor polling loop.\nAssistant: Sure."

DARTS = "Mentioned plans to play darts and test their local LLM model runner."
TITLES = "Discussed generating succinct titles and git branch names from session descriptions."


# ------------------------------------------------------------ write: summary #

@pytest.mark.parametrize("reply", [
    "NONE",
    "None.",
    "None - this was small talk.",
    "Nothing lasting was discussed.",
    "No lasting content in this conversation.",
    "N/A",
    "NONE\nDiscussed the weather and a dart match.",
])
def test_summary_none_reply_stores_nothing(reply):
    assert summarize_session(lambda p: reply, SESSION) == ""


@pytest.mark.parametrize("reply", [
    "Exchanged pleasantries and said goodbye.",
    "Small talk about the weekend.",
    "Greeted the assistant and asked how it was doing.",
    "Had a brief chat about darts.",
])
def test_summary_pleasantry_only_is_rejected(reply):
    assert summarize_session(lambda p: reply, SESSION) == ""


def test_summary_real_work_is_still_stored():
    good = "Planned the sensor polling loop for the greenhouse controller."
    assert summarize_session(lambda p: good, SESSION) == good


def test_nonword_none_prefix_is_not_a_decline():
    good = "Nonexistent-path handling was added to the CSV importer."
    assert summarize_session(lambda p: good, SESSION) == good


def test_episode_prompt_tells_the_model_it_may_decline():
    seen = []
    summarize_session(lambda p: seen.append(p) or "NONE", SESSION)
    assert seen and seen[0].startswith(_EPISODE_PROMPT)
    assert "exactly NONE" in seen[0] and "small talk" in seen[0]


# ------------------------------------------------- write: episodic capture #

@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    (tmp_path / "sessions").mkdir()
    return tmp_path


def _write_session(home, name, user_text, mtime):
    rows = [{"type": "user", "data": {"content": user_text}},
            {"type": "llm", "data": {"content": "Sure."}}]
    p = home / "sessions" / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))


def _episodes(store):
    return [r for r in store.all() if r.kind == "episodic"]


def test_greeting_session_is_never_summarised(memhome):
    _write_session(memhome, "hello", "hi, thanks!", 1000)
    prompts = []

    def complete(p):
        prompts.append(p)
        return "Discussed greetings."

    store = plug._chat_store()
    assert plug._store_episodes(store, complete, embed_fn=None, now=1e9) == 0
    assert _episodes(store) == []
    assert prompts == []                       # not even a model call
    # The watermark moved past it: a rerun does not look at it again.
    assert plug._store_episodes(store, complete, embed_fn=None, now=1e9) == 0
    assert prompts == []


def test_small_talk_declined_by_model_creates_no_record(memhome):
    _write_session(memhome, "darts", "Next week I plan to play darts with friends", 1000)
    store = plug._chat_store()
    assert plug._store_episodes(store, lambda p: "NONE", embed_fn=None, now=1e9) == 0
    assert _episodes(store) == []


def test_substantive_session_still_becomes_an_episode(memhome):
    _write_session(memhome, "real", "Help me debug the flaky upload test in CI", 1000)
    store = plug._chat_store()
    n = plug._store_episodes(
        store, lambda p: "Chased down the flaky CI upload failure.", embed_fn=None, now=1e9)
    assert n == 1
    assert [r.text for r in _episodes(store)] == ["Chased down the flaky CI upload failure."]


# --------------------------------------------------------------- read: recall #

def _store(tmp_path):
    s = MemoryStore("owner", "chat", root=tmp_path)
    for t in (DARTS, TITLES):
        s.add(MemoryRecord(text=t, kind="episodic", source="synth", importance=0.4))
    return s


@pytest.mark.parametrize("query", [
    "which model should I use for summaries",
    "is the local server running",
    "write a poem about autumn",
])
def test_one_generic_word_does_not_inject_episodes(tmp_path, query):
    s = _store(tmp_path)
    assert s.recall(query, k=6, now=NOW) == []


def test_two_shared_content_words_still_recall_the_episode(tmp_path):
    s = _store(tmp_path)
    hits = s.recall("how do I test the local LLM model runner", k=6, now=NOW)
    assert [h.text for h in hits] == [DARTS]


def test_single_word_query_can_still_hit_an_episode(tmp_path):
    s = _store(tmp_path)
    assert [h.text for h in s.recall("darts", k=6, now=NOW)] == [DARTS]


def test_user_fact_keeps_single_word_lexical_hit(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryRecord(text="User runs a local model on a laptop", source="user"))
    hits = s.recall("is the local server running", k=6, now=NOW)
    assert [h.text for h in hits] == ["User runs a local model on a laptop"]
