# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model call that RAISES during memory consolidation must be reported as a
failure on every surface, never as a clean "no new facts" run.

Covers the three model-calling helpers (extract, summarize_session, _decide),
run_consolidation, the episodic pass and its watermark, synthesize_memory, the
scheduled-job runner and the manual consolidate route. A reply that merely does
not parse stays a successful empty run.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi import HTTPException

from localm.memory import ConsolidationModelError, MemoryRecord, MemoryStore
from localm.memory.consolidate import (_decide, extract, run_consolidation,
                                       summarize_session)
from localm.plugins.builtin.jobs import runner
from localm.plugins.builtin.jobs.store import Job
from localm.plugins.builtin.memory import plug

NOW = 1_800_000_000.0
_SETTLED = 1_000_000.0


@pytest.fixture(autouse=True)
def _allow_writes(monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")


def _boom(_prompt):
    raise RuntimeError("engine evicted mid-run")


def _store(tmp_path):
    return MemoryStore("owner", "chat", "", root=tmp_path)


def _store_bytes(store):
    p = store.path
    return p.read_bytes() if p.is_file() else None


# --------------------------------------------------------------------------- #
#  The helpers                                                                 #
# --------------------------------------------------------------------------- #

def test_extract_raises_when_the_model_call_raises():
    with pytest.raises(ConsolidationModelError) as ei:
        extract(_boom, "User: my name is Ada")
    assert "RuntimeError" in str(ei.value) and "evicted" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_extract_on_an_unparseable_reply_is_still_empty():
    assert extract(lambda p: "not json at all", "User: my name is Ada") == []


def test_summarize_session_raises_when_the_model_call_raises():
    with pytest.raises(ConsolidationModelError):
        summarize_session(_boom, "User: let us plan the database migration")


def test_decide_raises_when_the_model_call_raises():
    with pytest.raises(ConsolidationModelError):
        _decide(_boom, "User prefers VS Code", "User prefers Vim")


# --------------------------------------------------------------------------- #
#  run_consolidation                                                           #
# --------------------------------------------------------------------------- #

def test_run_consolidation_reports_a_failed_extract_and_writes_nothing(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryRecord(text="User prefers metric units", source="user",
                       created=NOW, last_used=NOW))
    before = _store_bytes(s)

    res = run_consolidation(s, "User: my name is Ada", _boom, now=NOW)

    assert _store_bytes(s) == before
    assert res["status"] == "error"
    assert res["reason"] == "model_failed"
    assert "evicted" in res["error"]
    assert res["added"] == 0


def test_run_consolidation_on_an_unparseable_reply_is_still_ok(tmp_path):
    res = run_consolidation(_store(tmp_path), "User: my name is Ada",
                            lambda p: "not json", now=NOW)
    assert res["status"] == "ok" and res["added"] == 0 and "error" not in res


def test_run_consolidation_reports_a_failed_decide_and_writes_nothing(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryRecord(text="User prefers Vim as their editor", source="synth",
                       importance=0.7, created=NOW, last_used=NOW))
    before = _store_bytes(s)
    facts = json.dumps({"facts": [
        {"fact": "User is named Sam", "confidence": 0.9},
        {"fact": "User now prefers VS Code as their editor", "confidence": 0.9}]})
    decided = []

    def complete(prompt):
        if "Extract ONLY durable" in prompt:
            return facts
        decided.append(prompt)
        raise RuntimeError("decode failure")

    res = run_consolidation(s, "I am Sam and I switched to VS Code", complete, now=NOW)

    assert decided, "the decide step never ran - broken test setup"
    assert _store_bytes(s) == before
    assert [r.text for r in s.all()] == ["User prefers Vim as their editor"]
    assert res["status"] == "error" and res["reason"] == "model_failed"
    assert (res["added"], res["updated"], res["deleted"], res["noop"]) == (0, 0, 0, 0)


# --------------------------------------------------------------------------- #
#  The episodic pass, synthesize_memory, the runner and the route              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [{"type": "user", "data": {"content":
                                      "let us plan the Postgres migration for the "
                                      "billing service and its rollback steps"}},
            {"type": "llm", "data": {"content": "Here is a plan for the migration."}}]
    p = sdir / "s1.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (_SETTLED, _SETTLED))
    return tmp_path


def test_a_failed_episode_summary_is_reported_and_retried_next_run(memhome):
    store = plug._chat_store()
    errors: list = []

    assert plug._store_episodes(store, _boom, embed_fn=None, errors=errors) == 0
    assert errors and "evicted" in errors[0]

    stored = plug._store_episodes(
        store, lambda p: "Planned the billing service Postgres migration",
        embed_fn=None)
    assert stored == 1, "the failed session was marked processed and never retried"


def _add_session(home, stem, mtime, topic):
    rows = [{"type": "user", "data": {"content":
                                      f"let us work through {topic} in detail and "
                                      "decide the next steps for it"}},
            {"type": "llm", "data": {"content": f"Here is a plan for {topic}."}}]
    p = home / "sessions" / f"{stem}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    os.utime(p, (mtime, mtime))


def _poisoned(prompt):
    if "Postgres migration" in prompt:
        raise RuntimeError("decode failure on this session")
    return "Worked through the release checklist"


def test_a_session_that_keeps_failing_is_skipped_after_the_attempt_cap(memhome):
    _add_session(memhome, "s2", _SETTLED + 1, "the release checklist")
    store = plug._chat_store()

    for _ in range(plug.EPISODIC_MAX_ATTEMPTS - 1):
        errors: list = []
        assert plug._store_episodes(store, _poisoned, embed_fn=None, errors=errors,
                                    model_answered=True) == 0
        assert errors

    errors = []
    assert plug._store_episodes(store, _poisoned, embed_fn=None, errors=errors,
                                model_answered=True) == 1
    assert errors
    assert [r.text for r in store.all() if r.kind == "episodic"] == [
        "Worked through the release checklist"]


def test_a_session_is_never_skipped_while_no_model_call_succeeds(memhome):
    _add_session(memhome, "s2", _SETTLED + 1, "the release checklist")
    store = plug._chat_store()

    for _ in range(plug.EPISODIC_MAX_ATTEMPTS + 2):
        assert plug._store_episodes(store, _boom, embed_fn=None, errors=[],
                                    model_answered=False) == 0

    def working(prompt):
        if "Postgres migration" in prompt:
            return "Planned the billing service Postgres migration"
        return "Worked through the release checklist"

    assert plug._store_episodes(store, working, embed_fn=None) == 2


def test_synthesize_memory_reports_a_failed_model_call(memhome):
    res = plug.synthesize_memory(_boom)
    assert res["status"] == "error"
    assert res["reason"] == "model_failed"
    assert "evicted" in res["error"]
    assert res["facts"] == []


class _Engine:
    loaded = True

    def __init__(self, reply=None):
        self.display_name = "stub-model"
        self.active_requests = 0
        self._reply = reply

    def chat_stream(self, messages, **kw):
        if self._reply is None:
            raise RuntimeError("engine evicted mid-run")
        yield self._reply


def _memory_job():
    return Job(name="t", task_kind="memory", prompt="", schedule_kind="interval",
               schedule=60)


def test_the_memory_job_fails_when_the_model_call_raises(memhome):
    with pytest.raises(RuntimeError) as ei:
        runner._run_memory(_memory_job(), engine=_Engine())
    msg = str(ei.value)
    assert "memory synthesis failed" in msg and "evicted" in msg
    assert "no new durable facts" not in msg

    record = runner.run_job(_memory_job(), engine=_Engine())
    assert record["status"] == "error"
    assert "evicted" in record["error"]


def test_the_memory_job_with_no_facts_still_reports_a_clean_run(memhome):
    out = runner._run_memory(_memory_job(), engine=_Engine(json.dumps({"facts": []})))
    assert out.startswith("memory synthesis: no new durable facts found")


def test_the_consolidate_route_returns_500_when_the_model_call_raises(
        memhome, monkeypatch):
    monkeypatch.setattr(plug, "_live_engine", lambda: _Engine())

    with pytest.raises(HTTPException) as ei:
        asyncio.run(plug.memory_consolidate(request=None))

    assert ei.value.status_code == 500
    assert "model call failed" in ei.value.detail and "evicted" in ei.value.detail


def test_the_background_pass_logs_a_failed_model_call_as_a_warning(memhome, monkeypatch,
                                                                    caplog):
    monkeypatch.setattr(plug, "_live_engine", lambda: _Engine())

    with caplog.at_level("INFO", logger="localm"):
        plug._auto_consolidate_bg()

    hits = [r for r in caplog.records if "auto-consolidate: pass complete" in r.getMessage()]
    assert hits and hits[-1].levelname == "WARNING"
    assert "status=error" in hits[-1].getMessage() and "evicted" in hits[-1].getMessage()
