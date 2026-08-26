# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory consolidation: distil durable user facts from recent session logs into
the structured chat memory store (localm/memory) via the ADD/UPDATE/DELETE/NO_OP
loop, runnable as the jobs "memory" task. The model call is injected so the logic
is testable without a model; privacy mode must SKIP and say so (never a silent
success, never a model call)."""

import json

import pytest

from localm.memory import MemoryRecord
from localm.plugins.builtin.memory import plug


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    # writes_allowed("chat") (used by the consolidation gate) reads the mode env.
    monkeypatch.setenv("LOCALM_MODE", "log")
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [
        {"type": "user", "data": {"content": "My name is Sam and I use Rust daily."}},
        {"type": "llm", "data": {"content": "Nice to meet you, Sam."}},
        {"type": "system", "data": {"msg": "session started"}},
    ]
    (sdir / "s1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def _facts_stub(*facts, decision="NO_OP", conf=0.9):
    """A model stub: returns the given facts for the extract prompt and *decision*
    for a consolidation decision prompt."""
    payload = json.dumps({"facts": [{"fact": t, "confidence": c} for t, c in facts]})
    dec = json.dumps({"decision": decision, "confidence": conf})

    def complete(prompt):
        if "Extract ONLY durable" in prompt:
            return payload
        if "Decide the single best action" in prompt:
            return dec
        return "{}"
    return complete


def test_synthesize_adds_facts_to_store(memhome):
    out = plug.synthesize_memory(
        _facts_stub(("User is named Sam", 0.9), ("User uses Rust daily", 0.8)))
    assert out["status"] == "ok"
    assert out["added"] == 2
    texts = {r.text for r in plug._chat_store().all()}
    assert any("Sam" in t for t in texts) and any("Rust" in t for t in texts)


def test_synthesize_dedupes_against_existing(memhome):
    store = plug._chat_store()
    store.add(MemoryRecord(text="User is named Sam", source="user", importance=1.0))
    out = plug.synthesize_memory(
        _facts_stub(("User is named Sam", 0.9), ("User uses Rust daily", 0.8)))
    assert out["added"] == 1                          # the near-duplicate is skipped
    assert out["facts"] == ["User uses Rust daily"]


def test_synthesize_caps_additions(memhome):
    # distinct facts (so the near-dup candidate filter does not collapse them)
    facts = [("User is a data scientist", 0.9), ("User lives in Berlin", 0.9),
             ("User drives an electric car", 0.9), ("User plays the violin", 0.9),
             ("User speaks French fluently", 0.9), ("User has two cats", 0.9),
             ("User runs marathons", 0.9), ("User collects vinyl records", 0.9)]
    out = plug.synthesize_memory(_facts_stub(*facts, decision="ADD"), max_facts=5)
    assert out["added"] == 5                           # capped at max_facts


def test_privacy_mode_skips_and_never_calls_model(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    calls = {"n": 0}

    def fake(p):
        calls["n"] += 1
        return json.dumps({"facts": [{"fact": "x", "confidence": 1.0}]})

    out = plug.synthesize_memory(fake)
    assert out["status"] == "skipped" and out["reason"] == "privacy"
    assert out["added"] == 0
    assert calls["n"] == 0                             # no model call, no write
    assert not (tmp_path / "memory").exists()


def test_no_sessions_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "log")
    out = plug.synthesize_memory(_facts_stub(("x", 0.9)))
    assert out["status"] == "skipped" and out["reason"] == "no_sessions"


def test_recent_sessions_text_keeps_turns_drops_system(memhome):
    txt = plug._recent_sessions_text()
    assert "My name is Sam" in txt
    assert "Nice to meet you" in txt
    assert "session started" not in txt               # system rows excluded


def test_legacy_flat_memory_migrated_once(memhome):
    (memhome / "chat-memory.md").write_text(
        "- User prefers dark mode\n- User is in Berlin\n", encoding="utf-8")
    store = plug._chat_store()
    plug._migrate_legacy(store)
    texts = {r.text for r in store.all()}
    assert "User prefers dark mode" in texts and "User is in Berlin" in texts
    assert all(r.source == "import" for r in store.all())
    # second call does not re-import (marker present)
    n = len(store.all())
    plug._migrate_legacy(store)
    assert len(plug._chat_store().all()) == n


def test_memory_job_needs_no_prompt_but_chat_does():
    from localm.plugins.builtin.jobs.store import Job
    j = Job(name="m", task_kind="memory", prompt="",
            schedule_kind="interval", schedule=3600)
    assert j.task_kind == "memory"
    with pytest.raises(ValueError):
        Job(name="c", task_kind="chat", prompt="",
            schedule_kind="interval", schedule=3600)


def test_run_job_memory_kind(memhome):
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.store import Job

    class FakeEng:
        def chat_stream(self, messages):
            yield '{"facts": [{"fact": "User is named Sam", "confidence": 0.9}]}'

    job = Job(name="m", task_kind="memory", prompt="",
              schedule_kind="interval", schedule=3600)
    res = runner.run_job(job, engine=FakeEng())
    assert res["status"] == "ok"
    assert "Sam" in res["output"]
    assert any("Sam" in r.text for r in plug._chat_store().all())


def test_run_job_memory_surfaces_pending_corrections(memhome):
    # A background memory job tells the user when a saved fact has a pending
    # supersede suggestion, reporting the total outstanding.
    from localm.plugins.builtin.jobs import runner
    from localm.plugins.builtin.jobs.store import Job

    plug._chat_store().add(MemoryRecord(text="User lives in Berlin", source="user",
                                        importance=0.8))
    sdir = memhome / "sessions"
    (sdir / "move.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "data": {"content": "I live in Munich now, not Berlin."}},
        {"type": "llm", "data": {"content": "Understood, Munich it is."}},
    ]), encoding="utf-8")

    class FakeEng:
        def chat_stream(self, messages):
            p = messages[0]["content"]
            if "Extract ONLY durable" in p:
                yield '{"facts": [{"fact": "User lives in Munich", "confidence": 0.9}]}'
            elif "Decide the single best action" in p:
                yield '{"decision": "UPDATE", "confidence": 0.9}'
            else:
                yield "{}"

    job = Job(name="m", task_kind="memory", prompt="",
              schedule_kind="interval", schedule=3600)
    res = runner.run_job(job, engine=FakeEng())
    assert res["status"] == "ok"
    assert "await review" in res["output"], res["output"]     # the surfacing
    # the trusted fact is untouched and the correction is pending, not applied
    assert any("Berlin" in r.text for r in plug._chat_store().all())
    assert len(plug._chat_store().corrections()) == 1
