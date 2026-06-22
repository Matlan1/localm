# SPDX-License-Identifier: AGPL-3.0-or-later
"""A2: memory auto-synthesis. Distil durable user facts from recent session logs
into chat-memory.md, runnable as a jobs "memory" task. The model call is injected
so the deterministic logic is testable without a model; privacy mode must SKIP
and say so (never a silent success)."""

import json

import pytest

from localm.plugins.builtin.chat import plug


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
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


def test_synthesize_appends_facts(memhome):
    out = plug.synthesize_memory(lambda p: "- User is named Sam\n- Uses Rust daily")
    assert out["status"] == "ok"
    assert out["added"] == 2
    mem = (memhome / "chat-memory.md").read_text(encoding="utf-8")
    assert "Sam" in mem and "Rust" in mem


def test_synthesize_dedupes_against_existing(memhome):
    (memhome / "chat-memory.md").write_text("- User is named Sam\n", encoding="utf-8")
    out = plug.synthesize_memory(lambda p: "- user is named sam\n- Uses Rust daily")
    assert out["added"] == 1                 # the casefold-duplicate is dropped
    assert out["facts"] == ["Uses Rust daily"]


def test_synthesize_strips_nested_and_star_bullets(memhome):
    # A real model emitted nested bullets ("- - Name: Alex"); the fact must come
    # out clean, not "- Name: Alex". Also handle "* " bullets.
    out = plug.synthesize_memory(lambda p: "- - Name: Alex\n*  Uses Go")
    assert out["facts"] == ["Name: Alex", "Uses Go"]
    mem = (memhome / "chat-memory.md").read_text(encoding="utf-8")
    assert "- - " not in mem
    assert "- Name: Alex" in mem and "- Uses Go" in mem


def test_synthesize_caps_additions(memhome):
    facts = "\n".join(f"- fact {i}" for i in range(50))
    out = plug.synthesize_memory(lambda p: facts, max_facts=5)
    assert out["added"] == 5


def test_privacy_mode_skips_and_never_calls_model(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: False)
    calls = {"n": 0}

    def fake(p):
        calls["n"] += 1
        return "- something"

    out = plug.synthesize_memory(fake)
    assert out["status"] == "skipped" and out["reason"] == "privacy"
    assert out["added"] == 0
    assert calls["n"] == 0                    # no model call, no write
    assert not (tmp_path / "chat-memory.md").exists()


def test_no_sessions_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    out = plug.synthesize_memory(lambda p: "- x")
    assert out["status"] == "skipped" and out["reason"] == "no_sessions"


def test_recent_sessions_text_keeps_turns_drops_system(memhome):
    txt = plug._recent_sessions_text()
    assert "My name is Sam" in txt
    assert "Nice to meet you" in txt
    assert "session started" not in txt       # system rows excluded


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
            yield "- User is named Sam"

    job = Job(name="m", task_kind="memory", prompt="",
              schedule_kind="interval", schedule=3600)
    res = runner.run_job(job, engine=FakeEng())
    assert res["status"] == "ok"
    assert "Sam" in res["output"]
    assert "Sam" in (memhome / "chat-memory.md").read_text(encoding="utf-8")
