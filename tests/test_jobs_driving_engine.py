# SPDX-License-Identifier: AGPL-3.0-or-later
"""jobs/runner.py's _run_memory and _run_chat must wrap their generation calls in
http_server.driving_engine, or a scheduled job running on a quiet server (no
concurrent HTTP traffic) leaves nothing marking the model as in-use and idle-unload
can unload it mid-run.

Checked from OUTSIDE the wrap, at the point the underlying call actually runs, so
a forgotten or misplaced `with driving_engine(...)` fails these regardless of
implementation detail.
"""

from __future__ import annotations

from localm.plugins.builtin.jobs import runner
from localm.plugins.builtin.jobs.store import Job


class _FakeEngine:
    def __init__(self, name="job-model"):
        self.display_name = name
        self.loaded = True
        self.active_requests = 0
        self.active_requests_during_call = None

    def chat_stream(self, messages, **kw):
        self.active_requests_during_call = self.active_requests
        yield "ok"


def _make_job(**kw):
    base = dict(name="t", task_kind="chat", prompt="hi",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


def test_run_chat_pins_the_engine_busy_for_the_whole_tool_loop(monkeypatch):
    eng = _FakeEngine()

    def fake_run_chat_with_web(engine, prompt, **kw):
        return f"active_requests_during_call={engine.active_requests}"

    monkeypatch.setattr(
        "localm.plugins.builtin.jobs.webtool.run_chat_with_web",
        fake_run_chat_with_web)

    result = runner._run_chat(_make_job(prompt="hi"), engine=eng)

    assert result == "active_requests_during_call=1", (
        "the engine was not pinned busy during run_chat_with_web - a scheduled "
        "chat job on a quiet server has nothing marking the model as in-use")
    assert eng.active_requests == 0, "the pin must be released once the call returns"


def test_run_memory_pins_the_engine_busy_for_the_whole_synthesis_pass(monkeypatch):
    eng = _FakeEngine("memory-model")
    calls = []

    def fake_synthesize_memory(complete, **kw):
        # synthesize_memory can call complete() several times (one per
        # candidate), so the pin is checked here, from OUTSIDE any individual
        # chat_stream call.
        calls.append(eng.active_requests)
        complete("dummy prompt")
        return {"status": "ok", "added": 0}

    monkeypatch.setattr(
        "localm.plugins.builtin.memory.plug.synthesize_memory",
        fake_synthesize_memory)

    result = runner._run_memory(_make_job(task_kind="memory"), engine=eng)

    assert calls == [1], (
        "the engine was not pinned busy during synthesize_memory - a scheduled "
        "memory job on a quiet server has nothing marking the model as in-use")
    assert eng.active_requests_during_call == 1
    assert eng.active_requests == 0, "the pin must be released once the call returns"
    assert isinstance(result, str)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
