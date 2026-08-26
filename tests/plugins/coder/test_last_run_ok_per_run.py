# SPDX-License-Identifier: AGPL-3.0-or-later
"""``last_run_ok`` is PER-RUN; the session-wide failure lesson survives it.

``Agent._last_run_ok`` is re-armed by ``_loop`` at the start of every run,
alongside ``_stop_requested`` and ``_user_stopped``, so it means "the last run
failed" rather than "any run this session failed". Both consumers read it that
way: the GUI's per-turn badge (the final event's ``"ok"``) and the CLI's exit
code.

``session.py``'s close-time episodic reflection answers a DIFFERENT question -
did anything fail this session, i.e. is there a failure lesson worth a
1024-token reflection - and reads ``_had_any_failure``, the session-level
record ``_loop`` keeps for it.
"""

from __future__ import annotations

import json
import queue
import time

import pytest

from localm.plugins.coder.audit import SessionMode
from tests.conftest import final_answer as _final_answer


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


class _Scripted:
    """One canned response per chat() call, repeating the last.

    The close-time reflection is a chat() call too; it is the only one made with
    max_tokens=1024, so it is answered separately (and counted) instead of eating
    a scripted turn response."""

    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 5}

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.reflections: list = []

    def chat(self, messages, **kw):
        if kw.get("max_tokens") == 1024:
            self.reflections.append(kw)
            return ('{"summary": "s", "what_worked": "w", "what_failed": "f", '
                    '"lesson": "l"}')
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def chat_stream(self, messages, **kw):
        yield self.chat(messages, **kw)


def _tc(name: str, **args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "args": args}) + "</tool_call>"


# The per-tool circuit breaker aborts after 4 identical failures, so this run
# fails through the real loop with no hand-set flags.
_BREAKER_RUN = [_tc("read_file", path="missing.txt")] * 4


def _agent(tmp_path, responses, **kw):
    from localm.plugins.coder.agent import Agent
    kw.setdefault("auto_approve", True)
    kw.setdefault("self_verify", False)
    kw.setdefault("max_turns", 10)
    kw.setdefault("mode", SessionMode.LOG)
    return Agent(_Scripted(responses), cwd=tmp_path, **kw)


# --------------------------------------------------------------------------- #
#  A clean turn after a failed one reports ok                                  #
# --------------------------------------------------------------------------- #

def test_clean_run_after_a_failed_run_reports_ok(home, tmp_path):
    """At the Agent level: two real runs, the first tripping the circuit
    breaker and the second answering cleanly. The second reports ok."""
    agent = _agent(tmp_path, _BREAKER_RUN + ["All clean now."])

    first = agent.run_task("read the missing file")
    assert "circuit breaker" in first
    assert agent.last_run_ok is False

    second = agent.run_task("now just say something")
    assert _final_answer(second) == "All clean now."
    assert agent.last_run_ok is True, (
        "a clean run after a failed one still reported failure - _loop never "
        "re-armed _last_run_ok, so one bad turn poisoned the whole session")


def test_gui_final_event_reports_the_clean_turn_as_ok(tmp_path):
    """Through the real GUI session path: the done event's "ok" comes straight
    from agent.last_run_ok, once per turn."""
    from localm.plugins.coder.sessions import CoderSession

    session = CoderSession(
        tmp_path, _Scripted(_BREAKER_RUN + ["All clean now."]),
        auto_approve=True, max_turns=10)
    try:
        assert session.send_message("read the missing file") == "started"
        first = _drain_final(session)
        assert first["ok"] is False, "the failed turn should report failed"

        _wait_idle(session)
        assert session.send_message("now just say something") == "started"
        second = _drain_final(session)
        assert "All clean now." in second["text"]
        assert second["ok"] is True, (
            "the GUI labelled a healthy turn as failed because the previous "
            "turn had failed")
    finally:
        session.close()


def test_a_stopped_run_does_not_poison_the_next_one(home, tmp_path):
    """The same property via the stop path rather than a breaker: a user stop
    clears _last_run_ok too, and the next turn still reports ok."""
    agent = _agent(tmp_path, ["partial answer", "All clean now."])
    original_chat = agent.backend.chat

    def _chat_then_stop(messages, **kw):
        out = original_chat(messages, **kw)
        agent.request_stop()
        return out

    agent.backend.chat = _chat_then_stop
    assert _final_answer(agent.run_task("long thing")) == "partial answer"
    assert agent.last_run_ok is False and agent._user_stopped is True

    agent.backend.chat = original_chat
    assert _final_answer(agent.run_task("something else")) == "All clean now."
    assert agent.last_run_ok is True
    assert agent._user_stopped is False


# --------------------------------------------------------------------------- #
#  The session-level failure lesson is kept                                    #
# --------------------------------------------------------------------------- #

def test_session_that_failed_then_recovered_still_reflects_at_close(home, tmp_path):
    """The failed first run is still a lesson at close, even though the session
    ended on a clean run and wrote no files: the close-time reflection reads the
    session-level marker, not the per-run _last_run_ok."""
    agent = _agent(tmp_path, _BREAKER_RUN + ["All clean now."])
    agent._episode_task = "read the missing file"

    agent.run_task("read the missing file")
    agent.run_task("now just say something")
    assert agent.last_run_ok is True         # per-run: the last run was fine
    assert agent._had_any_failure is True    # session-level: something did fail

    agent.close()                            # on_event None -> synchronous
    assert len(agent.backend.reflections) == 1, (
        "the failed run's lesson was lost when the session ended on a clean run")
    eps = agent._episode_store.all()
    assert len(eps) == 1
    # The session completed, so the episode says so; the failure is carried by
    # what_failed rather than by the outcome label.
    assert eps[0].outcome == "ok"


def test_all_clean_session_reflects_nothing(home, tmp_path):
    """The other half: the session-level flag does not arm itself on a healthy
    session, so quitting one pays for no model reflection."""
    agent = _agent(tmp_path, ["All clean now."])
    agent._episode_task = "say something"

    agent.run_task("say something")
    agent.run_task("say something else")
    assert agent._had_any_failure is False

    agent.close()
    assert agent.backend.reflections == []
    assert agent._episode_store.all() == []


def test_failed_only_session_still_reports_incomplete(home, tmp_path):
    """A single-run failure: not ok, reflects, and is recorded as an incomplete
    session."""
    agent = _agent(tmp_path, _BREAKER_RUN)
    agent._episode_task = "read the missing file"

    agent.run_task("read the missing file")
    assert agent.last_run_ok is False and agent._had_any_failure is True

    agent.close()
    assert len(agent.backend.reflections) == 1
    assert agent._episode_store.all()[0].outcome == "incomplete"


def test_reset_clears_the_session_failure_marker(home, tmp_path):
    """/clear drops the history and error trace a lesson would be built from,
    and drops the session failure marker with them."""
    agent = _agent(tmp_path, _BREAKER_RUN)
    agent._episode_task = "read the missing file"
    agent.run_task("read the missing file")
    assert agent._had_any_failure is True

    agent.reset()
    assert agent._had_any_failure is False
    assert agent.last_run_ok is True

    agent.close()
    assert agent.backend.reflections == []


# --------------------------------------------------------------------------- #
#  GUI event-queue helpers                                                     #
# --------------------------------------------------------------------------- #

def _drain_final(session, timeout: float = 15.0) -> dict:
    """Return the session's next "final" event, or fail the test."""
    seen: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = session.events.get(timeout=0.2)
        except queue.Empty:
            continue
        seen.append(ev["type"])
        if ev["type"] == "final":
            return ev
    raise AssertionError(f"no final event within {timeout}s (saw {seen})")


def _wait_idle(session, timeout: float = 15.0) -> None:
    """Wait for the worker thread to release the session.

    ``busy`` is cleared in the run thread's ``finally``, just AFTER the final
    event is pushed, so a send issued the instant that event arrives would race
    and be taken as a mid-task steering note instead of a new turn."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not session.busy:
            return
        time.sleep(0.02)
    raise AssertionError("session still busy after the final event")
