# SPDX-License-Identifier: AGPL-3.0-or-later
"""``last_run_ok`` is PER-RUN; the session-wide failure lesson survives it.

THE BUG. ``Agent._last_run_ok`` started True and was only ever set False - by
max_turns, either circuit breaker, or a stop. ``_loop`` re-armed
``_stop_requested`` and ``_user_stopped`` at the start of every run but not this
flag, so it was one-way for the life of the session. In a multi-turn session
(the REPL, or the GUI, which reports it per turn as the final event's ``"ok"``)
one failed turn labelled every LATER turn a failure too, however clean it was.
The user watched healthy turns come back marked failed.

THE FIX re-arms it in ``_loop``, which narrows its meaning from "any run this
session failed" to "the last run failed" - matching its name, its docstring, and
both consumers (the GUI's per-turn badge, the CLI's exit code).

THE COMPLICATION. ``session.py``'s close-time episodic reflection read the same
flag to answer a DIFFERENT question: did anything fail this session, i.e. is
there a failure lesson worth a 1024-token reflection (REG-594). Narrowing the
flag would have silently narrowed that to "did the LAST run fail", losing the
lesson from any session that failed and then recovered. ``_had_any_failure`` is
the session-level record ``_loop`` now keeps for it, so both questions have an
answer and neither borrows the other's.
"""

from __future__ import annotations

import json
import queue
import time

import pytest

from localm.plugins.coder.audit import SessionMode


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


# The per-tool circuit breaker aborts after 4 identical failures, so this is a
# run that fails for REAL through the real loop - no hand-set flags.
_BREAKER_RUN = [_tc("read_file", path="missing.txt")] * 4


def _agent(tmp_path, responses, **kw):
    from localm.plugins.coder.agent import Agent
    kw.setdefault("auto_approve", True)
    kw.setdefault("self_verify", False)
    kw.setdefault("max_turns", 10)
    kw.setdefault("mode", SessionMode.LOG)
    return Agent(_Scripted(responses), cwd=tmp_path, **kw)


# --------------------------------------------------------------------------- #
#  The regression: a clean turn after a failed one reports ok                  #
# --------------------------------------------------------------------------- #

def test_clean_run_after_a_failed_run_reports_ok(home, tmp_path):
    """THE REGRESSION, at the Agent level. Two real runs: the first trips the
    circuit breaker, the second answers cleanly. The second must report ok."""
    agent = _agent(tmp_path, _BREAKER_RUN + ["All clean now."])

    first = agent.run_task("read the missing file")
    assert "circuit breaker" in first
    assert agent.last_run_ok is False

    second = agent.run_task("now just say something")
    assert second == "All clean now."
    assert agent.last_run_ok is True, (
        "a clean run after a failed one still reported failure - _loop never "
        "re-armed _last_run_ok, so one bad turn poisoned the whole session")


def test_gui_final_event_reports_the_clean_turn_as_ok(tmp_path):
    """THE USER-VISIBLE SYMPTOM, through the real GUI session path: the done
    event's "ok" comes straight from agent.last_run_ok, once per turn."""
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
    """The same one-way-flag defect via the stop path rather than a breaker: a
    user stop clears _last_run_ok too, and the next turn must still report ok."""
    agent = _agent(tmp_path, ["partial answer", "All clean now."])
    original_chat = agent.backend.chat

    def _chat_then_stop(messages, **kw):
        out = original_chat(messages, **kw)
        agent.request_stop()
        return out

    agent.backend.chat = _chat_then_stop
    assert agent.run_task("long thing") == "partial answer"
    assert agent.last_run_ok is False and agent._user_stopped is True

    agent.backend.chat = original_chat
    assert agent.run_task("something else") == "All clean now."
    assert agent.last_run_ok is True
    assert agent._user_stopped is False


# --------------------------------------------------------------------------- #
#  The complication: the session-level failure lesson must NOT be lost         #
# --------------------------------------------------------------------------- #

def test_session_that_failed_then_recovered_still_reflects_at_close(home, tmp_path):
    """REG-594's feature must survive the narrowing: the failed first run is
    still a lesson at close, even though the session ended on a clean run and
    wrote no files. Reading only the (now per-run) _last_run_ok would drop it."""
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
    # The session DID complete in the end, so the episode says so; the lesson is
    # carried by what_failed, not by mislabelling the outcome.
    assert eps[0].outcome == "ok"


def test_all_clean_session_reflects_nothing(home, tmp_path):
    """The other half: the new session-level flag must not arm itself on a
    healthy session, or every quit would pay for a model reflection."""
    agent = _agent(tmp_path, ["All clean now."])
    agent._episode_task = "say something"

    agent.run_task("say something")
    agent.run_task("say something else")
    assert agent._had_any_failure is False

    agent.close()
    assert agent.backend.reflections == []
    assert agent._episode_store.all() == []


def test_failed_only_session_still_reports_incomplete(home, tmp_path):
    """Unchanged behaviour for the single-run failure: still not ok, still
    reflects, still recorded as an incomplete session."""
    agent = _agent(tmp_path, _BREAKER_RUN)
    agent._episode_task = "read the missing file"

    agent.run_task("read the missing file")
    assert agent.last_run_ok is False and agent._had_any_failure is True

    agent.close()
    assert len(agent.backend.reflections) == 1
    assert agent._episode_store.all()[0].outcome == "incomplete"


def test_reset_clears_the_session_failure_marker(home, tmp_path):
    """/clear drops the history and error trace a lesson would be built from, so
    it drops the marker too rather than leaving a reflection armed for a run
    whose context is gone."""
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
