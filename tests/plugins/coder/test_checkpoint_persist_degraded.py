# SPDX-License-Identifier: AGPL-3.0-or-later
"""CoderSession.persist_checkpoint() must not swallow a save failure silently
(AGENTS.md rule 5): a task that finishes fine but whose resume checkpoint
fails to save must still report ok=true (the task itself did not fail), but
the failure has to be VISIBLE - a warning event in the feed and a
"checkpoint_degraded" status on both session.info() and the task result -
rather than disappearing into a bare ``except Exception: pass``.

Privacy-mode and restricted sessions intentionally never persist a
checkpoint at all, so they must never report this status, even when the
underlying save call would itself raise if it were ever reached."""

import queue
import time

import pytest

from localm.plugins.coder.sessions import CoderSession

_WARNING_TEXT = "Task completed, but the resume checkpoint could not be saved."


@pytest.fixture
def make_session():
    """CoderSession factory that closes every session it made on teardown, so
    a "log"-mode session's audit-log file handle is released rather than
    leaking past the test."""
    made = []

    def _make(*a, **kw):
        session = CoderSession(*a, **kw)
        made.append(session)
        return session

    yield _make
    for session in made:
        session.close()


class ScriptedBackend:
    """Answers every chat() call with the same canned response."""

    model_id = "fake-model"
    last_usage = {"total_tokens": 7}
    native_tools = False
    supports_grammar = False

    def __init__(self, text="All done, nothing to do."):
        self.text = text

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        return self.text

    def chat_stream(self, messages, **kw):
        mid = max(1, len(self.text) // 2)
        yield self.text[:mid]
        yield self.text[mid:]


def _drain(session, *, until_types, timeout=10.0):
    """Collect events from the session queue until one of until_types appears."""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = session.events.get(timeout=0.2)
        except queue.Empty:
            continue
        events.append(ev)
        if ev["type"] in until_types:
            return events
    raise TimeoutError(f"No {until_types} event within {timeout}s: {events}")


def _boom():
    raise OSError("disk full")


def _position(history, event):
    """Index of *event* in *history* by IDENTITY, not value - two "final"
    events from two scripted turns can otherwise compare equal by content
    and make list.index() return the wrong one's position."""
    return next(i for i, e in enumerate(history) if e is event)


def test_checkpoint_save_failure_surfaces_a_warning_but_keeps_the_task_ok(tmp_path, make_session):
    session = make_session(tmp_path, ScriptedBackend(), auto_approve=True,
                           mode="log")
    session.agent.save_checkpoint = _boom

    assert session.send_message("say hi") == "started"
    events = _drain(session, until_types={"final"})
    session._thread.join(timeout=10)
    assert not session._thread.is_alive()

    final = events[-1]
    assert final["ok"] is True
    assert "All done" in final["text"]
    assert session.last_result["ok"] is True

    warnings = [e for e in session.history
                if e.get("type") == "info" and _WARNING_TEXT in e.get("text", "")]
    assert len(warnings) == 1, session.history
    assert _position(session.history, warnings[0]) > _position(session.history, final), \
        "the checkpoint warning must follow the task's own final event"

    assert session.checkpoint_degraded is True
    assert session.info()["checkpoint_degraded"] is True
    assert session.last_result["checkpoint_degraded"] is True


def test_checkpoint_degraded_clears_after_a_later_successful_save(tmp_path, make_session):
    session = make_session(tmp_path, ScriptedBackend(), auto_approve=True,
                           mode="log")
    real_save_checkpoint = session.agent.save_checkpoint
    session.agent.save_checkpoint = _boom

    session.send_message("say hi")
    _drain(session, until_types={"final"})
    session._thread.join(timeout=10)
    assert session.checkpoint_degraded is True

    session.agent.save_checkpoint = real_save_checkpoint
    session.send_message("say hi again")
    events = _drain(session, until_types={"final"})
    session._thread.join(timeout=10)

    assert events[-1]["ok"] is True
    assert session.checkpoint_degraded is False
    assert session.info()["checkpoint_degraded"] is False
    assert session.last_result["checkpoint_degraded"] is False
    second_final_pos = _position(session.history, events[-1])
    assert not any(_WARNING_TEXT in e.get("text", "")
                   for e in session.history[second_final_pos:])


def test_privacy_mode_session_never_reports_checkpoint_degraded(tmp_path, make_session):
    session = make_session(tmp_path, ScriptedBackend(), auto_approve=True,
                           mode="privacy")
    session.agent._messages = [{"role": "user", "content": "hi"}]
    session.agent.save_checkpoint = _boom       # would raise if ever called

    session.persist_checkpoint()

    assert session.checkpoint_degraded is False
    assert session.info()["checkpoint_degraded"] is False
    assert not any(_WARNING_TEXT in e.get("text", "") for e in session.history)


def test_restricted_session_never_reports_checkpoint_degraded(tmp_path, make_session):
    session = make_session(tmp_path, ScriptedBackend(), auto_approve=True,
                           mode="log", restricted=True)
    session.agent._messages = [{"role": "user", "content": "hi"}]
    session.agent.save_checkpoint = _boom       # would raise if ever called

    session.persist_checkpoint()

    assert session.checkpoint_degraded is False
    assert session.info()["checkpoint_degraded"] is False
    assert not any(_WARNING_TEXT in e.get("text", "") for e in session.history)
