# SPDX-License-Identifier: AGPL-3.0-or-later
"""A sub-agent's label reaches the human who has to approve its tool call.

Concurrent children's confirmations are serialised and the asker is named on both
surfaces. These tests drive the REAL chain - real Agent instances, the real
``_execute_tool`` confirmation branch, the real serialising wrapper, a real
``CoderSession`` and its real event queue - and pin both halves: the GUI prompt
CARRIES the label, and the terminal keeps its announcement line.

Only the LLM backend is substituted (no model call is ever made here), the same
shape as test_spawn_agent_confirmation.py.

Every labelling assertion has a FIRES-CONTROL: the same assertion against the
top-level agent's OWN prompt, which must stay unlabelled, so a build that
stamped a label onto every prompt fails here.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.confirm import handler_accepts_agent, invoke_confirm
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.sessions import CoderSession
from localm.plugins.coder.tools import parallel as par
from localm.plugins.coder.tools import tool_spawn_agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _write_call(rel: str = "child_output.txt") -> ToolCall:
    return ToolCall(name="write_file", args={"path": rel, "content": "hi"},
                    raw="", start=0, end=0)


class _Call:
    """A ToolCall-shaped object for the handler-protocol tests below."""

    def __init__(self, name: str = "run_shell", **args) -> None:
        self.name = name
        self.args = args


# --------------------------------------------------------------------------
# The protocol: an optional keyword, never a breaking one
# --------------------------------------------------------------------------

def test_a_legacy_one_argument_handler_is_still_called_unchanged():
    """Third-party handlers predate the extension and must keep working."""
    seen = []

    def legacy(call):                      # the original public signature
        seen.append(call.name)
        return True

    assert invoke_confirm(legacy, _Call("write_file"), agent="child1") is True
    assert seen == ["write_file"]
    assert handler_accepts_agent(legacy) is False


@pytest.mark.parametrize("handler, accepts", [
    (lambda call: True, False),
    (lambda call, agent=None: True, True),
    (lambda call, **kw: True, True),
    (lambda call, *, agent=None: True, True),
])
def test_only_handlers_that_opt_in_are_offered_the_label(handler, accepts):
    assert handler_accepts_agent(handler) is accepts


def test_a_handler_is_invoked_exactly_once_even_when_it_raises_typeerror():
    """A TypeError from INSIDE a handler must not be read as a signature mismatch.

    A try/except fallback would re-invoke a handler that may already have
    prompted, asking the human twice and taking the second answer. Signature
    inspection is used instead: one call, and the error propagates.
    """
    calls = []

    def broken(call, agent=None):
        calls.append(agent)
        raise TypeError("something inside the handler went wrong")

    with pytest.raises(TypeError):
        invoke_confirm(broken, _Call(), agent="child1")
    assert calls == ["child1"], "handler was retried after its own TypeError"


def test_the_answer_is_relayed_verbatim_and_never_invented():
    for answer in (True, False):
        assert invoke_confirm(lambda call: answer, _Call()) is answer
        assert invoke_confirm(lambda call, agent=None: answer, _Call(),
                              agent="child1") is answer


def test_a_callable_without_an_introspectable_signature_falls_back_safely():
    """No signature to read -> treat it as the original protocol, never guess.

    ``type`` and ``dict`` raise ValueError from inspect.signature on CPython;
    ``len`` does NOT (it reports ``(obj, /)``), so it exercises the ordinary
    no-such-parameter path instead. Both are pinned, so the except branch is
    genuinely entered.
    """
    import inspect
    for uninspectable in (type, dict):
        with pytest.raises((TypeError, ValueError)):
            inspect.signature(uninspectable)
        assert handler_accepts_agent(uninspectable) is False
    assert handler_accepts_agent(len) is False


def test_a_handler_whose_agent_argument_is_required_works_unlabelled_too():
    """The keyword is passed whenever the handler accepts it, value or not.

    Passing it only when a label EXISTS would make the call shape depend on the
    value, so a handler written ``def handler(call, agent)`` would work for a
    sub-agent's prompt and raise TypeError on the top-level agent's own.
    """
    seen = []

    def strict(call, agent):            # no default: legal, must not break
        seen.append(agent)
        return True

    assert invoke_confirm(strict, _Call(), agent="child1") is True
    assert invoke_confirm(strict, _Call()) is True
    assert seen == ["child1", None]


# --------------------------------------------------------------------------
# Who the label comes from
# --------------------------------------------------------------------------

def test_a_sub_agent_labels_its_prompts_and_the_top_level_agent_does_not(tmp_path):
    parent = Agent(_StubBackend(), cwd=tmp_path, name="localcoder")
    child = Agent(_StubBackend(), cwd=tmp_path, name="reviewer", parent=parent)
    assert child._confirm_agent_label() == "reviewer"
    assert parent._confirm_agent_label() is None


@pytest.mark.parametrize("empty", ["", None])
def test_a_child_with_no_usable_name_is_still_marked_as_a_sub_agent(tmp_path, empty):
    """``spawn_agent``'s name comes from the MODEL, so it can arrive empty.

    A falsy label would collapse to "no label" and make a delegated request look
    exactly like the user's own, so an unnamed child still gets a generic
    sub-agent marker.
    """
    parent = Agent(_StubBackend(), cwd=tmp_path)
    child = Agent(_StubBackend(), cwd=tmp_path, name=empty, parent=parent)
    assert child._confirm_agent_label() == "sub-agent"
    # The top-level agent is unlabelled, not a sub-agent.
    assert parent._confirm_agent_label() is None


def test_a_child_named_by_the_model_reaches_the_gui_even_when_the_name_is_empty(tmp_path):
    """The same gap end to end: an empty-named child must not look like the user."""
    session = _gui_session(tmp_path)
    try:
        _auto_answer(session)
        captured = {}

        def _fake_run_task(self, task):
            captured["child"] = self
            return "done"

        from unittest.mock import patch
        with patch.object(Agent, "run_task", _fake_run_task):
            res = tool_spawn_agent(tmp_path, "do work", name="",
                                   _parent_agent=session.agent)
        assert res.ok, res.output
        assert captured["child"]._execute_tool(_write_call(), interactive=False).ok
        assert [r["agent"] for r in _requests(session)] == ["sub-agent"]
    finally:
        session.close()


def test_a_real_child_reaches_the_parents_handler_with_its_own_name(tmp_path):
    """End to end through the REAL confirmation branch of a REAL spawned child."""
    seen = []

    def gui_shaped_handler(call, agent=None):
        seen.append((call.name, agent))
        return False                # reject: this tests routing, not the write

    parent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=False,
                   confirm_handler=gui_shaped_handler)
    captured = {}

    def _fake_run_task(self, task):
        captured["child"] = self
        return "child done"

    from unittest.mock import patch
    with patch.object(Agent, "run_task", _fake_run_task):
        res = tool_spawn_agent(tmp_path, "do work", name="reviewer",
                               _parent_agent=parent)
    assert res.ok, res.output

    child = captured["child"]
    assert not child._execute_tool(_write_call(), interactive=False).ok
    # The parent's own identical call carries no child label.
    assert not parent._execute_tool(_write_call(), interactive=False).ok

    assert seen == [("write_file", "reviewer"), ("write_file", None)]
    assert not (tmp_path / "child_output.txt").exists()


# --------------------------------------------------------------------------
# The GUI surface: the label reaches the browser event
# --------------------------------------------------------------------------

def _gui_session(cwd: Path) -> CoderSession:
    """A real GUI session: real Agent, real event queue, real confirm plumbing."""
    return CoderSession(cwd, _StubBackend(), auto_approve=False, auto_verify=False)


def _auto_answer(session: CoderSession, approved: bool = True) -> threading.Thread:
    """Answer confirmations the way the browser does: by confirm id, off the feed."""
    def _run():
        deadline = threading.Event()
        for _ in range(500):
            with session._lock:
                pending = list(session._pending)
            for confirm_id in pending:
                session.answer_confirm(confirm_id, approved)
            if pending:
                return
            deadline.wait(0.01)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _requests(session: CoderSession) -> list[dict]:
    """confirm_request events from the session's real replay buffer."""
    return [e for e in session.history if e.get("type") == "confirm_request"]


def test_the_gui_prompt_carries_the_asking_childs_label(tmp_path):
    """THE FIX. A child's confirmation reaches the browser NAMING the child."""
    session = _gui_session(tmp_path)
    try:
        _auto_answer(session)
        # == not is: a bound method is a fresh object on every attribute access.
        assert session.agent.confirm_handler == session._confirm
        assert session._confirm(_write_call(), agent="child1") is True

        assert len(_requests(session)) == 1
        assert _requests(session)[0]["agent"] == "child1", (
            "the GUI approval card cannot name the asking child")
    finally:
        session.close()


def test_fires_control_the_parents_own_gui_prompt_carries_no_child_label(tmp_path):
    """THE CONTROL. Same session, same call, no sub-agent: no label."""
    session = _gui_session(tmp_path)
    try:
        _auto_answer(session)
        assert session._confirm(_write_call()) is True

        assert len(_requests(session)) == 1
        assert _requests(session)[0]["agent"] is None
    finally:
        session.close()


def test_a_real_child_of_a_gui_session_is_named_on_the_browser_event(tmp_path):
    """The whole chain, unmocked: child Agent -> confirm chain -> GUI event.

    The child inherits the session's handler, and its identity must survive the
    trip to the browser event.
    """
    session = _gui_session(tmp_path)
    try:
        _auto_answer(session)
        captured = {}

        def _fake_run_task(self, task):
            captured["child"] = self
            return "done"

        from unittest.mock import patch
        with patch.object(Agent, "run_task", _fake_run_task):
            res = tool_spawn_agent(tmp_path, "do work", name="tester",
                                   _parent_agent=session.agent)
        assert res.ok, res.output

        assert captured["child"]._execute_tool(_write_call(), interactive=False).ok
        assert [r["agent"] for r in _requests(session)] == ["tester"]
    finally:
        session.close()


def test_the_serialised_parallel_wrapper_names_the_child_on_the_gui_event(tmp_path):
    """The parallel-dispatch path specifically: wrapper -> GUI session -> event."""
    session = _gui_session(tmp_path)
    try:
        _auto_answer(session)
        handler = par._serialised_confirm_handler(session.agent, "child2")
        assert handler is not None
        assert handler(_write_call()) is True

        assert [r["agent"] for r in _requests(session)] == ["child2"]
    finally:
        session.close()


def test_an_always_allowed_tool_still_says_which_child_used_the_grant(tmp_path):
    session = _gui_session(tmp_path)
    try:
        session.allowed_tools.add("run_shell")
        assert session._confirm(_Call("run_shell"), agent="child2") is True
        assert session._confirm(_Call("run_shell")) is True

        infos = [e["text"] for e in session.history if e.get("type") == "info"]
        assert any("sub-agent 'child2'" in t for t in infos)
        # The parent's own auto-approval names no child.
        assert any(t.startswith("run_shell auto-approved") for t in infos)
    finally:
        session.close()


def test_a_timed_out_child_confirmation_says_which_child_it_was(tmp_path, monkeypatch):
    """An unanswered prompt is rejected; the notice must still be attributable."""
    monkeypatch.setattr("localm.plugins.coder.sessions._confirm_timeout",
                        lambda: 0.05)
    session = _gui_session(tmp_path)
    try:
        assert session._confirm(_write_call(), agent="child3") is False
        infos = [e["text"] for e in session.history if e.get("type") == "info"]
        assert any("child3" in t and "timed out" in t for t in infos)

        assert session._confirm(_write_call()) is False
        assert any(t == "Confirmation for write_file timed out - rejected."
                   for t in [e["text"] for e in session.history
                             if e.get("type") == "info"])
    finally:
        session.close()


# --------------------------------------------------------------------------
# The terminal surface keeps the attribution it already had
# --------------------------------------------------------------------------

def test_the_console_announcement_still_names_the_child(monkeypatch):
    """The terminal half: the announcement line still precedes the prompt."""
    printed: list[str] = []
    order: list[str] = []

    class _RecordingConsole:
        def print(self, text, *a, **kw):
            printed.append(str(text))
            order.append("announce")

    monkeypatch.setattr("localm.plugins.coder.display.console", _RecordingConsole())

    def terminal_handler(call):        # one-argument, like the REPL's _confirm_tool
        order.append("prompt")
        return True

    class _Parent:
        confirm_handler = staticmethod(terminal_handler)
        _interactive = False

    handler = par._serialised_confirm_handler(_Parent(), "child1")
    assert handler(_Call("run_shell")) is True

    assert any("child1" in line and "run_shell" in line for line in printed)
    assert order == ["announce", "prompt"], "attribution must precede the prompt"


def test_the_wrapper_still_denies_when_the_channel_stays_busy(monkeypatch):
    """The fail-closed timeout branch is untouched by the labelling."""
    monkeypatch.setattr(par, "_CONFIRM_WAIT_S", 0.05)

    class _Parent:
        confirm_handler = staticmethod(lambda call, agent=None: True)
        _interactive = False

    handler = par._serialised_confirm_handler(_Parent(), "child1")
    par._CONFIRM_LOCK.acquire()
    try:
        done: list[bool] = []
        t = threading.Thread(target=lambda: done.append(handler(_Call())))
        t.start()
        t.join(timeout=10)
        assert done == [False], "a busy channel must DENY, not approve"
    finally:
        par._CONFIRM_LOCK.release()
