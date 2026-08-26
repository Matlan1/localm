# SPDX-License-Identifier: AGPL-3.0-or-later
"""A spawn_agent child must still be able to do destructive work in the default
interactive REPL.

Inheriting the parent's confirmation posture - ``auto_approve`` / ``dry_run`` /
``always_confirm`` / ``confirm_handler`` instead of hardcoding the child's
``auto_approve=True`` - is what stops a child BYPASSING that posture. Inherited
naively it also blocks every delegation.

In the default interactive REPL (``localm coder`` without ``--yes``) the CLI
builds the parent with ``auto_approve=False`` and ``confirm_handler=None`` - the
terminal REPL confirms via ``_confirm_tool``, not a handler. A child inheriting
``auto_approve=False`` AND ``confirm_handler=None`` always runs ``run_task`` ->
``_loop(interactive=False)``, so in execution.py it hits ``needs_confirm=True``
with no handler and ``interactive=False``, takes the fail-closed branch, and
DENIES every write/shell/git - even though the user is sitting right there and
the parent's own tools prompt fine on that same terminal.

"Inherit the parent's confirmation posture" means ASK, not block. So the child
inherits the parent's ACTUAL confirmation CHANNEL: the GUI handler when there is
one, and otherwise the parent's terminal prompt when the parent is genuinely
running an interactive loop.

The security property must survive intact, which is what the negatives here pin:
a parent that is NOT interactive and has no handler (a scheduled/unattended run)
must still fail closed, and a child must never self-approve.
"""

from unittest.mock import patch

import pytest

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import tool_spawn_agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _write_call(rel="child_output.txt"):
    return ToolCall(name="write_file", args={"path": rel, "content": "hi"},
                    raw="", start=0, end=0)


def _terminal_answer(answer, asked=None):
    """Patch the parent's REAL terminal prompt for a write_file confirmation.

    _confirm_tool routes write_file/edit_file through print_diff_preview +
    confirm_diff (a diff preview, not a bare y/N), so a test that only patched
    `confirm` would let the real prompt block on stdin. Non-write tools use
    `confirm`, patched separately where that matters.
    """
    def _confirm_diff(label):
        if asked is not None:
            asked.append(label)
        return answer
    return patch.multiple(
        "localm.plugins.coder.agent",
        confirm_diff=_confirm_diff,
        print_diff_preview=lambda old, new, path_label="": None,
    )


def _terminal_must_not_be_used(why):
    """Assert NOTHING reaches the terminal prompt. Guards BOTH channels -
    confirm_diff (write/edit) and confirm (everything else) - so a leak through
    either one fails loudly instead of blocking on real stdin."""
    def _boom(*a, **kw):
        pytest.fail(why)
    return patch.multiple(
        "localm.plugins.coder.agent",
        confirm=_boom, confirm_diff=_boom, print_diff_preview=_boom,
    )


def _spawn_and_capture_child(tmp_path, parent_kwargs, *, interactive_parent=False):
    """Spawn a real child via tool_spawn_agent, short-circuiting run_task so no
    LLM call is needed, and return the constructed child Agent.

    *interactive_parent* marks the parent as running an interactive REPL loop the
    way ``_loop(interactive=True)`` does, which is the state the CLI parent is
    actually in when the model calls spawn_agent.
    """
    parent = Agent(_StubBackend(), cwd=tmp_path, **parent_kwargs)
    if interactive_parent:
        parent._interactive = True
    captured = {}

    def _fake_run_task(self, task):
        captured["child"] = self
        return "child done"

    with patch.object(Agent, "run_task", _fake_run_task):
        result = tool_spawn_agent(tmp_path, "do work", _parent_agent=parent)
    assert result.ok, result.output
    return captured["child"]


# --------------------------------------------------------------------------- #
#  An interactive parent can delegate real work                                #
# --------------------------------------------------------------------------- #

class TestInteractiveDelegation:
    def test_child_asks_the_terminal_and_executes_when_approved(self, tmp_path):
        """`localm coder` without --yes, model delegates via spawn_agent, user
        approves at the terminal -> the work actually happens."""
        asked = []
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False}, interactive_parent=True)
        with _terminal_answer(True, asked):
            res = child._execute_tool(_write_call(), interactive=False)

        assert res.ok, res.output
        assert (tmp_path / "child_output.txt").read_text() == "hi"
        assert asked, "the child must ASK the user at the terminal, not hard-deny"

    def test_child_terminal_prompt_denies_when_user_says_no(self, tmp_path):
        """The other half of ASK: a 'no' at the terminal rejects the tool - and
        it must read as a user rejection, not as a fail-closed 'no channel'."""
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False}, interactive_parent=True)
        with _terminal_answer(False):
            res = child._execute_tool(_write_call(), interactive=False)

        assert not res.ok
        assert "rejected" in res.output.lower()
        assert not (tmp_path / "child_output.txt").exists()

    def test_child_shell_is_asked_not_denied_in_interactive_repl(self, tmp_path):
        """run_shell is the tool the finding names: delegated shell work must be
        approvable, not hard-denied."""
        call = ToolCall(name="run_shell", args={"command": "echo hi"},
                        raw="", start=0, end=0)
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False}, interactive_parent=True)
        seen = []
        with patch("localm.plugins.coder.agent.confirm",
                   lambda prompt: seen.append(prompt) or True):
            res = child._execute_tool(call, interactive=False)

        assert seen, "delegated run_shell must reach the terminal prompt"
        assert "requires confirmation" not in res.output.lower()

    def test_diff_preview_resolves_against_the_shared_cwd(self, tmp_path):
        """The parent's _confirm_tool renders the write_file diff against
        `self.cwd`. Parent and child share one cwd (spawn_agent passes the
        parent's), so the preview must show the child's real target file."""
        (tmp_path / "child_output.txt").write_text("before")
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False}, interactive_parent=True)
        seen = {}

        def _capture(old, new, path_label=""):
            seen["old"], seen["new"], seen["label"] = old, new, path_label

        with patch("localm.plugins.coder.agent.print_diff_preview", _capture), \
             patch("localm.plugins.coder.agent.confirm_diff", lambda label: True):
            res = child._execute_tool(_write_call(), interactive=False)

        assert res.ok, res.output
        assert seen["old"] == "before" and seen["new"] == "hi"
        assert seen["label"] == "child_output.txt"


# --------------------------------------------------------------------------- #
#  NEGATIVES: the bypass fix must survive intact                               #
# --------------------------------------------------------------------------- #

class TestConfirmationPostureStillEnforced:
    def test_unattended_parent_without_a_channel_still_fails_closed(self, tmp_path):
        """A parent that requires confirmation but is NOT interactive and has no
        handler (a scheduled/unattended run) has no way to ask. The child must
        still DENY - never self-approve. This is the original security fix."""
        child = _spawn_and_capture_child(tmp_path, {"auto_approve": False})
        res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok
        assert "confirmation" in res.output.lower() or "denied" in res.output.lower()
        assert not (tmp_path / "child_output.txt").exists()

    def test_gui_confirm_handler_still_wins_over_the_terminal(self, tmp_path):
        """A GUI/web parent passes confirm_handler; that must remain the channel
        (routing to the browser), never the server's terminal."""
        seen = []

        def deny(call):
            seen.append(call.name)
            return False

        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False, "confirm_handler": deny},
            interactive_parent=True)      # even if marked interactive
        with _terminal_must_not_be_used("a GUI parent must never prompt the "
                                        "server's terminal"):
            res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok
        assert seen == ["write_file"]

    def test_child_still_respects_parent_dry_run(self, tmp_path):
        child = _spawn_and_capture_child(
            tmp_path, {"dry_run": True}, interactive_parent=True)
        res = child._execute_tool(_write_call(), interactive=False)
        assert res.ok
        assert "dry-run" in res.output.lower()
        assert not (tmp_path / "child_output.txt").exists()

    def test_child_still_inherits_always_confirm_and_asks(self, tmp_path):
        """always_confirm on an auto_approve parent must still force a prompt;
        with an interactive parent that prompt is now answerable."""
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": True, "always_confirm": {"write_file"}},
            interactive_parent=True)
        asked = []
        with _terminal_answer(False, asked):
            res = child._execute_tool(_write_call(), interactive=False)
        assert not res.ok
        assert asked, "always_confirm must still force the prompt"

    def test_baseline_auto_approve_parent_needs_no_prompt(self, tmp_path):
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": True}, interactive_parent=True)
        with _terminal_must_not_be_used("must not prompt when auto-approving"):
            res = child._execute_tool(_write_call(), interactive=False)
        assert res.ok, res.output
        assert (tmp_path / "child_output.txt").exists()

    def test_restricted_parent_child_cannot_reach_shell_at_all(self, tmp_path):
        """Belt-and-suspenders: the confirmation channel must not become a way to
        approve a tool a restricted session may never run. disabled_tools is the
        hard gate and it runs BEFORE any confirmation."""
        call = ToolCall(name="run_shell", args={"command": "echo hi"},
                        raw="", start=0, end=0)
        child = _spawn_and_capture_child(
            tmp_path, {"auto_approve": False, "restricted": True},
            interactive_parent=True)
        with _terminal_must_not_be_used("a restricted child must not even ask"):
            res = child._execute_tool(call, interactive=False)
        assert not res.ok


# --------------------------------------------------------------------------- #
#  The parent's interactive flag is real, not test-only                        #
# --------------------------------------------------------------------------- #

def test_loop_records_the_interactive_flag_on_the_agent(tmp_path):
    """_interactive must be set by the real loop (that is what makes the CLI
    parent's terminal reachable), and default False on a fresh Agent."""
    agent = Agent(_StubBackend(), cwd=tmp_path)
    assert agent._interactive is False

    seen = {}

    def _fake_call_llm(self, messages, interactive):
        seen["interactive"] = self._interactive
        raise RuntimeError("stop the loop here")

    with patch.object(Agent, "_call_llm", _fake_call_llm):
        with pytest.raises(RuntimeError):
            agent.chat("hi")            # chat() -> _loop(interactive=True)
    assert seen["interactive"] is True

    with patch.object(Agent, "_call_llm", _fake_call_llm):
        with pytest.raises(RuntimeError):
            agent.run_task("hi")        # run_task() -> _loop(interactive=False)
    assert seen["interactive"] is False
