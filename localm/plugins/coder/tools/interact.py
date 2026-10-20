# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tools that talk to the user or switch plan mode: ask_user, enter_plan_mode,
exit_plan_mode.

Each receives the hidden ``_session`` (the calling Agent) and ``_interactive``
(whether that agent owns a terminal) arguments from ``_execute_tool``. A
terminal session asks on the terminal. A session with a ``confirm_handler`` and
no terminal (the GUI) gets plan approval through that handler as an
approve/reject card; it cannot carry a typed answer, so ask_user is unavailable
there. A run with neither returns an error telling the model to decide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import ToolResult

MAX_QUESTION = 2000
MAX_OPTIONS = 9
MAX_OPTION = 200
MAX_ANSWER = 4000
MAX_PLAN = 20000

_APPROVE = ("y", "yes", "approve", "ok")
_AUTORUN = ("a", "auto", "autorun", "auto-run")
_REJECT = ("n", "no")


def _channel(session: Any, interactive: bool) -> str:
    """"terminal", "handler" or "none": how *session* can reach the user."""
    if interactive:
        return "terminal"
    if getattr(session, "confirm_handler", None) is not None:
        return "handler"
    return "none"


def _top_level_only(session: Any, tool: str) -> Optional[ToolResult]:
    """An error for a sub-agent calling *tool*, else None."""
    if session is None:
        return ToolResult.error(f"{tool} is not available here: no session.")
    if getattr(session, "parent", None) is not None:
        return ToolResult.error(
            f"{tool} is not available to a sub-agent. Decide yourself, state "
            "the assumption in your final answer, and let the parent agent "
            "take it to the user.")
    return None


def _read_answer(prompt: str) -> Optional[str]:
    """One line from the terminal, or None on end of input. A Ctrl-C
    propagates and interrupts the turn."""
    from .. import display as _display
    try:
        return _display.console.input(prompt)
    except EOFError:
        return None


def tool_ask_user(cwd: Path, question=None, options=None,
                  _session=None, _interactive: bool = False) -> ToolResult:
    """Ask the user *question*, optionally offering numbered *options*, and
    return the answer."""
    blocked = _top_level_only(_session, "ask_user")
    if blocked is not None:
        return blocked
    if not isinstance(question, str) or not question.strip():
        return ToolResult.error(
            "ask_user needs a non-empty 'question' string.")
    question = question.strip()[:MAX_QUESTION]
    opts: list[str] = []
    if options is not None:
        if isinstance(options, str):
            options = [line for line in options.splitlines() if line.strip()]
        if not isinstance(options, list) or not all(
                isinstance(o, (str, int, float)) for o in options):
            return ToolResult.error(
                "ask_user 'options' must be a list of short answer strings.")
        opts = [" ".join(str(o).split())[:MAX_OPTION] for o in options]
        opts = [o for o in opts if o]
        if len(opts) > MAX_OPTIONS:
            return ToolResult.error(
                f"ask_user takes at most {MAX_OPTIONS} options, got {len(opts)}.")

    channel = _channel(_session, _interactive)
    if channel == "handler":
        return ToolResult.error(
            "ask_user is not available in this session: it has no way to "
            "collect a typed answer. Decide yourself, state the assumption, "
            "or ask the question in your reply and stop.")
    if channel == "none":
        return ToolResult.error(
            "ask_user is not available: this run is unattended and nobody can "
            "answer. Decide yourself and state the assumption in your final "
            "answer.")

    from .. import display as _display
    console = _display.console
    console.print()
    console.print(_display.Panel(
        _display._sanitized_text(question), title="Question from the agent",
        border_style="cyan", expand=False))
    for i, opt in enumerate(opts, 1):
        console.print(f"  [bold]{i}[/bold]. {_display.safe_markup(opt)}")
    hint = "number or your own answer" if opts else "your answer"
    raw = _read_answer(_display._sanitized_text(f"  Answer ({hint}): ",
                                                style="cyan"))
    if raw is None:
        return ToolResult.error(
            "ask_user got no answer: the terminal input ended. Decide "
            "yourself and state the assumption.")
    answer = raw.strip()[:MAX_ANSWER]
    if not answer:
        return ToolResult.success(
            "The user gave no answer. Decide yourself and state the "
            "assumption.", summary="no answer")
    if opts and answer.isdigit() and 1 <= int(answer) <= len(opts):
        answer = opts[int(answer) - 1]
    return ToolResult.success(f"The user answered: {answer}",
                              summary=f"answer: {answer[:60]}")


def tool_enter_plan_mode(cwd: Path, reason=None,
                         _session=None, _interactive: bool = False) -> ToolResult:
    """Switch plan mode on."""
    blocked = _top_level_only(_session, "enter_plan_mode")
    if blocked is not None:
        return blocked
    if _channel(_session, _interactive) == "none":
        return ToolResult.error(
            "enter_plan_mode is not available: this run is unattended, so "
            "nobody could approve a plan. Carry on with the task.")
    if _session.plan_mode:
        return ToolResult.success(
            "Plan mode is already on. Investigate with read-only tools, then "
            "call exit_plan_mode with your plan.", summary="already in plan mode")
    _session.set_plan_mode(True)
    if _interactive:
        from .. import display as _display
        _display.print_info("  Plan mode on: the agent can only read until you "
                            "approve its plan (/plan off to leave).")
    from ..agent.planmode import PLAN_MODE_NOTE
    return ToolResult.success(PLAN_MODE_NOTE, summary="plan mode on")


def _approve_on_terminal(plan: str) -> tuple[str, str]:
    """Show *plan* and ask for a verdict. Returns (verdict, feedback) with
    verdict "approve", "autorun" or "revise"."""
    from .. import display as _display
    console = _display.console
    console.print()
    console.print(_display.Panel(
        _display.Markdown(_display._strip_esc(plan)), title="Proposed plan",
        border_style="cyan"))
    console.print(
        "  [bold]y[/bold] approve   "
        "[bold]a[/bold] approve and auto-run until verified   "
        "[bold]n[/bold] or type what to change: revise")
    raw = _read_answer(_display._sanitized_text("  Plan? [y/a/n or feedback]: ",
                                                style="yellow"))
    if raw is None:
        return "revise", ""
    answer = raw.strip()
    low = answer.lower()
    if low in _APPROVE:
        return "approve", ""
    if low in _AUTORUN:
        return "autorun", ""
    if low in _REJECT:
        return "revise", ""
    for prefix in ("n:", "no:", "n -", "no -"):
        if low.startswith(prefix):
            return "revise", answer[len(prefix):].strip()[:MAX_ANSWER]
    return "revise", answer[:MAX_ANSWER]


def tool_exit_plan_mode(cwd: Path, plan=None,
                        _session=None, _interactive: bool = False) -> ToolResult:
    """Present *plan* for approval; leave plan mode when it is approved."""
    blocked = _top_level_only(_session, "exit_plan_mode")
    if blocked is not None:
        return blocked
    if not _session.plan_mode:
        return ToolResult.error(
            "exit_plan_mode is only for plan mode, which is off. Carry on "
            "with the task.")
    if not isinstance(plan, str) or not plan.strip():
        return ToolResult.error(
            "exit_plan_mode needs the whole plan as a non-empty markdown "
            "'plan' string.")
    plan = plan.strip()
    if len(plan) > MAX_PLAN:
        return ToolResult.error(
            f"The plan is {len(plan)} characters; keep it under {MAX_PLAN} "
            "and call exit_plan_mode again.")

    channel = _channel(_session, _interactive)
    if channel == "terminal":
        verdict, feedback = _approve_on_terminal(plan)
    elif channel == "handler":
        from ..confirm import invoke_confirm
        from ..parser import ToolCall
        call = ToolCall(name="exit_plan_mode", args={"plan": plan},
                        raw="", start=0, end=0)
        approved = invoke_confirm(_session.confirm_handler, call, agent=None)
        verdict, feedback = ("approve" if approved else "revise"), ""
    else:
        return ToolResult.error(
            "exit_plan_mode cannot get an approval: this run is unattended. "
            "Plan mode stays on; give the plan as your final answer.")

    if verdict == "revise":
        why = (f" The user's feedback: {feedback}" if feedback
               else " The user gave no reason; ask in your reply what to "
                    "change, or revise the plan.")
        return ToolResult.success(
            "The user did not approve the plan. Plan mode stays on." + why +
            " Revise the plan (investigate more if needed) and call "
            "exit_plan_mode again.", summary="plan not approved")

    _session.set_plan_mode(False)
    if verdict == "autorun":
        cmd = _session._begin_plan_autorun()
        if cmd is not None:
            from ..verify import command_text
            label = command_text(cmd)
            text = (
                "The user approved the plan and switched on auto-run. Plan "
                "mode is off. Carry the plan out now: tool calls run without "
                "asking for the rest of this task (the shell safety gate and "
                f"tools set to always confirm still ask), and `{label}` must "
                "exit 0 before the task finishes.")
        else:
            text = (
                "The user approved the plan and switched on auto-run. Plan "
                "mode is off. Carry the plan out now: tool calls run without "
                "asking for the rest of this task (the shell safety gate and "
                "tools set to always confirm still ask). No verification command was "
                "found for this project, so run the tests yourself before "
                "finishing.")
        if _interactive:
            from .. import display as _display
            _display.print_info(
                "  Plan approved, auto-run on for this task"
                + (f" (until `{_display.safe_markup(label)}` passes)."
                   if cmd is not None else " (no verification check found)."))
        return ToolResult.success(text, summary="plan approved, auto-run")
    if _interactive:
        from .. import display as _display
        _display.print_info("  Plan approved; plan mode off.")
    return ToolResult.success(
        "The user approved the plan. Plan mode is off. Carry the plan out "
        "now, step by step.", summary="plan approved")
