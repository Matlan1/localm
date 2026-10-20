# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan mode: a read-only investigation mode that ends with a plan the user
approves.

While plan mode is on, ``_execute_tool`` refuses every tool outside
``PLAN_MODE_TOOLS``. The list is an allowlist, so a newly added, dynamically
registered (MCP, plugin, skill) or otherwise unclassified tool is refused until
it is added here. A listed tool is still refused when its registry entry is
destructive, asks by default, or carries externally reported docs.

The mode is left by the user (``/plan off``) or by an approved
``exit_plan_mode`` call (tools/interact.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# The tools that can run in plan mode: they read the project, the session's own
# state or a job's output, and none writes a file, runs a model-chosen command,
# reaches the network or starts an agent.
PLAN_MODE_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_dir", "tree", "grep", "search_files", "find_references",
    "git_status", "git_diff", "git_log",
    "check_shell_job", "check_agent_job",
    "read_env",
    "set_todos", "read_todos",
    "rag_list_collections",
    "list_skills", "use_skill",
    "ask_user", "enter_plan_mode", "exit_plan_mode",
})

# The tools that talk to the user or switch the mode. Each runs alone in its
# batch (never in the parallel pool) and receives the hidden ``_session`` and
# ``_interactive`` arguments from ``_execute_tool``.
INTERACTION_TOOLS: frozenset[str] = frozenset({
    "ask_user", "enter_plan_mode", "exit_plan_mode",
})

PLAN_MODE_NOTE = (
    "[plan mode] You are in plan mode. Investigate with read-only tools only "
    "(read_file, grep, search_files, tree, find_references, git_status, "
    "git_diff, git_log); every tool that writes a file, runs a command, uses "
    "the network or starts an agent is refused. Use ask_user to clarify a "
    "requirement you cannot settle from the code. When you have a concrete "
    "plan, call exit_plan_mode with the whole plan in markdown (files to "
    "change, the changes, how to verify them); the user approves it or asks "
    "for a revision. Do not describe edits as done while in plan mode."
)


def plan_mode_refusal(name: str, tool_def: Any) -> Optional[str]:
    """The refusal message for running tool *name* in plan mode, or None when
    it may run.

    *tool_def* is the tool's registry entry, or None for an unknown tool (the
    caller's unknown-tool error then applies, so None is returned).
    """
    if tool_def is None:
        return None
    flagged = (getattr(tool_def, "destructive", False) is True
               or getattr(tool_def, "ask_by_default", False) is True
               or getattr(tool_def, "untrusted_docs", False) is True)
    if name in PLAN_MODE_TOOLS and not flagged:
        return None
    return (
        f"'{name}' was not run: plan mode is on, and only read-only tools can "
        "run in plan mode (reading, searching, git status/diff/log, the task "
        "list, ask_user). Finish investigating, then call exit_plan_mode with "
        "your plan; once the user approves it, plan mode ends and this tool "
        "can run.")


class _PlanModeMixin:
    """Plan-mode state on the Agent."""

    if TYPE_CHECKING:
        # Set by Agent (core.py), which this class is mixed into.
        cwd: Path
        restricted: bool
        verify_cmd: Any
        _auto_approve: bool

        def _emit(self, event_type: str, **data) -> None: ...
        def _add_user(self, content) -> None: ...

    _plan_mode: bool = False
    # (auto_approve, verify_cmd) as they were before an approved plan switched
    # on auto-run, or None when auto-run is not active.
    _plan_autorun_saved: Optional[tuple] = None

    @property
    def plan_mode(self) -> bool:
        """Whether plan mode is on for this agent or any ancestor."""
        if self._plan_mode:
            return True
        parent = getattr(self, "parent", None)
        return bool(parent is not None and getattr(parent, "plan_mode", False))

    def set_plan_mode(self, on: bool) -> None:
        """Switch this agent's own plan mode on or off."""
        self._plan_mode = bool(on)
        self._emit("info", text="plan mode on" if on else "plan mode off")

    def _add_user_with_plan_note(self, content) -> None:
        """Add user message *content*, with the plan-mode reminder appended
        while plan mode is on."""
        if self.plan_mode:
            from localm.textguard import compose
            content = compose(content, "\n\n" + PLAN_MODE_NOTE)
        self._add_user(content)

    def _begin_plan_autorun(self) -> Optional[Any]:
        """Approve gated tools without asking and require the verification
        check for the rest of the current run.

        ``auto_approve`` is switched on (a shell-guard verdict, an
        ``always_confirm`` tool and a loosely formatted call still ask). When no
        verification command is set, the project's check is detected and used.
        Both are restored by ``_end_plan_autorun`` when the run ends. Returns
        the verification command in force, or None when there is none.
        """
        if self._plan_autorun_saved is None:
            self._plan_autorun_saved = (self._auto_approve, self.verify_cmd)
        self._auto_approve = True
        if self.verify_cmd is None and not self.restricted:
            try:
                from ..verify import detect_verify_command
                self.verify_cmd = detect_verify_command(self.cwd)
            except Exception as e:
                from localm.debuglog import logger
                logger.debug("plan auto-run: verify command detection failed: %r", e)
                self._emit("info", text=f"no verification check detected ({e})")
        return self.verify_cmd

    def _end_plan_autorun(self) -> None:
        """Restore the ``auto_approve`` and ``verify_cmd`` that were in force
        before ``_begin_plan_autorun``. No-op when auto-run is not active."""
        saved = self._plan_autorun_saved
        if saved is None:
            return
        self._plan_autorun_saved = None
        self._auto_approve, self.verify_cmd = saved
