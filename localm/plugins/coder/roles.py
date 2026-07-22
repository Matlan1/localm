# SPDX-License-Identifier: AGPL-3.0-or-later
"""Role presets for spawned sub-agents: a focused mission plus a narrowed toolset.

A parent delegating "review this diff" wants a reviewer, not a second full agent.
Before this existed, ``spawn_agent`` handed the child the parent's whole toolset,
so a child asked only to read and judge could still ``write_file``, ``run_shell``
and ``git_push``. A role fixes both halves at once: the mission text tells the
model what it is for, and ``allowed_tools`` is the set the child is actually left
holding.

THE INVARIANT: a role only ever REMOVES capability. It is applied as
``disabled_tools | (everything_registered - allowed_tools)`` (see
``Agent._apply_role_toolset``), a union with what the parent already forbade, so
a role can never hand back a tool the parent disabled or that a restricted,
shareable session forbids. Narrowing is a subset operation, never a substitution.

The allowlists follow the reasoning already written for ``SAFE_RESTRICTED_TOOLS``
(tools/registry.py): the excluded tools are the ones that execute code
(run_shell/run_tests, where a planted conftest runs), run git hooks
(git_commit/git_push), reach the network (fetch_url/web_search), or disclose
secrets (read_env). Because the set is an ALLOWLIST, a newly added tool - and
every dynamically registered MCP/plugin/skill tool - is denied to a role by
default rather than silently inherited.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolePreset:
    """One selectable sub-agent role."""

    name:          str
    summary:       str             # one-liner shown in the spawn_agent tool docs
    mission:       str             # the brief injected into the child's prompt
    allowed_tools: frozenset[str]  # the ONLY tools this role may call


# Building blocks. Named so a role reads as a sum of capabilities and a reviewer
# demonstrably has no write capability in its definition, not merely in prose.
_READ_ONLY = frozenset({
    "read_file", "list_dir", "tree", "grep", "search_files",
})
# Inspecting git state runs no hook and changes nothing (unlike git_commit /
# git_create_branch / git_push, which are excluded from every role below).
_GIT_INSPECT = frozenset({"git_status", "git_diff", "git_log"})
_EDIT = frozenset({
    "write_file", "edit_file", "patch_file", "search_replace",
    "edit_notebook_cell",
})


ROLE_PRESETS: dict[str, RolePreset] = {
    "reviewer": RolePreset(
        name="reviewer",
        summary="read-only code review; can read and inspect git, cannot change anything",
        mission=(
            "Review the code and report what you find. You are a reader, not an "
            "editor: you have no write, shell, or git-write tools, so do not "
            "plan changes you cannot make. Ground every point in something you "
            "actually read - quote the file and line. Report the problems you "
            "can defend, say plainly when you find none, and never approve code "
            "you did not read."
        ),
        allowed_tools=_READ_ONLY | _GIT_INSPECT,
    ),
    "researcher": RolePreset(
        name="researcher",
        summary="read-only investigation of the working tree; no writes, git, or network",
        mission=(
            "Investigate the question and report what the code actually does. "
            "You can only read the current working tree - no writing, no shell, "
            "no git, no network - so answer from files you have opened, never "
            "from assumption. Cite the file and line behind each claim, and say "
            "explicitly when something is not there rather than guessing."
        ),
        allowed_tools=_READ_ONLY,
    ),
    "test-writer": RolePreset(
        name="test-writer",
        summary="write and run tests; no shell, no git writes, no network",
        mission=(
            "Write tests and run them. Read the code under test first so the "
            "tests assert its real behaviour, then cover the edge and failure "
            "cases, not just the happy path. Use run_tests to check your work - "
            "a test you never ran is not evidence. You cannot commit or push; "
            "leave the changes for the parent to review."
        ),
        allowed_tools=_READ_ONLY | _GIT_INSPECT | _EDIT | frozenset({"run_tests"}),
    ),
}


def resolve_role(name: str | None) -> RolePreset | None:
    """The preset for ``name``, or None when no role was requested.

    Raises ValueError on an unrecognised name. Falling back to "no role" would
    silently hand the child the parent's FULL toolset - the exact over-capable
    child this module exists to prevent - so a typo fails loudly instead.
    """
    if name is None:
        return None
    # A model can emit anything for an argument. Coercing a non-string here (say
    # {"role": 123} or a list) would guess at intent; raising the SAME clear
    # ValueError keeps the fail-closed path single and tells it what to send.
    if not isinstance(name, str):
        raise ValueError(
            f"unknown role {name!r}: role must be a string. "
            f"Available roles: {', '.join(sorted(ROLE_PRESETS))}"
        )
    key = name.strip().lower()
    if not key:
        return None
    # Accept the underscore spelling too: a model that has seen "test_writer"
    # elsewhere in the tool schema should not be punished for the separator.
    key = key.replace("_", "-")
    preset = ROLE_PRESETS.get(key)
    if preset is None:
        raise ValueError(
            f"unknown role '{name}'. Available roles: {', '.join(sorted(ROLE_PRESETS))}"
        )
    return preset


def role_names() -> list[str]:
    """Role names, sorted - for tool docs and error messages."""
    return sorted(ROLE_PRESETS)


def role_catalogue() -> str:
    """``name (summary)`` lines for the spawn_agent parameter description."""
    return "; ".join(f"{n} ({ROLE_PRESETS[n].summary})" for n in role_names())
