# SPDX-License-Identifier: AGPL-3.0-or-later
"""How work done by an ISOLATED child agent is surfaced to the user.

A child that runs in its own git worktree does not touch the parent's working
tree. Its work lives on a branch. This module holds the one shared way both
child-dispatch features (worktree-parallel dispatch and background sub-agent
jobs) tell the user that such work exists, so the presentation is identical
whichever feature produced it.

DELEGATED WORK IS NEVER FOLDED INTO session_diff()
--------------------------------------------------
``session_diff()`` is not merely displayed. It is an INPUT to two model-facing
loops:

- agent/loop.py passes it to ``reviewer.review_feedback(...)``, whose reply is
  injected back into the agent as a user message. Foreign hunks would make the
  self-reviewer critique changes that are not in the tree and instruct the agent
  to fix things it cannot see.
- agent/session.py passes it to reflect_and_store, so it becomes EPISODIC
  MEMORY. A lesson would be stored against a diff that does not match the repo.

There is a mechanical hazard too: ``_track_write`` keys each entry relative to the
writing agent's OWN cwd while ``session_diff`` re-resolves those keys against
``self.cwd``, and ``changed_files`` does the same. A foreign key therefore names a
DIFFERENT file in the parent, so merging one either fabricates a diff that was
never made or silently reports nothing at all.

So the invariant is absolute: ``session_diff()`` and ``changed_files()`` describe
the PARENT's tree and nothing else, ever.

WHERE THE APPEND IS AND IS NOT WIRED (site-selective)
------------------------------------------------------
"session_diff() is unmodified" is NOT a sufficient invariant, because the
contamination can happen at the CALL SITE instead. The binding invariant is: THE
SELF-REVIEWER AND THE EPISODE NEVER RECEIVE FOREIGN-TREE HUNKS.

- APPENDED (human-facing): the ``/diff`` and ``/changes`` REPL commands.
- NOT APPENDED (model-facing): agent/loop.py and agent/session.py. The episode
  still learns THAT delegation happened, via structured fields, but is never fed
  hunks it cannot reconcile against the repo.

A test pins this by inspecting the source of those two modules.

NEVER UNDER A PATH FILTER
-------------------------
``/diff <path>`` asks about ONE file. A delegated section rendered there would
answer a question the user did not ask, and in the HTTP equivalent
(builtin/coder/plug.py) a non-empty result for an unchanged path defeats its
``404``. So the section renders only on the unfiltered view.

RENDERED SEPARATELY, NEVER INSIDE THE DIFF DOCUMENT
----------------------------------------------------
``/diff`` renders the parent's diff through ``Syntax(diff, "diff", ...)``, i.e. as
ONE applicable patch. The delegated section is printed as its OWN console block
after it, never concatenated into that string, so the foreign hunks cannot read as
part of a patch the user might try to apply. The heading and the per-child branch
line say plainly that the work is not in this tree. The branch is quoted because
it is the durable artifact: the worktree is transient and may already have been
removed.
"""

from __future__ import annotations

from dataclasses import dataclass


# Per-child cap on inlined diff text. The full diff is always reachable via the
# branch command, and /diff is invoked repeatedly during a session - dumping two
# unbounded child diffs on every invocation would drown the parent's own changes.
_MAX_INLINE_DIFF_CHARS = 2_000


@dataclass(frozen=True)
class DelegatedChangeSet:
    """One isolated child's work: where it lives, and what it changed."""

    label: str               # the child agent's name
    branch: str              # durable artifact holding the work
    file_count: int = 0
    source: str = ""         # "parallel" | "background"
    status: str = "ok"       # ok | error | timeout
    base: str = ""           # base ref, for the suggested view command
    diff: str = ""           # captured diff TEXT, shown inline
    summary: str = ""

    def view_command(self) -> str:
        """The exact command that shows this change-set in full."""
        if not self.branch:
            return ""
        base = (self.base or "HEAD")[:12] if self.base else "HEAD"
        return f"git diff {base}..{self.branch}"


def render_footer(items: list[DelegatedChangeSet]) -> str:
    """The shared delegated-work section. Empty string when there is none.

    The "" for the empty case lets each display site render this
    unconditionally, so wiring it in changes nothing for a session that never
    delegated. Callers must print it as its OWN block, never concatenated into a
    diff document (see the module docstring).
    """
    live = [i for i in items if i.branch]
    if not live:
        return ""

    lines = [
        "Delegated work (NOT in your working tree):",
        "These changes live on branches and have not been merged.",
        "",
    ]
    for i in live:
        files = f"{i.file_count} file(s)" if i.file_count else "no file changes"
        origin = f", {i.source}" if i.source else ""
        lines.append(f"  {i.label} [{i.status}{origin}]  {files}")
        lines.append(f"    branch: {i.branch}")
        lines.append(f"    view:   {i.view_command()}")
        if i.summary:
            lines.append(f"    {i.summary}")
        if i.diff:
            body = i.diff
            if len(body) > _MAX_INLINE_DIFF_CHARS:
                body = (body[:_MAX_INLINE_DIFF_CHARS]
                        + f"\n... [truncated - full diff: {i.view_command()}]")
            lines.append("")
            lines.extend("    " + ln for ln in body.splitlines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def record(agent, changeset: DelegatedChangeSet) -> None:
    """Attach *changeset* to *agent*'s delegated list.

    ``Agent.__init__`` creates the list, but this still tolerates its absence:
    the parent here is whatever object the tool was handed, and the test doubles
    that stand in for an Agent do not inherit its ``__init__``.
    """
    existing = getattr(agent, "_delegated", None)
    if existing is None:
        existing = []
        agent._delegated = existing
    existing.append(changeset)


def footer_for(agent) -> str:
    """The footer for whatever *agent* has delegated so far ("" when nothing)."""
    return render_footer(list(getattr(agent, "_delegated", []) or []))
