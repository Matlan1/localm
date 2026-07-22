# SPDX-License-Identifier: AGPL-3.0-or-later
"""How work done by an ISOLATED child agent is surfaced to the user.

A child that runs in its own git worktree does not touch the parent's working
tree. Its work lives on a branch. This module holds the one shared way both
child-dispatch features (worktree-parallel dispatch and background sub-agent
jobs) tell the user that such work exists, so the presentation is identical
whichever feature produced it.

WHY DELEGATED WORK IS NEVER FOLDED INTO session_diff()
-------------------------------------------------------
``session_diff()`` is not merely displayed. It is an INPUT to two model-facing
loops, so contaminating it corrupts behaviour, not just a view:

- agent/loop.py:381 passes it to ``reviewer.review_feedback(...)``, whose reply is
  injected back into the agent as a user message. Foreign hunks would make the
  self-reviewer critique changes that are not in the tree and instruct the agent
  to fix things it cannot see.
- agent/session.py:190 passes it to reflect_and_store, so it becomes EPISODIC
  MEMORY. A lesson would be stored against a diff that does not match the repo.

There is a mechanical hazard too: ``_track_write`` keys each entry relative to the
writing agent's OWN cwd (persistence.py:86-90) while ``session_diff`` re-resolves
those keys against ``self.cwd`` (persistence.py:67) and ``changed_files`` does the
same (persistence.py:38). A foreign key therefore names a DIFFERENT file in the
parent, so merging one either fabricates a diff that was never made or silently
reports nothing at all.

So the invariant is absolute: ``session_diff()`` and ``changed_files()`` describe
the PARENT's tree and nothing else, ever.

WHY A POINTER AND NOT THE DIFF TEXT
-----------------------------------
The ``/diff`` command renders its result with ``Syntax(diff, "diff", ...)``
(cli/repl.py:272), i.e. as ONE diff document. Appending another tree's hunks would
produce something that reads as directly applicable and is not - the user could
reasonably try to apply it, or assume it was already applied. So the footer names
the branch and the file count and gives the exact command to view it, rather than
inlining foreign hunks. The branch is the durable artifact (the worktree is
transient and may already be gone), so a branch reference stays correct.

Joint design decision by the parallel-dispatch and background-spawn work,
2026-07-22, delegated to those two by the maintainer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelegatedChangeSet:
    """One isolated child's work, as a POINTER to where it actually lives."""

    label: str               # the child agent's name
    branch: str              # durable artifact holding the work
    file_count: int = 0
    source: str = ""         # "parallel" | "background"
    status: str = "ok"       # ok | error | timeout
    base: str = ""           # base ref, for the suggested view command

    def view_command(self) -> str:
        """The exact command that shows this change-set."""
        if not self.branch:
            return ""
        base = (self.base or "HEAD")[:12] if self.base else "HEAD"
        return f"git diff {base}..{self.branch}"


def render_footer(items: list[DelegatedChangeSet]) -> str:
    """The shared footer naming delegated work. Empty string when there is none.

    Returning "" for the empty case is deliberate: it lets every display site
    append this unconditionally, so wiring it in changes nothing at all for a
    session that never delegated.
    """
    live = [i for i in items if i.branch]
    if not live:
        return ""

    width = max(len(i.label) for i in live)
    lines = ["Delegated changes (not in this tree):"]
    for i in live:
        files = f"{i.file_count} file(s)" if i.file_count else "no file changes"
        flag = "" if i.status == "ok" else f"  [{i.status}]"
        lines.append(f"  {i.label.ljust(width)}   {files:<16} {i.view_command()}{flag}")
    return "\n".join(lines)


def record(agent, changeset: DelegatedChangeSet) -> None:
    """Attach *changeset* to *agent*'s delegated list.

    The list is created on first use rather than in ``Agent.__init__`` so this
    feature does not edit agent/core.py, which a separate in-flight change is
    already modifying. Promote it to ``__init__`` once both have landed.
    """
    existing = getattr(agent, "_delegated", None)
    if existing is None:
        existing = []
        agent._delegated = existing
    existing.append(changeset)


def footer_for(agent) -> str:
    """The footer for whatever *agent* has delegated so far ("" when nothing)."""
    return render_footer(list(getattr(agent, "_delegated", []) or []))
