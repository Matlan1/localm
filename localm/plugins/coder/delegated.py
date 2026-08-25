# SPDX-License-Identifier: AGPL-3.0-or-later
"""How work done by an ISOLATED child agent is surfaced to the user."""

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
    """The shared delegated-work section."""
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
    """Attach *changeset* to *agent*'s delegated list."""
    existing = getattr(agent, "_delegated", None)
    if existing is None:
        existing = []
        agent._delegated = existing
    existing.append(changeset)


def footer_for(agent) -> str:
    """The footer for whatever *agent* has delegated so far ('' when nothing)."""
    return render_footer(list(getattr(agent, "_delegated", []) or []))
