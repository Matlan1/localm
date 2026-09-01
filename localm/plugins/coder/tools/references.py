# SPDX-License-Identifier: AGPL-3.0-or-later
"""find_references: query the running session's live ProjectMap reverse-
reference index (indexer.py's per-file FileSummary.refs) for call sites of a
symbol - "who else calls this" blast-radius awareness without a grep
round-trip per edit.

Needs the running session for its live, incrementally-maintained ProjectMap,
so the dispatcher injects it as a hidden `_session` arg
(agent/constants.py's _PROJECT_MAP_TOOLS), exactly like tools/tasks.py's todo
tools use `_session` for the session's todo state.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult

# How many hits to list before "N more" - grep's tool description uses the
# same shape ("... shown; raise the cap when a result says it was capped").
_MAX_RESULTS = 200


def tool_find_references(cwd: Path, symbol: str = "", _session=None) -> ToolResult:
    """Call sites of *symbol* found by the project map's reverse index.

    Best-effort and single-repo: a textual `symbol(` match outside its own
    definition line, not a real call graph. A shadowed local of the same
    name, or a call reached only through an alias or attribute access, is
    not distinguished from a genuine hit - use grep for an exhaustive search.
    """
    if _session is None or not hasattr(_session, "_project_map"):
        return ToolResult.error(
            "find_references needs an agent session and none was supplied - "
            "nothing was searched.")
    name = (symbol or "").strip()
    if not name:
        return ToolResult.error("find_references needs a non-empty `symbol`.")

    hits = _session._project_map.find_references(name)
    if not hits:
        return ToolResult.success(
            f"No call sites found for '{name}' in the indexed project.\n"
            "The index is best-effort (regex, not a real call graph) and "
            "only covers files the project map tracks - grep for the name "
            "directly for an exhaustive search.",
            summary=f"0 reference(s) to '{name}'",
        )

    shown = hits[:_MAX_RESULTS]
    lines = [f"  {path}:{lineno}" for path, lineno in shown]
    output = f"{len(hits)} call site(s) of '{name}':\n" + "\n".join(lines)
    trunc = len(hits) > _MAX_RESULTS
    if trunc:
        output += f"\n... ({len(hits) - _MAX_RESULTS} more)"
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{len(hits)} reference(s) to '{name}'",
        truncated=trunc,
    )
