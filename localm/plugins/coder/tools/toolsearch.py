# SPDX-License-Identifier: AGPL-3.0-or-later
"""``search_tools``: full usage of deferred tools, on demand.

The system prompt documents the core tools in full and lists the deferred ones
(``registry.DEFERRED_TOOLS`` plus every MCP and plugin tool) by name and one
line. This tool returns the full catalogue block of the tools matching a query,
in the same format the system prompt uses, and marks them activated on the
session so a native-tools backend declares their schemas from the next request
on. The system prompt itself is never rebuilt by this tool.

Only tools enabled for the calling session are searched or returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ToolResult

# Most tools one keyword search returns.
MAX_MATCHES = 5

_SELECT_PREFIX = "select:"


@dataclass
class _ForeignDocsResult(ToolResult):
    """A search result carrying an MCP or plugin tool's docs; read by
    provenance.is_untrusted_tool, which fences it as external content."""
    untrusted_output: bool = True

_WORD = re.compile(r"[a-z0-9]+")


def _score(name: str, description: str, words: list[str]) -> int:
    """Keyword score of one tool: a whole name segment scores 3, a substring of
    the name 2, a word of the description 1; a query equal to the name wins."""
    name_l = name.lower()
    segments = set(name_l.split("_"))
    desc_words = set(_WORD.findall(description.lower()))
    score = 0
    for w in words:
        if w in segments:
            score += 3
        elif w in name_l:
            score += 2
        if w in desc_words:
            score += 1
    if "_".join(words) == name_l:
        score += 10
    return score


def _candidates(session) -> dict[str, Any]:
    """The registry entries the session may call, minus search_tools itself, in
    registry order."""
    from .registry import TOOL_REGISTRY, TOOL_SEARCH_TOOL
    disabled = frozenset(getattr(session, "disabled_tools", frozenset()) or ())
    return {n: t for n, t in TOOL_REGISTRY.items()
            if n not in disabled and n != TOOL_SEARCH_TOOL}


def tool_search_tools(cwd: Path, query: str | None = None, _session=None,
                      **_) -> ToolResult:
    """Return the full usage of the enabled tools matching *query* and activate
    them on *_session*.

    *query* is either keywords (ranked, at most ``MAX_MATCHES`` results) or
    ``select:name1,name2`` (exact names; unknown or disabled names are reported).
    An error result for a missing session or an empty query. A result whose
    matches include a tool flagged ``untrusted_docs`` is a ``_ForeignDocsResult``
    (``untrusted_output`` True), so the agent fences it as external content."""
    from ..prompts import tool_doc_block
    from .registry import is_deferred
    if _session is None:
        return ToolResult.error("search_tools is only available inside an agent session")
    q = (query or "").strip()
    if not q:
        return ToolResult.error(
            'search_tools needs a query: keywords, or "select:name1,name2"')

    cands = _candidates(_session)
    missing: list[str] = []
    if q.lower().startswith(_SELECT_PREFIX):
        wanted = [w.strip() for w in q[len(_SELECT_PREFIX):].split(",") if w.strip()]
        names = []
        for w in wanted:
            if w in cands:
                if w not in names:
                    names.append(w)
            else:
                missing.append(w)
    else:
        words = _WORD.findall(q.lower())
        scored = [(s, i, n) for i, (n, t) in enumerate(cands.items())
                  if (s := _score(n, str(t.description), words)) > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        names = [n for _s, _i, n in scored[:MAX_MATCHES]]

    if not names:
        deferred = [n for n, t in cands.items() if is_deferred(n, t)]
        listing = ", ".join(deferred) if deferred else "none"
        head = (f"No enabled tool named: {', '.join(missing)}." if missing
                else f"No tool matches {q!r}.")
        foreign = any(getattr(cands[n], "untrusted_docs", False) is True for n in deferred)
        return (_ForeignDocsResult if foreign else ToolResult).success(
            f"{head} Tools listed without their usage: {listing}. "
            'Try other keywords, or "select:<name>".',
            summary="no matching tool")

    activate = getattr(_session, "_activate_tools", None)
    if callable(activate):
        activate(names)

    blocks = [tool_doc_block(n, cands[n]) for n in names]
    out = (f"Full usage of {len(names)} tool(s); call them like any other tool:\n\n"
           + "\n\n".join(blocks))
    if missing:
        out += f"\n\nNot available in this session: {', '.join(missing)}"
    foreign = any(getattr(cands[n], "untrusted_docs", False) is True for n in names)
    return (_ForeignDocsResult if foreign else ToolResult).success(
        out, summary=", ".join(names))
