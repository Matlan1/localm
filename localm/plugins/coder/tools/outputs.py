# SPDX-License-Identifier: AGPL-3.0-or-later
"""read_output: read back a tool output that was too large to show in full.

The spilled text lives in memory only (see ``..spill``); this tool reads it by
line range or by pattern, with line numbers, so the model can reach the part
of a long test log or search result that the truncated view left out.
"""

from __future__ import annotations

from pathlib import Path

from ..spill import active_store
from .base import ToolResult
from .files import (
    _ModelRegexInvalid, _ModelRegexTooExpensive, _ModelRegexTooSlow,
    _ModelRegexUnavailable, _compile_model_pattern, _model_regex_flags,
    _run_model_regex,
)

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 400
_MAX_MATCHES = 100
_MAX_CHARS = 8_000


def tool_read_output(cwd: Path, id: str, offset: int = 0, limit: int = 0,
                     pattern: str = "") -> ToolResult:
    """Lines *offset*..*offset+limit* (1-based) of spilled output *id*, or the
    lines matching *pattern*, numbered."""
    store = active_store()
    sid = str(id or "").strip()
    entry = store.get(sid) if store is not None else None
    if entry is None:
        known = ", ".join(store.ids()) if store is not None else ""
        return ToolResult.error(
            f"No kept output named {sid!r}"
            + (f"; kept outputs: {known}." if known else "; none are kept now.")
            + " Re-run the command to see its output again.")
    lines = entry.text.splitlines()
    total = len(lines)

    if pattern:
        try:
            rx = _compile_model_pattern(pattern, _model_regex_flags(ignore_case=True))
        except _ModelRegexInvalid as e:
            return ToolResult.error(f"Invalid regex: {e}")
        except _ModelRegexUnavailable as e:
            return ToolResult.error(str(e))
        picked: list[int] = []
        try:
            for i, line in enumerate(lines):
                if _run_model_regex(rx.search, line):
                    picked.append(i)
                    if len(picked) >= _MAX_MATCHES:
                        break
        except (_ModelRegexTooSlow, _ModelRegexTooExpensive) as e:
            return ToolResult.error(str(e))
        if not picked:
            return _result(entry, f"no line of {sid} ({total} lines) matches {pattern!r}",
                           [], total, summary=f"{sid}: 0 matches")
        capped = len(picked) >= _MAX_MATCHES
        note = (f"first {_MAX_MATCHES} matching lines" if capped
                else f"{len(picked)} matching line(s)")
        return _result(entry, f"{note} of {sid} ({total} lines)",
                       [(i + 1, lines[i]) for i in picked], total,
                       summary=f"{sid}: {len(picked)} match(es)")

    start = max(1, int(offset or 1))
    if start > total:
        return ToolResult.error(f"offset {start} is past the end of {sid} ({total} lines)")
    count = min(int(limit) if limit else _DEFAULT_LIMIT, _MAX_LIMIT)
    end = min(total, start + max(1, count) - 1)
    return _result(entry, f"lines {start}-{end} of {sid} ({total} lines)",
                   [(n, lines[n - 1]) for n in range(start, end + 1)], total,
                   summary=f"{sid}: lines {start}-{end} of {total}")


def _result(entry, header: str, numbered: list[tuple[int, str]], total: int,
            summary: str) -> ToolResult:
    body: list[str] = []
    used = 0
    cut = False
    for n, line in numbered:
        row = f"{n}\t{line}"
        if used + len(row) > _MAX_CHARS:
            cut = True
            break
        body.append(row)
        used += len(row) + 1
    tail = ("\n[stopped at the size limit; ask for a smaller range]" if cut else "")
    out = ToolResult(
        ok=True,
        output=f"[{header}, from {entry.tool}]\n" + "\n".join(body) + tail,
        summary=summary,
        truncated=cut,
        untrusted_output=entry.untrusted,
    )
    return out
