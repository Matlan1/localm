# SPDX-License-Identifier: AGPL-3.0-or-later
"""outline_file: the classes, functions and methods a source file declares, with
the exact line span of each, so a large file can be navigated by structure and
then read one span at a time with ``read_file(path, offset, limit)``.

Stateless and read-only: it reads one file inside the confined cwd and needs no
session (unlike find_references)."""

from __future__ import annotations

from pathlib import Path

from ..outline import format_outline, outline_source
from .base import _MAX_OUTPUT, ToolResult, _confine

# Largest file outlined. A bigger file is almost always generated or data, and
# ast.parse on it would cost seconds of CPU for an outline nobody can use.
_MAX_OUTLINE_BYTES = 2_000_000

_REGEX_LABEL = ("regex (approximate: declarations found by pattern, end lines "
                "estimated from indentation, not parsed)")


def tool_outline_file(cwd: Path, path: str) -> ToolResult:
    """Outline the file at *path* (relative to cwd)."""
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if p.is_dir():
        return ToolResult.error(
            f"{path} is a directory; outline_file takes a source file. "
            "Use tree or list_dir to see a directory.")
    if not p.is_file():
        return ToolResult.error(f"Not a file: {p}")
    try:
        size = p.stat().st_size
        if size > _MAX_OUTLINE_BYTES:
            return ToolResult.error(
                f"{path} is {size} bytes, over the {_MAX_OUTLINE_BYTES}-byte outline "
                "limit; use grep to find what you need.")
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd.resolve()) if p.is_relative_to(cwd.resolve()) else p
    outline = outline_source(text, p.suffix)
    method = "ast" if outline.method == "ast" else _REGEX_LABEL

    if not outline.entries:
        output = (f"<path>{rel}</path>\n<outline lines=\"{outline.lines}\" method=\"{method}\">\n"
                  "(no class or function declarations found)\n</outline>")
        return ToolResult.success(output, summary=f"{rel} - no declarations ({outline.method})")

    budget = _MAX_OUTPUT - 400
    body, shown, note = format_outline(outline, budget)
    output = (f"<path>{rel}</path>\n<outline lines=\"{outline.lines}\" method=\"{method}\">\n"
              f"{body}\n</outline>\n"
              "Read one span with read_file(path, offset=<first line>, "
              "limit=<last - first + 1>).")
    if note:
        output += f"\n[outline cut: {note}]"
    total = len(outline.entries)
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{rel} - {total} declaration(s) ({outline.method})",
        truncated=bool(note),
    )
