# SPDX-License-Identifier: AGPL-3.0-or-later
"""outline_file: the classes, functions and methods a source file declares, with
the exact line span of each, so a large file can be navigated by structure and
then read one span at a time with ``read_file(path, offset, limit)``.

Stateless and read-only: it reads one file inside the confined cwd and needs no
session (unlike find_references)."""

from __future__ import annotations

from pathlib import Path

from ..outline import format_outline, outline_source, supported_suffixes
from .base import _MAX_OUTPUT, ToolResult, _confine

# Largest file outlined. A bigger file is almost always generated or data, and
# ast.parse on it would cost seconds of CPU for an outline nobody can use.
_MAX_OUTLINE_BYTES = 2_000_000

# Characters kept free in the result for the continuation note.
_NOTE_RESERVE = 200

_REGEX_LABEL = ("regex (approximate: declarations found by pattern, end lines "
                "estimated from indentation, not parsed)")


def tool_outline_file(cwd: Path, path: str, from_line: int = 0) -> ToolResult:
    """Outline the file at *path* (relative to cwd).

    *from_line* (1-based, optional) lists only the declarations that end at or
    after that line, to continue an outline that was cut."""
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
        start_from = max(1, int(from_line or 1))
    except (TypeError, ValueError):
        return ToolResult.error(f"from_line must be a line number, got {from_line!r}.")
    try:
        size = p.stat().st_size
        if size > _MAX_OUTLINE_BYTES:
            return ToolResult.error(
                f"{path} is {size} bytes, over the {_MAX_OUTLINE_BYTES}-byte outline "
                "limit; use grep to find what you need.")
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        return ToolResult.error(str(e))

    resolved_cwd = cwd.resolve()
    rel = p.relative_to(resolved_cwd) if p.is_relative_to(resolved_cwd) else p
    outline = outline_source(text, p.suffix)

    if outline.method == "unsupported":
        shown_ext = p.suffix or "extensionless"
        return ToolResult.error(
            f"outline_file has no support for {shown_ext} files (it outlines "
            f"{', '.join(supported_suffixes())}); use read_file or grep on {path}.")

    method = "ast" if outline.method == "ast" else _REGEX_LABEL
    header = f"<path>{rel}</path>\n<outline lines=\"{outline.lines}\" method=\"{method}\">\n"
    if not outline.entries:
        output = header + "(no class or function declarations found)\n</outline>"
        return ToolResult.success(output, summary=f"{rel} - no declarations ({outline.method})")

    tail = ("\n</outline>\nRead one span with read_file(path, offset=<first line>, "
            "limit=<last - first + 1>).")
    budget = _MAX_OUTPUT - len(header) - len(tail) - _NOTE_RESERVE
    fmt = format_outline(outline, budget, start_from)
    if not fmt.shown:
        return ToolResult.success(
            header + f"(no declarations end at or after line {start_from})\n</outline>",
            summary=f"{rel} - nothing from line {start_from}")
    output = header + fmt.body + tail
    cut = fmt.next_from is not None
    if cut:
        output += (f"\n[outline cut: {fmt.shown} of {fmt.total} declaration(s) shown; "
                   f"continue with outline_file(path, from_line={fmt.next_from})]")
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{rel} - {fmt.shown} of {fmt.total} declaration(s) ({outline.method})"
                if cut else f"{rel} - {fmt.total} declaration(s) ({outline.method})",
        truncated=cut,
    )
