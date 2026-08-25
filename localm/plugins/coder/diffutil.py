# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared diff computation for write_file/edit_file/patch_file tool calls (CODER-1)."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional


def read_old_content_checked(cwd: Path, path_arg: str) -> tuple[str, bool]:
    """``(content, readable)`` for diff purposes."""
    if not path_arg:
        return "", True
    abs_path = (cwd / path_arg).resolve()
    if not abs_path.is_file():
        return "", True                    # a new file: an honest empty
    try:
        return abs_path.read_text(encoding="utf-8", errors="replace"), True
    except Exception:
        return "", False                   # EXISTS, and we could not read it


def read_old_content(cwd: Path, path_arg: str) -> str:
    """The file's current text content for diff purposes, or '' when it doesn't exist yet (a new file) or can't be decoded/read."""
    return read_old_content_checked(cwd, path_arg)[0]


def resolve_new_content(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """The new_content a write_file/edit_file call would produce, given the file's old_content."""
    if tool_name == "write_file":
        return args.get("content", "")
    if tool_name == "edit_file":
        old_str = args.get("old", "")
        new_str = args.get("new", "")
        return old_content.replace(old_str, new_str, 1)
    return None


def compute_multifile_diff(cwd: Path, edits: object) -> Optional[str]:
    """Unified diff an ``edit_files`` call would produce, concatenated over every file it touches, or None when nothing would change OR when the batch would be REJECTED (a malformed item, or an ``old`` that does not match)."""
    if not isinstance(edits, list) or not edits:
        return None
    current: dict[str, str] = {}
    order:   list[str] = []
    labels:  dict[str, str] = {}   # canonical key -> the path as the caller wrote it
    for item in edits:
        if not isinstance(item, dict):
            return None
        path_arg = str(item.get("path") or "")
        old_str  = str(item.get("old") or "")
        new_str  = str(item.get("new") or "")
        if not path_arg or not old_str:
            return None
        # Key on the RESOLVED path, as tool_edit_files does, so "a.py" and
        # "./a.py" in one batch compose into one hunk instead of two
        # contradictory ones.
        try:
            key = str((cwd / path_arg).resolve())
        except Exception:
            key = path_arg
        if key not in current:
            current[key] = read_old_content(cwd, path_arg)
            order.append(key)
            labels[key] = path_arg
        if old_str not in current[key]:
            # The real tool rejects the WHOLE batch on a miss. Returning a
            # partial diff here would make patch mode report success for a
            # change edit_files would never apply.
            return None
        current[key] = current[key].replace(old_str, new_str, 1)
    chunks = []
    for key in order:
        path_arg = labels[key]
        old_content = read_old_content(cwd, path_arg)
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            current[key].splitlines(keepends=True),
            fromfile=f"a/{path_arg}",
            tofile=f"b/{path_arg}",
        ))
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "".join(chunks) if chunks else None


def compute_search_replace_diff(cwd: Path, pattern: str, replacement: str,
                                glob_pattern: str = "**/*") -> Optional[str]:
    """Unified diff a ``search_replace`` call would produce, concatenated over every file it would touch, or None when nothing would change or the pattern is invalid."""
    from .tools.files import tool_search_replace
    result = tool_search_replace(cwd, pattern, replacement, glob_pattern, dry_run=True)
    if not result.ok or not result.changes:
        return None
    chunks = []
    for rel, old_bytes, new_text in result.changes:
        old_text = old_bytes.decode("utf-8", errors="replace")
        # Normalise CRLF/CR the same way text-mode read (and so the
        # substitution ToolResult.changes' new_text is built from) already
        # does: old_bytes is deliberately RAW (see ToolResult.changes' own
        # docstring - the changed-files/undo tracker needs the exact on-disk
        # bytes), but new_text is always LF-only, having been produced from
        # a normalised read. Diffing raw CRLF against normalised LF would
        # flag every line's ending as changed on a CRLF file, even where the
        # sweep touched nothing on that line.
        #
        # SAME root cause as tool_search_replace's own two-reads split
        # (files.py - old_bytes raw for tracking, read_text() for the
        # substitution): one raw-vs-normalised mismatch, two symptoms in two
        # subsystems. Removing this normalise step (or fixing files.py's
        # split back to one read) will look correct on Linux and reopen both.
        old_text = old_text.replace("\r\n", "\n").replace("\r", "\n")
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "".join(chunks) if chunks else None


def compute_tool_diff(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """Unified diff a write_file/edit_file/patch_file tool call would produce against *old_content*, or None when nothing would change or the tool isn't one of the three diff-producing writes."""
    if tool_name == "patch_file":
        diff = args.get("diff", "")
        return diff if diff else None

    new_content = resolve_new_content(tool_name, args, old_content)
    if new_content is None:
        return None

    path_arg = args.get("path", "")
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path_arg}",
        tofile=f"b/{path_arg}",
    ))
    return "".join(diff_lines) if diff_lines else None
