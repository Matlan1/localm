# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared diff computation for write_file/edit_file/patch_file tool calls (CODER-1).

Previously implemented three separate times (agent/execution.py's
``_patch_mode_intercept`` and ``_confirm_tool``, sessions.py's ``_diff_preview``),
each independently reading the file's current content and branching by tool name
to work out what would change. A fix to one copy (a tool-arg rename, new-file
handling) was easy to apply to only one or two of the three and let the others
silently diverge - this module is the one place that logic lives now.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional


def read_old_content(cwd: Path, path_arg: str) -> str:
    """The file's current text content for diff purposes, or "" when it
    doesn't exist yet (a new file) or can't be decoded/read."""
    if not path_arg:
        return ""
    abs_path = (cwd / path_arg).resolve()
    if not abs_path.is_file():
        return ""
    try:
        return abs_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def resolve_new_content(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """The new_content a write_file/edit_file call would produce, given the
    file's old_content. None for patch_file (no new_content concept - the
    diff is supplied directly) or any other tool name."""
    if tool_name == "write_file":
        return args.get("content", "")
    if tool_name == "edit_file":
        old_str = args.get("old", "")
        new_str = args.get("new", "")
        return old_content.replace(old_str, new_str, 1)
    return None


def compute_multifile_diff(cwd: Path, edits: object) -> Optional[str]:
    """
    Unified diff an ``edit_files`` call would produce, concatenated over every
    file it touches, or None when nothing would change.

    Each file is read once and successive edits to it compose, so the diff
    matches what the tool would actually write. Unlike the single-file helpers
    this needs *cwd*, because the paths live inside the edit items.
    """
    if not isinstance(edits, list):
        return None
    current: dict[str, str] = {}
    order:   list[str] = []
    for item in edits:
        if not isinstance(item, dict):
            continue
        path_arg = str(item.get("path") or "")
        old_str  = str(item.get("old") or "")
        new_str  = str(item.get("new") or "")
        if not path_arg or not old_str:
            continue
        if path_arg not in current:
            current[path_arg] = read_old_content(cwd, path_arg)
            order.append(path_arg)
        current[path_arg] = current[path_arg].replace(old_str, new_str, 1)
    chunks = []
    for path_arg in order:
        old_content = read_old_content(cwd, path_arg)
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            current[path_arg].splitlines(keepends=True),
            fromfile=f"a/{path_arg}",
            tofile=f"b/{path_arg}",
        ))
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "".join(chunks) if chunks else None


def compute_tool_diff(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """
    Unified diff a write_file/edit_file/patch_file tool call would produce
    against *old_content*, or None when nothing would change or the tool
    isn't one of the three diff-producing writes.

    ``args["path"]`` is used only for the diff's a/ and b/ file labels.
    ``edit_files`` is not handled here - it spans several files, so it has no
    single *old_content*; use :func:`compute_multifile_diff` for it.
    """
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
