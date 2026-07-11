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


def compute_tool_diff(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """
    Unified diff a write_file/edit_file/patch_file tool call would produce
    against *old_content*, or None when nothing would change or the tool
    isn't one of the three diff-producing writes.

    ``args["path"]`` is used only for the diff's a/ and b/ file labels.
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
