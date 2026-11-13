# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared diff computation for write_file/edit_file/patch_file tool calls.

The single place that reads a file's current content and branches by tool name
to work out what a call would change. Used by agent/execution.py's
``_patch_mode_intercept`` and ``_confirm_tool`` and by sessions.py's
``_diff_preview``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional


def read_old_content_checked(cwd: Path, path_arg: str) -> tuple[str, bool]:
    """``(content, readable)`` for diff purposes.

    ``readable`` is False ONLY when the file EXISTS and could not be read (a
    lock, a permission denial). A file that does not exist yet is ``("", True)``:
    a new file is a known, honest empty, not a failure.

    Use this at any CONSENT surface. ``read_old_content`` collapses the two into
    ``""``, and ``print_diff_preview``'s own contract reads ``""`` as "the file
    doesn't exist yet", so an overwrite of a file that merely failed to READ
    would render as a pure addition with nothing deleted. The ``is_file()``
    check below runs BEFORE the read, so a failure is known to be on an existing
    file.
    """
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
    """The file's current text content for diff purposes, or "" when it
    doesn't exist yet (a new file) or can't be decoded/read.

    Collapses those two cases for DISPLAY and patch-record paths. Do NOT use it
    where the result is shown to a user who is about to approve a write; see
    read_old_content_checked, which keeps them apart."""
    return read_old_content_checked(cwd, path_arg)[0]


def resolve_new_content(tool_name: str, args: dict, old_content: str) -> Optional[str]:
    """The new_content a write_file/edit_file call would produce, given the
    file's old_content. For edit_file this is the tool's own resolution
    (``planned_edit_text``: an exact match first, else a single
    whitespace-tolerant one, re-indented when that keeps the file parsing),
    and *old_content* unchanged when the tool would refuse the call. None for
    patch_file (no new_content concept - the diff is supplied directly) or any
    other tool name."""
    if tool_name == "write_file":
        return args.get("content", "")
    if tool_name == "edit_file":
        from .tools.files import planned_edit_text
        planned = planned_edit_text(Path(str(args.get("path", ""))), old_content,
                                    str(args.get("old", "") or ""),
                                    str(args.get("new", "") or ""))
        return old_content if planned is None else planned
    return None


def compute_multifile_diff(cwd: Path, edits: object) -> Optional[str]:
    """
    Unified diff an ``edit_files`` call would produce, concatenated over every
    file it touches, or None when nothing would change OR when the batch would
    be REJECTED.

    The batch is planned by ``edit_files``' own planner (``plan_edit_files``),
    so the diff is what the tool writes: the same match rules (exact, else one
    whitespace-tolerant match), successive edits to one file composed, paths
    confined to *cwd*, and the whole batch rejected on any miss or syntax
    break. Unlike the single-file helpers this needs *cwd*, because the paths
    live inside the edit items.
    """
    from .tools.files import plan_edit_files
    plan = plan_edit_files(cwd, edits)
    if isinstance(plan, str):
        return None
    chunks = []
    for p in plan.targets:
        label = plan.labels[p]
        diff_lines = list(difflib.unified_diff(
            plan.before[p].splitlines(keepends=True),
            plan.after[p].splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        ))
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "".join(chunks) if chunks else None


def compute_search_replace_diff(cwd: Path, pattern: str, replacement: str,
                                glob_pattern: str = "**/*",
                                restricted: bool = False) -> Optional[str]:
    """
    Unified diff a ``search_replace`` call would produce, concatenated over
    every file it would touch, or None when nothing would change or the
    pattern is invalid.

    Unlike write_file/edit_file/edit_files, search_replace's target files are
    discovered at RUNTIME (a glob + regex sweep) rather than named in the call
    args, so there is no single old_content to diff against ahead of time. This
    calls the real tool with its own ``dry_run=True``, which is the SAME
    matching pass a real apply runs (see ToolResult.changes), rather than
    re-implementing the sweep here. *restricted* applies the same filter a
    restricted session's real apply does, so the diff never covers a file
    inside a ``.localcoder`` directory.
    """
    from .tools.files import tool_search_replace
    result = tool_search_replace(cwd, pattern, replacement, glob_pattern,
                                 dry_run=True, _restricted=restricted)
    if not result.ok or not result.changes:
        return None
    chunks = []
    for rel, old_bytes, new_text in result.changes:
        old_text = old_bytes.decode("utf-8", errors="replace")
        # ToolResult.changes carries old_bytes RAW (the changed-files/undo
        # tracker needs the exact on-disk bytes) while new_text is always
        # LF-only, having been produced from a normalised read. Normalise here
        # so a CRLF file does not diff as though every line ending changed.
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
