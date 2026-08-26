# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model-owned task list: a small, deterministic state store the model writes
its own plan into.

The list lives on the Agent, OUTSIDE ``_messages``, so compaction
(agent/context.py summarises everything but the last 4 messages at 90% fill)
cannot touch it, and it rides along in the resume checkpoint
(HOME/checkpoints/<digest>.json) so it also survives a pause/resume.

This is a state store, not a planner: it holds exactly what the model wrote, in
the order the model wrote it. Nothing here schedules, approves, or reasons about
the plan.

Item format is one line per task, so a small model can produce it without nested
JSON:

    "[ ] read the failing test"   pending
    "[>] fix the parser"          in progress
    "[x] run the suite"           done

Parsing is forgiving about the shapes a model emits (a ``- `` bullet, ``1. ``
numbering, ``[X]``/``[*]``, a newline-joined string instead of an array, or the
``{"text": ..., "status": ...}`` dict form) and strict about the result: every
item ends up as ``{"text": str, "status": one of _STATUSES}``.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult

PENDING     = "pending"
IN_PROGRESS = "in_progress"
DONE        = "done"

_STATUSES = (PENDING, IN_PROGRESS, DONE)

# Leading "[m] " markers, lower-cased. Anything else in the brackets (or no
# bracket at all) is pending, so an unrecognised marker degrades to "not done"
# rather than silently counting as finished.
_MARKERS = {
    "x": DONE, "v": DONE, "✓": DONE, "✔": DONE,
    ">": IN_PROGRESS, "*": IN_PROGRESS, "~": IN_PROGRESS, "@": IN_PROGRESS,
    "": PENDING, " ": PENDING, "-": PENDING, ".": PENDING,
}

# Word forms accepted in the dict item form ({"text": ..., "status": ...}).
_STATUS_WORDS = {
    "pending": PENDING, "todo": PENDING, "open": PENDING, "not_started": PENDING,
    "in_progress": IN_PROGRESS, "in-progress": IN_PROGRESS, "active": IN_PROGRESS,
    "doing": IN_PROGRESS, "wip": IN_PROGRESS, "started": IN_PROGRESS,
    "done": DONE, "completed": DONE, "complete": DONE, "finished": DONE,
}

# Bounds on the list, which is written into every checkpoint and, once
# compaction has run, into the session summary.
MAX_ITEMS = 40
MAX_TEXT  = 200

_STATUS_GLYPH = {PENDING: "[ ]", IN_PROGRESS: "[>]", DONE: "[x]"}


def _clean_text(value: str) -> str:
    """Collapse whitespace and cap the length of one task's text."""
    text = " ".join(str(value).split())
    return text[:MAX_TEXT]


def _parse_line(raw: str) -> dict | None:
    """Parse one ``[m] text`` line into a todo dict, or None if it has no text.

    Strips a leading list bullet (``- ``, ``* ``, ``1. ``) before reading the
    status marker.
    """
    line = str(raw).strip()
    if not line:
        return None

    # Leading bullet / numbering, e.g. "- [x] done thing" or "2. [ ] next".
    if line[:2] in ("- ", "* ", "+ "):
        line = line[2:].lstrip()
    else:
        head, sep, rest = line.partition(". ")
        if sep and head.isdigit():
            line = rest.lstrip()

    status = PENDING
    if line.startswith("["):
        close = line.find("]")
        if close != -1:
            marker = line[1:close].strip().lower()
            # A multi-character marker is a word ("[done]"), a single one is a
            # glyph ("[x]"). Only a RECOGNISED marker is consumed: an unknown
            # bracket is part of the task the model wrote ("[api] fix the
            # handler") and stays in the text rather than being silently eaten.
            known = _STATUS_WORDS.get(marker, _MARKERS.get(marker))
            if known is not None:
                status = known
                line = line[close + 1:].lstrip()

    text = _clean_text(line)
    if not text:
        return None
    return {"text": text, "status": status}


def _parse_item(item) -> dict | None:
    """Parse one item of any accepted shape (line string or status dict)."""
    if isinstance(item, dict):
        text = _clean_text(item.get("text") or item.get("task") or item.get("content") or "")
        if not text:
            return None
        raw_status = str(item.get("status") or "").strip().lower()
        status = _STATUS_WORDS.get(raw_status, PENDING)
        return {"text": text, "status": status}
    if isinstance(item, (str, int, float)):
        return _parse_line(str(item))
    return None   # a list, None, or anything else is not a task


def normalize_todos(raw) -> list[dict]:
    """Canonicalise *raw* into ``[{"text": str, "status": <one of _STATUSES>}]``.

    The single entry point for every source of todos: the tool's own argument,
    and the ``todos`` key read back out of a checkpoint file (which is plain
    user-writable JSON, so it must be treated as untrusted shape). Unparseable
    items are DROPPED, never crash - the count difference is what the caller
    reports, so a partly-bad write is visible rather than silent.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        # A model that ignored the array type and sent one newline-joined blob.
        raw = raw.splitlines()
    if not isinstance(raw, (list, tuple)):
        return []
    todos: list[dict] = []
    for item in raw:
        parsed = _parse_item(item)
        if parsed is not None:
            todos.append(parsed)
        if len(todos) >= MAX_ITEMS:
            break
    return todos


def render_todos(todos: list[dict]) -> str:
    """The task list as the checklist the model wrote, one line per item."""
    return "\n".join(f"{_STATUS_GLYPH[t['status']]} {t['text']}" for t in todos)


def todos_summary(todos: list[dict]) -> str:
    """One-line progress display: counts plus the in-progress item, if any."""
    if not todos:
        return "no tasks"
    done = sum(1 for t in todos if t["status"] == DONE)
    active = next((t["text"] for t in todos if t["status"] == IN_PROGRESS), "")
    head = f"{done}/{len(todos)} done"
    return f"{head} - now: {active}" if active else head


def _session_todos(_session):
    """The agent whose task list this call operates on, or an error result.

    There is no module-level fallback store: a set_todos with nowhere to write
    returns an error rather than reporting success. The dispatcher injects the
    session for these tools (agent/execution.py), so this only fires on direct
    misuse.
    """
    if _session is None or not hasattr(_session, "get_todos"):
        return ToolResult.error(
            "The task list needs an agent session and none was supplied - "
            "nothing was recorded.")
    return None


def tool_set_todos(cwd: Path, items=None, _session=None) -> ToolResult:
    """
    Replace the session task list with *items*.

    Whole-list replace, not append: the model's latest call is the complete
    list.
    """
    err = _session_todos(_session)
    if err is not None:
        return err
    if items is None:
        return ToolResult.error(
            "set_todos needs an 'items' list, e.g. "
            '["[x] read the test", "[>] fix the parser", "[ ] run the suite"]')
    if not isinstance(items, (list, tuple, str)):
        return ToolResult.error(
            f"set_todos 'items' must be a list of task lines, got {type(items).__name__}.")

    todos = normalize_todos(items)
    supplied = len(items.splitlines()) if isinstance(items, str) else len(items)
    if supplied and not todos:
        return ToolResult.error(
            "None of the supplied items had any task text - nothing was recorded. "
            'Each item is one line, e.g. "[ ] run the tests".')

    _session.set_todos(todos)

    if not todos:
        return ToolResult.success("Task list cleared.", summary="todos cleared")
    output = render_todos(todos)
    dropped = supplied - len(todos)
    if dropped > 0:
        # Report what was kept from a partly-rejected write.
        output += (f"\n\n[{dropped} of {supplied} items had no task text (or the "
                   f"{MAX_ITEMS}-item cap was hit) and were not recorded.]")
    return ToolResult.success(output, summary=f"todos: {todos_summary(todos)}")


def tool_read_todos(cwd: Path, _session=None) -> ToolResult:
    """Return the current session task list (survives compaction and resume)."""
    err = _session_todos(_session)
    if err is not None:
        return err
    todos = _session.get_todos()
    if not todos:
        return ToolResult.success(
            "The task list is empty. Call set_todos to write one.",
            summary="todos: none")
    return ToolResult.success(render_todos(todos),
                              summary=f"todos: {todos_summary(todos)}")
