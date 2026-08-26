# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Persistent project memory for localcoder.

Looks for memory files in cwd (in priority order):
  1. LOCALCODER.md
  2. .localcoder/memory.md

Both files use free-form markdown. The content is injected into the system
prompt under a "## Project Memory" heading so the LLM always sees it, capped at
_MAX_MEMORY_CHARS so it cannot crowd out the repo map and the conversation.

The file is USER-managed, not agent-managed: it changes only when the user runs
/remember or /forget (or edits it by hand). The agent has no tool that writes it,
and close-time reflection writes episodes to the localm home data dir, not here.

REPL commands:
  /remember <text>     append a bullet to the memory file
  /forget  <pattern>   remove all bullets whose text contains <pattern>
                       (case-insensitive substring match)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


_CANDIDATES = ["LOCALCODER.md", ".localcoder/memory.md"]


# Injection budgets, in characters.
#
# Everything under these headings is paid on EVERY turn, and it competes with the
# repo map (capped at _MAX_MAP_CHARS = 3000, indexer.py) and the conversation
# itself.
#
# An oversized system prompt is UNRECOVERABLE at runtime: _compact_history
# (agent/context.py) only rewrites the message list, never the system prompt,
# while context_chars() counts the system prompt in the fill ratio. A runaway
# memory file therefore leaves the agent permanently past its auto-compact
# threshold, shredding real conversation history every turn without ever getting
# back under budget.
#
# These are runaway guards, not routine trimmers, and should never fire in normal
# use. When one does fire the drop is surfaced both to the model (an in-band
# notice) and to the user (see memory_warning / custom_instructions_warning),
# never hidden.
_MAX_MEMORY_CHARS = 3_000
_MAX_CUSTOM_INSTRUCTIONS_CHARS = 3_000


def _split_for_injection(text: str, limit: int) -> tuple:
    """
    Split *text* into ``(kept, omitted_chars)`` at the injection budget.

    Returns ``(text, 0)`` when it already fits. Otherwise the kept part is cut
    back to a line boundary (unless that would cost more than half the budget),
    because slicing mid-line can invert a bullet's meaning: "- never force-push
    to master" truncated at "- never force-push" reads as the opposite advice.

    Both the in-band notice and the user-facing warning derive their numbers from
    this one function, so they can never disagree about how much was dropped.
    """
    if len(text) <= limit:
        return text, 0

    head = text[:limit]
    cut = head.rfind("\n")
    if cut > limit // 2:      # whole lines, but not at the cost of most of the budget
        head = head[:cut]
    head = head.rstrip()
    return head, len(text) - len(head)


def _cap_for_injection(text: str, limit: int, what: str, remedy: str) -> str:
    """
    Trim *text* to the injection budget, leaving a notice the MODEL can see.

    Returns *text* unchanged when it fits, so the common case is byte-identical
    to no capping at all. When it does not fit, a bracketed notice naming the
    omitted character count is appended, so the model reads a file it knows is
    partial instead of silently trusting a truncated one.
    """
    kept, omitted = _split_for_injection(text, limit)
    if not omitted:
        return kept
    return (
        f"{kept}\n\n[... {omitted} characters of {what} omitted: the file is over "
        f"the {limit}-character injection budget. {remedy} ...]"
    )


def _overflow_warning(path: Optional[Path], raw: str, limit: int,
                      what: str, remedy: str) -> str:
    """User-facing warning when *raw* does not fit *limit*, else an empty string."""
    if path is None:
        return ""
    kept, omitted = _split_for_injection(raw, limit)
    if not omitted:
        return ""
    return (
        f"{what} {path.name} is {len(raw)} characters; only the first {len(kept)} "
        f"are injected into the system prompt ({omitted} omitted). {remedy}"
    )


def find_memory_file(cwd: Path) -> Optional[Path]:
    """Return the first memory file that exists under cwd, or None."""
    for name in _CANDIDATES:
        p = cwd / name
        if p.is_file():
            return p
    return None


def default_memory_file(cwd: Path) -> Path:
    """Preferred path when creating a new memory file from scratch."""
    return cwd / "LOCALCODER.md"


def _read_injectable(p: Optional[Path]) -> tuple:
    """
    Read *p* for injection. Returns ``(stripped_text, unreadable)``.

    ``unreadable`` distinguishes two cases: a file that is simply ABSENT (normal,
    nothing to say) versus one that EXISTS but could not be read (a locked,
    corrupt, or permission-denied file). The second is surfaced by the *_warning
    helpers below. Either way the text is empty, so an unreadable file degrades to
    "no memory" instead of killing the session.
    """
    if p is None:
        return "", False
    try:
        return p.read_text(encoding="utf-8").strip(), False
    except OSError:
        return "", True


_MEMORY_REMEDY = "Trim it, or use /forget <pattern> to drop stale entries."


def load_memory(cwd: Path) -> str:
    """Return memory file content, stripped and capped for injection, or empty
    string if none exists.

    Capped at _MAX_MEMORY_CHARS: content over the budget is cut back and leaves a
    visible notice, never dropped silently. Use :func:`memory_warning` to tell the
    user when that happened.
    """
    text, _ = _read_injectable(find_memory_file(cwd))
    return _cap_for_injection(text, _MAX_MEMORY_CHARS, "project memory", _MEMORY_REMEDY)


def memory_warning(cwd: Path) -> str:
    """
    A user-facing warning about the project-memory file, or "" when all is well.

    Covers both ways the injected memory can differ from what is on disk: the
    file is over the injection budget, or it exists but could not be read.
    """
    p = find_memory_file(cwd)
    raw, unreadable = _read_injectable(p)
    if unreadable:
        return (f"Project memory {p} could not be read; the agent is running "
                "without it.")
    return _overflow_warning(p, raw, _MAX_MEMORY_CHARS, "Project memory", _MEMORY_REMEDY)


# User-authored custom instructions, distinct from the project MEMORY above:
# memory is a running list of project FACTS added by the user with /remember,
# whereas system.md is hand-written guidance the user wants the agent to follow
# (conventions, style, constraints). It is injected into the system prompt under
# "## User Instructions".
CUSTOM_INSTRUCTIONS_FILE = ".localcoder/system.md"

_INSTRUCTIONS_REMEDY = "Shorten it so the whole file is followed."


def custom_instructions_file(cwd: Path) -> Path:
    """Path to the custom-instructions file for *cwd* (may not exist)."""
    return cwd / CUSTOM_INSTRUCTIONS_FILE


def _existing_instructions_file(cwd: Path) -> Optional[Path]:
    p = custom_instructions_file(cwd)
    return p if p.is_file() else None


def load_custom_instructions(cwd: Path) -> str:
    """Return the contents of ``.localcoder/system.md`` (stripped and capped for
    injection), or empty string when the file does not exist or cannot be read.

    Capped at _MAX_CUSTOM_INSTRUCTIONS_CHARS, loudly: the model sees a notice that
    the file was cut, and :func:`custom_instructions_warning` tells the user which
    of their instructions are not being followed.
    """
    text, _ = _read_injectable(_existing_instructions_file(cwd))
    return _cap_for_injection(text, _MAX_CUSTOM_INSTRUCTIONS_CHARS,
                              "user instructions", _INSTRUCTIONS_REMEDY)


def cap_user_instructions(text: str) -> str:
    """Cap an explicit ``--system`` string to the same budget as system.md.

    The flag bypasses :func:`load_custom_instructions` entirely, so without this
    the documented way to set user instructions would still be uncapped and the
    "## User Instructions" section would stay unbounded.
    """
    return _cap_for_injection(text.strip(), _MAX_CUSTOM_INSTRUCTIONS_CHARS,
                              "user instructions", _INSTRUCTIONS_REMEDY)


def custom_instructions_warning(cwd: Path, override: Optional[str] = None) -> str:
    """A user-facing warning about the user instructions, or "" when fine.

    When *override* is given (the ``--system`` flag) it is what actually gets
    injected, so the file on disk is not consulted.
    """
    if override is not None:
        raw = override.strip()
        kept, omitted = _split_for_injection(raw, _MAX_CUSTOM_INSTRUCTIONS_CHARS)
        if not omitted:
            return ""
        return (
            f"The --system instructions are {len(raw)} characters; only the first "
            f"{len(kept)} are injected into the system prompt ({omitted} omitted). "
            f"{_INSTRUCTIONS_REMEDY}"
        )

    p = _existing_instructions_file(cwd)
    raw, unreadable = _read_injectable(p)
    if unreadable:
        return (f"User instructions {p} could not be read; the agent is running "
                "without them.")
    return _overflow_warning(p, raw, _MAX_CUSTOM_INSTRUCTIONS_CHARS,
                             "User instructions", _INSTRUCTIONS_REMEDY)


def remember(cwd: Path, text: str) -> Path:
    """
    Append a new bullet point to the memory file.

    Creates ``LOCALCODER.md`` in cwd if no memory file exists yet.
    Silently skips if the exact bullet is already present.

    Returns the path of the file that was written.
    """
    text = text.strip()
    if not text:
        raise ValueError("Memory entry cannot be empty.")

    bullet = f"- {text}"
    p = find_memory_file(cwd) or default_memory_file(cwd)

    if p.is_file():
        existing = p.read_text(encoding="utf-8")
        # Exact duplicate guard
        if bullet in existing.splitlines():
            return p
        # Ensure there's a newline separator before the new bullet
        sep = "" if existing.endswith("\n") else "\n"
        new_content = existing + sep + bullet + "\n"
    else:
        # First entry - create with a header
        p.parent.mkdir(parents=True, exist_ok=True)
        new_content = f"# Project Memory\n\n{bullet}\n"

    p.write_text(new_content, encoding="utf-8")
    return p


def forget(cwd: Path, pattern: str) -> tuple:
    """
    Remove all bullet lines whose text contains *pattern* (case-insensitive).

    Non-bullet lines (headers, blank lines, etc.) are always preserved.

    Returns ``(file_path, removed_count)``.
    Returns ``(None, 0)`` if no memory file exists.
    """
    p = find_memory_file(cwd)
    if p is None:
        return None, 0

    lo = pattern.strip().lower()
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

    kept: list[str] = []
    removed = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("- ") and lo in stripped[2:].lower():
            removed += 1
        else:
            kept.append(line)

    p.write_text("".join(kept), encoding="utf-8")
    return p, removed
