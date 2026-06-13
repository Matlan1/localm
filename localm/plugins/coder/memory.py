"""
Persistent project memory for localcoder.

Looks for memory files in cwd (in priority order):
  1. LOCALCODER.md
  2. .localcoder/memory.md

Both files use free-form markdown. The content is injected into the system
prompt under a "## Project Memory" heading so the LLM always sees it.

REPL commands:
  /remember <text>     append a bullet to the memory file
  /forget  <pattern>   remove all bullets whose text contains <pattern>
                       (case-insensitive substring match)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


_CANDIDATES = ["LOCALCODER.md", ".localcoder/memory.md"]


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


def load_memory(cwd: Path) -> str:
    """Return memory file content, stripped, or empty string if none exists."""
    p = find_memory_file(cwd)
    if p is None:
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


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
