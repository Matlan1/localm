# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session overlay of the writes patch mode captured instead of applying.

Patch mode never touches disk, so a later read, grep or edit of a file the
agent already changed would otherwise see the original. The overlay holds the
text each captured write produced; read_file, grep, search_replace and the
next captured edit consult it, and the patch is one unified diff per file from
the content on disk to the final overlaid content.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

from .diffutil import read_old_content


class PatchOverlay:
    """Captured file contents for one patch-mode session."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = Path(cwd)
        self._original: dict[str, str] = {}
        self._current: dict[str, str] = {}
        self._labels: dict[str, str] = {}

    def _key(self, path: Path | str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.cwd / p
        try:
            return str(p.resolve())
        except OSError:
            return str(p)

    def _label(self, path: Path | str) -> str:
        p = Path(path)
        if not p.is_absolute():
            return p.as_posix()
        try:
            return p.resolve().relative_to(self.cwd.resolve()).as_posix()
        except (ValueError, OSError):
            return p.as_posix()

    def current(self, path: Path | str) -> Optional[str]:
        """The overlaid text of *path*, or None when no captured write touched it."""
        return self._current.get(self._key(path))

    def read(self, path_arg: str) -> str:
        """The text a tool would see for *path_arg*: the overlay, else disk
        ('' for a file that does not exist)."""
        text = self.current(path_arg)
        if text is not None:
            return text
        return read_old_content(self.cwd, path_arg)

    def record(self, path: Path | str, new_text: str) -> None:
        """Store *new_text* as the content of *path*. The first record of a
        path remembers what was on disk, which the final diff starts from."""
        key = self._key(path)
        if key not in self._original:
            self._original[key] = read_old_content(self.cwd, str(path))
            self._labels[key] = self._label(path)
        self._current[key] = new_text

    def diff(self) -> str:
        """One unified diff per touched file, original to final, in the order
        the files were first touched. A file whose final content equals its
        original contributes nothing."""
        chunks = []
        for key, original in self._original.items():
            label = self._labels[key]
            lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                self._current[key].splitlines(keepends=True),
                fromfile=f"a/{label}", tofile=f"b/{label}"))
            if lines:
                chunks.append("".join(lines))
        return "".join(chunks)

    def has_changes(self) -> bool:
        return any(self._current[k] != o for k, o in self._original.items())

    def clear(self) -> None:
        self._original.clear()
        self._current.clear()
        self._labels.clear()

