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
import re
import threading
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
        self._lock = threading.RLock()

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
        with self._lock:
            if key not in self._original:
                self._original[key] = read_old_content(self.cwd, str(path))
                self._labels[key] = self._label(path)
            self._current[key] = new_text

    def diff(self) -> str:
        """One unified diff per touched file, original to final, in the order
        the files were first touched. A file whose final content equals its
        original contributes nothing. A last line without a newline is followed
        by git's "\\ No newline at end of file" marker, so the diffs of several
        files concatenate into one patch ``git apply`` accepts."""
        with self._lock:
            items = [(self._labels[k], o, self._current[k])
                     for k, o in self._original.items()]
        chunks = []
        for label, original, current in items:
            for line in difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{label}", tofile=f"b/{label}"):
                chunks.append(line if line.endswith("\n")
                              else line + "\n\\ No newline at end of file\n")
        return "".join(chunks)

    def has_changes(self) -> bool:
        with self._lock:
            return any(self._current[k] != o for k, o in self._original.items())

    def new_files(self, base: Path, glob: str) -> list[Path]:
        """Overlaid files that do not exist on disk, below *base* and matching
        *glob* the way ``Path.glob`` reads it, for sweeps that enumerate disk."""
        rx = _glob_regex(glob)
        base_resolved = Path(base).resolve()
        found = []
        with self._lock:
            keys = list(self._current)
        for key in keys:
            path = Path(key)
            if path.exists():
                continue
            try:
                rel = path.relative_to(base_resolved).as_posix()
            except ValueError:
                continue
            if rx.fullmatch(rel):
                found.append(base_resolved / rel)
        return found

    def clear(self) -> None:
        with self._lock:
            self._original.clear()
            self._current.clear()
            self._labels.clear()


def _glob_regex(glob: str) -> re.Pattern[str]:
    """A regex for a ``Path.glob`` pattern: ``**/`` any directories, ``*`` and
    ``?`` within one path segment."""
    out = []
    i = 0
    g = glob.replace("\\", "/")
    while i < len(g):
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        elif g[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("".join(out))
