# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saving a collection's files as ONE change.

A save stages the new files as ``<file>.<id>.next`` beside the current ones and
then writes the marker ``save.pending``, which names that id and the files the
save holds. The marker is the moment the save takes effect: before it the
collection is what it was, after it the collection is what the staged files say.
The staged files are then moved onto their names and the marker is removed.

A process that ends before the marker leaves the collection unchanged. One that
ends after it leaves the marker: every reader takes a staged file where there
still is one (``read_current``), and the next writer finishes the moves
(``finish_save``, under the collection's write lock)."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

from localm.debuglog import logger as _log
from localm.rag import store as _st
from localm.storekit import replace_atomic

from .cache import _files_changing

T = TypeVar("T")

#: The files a save writes; one a save does not hold is removed by it.
FILES = ("meta.json", "chunks.jsonl", "vectors.json")

#: The marker of a save that took effect and is not finished.
MARKER = "save.pending"

_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")

#: What an interrupted save or file write can leave behind.
_LEFTOVER_RE = re.compile(
    r"\A(?:(?:meta\.json|chunks\.jsonl|vectors\.json)\.[0-9a-f]{32}\.next"
    r"|(?:meta\.json|chunks\.jsonl|vectors\.json|save\.pending)"
    r"(?:\.[0-9a-f]{32}\.next)?\.\d+\.\d+\.tmp)\Z")


def _retrying(step: Callable[[], T]) -> T:
    """``step()``, retried for a moment while another process's handle on the
    file refuses it (Windows ``PermissionError``)."""
    for _ in range(10):
        try:
            return step()
        except PermissionError:
            time.sleep(0.02)
    return step()


def _unlink(path: Path) -> None:
    _retrying(lambda: path.unlink(missing_ok=True))


def _staged_path(coll_dir: Path, name: str, save_id: str) -> Path:
    return coll_dir / f"{name}.{save_id}.next"


def pending_save(coll_dir: Path) -> Optional[tuple[str, tuple[str, ...]]]:
    """(save id, the files it holds) of the save *coll_dir* has pending, or
    None. A marker that does not describe a save is logged and is None; one that
    exists and cannot be read raises ``OSError``."""
    try:
        text = _retrying(lambda: (coll_dir / MARKER).read_text(encoding="utf-8"))
        marker = json.loads(text)
        save_id, files = marker["id"], tuple(marker["files"])
        if not _ID_RE.match(save_id) or not set(files) <= set(FILES):
            raise ValueError("not a save marker")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (ValueError, TypeError, KeyError, RecursionError) as e:
        _log.warning("RAG collection %r: %s cannot be used (%s: %s); reading the "
                     "collection's files as they are", coll_dir.name, MARKER,
                     type(e).__name__, e)
        return None
    return save_id, files


def read_current(coll_dir: Path, name: str, read: Callable[[Path], T]) -> T:
    """``read(path)`` of *name*'s current file in *coll_dir*: the pending save's
    staged file while there is one, else the file itself. Raises what *read*
    raises, ``FileNotFoundError`` when there is no such file."""
    pending = pending_save(coll_dir)
    if pending is not None:
        save_id, files = pending
        try:
            return read(_staged_path(coll_dir, name, save_id))
        except FileNotFoundError:
            if name not in files:
                raise
    return read(coll_dir / name)


class StagedSave:
    """One save of *coll_dir*'s files. MUST be used under its write lock."""

    def __init__(self, coll_dir: Path) -> None:
        self.dir = coll_dir
        self.id = uuid.uuid4().hex
        self.files: list[str] = []

    def path(self, name: str) -> Path:
        """Where *name* is staged; no file when this save removes it."""
        return _staged_path(self.dir, name, self.id)

    def add(self, name: str, content: str | Iterable[str]) -> None:
        self.files.append(name)
        _st._storekit_atomic_write(self.path(name), content)

    def discard(self) -> None:
        """Remove what was staged, unless the save already took effect."""
        try:
            pending = pending_save(self.dir)
        except OSError:
            return
        if pending is not None and pending[0] == self.id:
            return
        for name in self.files:
            _remove_leftover(self.path(name))

    def commit(self) -> None:
        """Make the staged files the collection's and move them into place. A
        move that fails leaves the save in effect and pending, and is logged."""
        _st._storekit_atomic_write(
            self.dir / MARKER, json.dumps({"id": self.id, "files": self.files}))
        try:
            _apply(self.dir, self.id, self.files)
        except OSError as e:
            _log.warning("RAG collection %r: the save took effect, but not every "
                         "file could be moved into place (%s); the next write to "
                         "the collection finishes it", self.dir.name, e)


def _remove_leftover(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        _log.warning("RAG collection %r: could not remove %s (%s); it is not part "
                     "of the index", path.parent.name, path.name, e)


def _apply(coll_dir: Path, save_id: str, files: Iterable[str]) -> None:
    """Move a save's staged files into place, remove the files it does not hold,
    then its marker. Repeatable: a file already moved is skipped."""
    for name in FILES:
        if name not in files:
            _unlink(coll_dir / name)
            continue
        try:
            replace_atomic(_staged_path(coll_dir, name, save_id), coll_dir / name)
        except FileNotFoundError:
            continue
    _unlink(coll_dir / MARKER)


def finish_save(coll_dir: Path) -> None:
    """Finish a save that an earlier writer left pending, and remove what an
    interrupted one left behind. MUST run under the collection's write lock.
    Raises ``OSError`` when a file of the pending save cannot be moved. Behind a
    marker that does not describe a save, nothing is touched."""
    pending = pending_save(coll_dir)
    if pending is not None:
        with _files_changing(coll_dir):
            _apply(coll_dir, *pending)
        _log.info("RAG collection %r: finished an interrupted save", coll_dir.name)
    elif (coll_dir / MARKER).exists():
        return
    try:
        left = [p for p in coll_dir.iterdir() if _LEFTOVER_RE.match(p.name)]
    except OSError:
        return
    for path in left:
        _remove_leftover(path)
