# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated media held only while this process runs.

A result that must not be kept (a privacy session with ``audit.privacy_keeps``
off for media) is generated into this process's own folder,
``<data dir>/run/private-media/<pid>-<random>/<kind>/``, where it can be shown,
downloaded and moved like a gallery file. The folder is deleted when the process
ends (:func:`discard_all`); :func:`sweep_leftovers` deletes the folders of
processes that ended without doing so. No owner index, sidecar or other record
of a held file is written: its owner lives in memory here.

Whether a folder's process still runs is decided by an OS lock on
``<pid>-<random>.lock`` beside it, which the system drops when that process ends
for any reason."""

from __future__ import annotations

import atexit
import os
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import IO, Optional

from localm.debuglog import logger

KINDS = ("image", "music", "video")

_REMOVE_ATTEMPTS = 5
_REMOVE_RETRY_SECONDS = 0.2

_LOCK = threading.RLock()


def root_dir() -> Path:
    """The folder that holds every process's folder of held media."""
    from localm.config import home_dir
    return home_dir() / "run" / "private-media"


def _try_lock(f: IO) -> bool:
    """Take an exclusive OS lock on the open file *f* without waiting; False when
    another holder has it."""
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


class _Session:
    """This process's folder, its held lock, and the owner of each file in it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pid = os.getpid()
        self.id = f"{self.pid}-{secrets.token_hex(6)}"
        self.dir = root / self.id
        self.lock_path = root / f"{self.id}.lock"
        self.owners: dict[tuple[str, str], Optional[str]] = {}
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        self._file = open(self.lock_path, "a+b")
        try:
            if not _try_lock(self._file):
                raise OSError(f"cannot lock {self.lock_path}")
            self.dir.mkdir(mode=0o700)
        except OSError:
            self._file.close()
            raise

    def release(self) -> None:
        self._file.close()


_session: Optional[_Session] = None
_exit_registered = False


def _current() -> Optional[_Session]:
    """This process's session, or None when it has none for the current data dir."""
    s = _session
    if s is None or s.pid != os.getpid() or s.root != root_dir():
        return None
    return s


def folder(kind: str) -> Path:
    """This process's folder for held *kind* files, created on first use (which
    also removes what earlier processes left). Raises ``OSError`` when it cannot
    be created."""
    global _session, _exit_registered
    if kind not in KINDS:
        raise ValueError(f"unknown media kind {kind!r}")
    with _LOCK:
        s = _current()
        if s is None:
            discard_all()
            sweep_leftovers()
            s = _session = _Session(root_dir())
            if not _exit_registered:
                atexit.register(discard_all)
                _exit_registered = True
        d = s.dir / kind
        d.mkdir(mode=0o700, exist_ok=True)
        return d


def current_folder(kind: str) -> Optional[Path]:
    """The folder :func:`folder` returned for *kind*, or None when this process
    holds no file of that kind. Creates nothing."""
    s = _current()
    if s is None:
        return None
    d = s.dir / kind
    return d if d.is_dir() else None


def is_held(path) -> bool:
    """True when *path* is in this process's held folder."""
    s = _current()
    return s is not None and s.dir in Path(path).parents


def files(kind: str) -> list[Path]:
    """The held files of *kind*, newest first."""
    d = current_folder(kind)
    if d is None:
        return []
    found = []
    try:
        entries = list(d.iterdir())
    except FileNotFoundError:
        return []
    for p in entries:
        try:
            if p.is_file() and not p.name.startswith("."):
                found.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    return [p for _t, p in sorted(found, key=lambda x: x[0], reverse=True)]


def set_owner(kind: str, name: str, owner: Optional[str]) -> None:
    """Record *owner* for the held file *name*, in memory only."""
    with _LOCK:
        s = _current()
        if s is not None:
            s.owners[(kind, name)] = owner


def tracks(kind: str, name: str) -> bool:
    """True when an owner (None included) is recorded for the held file *name*."""
    s = _current()
    return s is not None and (kind, name) in s.owners


def owner_of(kind: str, name: str) -> Optional[str]:
    """The recorded owner of the held file *name*, or None."""
    s = _current()
    return None if s is None else s.owners.get((kind, name))


def rename(kind: str, old: str, new: str) -> None:
    """Carry the owner record of a held file across a rename."""
    with _LOCK:
        s = _current()
        if s is not None and (kind, old) in s.owners:
            s.owners[(kind, new)] = s.owners.pop((kind, old))


def forget(kind: str, name: str) -> None:
    """Drop the owner record of the held file *name*."""
    with _LOCK:
        s = _current()
        if s is not None:
            s.owners.pop((kind, name), None)


def _remove_tree(path: Path) -> bool:
    """Delete *path*, retrying while another program has a file in it open.
    Returns whether it is gone; a failure is logged as a warning."""
    error: Optional[OSError] = None
    for attempt in range(_REMOVE_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            if not path.exists():
                return True
        except OSError as e:
            error = e
        if attempt + 1 < _REMOVE_ATTEMPTS:
            time.sleep(_REMOVE_RETRY_SECONDS)
    logger.warning("generated media that must not be kept could not be removed from %s "
                   "(%s); localm removes it at its next start", path, error)
    return False


def discard_all() -> bool:
    """Delete everything this process holds. Returns False when a file could not
    be deleted (logged as a warning); the next :func:`sweep_leftovers` tries
    again."""
    global _session
    with _LOCK:
        s = _session
        _session = None
        if s is None or s.pid != os.getpid():
            return True
        gone = _remove_tree(s.dir)
        s.release()
        if gone:
            _unlink_lock(s.lock_path)
            _drop_empty(s.root)
        return gone


def _unlink_lock(path: Path) -> None:
    try:
        path.unlink()
    except OSError as e:
        logger.debug("held media: the lock file %s stays (%s)", path, e)


def _drop_empty(root: Path) -> None:
    try:
        root.rmdir()
    except OSError:
        pass   # See test_the_root_stays_while_another_process_holds_media.


def sweep_leftovers() -> int:
    """Delete the folders of processes that are gone. Returns how many were
    deleted. Never raises; a folder that could not be deleted is logged as a
    warning and tried again at the next call."""
    removed = 0
    root = root_dir()
    with _LOCK:
        own = _current()
        try:
            entries = sorted(root.iterdir())
        except FileNotFoundError:
            return 0
        except OSError as e:
            logger.warning("held media: cannot list %s (%s); media an earlier run left may "
                           "still be there", root, e)
            return 0
        locks = {p.stem: p for p in entries if p.suffix == ".lock"}
        ids = set(locks) | {p.name for p in entries if p.suffix != ".lock"}
        for sid in sorted(ids):
            if own is not None and sid == own.id:
                continue
            lock = locks.get(sid)
            if lock is not None and not _process_gone(lock):
                continue
            left = root / sid
            if left.exists():
                if not _remove_tree(left):
                    continue
                removed += 1
            if lock is not None:
                _unlink_lock(lock)
        _drop_empty(root)
    if removed:
        logger.info("removed generated media left by %d earlier run(s)", removed)
    return removed


def _process_gone(lock: Path) -> bool:
    """True when no process holds the lock file *lock*."""
    try:
        with open(lock, "a+b") as f:
            return _try_lock(f)
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("held media: cannot check %s (%s); its folder is left alone", lock, e)
        return False
