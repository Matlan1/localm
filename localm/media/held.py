# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generated media held only while this process runs.

A result that must not be kept (a privacy session with ``audit.privacy_keeps``
off for media) is generated into this process's own folder,
``<data dir>/run/private-media/<pid>-<random>/<kind>/``, where it can be shown,
downloaded and moved like a gallery file. The folder is deleted when the process
ends (:func:`discard_all`); :func:`sweep_leftovers` deletes the folders of
processes that ended without doing so. No owner index, sidecar or other record
of a held file is written: its owner lives in memory here.

A private ComfyUI job in progress is noted in ``<pid>-<random>/comfy/`` by its
address and job id, so a later server start can say that one was cut off. The
note stays until the app has shown it.

Whether a folder's process still runs is decided by an OS lock on
``<pid>-<random>.lock`` beside it, which the system drops when that process ends
for any reason."""

from __future__ import annotations

import atexit
import contextlib
import filecmp
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import IO, Callable, Iterator, Optional

from localm.debuglog import logger
from localm.vram import MEDIA_CLOSING as CLOSING

KINDS = ("image", "music", "video")

_REMOVE_ATTEMPTS = 5
_REMOVE_RETRY_SECONDS = 0.2

_LOCK = threading.RLock()

_COMFY_NOTES = "comfy"
_ENDED = ".ended"

#: The owner of a held file made while no key existed, once one does: nobody's id,
#: so only the owner key has it.
OWNER_ONLY = "owner-only"

_END_LIMIT_S = 12.0

_runs: list[list[Callable[[], bool]]] = []
_runs_cv = threading.Condition()


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
            if not (_try_lock(self._file) and self.lock_file_in_place()):
                raise OSError(f"cannot lock {self.lock_path}")
            self.dir.mkdir(mode=0o700)
        except OSError:
            self._file.close()
            raise

    def lock_file_in_place(self) -> bool:
        """Whether the locked file is still the one at ``lock_path``; another
        process's sweep can remove it before the lock is taken."""
        if sys.platform == "win32":
            return True   # An open file cannot be removed there.
        try:
            return os.path.samestat(os.fstat(self._file.fileno()), os.stat(self.lock_path))
        except FileNotFoundError:
            return False

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


def _open_session() -> _Session:
    """This process's session, created when there is none (which also removes what
    earlier processes left), its folder made again when it vanished. Caller holds
    ``_LOCK``; raises ``OSError``."""
    global _session, _exit_registered
    s = _current()
    if s is None or not s.lock_file_in_place():
        discard_all()
        sweep_leftovers()
        s = _session = _Session(root_dir())
        if not _exit_registered:
            atexit.register(discard_all)
            _exit_registered = True
    s.dir.mkdir(mode=0o700, exist_ok=True)
    return s


def folder(kind: str) -> Path:
    """This process's folder for held *kind* files, created on first use. Raises
    ``OSError`` when it cannot be created and while localm is stopping."""
    if kind not in KINDS:
        raise ValueError(f"unknown media kind {kind!r}")
    with _LOCK:
        if CLOSING.is_set():
            raise OSError("localm is stopping")
        d = _open_session().dir / kind
        d.mkdir(mode=0o700, exist_ok=True)
        return d


def note_comfy_job(token: str, api_url: str, prompt_id: str) -> Optional[Path]:
    """Record, with no content, that a private job was sent to the ComfyUI at
    *api_url*, in the note that is returned, until :func:`forget_comfy_job` of it.
    Never raises ``OSError``: without the record (None) a job cut off by a crash is
    not reported at the next start."""
    try:
        with _LOCK:
            note = _open_session().dir / _COMFY_NOTES / token
            note.parent.mkdir(mode=0o700, exist_ok=True)
            note.write_text(json.dumps([api_url, prompt_id]), encoding="utf-8")
            return note
    except OSError as e:
        logger.warning("held media: a private ComfyUI job could not be noted (%s); if localm "
                       "is killed now, its next start does not report it", e)
        return None


def forget_comfy_job(note: Path) -> None:
    """Drop the *note* of an ended job. One that cannot be deleted now is marked as
    ended and deleted later. Never raises ``OSError``."""
    with _LOCK:
        try:
            note.unlink(missing_ok=True)
        except OSError:
            try:
                note.with_name(note.name + _ENDED).touch()
            except OSError as e:
                logger.warning("held media: the note of an ended ComfyUI job stays (%s); a "
                               "later start may report that job as cut off", e)


def _drop_ended_notes(session_dir: Path) -> bool:
    """Delete the notes marked as ended in *session_dir*. Returns whether they are
    gone."""
    gone = True
    for mark in list((session_dir / _COMFY_NOTES).glob("*" + _ENDED)):
        try:
            mark.with_name(mark.name.removesuffix(_ENDED)).unlink(missing_ok=True)
            mark.unlink(missing_ok=True)
        except OSError as e:
            gone = False
            logger.debug("held media: the note of an ended ComfyUI job stays (%s)", e)
    return gone


def _cut_off_jobs(session_dir: Path) -> list[tuple[str, Path]]:
    """What to tell the owner about each ComfyUI job noted in *session_dir* that did
    not end, with its note."""
    told = []
    d = session_dir / _COMFY_NOTES
    names = {p.name for p in d.iterdir()} if d.is_dir() else set()
    for name in sorted(names):
        if name.endswith(_ENDED) or name + _ENDED in names:
            continue
        p = d / name
        try:
            url, prompt_id = json.loads(p.read_text(encoding="utf-8"))
            url, prompt_id = (re.sub(r"[^\x21-\x7e]", "", str(x))[:200]
                              for x in (url, prompt_id))
        except (OSError, ValueError, TypeError, RecursionError):
            url, prompt_id = "an unknown address", "unknown"
        told.append((f"A private ComfyUI job was cut off when localm ended. The ComfyUI at {url} "
                     "may still run it or keep its files, the image it was given and its history "
                     f"entry (job {prompt_id}): remove them there.", p))
    return told


def dismiss_note(note: Path) -> None:
    """Delete the *note* of a cut-off ComfyUI job, which the owner has seen, and what
    is empty around it. Never raises ``OSError``."""
    with _LOCK:
        try:
            note.unlink(missing_ok=True)
            note.parent.rmdir()
            note.parent.parent.rmdir()
        except OSError as e:
            logger.debug("held media: %s stays (%s)", note.parent, e)
        _drop_empty(root_dir())


def current_folder(kind: str) -> Optional[Path]:
    """The folder :func:`folder` returned for *kind*, or None when this process
    holds no file of that kind. Creates nothing."""
    s = _current()
    if s is None:
        return None
    d = s.dir / kind
    return d if d.is_dir() else None


def is_held(path) -> bool:
    """True when *path* is in a held folder, of this process or an ended one."""
    return root_dir() in Path(path).parents


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
    """The recorded owner of the held file *name*, or None. A file recorded without
    one is :data:`OWNER_ONLY` while a key exists."""
    s = _current()
    owner = None if s is None else s.owners.get((kind, name))
    if owner is None and tracks(kind, name):
        from localm.auth import any_key_configured
        return OWNER_ONLY if any_key_configured() else None
    return owner


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


def save_copy(source: Path, target: Path,
              created: Optional[Callable[[], None]] = None) -> None:
    """Copy the held file *source* to the new file *target*, which gets the rights
    of its own folder, and compare the two. *created* is called once *target*
    exists, before its content does. Raises ``OSError``, leaving no *target*, when
    it cannot (``FileExistsError`` when *target* exists)."""
    with open(target, "xb") as dst:
        try:
            if created is not None:
                created()
            with open(source, "rb") as src:
                shutil.copyfileobj(src, dst)
            dst.close()
            if not filecmp.cmp(source, target, shallow=False):
                raise OSError("the copy differs from the original")
        except BaseException:
            dst.close()
            target.unlink(missing_ok=True)
            raise


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


def _clear(session_dir: Path) -> bool:
    """Delete *session_dir*, or all of it but its ComfyUI notes while one must stay.
    Returns whether that is gone."""
    if not _drop_ended_notes(session_dir) or _cut_off_jobs(session_dir):
        return all([_remove_tree(p) for p in session_dir.iterdir() if p.name != _COMFY_NOTES])
    return _remove_tree(session_dir)


def discard_all() -> bool:
    """Delete everything this process holds, except the notes of ComfyUI jobs that
    did not end, which a later start reports. Returns False when a file could not
    be deleted (logged as a warning); the next :func:`sweep_leftovers` tries
    again."""
    global _session
    with _LOCK:
        s = _session
        _session = None
        if s is None or s.pid != os.getpid():
            return True
        gone = _clear(s.dir)
        s.release()
        if gone:
            _unlink_lock(s.lock_path)
            _drop_empty(s.root)
        return gone


def _unlink_lock(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.debug("held media: the lock file %s stays (%s)", path, e)


def _drop_empty(root: Path) -> None:
    try:
        root.rmdir()
    except OSError:
        pass   # See test_the_root_stays_while_another_process_holds_media.


def sweep_leftovers(tell: Optional[Callable[[str, Path], None]] = None) -> int:
    """Delete the folders of processes that are gone, and the owner index entries
    of the files in them. Returns how many folders were deleted. Each ComfyUI job
    such a process left unfinished is given to *tell*, with the note that stays
    until :func:`dismiss_note`. Never raises ``OSError``; a folder that could not be
    deleted is logged as a warning and tried again at the next call."""
    removed = 0
    told: list[tuple[str, Path]] = []
    names: list[tuple[str, str]] = []
    root = root_dir()
    with _LOCK:
        own = _current()
        try:
            entries = list(root.iterdir())
        except FileNotFoundError:
            return 0
        except OSError as e:
            logger.warning("held media: cannot list %s (%s); media an earlier run left may "
                           "still be there", root, e)
            return 0
        ids = {p.stem for p in entries if p.suffix == ".lock"}
        ids |= {p.name for p in entries if _is_plain_dir(p)}
        for sid in sorted(ids):
            lock, left = root / f"{sid}.lock", root / sid
            if own is not None and sid == own.id:
                continue
            with _of_gone_process(lock) as gone:
                if not gone:
                    continue
                if _is_plain_dir(left):
                    files_left = [(p.parent.name, p.name) for p in left.glob("*/*")]
                    if not _clear(left):
                        continue
                    names += [n for n in files_left if n[0] in KINDS]
                    removed += not left.exists()
                    told += _cut_off_jobs(left)
                if sys.platform != "win32":
                    _unlink_lock(lock)
            if sys.platform == "win32":
                _unlink_lock(lock)
        _drop_empty(root)
    if removed:
        logger.info("removed generated media left by %d earlier run(s)", removed)
    if names:
        from localm.media import gallery
        gallery.forget_never_kept(names)
    if tell is not None:
        for text, note in told:
            logger.warning("%s", text)
            tell(text, note)
    return removed


def _is_plain_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and not path.is_junction()


@contextlib.contextmanager
def _of_gone_process(lock: Path) -> Iterator[bool]:
    """Whether no process holds the lock file *lock* (none does when there is no such
    file). When so, the lock is taken for the ``with`` block."""
    try:
        f = open(lock, "r+b")
    except FileNotFoundError:
        yield True
        return
    except OSError as e:
        logger.warning("held media: cannot check %s (%s); its folder is left alone", lock, e)
        yield False
        return
    with f:
        yield _try_lock(f)


@contextlib.contextmanager
def run(not_kept: Callable[[], bool],
        cancel: Optional[Callable[[], None]] = None) -> Iterator[Callable[[], bool]]:
    """Hold this while a generation runs whose result must not be kept once
    *not_kept* says so; *not_kept* is called from other threads. Yields a function:
    whether the generation must end because localm is stopping, which then calls
    *cancel*."""
    entry = [not_kept]

    def ending() -> bool:
        if not (CLOSING.is_set() and not_kept()):
            return False
        if cancel is not None:
            cancel()
        return True
    with _runs_cv:
        _runs.append(entry)
    try:
        yield ending
    finally:
        with _runs_cv:
            _runs[:] = [r for r in _runs if r is not entry]
            _runs_cv.notify_all()


def private_run_wait() -> float:
    """How many seconds a stop waits at most for the generations running now whose
    result must not be kept; 0 when there is none."""
    with _runs_cv:
        return _END_LIMIT_S if any(not_kept() for (not_kept,) in _runs) else 0.0


def end_private_runs() -> bool:
    """End every running generation whose result must not be kept, refuse new ones,
    and wait (``_END_LIMIT_S`` at most) until they have ended and removed what they
    left in ComfyUI. Other generations are left alone. Returns whether they ended;
    one that did not is logged as a warning. ``CLOSING`` stays set."""
    CLOSING.set()
    with _runs_cv:
        ended = _runs_cv.wait_for(lambda: not private_run_wait(), _END_LIMIT_S)
    if not ended:
        logger.warning("a private generation did not end within %g s of the stop; what it "
                       "writes from now on is removed at the next start", _END_LIMIT_S)
    return ended
