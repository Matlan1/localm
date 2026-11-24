# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record of an update, its lock and its phases, and the end of an update:
finished (:func:`finish`) or undone (:func:`restore`).

The record is on disk before anything changes and is removed last. While it is
there and its lock is free, nobody is updating: :func:`restore` then puts the old
version back, whatever the phase, or removes what a finished update left.
"""

from __future__ import annotations

import contextlib
import os
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import disk, mover, packages
from .disk import Unsafe
from .mover import StepFailed, Unreadable

__all__ = ["Busy", "Lock", "Record", "Result", "StepFailed", "Unreadable", "Unsafe",
           "advance", "cleanup", "finish", "locked", "read", "restore", "start"]

FORMAT = "localm-update-job/1"
MAX_RECORD_BYTES = 64 * 1024

ENDING, MOVING, STEPS, STARTING, DONE, RESTORING = (
    "ending", "moving", "steps", "starting", "done", "restoring")
#: Every phase, in order; an update that fails goes to RESTORING from any but DONE.
PHASES = (ENDING, MOVING, STEPS, STARTING, DONE, RESTORING)

#: Seconds :class:`Lock` waits, longer than another process's :func:`locked` has it.
LOCK_WAIT_S = 2.0
_FINISHED = "finished"
_RESERVE = "reserve"
_RESERVE_BYTES = 256 * 1024


class Busy(Exception):
    """Another process is updating this install."""


# ------------------------------------------------------------------- lock

class Lock:
    """The lock of the install at *root*, a context manager. The system frees it
    when its holder ends. Entering raises :class:`Busy` when another holder has it
    for *wait_s* seconds, and :class:`Unsafe` or OSError when it cannot be used."""

    def __init__(self, root, *, wait_s: float = LOCK_WAIT_S) -> None:
        self.root, self.wait_s = Path(root), wait_s
        self._fd: Optional[int] = None

    def __enter__(self) -> Lock:
        path = mover.update_folder(self.root, create=True) / "lock"
        fd = disk.open_lock(path)
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                disk.lock_file(fd)
            except (PermissionError, BlockingIOError) as e:
                if time.monotonic() < deadline:
                    time.sleep(0.05)
                    continue
                os.close(fd)
                raise Busy(f"another process is updating {self.root}") from e
            except BaseException:
                os.close(fd)
                raise
            self._fd = fd
            return self

    def __exit__(self, *_exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        """Whether this object holds the lock now."""
        return self._fd is not None


def locked(root) -> bool:
    """Whether a process holds the lock of the install at *root* now. Creates
    nothing where no update was ever made."""
    if disk.lstat(mover.update_folder(root) / "lock") is None:
        return False
    try:
        with Lock(root, wait_s=0):
            return False
    except Busy:
        return True


# ------------------------------------------------------------------ record

@dataclass
class Record:
    """One update: its id (32 hex digits), its phase and the versions."""
    id: str
    phase: str
    old: str
    new: str


def record_path(root) -> Path:
    """Where the record of the install at *root* is."""
    return mover.update_folder(root) / "job.json"


def read(root) -> Optional[Record]:
    """The record of the install at *root*, None when no update is unfinished.
    Raises :class:`Unreadable` for a damaged one, :class:`Unsafe` for a link."""
    try:
        raw = disk.read_json(record_path(root), MAX_RECORD_BYTES)
    except ValueError as e:
        raise Unreadable(str(e)) from None
    if raw is None:
        return None
    if (not isinstance(raw, dict) or set(raw) != {"format", "id", "phase", "old", "new"}
            or raw["format"] != FORMAT or raw["phase"] not in PHASES
            or not all(isinstance(raw[k], str) and len(raw[k]) <= 64 for k in ("old", "new"))):
        raise Unreadable("job.json is not the record of an update")
    return Record(mover.check_id(raw["id"]), raw["phase"], raw["old"], raw["new"])


def _write(lock: Lock, record: Record) -> None:
    if not lock.held:
        raise ValueError("the lock of the install is not held")
    disk.write_json(record_path(lock.root), {"format": FORMAT, **asdict(record)})


def start(lock: Lock, update_id: str, *, old: str, new: str) -> Record:
    """Begin the update *update_id*, whose folder already holds ``new``: write its
    record in phase ENDING. Raises :class:`Busy` when another update is unfinished,
    :class:`Unreadable`, :class:`Unsafe` or OSError."""
    if read(lock.root) is not None:
        raise Busy("an earlier update of this install is not finished")
    work = mover.work_folder(lock.root, update_id)
    (work / "plans").mkdir(parents=True, exist_ok=True)
    with open(work / _RESERVE, "wb") as f:
        f.write(b"\0" * _RESERVE_BYTES)
    record = Record(mover.check_id(update_id), ENDING, old, new)
    _write(lock, record)
    return record


def advance(lock: Lock, phase: str) -> Record:
    """Write *phase*, the next one of :data:`PHASES` or a repeat of the current one,
    into the record before that phase begins. Raises ValueError for any other phase
    or without a record, and OSError."""
    record = read(lock.root)
    if record is None:
        raise ValueError("there is no update to advance")
    steps = PHASES.index(phase) - PHASES.index(record.phase)
    if phase in (DONE, RESTORING) or steps not in (0, 1):
        raise ValueError(f"an update cannot go from {record.phase} to {phase}")
    disk.sync_disk()
    record.phase = phase
    _write(lock, record)
    return record


# ------------------------------------------------------------------ the end

@dataclass
class Result:
    """How an update ended. *state*: ``none`` (there was none), ``old`` (the old
    version is back), ``new`` (it was finished) or ``failed``. *errors* names every
    file that is not as it should be; *notes* are lines for the user."""
    state: str
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def finish(lock: Lock) -> Result:
    """End the update of *lock*'s install as done: the new version stays, files the
    user had edited go to ``kept``, everything else of the update is deleted. When
    a plan cannot be read or an edited file cannot be moved nothing is deleted and a
    note names the folder. Only from phase STARTING, or DONE when this was
    interrupted. Raises ValueError for another phase, :class:`Unreadable` for the
    record, :class:`Unsafe` or OSError."""
    record = read(lock.root)
    if record is None or record.phase not in (STARTING, DONE):
        raise ValueError("there is no started update to finish")
    disk.sync_disk()
    _set_phase(lock, record, DONE)
    result = Result("new")
    try:
        for plan in mover.read_plans(lock.root, record.id):
            if plan.step == mover.FILES:
                _keep_edited(lock.root, record.id, plan, result)
    except (Unreadable, Unsafe, disk.Held, OSError) as e:
        disk.unlink(record_path(lock.root))
        result.notes.append("What the update replaced stays in "
                            f"{mover.update_folder(lock.root) / record.id}: {e}")
        return result
    _close(lock, record, result)
    return result


def restore(lock: Lock, *, wait_s: Optional[float] = None) -> Result:
    """End the unfinished update of *lock*'s install, if there is one: put the old
    version back, every step in reverse, and check every file; a finished one is
    only cleaned up. Safe to repeat and to interrupt. While a file cannot be put
    back the record stays and the result is ``failed``.

    Raises :class:`Unreadable` when the record or a plan cannot be read (nothing is
    changed then), :class:`Unsafe` for a link in the place of one of the update's
    folders, and OSError."""
    if not lock.held:
        raise ValueError("the lock of the install is not held")
    try:
        record = read(lock.root)
    except Unreadable:
        record = _only_unfinished(lock.root)
        if record is None:
            raise
    if record is None:
        cleanup(lock)
        return Result("none")
    if record.phase == DONE:
        return finish(lock)
    result = Result("old")
    if record.phase != ENDING:
        plans = mover.read_plans(lock.root, record.id)
        _set_phase(lock, record, RESTORING)
        for plan in reversed(plans):
            result.errors += _back(lock, record, plan, wait_s)
    if result.errors:
        result.state = "failed"
        return result
    disk.sync_disk()
    _close(lock, record, result)
    if record.phase != ENDING:
        result.notes.insert(0, f"The update to {record.new} was undone. localm {record.old} is "
                               "in place." if record.new else "The unfinished update was undone.")
    return result


def _back(lock: Lock, record: Record, plan: mover.Plan, wait_s: Optional[float]) -> list:
    if plan.step == mover.PACKAGES:
        return packages.back(lock.root, record.id, plan, wait_s=wait_s)
    return mover.back(lock.root, record.id, plan, wait_s=wait_s)


def _set_phase(lock: Lock, record: Record, phase: str) -> None:
    """Write *phase* into the record, with the space of the reserve file when the
    disk is full."""
    if record.phase == phase:
        return
    record.phase = phase
    try:
        _write(lock, record)
    except OSError:
        reserve = mover.work_folder(lock.root, record.id) / _RESERVE
        if disk.lstat(reserve) is None:
            raise
        disk.unlink(reserve)
        _write(lock, record)


def _only_unfinished(root) -> Optional[Record]:
    """The record a damaged ``job.json`` stands for: the one update of *root* that
    has plans and did not end. None when there is none or more than one."""
    found = []
    for update_id in _ids(root):
        work = mover.work_folder(root, update_id)
        if disk.lstat(work / _FINISHED) is None and mover.has_plans(root, update_id):
            found.append(update_id)
    if len(found) != 1:
        return None
    return Record(found[0], MOVING, "", "")


def _ids(root) -> list:
    folder = mover.update_folder(root)
    if disk.lstat(folder) is None:
        return []
    return sorted(n for n in os.listdir(folder) if mover.is_id(n))


def _keep_edited(root, update_id: str, plan: mover.Plan, result: Result) -> None:
    a = mover.area(root, update_id, plan.step)
    for op in plan.ops:
        if not op.edited:
            continue
        old, st = a.beside(a.old, op.path)
        if st is not None:
            kept, kept_st = a.beside(a.kept, op.path)
            if kept_st is None:
                disk.move(old, kept, disk.Wait())
    if disk.lstat(a.kept) is not None:
        result.notes.append(f"Files you had changed are in {a.kept}.")


def _close(lock: Lock, record: Record, result: Result) -> None:
    """Mark the update as ended, remove its record, then delete its folders. When it
    cannot be marked its folders stay and a note names them."""
    work = mover.work_folder(lock.root, record.id)
    if disk.lstat(work) is not None:
        try:
            disk.write_json(work / _FINISHED, {})
        except OSError as e:
            result.notes.append(f"What the update left stays in {work}: {e}")
    disk.unlink(record_path(lock.root))
    cleanup(lock)
    kept = mover.update_folder(lock.root) / "kept" / record.id
    if disk.lstat(kept) is not None and not result.notes:
        result.notes.append(f"Files that were neither of the two versions are in {kept}.")


def cleanup(lock: Lock) -> list:
    """Delete what ended updates of *lock*'s install left. A file a program still
    has loaded stays for the next call. Returns the folders that are left."""
    if not lock.held:
        raise ValueError("the lock of the install is not held")
    left = []
    with contextlib.suppress(OSError):   # a write that never took effect
        disk.unlink(record_path(lock.root).with_name("job.json.tmp"))
    for update_id in _ids(lock.root):
        try:
            if not _delete(lock.root, update_id):
                left.append(str(mover.update_folder(lock.root) / update_id))
        except (OSError, Unsafe) as e:
            left.append(f"{update_id}: {e}")
    with contextlib.suppress(OSError):   # not empty, or never made
        os.rmdir(disk.real_folder(lock.root / ".venv") / mover.VENV_FOLDER)
    return left


def _delete(root, update_id: str) -> bool:
    """Delete the folders of *update_id* when it ended, its end mark last. Returns
    False when a part is left."""
    work = mover.work_folder(root, update_id)
    if disk.lstat(work / _FINISHED) is None:
        with contextlib.suppress(OSError):   # only an empty one goes
            os.rmdir(work)
        return True
    gone = disk.remove_tree(mover.area(root, update_id, mover.PACKAGES).old.parent)
    for name in sorted(os.listdir(work)):
        path = work / name
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) or disk.is_link(st):
            gone = disk.remove_tree(path) and gone
        elif name != _FINISHED:
            disk.unlink(path)
    return gone and disk.remove_tree(work)

