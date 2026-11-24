# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record of an update, its lock and its phases, and the end of an update:
finished (:func:`finish`) or undone (:func:`restore`).

The record is on disk before anything changes and is removed last. While it is
there and its lock is free, nobody is updating: :func:`restore` then puts the old
version back, whatever the phase, or ends an update that reached DONE.

The update's own folder holds the same facts (``state.json``), written before the
record: it decides, a damaged record is rebuilt from it, and an update that reached
DONE there is never undone. It says that the update is over before the record is
removed. Nothing is deleted of an update whose files are not as it says.
"""

from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import disk, mover, packages
from .disk import Blocked, Held, Unsafe
from .mover import StepFailed, Unreadable

__all__ = ["Blocked", "Busy", "Held", "Lock", "Record", "Result", "StepFailed", "Unreadable", "Unsafe",
           "advance", "cleanup", "discard", "finish", "locked", "read", "restore", "start"]

FORMAT = "localm-update-job/1"
MAX_RECORD_BYTES = 64 * 1024

ENDING, MOVING, STEPS, STARTING, DONE, RESTORING = (
    "ending", "moving", "steps", "starting", "done", "restoring")
#: Every phase, in order; an update that fails goes to RESTORING from any but DONE.
PHASES = (ENDING, MOVING, STEPS, STARTING, DONE, RESTORING)

# What ``state.json`` says of an update that is over: undone, or done with files of
# the user that may still have to be handed over.
_ENDED, _LEFT = "ended", "left"
_STATE = "state.json"
#: Seconds :class:`Lock` waits, longer than another process's :func:`locked` has it.
LOCK_WAIT_S = 2.0
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
        try:
            disk.lock_file(fd, disk.Wait(self.wait_s))
        except (PermissionError, BlockingIOError) as e:
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
    Raises :class:`Unreadable` for a damaged one, :class:`Blocked` for one that
    cannot be read now, :class:`Unsafe` for a link."""
    return _parse(record_path(root), PHASES)


def _parse(path: Path, phases) -> Optional[Record]:
    try:
        raw = disk.read_json(path, MAX_RECORD_BYTES)
    except ValueError as e:
        raise Unreadable(str(e)) from None
    if raw is None:
        return None
    if (not isinstance(raw, dict) or set(raw) != {"format", "id", "phase", "old", "new"}
            or raw["format"] != FORMAT or raw["phase"] not in phases
            or not all(isinstance(raw[k], str) and len(raw[k]) <= 64 for k in ("old", "new"))):
        raise Unreadable(f"{path.name} is not the record of an update")
    return Record(mover.check_id(raw["id"]), raw["phase"], raw["old"], raw["new"])


def _state(root, update_id: str) -> Optional[Record]:
    """What the folder of the update *update_id* says of it, None when it says
    nothing. A damaged one without plans beside it says nothing. Raises like
    :func:`read`."""
    try:
        record = _parse(mover.work_folder(root, update_id) / _STATE, (*PHASES, _ENDED, _LEFT))
        if record is not None and record.id != update_id:
            raise Unreadable(f"{_STATE} of {update_id} is that of another update")
    except Unreadable:
        if mover.has_plans(root, update_id):
            raise
        return None
    return record


def _write_state(lock: Lock, record: Record) -> None:
    disk.write_json(mover.work_folder(lock.root, record.id) / _STATE,
                    {"format": FORMAT, **asdict(record)})


def _held(lock: Lock) -> None:
    if not lock.held:
        raise ValueError("the lock of the install is not held")


def _write(lock: Lock, record: Record) -> None:
    _held(lock)
    _write_state(lock, record)
    disk.write_json(record_path(lock.root), {"format": FORMAT, **asdict(record)})


def _set_phase(lock: Lock, record: Record, phase: str) -> None:
    """Write *phase* into the record, with the space of the reserve file when the
    disk is full."""
    record.phase = phase
    try:
        _write(lock, record)
    except OSError:
        if not _free_reserve(lock.root, record.id):
            raise
        _write(lock, record)


def _free_reserve(root, update_id: str) -> bool:
    reserve = mover.work_folder(root, update_id) / _RESERVE
    if disk.lstat(reserve) is None:
        return False
    disk.unlink(reserve)
    return True


def start(lock: Lock, update_id: str, *, old: str, new: str) -> Record:
    """Begin the update *update_id*, whose folder already holds ``new``: write its
    record in phase ENDING, and delete what others left (:func:`cleanup`). Raises :class:`Busy` when another update is unfinished,
    :class:`Unreadable`, :class:`Blocked`, :class:`Unsafe`, :class:`Held` or OSError."""
    going = _current(lock.root)
    if read(lock.root) is not None or (going is not None and going.phase != ENDING):
        raise Busy("an earlier update of this install is not finished")
    cleanup(lock, keep=update_id)
    work = mover.work_folder(lock.root, update_id)
    (work / "plans").mkdir(parents=True, exist_ok=True)
    with open(work / _RESERVE, "wb") as f:
        f.write(b"\0" * _RESERVE_BYTES)
        f.flush()
        os.fsync(f.fileno())
    record = Record(mover.check_id(update_id), ENDING, old, new)
    _write(lock, record)
    return record


def advance(lock: Lock, phase: str) -> Record:
    """Write *phase*, the next one of :data:`PHASES` or a repeat of the current one,
    into the record before that phase begins. Raises ValueError for any other phase
    or without a record, and what :func:`start` raises."""
    record = _current(lock.root)
    if record is None:
        raise ValueError("there is no update to advance")
    steps = PHASES.index(phase) - PHASES.index(record.phase)
    if phase in (DONE, RESTORING) or steps not in (0, 1):
        raise ValueError(f"an update cannot go from {record.phase} to {phase}")
    disk.sync_disk()
    _set_phase(lock, record, phase)
    return record


# --------------------------------------------------------- which update it is

def _ids(root) -> list:
    folder = mover.update_folder(root)
    return [] if disk.lstat(folder) is None else _listed(folder)


def _listed(folder: Path) -> list:
    return sorted(n for n in os.listdir(folder) if mover.is_id(n))


def _current(root) -> Optional[Record]:
    """The unfinished update of the install at *root*, None when there is none. A
    damaged or missing record is rebuilt from the folder of the one update that did
    not end. Raises :class:`Unreadable` when that cannot be told, :class:`Blocked`
    when a file cannot be read now."""
    try:
        record = read(root)
    except Unreadable:
        return _rebuilt(root, damaged=True)
    if record is None:
        return _rebuilt(root, damaged=False)
    state = _state(root, record.id)
    if state is not None:
        record.phase = state.phase
    if record.phase == DONE and disk.lstat(mover.work_folder(root, record.id)) is None:
        record.phase = _ENDED
    return record


def _rebuilt(root, *, damaged: bool) -> Optional[Record]:
    going: list = []
    unchanged: list = []
    for update_id in _ids(root):
        state = _state(root, update_id)
        if state is None and mover.has_plans(root, update_id):
            raise Unreadable(f"the update {update_id} has plans and no {_STATE}")
        if state is None or state.phase in (_ENDED, _LEFT):
            continue
        (unchanged if state.phase == ENDING else going).append(state)
    if len(going) == 1:
        return going[0]
    if going or (damaged and not unchanged):
        raise Unreadable("the record of the update is damaged, and which update it was "
                         "cannot be told")
    return unchanged[0] if unchanged else None


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
    user had edited go to the area's ``kept`` folder, everything else of the update
    is deleted. From DONE on it is never undone; interrupted, it is done again. A
    file of the user that cannot be moved stays with the update's folder, which a
    note names and a later :func:`cleanup` tries again.

    Only from phase STARTING, or DONE when this was interrupted. Raises ValueError
    for another phase, and what :func:`start` raises."""
    _held(lock)
    record = _current(lock.root)
    if record is None or record.phase not in (STARTING, DONE):
        raise ValueError("there is no started update to finish")
    return _finish(lock, record)


def _finish(lock: Lock, record: Record) -> Result:
    result = Result("new")
    _check_fits(lock.root, record.id, DONE)
    if record.phase != DONE:
        disk.sync_disk()
        _set_phase(lock, record, DONE)
    _hand_over(lock.root, record.id, result)
    _close(lock, record, _LEFT, result)
    return result


def _check_fits(root, update_id: str, phase: str) -> None:
    """Raise :class:`Unreadable` when the files of the plans say that the update
    *update_id* is not in *phase*: DONE and LEFT mean that every step was made or is
    undone, any other that nothing of one is left to put back."""
    if not mover.has_plans(root, update_id):
        return
    for plan in mover.read_plans(root, update_id):
        rel = mover.unfinished(root, update_id, plan, made=phase in (DONE, _LEFT))
        if rel is not None:
            raise Unreadable(f"{_STATE} says that the update is {phase}, and {rel} is not as "
                             "that means")


def restore(lock: Lock, *, wait_s: Optional[float] = None) -> Result:
    """End the unfinished update of *lock*'s install, if there is one: put the old
    version back, every step in reverse, and check every file; one that reached DONE
    is finished instead. Safe to repeat and to interrupt. While a file cannot be put
    back the record stays and the result is ``failed``.

    Raises :class:`Unreadable` when the record or a plan is damaged and
    :class:`Blocked` when it cannot be read now (nothing is changed then),
    :class:`Unsafe` for a link in the place of one of the update's folders,
    :class:`Held` and OSError."""
    _held(lock)
    record = _current(lock.root)
    if record is None:
        cleanup(lock)
        return Result("none")
    if record.phase == DONE:
        return _finish(lock, record)
    if record.phase in (_ENDED, _LEFT):
        result = Result("none")
        _check_fits(lock.root, record.id, record.phase)
        _close(lock, record, record.phase, result)
        return result
    result = Result("old")
    if record.phase == ENDING:
        _check_fits(lock.root, record.id, ENDING)
    else:
        plans = mover.read_plans(lock.root, record.id)
        try:
            _set_phase(lock, record, RESTORING)
        except OSError as e:
            result.notes.append(f"The record of the update could not be written: {e}")
        _free_reserve(lock.root, record.id)
        for plan in reversed(plans):
            out = _back(lock, record, plan, wait_s)
            result.errors += out.errors
            result.notes += out.notes
        if result.errors:
            result.state = "failed"
            return result
        result.notes.insert(0, f"The update to {record.new} was undone. localm {record.old} is "
                               "in place.")
    _close(lock, record, _ENDED, result)
    return result


def _back(lock: Lock, record: Record, plan: mover.Plan,
          wait_s: Optional[float]) -> mover.Outcome:
    step = packages if plan.step == mover.PACKAGES else mover
    return step.back(lock.root, record.id, plan, wait_s=wait_s)


def _hand_over(root, update_id: str, result: Result) -> bool:
    """Move the user's own files an update replaced to the ``kept`` folders, on disk
    when this returns. Safe to repeat. Returns False when one could not be moved,
    which is noted."""
    try:
        for plan in mover.read_plans(root, update_id) if mover.has_plans(root, update_id) else ():
            a = mover.area(root, update_id, plan.step)
            moved = 0
            for op in plan.ops:
                old, st = a.beside(a.old, op.path) if op.edited else (a.old, None)
                if st is not None:
                    disk.move(old, mover.free(a, op.path), disk.Wait(), through=True)
                    moved += 1
            if moved:
                result.notes.append(f"{moved} file(s) that were your own are in {a.kept}.")
    except (Unreadable, Unsafe, Held, Blocked, OSError) as e:
        result.notes.append("What the update replaced stays in "
                            f"{mover.update_folder(root) / update_id}: {e}")
        return False
    disk.sync_disk()
    return True


def _close(lock: Lock, record: Record, over: str, result: Result) -> None:
    """Say in its folder that the update is *over*, remove its record, then delete
    its folders. When that cannot be written the record stays and a note says so."""
    disk.sync_disk()
    try:
        if disk.lstat(mover.work_folder(lock.root, record.id)) is not None:
            _set_state(lock, record, over)
        disk.remove_through(record_path(lock.root))
    except (OSError, Held) as e:
        result.notes.append(f"The record of the update stays for now: {e}")
        return
    cleanup(lock)


def _set_state(lock: Lock, record: Record, over: str) -> None:
    ended = Record(record.id, over, record.old, record.new)
    try:
        _write_state(lock, ended)
    except OSError:
        if not _free_reserve(lock.root, record.id):
            raise
        _write_state(lock, ended)


def discard(lock: Lock, update_id: str) -> bool:
    """Delete an update that was prepared and never started (:func:`start`). Returns
    whether it is gone. Raises ValueError for one that was started, :class:`Unsafe`
    or OSError."""
    _held(lock)
    work = mover.work_folder(lock.root, update_id)
    if disk.lstat(work / _STATE) is not None:
        raise ValueError("the update was started")
    return disk.remove_tree(mover.venv_folder(lock.root, update_id)) and disk.remove_tree(work)


def cleanup(lock: Lock, *, keep: Optional[str] = None) -> list:
    """Delete what ended updates of *lock*'s install left, nothing of an update that
    did not end, and the staged packages of those that were never started, except the
    update *keep*. A file a program still has loaded stays for the next call. Returns
    the folders that are left."""
    _held(lock)
    left = []
    for name in ("job.json.tmp", "job.json.gone"):
        with contextlib.suppress(OSError, Held):   # a write that never took effect
            disk.unlink(record_path(lock.root).with_name(name))
    venv = disk.real_folder(lock.root / ".venv") / mover.VENV_FOLDER
    ids = set(_ids(lock.root))
    with contextlib.suppress(OSError):   # nothing is staged
        ids.update(_listed(venv))
    for update_id in sorted(ids - {keep}):
        try:
            if not _delete(lock, update_id):
                left.append(str(mover.update_folder(lock.root) / update_id))
        except (OSError, Unsafe, Unreadable, Blocked, Held) as e:
            left.append(f"{update_id}: {e}")
    with contextlib.suppress(OSError):   # not empty, or never made
        os.rmdir(venv)
    return left


def _delete(lock: Lock, update_id: str) -> bool:
    """Delete the folders of *update_id* when it ended, its plans first and what says
    that it ended last. Returns False when a part is left."""
    root = lock.root
    work = mover.work_folder(root, update_id)
    state = _state(root, update_id)
    if state is None:
        mine = mover.venv_folder(root, update_id)
        if mover.has_plans(root, update_id) or disk.lstat(mine / "old") is not None:
            return False
        with contextlib.suppress(OSError):   # only an empty one goes
            os.rmdir(work)
        return disk.remove_tree(mine)
    if state.phase not in (_ENDED, _LEFT, ENDING) or (
            state.phase == ENDING and not _is_forgotten(root, update_id)):
        return True
    _check_fits(root, update_id, state.phase)
    if state.phase == _LEFT and not _hand_over(root, update_id, Result("new")):
        return False
    if state.phase == ENDING:
        state.phase = _ENDED
        _write_state(lock, state)
    if not disk.remove_tree(work / "plans"):
        return False
    gone = disk.remove_tree(mover.venv_folder(root, update_id))
    for name in sorted(os.listdir(work)):
        path = work / name
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) or disk.is_link(st):
            gone = disk.remove_tree(path) and gone
        elif name != _STATE:
            disk.unlink(path)
    return gone and disk.remove_tree(work)


def _is_forgotten(root, update_id: str) -> bool:
    """Whether the update *update_id*, which changed nothing, has no record."""
    try:
        record = read(root)
    except Unreadable:
        return False
    return record is None or record.id != update_id
