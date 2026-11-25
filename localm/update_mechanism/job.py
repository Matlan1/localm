# SPDX-License-Identifier: AGPL-3.0-or-later
"""The record of an update, its lock and its phases, and the end of an update:
finished (:func:`finish`) or undone (:func:`restore`).

The record is on disk before anything changes and is removed last. While it is
there and its lock is free, nobody is updating: :func:`restore` then puts the old
version back, whatever the phase, or ends an update that reached DONE.

The update's own folder says how far it came, by the files that are there and not by
what they hold: with ``done`` it is done and never undone, with ``ended`` it was
undone, with neither it is put back by its plans. ``state.json`` there repeats the
record, for the versions, and the phase of an update without plans, when the record is
damaged.
"""

from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

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

#: The files that say that an update is over: :data:`DONE`, or undone.
_ENDED = "ended"
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


def _over(root, update_id: str) -> Optional[str]:
    """:data:`DONE` or ``ended`` when the update *update_id* is, None before."""
    work = mover.work_folder(root, update_id)
    return next((name for name in (DONE, _ENDED) if disk.lstat(work / name) is not None), None)


def _held(lock: Lock) -> None:
    if not lock.held:
        raise ValueError("the lock of the install is not held")


def _write(lock: Lock, record: Record) -> None:
    _held(lock)
    raw = {"format": FORMAT, **asdict(record)}
    disk.write_json(mover.work_folder(lock.root, record.id) / _STATE, raw)
    disk.write_json(record_path(lock.root), raw)


def _set_phase(lock: Lock, record: Record, phase: str) -> None:
    record.phase = phase
    _with_reserve(lock, record, lambda: _write(lock, record))


def _set_over(lock: Lock, record: Record, over: str) -> None:
    """Say in its folder, on disk when this returns, that the update is *over*."""
    path = mover.work_folder(lock.root, record.id) / over
    _with_reserve(lock, record, lambda: disk.write_json(path, {}))


def _with_reserve(lock: Lock, record: Record, write: Callable[[], None]) -> None:
    """Call *write*, again with the space of the reserve file when the disk is full."""
    try:
        write()
    except OSError:
        if not _free_reserve(lock.root, record.id):
            raise
        write()


def _free_reserve(root, update_id: str) -> bool:
    reserve = mover.work_folder(root, update_id) / _RESERVE
    if disk.lstat(reserve) is None:
        return False
    disk.unlink(reserve)
    return True


def start(lock: Lock, update_id: str, *, old: str, new: str) -> Record:
    """Begin the update *update_id*, whose folder already holds ``new``: write its
    record in phase ENDING, and delete what others left (:func:`cleanup`). Raises
    :class:`Busy` when another update is unfinished, :class:`Unreadable`,
    :class:`Blocked`, :class:`Unsafe`, :class:`Held` or OSError."""
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
    """The unfinished update of the install at *root*, None when there is none. Its
    phase is DONE or ``ended`` exactly when its folder says so. Raises
    :class:`Unreadable` when it cannot be told, :class:`Blocked` when a file cannot be
    read now."""
    damaged = None
    try:
        record = read(root)
    except Unreadable as e:
        record, damaged = None, e
    if record is None:
        record = _from_folders(root, damaged)
    if record is not None:
        over = _over(root, record.id)
        gone = disk.lstat(mover.work_folder(root, record.id)) is None
        if over is not None or record.phase == DONE:
            record.phase = over or (_ENDED if gone else RESTORING)
    return record


def _from_folders(root, damaged: Optional[Unreadable]) -> Optional[Record]:
    """The update that is not over, from the folders of the updates; with a *damaged*
    record also one that is over or has changed nothing."""
    going: list = []
    others: list = []
    for update_id in _ids(root):
        over = _over(root, update_id) is not None
        if not over and mover.has_plans(root, update_id):
            going.append(update_id)
        elif damaged or (
                not over and disk.lstat(mover.work_folder(root, update_id) / _STATE) is not None):
            others.append(update_id)
    if len(going) > 1 or (damaged and not going + others):
        raise mover.lost_track(root, going, str(damaged or "more than one update is unfinished"))
    for update_id in (going + others)[:1]:
        with contextlib.suppress(Unreadable):   # it only tells the versions and the phase
            said = _parse(mover.work_folder(root, update_id) / _STATE, PHASES)
            if said is not None and said.id == update_id:
                return said
        return Record(update_id, RESTORING if going else ENDING, "", "")
    return None


# ------------------------------------------------------------------ the end

@dataclass
class Result:
    """How an update ended. *state*: ``none`` (there was none), ``old`` (the old
    version is back), ``new`` (it was finished) or ``failed``. *errors* names every
    file and package that is not as it should be, *packages* the packages among them
    (:class:`.mover.Outcome`): installing exactly those again and calling again puts
    the old version back. *notes* are lines for the user; *held* has each
    :class:`.disk.Held` behind an error or note. Calling again goes on from there."""
    state: str
    errors: list = field(default_factory=list)
    packages: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    held: list = field(default_factory=list)

    def note(self, text: str, e: BaseException) -> None:
        """Add the note *text* on the error *e*."""
        self.notes.append(f"{text}: {e}")
        if isinstance(e, Held):
            self.held.append(e)


def finish(lock: Lock, *, ask: Optional[disk.Ask] = None) -> Result:
    """End the update of *lock*'s install as done: the new version stays, files the
    user had edited go to the area's ``kept`` folder, everything else of the update
    is deleted. From DONE on it is never undone; interrupted, it is done again. A
    file of the user that cannot be moved stays with the update's folder, which a
    note names and a later :func:`cleanup` tries again; *ask* is called about it
    first (:class:`.disk.Wait`).

    Only from phase STARTING, or DONE when this was interrupted. Raises ValueError
    for another phase, :class:`Unreadable` when a step is not made, and what
    :func:`start` raises."""
    _held(lock)
    record = _current(lock.root)
    if record is None or record.phase not in (STARTING, DONE):
        raise ValueError("there is no started update to finish")
    if record.phase != DONE:
        for plan in mover.read_plans(lock.root, record.id):
            rel = mover.unfinished(lock.root, record.id, plan)
            if rel is not None:
                raise Unreadable(f"the update is not finished: {rel} is not as it leaves it")
        disk.sync_disk()
        _set_over(lock, record, DONE)
        _set_phase(lock, record, DONE)
    return _finish(lock, record, ask)


def _finish(lock: Lock, record: Record, ask: Optional[disk.Ask]) -> Result:
    result = Result("new")
    _hand_over(lock.root, record.id, result, ask)
    _close(lock, record, DONE, result)
    return result


def restore(lock: Lock, *, wait_s: Optional[float] = None,
            ask: Optional[disk.Ask] = None) -> Result:
    """End the unfinished update of *lock*'s install, if there is one: put the old
    version back, every step in reverse, and check every file; one that reached DONE
    is finished instead. Safe to repeat and to interrupt. *ask* is called about a
    file that is still held after *wait_s* (:class:`.disk.Wait`). While a file
    or package cannot be put back the record stays and the result is ``failed``.

    Raises :class:`Unreadable`, which says what to do, when a plan is damaged or the
    update cannot be told, :class:`Blocked` when the record or a plan cannot be read
    now (nothing is changed then), :class:`Unsafe` for a link in the place of one of
    the update's folders, :class:`Held` and OSError."""
    _held(lock)
    record = _current(lock.root)
    if record is None:
        cleanup(lock)
        return Result("none")
    if record.phase == DONE:
        return _finish(lock, record, ask)
    result = Result("none" if record.phase == _ENDED else "old")
    if record.phase != _ENDED and (
            record.phase != ENDING or mover.has_plans(lock.root, record.id)):
        plans = mover.read_plans(lock.root, record.id)
        try:
            _set_phase(lock, record, RESTORING)
        except OSError as e:
            result.note("The record of the update could not be written", e)
        _free_reserve(lock.root, record.id)
        for plan in reversed(plans):
            earlier = {d for p in plans if p.number < plan.number and p.step == plan.step
                       for d in p.dists}
            step = (partial(packages.back, earlier=earlier) if plan.step == mover.PACKAGES
                    else mover.back)
            out = step(lock.root, record.id, plan, wait_s=wait_s, ask=ask)
            result.errors += out.errors
            result.packages += out.packages
            result.notes += out.notes
            result.held += out.held
        if result.errors:
            result.state = "failed"
            return result
        result.notes.insert(0, f"The update to {record.new} was undone. localm {record.old} is "
                               "in place." if record.new else "The update was undone.")
    _close(lock, record, _ENDED, result)
    return result


def _hand_over(root, update_id: str, result: Result, ask: Optional[disk.Ask] = None) -> bool:
    """Move the user's own files an update replaced to the ``kept`` folders, on disk
    when this returns. Safe to repeat. Returns False when one could not be moved,
    which is noted."""
    wait = disk.Wait(ask=ask)
    try:
        for plan in mover.read_plans(root, update_id) if mover.has_plans(root, update_id) else ():
            a = mover.area(root, update_id, plan.step, plan.number)
            moved = 0
            for op in plan.ops:
                old, st = a.beside(a.old, op.path) if op.edited else (a.old, None)
                if st is not None:
                    disk.move(old, mover.free(a, op.path), wait, through=True)
                    moved += 1
            if moved:
                result.notes.append(f"{moved} file(s) that were your own are in {a.kept}.")
    except (Unreadable, Unsafe, Held, OSError) as e:
        result.note(f"What the update replaced stays in {mover.update_folder(root) / update_id}", e)
        return False
    disk.sync_disk()
    return True


def _close(lock: Lock, record: Record, over: str, result: Result) -> None:
    """Say in its folder that the update is *over*, remove its record, then delete
    its folders. When that cannot be written the record stays and a note says so."""
    disk.sync_disk()
    try:
        work = mover.work_folder(lock.root, record.id)
        if disk.lstat(work) is not None and disk.lstat(work / over) is None:
            _set_over(lock, record, over)
        disk.remove_through(record_path(lock.root))
    except (OSError, Held) as e:
        result.note("The record of the update stays for now", e)
        return
    cleanup(lock)


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
        except (OSError, Unsafe, Unreadable, Held) as e:
            left.append(f"{update_id}: {e}")
    with contextlib.suppress(OSError):   # not empty, or never made
        os.rmdir(venv)
    return left


def _delete(lock: Lock, update_id: str) -> bool:
    """Delete the folders of *update_id* when it is over, its plans first and what says
    that it is over last. Returns False when a part is left."""
    root = lock.root
    work = mover.work_folder(root, update_id)
    over = _over(root, update_id)
    mine = mover.venv_folder(root, update_id)
    if over is None:
        if disk.lstat(work / _STATE) is None:
            if mover.has_plans(root, update_id) or disk.lstat(mine / "old") is not None:
                return False
            with contextlib.suppress(OSError):   # only an empty one goes
                os.rmdir(work)
            return disk.remove_tree(mine)
        record = read(root)
        if mover.has_plans(root, update_id) or (record is not None and record.id == update_id):
            return True
    if over == DONE and not _hand_over(root, update_id, Result("new")):
        return False
    if disk.lstat(work / "plans") is not None:
        os.rename(work / "plans", work / "gone")
    gone = disk.remove_tree(mine)
    for name in sorted(os.listdir(work)):
        path = work / name
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) or disk.is_link(st):
            gone = disk.remove_tree(path) and gone
        elif name not in (_STATE, over):
            disk.unlink(path)
    if gone:
        disk.unlink(work / _STATE)
    return gone and disk.remove_tree(work)
