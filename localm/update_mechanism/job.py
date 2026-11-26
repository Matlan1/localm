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

from . import disk, mover, packages, shared, steps_data
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
ENDED = "ended"
STATE = "state.json"
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
        #: The locks of the parts of this update, by their folder.
        self.parts: dict = {}
        #: The update that is about to start; :func:`cleanup` leaves it alone.
        self.keep: Optional[str] = None

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
        for part in self.parts.values():
            part.__exit__()
        self.parts.clear()
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


def over_of(root, update_id: str) -> Optional[str]:
    """:data:`DONE` or :data:`ENDED` when the update *update_id* is, None before."""
    work = mover.work_folder(root, update_id)
    return next((name for name in (DONE, ENDED) if disk.lstat(work / name) is not None), None)


def _held(lock: Lock) -> None:
    if not lock.held:
        raise ValueError("the lock of the install is not held")


def _write(lock: Lock, record: Record) -> None:
    _held(lock)
    raw = {"format": FORMAT, **asdict(record)}
    disk.write_json(mover.work_folder(lock.root, record.id) / STATE, raw)
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
    going = current(lock.root)
    if read(lock.root) is not None or (going is not None and going.phase != ENDING):
        raise Busy("an earlier update of this install is not finished")
    lock.keep = update_id
    cleanup(lock)
    work = mover.work_folder(lock.root, update_id)
    (work / "plans").mkdir(parents=True, exist_ok=True)
    with open(work / _RESERVE, "wb") as f:
        f.write(b"\0" * _RESERVE_BYTES)
        f.flush()
        os.fsync(f.fileno())
    record = Record(mover.check_id(update_id), ENDING, old, new)
    _write(lock, record)
    lock.keep = None
    return record


def advance(lock: Lock, phase: str) -> Record:
    """Write *phase*, the next one of :data:`PHASES` or a repeat of the current one,
    into the record before that phase begins. Raises ValueError for any other phase
    or without a record, and what :func:`start` raises."""
    record = current(lock.root)
    if record is None:
        raise ValueError("there is no update to advance")
    steps = PHASES.index(phase) - PHASES.index(record.phase)
    if phase in (DONE, RESTORING) or steps not in (0, 1):
        raise ValueError(f"an update cannot go from {record.phase} to {phase}")
    disk.sync_disk()
    _set_phase(lock, record, phase)
    return record


# --------------------------------------------------------- which update it is

def ids(root) -> list:
    """The ids of the updates that have a folder in the install at *root*."""
    folder = mover.update_folder(root)
    return [] if disk.lstat(folder) is None else _listed(folder)


def _listed(folder: Path) -> list:
    return sorted(n for n in os.listdir(folder) if mover.is_id(n))


def current(root) -> Optional[Record]:
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
        over = over_of(root, record.id)
        gone = disk.lstat(mover.work_folder(root, record.id)) is None
        if over is not None or record.phase == DONE:
            record.phase = over or (ENDED if gone else RESTORING)
    return record


def _from_folders(root, damaged: Optional[Unreadable]) -> Optional[Record]:
    """The update that is not over, from the folders of the updates; with a *damaged*
    record also one that is over or has changed nothing."""
    going: list = []
    others: list = []
    for update_id in ids(root):
        over = over_of(root, update_id) is not None
        if not over and mover.has_plans(root, update_id):
            going.append(update_id)
        elif damaged or (
                not over and disk.lstat(mover.work_folder(root, update_id) / STATE) is not None):
            others.append(update_id)
    if len(going) > 1 or (damaged and not going + others):
        raise mover.lost_track(root, going, str(damaged or "more than one update is unfinished"))
    for update_id in (going + others)[:1]:
        with contextlib.suppress(Unreadable):   # it only tells the versions and the phase
            said = _parse(mover.work_folder(root, update_id) / STATE, PHASES)
            if said is not None and said.id == update_id:
                return said
        return Record(update_id, RESTORING if going else ENDING, "", "")
    return None


# ------------------------------------------------------------------- parts
#
# A part is a component in a folder of its own (a runtime, the managed ComfyUI) with
# a record, a lock and plans of its own there, under the id of the update it belongs
# to. The update's folder names the parts; they end when it ends.

PARTS = "parts.json"
OWNER = "owner.json"
#: In the folder of an update that is prepared: the lock of the process that prepares
#: it, and the name of a folder of it beside its own.
BUSY, BESIDE = "busy", "beside.json"
#: What setup-llama adds to the name of the folder it fills, for the one it unpacks in.
INCOMING = ".incoming"


def part_path(root, folder) -> str:
    """*folder* as an update of *root* names it: relative to *root* when inside it."""
    whole, inner = disk.real_folder(root), disk.real_folder(folder)
    if whole in inner.parents:
        return inner.relative_to(whole).as_posix()
    return disk.short_path(inner)


def _json(path: Path):
    try:
        return disk.read_json(path, MAX_RECORD_BYTES)
    except ValueError as e:
        raise Unreadable(str(e)) from None


def part_folders(root, update_id: str) -> list:
    """The folders of the parts of the update *update_id*, in their order. Raises
    :class:`Unreadable` or :class:`Blocked`."""
    raw = _json(mover.work_folder(root, update_id) / PARTS)
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(f, str) and f for f in raw):
        raise Unreadable(f"{PARTS} does not name the parts of an update")
    return [Path(root) / f for f in raw]


def _lock_part(lock: Lock, folder: Path) -> Lock:
    key = disk.real_folder(folder)
    part = lock.parts.get(key)
    if part is None:
        part = Lock(folder)
        part.__enter__()
        lock.parts[key] = part
    return part


def own_part(root, update_id: str, folder) -> Path:
    """The folder of the update *update_id* in the component folder *folder*, marked
    as belonging to the install at *root*, by a relative path when *folder* is inside
    it. Raises OSError, :class:`Held` or
    :class:`Unsafe`."""
    mover.update_folder(folder, create=True)
    work = mover.work_folder(folder, update_id)
    (work / "plans").mkdir(parents=True, exist_ok=True)
    whole, inner = disk.real_folder(root), disk.real_folder(folder)
    named = ("/".join([".."] * (len(inner.parts) - len(whole.parts)))
             if whole in inner.parents else disk.short_path(whole))
    disk.write_json(work / OWNER, {"root": named})
    return work


def owner(folder, update_id: str) -> Optional[Path]:
    """The install whose update *update_id* the component folder *folder* is a part
    of, None for a component changed on its own. Raises :class:`Unreadable` or
    :class:`Blocked`."""
    raw = _json(mover.work_folder(folder, update_id) / OWNER)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("root"), str):
        raise Unreadable(f"{OWNER} does not name an install")
    return disk.real_folder(disk.real_folder(folder) / raw["root"])


def add_part(lock: Lock, folder) -> Lock:
    """Make the component folder *folder* a part of the update of *lock*'s install,
    which is in phase STEPS, and start its record. Returns the lock of the part,
    which the update holds until *lock* is released. Raises :class:`Busy` and what
    :func:`start` raises."""
    record = current(lock.root)
    if record is None or record.phase != STEPS:
        raise ValueError("the update is not in its steps")
    own_part(lock.root, record.id, folder)
    named = [part_path(lock.root, f) for f in part_folders(lock.root, record.id)]
    here = part_path(lock.root, folder)
    if here not in named:
        disk.write_json(mover.work_folder(lock.root, record.id) / PARTS, [*named, here])
    part = _lock_part(lock, Path(folder))
    earlier = current(part.root)
    if earlier is not None and earlier.id == record.id:
        return part
    if earlier is not None:
        result = settle(part)
        if result.errors:
            raise Busy("; ".join(result.errors))
    start(part, record.id, old=record.old, new=record.new)
    advance(part, MOVING)
    advance(part, STEPS)
    return part


def _end_parts(lock: Lock, update_id: str, end) -> Result:
    """Run *end* on each part of the update *update_id* that has not ended, the last
    one first. Returns their errors and notes."""
    all_parts = Result("parts")
    for folder in reversed(part_folders(lock.root, update_id)):
        try:
            if disk.lstat(folder) is None:
                gone = disk.lstat(folder.parent) is not None and bool(os.listdir(folder.parent))
                (all_parts.notes if gone else all_parts.errors).append(
                    f"{folder} was removed and stays so" if gone else
                    f"{folder} cannot be reached (if it was deleted, create it as an empty folder)")
                continue
            part = _lock_part(lock, folder)
            record = current(part.root)
            if record is None or record.id != update_id:
                continue
            result = end(part)
            all_parts.errors += result.errors
            all_parts.notes += result.notes
            all_parts.held += result.held
            all_parts.packages += [{**p, "part": part_path(lock.root, folder)}
                                   for p in result.packages]
        except (Busy, Unreadable, Unsafe, Held, OSError, ValueError) as e:
            all_parts.errors.append(f"{folder}: {e}")
            if isinstance(e, (Held, Busy)):
                all_parts.held.append(e)
    return all_parts


def settle(lock: Lock, *, wait_s: Optional[float] = None) -> Result:
    """End what an interrupted change left in the component folder of *lock*, or do
    nothing (state ``busy``) when it is a part of an update that has not ended. A
    part whose update says that it is done is ended as done; everything else is put
    back, waiting *wait_s* for held files. Raises like :func:`restore`."""
    record = current(lock.root)
    if record is None:
        cleanup(lock)
        return Result("none")
    whole = owner(lock.root, record.id)
    if whole is not None:
        of = read(whole) if disk.lstat(whole) is not None else None
        if of is not None and of.id == record.id:
            return Result("busy")
        if record.phase == DONE or (
                record.phase == STARTING and over_of(whole, record.id) == DONE):
            return finish(lock)
    return restore(lock, wait_s=wait_s)


def abandon(lock: Lock) -> None:
    """End as it is a change of the component of *lock* that cannot be put back, or
    whose record is damaged, before all files of the component are replaced; what it
    kept of the old files goes. Raises :class:`Busy` for a part of an update, OSError
    or :class:`Unsafe`."""
    _held(lock)
    for update_id in ids(lock.root):
        whole = owner(lock.root, update_id)
        if (whole is not None and disk.lstat(mover.update_folder(whole) / update_id) is not None
                and over_of(whole, update_id) != DONE):
            raise Busy("an update of localm is not finished")
    for update_id in ids(lock.root):
        if disk.lstat(mover.work_folder(lock.root, update_id) / STATE) is None:
            continue
        if not (remove_staged(lock.root, update_id)
                and disk.remove_tree(mover.work_folder(lock.root, update_id))):
            raise OSError(f"{mover.update_folder(lock.root) / update_id} cannot be deleted")
    disk.remove_through(record_path(lock.root))


# ------------------------------------------------------------------ the end

@dataclass
class Result:
    """How an update ended. *state*: ``none`` (there was none), ``old`` (the old
    version is back), ``new`` (it was finished) or ``failed``. *errors* names every
    file and package that is not as it should be, *packages* the packages among them
    (:class:`.mover.Outcome`): installing exactly those again and calling again puts
    the old version back. *notes* are lines for the user; *held* has each
    :class:`.disk.Held` and :class:`Busy` behind an error or note. Calling again goes on from there."""
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
    record = current(lock.root)
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
    try:
        parts = _end_parts(lock, record.id, lambda part: finish(part, ask=ask))
        result.notes += parts.notes + parts.errors
        result.held += parts.held
    except (Unreadable, Unsafe, Held, OSError) as e:
        result.note("The parts of the update could not be ended", e)
    _hand_over(lock.root, record.id, result, ask)
    _close(lock, record, DONE, result)
    return result


def undone_note(new, old) -> str:
    """The note :func:`restore` gives an update from *old* to *new* it undid."""
    return (f"The update to {new} was undone. localm {old} is in place." if new
            else "The update was undone.")


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
    record = current(lock.root)
    if record is None:
        cleanup(lock)
        return Result("none")
    if record.phase == DONE:
        return _finish(lock, record, ask)
    result = Result("none" if record.phase == ENDED else "old")
    undone = undone_note(record.new, record.old)
    if record.phase != ENDED and (
            record.phase != ENDING or mover.has_plans(lock.root, record.id)):
        plans = mover.read_plans(lock.root, record.id)
        try:
            _set_phase(lock, record, RESTORING)
        except OSError as e:
            result.note("The record of the update could not be written", e)
        _free_reserve(lock.root, record.id)
        parts = _end_parts(lock, record.id, lambda part: restore(part, wait_s=wait_s, ask=ask))
        result.errors += parts.errors
        result.packages += parts.packages
        result.notes += [note for note in parts.notes if note != undone]
        result.held += parts.held
        for plan in reversed(plans):
            earlier = {d for p in plans if p.number < plan.number and p.step == plan.step
                       for d in p.dists}
            step = (partial(packages.back, earlier=earlier) if plan.step in mover.VENVS
                    else mover.back)
            out = step(lock.root, record.id, plan, wait_s=wait_s, ask=ask)
            result.errors += out.errors
            result.packages += out.packages
            result.notes += out.notes
            result.held += out.held
        result.errors += steps_data.back(lock.root, record.id)
        if result.errors:
            result.state = "failed"
            return result
        result.notes.insert(0, undone)
    _close(lock, record, ENDED, result)
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


def discard(root, update_id: str) -> bool:
    """Delete what was prepared for the update *update_id* of the install or component
    *root*, and nothing of one that was started (:func:`start`). Returns whether it is
    gone. Raises :class:`Unsafe` or OSError."""
    work = mover.work_folder(root, update_id)
    if disk.lstat(work / STATE) is not None:
        return False
    try:
        beside = disk.read_json(work / BESIDE, MAX_RECORD_BYTES)
    except ValueError:
        beside = None
    gone = remove_staged(root, update_id)
    if isinstance(beside, str) and "/" not in disk.clean(beside):
        for name in (beside, beside + INCOMING):
            gone = disk.remove_tree(disk.real_folder(root) / name) and gone
    return gone and disk.remove_tree(work)


def hold(root, update_id: str) -> int:
    """Lock the folder of the update *update_id*, which this process prepares, until the
    descriptor this returns is closed. Raises OSError or :class:`Unsafe`."""
    fd = disk.open_lock(mover.work_folder(root, update_id) / BUSY)
    try:
        disk.lock_file(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _prepared(root, update_id: str) -> Optional[bool]:
    """Whether a running process prepares the update *update_id*, which was not started,
    or the update it is a part of still needs it. None when nothing tells."""
    work = mover.work_folder(root, update_id)
    whole = owner(root, update_id)
    if whole is not None:
        mine = mover.work_folder(whole, update_id)
        return disk.lstat(mine) is not None and (
            disk.lstat(mine / STATE) is not None or _prepared(whole, update_id) is not False)
    if disk.lstat(work / BUSY) is None:
        return None
    try:
        os.close(hold(root, update_id))
    except (PermissionError, BlockingIOError):
        return True
    return False


def remove_staged(root, update_id: str) -> bool:
    """Delete what the update *update_id* keeps in the venvs of *root*. Returns
    whether it is gone."""
    folders = [mover.venv_folder(root, update_id, step) for step in mover.VENVS]
    return all([disk.remove_tree(folder) for folder in folders])


def cleanup(lock: Lock, *, keep: Optional[str] = None) -> list:
    """Delete what ended updates of *lock*'s install left, nothing of an update that
    did not end, and what was prepared by a process that ended (:func:`hold`) for one
    that was never started, except the updates *keep* and ``lock.keep``. A file a
    program still has loaded stays for the next call. Returns the folders that are
    left."""
    _held(lock)
    left = []
    for name in ("job.json.tmp", "job.json.gone", "started"):
        with contextlib.suppress(OSError, Held):   # a write that never took effect
            disk.unlink(record_path(lock.root).with_name(name))
    with contextlib.suppress(OSError, Held, Unsafe):   # stays for the next call
        _unread_results(lock.root)
    staged = disk.real_folder(lock.root / ".venv") / mover.VENV_FOLDER
    known = set(ids(lock.root))
    with contextlib.suppress(OSError):   # nothing is staged
        known.update(_listed(staged))
    for update_id in sorted(known - {keep, lock.keep}):
        try:
            if not _delete(lock, update_id):
                left.append(str(mover.update_folder(lock.root) / update_id))
        except (OSError, Unsafe, Unreadable, Held) as e:
            left.append(f"{update_id}: {e}")
    for venv in mover.VENVS.values():
        with contextlib.suppress(OSError):   # not empty, or never made
            os.rmdir(disk.real_folder(lock.root / venv) / mover.VENV_FOLDER)
    return left


#: In a result in the update folder: ``<pid>-<start>`` of the ``localm update`` that
#: waits for it.
RESULT_FOR = "for"


def result_name(update_id: str) -> str:
    """The name, in the update folder, of how the update *update_id* ended, for the
    ``localm update`` that waits for it."""
    return f"result-{update_id}.json"


def _unread_results(root) -> None:
    """Delete each result in the update folder of *root* (:func:`result_name`) whose
    ``localm update`` (:data:`RESULT_FOR`) no longer runs, and each that names none.
    Leaves them all when the processes of the install cannot be listed."""
    folder = mover.update_folder(root)
    names = [n for n in os.listdir(folder) if n.startswith("result-") and n.endswith(".json")]
    procs = shared.also("procs") if names else None
    if procs is None:
        return
    running = {f"{e.pid}-{e.start}"
               for e in procs.list(disk.short_path(disk.real_folder(root)))}
    for name in names:
        try:
            said = disk.read_json(folder / name, MAX_RECORD_BYTES)
        except ValueError:
            said = None
        if not isinstance(said, dict) or said.get(RESULT_FOR) not in running:
            disk.unlink(folder / name)


def _waited_for(root, update_id: str) -> bool:
    """Whether a part of the update *update_id*, which is done, has not ended or cannot be
    reached; it ends as done while the update's folder says so."""
    for folder in part_folders(root, update_id):
        if disk.lstat(folder) is None:
            return True
        record = current(folder)
        if record is not None and record.id == update_id:
            return True
    return False


def _delete(lock: Lock, update_id: str) -> bool:
    """Delete the folders of *update_id* when it is over, its plans first and what says
    that it is over last; that stays while a part of it waits (:func:`_waited_for`).
    Returns False when something is left."""
    root = lock.root
    work = mover.work_folder(root, update_id)
    over = over_of(root, update_id)
    mine = mover.venv_folder(root, update_id)
    if over is None:
        if disk.lstat(work / STATE) is None:
            used = _prepared(root, update_id)
            if used:
                return True
            if mover.has_plans(root, update_id) or disk.lstat(mine / "old") is not None:
                return False
            if used is False:
                return discard(root, update_id)
            with contextlib.suppress(OSError):   # only an empty one goes
                os.rmdir(work)
            return disk.remove_tree(mine)
        record = read(root)
        if mover.has_plans(root, update_id) or (record is not None and record.id == update_id):
            return True
    if over == DONE and not _hand_over(root, update_id, Result("new")):
        return False
    waited = over == DONE and _waited_for(root, update_id)
    if disk.lstat(work / "plans") is not None:
        os.rename(work / "plans", work / "gone")
    gone = remove_staged(root, update_id)
    for name in sorted(os.listdir(work)):
        path = work / name
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) or disk.is_link(st):
            gone = disk.remove_tree(path) and gone
        elif name not in (STATE, over) and not (waited and name == PARTS):
            disk.unlink(path)
    if waited:
        return False
    if gone:
        disk.unlink(work / STATE)
    return gone and disk.remove_tree(work)
