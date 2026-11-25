# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replacing a component in a folder of its own: the native runtimes, the managed
ComfyUI, the uv inside the install.

The new files are prepared beside the installed ones (:class:`Work`) while those stay
in use. :func:`change` then moves them in under a record, so that a failure, or a
process that ends at any moment, leaves the complete old component: at once, or when
:func:`settle` next finds the record. The same moves run as a part of an update
(:mod:`.steps_update`), where they end together with it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional

from . import disk, job, mover, shared
from .disk import Held, Unsafe
from .mover import Plan, Refused, StepFailed, Unreadable

#: Seconds a user of a component waits while another process moves its files.
SETTLE_WAIT_S = 120.0
#: Bytes left free on a volume besides what a step writes there.
SPARE_BYTES = 64 * 1024 * 1024
#: Seconds between two tries to put a change back that did not meet a held file.
_RETRY_S = 0.5
#: Seconds a user of a component waits for held files that are to be put back.
_USE_WAIT_S = 2.0


class Failed(Exception):
    """A change was not made and the old files are back.

    *reason* says why. *errors* names every file that could not be put back; the
    component is as it was only when it is empty. *held* has each :class:`.disk.Held`
    behind it."""

    def __init__(self, reason: str, errors: Optional[list] = None,
                 held: Optional[list] = None) -> None:
        self.reason, self.errors, self.held = reason, list(errors or ()), list(held or ())
        super().__init__("; ".join([reason, *self.errors]))


# -------------------------------------------------------------------- space

def require_space(folder, need: int, what: str) -> None:
    """Raise :class:`.mover.Refused` unless the volume of *folder* has *need* bytes
    for *what* and :data:`SPARE_BYTES` more. Raises OSError when it cannot tell."""
    path = Path(folder)
    while disk.lstat(path) is None and path != path.parent:
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < need + SPARE_BYTES:
        raise Refused(f"{what} needs {_size(need + SPARE_BYTES)} on the disk of {folder}, and "
                      f"{_size(free)} are free")


def _size(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


# --------------------------------------------------------------- preparing

class Place:
    """The folder of the update *update_id* in the component folder *root*, to prepare
    new files in."""

    def __init__(self, root, update_id: str) -> None:
        self.root, self.id = Path(root), update_id

    @property
    def path(self) -> Path:
        """The folder itself."""
        return mover.work_folder(self.root, self.id)

    def new(self, step: str) -> Path:
        """The folder that holds the new files of *step* (:data:`.mover.STEPS`)."""
        folder = mover.new_folder(self.root, self.id, step)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def recorded(self) -> None:
        """Called when the record of the change is on disk: it, not this process, keeps
        the folder from then on."""

    def beside(self, name: str) -> Path:
        """The path *name* in the component folder, for new files that are tested at a
        short path before they are moved to :meth:`new`; it is deleted with this
        place."""
        self.path.mkdir(parents=True, exist_ok=True)
        disk.write_json(self.path / job.BESIDE, disk.clean(name))
        return disk.real_folder(self.root) / name


class Work(Place):
    """A :class:`Place` of its own for one change; a context manager. It is deleted at
    the end unless a change under its id was started; one that a killed process left is
    deleted by the next one. *update_id*: of an update whose folder is there."""

    def __init__(self, root, update_id: Optional[str] = None) -> None:
        super().__init__(root, update_id or uuid.uuid4().hex)
        self._fd: Optional[int] = None

    def __enter__(self) -> Work:
        sweep(self.root)
        mover.update_folder(self.root, create=True)
        (self.path / "plans").mkdir(parents=True, exist_ok=True)
        self._fd = job.hold(self.root, self.id)
        return self

    def recorded(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __exit__(self, *_exc) -> None:
        self.recorded()
        with contextlib.suppress(Unreadable, Unsafe, Held, OSError):
            job.discard(self.root, self.id)


def sweep(root) -> None:
    """Delete what earlier changes of the component folder *root* left that no record
    and no running process needs. Does nothing while a change runs or is unfinished,
    and leaves what cannot be read or deleted."""
    with contextlib.suppress(job.Busy, Unreadable, Unsafe, Held, OSError):
        folder = mover.update_folder(root)
        if disk.lstat(folder) is None or job.read(root) is not None:
            return
        with job.Lock(root, wait_s=0) as lock:
            job.cleanup(lock)


# ------------------------------------------------------------------- moving

def staged_files(folder: Path) -> list:
    """Every file and link below *folder*, as POSIX paths, compiled Python files
    excepted."""
    found = []
    for here, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d != disk.PYCACHE]
        links = [d for d in dirs if disk.is_link(os.lstat(Path(here) / d))]
        dirs[:] = [d for d in dirs if d not in links]
        for name in [*names, *links]:
            found.append((Path(here) / name).relative_to(folder).as_posix())
    return sorted(found)


def _same_content(a: Path, a_st: os.stat_result, b: Path, b_st: os.stat_result) -> bool:
    if disk.is_link(a_st) or disk.is_link(b_st):
        return disk.is_link(a_st) and disk.is_link(b_st) and os.readlink(a) == os.readlink(b)
    return (mover._executable(a_st) == mover._executable(b_st)
            and shared.updater_components._same_file(a, b))


def plan_staged(root, update_id: str, step: str, *, remove: Iterable[str] = ()) -> Plan:
    """The plan that makes the files prepared for *step* (:meth:`Work.new`) those of
    its area, and takes the files *remove* out, those that are staged excepted. A
    staged file equal to the installed one is left out. Files are known by size and
    time (links by target). Reads only.

    Raises :class:`.mover.Refused` for a name that is taken by something else,
    :class:`.disk.Unsafe` and OSError."""
    number = mover.next_number(root, update_id)
    a = mover.area(root, update_id, step, number)
    a.new.mkdir(parents=True, exist_ok=True)
    staged = staged_files(a.new)
    disk.flush_files([a.beside(a.new, rel)[0] for rel in staged])
    new: dict = {}
    for rel in staged:
        source, source_st = a.beside(a.new, rel)
        path, st = a.find(rel)
        assert source_st is not None
        if st is None or (stat.S_ISDIR(st.st_mode) and not disk.is_link(st)) or (
                not _same_content(path, st, source, source_st)):
            new[rel] = disk.quick_identity(source)
    return mover.build(a, step, number, new, set(remove) - set(staged), identify=disk.quick_identity,
                       edited=lambda _rel, _old: False)


def move_in(root, update_id: str, step: str, *, remove: Iterable[str] = (),
            wait_s: Optional[float] = None, ask: Optional[disk.Ask] = None) -> int:
    """Plan (:func:`plan_staged`), write the plan and make its moves. *ask* is called
    about a file that is still held after *wait_s* (:class:`.disk.Wait`). Returns how
    many files it changed. Raises like :func:`plan_staged`, and :class:`StepFailed`."""
    plan = plan_staged(root, update_id, step, remove=remove)
    if plan.ops:
        mover.write_plan(root, update_id, plan)
        mover.forward(root, update_id, plan, wait_s=wait_s, ask=ask)
    return len(plan.ops)


#: What :func:`change` and an update's steps treat as a step that failed.
STEP_ERRORS = (StepFailed, Refused, Held, Unsafe, Unreadable, job.Busy, OSError)


def change(work: Place, stages: Iterable[Callable[[], object]], *, old: str = "", new: str = "",
           prove: Optional[Callable[[], Optional[str]]] = None,
           wait_s: Optional[float] = None, ask: Optional[disk.Ask] = None,
           whole: bool = False, reinstall: Optional[Reinstall] = None) -> job.Result:
    """Carry out *stages* (calls that plan and move what was prepared in *work*) under
    a record of *work*'s folder, then *prove*, which returns None or why the result
    does not work. Raises :class:`Failed`, with the old files back, when any of it
    fails. Ends what an interrupted earlier change left first; with *whole*, for a
    change that replaces the whole component, also one that cannot be put back.
    *reinstall* as in :func:`put_back`."""
    with job.Lock(work.root) as lock:
        lock.keep = work.id
        lost = _unsettled(lambda: put_back(lambda: job.settle(lock), reinstall))
        if lost and not whole:
            raise Failed("an interrupted change cannot be undone", lost)
        if lost:
            job.abandon(lock)
        job.start(lock, work.id, old=old, new=new)
        try:
            work.recorded()
            job.advance(lock, job.MOVING)
            job.advance(lock, job.STEPS)
            for stage in stages:
                stage()
            why = prove() if prove else None
            if why:
                raise Refused(why)
            job.advance(lock, job.STARTING)
            return job.finish(lock, ask=ask)
        except BaseException as e:
            try:
                back = put_back(lambda: job.restore(lock, wait_s=wait_s, ask=ask), reinstall)
            except STEP_ERRORS as again:
                back = job.Result("failed", [str(again)])
                back.held += [again] if isinstance(again, Held) else []
            if isinstance(e, StepFailed) and not isinstance(e.__cause__, (Exception, type(None))):
                raise e.__cause__ from None
            if not isinstance(e, STEP_ERRORS):
                raise
            if back.state == "new":
                return back
            reason = e.reason if isinstance(e, StepFailed) else str(e)
            held = e.held if isinstance(e, StepFailed) else [e] if isinstance(e, Held) else []
            raise Failed(reason, back.errors, [*held, *back.held]) from (
                e.__cause__ if isinstance(e, StepFailed) else e)


#: Installs the packages a way back names (``job.Result.packages``) again. Raises one of
#: :data:`STEP_ERRORS`.
Reinstall = Callable[[list], object]


def put_back(restore: Callable[[], job.Result], reinstall: Optional[Reinstall],
             repair: str = "Run localm's setup again, which repairs that.") -> job.Result:
    """What *restore*, a way back, returns once *reinstall* has installed the packages
    it named again. When some stay named, its last error names them and says *repair*."""
    result = restore()
    for _round in range(3):
        if not result.packages or reinstall is None:
            break
        try:
            reinstall(result.packages)
        except STEP_ERRORS as e:
            result.errors.append(str(e))
            if isinstance(e, Held):
                result.held.append(e)
            break
        result = restore()
    if result.packages and reinstall is not None:
        named = ", ".join(sorted({f"{p['name']} {p['version']}" for p in result.packages}))
        result.errors.append(f"These Python packages are not complete: {named}. {repair}")
    return result


def _unsettled(settled: Callable[[], Optional[job.Result]]) -> list:
    """Why what an interrupted change left, as *settled* ended it, can never be put
    back: a lost file or a damaged record. Empty when it is. Raises :class:`Failed`
    while an update or a program that holds a file has to end first."""
    for again in (True, True, False):
        try:
            result = settled()
        except Unreadable as e:
            return [str(e)]
        if result is None:
            return []
        if not again or not result.errors or result.held:
            break
        time.sleep(_RETRY_S)   # a passing error is no lost file
    if result.state == "busy":
        raise Failed("its files are being replaced, or an update of localm is not finished")
    if result.errors and result.held:
        raise Failed("an interrupted change could not be undone", result.errors, result.held)
    return result.errors


def mend(root) -> bool:
    """:func:`settle`, before the whole component in *root* is replaced. Returns
    whether an interrupted change can never be undone, so that only replacing it helps
    (:func:`change` with *whole*). Raises :class:`Failed` when an update or a program
    that holds a file has to end first, and one of :data:`STEP_ERRORS`."""
    return bool(_unsettled(lambda: settle(root)))


def put_folder(root, name: str, staged, *, prove: Optional[Callable[[], Optional[str]]] = None,
               ) -> job.Result:
    """Make the folder *staged*, which is on the volume of *root* and is gone
    afterwards, the folder *name* of the component folder *root*, in place of one that
    is there. *prove* as in :func:`change`. Raises :class:`Failed`, with *root* as it
    was, and one of :data:`STEP_ERRORS` with nothing changed."""
    root = Path(root)
    with Work(root) as work:
        disk.move(Path(staged), work.new(mover.WHOLE) / name, disk.Wait())
        return change(work, [lambda: move_in(root, work.id, mover.WHOLE,
                                             remove=folder_files(root, name))],
                      prove=prove, whole=True)


def folder_files(root, name: str) -> list:
    """The files of the folder *name* in *root*, as paths relative to *root*."""
    return [f"{name}/{rel}" for rel in staged_files(disk.real_folder(root) / name)]


def settle(root, wait_s: float = SETTLE_WAIT_S) -> Optional[job.Result]:
    """Called before the component folder *root* is used: end what an interrupted
    change left in it (:func:`.job.settle`), or wait while a change moves its files.
    Returns None when there was nothing to do, also while an update that has moved
    them all runs; else how the change ended. The state is ``busy`` when its files
    were still being moved after *wait_s* seconds, or an update has to end first.
    Raises :class:`Unreadable`, :class:`Unsafe` or OSError."""
    if disk.lstat(job.record_path(root)) is None:
        return None
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with job.Lock(root, wait_s=0) as lock:
                result = job.settle(lock, wait_s=None if wait_s else _USE_WAIT_S)
            return None if result.state == "none" else result
        except job.Busy:
            record = job.read(root)
            if record is None or record.phase in (job.STARTING, job.DONE):
                return None
            if time.monotonic() >= deadline:
                return job.Result("busy")
            time.sleep(0.1)


def unsettled(root, *, wait_s: float = 0.0, repair: str = "") -> Optional[str]:
    """:func:`settle`, by whoever uses the component folder *root*. Returns None, or why
    its files may be incomplete, with *repair* for what cannot be put back later.
    Costs one file check when there is nothing to do; never raises."""
    try:
        result = settle(root, wait_s=wait_s)
    except STEP_ERRORS as e:
        return f"an interrupted change of {root} could not be undone: {e}"
    if result is None or (result.state != "busy" and not result.errors):
        return None
    if not result.errors:
        return f"the files in {root} are being replaced"
    return (f"an interrupted change of {root} could not be undone: " + "; ".join(result.errors)
            + (f". {repair}" if repair and not result.held else ""))
