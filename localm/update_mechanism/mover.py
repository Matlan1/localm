# SPDX-License-Identifier: AGPL-3.0-or-later
"""The plan of one step of an update, the file moves it makes and the way back.

A step moves files inside one :class:`Area`. Its :class:`Plan` names every file it
replaces, adds or removes, with the identity of the old and of the new file, and is
on disk before the first move, and :class:`Marks` say of which files it moved. The
way back (:func:`back`) looks at each file of the plan where it is now and never at
how far the step came, so it can be repeated and interrupted. A file that is neither
version is never deleted: it stays where it is, or goes to the area's *kept* folder
when it is in the way of an old file. A file that is back is the user's from then on,
but for the packages that are to be whole (:func:`back`).
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Optional

from . import disk, shared
from .disk import Held, Unsafe, Wait

FORMAT = "localm-update-plan/1"
FOLDER = ".localm-update"
#: The folder in the venv that holds what a package step replaced and puts in.
VENV_FOLDER = ".localm-old"
#: The folder in the venv that holds files of the user; never deleted here.
VENV_KEPT = ".localm-kept"
MAX_PLAN_BYTES = 256 * 1024 * 1024

REPLACE, ADD, REMOVE = "replace", "add", "remove"
_KINDS = (REPLACE, ADD, REMOVE)
_ID = re.compile(r"[0-9a-f]{32}")
_PLAN_NAME = re.compile(r"[0-9]{2}-([a-z]+)\.json")
_MOVED = re.compile(r"moved-([0-9]{2})")
_MARK = re.compile(r"([+-])([0-9]{1,9})")
_TOP = frozenset(n.casefold() for n in shared.apply_update.NEVER_TOUCH) | {FOLDER}


class Refused(Exception):
    """The step cannot be made; nothing was changed."""


class Unreadable(Exception):
    """A record or plan of an update that is damaged."""


@dataclass
class Outcome:
    """What a way back left. *errors*: one line per file or package that is not as it
    was before the step; *packages*: ``{"step", "name", "version"}`` of each such
    package; *notes*: lines on files of somebody else that were found; *kept*: the
    paths moved to the *kept* folder; *held*: each :class:`.disk.Held` behind an
    error."""
    errors: list = field(default_factory=list)
    packages: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    held: list = field(default_factory=list)

    def fail(self, e: BaseException) -> None:
        """Take note of the error *e*."""
        self.errors.append(str(e))
        if isinstance(e, Held):
            self.held.append(e)


class StepFailed(Exception):
    """A step failed and its files were moved back.

    *reason* says why. *errors* names everything that could not be put back, *packages*
    the packages among it (:class:`Outcome`); the install is as it was before the step
    only when *errors* is empty; then the step can be made again, with the same plan
    when a file could not be moved, with a new one when a check after the moves failed.
    *notes* names the files of somebody else that were found, *held* has each
    :class:`.disk.Held` that stopped the step or its way back."""

    def __init__(self, cause: BaseException, outcome: Optional[Outcome] = None) -> None:
        outcome = outcome or Outcome()
        reason = str(cause) or type(cause).__name__
        self.reason, self.errors, self.notes = reason, outcome.errors, outcome.notes
        self.packages = outcome.packages
        self.held = [*([cause] if isinstance(cause, Held) else []), *outcome.held]
        super().__init__("; ".join([reason, *self.errors]))


# -------------------------------------------------------------------- areas

def is_id(update_id) -> bool:
    """Whether *update_id* is 32 lowercase hex digits, the id of an update."""
    return isinstance(update_id, str) and bool(_ID.fullmatch(update_id))


def check_id(update_id) -> str:
    """*update_id*, when :func:`is_id`. Raises :class:`Unreadable`."""
    if not is_id(update_id):
        raise Unreadable("the record of the update names no update")
    return update_id


def update_folder(root, *, create: bool = False) -> Path:
    """``<root>/.localm-update``. Raises :class:`Unsafe` when it is a link or not a
    folder."""
    folder = disk.real_folder(root) / FOLDER
    if create:
        folder.mkdir(exist_ok=True)
    return plain_folder(folder)


def plain_folder(folder: Path) -> Path:
    """*folder*, when it is a folder or nothing. Raises :class:`Unsafe`."""
    st = disk.lstat(folder)
    if st is not None and (disk.is_link(st) or not stat.S_ISDIR(st.st_mode)):
        raise Unsafe(f"{folder} is a link or not a folder")
    return folder


def work_folder(root, update_id: str) -> Path:
    """The folder of the update *update_id*. Raises :class:`Unsafe` when it is a
    link or not a folder."""
    return plain_folder(update_folder(root) / check_id(update_id))


class Area:
    """The folder a step changes (*base*), and where it keeps what it replaced
    (*old*), what it puts in or takes out again (*new*) and the files of somebody
    else (*kept*).

    *guard*: the folders an update never touches are refused. *exact*: every part of
    a path counts only under its exact spelling, else the ones that two paths given
    to :meth:`watch` spell differently. *closed*: top-level names of
    *base* that are never changed. *stay*: folders that are not removed when they
    become empty."""

    def __init__(self, base: Path, old: Path, new: Path, kept: Path, *, guard: bool = False,
                 exact: bool = False, closed: Iterable[str] = (),
                 stay: Iterable[Path] = ()) -> None:
        self.base, self.old, self.new, self.kept = base, old, new, kept
        self.guard, self.exact = guard, exact
        self.closed = frozenset(n.casefold() for n in closed)
        self.stay = frozenset([base, *stay])
        self._plain: set = set()
        self._twins: set = set()
        self._guarded = self._guards() if guard else set()

    def watch(self, paths: Iterable[str]) -> None:
        """Take note of the paths a plan names."""
        seen: dict = {}
        for rel in paths:
            while rel and seen.get(rel.casefold()) != rel:
                if seen.setdefault(rel.casefold(), rel) != rel:
                    self._twins.add(rel.casefold())
                    seen[rel.casefold()] = rel
                rel = rel.rpartition("/")[0]

    def _guards(self) -> set:
        names = [*shared.apply_update.NEVER_TOUCH, *shared.apply_update.PRESERVE_WITHIN, FOLDER]
        found = (disk.lstat(self.base / n) for n in names)
        return {(st.st_dev, st.st_ino) for st in found if st is not None and st.st_ino}

    def inside(self, rel: str) -> bool:
        """Whether *rel* is a path a step of this area may change."""
        folded = rel.casefold()
        top = folded.split("/", 1)[0]
        if top in self.closed:
            return False
        return not self.guard or (
            top not in _TOP and not shared.apply_update._within_preserved(folded))

    def find(self, rel: str) -> tuple:
        """``(path, stat)`` of *rel* in *base*; stat is None when there is nothing.
        Raises :class:`Unsafe` for a path this area does not cover or that leads
        through a link."""
        disk.clean(rel)
        if not self.inside(rel):
            raise Unsafe(f"{rel} is not changed by an update")
        return self._walk(self.base, rel)

    def beside(self, folder: Path, rel: str) -> tuple:
        """Like :meth:`find`, for *rel* in *folder*, one of *old*, *new* and
        *kept*."""
        disk.clean(rel)
        return self._walk(folder, rel)

    def _walk(self, base: Path, rel: str) -> tuple:
        path, st = base, None
        parts = rel.split("/")
        for i, part in enumerate(parts):
            path = path / part
            last = i == len(parts) - 1
            exact = self.exact or self._is_twin("/".join(parts[:i + 1]))
            if not last and not exact and path in self._plain:
                continue
            st = disk.lstat(path)
            if st is not None and (st.st_dev, st.st_ino) in self._guarded:
                raise Unsafe(f"{rel} is not changed by an update")
            if st is None or (exact and not disk.spelled(path)):
                return base.joinpath(*parts), None
            if not last:
                if disk.is_link(st):
                    raise Unsafe(f"{path} is a link")
                if not stat.S_ISDIR(st.st_mode):
                    return base.joinpath(*parts), None
                self._plain.add(path)
        return path, st

    def _is_twin(self, rel: str) -> bool:
        return bool(self._twins) and rel.casefold() in self._twins

    def respell(self, rel: str, wait: Wait) -> None:
        """Give each folder above *rel* in *base* the spelling it has in *rel*, where
        :meth:`watch` saw another one. Raises :class:`.disk.Held` or OSError."""
        parts = rel.split("/")
        for i in range(1, len(parts) if self._twins else 0):
            path = self.base.joinpath(*parts[:i])
            if (not self._is_twin("/".join(parts[:i])) or disk.lstat(path) is None
                    or disk.spelled(path)):
                continue
            for name in os.listdir(path.parent):
                if name.casefold() == path.name.casefold():
                    disk.patient(wait, path, partial(os.rename, path.with_name(name), path))

    def prune(self, folder: Path) -> None:
        """Remove *folder* and the folders above it while each is empty, except
        *base* and *stay*."""
        stop = folder
        while stop != self.base and stop not in self.stay and self.base in stop.parents:
            stop = stop.parent
        disk.prune(folder, stop)


def venv_of(root, step: str = "packages") -> Path:
    """The venv the package step *step* of *root* changes, with a link to it followed.
    Raises :class:`Refused` when there is none."""
    path = Path(root) / VENVS[step]
    venv = disk.real_folder(path)
    if not (venv / "pyvenv.cfg").is_file():
        raise Refused(f"{path} is not a Python environment")
    return venv


def venv_folder(root, update_id: str, step: str = "packages") -> Path:
    """The folder in the venv of *step* that belongs to the update *update_id*."""
    venv = disk.real_folder(Path(root) / VENVS[step])
    return plain_folder(plain_folder(venv / VENV_FOLDER) / check_id(update_id))


def area(root, update_id: str, step: str, number: int) -> Area:
    """The area of the plan *number* of *step*, one of :data:`STEPS`, of the update
    *update_id*."""
    work = work_folder(root, update_id)
    if step == FILES:
        kept = plain_folder(plain_folder(update_folder(root) / "kept") / update_id)
        return Area(disk.real_folder(root), old_folder(work, number),
                    plain_folder(work / "new"), kept, guard=True, exact=True)
    if step in BASES:
        mine = plain_folder(work / step)
        kept = plain_folder(plain_folder(update_folder(root) / "kept") / update_id)
        return Area(plain_folder(disk.real_folder(root) / BASES[step]),
                    old_folder(mine, number), new_folder(root, update_id, step),
                    plain_folder(kept / step), exact=step == TREE,
                    closed=(FOLDER, VENV_FOLDER, VENV_KEPT))
    if step in VENVS:
        venv = disk.real_folder(Path(root) / VENVS[step])
        mine = venv_folder(root, update_id, step)
        kept = plain_folder(plain_folder(venv / VENV_KEPT) / update_id)
        site = shared.updater_components.site_packages(venv)
        stay = [venv / "Scripts", venv / "bin", *([site, *site.parents] if site else [])]
        return Area(venv, old_folder(mine, number), stage_folder(root, update_id, number, step),
                    kept,
                    closed=(VENV_FOLDER, VENV_KEPT, "pyvenv.cfg", ".lock"), stay=stay)
    raise Unreadable(f"the plan of the update names an unknown step {step[:40]!r}")


def old_folder(mine: Path, number: int) -> Path:
    """Where the plan *number* keeps what it replaced, in the folder *mine* of its
    update."""
    return plain_folder(plain_folder(mine / "old") / f"{number:02d}")


def new_folder(root, update_id: str, step: str) -> Path:
    """Where the new files of *step*, one of :data:`BASES`, are prepared."""
    return plain_folder(plain_folder(work_folder(root, update_id) / step) / "new")


def stage_folder(root, update_id: str, number: Optional[int] = None,
                 step: str = "packages") -> Path:
    """The staging venv of the plan *number* of the package step *step* of the update
    *update_id*, None for the next one. The plans of a step are counted from 01, in
    their order."""
    folder = _plans_folder(root, update_id)
    names = os.listdir(folder) if disk.lstat(folder) is not None else ()
    nth = 1 + sum(n.endswith(f"-{step}.json") and (number is None or n < f"{number:02d}")
                  for n in names)
    return plain_folder(plain_folder(venv_folder(root, update_id, step) / "stage") / f"{nth:02d}")


FILES, PACKAGES = "files", "packages"
#: A step on the uv inside the install, on the whole folder of a job (file names
#: compared exactly), on the whole folder of a job, and on the venv inside the folder
#: of a job.
TREE, WHOLE, VENV, UV = "tree", "whole", "venv", "uv"
#: The venv of each package step, and the folder of each other step, below the root.
VENVS = {PACKAGES: ".venv", VENV: "venv"}
BASES = {UV: ".uv", TREE: "", WHOLE: ""}
#: The steps a plan can be of.
STEPS = (FILES, *VENVS, *BASES)


# -------------------------------------------------------------------- plans

@dataclass
class Op:
    """One file of a plan. *old* and *new* are identities (:mod:`.disk`); a removed
    file has no *new*, an added one no *old*. *swap*: the file goes first, and one
    that is replaced is never absent. *edited*: the old file is the user's own.
    *release*: what else counts as the old file, as it was before the user or as another
    package that lists it has it. *dist*: the Python package the file belongs to."""
    path: str
    kind: str
    old: Optional[dict] = None
    new: Optional[dict] = None
    swap: bool = False
    edited: bool = False
    release: list = field(default_factory=list)
    dist: str = ""


@dataclass
class Plan:
    """What one step changes. *folders* are the folders it creates. *dists* are the
    canonical names of the Python packages it changes, *before* the ``[canonical name,
    dist-info folder]`` of every package there was."""
    step: str
    number: int
    ops: list = field(default_factory=list)
    folders: list = field(default_factory=list)
    dists: list = field(default_factory=list)
    before: list = field(default_factory=list)


def _identity(value) -> Optional[dict]:
    if value is None:
        return None
    keys = set(value)
    if not all(type(value[k]) is int for k in keys & {"size", "mtime_ns"}):
        raise ValueError("an identity is not made of numbers")
    if keys == {"link"} and isinstance(value["link"], str):
        return value
    if keys == {"size", "mtime_ns"}:
        return value
    for checksum, least in (("sha256", {"size"}), ("copy", {"size", "mtime_ns"})):
        if ({checksum} | least <= keys <= {checksum, "size", "mtime_ns"}
                and shared.update_package.is_sha256(value[checksum])):
            return value
    raise ValueError("an identity is malformed")


def _parse_op(raw) -> Op:
    if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(Op)}:
        raise ValueError("an entry is malformed")
    op = Op(disk.clean(raw["path"]), raw["kind"], _identity(raw["old"]), _identity(raw["new"]),
            raw["swap"], raw["edited"], [_identity(i) for i in raw["release"]], raw["dist"])
    if (op.kind not in _KINDS or type(op.swap) is not bool or type(op.edited) is not bool
            or not isinstance(op.dist, str) or None in op.release
            or (op.old is None) != (op.kind == ADD) or (op.new is None) != (op.kind == REMOVE)
            or (op.swap and op.kind == REMOVE)):
        raise ValueError(f"the entry for {op.path} is malformed")
    return op


def parse(raw) -> Plan:
    """The :class:`Plan` in the JSON value *raw*. Raises :class:`Unreadable` for
    anything that is not one."""
    try:
        if (not isinstance(raw, dict) or raw.get("format") != FORMAT
                or set(raw) != {"format", *(f.name for f in fields(Plan))}
                or raw["step"] not in STEPS or type(raw["number"]) is not int
                or not all(isinstance(raw[k], list)
                           for k in ("ops", "folders", "dists", "before"))
                or not all(isinstance(d, str) for d in raw["dists"])
                or not all(isinstance(d, list) and [*map(type, d)] == [str, str]
                           for d in raw["before"])):
            raise ValueError("it is malformed")
        ops = [_parse_op(o) for o in raw["ops"]]
        if len({op.path for op in ops}) != len(ops):
            raise ValueError("it names a file twice")
        return Plan(raw["step"], raw["number"], ops, [disk.clean(f) for f in raw["folders"]],
                    raw["dists"], raw["before"])
    except (ValueError, TypeError, KeyError, AttributeError, Unsafe) as e:
        raise Unreadable(f"a plan of the update cannot be used: {e}") from None


def _plans_folder(root, update_id: str) -> Path:
    return plain_folder(work_folder(root, update_id) / "plans")


def write_plan(root, update_id: str, plan: Plan) -> None:
    """Put *plan* on disk. Raises :class:`Unreadable` for a plan that could not be
    read back, :class:`.disk.Held` or OSError."""
    raw = {"format": FORMAT, **asdict(plan)}
    parse(raw)
    disk.write_json(_plans_folder(root, update_id) / f"{plan.number:02d}-{plan.step}.json", raw)


def read_plans(root, update_id: str) -> list:
    """Every plan of the update *update_id*, in the order of the steps. Raises
    :class:`Unreadable` when one is damaged or missing, and :class:`.disk.Blocked` when
    one cannot be read now."""
    folder = _plans_folder(root, update_id)
    if disk.lstat(folder) is None:
        raise lost_track(root, [update_id], "its plans are missing")
    plans = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        match = _PLAN_NAME.fullmatch(name)
        try:
            plan = parse(disk.read_json(folder / name, MAX_PLAN_BYTES))
        except (ValueError, Unreadable) as e:
            raise lost_track(root, [update_id], f"{name}: {e}") from None
        if not match or match[1] != plan.step or int(name[:2]) != plan.number:
            raise lost_track(root, [update_id], f"{name} is not a plan of the update")
        plans.append(plan)
    numbers = [plan.number for plan in plans]
    if numbers != [*range(1, len(plans) + 1)] or not _moved(root, update_id) <= set(numbers):
        raise lost_track(root, [update_id], "a plan is missing")
    return plans


def _moved(root, update_id: str) -> set:
    """The numbers of the plans of which something was moved."""
    work = work_folder(root, update_id)
    found = {int(m[1]) for m in map(_MOVED.fullmatch, os.listdir(work)) if m}
    for mine in (work, *(plain_folder(work / step) for step in BASES),
                 *(venv_folder(root, update_id, step) for step in VENVS)):
        old = plain_folder(mine / "old")
        if disk.lstat(old) is not None:
            found |= {int(n) for n in os.listdir(old) if n.isdecimal()}
    return found


def lost_track(root, ids: Iterable[str], why: str) -> Unreadable:
    """The error that says that the updates *ids* can be neither finished nor undone,
    and what to do."""
    record = update_folder(root) / "job.json"
    paths = " and ".join(disk.short_path(p) for p in (*(record.parent / i for i in ids), record))
    return Unreadable(f"{why}, so the update can be neither finished nor undone. Install localm "
                      f"again (repair or setup), then delete {paths}")


def has_plans(root, update_id: str) -> bool:
    """Whether the update *update_id* has plans."""
    folder = _plans_folder(root, update_id)
    return disk.lstat(folder) is not None and any(n.endswith(".json") for n in os.listdir(folder))


def next_number(root, update_id: str) -> int:
    """The number of the next plan of the update *update_id*."""
    plans = read_plans(root, update_id) if has_plans(root, update_id) else ()
    return 1 + max((p.number for p in plans), default=0)


class Marks:
    """What the engine did to the files of a plan: ``+n`` is written before it changes
    the file of operation *n*, ``-n`` before it puts that file back, ``.`` when no
    change is under way and ``=`` when every file is back.

    *changed* and *back* are the numbers last written with that sign, *last* is the
    one whose change may not have been made, *over* says that ``=`` was last. A
    context manager. Raises :class:`Unreadable` or :class:`.disk.Blocked`."""

    def __init__(self, root, update_id: str, plan: Plan) -> None:
        self.path = work_folder(root, update_id) / f"moved-{plan.number:02d}"
        self.changed: set = set()
        self.back: set = set()
        self.last: Optional[int] = None
        self.over = False
        self._fd: Optional[int] = None
        try:
            text = (disk.read_file(self.path, MAX_PLAN_BYTES) or b"").decode("ascii", "replace")
        except ValueError as e:
            raise Unreadable(str(e)) from None
        for line in text.split("\n")[:-1]:
            self._take(line)

    def _take(self, line: str) -> None:
        match = _MARK.fullmatch(line)
        if match:
            self.last, self.over = int(match[2]), False
            to, out = ((self.changed, self.back) if match[1] == "+" else (self.back, self.changed))
            to.add(self.last)
            out.discard(self.last)
        elif line in (".", "="):
            self.last, self.over = None, line == "="

    def note(self, line: str, wait: Optional[Wait] = None) -> None:
        """Write *line*. Raises :class:`.disk.Held` or OSError."""
        if self._fd is None:
            self._fd = disk.patient(wait or Wait(), self.path, lambda: disk.open_lock(self.path))
            os.lseek(self._fd, 0, os.SEEK_END)
            os.write(self._fd, b"\n")
        os.write(self._fd, line.encode("ascii") + b"\n")
        self._take(line)

    def settle(self, line: str, wait: Optional[Wait] = None) -> None:
        """Write *line*; all of them are on disk when this returns."""
        self.note(line, wait)
        assert self._fd is not None
        os.fsync(self._fd)

    def __enter__(self) -> Marks:
        return self

    def __exit__(self, *_exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------- making a plan

def build(a: Area, step: str, number: int, new: Mapping, leaving: Iterable[str], *,
          identify: Callable[[Path], dict], edited: Callable[[str, dict], bool],
          swap: Iterable[str] = (), keep_equal: bool = False) -> Plan:
    """The plan that puts the files *new* (``{path: identity}``, staged in the area's
    *new* folder) into *a* and takes the files *leaving* out, as far as they are
    there. Reads only.

    *identify* gives the identity of a file that is there, *edited* whether a file
    that goes is the user's own; an identity without a checksum gets that of the
    content. *keep_equal*: a file that is already the new one stays. Raises :class:`Refused` for a path that is taken by something else or on
    another volume, :class:`Unsafe` for one through a link, :class:`.disk.Held` or
    OSError."""
    swap = list(swap)
    wait = Wait()

    def known(path: Path) -> dict:
        return disk.with_copy(path, identify(path), wait)
    a.watch([*(set(leaving) - set(new)), *new])
    removes: dict = {}
    changes: list = []
    taken: list = []
    for rel in sorted(set(leaving) - set(new)):
        path, st = a.find(rel)
        if st is not None and not _is_folder(st):
            old = known(path)
            removes[rel] = Op(rel, REMOVE, old, edited=edited(rel, old))
    for rel in sorted(new):
        path, st = a.find(rel)
        if st is None or _is_folder(st):
            changes.append(Op(rel, ADD, new=new[rel], swap=rel in swap))
            taken.append(rel)
            other = _other_spelling(path) if st is None else None
            if other is not None:
                name = other.relative_to(a.base).as_posix()
                if name not in removes:
                    old = known(other)
                    removes[name] = Op(name, REMOVE, old, edited=edited(name, old))
            continue
        old = known(path)
        if (keep_equal and not differs(old, new[rel])
                and _executable(st) == _executable(a.beside(a.new, rel)[1])):
            continue
        changes.append(Op(rel, REPLACE, old, new[rel], swap=rel in swap, edited=edited(rel, old)))
    gone = {rel.casefold() for rel in removes}
    for rel in taken:
        _check_taken(a, rel, gone)
    _check_volume(a, [*removes, *(op.path for op in changes)])
    first = sorted((op for op in changes if op.swap), key=lambda op: swap.index(op.path))
    rest = [op for op in changes if not op.swap]
    return Plan(step, number, [*first, *(removes[r] for r in sorted(removes)), *rest],
                _created(a, changes))


def plan_files(root, update_id: str, new_files: Mapping, old_files: Optional[Mapping] = None, *,
               swap: Iterable[str] = (), protect: Iterable[str]) -> Plan:
    """The plan that turns the install at *root* into the release extracted in
    ``<id>/new``. Reads only.

    *new_files* and *old_files* map the paths of the new release and of the
    installed one to ``{"sha256", "size"}`` (the ``files`` of their statements).
    A file only *old_files* names is removed; with None nothing is removed and every
    file that differs counts as the user's own. *swap* are the paths that go first,
    in that order, and are never absent. *protect* are the user's own folders below
    *root* besides the ones no update touches (a data or models folder kept there),
    relative to *root*; they are left alone with everything in them.

    Raises :class:`Refused` for an extracted file that differs from its statement,
    and what :func:`build` raises."""
    number = next_number(root, update_id)
    a = area(root, update_id, FILES, number)
    protect = [disk.clean(p) for p in protect]
    package = shared.update_package
    stated = dict(new_files)
    own, own_st = a.beside(a.new, package.NAME)
    if own_st is not None:
        stated.setdefault(package.NAME, package._file_entry(own))

    def covered(rel: str) -> bool:
        disk.clean(rel)
        return a.inside(rel) and not _protected(rel, protect)

    new = {rel: {k: stated[rel][k] for k in ("sha256", "size")} for rel in stated if covered(rel)}
    wait = Wait()
    for rel in sorted(new):
        source, source_st = a.beside(a.new, rel)
        if source_st is None or not disk.same(source, new[rel], wait):
            raise Refused(f"{rel} of the new version is missing or damaged")

    def edited(rel: str, old: dict) -> bool:
        if rel == package.NAME:
            return False
        if old_files is None:
            return True
        return rel not in old_files or differs(old, old_files[rel])

    plan = build(a, FILES, number, new,
                 [rel for rel in old_files or () if covered(rel)],
                 identify=lambda path: disk.content_identity(path, wait), edited=edited,
                 swap=swap, keep_equal=True)
    for op in plan.ops:
        was = (old_files or {}).get(op.path) if op.edited else None
        op.release = [{k: was[k] for k in ("sha256", "size")}] if was else []
    return plan


def _is_folder(st: os.stat_result) -> bool:
    return stat.S_ISDIR(st.st_mode) and not disk.is_link(st)


def _executable(st: Optional[os.stat_result]) -> bool:
    return os.name != "nt" and st is not None and bool(st.st_mode & 0o111)


def _protected(rel: str, protect: Iterable[str]) -> bool:
    folded = rel.casefold()
    return any(folded == p.casefold() or folded.startswith(p.casefold() + "/") for p in protect)


def differs(identity: Mapping, stated: Mapping) -> bool:
    """Whether the content identity *identity* is not the one *stated*."""
    return any(identity[k] != stated[k] for k in ("sha256", "size"))


def _check_volume(a: Area, paths: Iterable[str]) -> None:
    volume = os.lstat(a.new).st_dev
    seen: set = set()
    for rel in paths:
        parent = a.base.joinpath(*rel.split("/")).parent
        while parent not in seen:
            seen.add(parent)
            st = disk.lstat(parent)
            if st is not None:
                if st.st_dev != volume:
                    raise Refused(f"{rel} is on another volume than {a.new}")
                break
            parent = parent.parent


def _other_spelling(path: Path) -> Optional[Path]:
    """The regular file that *path* reaches under another spelling of its last part,
    None when there is none."""
    st = disk.lstat(path)
    if st is None or disk.spelled(path) or disk.is_link(st) or not stat.S_ISREG(st.st_mode):
        return None
    for name in os.listdir(path.parent):
        other = path.with_name(name)
        if name != path.name and os.path.samestat(os.lstat(other), st):
            return other
    return None


def _check_taken(a: Area, rel: str, gone: set) -> None:
    """Raise :class:`Refused` when something stands where *rel* is added, in this or
    another spelling, that the files *gone* (case folded) do not clear away."""
    parts = rel.split("/")
    for i in range(1, len(parts) + 1):
        path = a.base.joinpath(*parts[:i])
        st = disk.lstat(path)
        if st is None:
            return
        if i < len(parts) and _is_folder(st) and disk.spelled(path):
            continue
        if not _is_folder(st):
            if "/".join(parts[:i]).casefold() not in gone:
                raise Refused(f"{path} is in the place of {rel}")
            continue
        for folder, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d != disk.PYCACHE]
            for name in names:
                inner = (Path(folder) / name).relative_to(a.base).as_posix()
                if inner.casefold() not in gone:
                    raise Refused(f"{path} is in the place of {rel} and holds {inner}")


def _created(a: Area, ops: list) -> list:
    found: set = set()
    for op in ops:
        parts = op.path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            rel = "/".join(parts[:i])
            if rel in found:
                continue
            st = a.find(rel)[1]
            if st is None or not stat.S_ISDIR(st.st_mode):
                found.add(rel)
    return sorted(found)


# -------------------------------------------------------------- the way there

def forward(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None,
            ask: Optional[disk.Ask] = None,
            then: Optional[Callable[[Area], None]] = None,
            back: Optional[Callable[..., Outcome]] = None) -> None:
    """Make the moves of *plan*, which is already on disk (:func:`write_plan`), check
    every file and call *then*. *ask* is called about a file that is still held after
    *wait_s*; while it returns True the step waits on. Raises :class:`StepFailed`, with everything moved
    back (by *back*, :func:`back` by default), for any exception on the way,
    KeyboardInterrupt included: a move that fails, a held file, a file that is not
    what the plan says."""
    a = area(root, update_id, plan.step, plan.number)
    a.watch(op.path for op in plan.ops)
    try:
        wait = Wait(wait_s, ask)
        disk.flush_files([a.beside(a.new, op.path)[0] for op in plan.ops if op.new], wait)
        with Marks(root, update_id, plan) as marks:
            _go(a, plan, wait, marks)
        wrong = check(a, plan, wait)
        if wrong:
            raise Refused("; ".join(wrong[:5]))
        if then is not None:
            then(a)
    except BaseException as e:
        out = (back or _back)(root, update_id, plan, wait_s=wait_s, ask=ask)
        raise StepFailed(e, out) from e


def _go(a: Area, plan: Plan, wait: Wait, marks: Marks) -> None:
    """The moves, in groups with the disk flushed between them: the old files are
    safe before a new one takes their place."""
    numbers = {op.path: n for n, op in enumerate(plan.ops)}
    swaps = [op for op in plan.ops if op.swap]
    rest = [op for op in plan.ops if not op.swap]
    for op in swaps:
        if op.old is not None:
            path, st = a.find(op.path)
            _expect_old(op, path, st)
            disk.copy_whole(path, a.beside(a.old, op.path)[0], wait)
    disk.sync_disk()
    for op in swaps:
        path = a.find(op.path)[0]
        marks.note(f"+{numbers[op.path]}", wait)
        if op.old is None:
            disk.move(a.beside(a.new, op.path)[0], path, wait)
        else:
            disk.put_over(a.beside(a.new, op.path)[0], path, wait)
    for op in rest:
        if op.old is not None:
            path, st = a.find(op.path)
            _expect_old(op, path, st)
            marks.note(f"+{numbers[op.path]}", wait)
            disk.move(path, a.beside(a.old, op.path)[0], wait)
    for op in rest:
        if op.old is None:
            marks.note(f"+{numbers[op.path]}", wait)
    marks.settle(".", wait)
    disk.sync_disk()
    for op in rest:
        if op.kind == REMOVE:
            a.prune(a.find(op.path)[0].parent)
    for op in rest:
        if op.new is not None:
            path, st = a.find(op.path)
            if st is not None and _is_folder(st):
                a.prune(path)
                if disk.lstat(path) is not None:
                    disk.clear(path, wait)
            a.respell(op.path, wait)
            disk.move(a.beside(a.new, op.path)[0], path, wait)


def _expect_old(op: Op, path: Path, st: Optional[os.stat_result]) -> None:
    if st is None or op.old is None or not disk.unchanged(path, op.old):
        raise Refused(f"{op.path} changed while the update was prepared")


def check(a: Area, plan: Plan, wait: Optional[Wait] = None) -> list:
    """What differs in *a* from the state *plan* leads to, one line per file."""
    wrong = []
    for op in plan.ops:
        path, st = a.find(op.path)
        if op.new is None:
            if st is not None and not stat.S_ISDIR(st.st_mode):
                wrong.append(f"{op.path} is still there")
        elif not disk.same(path, op.new, wait):
            wrong.append(f"{op.path} is not the file of the new version")
    return wrong


def unfinished(root, update_id: str, plan: Plan) -> Optional[str]:
    """The first path that says that the moves of *plan* are neither made nor all
    taken back, None when nothing says so. A file the user changed since says
    nothing."""
    a = area(root, update_id, plan.step, plan.number)
    a.watch(op.path for op in plan.ops)
    with Marks(root, update_id, plan) as marks:
        if marks.over:
            return None
        for n, op in enumerate(plan.ops):
            path = a.find(op.path)[0]
            if op.new is not None:
                left = a.beside(a.new, op.path)[1] is not None and not disk.same(path, op.new)
            else:
                left = n not in marks.changed and op.old is not None and disk.same(path, op.old)
            if left:
                return op.path
    return None


# --------------------------------------------------------------- the way back

def back(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None,
         ask: Optional[disk.Ask] = None, wait: Optional[Wait] = None,
         out: Optional[Outcome] = None, whole: Iterable[str] = (),
         then: Optional[Callable[[], None]] = None) -> Outcome:
    """Put back what *plan* replaced and take out what it added, from what is on
    disk now, check every file and call *then*. Safe to repeat, and to run on a plan of
    which nothing or only a part was done.

    A file that is neither version stays; only one in the way of an old file, or in
    a folder in its way, is moved, to the area's *kept* folder, and noted. A file the
    update did not move, or that is back already, is the user's. One it moved is lost
    when its old copy is gone and no file with that content is at its place; putting
    it there heals that. The packages *whole* have no file of the user; *then* checks
    them. *out* is added to and returned."""
    out = out or Outcome()
    kept = ""
    try:
        a = area(root, update_id, plan.step, plan.number)
        kept = str(a.kept)
        a.watch(op.path for op in plan.ops)
        with Marks(root, update_id, plan) as marks:
            _Back(a, plan, wait or Wait(wait_s, ask), out, marks, whole).run()
            if then is not None:
                then()
            with contextlib.suppress(OSError, Held):   # a file without a mark is still looked at
                marks.settle("." if out.errors else "=")
    except (OSError, Held, Unsafe, Unreadable) as e:
        out.fail(e)
    if out.kept:
        out.notes.append(f"Files of neither version are in {kept}: {some(out.kept)}.")
    return out


_back = back


def some(names: Iterable[str]) -> str:
    """The first of *names*, and how many more there are."""
    names = sorted(names)
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def is_old(a: Area, op: Op, wait: Optional[Wait] = None) -> bool:
    """Whether the old file of *op*, or what else counts as it, is at its place."""
    path, st = a.find(op.path)
    return op.old is not None and st is not None and any(
        disk.same(path, i, wait) for i in (op.old, *op.release))


def by_installer(rel: str) -> bool:
    """Whether an installer writes the file *rel* of a venv itself, so that it can
    differ from one install of a package to the next."""
    folder, _, name = rel.rpartition("/")
    if folder.endswith(".dist-info"):
        return name in ("RECORD", "WHEEL", "INSTALLER", "REQUESTED", "direct_url.json",
                        "uv_cache.json", "uv_build.json")
    return rel.endswith(".pth") or rel.split("/", 1)[0] in ("Scripts", "bin")


class _Back:
    """One run of the way back of *plan* in *a*."""

    def __init__(self, a: Area, plan: Plan, wait: Wait, out: Outcome, marks: Marks,
                 whole: Iterable[str]) -> None:
        self.a, self.plan, self.wait, self.out, self.marks = a, plan, wait, out, marks
        self.whole = frozenset(whole)
        self.numbers = {op.path: n for n, op in enumerate(plan.ops)}
        self.unsure = marks.last
        self.failed: set = set()
        self.left: set = set()
        self.theirs: list = []
        self.sources: dict = {}

    def run(self) -> None:
        for op in self.plan.ops:   # before anything moves
            if op.old is not None:
                self._each(op, self._find_source)
        swaps = [op for op in self.plan.ops if op.swap]
        rest = [op for op in self.plan.ops if not op.swap]
        created = frozenset(self.plan.folders)
        for op in reversed(rest):
            self._each(op, self._take_out)
        for op in reversed(rest):
            if op.old is None:
                self._each(op, lambda op: self._prune_created(op, created))
        for op in reversed(rest):
            self._each(op, self._put_back)
        for op in reversed(swaps):
            self._each(op, self._take_out if op.old is None else self._put_back)
        disk.sync_disk()
        for op in self.plan.ops:
            self._each(op, self._verify)
        self._tell()

    def _each(self, op: Op, action: Callable[[Op], None]) -> None:
        if op.path in self.failed or op.path in self.left:
            return
        try:
            action(op)
        except (OSError, Held, Unsafe, Refused) as e:
            self._fail(op, e)

    def _fail(self, op: Op, why: Exception) -> None:
        self.failed.add(op.path)
        self.out.fail(why)

    def _mark(self, sign: str, op: Op) -> None:
        with contextlib.suppress(OSError, Held):   # a file without a mark is still looked at
            self.marks.note(f"{sign}{self.numbers[op.path]}", self.wait)

    # -- what there is to put back

    def _find_source(self, op: Op) -> None:
        assert op.old is not None
        path, st = self.a.beside(self.a.old, op.path)
        live, there = self.a.find(op.path)
        if st is not None and disk.same(path, op.old, self.wait):
            self.sources[op.path] = path
        elif is_old(self.a, op, self.wait):
            self._leave(op)
        elif op.dist or (st is None and not self._was_moved(op)):
            self._leave(op, theirs=not op.dist)
        elif op.edited and not op.release and (
                self.whole and (".dist-info/" in op.path or disk.lstat(self.a.old) is None)
                if there is None else not _is_folder(there) and (
                    op.new is None or not disk.same(live, op.new, self.wait))):
            self._leave(op)
        else:
            raise Refused(f"{op.path}: its old version is lost. Put that version of the file back "
                          "in its place.")

    def _was_moved(self, op: Op) -> bool:
        n = self.numbers[op.path]
        return n in self.marks.changed and (n != self.unsure or (
            op.new is not None and disk.same(self.a.find(op.path)[0], op.new, self.wait)))

    def _leave(self, op: Op, *, theirs: bool = False) -> None:
        """Leave what is at the place of *op*; what the update still has of the old
        file goes to *kept*."""
        self.left.add(op.path)
        self.theirs += [op.path] if theirs else []
        path, st = self.a.beside(self.a.old, op.path)
        if st is not None:
            disk.move(path, free(self.a, op.path), self.wait, through=True)
            self.out.kept.append(op.path)
        if self.numbers[op.path] in self.marks.changed:
            self._mark("-", op)

    # -- moving

    def _take_out(self, op: Op) -> None:
        path, st = self.a.find(op.path)
        if st is None:
            return
        if op.old is None:
            if not _is_folder(st):
                self._take_out_added(op, path)
        elif op.path in self.sources:
            if _is_folder(st):
                self.a.prune(path)
                if disk.lstat(path) is not None:
                    self._clear_folder(op.path)
            elif not disk.same(path, op.old, self.wait):
                aside(self.a, op, path, self.wait, self.out)

    def _take_out_added(self, op: Op, path: Path) -> None:
        n = self.numbers[op.path]
        same = op.new is not None and disk.same(path, op.new, self.wait)
        moved = n in self.marks.changed and self.a.beside(self.a.new, op.path)[1] is None
        if op.dist not in self.whole and (n in self.marks.back or not (same or moved)):
            self._leave(op, theirs=True)
        else:
            aside(self.a, op, path, self.wait, self.out)
            self._mark("-", op)

    def _clear_folder(self, rel: str) -> None:
        base = self.a.base.joinpath(*rel.split("/"))
        for folder, dirs, names in os.walk(base):
            links = [d for d in dirs if disk.is_link(os.lstat(Path(folder) / d))]
            dirs[:] = [d for d in dirs if d not in links and d != disk.PYCACHE]
            for name in [*names, *links]:
                path = Path(folder) / name
                inner = path.relative_to(self.a.base).as_posix()
                if inner not in self.numbers:
                    aside(self.a, Op(inner, REMOVE), path, self.wait, self.out)
        for folder, _dirs, _names in os.walk(base, topdown=False):
            self.a.prune(Path(folder))
        if disk.lstat(base) is not None:
            disk.clear(base, self.wait)

    def _prune_created(self, op: Op, created: frozenset) -> None:
        parent = op.path.rpartition("/")[0]
        while parent:
            if parent in created:
                path = self.a.find(parent)[0]
                disk.prune(path, path.parent)
            parent = parent.rpartition("/")[0]

    def _put_back(self, op: Op) -> None:
        source = self.sources.get(op.path)
        if source is None or op.old is None:
            return
        self.a.respell(op.path, self.wait)
        path, st = self.a.find(op.path)
        if st is not None and _is_folder(st):
            self._clear_folder(op.path)
            st = disk.lstat(path)
        self._mark("-", op)
        try:
            if st is None:
                disk.move(source, path, self.wait)
            elif not disk.same(path, op.old, self.wait):
                if op.new is None or not disk.same(path, op.new, self.wait):
                    aside(self.a, op, path, self.wait, self.out)
                else:
                    with contextlib.suppress(OSError, Held):   # only a second try needs it
                        disk.copy_whole(path, self.a.beside(self.a.new, op.path)[0], Wait(0))
                disk.put_over(source, path, self.wait)
            elif st.st_mode != os.lstat(source).st_mode:
                disk.put_over(source, path, self.wait)
            else:
                disk.unlink(source)
        except BaseException:
            self._mark("+", op)
            raise

    def _verify(self, op: Op) -> None:
        path, st = self.a.find(op.path)
        if op.old is None:
            if st is not None and not stat.S_ISDIR(st.st_mode):
                raise Refused(f"{op.path} was added and is still there")
        elif not disk.same(path, op.old, self.wait):
            raise Refused(f"{op.path} is not the file it was before the update")

    def _tell(self) -> None:
        if self.theirs:
            self.out.notes.append(f"In {self.a.base}: changed by somebody, and stays so: "
                                  f"{some(self.theirs)}.")
        above: set = set()
        for op in self.plan.ops:
            parent = op.path.rpartition("/")[0] if op.old is not None else ""
            while parent and parent not in above:
                above.add(parent)
                parent = parent.rpartition("/")[0]
        for rel in self.plan.folders:
            path, st = self.a.find(rel)
            if st is not None and _is_folder(st) and rel not in above:
                self.out.notes.append(f"{path} holds files of neither version and stays.")


def aside(a: Area, op: Op, path: Path, wait: Wait, out: Outcome) -> None:
    """Move the file at *path* out of the way: to *new* when it is the new file of
    *op*, else to *kept* under a free name, which is noted. Raises
    :class:`.disk.Held` or OSError."""
    target, st = a.beside(a.new, op.path)
    if st is None and op.new is not None and disk.same(path, op.new, wait):
        disk.move(path, target, wait)
        return
    disk.move(path, free(a, op.path), wait, through=True)
    out.kept.append(op.path)


def free(a: Area, rel: str) -> Path:
    """A path in *kept* for *rel* where nothing is."""
    target, st = a.beside(a.kept, rel)
    n = 0
    while st is not None or disk.lstat(target) is not None:
        n += 1
        target, st = a.beside(a.kept, f"{rel}.{n}")
    return target
