# SPDX-License-Identifier: AGPL-3.0-or-later
"""The plan of one step of an update, the file moves it makes and the way back.

A step moves files inside one :class:`Area`. Its :class:`Plan` names every file it
replaces, adds or removes, with the identity of the old and of the new file, and is
on disk before the first move. The way back (:func:`back`) looks at each file of
the plan where it is now and never at how far the step came, so it can be repeated
and interrupted.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from . import disk, shared
from .disk import Held, Unsafe, Wait

FORMAT = "localm-update-plan/1"
FOLDER = ".localm-update"
#: The folder in the venv that holds what a package step replaced.
VENV_FOLDER = ".localm-old"
MAX_PLAN_BYTES = 256 * 1024 * 1024

REPLACE, ADD, REMOVE = "replace", "add", "remove"
_KINDS = frozenset({REPLACE, ADD, REMOVE})
_ID = re.compile(r"[0-9a-f]{32}")
_PLAN_NAME = re.compile(r"[0-9]{2}-([a-z]+)\.json")
_TOP = frozenset(n.casefold() for n in shared.apply_update.NEVER_TOUCH) | {FOLDER}


class Refused(Exception):
    """The step cannot be made; nothing was changed."""


class Unreadable(Exception):
    """A record or plan of an update that cannot be read."""


class StepFailed(Exception):
    """A step failed and its files were moved back.

    *reason* says why. *errors* names every file that could not be put back; the
    install is as it was before the step only when it is empty."""

    def __init__(self, reason: str, errors: Optional[list] = None) -> None:
        self.reason, self.errors = reason, list(errors or ())
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
    st = disk.lstat(folder)
    if st is not None and (disk.is_link(st) or not stat.S_ISDIR(st.st_mode)):
        raise Unsafe(f"{folder} is not a folder of the install")
    return folder


def _plain_folder(folder: Path) -> Path:
    st = disk.lstat(folder)
    if st is not None and (disk.is_link(st) or not stat.S_ISDIR(st.st_mode)):
        raise Unsafe(f"{folder} is a link")
    return folder


def work_folder(root, update_id: str) -> Path:
    """The folder of the update *update_id*. Raises :class:`Unsafe` when it is a
    link or not a folder."""
    return _plain_folder(update_folder(root) / check_id(update_id))


class Area:
    """The folder a step changes (*base*), and where it keeps what it replaced
    (*old*), what it puts in or takes out again (*new*) and what is neither
    (*kept*). *exact*: a file counts as present only under its exact spelling, and
    the folders an update never touches are refused."""

    def __init__(self, base: Path, old: Path, new: Path, kept: Path, *, exact: bool) -> None:
        self.base, self.old, self.new, self.kept, self.exact = base, old, new, kept, exact
        self._plain: set = set()
        self._guarded = self._guards() if exact else set()

    def _guards(self) -> set:
        names = [*shared.apply_update.NEVER_TOUCH, *shared.apply_update.PRESERVE_WITHIN, FOLDER]
        found = (disk.lstat(self.base / n) for n in names)
        return {(st.st_dev, st.st_ino) for st in found if st is not None and st.st_ino}

    def inside(self, rel: str) -> bool:
        """Whether *rel* is a path a step of this area may change."""
        if not self.exact:
            return rel.split("/", 1)[0] != VENV_FOLDER
        folded = rel.casefold()
        return (folded.split("/", 1)[0] not in _TOP
                and not shared.apply_update._within_preserved(folded))

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
            if not last and not self.exact and path in self._plain:
                continue
            st = disk.lstat(path)
            if st is not None and (st.st_dev, st.st_ino) in self._guarded:
                raise Unsafe(f"{rel} is not changed by an update")
            if st is None or (self.exact and not disk.spelled(path)):
                return base.joinpath(*parts), None
            if not last:
                if disk.is_link(st):
                    raise Unsafe(f"{path} is a link")
                if not stat.S_ISDIR(st.st_mode):
                    return base.joinpath(*parts), None
                self._plain.add(path)
        return path, st


def venv_of(root) -> Path:
    """The venv of the install at *root*, with a link to it followed. Raises
    :class:`Refused` when there is none."""
    venv = disk.real_folder(Path(root) / ".venv")
    if not (venv / "pyvenv.cfg").is_file():
        raise Refused(f"{Path(root) / '.venv'} is not a Python environment")
    return venv


def area(root, update_id: str, step: str) -> Area:
    """The area of *step*, one of :data:`STEPS`, of the update *update_id*."""
    work = work_folder(root, update_id)
    if step == FILES:
        kept = _plain_folder(_plain_folder(update_folder(root) / "kept") / update_id)
        return Area(disk.real_folder(root), _plain_folder(work / "old"),
                    _plain_folder(work / "new"), kept, exact=True)
    if step == PACKAGES:
        venv = disk.real_folder(Path(root) / ".venv")
        mine = _plain_folder(_plain_folder(venv / VENV_FOLDER) / check_id(update_id))
        return Area(venv, _plain_folder(mine / "old"), _plain_folder(mine / "new"),
                    _plain_folder(mine / "new"), exact=False)
    raise Unreadable(f"the plan of the update names an unknown step {step[:40]!r}")


FILES, PACKAGES = "files", "packages"
#: The steps a plan can be of.
STEPS = (FILES, PACKAGES)


# -------------------------------------------------------------------- plans

@dataclass
class Op:
    """One file of a plan. *old* and *new* are identities (:mod:`.disk`); a removed
    file has no *new*, an added one no *old*. *swap*: the file is never absent.
    *edited*: the old file is the user's own version of it."""
    path: str
    kind: str
    old: Optional[dict] = None
    new: Optional[dict] = None
    swap: bool = False
    edited: bool = False


@dataclass
class Plan:
    """What one step changes. *folders* are the folders it creates. *before* is the
    listing of the whole area (:func:`listing`) where the step also runs another
    program in it."""
    step: str
    number: int
    ops: list = field(default_factory=list)
    folders: list = field(default_factory=list)
    before: Optional[dict] = None


def _identity(value, where: str) -> Optional[dict]:
    """*value*, when it is None or an identity. Raises :class:`Unreadable`."""
    if value is None:
        return None
    if isinstance(value, dict):
        keys = set(value)
        numbers = all(isinstance(value[k], int) and not isinstance(value[k], bool)
                      for k in keys & {"size", "mtime_ns"})
        if keys == {"link"} and isinstance(value["link"], str):
            return value
        if numbers and keys == {"size", "mtime_ns"}:
            return value
        if (numbers and {"sha256", "size"} <= keys <= {"sha256", "size", "mtime_ns"}
                and shared.update_package.is_sha256(value["sha256"])):
            return value
    raise Unreadable(f"the plan of the update has no usable identity for {where[:80]!r}")


def _parse_op(raw) -> Op:
    if not isinstance(raw, dict) or set(raw) != {"path", "kind", "old", "new", "swap", "edited"}:
        raise Unreadable("the plan of the update has a malformed entry")
    path = disk.clean(raw["path"])
    op = Op(path, raw["kind"], _identity(raw["old"], path), _identity(raw["new"], path),
            raw["swap"], raw["edited"])
    if (op.kind not in _KINDS or not isinstance(op.swap, bool) or not isinstance(op.edited, bool)
            or (op.old is None) != (op.kind == ADD) or (op.new is None) != (op.kind == REMOVE)
            or (op.swap and op.kind != REPLACE)):
        raise Unreadable(f"the plan of the update has a malformed entry for {path}")
    return op


def _parse_before(raw) -> Optional[dict]:
    if raw is None:
        return None
    if (not isinstance(raw, dict) or set(raw) != {"files", "folders"}
            or not isinstance(raw["files"], dict) or not isinstance(raw["folders"], list)):
        raise Unreadable("the plan of the update has a malformed listing")
    for rel, identity in raw["files"].items():
        if _identity(identity, rel) is None:
            raise Unreadable("the plan of the update has a malformed listing")
    if not all(isinstance(rel, str) for rel in raw["folders"]):
        raise Unreadable("the plan of the update has a malformed listing")
    return raw


def parse(raw) -> Plan:
    """The :class:`Plan` in the JSON value *raw*. Raises :class:`Unreadable` or
    :class:`Unsafe`."""
    if (not isinstance(raw, dict) or raw.get("format") != FORMAT
            or set(raw) != {"format", "step", "number", "ops", "folders", "before"}
            or raw["step"] not in STEPS or not isinstance(raw["number"], int)
            or not isinstance(raw["ops"], list) or not isinstance(raw["folders"], list)):
        raise Unreadable("the plan of the update is malformed")
    ops = [_parse_op(o) for o in raw["ops"]]
    if len({op.path for op in ops}) != len(ops):
        raise Unreadable("the plan of the update names a file twice")
    return Plan(raw["step"], raw["number"], ops, [disk.clean(f) for f in raw["folders"]],
                _parse_before(raw["before"]))


def _plans_folder(root, update_id: str) -> Path:
    return _plain_folder(work_folder(root, update_id) / "plans")


def write_plan(root, update_id: str, plan: Plan) -> None:
    """Put *plan* on disk. Raises OSError, and :class:`Unreadable` or
    :class:`Unsafe` for a plan that could not be read back."""
    raw = {"format": FORMAT, **asdict(plan)}
    parse(raw)
    disk.write_json(_plans_folder(root, update_id) / f"{plan.number:02d}-{plan.step}.json", raw)


def read_plans(root, update_id: str) -> list:
    """Every plan of the update *update_id*, in the order of the steps. Raises
    :class:`Unreadable` when one cannot be read or their folder is missing."""
    folder = _plans_folder(root, update_id)
    if disk.lstat(folder) is None:
        raise Unreadable(f"the folder of the update ({folder.parent}) is missing")
    plans = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        match = _PLAN_NAME.fullmatch(name)
        try:
            plan = parse(disk.read_json(folder / name, MAX_PLAN_BYTES))
        except ValueError as e:
            raise Unreadable(str(e)) from None
        if not match or match[1] != plan.step or int(name[:2]) != plan.number:
            raise Unreadable(f"{name} is not a plan of the update")
        plans.append(plan)
    return plans


def has_plans(root, update_id: str) -> bool:
    """Whether the update *update_id* has plans and all of them can be read."""
    try:
        return bool(read_plans(root, update_id))
    except (Unreadable, Unsafe, OSError):
        return False


def next_number(root, update_id: str) -> int:
    """The number of the next plan of the update *update_id*."""
    return 1 + max((p.number for p in read_plans(root, update_id)), default=0)


# ----------------------------------------------------- the plan of the files

def _executable(st: os.stat_result) -> bool:
    return os.name != "nt" and bool(st.st_mode & 0o111)


def _protected(rel: str, protect: Iterable[str]) -> bool:
    folded = rel.casefold()
    return any(folded == p.casefold() or folded.startswith(p.casefold() + "/") for p in protect)


def plan_files(root, update_id: str, new_files: Mapping, old_files: Optional[Mapping] = None, *,
               swap: Iterable[str] = (), protect: Iterable[str]) -> Plan:
    """The plan that turns the install at *root* into the release extracted in
    ``<id>/new``. Reads only.

    *new_files* and *old_files* map the paths of the new release and of the
    installed one to ``{"sha256", "size"}`` (the ``files`` of their statements).
    A file only *old_files* names is removed; with None nothing is removed and no
    file counts as edited. *swap* are the paths that are never absent; they go
    first, in that order. *protect* are the user's own folders below *root* besides
    the ones no update touches (a data or models folder kept there), relative to
    *root*; they are left alone with everything in them.

    Raises :class:`Refused` for an extracted file that differs from its statement,
    a path on another volume or a name that is taken by something else,
    :class:`Unsafe` for a path outside *root* or through a link, and OSError."""
    a = area(root, update_id, FILES)
    swap, protect = list(swap), [disk.clean(p) for p in protect]
    volume = os.lstat(a.new).st_dev
    package = shared.update_package
    stated = dict(new_files)
    own, own_st = a.beside(a.new, package.NAME)
    if own_st is not None:
        stated.setdefault(package.NAME, package._file_entry(own))

    def covered(rel: str) -> bool:
        disk.clean(rel)
        return a.inside(rel) and not _protected(rel, protect)

    removes, changes, taken = [], [], []
    for rel in sorted(set(old_files or ()) - set(stated)):
        if not covered(rel):
            continue
        path, st = a.find(rel)
        if st is None:
            continue
        old = disk.content_identity(path)
        removes.append(Op(rel, REMOVE, old, edited=_differs(old, (old_files or {})[rel])))
    for rel in sorted(stated):
        if not covered(rel):
            continue
        source, source_st = a.beside(a.new, rel)
        want = {k: stated[rel][k] for k in ("sha256", "size")}
        if source_st is None or not disk.same(source, want):
            raise Refused(f"{rel} of the new version is missing or damaged")
        path, st = a.find(rel)
        if st is None or (stat.S_ISDIR(st.st_mode) and not disk.is_link(st)):
            changes.append(Op(rel, ADD, new=want))
            taken.append(rel)
            continue
        old = disk.content_identity(path)
        if not _differs(old, want) and _executable(st) == _executable(source_st):
            continue
        edited = (old_files is not None and rel != package.NAME
                  and (rel not in old_files or _differs(old, old_files[rel])))
        changes.append(Op(rel, REPLACE, old, want, swap=rel in swap, edited=edited))
    leaving = {op.path for op in removes}
    for rel in taken:
        _check_taken(a, rel, leaving)
    for op in [*removes, *changes]:
        parent = a.base.joinpath(*op.path.split("/")).parent
        while disk.lstat(parent) is None:
            parent = parent.parent
        if os.lstat(parent).st_dev != volume:
            raise Refused(f"{op.path} is on another volume than {a.base}")
    first = sorted((op for op in changes if op.path in swap), key=lambda op: swap.index(op.path))
    rest = [op for op in changes if op.path not in swap]
    return Plan(FILES, next_number(root, update_id), [*first, *removes, *rest],
                _created(a, changes))


def _differs(identity: dict, stated: Mapping) -> bool:
    return any(identity[k] != stated[k] for k in ("sha256", "size"))


def _check_taken(a: Area, rel: str, leaving: set) -> None:
    """Raise :class:`Refused` when something stands where *rel* is added, in this or
    another spelling, that the files *leaving* do not clear away."""
    gone = {p.casefold() for p in leaving}
    parts = rel.split("/")
    for i in range(1, len(parts) + 1):
        path = a.base.joinpath(*parts[:i])
        st = disk.lstat(path)
        if st is None:
            return
        whole = i == len(parts)
        if (not whole and disk.spelled(path) and stat.S_ISDIR(st.st_mode)
                and not disk.is_link(st)):
            continue
        if disk.is_link(st) or not stat.S_ISDIR(st.st_mode):
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
    found = set()
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

def forward(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None) -> None:
    """Make the moves of *plan*, a plan of :func:`plan_files` already on disk
    (:func:`write_plan`), and check every file. Raises :class:`StepFailed`, with
    everything moved back, when a move fails, a file is held or a file is not what
    the plan says."""
    a = area(root, update_id, plan.step)
    try:
        disk.flush_files(a.beside(a.new, op.path)[0] for op in plan.ops if op.new)
        wait = Wait(wait_s)
        for op in plan.ops:
            _do(a, op, wait)
        wrong = check(a, plan)
        if wrong:
            raise Refused("; ".join(wrong[:5]))
    except (OSError, Held, Unsafe, Refused) as e:
        raise StepFailed(str(e), back(root, update_id, plan, wait_s=wait_s)) from e


def _do(a: Area, op: Op, wait: Wait) -> None:
    path, st = a.find(op.path)
    if op.old is None:
        if st is not None and stat.S_ISDIR(st.st_mode) and not disk.is_link(st):
            disk.prune(path, a.base)
    elif op.swap:
        _expect_old(op, path, st)
        disk.copy_whole(path, a.beside(a.old, op.path)[0], wait)
        disk.put_over(a.beside(a.new, op.path)[0], path, wait)
        return
    else:
        take_out(a, op, wait)
    if op.new is None:
        disk.prune(path.parent, a.base)
    else:
        disk.move(a.beside(a.new, op.path)[0], path, wait)


def _expect_old(op: Op, path: Path, st: Optional[os.stat_result]) -> None:
    if st is None or op.old is None or not disk.unchanged(path, op.old):
        raise Refused(f"{op.path} changed while the update was prepared")


def take_out(a: Area, op: Op, wait: Wait) -> None:
    """Move the old file of *op* to where the update keeps it. Raises
    :class:`Refused` when it is not that file any more, :class:`.disk.Held` or
    OSError."""
    path, st = a.find(op.path)
    _expect_old(op, path, st)
    disk.move(path, a.beside(a.old, op.path)[0], wait)


def check(a: Area, plan: Plan) -> list:
    """What differs in *a* from the state *plan* leads to, one line per file."""
    wrong = []
    for op in plan.ops:
        path, st = a.find(op.path)
        if op.new is None:
            if st is not None and not stat.S_ISDIR(st.st_mode):
                wrong.append(f"{op.path} is still there")
        elif not disk.same(path, op.new):
            wrong.append(f"{op.path} is not the file of the new version")
    return wrong


# --------------------------------------------------------------- the way back

def back(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None) -> list:
    """Put back what *plan* replaced and take out what it added, from what is on
    disk now, and check every file. Safe to repeat, and to run on a plan of which
    nothing or only a part was done. A file somebody wrote meanwhile in the place of
    one this step moved goes to the area's *kept* folder. Returns one line per file
    that is not as it was before the step; empty when all are."""
    a = area(root, update_id, plan.step)
    wait = Wait(wait_s)
    errors: dict = {}
    created = frozenset(plan.folders)
    for op in reversed(plan.ops):
        try:
            _undo(a, op, wait)
            if op.old is None:
                _prune_created(a, op.path, created)
        except (OSError, Held, Unsafe, Refused) as e:
            errors.setdefault(op.path, str(e))
    for op in plan.ops:
        if op.path in errors:
            continue
        try:
            path, st = a.find(op.path)
            if op.old is None:
                if st is not None and not stat.S_ISDIR(st.st_mode):
                    errors[op.path] = f"{op.path} was added and is still there"
            elif not disk.same(path, op.old) and (st is None or not _edited_since(a, op, path)):
                errors[op.path] = f"{op.path} is not the file it was before the update"
        except (OSError, Unsafe) as e:
            errors[op.path] = f"{op.path} cannot be checked: {e}"
    return list(errors.values())


def _edited_since(a: Area, op: Op, path: Path) -> bool:
    """Whether the file at *path*, which is not the old one, is somebody's later
    version of it that this update did not touch. Only in an area whose plan knows
    the new files."""
    return a.exact and (op.new is None or not disk.same(path, op.new))


def _undo(a: Area, op: Op, wait: Wait) -> None:
    path, st = a.find(op.path)
    present = st is not None and (disk.is_link(st) or not stat.S_ISDIR(st.st_mode))
    kept, kept_st = a.beside(a.old, op.path)
    if op.old is None:
        if present:
            aside(a, op, path, wait)
        return
    if kept_st is None:
        return
    if not disk.same(kept, op.old):
        raise Refused(f"{kept} is not the file that was replaced")
    if op.swap and present and op.new is not None and (
            disk.same(path, op.old) or disk.same(path, op.new)):
        disk.put_over(kept, path, wait)
        return
    if present:
        aside(a, op, path, wait)
    disk.move(kept, path, wait)


def _prune_created(a: Area, rel: str, created: frozenset) -> None:
    parent = rel.rpartition("/")[0]
    while parent:
        if parent in created:
            path = a.find(parent)[0]
            disk.prune(path, path.parent)
        parent = parent.rpartition("/")[0]


def aside(a: Area, op: Op, path: Path, wait: Wait) -> None:
    """Move the file at *path* out of the install: to *new* when it is the new file
    of *op*, else to *kept* under a free name. Raises :class:`.disk.Held` or
    OSError."""
    target, st = a.beside(a.new, op.path)
    if st is not None or op.new is None or not disk.same(path, op.new):
        target, st = a.beside(a.kept, op.path)
        n = 0
        while st is not None:
            n += 1
            target, st = a.beside(a.kept, f"{op.path}.{n}")
    disk.move(path, target, wait)
