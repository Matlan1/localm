# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replacing Python packages in the venv of the install.

uv installs the new packages into a staging venv inside the venv
(:func:`stage_packages`), which holds besides them only the metadata of the packages
that stay, so uv chooses as it would in the venv. From there their files are moved
into place by :mod:`.mover`, like the files of a release (:func:`plan_packages`,
:func:`forward`). Nothing else writes to the venv, so the way back needs neither uv
nor a package source, and a file nobody planned is somebody else's.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import os
import posixpath
import re
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import disk, mover, shared
from .disk import Held, Unsafe, Wait
from .mover import PACKAGES, REMOVE, Area, Op, Outcome, Plan, Refused, Unreadable

_OWN = "localm"
_STAGED = "staged.json"
_LINE = re.compile(r" ([-+~]) ([A-Za-z0-9][A-Za-z0-9._-]*)(?:==(\S+)(?: \((.*)\))?| @ (.+))")
_COUNT = re.compile(r"Would (uninstall|install) (\d+) packages?")
_OTHER_VENV = re.compile(r"Would (?:replace|create|remove)\b.*\benvironment\b")
# What uv reads of an installed distribution to know that it is there.
_METADATA = ("METADATA", "WHEEL", "INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json",
             "uv_build.json")

#: Runs a command: ``(argv, cwd, env) -> (exit status, output)``.
Runner = Callable[[list, str, dict], tuple]


def run_hidden(argv: list, cwd: str, env: dict) -> tuple:
    """The default :data:`Runner`: runs *argv* without a window, output and errors
    together."""
    flags: dict = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {})
    done = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", check=False, **flags)
    return done.returncode, done.stdout


def command(argv: Iterable, *, project: bool = False, network: bool = False) -> list:
    """The uv command *argv* (``<uv> sync ...`` or ``<uv> pip install ...``) as this
    step runs it: offline unless *network*, and a sync without the project unless
    *project*."""
    argv = [str(a) for a in argv]
    extra = [] if network else ["--offline"]
    if argv[1:2] == ["sync"] and not project:
        extra.append("--no-install-project")
    return [*argv, *(e for e in extra if e not in argv)]


def _environment(env: Mapping) -> dict:
    out = {k: v for k, v in env.items()
           if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "UV_COMPILE_BYTECODE",
                        "FORCE_COLOR", "CLICOLOR_FORCE")}
    out["NO_COLOR"] = "1"
    return out


def _into(argv: list, root, env: dict, stage: Path) -> tuple:
    """*argv*, *root* and *env*, changed to install into the venv *stage*."""
    target = disk.short_path(stage)
    if argv[1:2] == ["sync"]:
        return argv, root, {**env, "UV_PROJECT_ENVIRONMENT": target}
    out: list = []
    skip = False
    for arg in argv:
        if skip or arg.startswith("--python="):
            skip = False
        elif arg in ("--python", "-p"):
            skip = True
        else:
            out.append(arg)
    return [*out, "--python", target], root, env


def _last(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[-3:])[-600:] or "no output"


def changes(output: str) -> tuple:
    """``(removed, installed)`` from the output of a uv dry run, each a list of
    ``(canonical name, version, local)``; *local* is true for a package from a folder
    or file of this machine, whose version may be empty. Raises :class:`Refused` when
    uv would make or replace a venv, or when its counts do not match the packages it
    names."""
    canon = shared.updater_components._canon
    removed, installed = [], []
    counts = {"uninstall": 0, "install": 0}
    for line in output.splitlines():
        if _OTHER_VENV.search(line):
            raise Refused(f"uv would not use this Python environment: {line.strip()}")
        count = _COUNT.match(line)
        if count:
            counts[count[1]] += int(count[2])
        found = _LINE.fullmatch(line.rstrip())
        if found:
            source = found[4] or found[5] or ""
            entry = (canon(found[2]), found[3] or "", "file:" in source)
            if found[1] in "-~":
                removed.append(entry)
            if found[1] in "+~":
                installed.append(entry)
    if len(removed) != counts["uninstall"] or len(installed) != counts["install"]:
        raise Refused(f"what uv would do could not be read: {_last(output)}")
    return removed, installed


# ------------------------------------------------------ the files of a package

def _records(site: Path, base: Path, folder: str, name: str) -> dict:
    """``{path in the venv: (stated hash, stated size or None)}`` of the files of the
    distribution *folder* in *site*: its RECORD and everything in its own folder,
    compiled Python files excepted. Raises :class:`Refused` without a usable RECORD,
    and for a path outside the venv."""
    inner = site.relative_to(base).as_posix()
    try:
        rows = shared.updater_components.record_rows(site, folder)
    except (OSError, ValueError, csv.Error) as e:
        raise Refused(f"the package {name} has no list of its files (RECORD), so it cannot be "
                      f"replaced safely: {e}") from None
    if not rows:
        raise Refused(f"the package {name} has an empty list of its files (RECORD)")
    found: dict = {}
    for row in rows:
        text = row[0].replace("\\", "/") if os.name == "nt" else row[0]
        rel = posixpath.normpath(posixpath.join(inner, text))
        if (posixpath.isabs(text) or rel == ".." or rel.startswith("../")
                or (os.name == "nt" and ":" in text)):
            raise Refused(f"the package {name} lists a file outside the Python environment: "
                          f"{row[0][:200]}")
        try:
            disk.clean(rel)
        except Unsafe:
            raise Refused(f"the package {name} has no usable list of its files (RECORD): "
                          f"{row[0][:80]!r}") from None
        if disk.PYCACHE not in rel.split("/"):
            size = row[2] if len(row) > 2 else ""
            found[rel] = (row[1] if len(row) > 1 else "",
                          int(size) if size.isdecimal() and len(size) < 19 else None)
    for entry in os.listdir(site / folder):
        st = os.lstat(site / folder / entry)
        if disk.is_link(st) or stat.S_ISREG(st.st_mode):
            found.setdefault(f"{inner}/{folder}/{entry}", ("", None))
    return found


def _infos(site: Path, names: Iterable[str]) -> list:
    """``(name, folder)`` of every distribution in *site* that is one of *names*."""
    names = set(names)
    return [(n, folder) for n, _v, folder in shared.updater_components.dist_infos(site)
            if n in names]


def _site(venv: Path) -> Path:
    site = shared.updater_components.site_packages(venv)
    if site is None:
        raise Refused(f"{venv} has no site-packages")
    return site


# ------------------------------------------------------------------- staging

def stage_folder(root, update_id: str) -> Path:
    """The staging venv of the update *update_id*."""
    return mover.plain_folder(mover.venv_folder(root, update_id) / "stage")


def stage_packages(root, update_id: str, argv: Iterable, *, env: Mapping, project: bool = False,
                   run: Runner = run_hidden, more: bool = False, network: bool = False) -> bool:
    """Put what the uv command *argv* (:func:`command`) would install in *root* into
    the staging venv of *update_id*, and note what it would remove. With *more* it
    is added to what earlier commands staged, and uv sees that as installed; else
    the staging venv is made anew. uv may download only with *network*. Returns
    False, with nothing staged, when uv has nothing to do. The venv itself is not
    changed, so this can run while localm does.

    What is staged stays for :func:`.job.start` with this id; :func:`.job.cleanup`
    and the start of another update delete it. An *argv* with ``--reinstall`` makes
    :func:`forward` fail, as uv then always has something to do.

    Raises :class:`Refused`, naming the package, for one without a RECORD, with a
    file outside the venv or behind a link, for localm itself without *project*, for
    another package from a folder or file, and when uv cannot make its plan or
    stages something else than it planned (the staging venv is then of no use);
    :class:`.disk.Held` and OSError otherwise. It can then be called again."""
    venv = mover.venv_of(root)
    site = _site(venv)
    env = _environment(env)
    full = command(argv, project=project, network=network)
    stage = stage_folder(root, update_id)
    path = stage.parent / _STAGED
    staged = _Staged.read(path) if more and disk.lstat(path) is not None else None
    if staged is None:
        code, output = run([*full, "--dry-run"], str(root), env)
    else:
        target = _into(full, root, env, stage)
        code, output = run([*target[0], "--dry-run"], str(root), target[2])
    if code != 0:
        raise Refused(f"uv cannot prepare the packages: {_last(output)}")
    removed, installed = changes(output)
    if not removed and not installed:
        return False
    _check_allowed(removed + installed, project)
    leaving = _leaving(venv, site, [e for e in removed
                                    if staged is None or e[0] not in staged.installed])
    if staged is None:
        if not disk.remove_tree(stage):
            raise Held(next((p for p in stage.rglob("*") if p.is_file()), stage),
                       "it cannot be deleted")
        stage.parent.mkdir(parents=True, exist_ok=True)
        _run(run, [full[0], "venv", "--offline", "--no-config", "--relocatable",
                   *(["--system-site-packages"] if _system_site(venv) else []),
                   "--python", disk.short_path(_python(venv)), disk.short_path(stage)], root, env)
        _seed(site, _site(stage))
        staged = _Staged(set(), set(), listing(stage))
    else:
        disk.unlink(path)
    staged_site = _site(stage)
    for name, folder in leaving:
        inner = f"{staged_site.relative_to(stage).as_posix()}/{folder}/"
        disk.remove_tree(staged_site / folder)
        staged.base = {rel for rel in staged.base if not rel.startswith(inner)}
        staged.removed.add((name, folder))
    _run(run, *_into(full, root, env, stage))
    staged.installed -= {e[0] for e in removed}
    staged.installed |= {e[0] for e in installed}
    _check_stage(stage, staged_site, staged)
    staged.write(path)
    return True


@dataclass
class _Staged:
    """What is staged: the ``(name, dist-info folder)`` of the packages that go, the
    names of those that come, and the paths of the staging venv itself."""
    removed: set
    installed: set
    base: set

    @staticmethod
    def read(path: Path) -> _Staged:
        try:
            raw: dict = disk.read_json(path, mover.MAX_PLAN_BYTES) or {}
            return _Staged({(str(n), str(f)) for n, f in raw["removed"]},
                           {str(n) for n in raw["installed"]},
                           {str(rel) for rel in raw.get("base", ())})
        except (ValueError, TypeError, KeyError):
            raise Refused("no packages are staged for this update") from None

    def write(self, path: Path) -> None:
        disk.write_json(path, {"removed": sorted(self.removed),
                               "installed": sorted(self.installed), "base": sorted(self.base)})


def _run(run: Runner, argv: list, root, env: dict) -> None:
    code, output = run(argv, str(root), env)
    if code != 0:
        raise Refused(f"uv could not install the packages: {_last(output)}")


def _check_allowed(entries: list, project: bool) -> None:
    for name, _version, local in entries:
        if name == _OWN and not project:
            raise Refused("uv would replace localm itself")
        if local and name != _OWN:
            raise Refused(f"the package {name} comes from a folder or file of this machine, so "
                          "it cannot be replaced safely")


def _leaving(venv: Path, site: Path, removed: list) -> list:
    """``(name, dist-info folder)`` of the packages *removed*, as they are in the
    venv. Raises :class:`Refused` for one without a RECORD or behind a link."""
    out = []
    wanted = [n for n, _v, _l in removed]
    a = Area(venv, venv, venv, venv)
    for name in sorted(set(wanted)):
        have = _infos(site, [name])
        if not have or len(have) != wanted.count(name):
            raise Refused(f"the package {name} has no list of its files (RECORD), so it cannot "
                          "be replaced safely")
        for _n, folder in have:
            try:
                for rel in _records(site, venv, folder, name):
                    a.find(rel)
            except Unsafe as e:
                raise Refused(f"the package {name} cannot be replaced safely: {e}") from None
        out += have
    return out


def _system_site(venv: Path) -> bool:
    text = (venv / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace").lower()
    return bool(re.search(r"^\s*include-system-site-packages\s*=\s*true\s*$", text, re.M))


def _python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


def _seed(site: Path, staged_site: Path) -> None:
    """Copy the metadata of every distribution of *site*, and nothing else of it,
    into *staged_site*."""
    for entry in sorted(os.listdir(site)):
        source = site / entry
        if not entry.endswith((".dist-info", ".egg-info")):
            continue
        if source.is_file():
            shutil.copyfile(source, staged_site / entry)
            continue
        names = _METADATA if entry.endswith(".dist-info") else ("PKG-INFO",)
        (staged_site / entry).mkdir(exist_ok=True)
        for name in names:
            if (source / name).is_file():
                shutil.copy2(source / name, staged_site / entry / name)


def _check_stage(stage: Path, site: Path, staged: _Staged) -> None:
    """Raise :class:`Refused` unless what uv added to *stage* is exactly the files of
    the packages *staged*, all of them complete, none naming the staging venv."""
    stated: dict = {}
    found = _infos(site, staged.installed)
    for name, folder in found:
        stated.update(_records(site, stage, folder, name))
    names = sorted(n for n, _f in found)
    if names != sorted(staged.installed):
        raise Refused("uv staged other packages than it planned: "
                      + ", ".join(sorted({*names} ^ staged.installed) or names))
    added = listing(stage) - staged.base
    if added != set(stated):
        raise Refused("uv staged other files than its packages list: "
                      + ", ".join(sorted(added ^ set(stated))[:5]))
    for rel, (_hash, size) in stated.items():
        path = stage.joinpath(*rel.split("/"))
        st = os.lstat(path)
        if size is not None and not disk.is_link(st) and st.st_size != size:
            raise Refused(f"{rel} was not staged completely")
        if _may_name_a_path(rel) and not disk.is_link(st) and _names(path, stage):
            raise Refused(f"{rel} names the staging folder")


def _may_name_a_path(rel: str) -> bool:
    return (".dist-info/" in rel or rel.endswith(".pth")
            or rel.split("/", 1)[0] in ("Scripts", "bin"))


def _names(path: Path, stage: Path) -> bool:
    text = disk.short_path(stage)
    data = path.read_bytes().lower()
    for form in {text, text.replace("\\", "/"), text.replace("\\", "\\\\")}:
        for encoding in ("utf-8", "utf-16-le"):
            if form.encode(encoding).lower() in data:
                return True
    return False


def listing(venv: Path) -> set:
    """The paths of every file and link in *venv* except compiled Python files and
    uv's lock file. Raises OSError."""
    found: set = set()
    todo = [("", venv)]
    while todo:
        prefix, folder = todo.pop()
        for name in os.listdir(folder):
            rel = prefix + name
            if name == disk.PYCACHE or rel == ".lock":
                continue
            st = os.lstat(folder / name)
            if stat.S_ISDIR(st.st_mode) and not disk.is_link(st):
                todo.append((rel + "/", folder / name))
            else:
                found.add(rel)
    return found


# ---------------------------------------------------------------- the moves

def plan_packages(root, update_id: str) -> Plan:
    """The plan that moves what :func:`stage_packages` staged into the venv and takes
    out what it replaces. Every file in the place of a new one is moved aside, also
    one of another package or of the user, whose own file is kept. Reads only.

    Raises :class:`Refused` when nothing is staged or the packages to replace are no
    longer those, and what :func:`.mover.build` raises."""
    venv = mover.venv_of(root)
    site = _site(venv)
    stage = stage_folder(root, update_id)
    staged = _Staged.read(stage.parent / _STAGED)
    removed, installed = staged.removed, staged.installed
    touched = {n for n, _f in removed} | installed
    if sorted(_infos(site, touched)) != sorted(removed):
        raise Refused("the packages of the Python environment changed since the update was "
                      "prepared")
    staged_site = _site(stage)
    new: dict = {}
    dists: dict = {}
    for name, folder in _infos(staged_site, installed):
        for rel, row in _records(staged_site, stage, folder, name).items():
            new[rel] = disk.quick_identity(stage.joinpath(*rel.split("/")))
            stated = _stated(row)
            if stated is not None and "link" not in new[rel]:
                new[rel]["copy"] = stated["sha256"]
            dists[rel] = name
    leaving: dict = {}
    for name, folder in removed:
        leaving.update(_records(site, venv, folder, name))
    owners: dict = {}
    for name, _v, folder in shared.updater_components.dist_infos(site):
        with contextlib.suppress(Refused):   # a package that stays needs no list
            owners.update(dict.fromkeys(_records(site, venv, folder, name), name))

    def edited(rel: str, old: dict) -> bool:
        stated = _stated(leaving.get(rel))
        return rel not in owners or (stated is not None and "link" not in old
                                     and stated != {"sha256": old["copy"], "size": old["size"]})

    a = mover.area(root, update_id, PACKAGES)
    plan = mover.build(a, PACKAGES, mover.next_number(root, update_id), new, leaving,
                       identify=disk.quick_identity, edited=edited)
    for op in plan.ops:
        op.dist = (dists if op.old is None else owners).get(op.path, "")
        op.release = _stated(leaving.get(op.path)) if op.edited else None
    plan.dists = sorted(touched)
    plan.venv = mover.venv_mark(a.base)
    return plan


def _stated(row: Optional[tuple]) -> Optional[dict]:
    """What a row of :func:`_records` states of the content, None for nothing."""
    digest, size = row or ("", None)
    try:
        raw = base64.urlsafe_b64decode(digest.removeprefix("sha256=") + "=")
    except ValueError:
        return None
    if size is None or not digest.startswith("sha256=") or len(raw) != 32:
        return None
    return {"sha256": raw.hex(), "size": size}


def forward(root, update_id: str, plan: Plan, argv: Iterable, *, env: Mapping,
            project: bool = False, run: Runner = run_hidden,
            wait_s: Optional[float] = None, ask: Optional[disk.Ask] = None) -> None:
    """Make the moves of *plan* (:func:`plan_packages`), then ask uv whether it has
    anything left to do for *argv*, the last command staged. *ask* as in
    :func:`.mover.forward`. Raises
    :class:`.mover.StepFailed`, with everything moved back, when any of it fails or
    is interrupted. Other programs' uv waits during the moves."""
    locked = contextlib.ExitStack()

    def nothing_left(_a: Area) -> None:
        locked.close()
        code, output = run([*command(argv, project=project), "--dry-run"], str(root),
                           _environment(env))
        if code != 0:
            raise Refused(f"uv cannot check the packages: {_last(output)}")
        removed, installed = changes(output)
        if removed or installed:
            raise Refused("uv would still change packages: "
                          + ", ".join(sorted({e[0] for e in removed + installed})))

    def undo(*args, **kwargs) -> Outcome:
        locked.close()
        return back(*args, **kwargs)
    with locked:
        try:
            venv = mover.area(root, update_id, plan.step).base
            locked.enter_context(_uv_waits(plan, venv, Wait(wait_s, ask)))
        except (OSError, Held, Unsafe) as e:
            raise mover.StepFailed(e) from e
        mover.forward(root, update_id, plan, wait_s=wait_s, ask=ask, then=nothing_left, back=undo)


@contextlib.contextmanager
def _uv_waits(plan: Plan, venv: Path, wait: Wait) -> Iterator[None]:
    """Hold uv's lock of *venv* for the block when *plan* is of the venv of the
    install. Raises :class:`.disk.Held` when uv has it for longer than *wait*,
    OSError otherwise."""
    if plan.step != PACKAGES:
        yield
        return
    path = venv / ".lock"

    def lock() -> int:
        fd = disk.open_lock(path)
        try:
            disk.lock_file(fd)
        except BaseException:
            os.close(fd)
            raise
        return fd
    fd = disk.patient(wait, path, lock, "uv is at work in this Python environment")
    try:
        yield
    finally:
        os.close(fd)


def back(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None,
         ask: Optional[disk.Ask] = None) -> Outcome:
    """:func:`.mover.back` for a package step. A package the update has changed and
    somebody installed in another form since is first moved to the *kept* folder as a
    whole, which is noted. One whose old version cannot be put back since somebody
    installed another version or removed it stays as it is."""
    out = Outcome()
    try:
        a = mover.area(root, update_id, PACKAGES)
        a.watch(op.path for op in plan.ops)
        site = shared.updater_components.site_packages(a.base)
        if disk.lstat(Path(root) / ".venv") is not None and not a.base.is_dir():
            raise Refused(f"{Path(root) / '.venv'} leads nowhere")
        if site is None or mover.made_anew(a, plan):
            return mover.back(root, update_id, plan, wait_s=wait_s, ask=ask, out=out)
        with _uv_waits(plan, a.base, Wait(wait_s, ask)):
            with mover.Marks(root, update_id, plan) as marks:
                had = {op.dist for op in plan.ops if op.old is not None}
                changed = {op.dist for n, op in enumerate(plan.ops)
                           if a.beside(a.old, op.path)[1] is not None
                           or (op.dist not in had and n in marks.changed)}
                foreign = _foreign(a, site, plan)
                users = {name: set() for name in changed & had
                         if _is_users(a, site, plan, name, foreign, marks)}
            for name, folder in foreign:
                if name in had - changed:
                    users.setdefault(name, {op.path for op in plan.ops if op.dist == name})
                if name in users:
                    users[name] |= _listed(site, a.base, folder, name)
            _set_aside(a, site, plan, {f for f in foreign if f[0] in changed - set(users)},
                       Wait(wait_s, ask), out)
            return mover.back(root, update_id, plan, wait_s=wait_s, ask=ask, out=out, users=users,
                              heal=lambda *args: _heal(a, *args))
    except (OSError, Held, Unsafe, Unreadable, Refused) as e:
        out.fail(e)
        return out


def _heal(a: Area, dist: str, ops: list, is_old: Callable[[Op], bool]) -> Optional[str]:
    """None when the package *dist* is completely there in its old version, but for
    what an installer writes itself, else that version."""
    version = ""
    whole = True
    for op in ops:
        folder, _, name = op.path.rpartition("/")
        if name == "METADATA" and folder.endswith(".dist-info"):
            version = folder.rpartition("/")[2].removesuffix(".dist-info").partition("-")[2]
        written = _may_name_a_path(op.path) and name != "METADATA"
        needed = not folder.endswith(".dist-info") or name == "RECORD"
        whole = whole and (is_old(op) or (
            written and (not needed or a.find(op.path)[1] is not None)))
    return None if whole and version else f"{dist} {version}".rstrip()


def _is_users(a: Area, site: Path, plan: Plan, name: str, foreign: set,
              marks: mover.Marks) -> bool:
    """Whether the old version of the package *name* cannot be put back, because
    somebody installed another one over what was still of it, or removed it."""
    changed = [op.path for n, op in enumerate(plan.ops)
               if op.dist == name and op.old is not None
               and (n not in marks.changed or n == marks.last)
               and a.beside(a.old, op.path)[1] is None
               and not disk.same(a.find(op.path)[0], op.old)]
    if any(rel.endswith(".dist-info/METADATA") and a.find(rel)[1] is None for rel in changed):
        return True
    return bool(changed) and any(
        f[0] == name and _is_complete(site, a.base, f[1], name) for f in foreign)


def _is_complete(site: Path, venv: Path, folder: str, name: str) -> bool:
    try:
        listed = _records(site, venv, folder, name)
    except Refused:
        return False
    return all(disk.same(venv.joinpath(*rel.split("/")), stated) for rel, stated in
               ((rel, _stated(row)) for rel, row in listed.items()) if stated is not None)


def _listed(site: Path, venv: Path, folder: str, name: str) -> set:
    try:
        return set(_records(site, venv, folder, name))
    except Refused:
        inner = site.relative_to(venv).as_posix()
        return {f"{inner}/{folder}/{e}" for e in os.listdir(site / folder)}


def _foreign(a: Area, site: Path, plan: Plan) -> set:
    """``(name, dist-info folder)`` of those of the packages of *plan* that are there
    in a form that is neither the old nor the new one."""
    found: set = set()
    inner = site.relative_to(a.base).as_posix()
    ops = {op.path: op for op in plan.ops}
    for name, folder in _infos(site, plan.dists):
        record = f"{inner}/{folder}/RECORD"
        op = ops.get(record)
        path, st = a.find(record)
        if op is not None and st is not None and any(
                disk.same(path, i) for i in (op.old, op.new) if i is not None):
            continue
        metadata = ops.get(f"{inner}/{folder}/METADATA")
        if st is not None or metadata is None:
            found.add((name, folder))
    return found


def _set_aside(a: Area, site: Path, plan: Plan, foreign: set, wait: Wait, out: Outcome) -> None:
    inner = site.relative_to(a.base).as_posix()
    ops = {op.path: op for op in plan.ops}
    for name, folder in sorted(foreign):
        record = f"{inner}/{folder}/RECORD"
        last = (record, f"{inner}/{folder}/METADATA")
        listed = _listed(site, a.base, folder, name)
        for rel in sorted(listed, key=lambda rel: (rel in last, rel == last[1], rel)):
            path, st = a.find(rel)
            op = ops.get(rel, Op(rel, REMOVE))
            if st is not None and op.old is None:
                mover.aside(a, op, path, wait, out)
                a.prune(path.parent)
        out.notes.append(f"The package {name} ({folder}) was installed by somebody since; it is "
                         f"in {a.kept}.")
    for folder in os.listdir(site):
        name = folder.removesuffix(".dist-info").partition("-")[0]
        if folder.endswith(".dist-info") and shared.updater_components._canon(name) in plan.dists:
            a.prune(site / folder)
