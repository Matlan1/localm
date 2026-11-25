# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Python packages of an update, staged before anything is replaced
(:mod:`.packages`) and moved in later without a network.

:class:`Future` is the staging venv as the environment the code of the new version runs
in while the old one is still in use.
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Optional

from . import disk, mover, packages, shared
from .mover import Refused
from .steps import require_space

Say = Callable[[str], None]
#: The extras the PyTorch step installs (``updater_components.torch_step``).
TORCH_EXTRAS = frozenset({"gpu", "hf"})
_LINK = "_localm_update.pth"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: The PyTorch builds that have an index of their own.
_TORCH_BUILD = re.compile(r"==[^+]+\+(cpu|xpu|cu[0-9]+|rocm[0-9.]+)$")
_TORCH_INDEX = "https://download.pytorch.org/whl/"
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*")
# A wheel in the download cache, unpacked there and, in the worst case, copied twice.
_SPACE_FACTOR = 4


def sync_argv(uv, extras: Iterable[str]) -> list:
    """``localm.pyinstall.sync_command`` of the uv at *uv*."""
    argv = [str(uv), "sync", "--frozen", "--inexact", "--no-config"]
    for extra in extras:
        argv += ["--extra", str(extra)]
    return argv


def _project(tree) -> dict:
    try:
        with open(Path(tree) / "pyproject.toml", "rb") as f:
            return tomllib.load(f).get("project") or {}
    except (OSError, ValueError, RecursionError) as e:
        raise Refused(f"the project file of {tree} cannot be read: {e}") from None


def scripts(tree) -> dict:
    """The console and window scripts the project in *tree* installs."""
    project = _project(tree)
    return {kind: project.get(kind) or {} for kind in ("scripts", "gui-scripts")}


def used_extras(root, new, uv, *, env: Mapping, run: packages.Runner = packages.run_hidden) -> list:
    """The extras the dependency step of the install at *root* installs from the tree
    *new*: the ones every update installs, and each other extra both trees name whose
    packages the install's environment has. Raises :class:`.mover.Refused`."""
    always = list(shared.apply_update.DEPS_EXTRAS)
    here = set(_project(root).get("optional-dependencies") or {})
    there = set(_project(new).get("optional-dependencies") or {})
    used = [e for e in always if e in there]
    for extra in sorted((here & there) - set(used) - TORCH_EXTRAS):
        argv = [*packages.command(sync_argv(uv, [extra])), "--dry-run"]
        code, output = run(argv, str(root), packages._environment(env))
        if code != 0:
            continue
        removed, installed = packages.changes(output)
        if {e[0] for e in installed} <= {e[0] for e in removed}:
            used.append(extra)
    return used


def _wheel_sizes(new) -> dict:
    """``{(canonical name, version): bytes}`` of the largest file the lock of the tree
    *new* names for each package."""
    try:
        with open(Path(new) / "uv.lock", "rb") as f:
            entries = tomllib.load(f).get("package") or []
    except (OSError, ValueError, RecursionError):
        return {}
    canon = shared.updater_components._canon
    sizes: dict = {}
    for entry in entries:
        files = [w.get("size") for w in entry.get("wheels") or []]
        files.append((entry.get("sdist") or {}).get("size"))
        known = [s for s in files if isinstance(s, int)]
        if known and isinstance(entry.get("version"), str):
            sizes[(canon(str(entry.get("name"))), entry["version"])] = max(known)
    return sizes


def python_in(venv: Path) -> Path:
    """The interpreter of the Python environment at *venv*."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class Future:
    """The staged packages of the update *update_id* of the install at *root* as an
    environment (:func:`.packages.stage_packages`): it has the packages that change,
    and finds every other one in the install's venv. The new tree is *tree*
    (``<id>/new`` when not given). *uv* is the full path of the uv to run."""

    def __init__(self, root, update_id: str, uv, *, env: Mapping, say: Say, tree=None,
                 run: packages.Runner = packages.run_hidden) -> None:
        self.root, self.id, self.uv, self.say, self._run = Path(root), update_id, str(uv), say, run
        self.work = mover.work_folder(root, update_id)
        self.tree = Path(disk.short_path(self.work / "new" if tree is None else Path(tree)))
        self.venv = mover.venv_of(root)
        self.env = dict(env)

    @property
    def stage(self) -> Path:
        """The staging venv."""
        return packages.stage_folder(self.root, self.id)

    @property
    def staged(self) -> bool:
        """Whether any package is staged."""
        return disk.lstat(self.stage / "pyvenv.cfg") is not None

    def _link(self) -> Optional[Path]:
        site = shared.updater_components.site_packages(self.stage)
        return None if site is None else site / _LINK

    @property
    def python(self) -> Path:
        """The interpreter that runs the new tree: the stage's, linked to the packages
        of the install's venv, or the install's when nothing is staged."""
        if not self.staged:
            return Path(disk.short_path(python_in(self.venv)))
        link, site = self._link(), shared.updater_components.site_packages(self.venv)
        assert link is not None and site is not None
        link.write_text(f"import site; site.addsitedir({disk.short_path(site)!r})\n",
                        encoding="utf-8")
        return Path(disk.short_path(python_in(self.stage)))

    def add(self, argv: list, *, project: bool = False) -> bool:
        """Stage what the uv command *argv* (``uv sync ...``, or ``uv pip install -p
        .venv ...``) changes, with the network, besides what is staged already. Returns
        whether it changes a package. Raises :class:`.mover.Refused` and OSError."""
        link = self._link()
        if link is not None:
            link.unlink(missing_ok=True)
        full, cwd, env = packages._into(
            packages.command(argv, project=project, network=True), self.tree,
            packages._environment(self.env), self.stage if self.staged else self.venv)
        code, output = self._run([*full, "--dry-run"], str(cwd), env)
        if code != 0:
            raise Refused(f"uv cannot prepare the new Python packages: {packages._last(output)}")
        installed = packages.changes(output)[1]
        if installed:
            sizes = _wheel_sizes(self.tree)
            require_space(self.venv, _SPACE_FACTOR * sum(sizes.get(e[:2], 0) for e in installed),
                          "fetching the new Python packages")
            names = sorted(f"{e[0]} {e[1]}".strip() for e in installed)
            more = f" and {len(names) - 12} more" if len(names) > 12 else ""
            self.say(f"Fetching Python packages: {', '.join(names[:12])}{more} ...")
        return packages.stage_packages(self.root, self.id, argv, env=self.env, project=project,
                                       run=self._run, more=True, tree=self.tree, network=True)

    def sync(self, extras: list) -> Optional[dict]:
        """Stage the dependencies of the new tree with *extras*. Returns the step to
        record (:func:`run_packages`), None when the install's venv already has them.
        Raises :class:`.mover.Refused` and OSError."""
        project = scripts(self.root) != scripts(self.tree)
        changed = self.add(sync_argv(self.uv, extras), project=project)
        return {"extras": list(extras), "project": project} if changed else None


def run_packages(root, update_id: str, sync: Optional[Mapping], more: list, uv, *, env: Mapping,
                 run: packages.Runner = packages.run_hidden,
                 ask: Optional[disk.Ask] = None) -> None:
    """Move the staged packages in. *ask* as in :func:`.steps.move_in`. *sync* is a step of :meth:`Future.sync` or None,
    *more* the ``uv pip`` commands staged after it. Raises :class:`.mover.Refused` with
    nothing changed, :class:`.mover.StepFailed`, OSError."""
    plan = packages.plan_packages(root, update_id)
    mover.write_plan(root, update_id, plan)
    if more:
        argv, project = [str(uv), *more[-1]], False
    else:
        assert sync is not None
        argv, project = sync_argv(uv, sync["extras"]), bool(sync["project"])
    packages.forward(root, update_id, plan, argv, env=env, project=project, run=run, ask=ask)


# ------------------------------------------------- the old versions once more

def _locked(root) -> dict:
    """``{(canonical name, version): entry}`` of the lock file of the install at
    *root*, empty without a readable one."""
    try:
        with open(Path(root) / "uv.lock", "rb") as f:
            entries = tomllib.load(f).get("package") or []
    except (OSError, ValueError, RecursionError):
        return {}
    canon = shared.updater_components._canon
    return {(canon(str(e.get("name"))), str(e.get("version"))): e for e in entries}


def _hashes(entry: Mapping) -> list:
    files = [*(entry.get("wheels") or []), entry.get("sdist") or {}]
    return [f["hash"] for f in files if isinstance(f.get("hash"), str)]


def _requirements(root, wanted: Iterable) -> tuple:
    """``(lines, indexes, other lines, their indexes)`` of requirements files for the
    packages *wanted*, ``(name, version)``: with the checksums of the lock of the install
    at *root*, and those it does not name in that version."""
    locked = _locked(root)
    canon = shared.updater_components._canon
    lines, indexes, other, places = [], [], [], []
    for name, version in wanted:
        entry = locked.get((canon(name), version)) or {}
        source, sums = entry.get("source") or {}, _hashes(entry)
        if sums and (source.get("registry") or source.get("url")):
            at = f" @ {source['url']}" if source.get("url") else f"=={version}"
            lines.append(" ".join([f"{name}{at}", *(f"--hash={s}" for s in sums)]))
            if source.get("registry") and source["registry"] not in indexes:
                indexes.append(source["registry"])
        else:
            other.append(f"{name}=={version}")
            places += [r for (n, _v), e in locked.items() if n == canon(name)
                       and (r := (e.get("source") or {}).get("registry")) and r not in places]
    return lines, indexes, other, places


def index_args(argvs: Iterable) -> list:
    """The index options of the ``pip install`` command lines *argvs*."""
    out: list = []
    for argv in argvs:
        for n, arg in enumerate(argv):
            if arg in ("--index-url", "--extra-index-url") and n + 1 < len(argv):
                if ["--extra-index-url", argv[n + 1]] != out[-2:]:
                    out += ["--extra-index-url", argv[n + 1]]
    return out


def _own_indexes(wanted: Iterable) -> list:
    """The index options of the PyTorch builds among *wanted* requirements, which the
    local part of their version names (``2.9.0+cu126``)."""
    out: list = []
    for line in wanted:
        build = _TORCH_BUILD.search(line)
        if build and f"{_TORCH_INDEX}{build[1]}" not in out:
            out += ["--extra-index-url", f"{_TORCH_INDEX}{build[1]}"]
    return out


def reinstall(root, update_id: str, step: str, wanted: Iterable, uv, *, env: Mapping,
              indexes: Iterable = (), extras: Iterable = (), network: bool = True,
              run: packages.Runner = packages.run_hidden, wait_s: Optional[float] = None,
              ask: Optional[disk.Ask] = None) -> None:
    """Install exactly the packages *wanted*, ``(name, version)``, into the venv of the
    package step *step* of *root* with uv: from the index and with the checksums of the
    lock of *root* when it names that version, else from *indexes*
    (:func:`index_args`), the index of a PyTorch build, the one the lock names for
    another version and the default index; localm
    itself from *root*; without *network* from uv's cache only. Files in them that the
    user edited are copied to the *kept* folder first. *ask* is called
    about a file of theirs that is still held after *wait_s*. A venv of the install that
    is gone is made anew from its lock, with *extras*. Raises :class:`.disk.Held`, and
    :class:`.mover.Refused` and OSError: the packages may then be incomplete, and the way
    back names them again."""
    wanted = [(str(n), str(v)) for n, v in wanted]
    if not all(_NAME.fullmatch(n) and _VERSION.fullmatch(v) for n, v in wanted):
        raise Refused(f"these are no packages: {wanted!r}"[:200])
    offline = [] if network else ["--offline"]
    venv = disk.real_folder(Path(root) / mover.VENVS[step])
    work = mover.work_folder(root, update_id)
    commands: list = []
    if disk.lstat(venv / "pyvenv.cfg") is None:
        if step != mover.PACKAGES:
            raise Refused(f"{venv} is not a Python environment any more")
        commands.append(packages.command(sync_argv(uv, extras), network=network))
    else:
        wait = disk.Wait(wait_s, ask)
        _until_free(root, step, [n for n, _v in wanted], wait)
        _keep_edits(root, update_id, step, {n for n, _v in wanted}, wait)
        canon = shared.updater_components._canon
        own = [n for n, _v in wanted if canon(n) == packages._OWN]
        if own:
            commands.append([
                str(uv), "pip", "install", "--no-config", "--no-deps", "--reinstall", *offline,
                "--python", disk.short_path(python_in(venv)), "-e", disk.short_path(
                    disk.real_folder(root))])
        lines, registries, other, places = _requirements(
            root, [w for w in wanted if canon(w[0]) != packages._OWN])
        first = [a for n, r in enumerate(registries)
                 for a in ("--extra-index-url" if n else "--index-url", r)]
        for name, text, options in (("locked.txt", lines, ["--require-hashes", *first]),
                                    ("other.txt", other, [
                                        *indexes, *_own_indexes(other),
                                        *(a for r in places for a in ("--extra-index-url", r))])):
            if text:
                listing = work / name
                listing.write_text("".join(f"{line}\n" for line in text), encoding="utf-8")
                commands.append([
                    str(uv), "pip", "install", "--no-config", "--no-deps", "--reinstall",
                    "--index-strategy", "unsafe-best-match", *options, *offline, "--python",
                    disk.short_path(python_in(venv)), "-r", disk.short_path(listing)])
    for argv in commands:
        code, output = run(argv, str(root), packages._environment(env))
        if code != 0:
            named = ", ".join(f"{n} {v}" for n, v in wanted)
            raise Refused(f"{named} could not be installed again: {packages._last(output)}")


def _keep_edits(root, update_id: str, step: str, names: set, wait: disk.Wait) -> None:
    """Copy each file of the packages *names* that is neither the one its package lists
    nor one the update knows to the *kept* folder of the update's package step *step*.
    Raises :class:`.disk.Held` or OSError."""
    venv = disk.real_folder(Path(root) / mover.VENVS[step])
    site = shared.updater_components.site_packages(venv)
    canon = shared.updater_components._canon
    names = {canon(n) for n in names}
    found: dict = {}
    for name, folder in packages._infos(site, names) if site else ():
        for rel, row in packages._listed(site, venv, folder, name).items():
            found.setdefault(rel, []).append(packages._stated(row))
    plans = [p for p in mover.read_plans(root, update_id) if p.step == step] if (
        mover.has_plans(root, update_id)) else []
    known: dict = {}
    for plan in plans:
        for op in plan.ops:
            if op.dist in names:
                found.setdefault(op.path, [])
            known.setdefault(op.path, []).extend(
                i for i in (op.old, op.new, *op.release) if i is not None)
    if not plans:
        return
    a = mover.area(root, update_id, step, plans[-1].number)
    a.watch(found)
    for rel, stated in found.items():
        path, st = a.find(rel)
        if st is None or not stat.S_ISREG(st.st_mode) or mover.by_installer(rel):
            continue
        # A package that states nothing about its file does not tell an edit.
        said = [s for s in stated if s is not None]
        if (said or known.get(rel)) and not any(
                disk.same(path, i, wait) for i in (*said, *known.get(rel, ()))):
            target = mover.free(a, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            disk.copy_whole(path, target, wait)


def _until_free(root, step: str, names: Iterable, wait: disk.Wait) -> None:
    """Wait on Windows until no program holds a file of the packages *names*; one that
    only has it loaded does not. Raises :class:`.disk.Held`."""
    venv = disk.real_folder(Path(root) / mover.VENVS[step])
    site = shared.updater_components.site_packages(venv)
    if os.name != "nt" or site is None:
        return
    canon = shared.updater_components._canon
    for name, folder in packages._infos(site, {canon(n) for n in names}):
        try:
            listed = packages._records(site, venv, folder, name)
        except Refused:
            continue
        for rel in listed:
            path = venv.joinpath(*rel.split("/"))
            if disk.lstat(path) is not None:
                disk.patient(wait, path, partial(os.rename, path, path))


#: In the folder of an update of a component: the packages pip installs (one
#: ``name==version`` per line) and the wheels it installs them from.
PIP_LIST, PIP_WHEELS = "pip-install.txt", "wheels"


def run_pip(root, update_id: str, *, ask: Optional[disk.Ask] = None) -> None:
    """Move in the packages pip staged (:func:`.packages.note_staged`) for the venv of
    the component folder *root*. Raises like :func:`run_packages`."""
    plan = packages.plan_packages(root, update_id, mover.VENV)
    mover.write_plan(root, update_id, plan)
    packages.forward(root, update_id, plan, None, env={}, ask=ask)
