# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replacing Python packages in the venv of the install, with uv, so that the way
back needs neither uv nor a package source.

The files of exactly the packages uv would remove are renamed into a folder inside
the venv, then uv installs the new ones. The way back renames them again. What uv
would do is read from its own dry run, and what it did is compared with a listing of
the whole venv taken before: a change nobody planned is a failure.
"""

from __future__ import annotations

import contextlib
import os
import posixpath
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional

from . import disk, mover, shared
from .disk import Held, Unsafe, Wait
from .mover import PACKAGES, REMOVE, Area, Op, Plan, Refused, StepFailed

#: Seconds the way back waits for a uv that is still working in the venv.
UV_WAIT_S = 120.0
_OWN = "localm"
_LINE = re.compile(r" ([-+~]) ([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)(?: \(.*\))?")
_COUNT = re.compile(r"Would (uninstall|install) (\d+) packages?")
_OTHER_VENV = re.compile(r"Would (?:replace|create|remove)\b.*\benvironment\b")

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


def command(argv: Iterable, *, project: bool = False) -> list:
    """The uv command *argv* (``<uv> sync ...`` or ``<uv> pip install ...``) as this
    step runs it: offline, and a sync without the project unless *project*."""
    argv = [str(a) for a in argv]
    extra = ["--offline"]
    if argv[1:2] == ["sync"] and not project:
        extra.append("--no-install-project")
    return [*argv, *(e for e in extra if e not in argv)]


def _environment(env: Mapping) -> dict:
    out = {k: v for k, v in env.items()
           if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "FORCE_COLOR", "CLICOLOR_FORCE")}
    out["NO_COLOR"] = "1"
    return out


def _last(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[-3:])[-600:] or "no output"


def changes(output: str) -> tuple:
    """``(removed, installed)`` from the output of a uv dry run, each a list of
    ``(canonical name, version)``. Raises :class:`Refused` when uv would make or
    replace a venv, or when its counts do not match the packages it names."""
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
            entry = (canon(found[2]), found[3])
            if found[1] in "-~":
                removed.append(entry)
            if found[1] in "+~":
                installed.append(entry)
    if len(removed) != counts["uninstall"] or len(installed) != counts["install"]:
        raise Refused(f"what uv would do could not be read: {_last(output)}")
    return removed, installed


def _record_paths(site: Path, base: Path, folder: str, name: str) -> dict:
    """``{path in the venv: stated size or None}`` of the files of the distribution
    *folder* in *site*: its RECORD and everything in its own folder, compiled Python
    files excepted. Raises :class:`Refused` without a RECORD, and for a path outside
    the venv."""
    inner = site.relative_to(base).as_posix()
    try:
        rows = shared.updater_components.record_rows(site, folder)
    except (OSError, ValueError) as e:
        raise Refused(f"the package {name} has no list of its files (RECORD), so it cannot be "
                      f"replaced safely: {e}") from None
    if not rows:
        raise Refused(f"the package {name} has an empty list of its files (RECORD)")
    found: dict = {}
    for row in rows:
        text = row[0].replace("\\", "/") if os.name == "nt" else row[0]
        rel = posixpath.normpath(posixpath.join(inner, text))
        if posixpath.isabs(text) or rel == ".." or rel.startswith("../") or ":" in rel[:2]:
            raise Refused(f"the package {name} lists a file outside the Python environment: "
                          f"{row[0][:200]}")
        if disk.PYCACHE not in rel.split("/"):
            size = row[2] if len(row) > 2 else ""
            found[rel] = int(size) if size.isdigit() else None
    for entry in os.listdir(site / folder):
        st = os.lstat(site / folder / entry)
        if disk.is_link(st) or os.path.isfile(site / folder / entry):
            found.setdefault(f"{inner}/{folder}/{entry}", None)
    return found


def plan_packages(root, update_id: str, argv: Iterable, *, env: Mapping, project: bool = False,
                  run: Runner = run_hidden) -> Optional[Plan]:
    """The plan of the package step that runs the uv command *argv*
    (:func:`command`) in *root*, or None when uv has nothing to do. Changes nothing.

    Raises :class:`Refused`, naming the package, for one without a RECORD, with a
    file outside the venv, or for localm itself without *project*; also when uv
    cannot make its plan. Raises OSError."""
    venv = mover.venv_of(root)
    code, output = run([*command(argv, project=project), "--dry-run"], str(root),
                       _environment(env))
    if code != 0:
        raise Refused(f"uv cannot prepare the packages: {_last(output)}")
    removed, installed = changes(output)
    if not removed and not installed:
        return None
    site = shared.updater_components.site_packages(venv)
    if site is None:
        raise Refused(f"{venv} has no site-packages")
    if not project and any(name == _OWN for name, _v in removed + installed):
        raise Refused("uv would replace localm itself")
    before = mover.listing(venv)
    infos = shared.updater_components.dist_infos(site)
    paths: dict = {}
    for name in sorted({name for name, _v in removed}):
        have = [folder for n, _v, folder in infos if n == name]
        if not have or len(have) != sum(1 for n, _v in removed if n == name):
            raise Refused(f"the package {name} has no list of its files (RECORD), so it cannot "
                          "be replaced safely")
        for folder in have:
            paths.update(_record_paths(site, venv, folder, name))
    ops = [Op(rel, REMOVE, old=before["files"][rel])
           for rel in sorted(paths) if rel in before["files"]]
    return Plan(PACKAGES, mover.next_number(root, update_id), ops, [], before)


def forward(root, update_id: str, plan: Plan, argv: Iterable, *, env: Mapping,
            project: bool = False, run: Runner = run_hidden,
            wait_s: Optional[float] = None) -> None:
    """Carry out *plan*, a plan of :func:`plan_packages` for the same arguments that
    is already on disk: move the files aside, run uv, compare the venv with what was
    planned. Raises :class:`StepFailed`, with everything put back, when any of it
    fails."""
    try:
        a = mover.area(root, update_id, PACKAGES)
        site = shared.updater_components.site_packages(a.base)
        if site is None:
            raise Refused(f"{a.base} has no site-packages")
        wait = Wait(wait_s)
        for op in plan.ops:
            mover.take_out(a, op, wait)
        for parent in sorted({a.base.joinpath(*op.path.split("/")).parent for op in plan.ops},
                             key=lambda p: len(p.parts), reverse=True):
            disk.prune(parent, site if site in parent.parents else parent)
        code, output = run(_relocated(a, plan, command(argv, project=project)), str(root),
                           _environment(env))
        if code != 0:
            raise Refused(f"uv could not install the packages: {_last(output)}")
        added = _verify(a, site, plan)
        disk.flush_files(a.base.joinpath(*rel.split("/")) for rel in added)
    except (OSError, Held, Unsafe, Refused) as e:
        raise StepFailed(str(e), back(root, update_id, plan, wait_s=wait_s)) from e


def _relocated(a: Area, plan: Plan, argv: list) -> list:
    """*argv* with its program at the place it was moved to, when it is one of the
    files *plan* moved."""
    program = os.path.normcase(disk.real_folder(argv[0]))
    for op in plan.ops:
        if os.path.normcase(a.base.joinpath(*op.path.split("/"))) == program:
            return [disk.short_path(a.beside(a.old, op.path)[0]), *argv[1:]]
    return argv


def _verify(a: Area, site: Path, plan: Plan) -> list:
    """The files uv added. Raises :class:`Refused` when uv changed, removed or added
    a file it was not planned to, or left a package incomplete."""
    before = (plan.before or {})["files"]
    moved = {op.path for op in plan.ops}
    allowed: dict = {}
    inner = site.relative_to(a.base).as_posix()
    for name, _version, folder in shared.updater_components.dist_infos(site):
        metadata = f"{inner}/{folder}/METADATA"
        if metadata not in before or metadata in moved:
            allowed.update(_record_paths(site, a.base, folder, name))
    now = mover.listing(a.base)["files"]
    wrong = [f"{rel} was removed" for rel in before if rel not in now and rel not in moved]
    wrong += [f"{rel} was changed" for rel in before
              if rel in now and rel not in moved and now[rel] != before[rel]]
    added = [rel for rel in now if rel not in before or rel in moved]
    wrong += [f"{rel} was added" for rel in added if rel not in allowed]
    for rel, size in allowed.items():
        if rel not in now:
            wrong.append(f"{rel} is missing")
        elif size is not None and now[rel].get("size", size) != size:
            wrong.append(f"{rel} is incomplete")
    if wrong:
        raise Refused("uv did not do what it planned: " + "; ".join(sorted(wrong)[:5]))
    return added


def back(root, update_id: str, plan: Plan, *, wait_s: Optional[float] = None,
         uv_wait_s: float = UV_WAIT_S) -> list:
    """:func:`mover.back` for a package step, once no uv works in the venv any
    more. Returns one line per file that is not as it was before the step, or the
    reason when a uv was still working after *uv_wait_s* seconds."""
    try:
        venv = mover.venv_of(root)
        with _uv_lock(venv, uv_wait_s):
            return mover.back(root, update_id, plan, wait_s=wait_s)
    except (OSError, Unsafe, Refused) as e:
        return [str(e)]


@contextlib.contextmanager
def _uv_lock(venv: Path, wait_s: float) -> Iterator[None]:
    """Hold the lock uv takes on *venv* while it changes it. Raises
    :class:`Refused` when uv still has it after *wait_s* seconds."""
    fd = disk.open_lock(venv / ".lock")
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                disk.lock_file(fd)
                break
            except (PermissionError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise Refused(f"a package installation is still running in {venv}") from None
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)
