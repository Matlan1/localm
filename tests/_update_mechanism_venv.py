# SPDX-License-Identifier: AGPL-3.0-or-later
"""A venv made by hand with an installer that works like uv, and an update of it that
stops at a chosen operation, for the tests of ``localm.update_mechanism``."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from pathlib import Path

from localm.update_mechanism import disk, job, mover, packages
from tests import _update_mechanism as um

ID = um.UPDATE_ID
SITE = "Lib/site-packages" if os.name == "nt" else "lib/python3.12/site-packages"
#: The packages of the tests: name, version, files.
ALPHA1 = ("alpha", "1.0", {"alpha/__init__.py": "V = 1", "alpha/data.txt": "one",
                           "alpha/old.py": "x = 1", "alpha/same.py": "same"})
ALPHA2 = ("alpha", "2.0", {"alpha/__init__.py": "V = 2", "alpha/data.txt": "two",
                           "alpha/new.py": "x = 2", "alpha/same.py": "same"})
BETA = ("beta", "1.0", {"beta.py": "V = 1"})
GAMMA = ("gamma", "1.0", {"gamma.py": "V = 1"})


def venv(root: Path) -> Path:
    """A new venv without packages at *root*."""
    path = root / ".venv"
    (path / SITE).mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    return path


def install(where: Path, package: tuple) -> None:
    """Install *package* into the venv *where*, in place of another version of it."""
    name, version, files = package
    uninstall(where, name)
    site = where / SITE
    info = f"{name}-{version}.dist-info"
    content = {**files, f"{info}/METADATA": f"Name: {name}\nVersion: {version}\n",
               f"{info}/INSTALLER": "hand"}
    rows = []
    for rel, text in content.items():
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        path.write_bytes(text.encode())
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).decode()
        rows.append(f"{rel},sha256={digest.rstrip('=')},{len(text)}\n")
    (site / info / "RECORD").write_bytes("".join([*rows, f"{info}/RECORD,,\n"]).encode())


def uninstall(where: Path, name: str) -> None:
    """Take every installed version of *name* out, as far as its list tells."""
    site = where / SITE
    for info in list(site.glob(f"{name}-*.dist-info")):
        if not (info / "METADATA").exists():
            continue
        if (info / "RECORD").exists():
            for row in (info / "RECORD").read_text(encoding="utf-8").splitlines():
                path = site / row.split(",")[0]
                path.unlink(missing_ok=True)
                disk.prune(path.parent, site)
        shutil.rmtree(info, ignore_errors=True)


def stage(root: Path, leaving: list, coming: list, update_id: str = ID) -> None:
    """What ``stage_packages`` leaves: *leaving* are names, *coming* packages."""
    made = packages.stage_folder(root, update_id)
    site = made / SITE
    site.mkdir(parents=True)
    (made / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    base = packages.listing(made)
    for package in coming:
        install(made, package)
    have = packages._infos(root / ".venv" / SITE, leaving)
    packages._Staged(set(have), {p[0] for p in coming}, base).write(made.parent / "staged.json")


def installed(root: Path) -> Path:
    """An install with release files and a venv with alpha 1.0 and beta."""
    um.install(root)
    for package in (ALPHA1, BETA):
        install(root / ".venv" if (root / ".venv").exists() else venv(root), package)
    return root


def nothing_to_do(_argv, _cwd, _env) -> tuple:
    return 0, "Would make no changes"


def package_step(root: Path, coming=(ALPHA2, GAMMA), leaving=("alpha",)) -> mover.Plan:
    """Stage, plan and write the plan of a package step."""
    if disk.lstat(packages.stage_folder(root, ID)) is None:
        stage(root, list(leaving), list(coming))
    plan = packages.plan_packages(root, ID)
    mover.write_plan(root, ID, plan)
    return plan


def forward(root: Path, plan: mover.Plan, **kw) -> None:
    packages.forward(root, ID, plan, ["uv", "sync"], env={}, run=nothing_to_do, **kw)


def update(root: Path, *, stop_before: str = "") -> None:
    """A whole update of *root*, whose new release :func:`um.stage` extracted: files,
    packages, finish. *stop_before*: ``finish``."""
    files = um.stated({**um.NEW, **um.INTRUDING})
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        plan = mover.plan_files(root, ID, files, um.stated(um.OLD), swap=um.SWAP,
                                protect=["mydata"])
        mover.write_plan(root, ID, plan)
        mover.forward(root, ID, plan)
        job.advance(lock, job.STEPS)
        forward(root, package_step(root))
        job.advance(lock, job.STARTING)
        if stop_before != "finish":
            job.finish(lock)


def begun(root: Path) -> None:
    """An update in phase STEPS that has changed nothing."""
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)


def undo(root: Path, **kw) -> job.Result:
    with job.Lock(root) as lock:
        return job.restore(lock, **kw)


class Killed(BaseException):
    pass


class Points:
    """Every call that changes the disk is a point; from point number *at* on nothing
    changes any more, as after the end of the process. A context manager."""

    CALLS = ("rename", "replace", "unlink", "mkdir", "rmdir", "fsync", "write")

    def __init__(self, at: int = -1, end=None) -> None:
        self.at, self.count, self.dead, self.end = at, 0, False, end
        self.real: list = []

    def _wrap(self, owner, name: str) -> None:
        real = getattr(owner, name)
        self.real.append((owner, name, real))

        def call(*args, **kw):
            if self.count == self.at:
                if self.end is not None:
                    self.end()
                self.dead = True
            if self.dead:
                raise Killed
            self.count += 1
            return real(*args, **kw)
        setattr(owner, name, call)

    def __enter__(self) -> Points:
        for name in self.CALLS:
            self._wrap(os, name)
        self._wrap(json, "dump")
        self._wrap(disk, "replace_through")
        return self

    def __exit__(self, *exc) -> bool:
        for owner, name, real in self.real:
            setattr(owner, name, real)
        return exc[0] is Killed


def tree(root: Path) -> dict:
    """Files and folders of *root* besides the update's own and its kept files."""
    return um.tree(root, skip=(".localm-update", ".localm-run", ".venv/.lock",
                               ".venv/.localm-old", ".venv/.localm-kept"))
