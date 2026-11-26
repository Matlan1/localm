# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installed version's part of an update: download and verify the build, extract it
into the update's folder, let the program the build ships check it, list what else of
this install runs, and hand the update over to that program (``update_mechanism.applier``).

:func:`prepare` changes nothing of the install. :func:`hand_off` is the last act of the
process that starts the update: the applier holds the update from then on, and this
process restarts (a server) or exits (``localm update``). A build that ships no applier
(an older release) is installed by :func:`localm.updater.apply` instead
(``Prepared.legacy``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from localm._update_parent import HANDED_OFF  # noqa: F401  (``localm update`` exits with it)

#: Seconds the applier has to say that it holds the update.
READY_S = 60.0
#: Seconds the check of the new version may take (it may download packages).
CHECK_S = 3600.0
#: Whether the applier of a build is there, relative to its folder.
APPLIER = Path("localm") / "update_mechanism" / "applier.py"


@dataclass
class Prepared:
    """An update that is prepared and not yet handed over; see :func:`prepare`."""
    root: Path
    id: str
    old: str
    new: str
    target_version: Optional[str]
    verified: bool
    verify_reason: Optional[str]
    downgrade: bool
    keyset: Optional[dict]
    legacy: bool = False
    #: The other processes of this install, ``{"key", "pid", "purpose"}``.
    processes: list = field(default_factory=list)
    #: Files the user edited that the update replaces.
    edited: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    #: ``{"host", "port", "scheme"}`` of the server that restarts into the update.
    watch: Optional[dict] = None
    _work: object = None

    def discard(self) -> None:
        """Delete what was prepared."""
        work, self._work = self._work, None
        if work is not None:
            work.__exit__(None, None, None)


def base_python() -> str:
    """The Python the install's environment is made of, outside the files an update
    replaces."""
    return getattr(sys, "_base_executable", None) or sys.executable


def _hidden() -> dict:
    return ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt"
            else {})


def _env(home) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALM_UPDATE_")}
    env["LOCALM_HOME"] = str(home)
    return env


def prepare(asset_id, *, signature=None, target_version=None, allow_unverified=False,
            expect_sha256=None, download_opener=None, installed=None,
            say: Optional[Callable[[str], None]] = None) -> Prepared:
    """Download the build *asset_id*, check its signature and version as
    :func:`localm.updater.apply` does, extract it into ``<root>/.localm-update/<id>/new``
    and run its ``check``. Nothing of the install changes. Raises LocalmError or
    ``updater.UnverifiedBuild``; with nothing left behind."""
    import zipfile

    from localm import _apply_update as au
    from localm import _version, updater
    from localm.bugreport import LocalmError
    from localm.update_mechanism import mover, shared, steps_update
    say = say or (lambda _t: None)
    root = Path(installed) if installed else updater.repo_root()
    if installed is None and updater.is_site_packages_install(root):
        raise LocalmError("this localm was installed with pip",
                          reason="run 'pip install --upgrade localm' instead")
    if installed is None:
        updater.refuse_in_container_image("update")
    if allow_unverified and not expect_sha256:
        raise LocalmError("refusing to install an unverified build without its sha256",
                          reason="pass the sha256 the verification warning named")
    update_id = uuid.uuid4().hex
    work = steps_update.Preparing(root, update_id)
    work.__enter__()
    try:
        zip_path, unpacked = work.path / "build.zip", work.path / "unpacked"
        keyset, verified, reason = updater._verified_download(
            asset_id, zip_path, signature=signature, target_version=target_version,
            allow_unverified=allow_unverified, expect_sha256=expect_sha256,
            opener=download_opener)
        try:
            au.verify_zip(zip_path)
            tree = au.extract(zip_path, unpacked)
        except (ValueError, zipfile.BadZipFile) as e:
            raise LocalmError("refusing to install: the download is not a localm build",
                              reason=str(e)) from None
        new_folder = mover.work_folder(root, update_id) / "new"
        os.replace(tree, new_folder)
        shutil.rmtree(unpacked, ignore_errors=True)
        zip_path.unlink()
        vf = new_folder / "VERSION"
        new_version = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
        if target_version is None:
            updater._refuse_downgrade(new_version)
        else:
            updater._check_selected_version(new_version, str(target_version))
        old_version = _version.read_version()
        out = Prepared(root, update_id, old_version, new_version, target_version, verified,
                       reason, _version.is_newer(old_version, new_version), keyset, _work=work)
        if not (new_folder / APPLIER).is_file():
            out.legacy = True
            return out
        program = shared.copy_applier(new_folder / "localm", work.path / "applier")
        say(f"Checking localm {new_version}...")
        code = _run_check(program, root, update_id, say)
        try:
            found = json.loads((work.path / "check.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            found = {"refused": f"the check of the new version ended with {code}"}
        if code != 0 or found.get("refused"):
            raise LocalmError("the update was not started; nothing was changed",
                              reason=str(found.get("refused") or f"its check ended with {code}"))
        out.edited, out.notes = list(found.get("edited") or ()), list(found.get("notes") or ())
        out.processes = other_processes(root)
        return out
    except BaseException:
        work.__exit__(None, None, None)
        raise


def _run_check(program: Path, root: Path, update_id: str, say: Callable[[str], None]) -> int:
    from localm import config
    proc = subprocess.Popen(
        [base_python(), "-I", "-S", str(program), "check", str(root), update_id],
        cwd=str(root), env=_env(config.HOME_DIR), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", **_hidden())
    assert proc.stderr is not None
    for line in proc.stderr:
        if line.strip():
            say(line.rstrip())
    try:
        return proc.wait(timeout=CHECK_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def _key(entry) -> str:
    return f"{entry.pid}-{entry.start}"


def other_processes(root) -> list:
    """The running processes of the install at *root* that an update ends: every recorded
    one but this process, those above it and those it started."""
    from localm import procs
    from localm.update_mechanism import disk
    me = procs.current()
    if me is None:
        return []
    running = procs.list(disk.short_path(disk.real_folder(root)))
    by_key = {(e.pid, e.start): e for e in running}
    mine = {(me.pid, me.start)}
    up = me.parent
    while up and tuple(up) not in mine:
        mine.add(tuple(up))
        entry = by_key.get(tuple(up))
        up = entry.parent if entry else None

    def is_mine(e) -> bool:
        seen = set()
        while e is not None and (e.pid, e.start) not in seen:
            seen.add((e.pid, e.start))
            if (e.pid, e.start) in mine or (e.parent and tuple(e.parent) == (me.pid, me.start)):
                return True
            e = by_key.get(tuple(e.parent)) if e.parent else None
        return False

    return [{"key": _key(e), "pid": e.pid, "purpose": e.purpose}
            for e in running if e.recorded and not is_mine(e)]


def describe(processes: list) -> str:
    """The processes an update ends, for the user."""
    return ", ".join(f"{p['purpose']} (pid {p['pid']})" for p in processes)


def hand_off(prepared: Prepared, *, mode: str, confirmed: list, watch: Optional[dict] = None,
             relaunch: Optional[list] = None, say: Optional[Callable[[str], None]] = None) -> None:
    """Start the applier of *prepared* and return once it holds the update (its record
    is written). *confirmed* are the keys of the processes the user agreed to end.
    *mode* ``server``: this server restarts next (*watch* is ``{"host", "port",
    "scheme"}``, *relaunch* its command line); ``command``: this process exits with
    :data:`HANDED_OFF`. Raises LocalmError with the update discarded."""
    from localm import config, procs, updater
    from localm.bugreport import LocalmError
    from localm.update_mechanism import shared
    say = say or (lambda _t: None)
    me = procs.current()
    try:
        if me is None:
            raise LocalmError("the update was not started", reason=procs.problem())
        if prepared.keyset is not None:
            updater._save_keyset(prepared.keyset)
        if prepared.target_version is not None:
            updater._carry_trust_forward()
            if prepared.downgrade:
                updater._backup_data_for_downgrade(prepared.new)
        handoff = {"mode": mode, "starter": _key(me), "confirmed": list(confirmed),
                   "home": str(config.HOME_DIR), "old": prepared.old, "new": prepared.new,
                   "watch": watch, "relaunch": relaunch}
        program = prepared.root / ".localm-update" / prepared.id / "applier" / shared._PACKAGE.name
        flags = _hidden() if mode == "server" else (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
            else {"start_new_session": True})
        proc = subprocess.Popen(
            [base_python(), "-I", "-S", str(program), "apply", str(prepared.root), prepared.id],
            cwd=str(prepared.root), env=_env(config.HOME_DIR), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL if mode == "server" else None,
            text=True, encoding="utf-8", errors="replace", **flags)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(handoff) + "\n")
        proc.stdin.close()
        answer: list = []
        reader = threading.Thread(target=lambda: answer.append(proc.stdout.readline()),
                                  daemon=True)
        reader.start()
        reader.join(READY_S)
        if not answer or answer[0].strip() != "ready":
            if proc.poll() is None:
                proc.kill()
            raise LocalmError("the update was not started; nothing was changed",
                              reason=f"its program did not take it over (exit {proc.poll()})")
        say(f"localm {prepared.new} is being installed.")
    except BaseException:
        prepared.discard()
        raise
    work, prepared._work = prepared._work, None
    if work is not None:
        work.recorded()


def note_failed(prepared: Prepared, text: str) -> None:
    """Write that the update *prepared* was not started, with *text*, where
    :func:`last_result` reads it."""
    from localm import updater
    from localm.update_mechanism import applier, disk
    try:
        disk.write_json(updater._updates_dir() / applier.RESULT, {
            "state": "old", "from": prepared.old, "to": prepared.new, "version": prepared.old,
            "note": text, "notes": [], "lines": [], "at": time.time()})
    except (OSError, disk.Unsafe, disk.Held) as e:
        from localm.debuglog import logger
        logger.warning("could not record that the update was not started: %s", e)
