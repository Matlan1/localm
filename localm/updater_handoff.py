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
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from localm.update_mechanism.steps_update import Preparing

from localm._update_parent import HANDED_OFF  # noqa: F401  (``localm update`` exits with it)

#: Seconds the applier has to say that it holds the update.
READY_S = 60.0
#: Seconds the check of the new version may take (it may download packages).
CHECK_S = 3600.0
#: Whether the applier of a build is there, relative to its folder.
APPLIER = Path("localm") / "update_mechanism" / "applier.py"
#: The purposes of the processes an update may end because they come back by themselves:
#: an MCP server is started again by its client (``localm mcp``, and one the coder runs).
COMES_BACK = frozenset({"mcp", "mcp server"})


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
    #: ``{key: {"url", "token"}}`` of the other servers of this install that restart into
    #: the update (:func:`restart_servers`); their entries in :attr:`processes` carry
    #: ``"restarts": True``.
    servers: dict = field(default_factory=dict)
    _work: Optional[Preparing] = None

    def restarting(self) -> list:
        """``[key, purpose]`` of each other server that restarts into the update."""
        return [[p["key"], p["purpose"]] for p in self.processes if p["key"] in self.servers]

    def ended(self) -> list:
        """The other processes the update ends: those that do not restart."""
        return [p for p in self.processes if p["key"] not in self.servers]

    def lost(self) -> list:
        """The other processes the update ends that do not come back by themselves:
        neither a server that restarts nor one of :data:`COMES_BACK`."""
        return [p for p in self.ended() if p["purpose"] not in COMES_BACK]

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
        try:
            program = shared.copy_applier(new_folder / "localm", work.path / "applier")
        except OSError as e:
            raise LocalmError("the update was not started; nothing was changed",
                              reason=f"the program of the new version could not be copied: {e}"
                              ) from None
        say(f"Checking localm {new_version}...")
        code = _run_check(program, root, update_id, say)
        try:
            found = json.loads((work.path / "check.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            found = {"refused": f"the check of the new version did not end within {CHECK_S:.0f} s"
                     if code == -1 else f"the check of the new version ended with {code}"}
        if code != 0 or found.get("refused"):
            raise LocalmError("the update was not started; nothing was changed",
                              reason=str(found.get("refused") or f"its check ended with {code}"))
        out.edited, out.notes = list(found.get("edited") or ()), list(found.get("notes") or ())
        return look_around(out)
    except BaseException:
        work.__exit__(None, None, None)
        raise


def look_around(prepared: Prepared) -> Prepared:
    """Fill in what else of the install runs now: :attr:`Prepared.processes` and
    :attr:`Prepared.servers`. Returns *prepared*."""
    prepared.processes = other_processes(prepared.root)
    prepared.servers = servers_of(prepared.processes)
    for p in prepared.processes:
        p["restarts"] = p["key"] in prepared.servers
    return prepared


def _run_check(program: Path, root: Path, update_id: str, say: Callable[[str], None]) -> int:
    from localm import config
    proc = subprocess.Popen(
        [base_python(), "-I", "-S", str(program), "check", str(root), update_id],
        cwd=str(root), env=_env(config.HOME_DIR), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", **_hidden())
    assert proc.stderr is not None

    stderr = proc.stderr

    def read() -> None:
        for line in stderr:
            if line.strip():
                say(line.rstrip())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=CHECK_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        say(f"The check of the new version did not end within {CHECK_S:.0f} s.")
        code = None
    reader.join(5)
    return code if code is not None else -1


#: In ``<data folder>/logs``: what the program of a build said while it carried out an
#: update for a server (``hand_off`` mode ``server``).
APPLIER_LOG = "update-applier.log"


def _applier_log():
    """:data:`APPLIER_LOG`, open for appending, when localm may write a diagnostic trace
    (``audit.diagnostics_allowed``); else None, as when it cannot be opened (logged)."""
    from localm.audit import diagnostics_allowed
    from localm.debuglog import logger, logs_dir
    if not diagnostics_allowed():
        return None
    try:
        logs_dir().mkdir(parents=True, exist_ok=True)
        return open(logs_dir() / APPLIER_LOG, "a", encoding="utf-8")
    except OSError as e:
        logger.warning("the program of the update logs nothing: %s", e)
        return None


def _key(entry) -> str:
    return f"{entry.pid}-{entry.start}"


def ancestors(root) -> list:
    """``<pid>-<start>`` of each running process of the install at *root* above this one
    that an update neither ends nor names (``ancestors`` of the hand-off,
    :func:`_spared`)."""
    return [_key(e) for e in _spared(_family(root)[1])]


def _spared(above: list) -> list:
    """Of the processes *above* this one, nearest first: the launcher that shows the
    window of the app (``applier.LAUNCHER``), the ``localm update`` this process was
    started by to wait for the update (``_update_parent``), and the programs without a
    file of their own. Every other one is listed by :func:`other_processes`."""
    from localm import _update_parent
    recorded = [e for e in above if e.recorded]
    waits = (recorded[0] if recorded and recorded[0].purpose == "update"
             and os.environ.get(_update_parent.WORK) == "1" else None)
    return [e for e in above if not e.recorded or e.purpose == "launcher" or e is waits]


def _family(root) -> tuple:
    """``(running, above)``: the running processes of the install at *root*, and those
    of them above this process."""
    from localm import procs
    from localm.update_mechanism import disk
    me = procs.current()
    if me is None:
        return [], []
    running = procs.list(disk.short_path(disk.real_folder(root)))
    by_key = {(e.pid, e.start): e for e in running}
    above, seen = [], {(me.pid, me.start)}
    up = me.parent
    while up and tuple(up) not in seen:
        seen.add(tuple(up))
        entry = by_key.get(tuple(up))
        if entry is None:
            break
        above.append(entry)
        up = entry.parent
    return running, above


def other_processes(root) -> list:
    """The running processes of the install at *root* that an update ends: every recorded
    one but this process, those above it that go on (:func:`ancestors`) and those it
    started."""
    from localm import procs
    me = procs.current()
    running, above = _family(root)
    if me is None:
        return []
    by_key = {(e.pid, e.start): e for e in running}
    skip = {(e.pid, e.start) for e in _spared(above)}

    def started_by_me(e) -> bool:
        seen = set()
        while e is not None and (e.pid, e.start) not in seen:
            seen.add((e.pid, e.start))
            if (e.pid, e.start) == (me.pid, me.start):
                return True
            e = by_key.get(tuple(e.parent)) if e.parent else None
        return False

    return [{"key": _key(e), "pid": e.pid, "purpose": e.purpose} for e in running
            if e.recorded and (e.pid, e.start) not in skip and not started_by_me(e)]


def servers_of(processes: list) -> dict:
    """``{key: {"url", "token"}}`` of the servers among *processes* (:func:`other_processes`)
    that this data folder's instance registry lists, that are proven to be that server
    and that listen on this machine: they can be asked to restart."""
    from localm import config, instances
    from localm.bindhost import is_own_address, self_connect_host, url_host
    by_pid = {p["pid"]: p["key"] for p in processes}
    found: dict = {}
    for entry in instances.list_entries(config.HOME_DIR):
        key = by_pid.get(entry.get("pid"))
        if key is None or instances.entry_process_state(entry)[0] != "same":
            continue
        dial = self_connect_host(entry.get("host"))
        if not is_own_address(dial) or not entry.get("port"):
            continue
        found[key] = {"url": f"{entry.get('scheme') or 'http'}://{url_host(dial)}:"
                             f"{entry['port']}/v1", "token": entry.get("token")}
    return found


def restart_servers(prepared: Prepared, say: Callable[[str], None]) -> list:
    """Ask each other server of *prepared* to restart (``POST /v1/server/restart``); it
    then waits for the update at the gate of ``localm/__init__.py`` and runs the new
    version after it. Returns the keys of those that refused or could not be asked; the
    applier ends them and names them, and so does one that does not restart."""
    if not prepared.servers:
        return []
    import requests

    from localm.selfclient import self_request
    refused = []
    for key, server in sorted(prepared.servers.items()):
        try:
            reply = self_request("POST", "/server/restart", base_url=server["url"], timeout=5,
                                 instance_token=server.get("token"))
            if reply.status_code != 200:
                refused.append(key)
                say(f"The server {server['url']} did not restart for the update "
                    f"({reply.status_code}); the update ends it.")
        except requests.RequestException as e:
            refused.append(key)
            say(f"The server {server['url']} could not be asked to restart for the update "
                f"({e}); the update ends it.")
    return refused


def _not_restarting(prepared: Prepared, keys: list) -> None:
    """Tell the applier of *prepared* not to wait for the servers *keys*
    (``applier.NOT_RESTARTING``)."""
    if not keys:
        return
    from localm.update_mechanism import applier, disk
    path = prepared.root / ".localm-update" / prepared.id / applier.NOT_RESTARTING
    try:
        disk.write_json(path, list(keys))
    except (OSError, disk.Unsafe, disk.Held) as e:
        from localm.debuglog import logger
        logger.warning("the update waits for servers that did not restart: %s", e)


def describe(processes: list) -> str:
    """The processes an update ends or restarts, for the user."""
    return ", ".join(f"{p['purpose']} (pid {p['pid']})"
                     + (", restarts" if p.get("restarts") else "") for p in processes)


def hand_off(prepared: Prepared, *, mode: str, confirmed: list, watch: Optional[dict] = None,
             relaunch: Optional[list] = None, ask: bool = False,
             say: Optional[Callable[[str], None]] = None) -> None:
    """Start the applier of *prepared* and return once it holds the update (its record
    is written) and the other servers were asked to restart (:func:`restart_servers`).
    *confirmed* are the keys of the processes the user agreed to end or restart.
    *mode* ``server``: this server restarts next (*watch* is ``{"host", "port",
    "scheme"}``, *relaunch* its command line); ``command``: this process exits with
    :data:`HANDED_OFF`; with *ask*, the user at its terminal is asked "Try again" or
    "Cancel" about a file another program holds or a process that does not end. Raises
    LocalmError with the update discarded."""
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
                   "watch": watch, "restarting": prepared.restarting(),
                   "ancestors": ancestors(prepared.root), "ask": bool(ask)}
        program = prepared.root / ".localm-update" / prepared.id / "applier" / shared._PACKAGE.name
        flags = _hidden() if mode == "server" else (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
            else {"start_new_session": True})
        log = _applier_log() if mode == "server" else None
        try:
            proc = procs.start(
                [base_python(), "-I", "-S", str(program), "apply", str(prepared.root),
                 prepared.id, *map(str, relaunch or ())],
                purpose="update", cwd=str(prepared.root), env=_env(config.HOME_DIR),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=(log or subprocess.DEVNULL) if mode == "server" else None,
                text=True, encoding="utf-8", errors="replace", **flags)
        finally:
            if log is not None:
                log.close()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(handoff) + "\n")
        proc.stdin.close()
        answer: list = []
        stdout = proc.stdout
        reader = threading.Thread(target=lambda: answer.append(stdout.readline()),
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
    _not_restarting(prepared, restart_servers(prepared, say))


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
