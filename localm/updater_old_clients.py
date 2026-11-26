# SPDX-License-Identifier: AGPL-3.0-or-later
"""The checks of the first start after an older localm installed this build with its
own updater.

localm 0.3.2 to 0.3.4b4 update with the code they shipped: they copy the install's
old top-level entries to ``<home>/updates/backup``, put the new ones in place, run
their package step and start the new build, watched by their own update watchdog.
The first start of this build after such an update: ``<home>/updates/backup/VERSION``
names another version than ``VERSION``, and ``<root>/.localm-update/checked.json``
(:data:`CHECKED`) does not name the identity of that file (:func:`backup_identity`),
which every backup an old client makes has anew, or names it with checks still open
(:func:`due`).

It runs the checks (:func:`run_checks`): every file the release statement
(``update-package.json``) states is there as stated, the venv holds this version's
install record and dependencies (``updater_parts.venv_part``), and the new version,
started as its own process, answers its version (``update_mechanism.steps_probe``).
While an update watchdog of this install may run (:func:`watchdogs`), the version is
not asked: that watchdog judges the update, and the outcome stays open (``passed``
None in :data:`CHECKED`) until it has ended. A server then waits for it in a thread
and finishes the checks; any other start leaves them to a later start. The native
runtime and the components are brought to this version's pins by its own start-time
steps; the checks run once none is pending.

- Passed (all three, with no update watchdog of this install running): :data:`CHECKED`
  records it, and the old backup (with its ``-runtime`` and ``-components`` folders
  and ``applied_names.json``) is removed when it is this install's (:func:`mine`), no
  update watchdog of this install runs (from 0.3.3 on that watchdog runs from the
  backup), and the install still is the version that passed (:func:`tidy`).
- Failed and :func:`mine`: the backup is put back (``updater.rollback_last``). The
  put-back is recorded before it starts, so a start that is stopped while it runs
  leaves it to the next start; one that changed nothing is tried again at the next
  start, :data:`PUT_BACK_TRIES` times in all.
- Failed and not :func:`mine`: nothing is put back; the reason and how to repair it
  are shown. The backup then stays until the user removes it (:func:`kept_backup`,
  :func:`remove_kept_backup`).

After a put-back the previous version starts with the command this process was
started with (:func:`go_on`): this process starts it in its console once it answers
its version. While an update watchdog of the install judges a server, that watchdog
puts the previous version back; one of localm 0.3.4b3 or b4 then starts it itself
(``--relaunch-argv``), and for one of an earlier version a starter process
(:func:`_start_after_watchdogs`, this file run with only the standard library) waits
for it to end and runs the same command again.

The outcome goes to the console (stderr), to the result of the last update
(``update_mechanism.applier.RESULT``), which this version's Settings shows, and after
a put-back to the last runtime and component update results, which the Settings of
localm 0.3.3 to 0.3.4b4 show.

The module level imports only the standard library: ``scripts/rollback_update.py``
loads this file by path and calls :func:`refusal`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

#: In ``<root>/.localm-update``: which backup the checks ran for, and how they ended.
CHECKED = "checked.json"
#: In ``<root>/.localm-update``: the target of the store lock (``storekit.store_lock``,
#: file ``checking.lock``) a start holds while it runs the checks.
CHECKING = "checking"
#: Seconds a start waits for another start's checks before it leaves them to it.
CHECKING_WAIT_S = 2.0
#: The entries of ``<home>/updates`` that belong to an old client's backup.
BACKUP_PARTS = ("backup", "backup-runtime", "backup-components", "applied_names.json")
#: Seconds a server waits for an old update watchdog to end before it leaves the rest to
#: a later start.
WATCHDOG_WAIT_S = 900.0
#: How often a put-back that changed nothing is tried, at one start each.
PUT_BACK_TRIES = 3
#: Seconds the backup an old client makes may be older than this install's release
#: statement, or newer, and still be the backup of the update that installed it
#: (:func:`tied`). Old clients extract the release with ``zipfile`` (each file gets the
#: time of the extraction), then copy the install's old entries to the backup.
TIED_BEFORE_S = 600.0
TIED_AFTER_S = 6 * 3600.0
#: The file an old client (0.3.3 and later) writes last into its backup.
BACKUP_RECORD = ".localm-backup.json"
_FILES_TRIES = 3
_WATCHDOG_POLL_S = 2.0
_WATCHDOG = "update_watchdog.py"
_REMOVING = "-removing-"

Say = Callable[[str], None]


class Later(Exception):
    """A check could not be made now; the checks run again at the next start."""


class Refused(Exception):
    """A removal of the old backup that was not made, with the reason for the user."""


class Ending(str):
    """The note :func:`at_start` returns when this process must not go on, with what
    starts next (:func:`go_on`) in ``then``: ``"previous"`` the checks put localm
    ``old`` back and this process starts it; ``"watchdog"`` an update watchdog of
    localm ``old`` puts it back and starts it itself; ``"after_watchdog"`` one puts it
    back and starts nothing, so a starter waits for the watchdogs ``pids`` (None when
    they cannot be told) and starts the same command; None nothing starts. ``new`` is
    the version that failed."""

    then: Optional[str]
    old: str
    new: str
    pids: Optional[list]

    def __new__(cls, note: str, *, then: Optional[str] = None, old: str = "",
                new: str = "", pids: Optional[list] = None):
        self = super().__new__(cls, note)
        self.then, self.old, self.new, self.pids = then, old, new, pids
        return self


# ---------------------------------------------------------------- reading

def _folder(root) -> Path:
    return Path(root) / ".localm-update"


def backup_identity(updates_dir) -> Optional[dict]:
    """``{"version", "size", "mtime_ns", "ctime_ns", "ino"}`` of
    ``<updates_dir>/backup/VERSION``, which an old client's updater writes anew with
    every backup; None when there is none that can be read."""
    path = Path(updates_dir) / "backup" / "VERSION"
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            text = f.read(256).decode("utf-8").strip()
    except (OSError, ValueError):
        return None
    if not text:
        return None
    return {"version": text, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "ino": st.st_ino}


def read_checked(root) -> dict:
    """What :data:`CHECKED` of the install at *root* holds, empty when it is not there
    or cannot be read."""
    try:
        with open(_folder(root) / CHECKED, "rb") as f:
            data = json.loads(f.read(64 * 1024).decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def _open(checked: dict) -> bool:
    """Whether the checks recorded in *checked* are not finished: the version is still
    to be confirmed (``passed`` None), or a put-back is owed."""
    return checked.get("passed") is None or checked.get("putting_back") is True


def due(root, updates_dir, version: str) -> Optional[dict]:
    """The identity of the old backup whose update the checks are due for at this start
    of localm *version* at *root*, None when none is due: no backup, one of *version*,
    or :data:`CHECKED` names that backup and its checks are finished (they ran for it,
    with *version* or with a version an update installed after it)."""
    identity = backup_identity(updates_dir)
    if identity is None or identity["version"] == version:
        return None
    checked = read_checked(root)
    if checked.get("backup") == identity and not _open(checked):
        return None
    return identity


def refusal(install_root, updates_dir) -> Optional[str]:
    """Why ``rollback.*`` does not put the old backup in *updates_dir* back over the
    install at *install_root*, None when it may: the checks of its update passed, or
    they failed and the backup was not this install's (:func:`mine`), so it was left
    alone on purpose. While the checks are not finished (an update watchdog of the old
    client still judges the update) the backup may be put back."""
    identity = backup_identity(updates_dir)
    if identity is None:
        return None
    checked = read_checked(install_root)
    if checked.get("backup") != identity:
        return None
    version = checked.get("version")
    if checked.get("passed") is True:
        return (f"it is the backup of an update to localm {version} that is complete "
                "(its checks passed)")
    if checked.get("passed") is False and checked.get("mine") is not True:
        return (f"the update to localm {version} failed its checks and this backup is not "
                "provably the one that update made of this install")
    return None


def provable(root, updates_dir) -> bool:
    """Whether the old backup in *updates_dir* is provably the one of the install at
    *root*: its ``backup-components/state.json`` names *root* and, when it names the
    version the backup holds, that version."""
    updates_dir = Path(updates_dir)
    identity = backup_identity(updates_dir)
    try:
        with open(updates_dir / "backup-components" / "state.json", "rb") as f:
            state = json.loads(f.read(4 * 1024 * 1024).decode("utf-8"))
        named = state["root"]
        kept_for = state.get("backup_version")
    except (OSError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
        return False
    if identity is None or not isinstance(named, str) or not named:
        return False
    if kept_for and kept_for != identity["version"]:
        return False
    return _same_folder(named, root)


def _made_at(path: Path) -> float:
    """When the file *path* was created: its birth time where the system keeps one,
    else its change time. Raises OSError."""
    st = os.stat(path)
    return float(getattr(st, "st_birthtime", None) or st.st_ctime)


def tied(root, updates_dir) -> bool:
    """Whether the old backup in *updates_dir* was made by the update that installed the
    release the install at *root* states: the backup (its :data:`BACKUP_RECORD`, else
    its ``VERSION``) was written between :data:`TIED_BEFORE_S` before and
    :data:`TIED_AFTER_S` after the install's ``update-package.json``, which an old
    client extracts just before it makes the backup. A backup left by an earlier
    update, under a release unzipped by hand (which keeps the release's own file
    times), is not."""
    updates_dir = Path(updates_dir)
    try:
        statement = os.stat(Path(root) / "update-package.json").st_mtime
        record = updates_dir / "backup" / BACKUP_RECORD
        made = (os.stat(record).st_mtime if record.is_file()
                else _made_at(updates_dir / "backup" / "VERSION"))
    except OSError:
        return False
    return -TIED_BEFORE_S <= made - statement <= TIED_AFTER_S


def mine(root, updates_dir) -> bool:
    """Whether the old backup in *updates_dir* is this install's and of the update that
    installed this version: :func:`provable` and :func:`tied`."""
    return provable(root, updates_dir) and tied(root, updates_dir)


def _same_folder(a, b) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def _installed_version(root) -> str:
    try:
        return (Path(root) / "VERSION").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


# ---------------------------------------------------------------- the checks

def run_checks(root, home, version: str, *, say: Say, answers: bool = True) -> list:
    """Run the checks of the install at *root* of localm *version* with the data folder
    *home*: ``[(check, reason)]`` for each one that failed (``files``, ``parts``,
    ``version``), empty when all passed. Without *answers* the version is not asked.
    Raises :class:`Later` when the files cannot be compared now."""
    failed = []
    say("Checking the files of this install against the release ...")
    why = _files(root, version)
    if why:
        failed.append(("files", why))
    say("Checking the installed packages ...")
    from localm import updater_parts
    problems = updater_parts.venv_part(Path(root), version)["problems"]
    if problems:
        failed.append(("parts", "; ".join(problems)))
    if not answers:
        return failed
    say(f"Checking that localm {version} starts and answers its version ...")
    why = _answers(root, home, version)
    if why:
        failed.append(("version", why))
    return failed


def _files(root, version: str) -> Optional[str]:
    """Why the files of the install at *root* are not those of the release *version*
    states, None when they are. A file held by another program, or one that cannot be
    read, is tried again :data:`_FILES_TRIES` times a second apart; then
    :class:`Later` is raised."""
    from localm.update_mechanism import disk, restoring
    stated, differ = "", []
    for attempt in range(_FILES_TRIES):
        try:
            stated, differ = restoring.stated_differences(root)
            break
        except (ValueError, disk.Unsafe) as e:
            return f"its files could not be compared with the release: {e}"
        except (OSError, disk.Held) as e:
            if attempt == _FILES_TRIES - 1:
                raise Later(f"its files could not be read: {e}") from e
            time.sleep(1.0)
    if not stated:
        return "it has no release statement (update-package.json) that can be read"
    if stated != version:
        return f"its release statement is of localm {stated}, not {version}"
    if differ:
        more = f" and {len(differ) - 3} more" if len(differ) > 3 else ""
        return ("these files are missing or differ from the release: "
                + ", ".join(differ[:3]) + more)
    return None


def _answers(root, home, version: str) -> Optional[str]:
    """Why localm *version* of the install at *root*, started as its own process, does
    not answer its version, None when it does."""
    from localm import updater_finish
    from localm.update_mechanism import steps_probe
    return steps_probe.probe(updater_finish.interpreter(root), root, version=version,
                             home=home, env=os.environ)


# ---------------------------------------------------------------- old watchdogs

def watchdogs(root, *, untold: Optional[list] = None) -> Optional[list]:
    """The pids of the update watchdogs of the install at *root* that run: an old
    client's ``scripts/update_watchdog.py`` whose ``--install-root`` is *root*. None
    when that cannot be told: the processes cannot be listed, or the command line of
    a Python interpreter (:func:`_interpreter`) cannot be read or parsed while it
    still runs: one that ended between the listing and the read is listed again, and
    left out when it is gone. A process whose program is known and is not such an
    interpreter is not read: an old watchdog runs in one. Elsewhere than Windows, a
    process whose program cannot be read is read from ``/proc`` when it can be. On a
    system without ``/proc`` (macOS) the command lines come from ``ps``
    (:func:`_ps_command_line`). *untold*, when given, receives the
    ``install_manifest.list_processes`` row of each process that made the answer
    None."""
    from localm import install_manifest
    rows = install_manifest.list_processes()
    if rows is None:
        return None
    own = os.getpid()
    found, unread = [], []
    for row in rows:
        if row[0] == own:
            continue
        seen = _watchdog_of(row, root)
        if seen is None:
            unread.append(row)
        elif seen:
            found.append(row[0])
    if not unread:
        return found
    again = install_manifest.list_processes()
    if again is None:
        return None
    now = {row[0]: row for row in again}
    unknown = []
    for row in unread:
        still = now.get(row[0])
        if still is None or still[3] != row[3]:
            continue
        seen = _watchdog_of(still, root)
        if seen is None:
            unknown.append(still)
        elif seen:
            found.append(row[0])
    if unknown:
        if untold is not None:
            untold.extend(unknown)
        return None
    return found


def _watchdog_of(row, root) -> Optional[bool]:
    """Whether the process of the :func:`install_manifest.list_processes` *row* is an
    update watchdog of the install at *root*; None when that cannot be told."""
    pid, exe = row[0], row[2]
    if exe and not _interpreter(exe):
        return False
    if _by_ps():
        if not exe:
            return False
        line = _ps_command_line(pid)
        if line is None:
            return None
        if _WATCHDOG not in line:
            return False
        named = _option_in_line(line, "--install-root")
    else:
        argv = _argv(pid)
        if argv is None:
            return None if exe else False
        if not any(os.path.basename(a) == _WATCHDOG for a in argv):
            return False
        named = _option(argv, "--install-root")
    if named is None:
        return None
    return _same_folder(named, root)


def _relaunches(pid: int) -> bool:
    """Whether the update watchdog *pid* starts the previous version itself after its
    rollback: localm 0.3.4b3 and b4 pass it ``--relaunch-argv``, earlier ones have no
    such option. False when its command line cannot be read."""
    if _by_ps():
        line = _ps_command_line(pid)
        return line is not None and " --relaunch-argv" in line
    argv = _argv(pid)
    return argv is not None and any(a == "--relaunch-argv" or a.startswith("--relaunch-argv=")
                                    for a in argv)


def _interpreter(exe: str) -> bool:
    """Whether the program *exe* is one an old client's update watchdog runs in: a
    Python interpreter (``python``, ``pythonw``, ``python3.12``, a macOS framework
    ``Python``) or localm's own app program ``LocaLM.exe``, which is one."""
    name = os.path.basename(exe).lower()
    return name.startswith("python") or name == "localm.exe"


def _by_ps() -> bool:
    """Whether command lines are read from ``ps``: not Windows, and no ``/proc``."""
    return sys.platform != "win32" and not os.path.isdir("/proc")


def _may_run(root, since: float) -> bool:
    """Whether an update watchdog of the install at *root* runs (:func:`watchdogs`) or,
    when that cannot be told, may still run: an old client's watchdog gives up after 90
    seconds, so none runs :data:`WATCHDOG_WAIT_S` seconds after *since*, the time the
    checks first ran for the backup."""
    running = watchdogs(root)
    if running is None:
        return time.time() - since < WATCHDOG_WAIT_S
    return bool(running)


def _argv(pid: int) -> Optional[list]:
    """The arguments process *pid* was started with, None when they cannot be read: on
    Windows its command line split as the C runtime splits it, elsewhere from
    ``/proc``."""
    from localm import install_manifest
    if sys.platform == "win32":
        line = install_manifest.command_line(pid)
        return None if line is None else _split_windows(line)
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    return parts or None


def _ps_command_line(pid: int) -> Optional[str]:
    """``ps -o command= -p <pid>`` (the arguments joined by spaces), None when ps is not
    there, does not answer or shows nothing for *pid*."""
    import subprocess
    from localm import sysprog
    ps = sysprog.find("ps")
    if not ps:
        return None
    try:
        res = subprocess.run([ps, "-o", "command=", "-p", str(int(pid))],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def _option_in_line(line: str, name: str) -> Optional[str]:
    """The value of the option *name* in the command line *line* as ``ps`` shows it (the
    arguments joined by spaces): the text after it up to the next `` --``."""
    m = re.search(r"(?:^| )" + re.escape(name) + r"[ =](.+?)(?= --|$)", line)
    return m.group(1).strip() if m else None


def _split_windows(line: str) -> Optional[list]:
    """*line* split into arguments by ``CommandLineToArgvW``; None when it fails or on
    any other platform than Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    count = ctypes.c_int(0)
    argv = shell32.CommandLineToArgvW(line, ctypes.byref(count))
    if not argv:
        return None
    try:
        return [argv[i] for i in range(count.value)]
    finally:
        kernel32.LocalFree(ctypes.cast(argv, wintypes.HLOCAL))


def _option(argv: list, name: str) -> Optional[str]:
    """The value of the option *name* in *argv* (``name value`` or ``name=value``), None
    when it is not there."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg[len(name) + 1:]
    return None


# ---------------------------------------------------------------- the old backup

def remove_backup(updates_dir, root=None) -> list:
    """Remove the old backup in *updates_dir* (:data:`BACKUP_PARTS`), the venv copies its
    components state names (``updater_components.discard``) and what an earlier removal
    left (:func:`sweep`). The backup folder goes last, renamed aside first, so a removal
    that stops leaves a backup that is whole or none. Returns what could not be
    removed, one line each; nothing of the backup folder is removed while anything
    else is left."""
    from localm import updater_components as uc
    updates_dir = Path(updates_dir)
    left: list = []
    snap = updates_dir / "backup-components"
    if snap.exists():
        state = uc._read_json(snap / uc.STATE_FILE) or {}
        copies = [uc._own_copy(info, snap) for info in (state.get("components") or {}).values()]
        uc.discard(snap)
        left.extend(str(c) for c in [*copies, snap] if c is not None and os.path.lexists(c))
    for name in ("backup-runtime", "applied_names.json"):
        left.extend(_remove(updates_dir / name))
    if left:
        return left
    backup = updates_dir / "backup"
    if backup.exists():
        try:
            os.rename(backup, updates_dir / f"backup{_REMOVING}{os.getpid()}-{time.time_ns()}")
        except OSError as e:
            return [f"{backup}: {e}"]
    return sweep(updates_dir, root)


def sweep(updates_dir, root=None) -> list:
    """Remove what a removal of an old backup in *updates_dir* that stopped left: the
    backup folders renamed aside, and, while no ``backup-components`` folder is
    there, the venv copies beside the venv of the install at *root* whose marker
    names that folder (``updater_components._own_copy``). Returns what could not be
    removed."""
    updates_dir = Path(updates_dir)
    left: list = []
    try:
        asides = [p for p in updates_dir.iterdir() if p.name.startswith(f"backup{_REMOVING}")]
    except FileNotFoundError:
        asides = []
    except OSError as e:
        return [f"{updates_dir}: {e}"]
    for path in asides:
        left.extend(_remove(path))
    snap = updates_dir / "backup-components"
    if root is None or os.path.lexists(snap):
        return left
    from localm import updater_components as uc
    venv = Path(root) / ".venv"
    for kind in ("", "deps"):
        copy = uc._venv_copy_path(venv, snap, kind)
        if os.path.lexists(copy) and uc._own_copy({"venv_copy": str(copy)}, snap) is not None:
            left.extend(_remove(copy))
    return left


def _remove(path: Path) -> list:
    """Remove the file or folder *path*; what could not be removed, at most three lines."""
    errors: list = []
    if not os.path.lexists(path):
        return errors
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=lambda _f, p, e: errors.append(f"{p}: {e}"))
    else:
        try:
            os.unlink(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    return errors[:3]


def _size(updates_dir: Path) -> int:
    """The bytes of the files of the old backup in *updates_dir* (:data:`BACKUP_PARTS`),
    links not followed; a file that cannot be read counts 0."""
    total = 0
    for name in BACKUP_PARTS:
        path = updates_dir / name
        if path.is_file() and not path.is_symlink():
            with contextlib.suppress(OSError):
                total += os.lstat(path).st_size
            continue
        for folder, _dirs, files in os.walk(path):
            for f in files:
                with contextlib.suppress(OSError):
                    total += os.lstat(os.path.join(folder, f)).st_size
    return total


def kept_backup(root, home) -> Optional[dict]:
    """``{"version", "bytes"}`` of the old backup in the data folder *home* that stays
    because it cannot be proven to be the install at *root*'s: the checks of its update
    ran and ended (:data:`CHECKED` names it and owes nothing), it was not this install's
    when they ran, and is not now (:func:`mine`). None when there is none."""
    updates_dir = Path(home) / "updates"
    identity = backup_identity(updates_dir)
    if identity is None:
        return None
    checked = read_checked(root)
    if (checked.get("backup") != identity or _open(checked) or checked.get("mine") is True
            or mine(root, updates_dir)):
        return None
    return {"version": identity["version"], "bytes": _size(updates_dir)}


def remove_kept_backup(root, home) -> list:
    """Remove the backup :func:`kept_backup` names, which only the user's explicit choice
    does, with :func:`remove_backup`, holding the updater's apply lock and the checks'
    lock (:func:`_checking`). Raises :class:`Refused` with the reason when there is no
    such backup, an update watchdog of the install at *root* runs or may run, an update
    of the install is in progress, an update or rollback runs, or a start is checking
    the last update. Records the removal in :data:`CHECKED`. Returns what could not be
    removed, one line each."""
    from localm import updater
    from localm.bugreport import LocalmError
    from localm.update_mechanism import job
    updates_dir = Path(home) / "updates"
    none = ("there is no backup of an earlier update that is kept because it cannot be "
            "proven to be this install's")
    if kept_backup(root, home) is None:
        raise Refused(none)
    checked = read_checked(root)
    if _may_run(root, _since(checked)):
        raise Refused("an update watchdog of this install may still run; try again once it "
                      "has ended")
    if job.record_path(root).exists() or job.locked(root):
        raise Refused("an update of this install is in progress; try again once it has "
                      "finished")
    try:
        with updater._apply_lock("remove the backup of an earlier update"), \
                _checking(root) as held:
            if not held:
                raise Refused("a start of localm is checking the last update; try again")
            if kept_backup(root, home) is None:
                raise Refused(none)
            left = remove_backup(updates_dir, root)
    except LocalmError as e:
        raise Refused(f"{e.summary} ({e.reason})") from None
    if not left:
        try:
            _record(root, str(checked.get("version")), checked["backup"],
                    passed=checked.get("passed"), removed=True,
                    reasons=checked.get("reasons") or [], mine=False)
        except OSError as e:
            from localm.debuglog import logger
            logger.warning("update: the removal of the old backup could not be recorded: %s",
                           e)
    return left


def _record(root, version: str, identity: dict, *, passed: Optional[bool], removed: bool,
            reasons: list, mine: bool = False, putting_back: bool = False,
            tries: int = 0) -> None:
    """Write :data:`CHECKED` of the install at *root*: *passed* True when every check
    passed, False when one failed, None while the version is still to be confirmed;
    *mine* that the backup was this install's (:func:`mine`) when the checks ran;
    *putting_back* that a put-back is owed, *tries* how often it was tried; ``since``
    keeps when the checks first ran for *identity*. Raises OSError."""
    folder = _folder(root)
    folder.mkdir(exist_ok=True)
    tmp = folder / f"{CHECKED}.tmp"
    now = time.time()
    before = read_checked(root)
    since = before.get("since") if before.get("backup") == identity else None
    data = {"version": version, "backup": identity, "passed": passed, "removed": removed,
            "mine": mine, "putting_back": putting_back, "tries": tries,
            "reasons": [list(r) for r in reasons],
            "since": since if isinstance(since, (int, float)) else now, "at": now}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, folder / CHECKED)


def _result(home, root, *, state: str, old: str, new: str, version: str, note: str) -> None:
    """Write the result of the last update (``update_mechanism.applier.RESULT``) in the
    data folder *home*, for the install at *root*: *version* is the version now in
    place. Raises OSError."""
    from localm.update_mechanism import applier
    updates = Path(home) / "updates"
    updates.mkdir(parents=True, exist_ok=True)
    result = {"state": state, "from": old, "to": new, "version": version, "note": note,
              "notes": [], "lines": [], "root": str(Path(root).resolve()), "at": time.time()}
    tmp = updates / f"{applier.RESULT}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, updates / applier.RESULT)


def _tell_old_version(home, root, *, old: str, note: str) -> Optional[str]:
    """Put *note* into the last runtime and component update results in the data folder
    *home* (``runtime-update.json``, ``components-update.json``), under localm *old*,
    where localm 0.3.3 to 0.3.4b4 show it in Settings and ``localm doctor`` reads it.
    Returns why it could not be written, None when it was."""
    updates = Path(home) / "updates"
    now = time.time()
    try:
        runtime = updates / "runtime-update.json"
        try:
            st = json.loads(runtime.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            st = None
        if isinstance(st, dict):
            runtime.write_text(json.dumps(dict(st, note=note, version=old, at=now)),
                               encoding="utf-8")
        (updates / "components-update.json").write_text(json.dumps(
            {"state": "rolled_back", "components": [], "note": note, "error": None,
             "root": str(Path(root).resolve()), "version": old, "at": now}),
            encoding="utf-8")
    except OSError as e:
        return str(e)
    return None


def tidy(root, updates_dir, *, warn: Say, wait_s: float = 0.0) -> Optional[bool]:
    """Remove the old backup in *updates_dir* when the checks passed with it, it was
    this install's when they ran (:data:`CHECKED` says so; an earlier removal that
    stopped may have removed that proof) or is now (:func:`mine`), no update watchdog of
    the install at *root* runs or may run, and the install still is the version that
    passed (its ``VERSION``, and every file its release statement states,
    ``restoring.release_of``); with *wait_s*, wait that long for such a watchdog to
    end. True when it was removed (and :data:`CHECKED` says so), False when it stays
    for a later start, None when there is nothing to remove."""
    identity = backup_identity(updates_dir)
    checked = read_checked(root)
    if identity is None or checked.get("passed") is not True or checked.get("backup") != identity:
        sweep(updates_dir)
        return None
    if checked.get("mine") is not True and not mine(root, updates_dir):
        return None
    version = str(checked.get("version"))
    if not _no_watchdog(root, wait_s, _since(checked)):
        return False
    from localm.update_mechanism import restoring
    if _installed_version(root) != version or restoring.release_of(root) != version:
        warn(f"The backup of the update to localm {version} is kept: the install is no "
             "longer that release as it was checked.")
        return False
    left = remove_backup(updates_dir, root)
    if left:
        warn(f"The backup of the update to localm {version} could not be removed "
             "completely; a later start tries again: " + "; ".join(left[:3]))
        return False
    try:
        _record(root, version, identity, passed=True, removed=True, reasons=[], mine=True)
    except OSError as e:
        warn(f"Could not record that the backup was removed: {e}")
    return True


def _since(checked: dict) -> float:
    since = checked.get("since")
    return float(since) if isinstance(since, (int, float)) else time.time()


def _no_watchdog(root, wait_s: float, since: float) -> bool:
    """Whether no update watchdog of the install at *root* runs or may run
    (:func:`_may_run` with *since*), waiting up to *wait_s* seconds for one to end."""
    deadline = time.monotonic() + wait_s
    while True:
        if not _may_run(root, since):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_WATCHDOG_POLL_S)


# ---------------------------------------------------------------- at a start

@contextlib.contextmanager
def _checking(root):
    """Hold the store lock (``storekit.store_lock``) on :data:`CHECKING` of the install
    at *root* for the block; yields False without it when another start, or another
    thread of this process, holds it (that one runs the checks). Raises OSError when
    the lock cannot be taken for another reason."""
    from localm import storekit
    folder = _folder(root)
    folder.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(storekit.store_lock(folder / CHECKING,
                                                    timeout=CHECKING_WAIT_S))
        except (storekit.ConfigLockTimeout, RuntimeError):
            yield False
            return
        yield True


def _how_to_repair(failed: list, version: str) -> str:
    kinds = {kind for kind, _why in failed}
    if kinds == {"parts"}:
        return "To repair the packages: localm update --repair."
    return (f"To repair it, unzip the localm {version} release over the install folder and "
            "run setup again.")


def at_start(*, root, home, version: str, say: Say, warn: Say, server: bool) -> Optional[str]:
    """At a start of localm *version* at *root* with the data folder *home*: run the
    checks when they are due (:func:`due`) and handle their outcome as the module
    docstring says, or remove an old backup that waits for it (:func:`tidy`; a *server*
    waits for the old watchdog in a thread and finishes open checks there). Returns the
    note for the user when this process must end because the checks failed and the old
    backup was put back, putting it back failed, or an old watchdog puts it back (an
    :class:`Ending` saying what starts next, for :func:`go_on`); else None. *say* gets each step,
    *warn* each problem; neither may raise."""
    updates_dir = Path(home) / "updates"
    identity = due(root, updates_dir, version)
    if identity is None:
        _tidy_now_or_later(root, home, version, warn=warn, server=server)
        return None
    if _steps_pending():
        return None
    try:
        with _checking(root) as held:
            if not held:
                return None
            note, wait = _check(root, home, version, identity, say=say, warn=warn,
                                server=server)
    except OSError as e:
        warn(f"The checks of the update to localm {version} could not start ({e}); they run "
             "at the next start.")
        return None
    if wait:
        _later(root, home, version, warn=warn)
    return note


def _check(root, home, version: str, identity: dict, *, say: Say, warn: Say,
           server: bool) -> tuple:
    """:func:`at_start` with the checks due for the backup *identity* and the lock held:
    ``(note, wait)``, *wait* True when a server is to finish the checks once the old
    update watchdog has ended (:func:`_later`)."""
    updates_dir = Path(home) / "updates"
    old = identity["version"]
    checked = read_checked(root)
    if checked.get("backup") == identity and checked.get("putting_back") is True:
        say(f"Putting localm {old} back, which an earlier start began ...")
        return _put_back(root, home, identity, old=old, new=version,
                         failed=[tuple(r) for r in checked.get("reasons") or []],
                         tries=int(checked.get("tries") or 0), say=say, warn=warn), False
    say(f"localm {old} installed localm {version}; checking the update ...")
    since = _since(checked) if checked.get("backup") == identity else time.time()
    watched = _may_run(root, since)
    try:
        failed = run_checks(root, home, version, say=say, answers=not watched)
    except Later as e:
        warn(f"The checks of the update to localm {version} could not finish ({e}); they "
             "run at the next start.")
        return None, False
    owned = mine(root, updates_dir)
    if watched:
        _keep_record(root, version, identity, warn, passed=None, reasons=failed, mine=owned)
        if failed:
            why = "; ".join(reason for _kind, reason in failed)
            pids = watchdogs(root)
            if not server:
                then, tail = None, (f"Start it again once the update watchdog of localm {old} "
                                    f"has put localm {old} back.")
            elif pids and any(_relaunches(pid) for pid in pids):
                then, tail = "watchdog", (f"The update watchdog of localm {old} puts localm "
                                          f"{old} back and starts it again with the same "
                                          "command.")
            else:
                then, tail = "after_watchdog", (f"The update watchdog of localm {old} puts "
                                                f"localm {old} back; then localm starts again "
                                                "by itself with the same command.")
            if then != "after_watchdog":
                tail += " If it does not, the next start of localm puts it back."
            note = f"The update to localm {version} failed its checks: {why}. {tail}"
            _write_result(home, root, warn, state="failed", old=old, new=version,
                          version=version, note=note)
            return Ending(note, then=then, old=old, new=version, pids=pids), False
        say(f"The files and packages are those of localm {version}; the update watchdog of "
            f"localm {old} checks that it answers, and the update is complete once it "
            "has ended.")
        return None, server
    if not failed:
        _keep_record(root, version, identity, warn, passed=True, reasons=[], mine=owned)
        say(f"The update from localm {old} to {version} is complete.")
        _tidy_now_or_later(root, home, version, warn=warn, server=server)
        return None, False
    if owned:
        return _put_back(root, home, identity, old=old, new=version, failed=failed, tries=0,
                         say=say, warn=warn), False
    why = "; ".join(reason for _kind, reason in failed)
    _keep_record(root, version, identity, warn, passed=False, reasons=failed, mine=False)
    note = (f"The update to localm {version} failed its checks: {why}. The backup of "
            f"localm {old} is not provably the one the update that installed localm "
            f"{version} made of this install, so nothing was put back. "
            + _how_to_repair(failed, version))
    warn(note)
    _write_result(home, root, warn, state="failed", old=old, new=version, version=version,
                  note=note)
    return None, False


def _keep_record(root, version, identity, warn: Say, **fields) -> None:
    try:
        _record(root, version, identity, removed=False, **fields)
    except OSError as e:
        warn(f"Could not record the checks of the update to localm {version}: {e}")


def _steps_pending() -> bool:
    """Whether a runtime or component step of the last update has not run yet."""
    from localm import updater_components, updater_runtime
    return (updater_runtime.read_pending() is not None
            or updater_components.read_pending() is not None)


def _tidy_now_or_later(root, home, version: str, *, warn: Say, server: bool) -> None:
    updates_dir = Path(home) / "updates"
    if tidy(root, updates_dir, warn=warn) is False and server:
        _later(root, home, version, warn=warn)


def _later(root, home, version: str, *, warn: Say) -> None:
    """In a server: in a thread, wait up to :data:`WATCHDOG_WAIT_S` seconds for the
    update watchdogs of the install at *root* to end (:func:`finish`)."""
    threading.Thread(target=finish, args=(root, home, version),
                     kwargs={"warn": warn, "wait_s": WATCHDOG_WAIT_S},
                     daemon=True, name="old-update-checks").start()


def finish(root, home, version: str, *, warn: Say, wait_s: float = 0.0) -> Optional[bool]:
    """Once no update watchdog of the install at *root* runs (waiting up to *wait_s*
    seconds): when the checks of the old backup's update are open with every check but
    the version passed, and the install still is localm *version*, ask the version
    now; then remove the backup when it may go (:func:`tidy`). A check that fails here
    is left to the next start, which puts the backup back. When the install is another
    version by then (the old watchdog put its version back under this process), says
    that this process must be started again. Returns :func:`tidy`'s answer, False when
    the watchdog did not end or the install is another version."""
    updates_dir = Path(home) / "updates"
    if not _no_watchdog(root, wait_s, _since(read_checked(root))):
        return False
    installed = _installed_version(root)
    if installed and installed != version:
        warn(f"localm {installed} is in place again: the update watchdog of the previous "
             f"version undid the update to localm {version} while this localm ran. Close "
             "localm and start it again.")
        return False
    identity = backup_identity(updates_dir)
    checked = read_checked(root)
    if (identity is not None and checked.get("backup") == identity
            and checked.get("passed") is None and not checked.get("putting_back")
            and not checked.get("reasons") and _installed_version(root) == version):
        try:
            with _checking(root) as held:
                if not held:
                    return False
                failed = run_checks(root, home, version, say=lambda _m: None)
                if failed:
                    warn(f"The update to localm {version} did not pass its checks after the "
                         "old version's update watchdog ended: "
                         + "; ".join(r for _k, r in failed)
                         + ". The next start of localm checks it again.")
                    return False
                _keep_record(root, version, identity, warn, passed=True, reasons=[],
                             mine=bool(checked.get("mine")))
        except (OSError, Later) as e:
            warn(f"The checks of the update to localm {version} could not finish ({e}); "
                 "they run at the next start.")
            return False
    return tidy(root, updates_dir, warn=warn)


def _put_back(root, home, identity: dict, *, old: str, new: str, failed: list, tries: int,
              say: Say, warn: Say) -> str:
    """Put the old backup back (``updater.rollback_last``) because the update from localm
    *old* to *new* failed with *failed*, after recording that the put-back is owed
    (try *tries* + 1); record the outcome and return the note for the user."""
    from localm import updater
    why = "its checks failed: " + "; ".join(reason for _kind, reason in failed)
    tries += 1
    _keep_record(root, new, identity, warn, passed=False, reasons=failed, mine=True,
                 putting_back=True, tries=tries)
    say(f"The update to localm {new} failed: {why}. Putting localm {old} back ...")
    try:
        updater.rollback_last(say=say)
    except Exception as e:
        detail = f"{getattr(e, 'summary', e)}: {getattr(e, 'reason', '') or ''}".rstrip(": ")
        if tries < PUT_BACK_TRIES:
            note = (f"The update to localm {new} failed ({why}) and could not be undone yet: "
                    f"{detail}. The next start of localm tries again.")
            _write_result(home, root, warn, state="failed", old=old, new=new, version=new,
                          note=note)
            return note
        _keep_record(root, new, identity, warn, passed=False, reasons=failed, mine=True,
                     putting_back=False, tries=tries)
        note = (f"The update to localm {new} failed ({why}) and could not be undone: "
                f"{detail}. localm doctor shows which parts disagree. "
                + _how_to_repair(failed, new))
        _write_result(home, root, warn, state="failed", old=old, new=new, version=new,
                      note=note)
        return note
    _keep_record(root, new, identity, warn, passed=False, reasons=failed, mine=True,
                 putting_back=False, tries=tries)
    note = (f"The update to localm {new} failed and was undone: {why}. localm {old} is in "
            "place again.")
    _write_result(home, root, warn, state="old", old=old, new=new, version=old, note=note)
    unwritten = _tell_old_version(home, root, old=old, note=note)
    if unwritten:
        warn(f"Could not record the reason where localm {old} shows it: {unwritten}")
    return Ending(note, then="previous", old=old, new=new)


def _write_result(home, root, warn: Say, **fields) -> None:
    try:
        _result(home, root, **fields)
    except OSError as e:
        warn(f"Could not record the result of the checks: {e}")


# ---------------------------------------------------------------- the version after
# Everything below imports only the standard library: it runs after the previous
# version's files are back, when ``localm`` on disk is that version's code.

#: Set on the process :func:`go_on` starts, to the version that failed its checks.
RELAUNCHED = "LOCALM_OLD_CLIENT_RELAUNCH"
#: The argument a starter (:func:`_start_after_watchdogs`) runs with.
_STARTER = "--start-after-watchdogs"
#: Seconds the previous version may take to answer ``--version``.
_VERSION_LIMIT_S = 180.0
#: Seconds the install must stay the previous version, while the watchdogs cannot be
#: told, before a starter takes their rollback as finished.
_SETTLED_S = 10.0
#: Variables of a restart that a new start of the previous version must not carry.
_RESTART_ENV = ("LOCALM_RESTART_IN_PROGRESS", "LOCALM_FINISH_UPDATE_AFTER_START",
                "LOCALM_UPDATE_STEPS_DONE", "LOCALM_UPDATE_ID")
_VERSION_LINE = re.compile(r"^localm, version (\S+)\s*$")


def go_on(note, *, root, argv: list, say: Say, warn: Say) -> int:
    """What this process does after :func:`at_start` returned *note*, and the exit
    status it ends with: with an :class:`Ending` whose ``then`` is ``"previous"``, start
    the previous version with *argv* (the arguments of this command,
    :func:`start_previous`); ``"after_watchdog"``, start a starter that runs *argv*
    again once the old update watchdogs have ended (:func:`_spawn_starter`), unless
    this process was itself started that way; otherwise 1."""
    then = getattr(note, "then", None)
    if then == "previous":
        return start_previous(root, old=note.old, new=note.new, argv=argv, say=say,
                              warn=warn)
    if then == "after_watchdog":
        if os.environ.get(RELAUNCHED):
            warn(f"Start localm again once localm {note.old} is in place.")
            return 1
        why = _spawn_starter(root, note, argv)
        if why:
            warn(f"localm could not arrange to start again by itself ({why}); start it "
                 f"again once the update watchdog of localm {note.old} has ended.")
    return 1


def _python(root) -> str:
    """The python of the install at *root*: its venv's, else the running one (as
    ``updater_finish.interpreter``)."""
    parts = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    own = Path(root).joinpath(".venv", *parts)
    return str(own) if own.is_file() else sys.executable


def _env(new: str) -> dict:
    """The environment of a new start after the update to *new* failed: this one without
    the variables of a restart (:data:`_RESTART_ENV`), with :data:`RELAUNCHED`."""
    env = {k: v for k, v in os.environ.items() if k not in _RESTART_ENV}
    env[RELAUNCHED] = new
    return env


def _has_console() -> bool:
    """Whether a program this process starts shares a console it can show: always
    outside Windows; on Windows, whether this process has a console window."""
    if os.name != "nt":
        return True
    import ctypes
    try:
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except (AttributeError, OSError):
        return False


def _flags() -> dict:
    """The creation flags of a program this process starts: none with a console to
    share, else no window (``CREATE_NO_WINDOW``) on Windows."""
    import subprocess
    if os.name == "nt" and not _has_console():
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _says_version(python: str, root, version: str, env: dict) -> Optional[str]:
    """Why ``localm --version`` of the install at *root*, run with *python*, does not
    name *version* within :data:`_VERSION_LIMIT_S` seconds, None when it does."""
    import subprocess
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        done = subprocess.run([python, "-m", "localm", "--version"], cwd=str(root), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=_VERSION_LIMIT_S,
                              **flags)
    except OSError as e:
        return f"it could not be started ({e})"
    except subprocess.TimeoutExpired:
        return f"it did not answer within {_VERSION_LIMIT_S:.0f} seconds"
    said = [m[1] for m in map(_VERSION_LINE.search, done.stdout.splitlines()) if m]
    if done.returncode == 0 and said == [version]:
        return None
    lines = [line.strip() for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    shown = said[0] if said else (lines[-1][-300:] if lines else f"exit status {done.returncode}")
    return f"it does not start as localm {version} ({shown})"


def _run(cmd: list, env: dict, cwd: str, warn: Say) -> int:
    """Run *cmd* in this console and return its exit status, with Ctrl+C left to it;
    1 when it cannot be started."""
    import signal
    import subprocess
    try:
        before = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        before = None
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, **_flags()).returncode
    except OSError as e:
        warn(f"localm could not be started ({e}).")
        return 1
    finally:
        if before is not None:
            signal.signal(signal.SIGINT, before)


def _repair_note(old: str) -> str:
    return (f"To repair it, unzip the localm {old} release over the install folder and run "
            "setup again.")


def start_previous(root, *, old: str, new: str, argv: list, say: Say, warn: Say) -> int:
    """Start localm *old*, which the checks put back in place of *new* at *root*, with
    the arguments *argv* in this console, once its ``VERSION`` names it and it answers
    ``--version`` (:func:`_says_version`); return its exit status. When it does not,
    say why and how to repair it and return 1."""
    installed = _installed_version(root)
    if installed != old:
        warn(f"localm {old} was put back, but the install is localm "
             f"{installed or '(no version that can be read)'} now, so it was not started. "
             + _repair_note(old))
        return 1
    env = _env(new)
    python = _python(root)
    why = _says_version(python, root, old, env)
    if why:
        warn(f"localm {old} is in place again, but {why}, so it was not started; its files "
             f"may not be whole. {_repair_note(old)}")
        return 1
    say(f"Starting localm {old} with the same command: localm {' '.join(argv)}".rstrip())
    return _run([python, "-m", "localm", *argv], env, os.getcwd(), warn)


def _port_of(argv: list) -> Optional[int]:
    """The port *argv* names with ``-p``/``--port``, None when it names none."""
    for i, arg in enumerate(argv):
        value = None
        if arg in ("-p", "--port") and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--port="):
            value = arg.split("=", 1)[1]
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _spawn_starter(root, note: Ending, argv: list) -> Optional[str]:
    """Start a starter (:func:`_start_after_watchdogs`, this file run by the install's
    python with only the standard library) in this console, which outlives this
    process; None when it started, else why not."""
    import subprocess
    payload = {"root": str(Path(root)), "old": note.old, "new": note.new,
               "pids": note.pids, "argv": list(argv), "cwd": os.getcwd(),
               "until": time.time() + WATCHDOG_WAIT_S, "port": _port_of(argv)}
    cmd = [_python(root), "-I", "-S", str(Path(__file__).resolve()), _STARTER,
           json.dumps(payload)]
    try:
        subprocess.Popen(cmd, cwd=os.getcwd(), env=_env(note.new), stdin=subprocess.DEVNULL,
                         close_fds=True, **_flags())
    except OSError as e:
        return str(e)
    return None


def _alive(pid: int) -> bool:
    """Whether process *pid* runs; True when that cannot be told."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _answers_on(port: int) -> bool:
    """Whether something answers ``GET /whoami`` on the loopback *port* (http or https)."""
    import ssl
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for scheme in ("http", "https"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                             urllib.request.HTTPSHandler(context=ctx))
        try:
            with opener.open(f"{scheme}://127.0.0.1:{int(port)}/whoami", timeout=3) as r:
                if r.status == 200:
                    return True
        except (OSError, ValueError):
            continue
    return False


def _wait_for_watchdogs(root: Path, old: str, pids: Optional[list], until: float, *,
                        sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.time) -> None:
    """Wait until the update watchdogs *pids* have ended or, when they cannot be told
    (None), until the install at *root* has been localm *old* for :data:`_SETTLED_S`
    seconds; at most until the time *until* of *clock*."""
    settled_at = None
    while clock() < until:
        if pids is not None:
            if not any(_alive(p) for p in pids):
                return
        elif _installed_version(root) == old:
            settled_at = clock() if settled_at is None else settled_at
            if clock() - settled_at >= _SETTLED_S:
                return
        else:
            settled_at = None
        sleep(_WATCHDOG_POLL_S)


def _start_after_watchdogs(payload: dict) -> int:
    """The starter :func:`_spawn_starter` runs: wait for the old update watchdogs
    (:func:`_wait_for_watchdogs`), then, unless something answers on the port the
    command names, run the command again with the install's python and return its
    exit status. When the watchdog put the previous version back, that version is
    checked to answer first (:func:`start_previous`); when it did not, this version
    starts again and its own checks put the previous version back and start it."""
    root, old, new = Path(payload["root"]), str(payload["old"]), str(payload["new"])
    argv = [str(a) for a in payload.get("argv") or []]
    pids = payload.get("pids")

    def say(message: str) -> None:
        print(f"localm: {message}", file=sys.stderr, flush=True)

    say(f"Waiting for the update watchdog of localm {old} to put localm {old} back ...")
    _wait_for_watchdogs(root, old, [int(p) for p in pids] if pids is not None else None,
                        float(payload["until"]))
    port = payload.get("port")
    if port and _answers_on(int(port)):
        say(f"localm already answers on port {port}, so it was not started again.")
        return 0
    os.chdir(payload.get("cwd") or str(root))
    installed = _installed_version(root)
    if installed == old:
        say(f"localm {old} is in place again: the update to localm {new} failed its checks.")
        return start_previous(root, old=old, new=new, argv=argv, say=say, warn=say)
    say(f"The update watchdog of localm {old} did not put localm {old} back; starting localm "
        f"{installed or new} again with the same command.")
    return _run([_python(root), "-m", "localm", *argv], _env(new), os.getcwd(), say)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == _STARTER:
        raise SystemExit(_start_after_watchdogs(json.loads(sys.argv[2])))
    raise SystemExit(2)
