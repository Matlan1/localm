# SPDX-License-Identifier: AGPL-3.0-or-later
"""The checks of the first start after an older localm installed this build with its
own updater.

localm 0.3.2 to 0.3.4b4 update with the code they shipped: they copy the install's
old top-level entries to ``<home>/updates/backup``, put the new ones in place, run
their package step and start the new build, watched by their own update watchdog.
The first start of this build after such an update (:func:`due`):
``<home>/updates/backup/VERSION`` names another version than ``VERSION``, and
``<root>/.localm-update/checked.json`` (:data:`CHECKED`) does not name the identity
of that file (:func:`backup_identity`), which every backup an old client makes has
anew.

It runs, once, the checks (:func:`run_checks`): every file the release statement
(``update-package.json``) states is there as stated, the venv holds this version's
install record and dependencies (``updater_parts.venv_part``), and the new version,
started as its own process, answers its version (``update_mechanism.steps_probe``); in
a server that the old client's update watchdog waits for, that watchdog asks it.
The native runtime and the components are brought to this version's pins by its own
start-time steps; the checks run once none is pending.

- Passed: :data:`CHECKED` records it, and the old backup (with its ``-runtime`` and
  ``-components`` folders and ``applied_names.json``) is removed when it is provably
  this install's (:func:`provable`) and no update watchdog of this install runs
  (:func:`watchdogs`): from 0.3.3 on that watchdog runs from the backup.
- Failed and provable: the backup is put back (``updater.rollback_last``).
- Failed and not provable: nothing is put back; the reason and how to repair it are
  shown.

Either way the outcome goes to the result of the last update
(``update_mechanism.applier.RESULT``), which Settings shows, and to the console.

The module level imports only the standard library: ``scripts/rollback_update.py``
loads this file by path and calls :func:`passed_backup`.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

#: In ``<root>/.localm-update``: which backup the checks ran for, and how they ended.
CHECKED = "checked.json"
#: In ``<root>/.localm-update``: held while a start runs the checks.
CHECKING = "checking"
#: Seconds after which a :data:`CHECKING` file is taken to be left by a start that ended.
CHECKING_STALE_S = 900.0
#: The entries of ``<home>/updates`` that belong to an old client's backup.
BACKUP_PARTS = ("backup", "backup-runtime", "backup-components", "applied_names.json")
#: Seconds a server waits for an old update watchdog to end before it leaves the backup
#: to a later start.
WATCHDOG_WAIT_S = 900.0
_WATCHDOG_POLL_S = 2.0
_WATCHDOG = "update_watchdog.py"
_REMOVING = "-removing-"

Say = Callable[[str], None]


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


def due(root, updates_dir, version: str) -> Optional[dict]:
    """The identity of the old backup whose update the checks are due for at this start
    of localm *version* at *root*, None when none is due: no backup, one of *version*,
    or :data:`CHECKED` names that backup (the checks ran for it, with *version* or with
    a version an update installed after it)."""
    identity = backup_identity(updates_dir)
    if identity is None or identity["version"] == version:
        return None
    return None if read_checked(root).get("backup") == identity else identity


def passed_backup(install_root, updates_dir) -> Optional[str]:
    """The version of this install whose checks passed with the old backup in
    *updates_dir*, None when they did not or did not run for it. That backup belongs to
    an update that is complete and is not put back by ``rollback.*``."""
    identity = backup_identity(updates_dir)
    if identity is None:
        return None
    checked = read_checked(install_root)
    if checked.get("passed") is True and checked.get("backup") == identity:
        return str(checked.get("version"))
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


def _same_folder(a, b) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- the checks

def run_checks(root, home, version: str, *, say: Say, answers: bool = True) -> list:
    """Run the checks of the install at *root* of localm *version* with the data folder
    *home*: ``[(check, reason)]`` for each one that failed (``files``, ``parts``,
    ``version``), empty when all passed. Without *answers* the version is not asked
    here: an old client's update watchdog asks this server for it."""
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
        say(f"The update watchdog of the old version checks that localm {version} answers.")
        return failed
    say(f"Checking that localm {version} starts and answers its version ...")
    why = _answers(root, home, version)
    if why:
        failed.append(("version", why))
    return failed


def _files(root, version: str) -> Optional[str]:
    """Why the files of the install at *root* are not those of the release *version*
    states, None when they are."""
    from localm.update_mechanism import disk, restoring
    try:
        stated, differ = restoring.stated_differences(root)
    except (OSError, ValueError, disk.Unsafe, disk.Held) as e:
        return f"its files could not be compared with the release: {e}"
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


# ---------------------------------------------------------------- the old backup

def watchdogs(root) -> Optional[list]:
    """The pids of the update watchdogs of the install at *root* that run (an old
    client's ``scripts/update_watchdog.py`` started with ``--install-root`` naming it),
    None when the processes cannot be listed."""
    from localm import install_manifest
    rows = install_manifest.list_processes()
    if rows is None:
        return None
    own = os.getpid()
    here = os.path.normcase(str(Path(root).resolve()))
    found = []
    for row in rows:
        pid = row[0]
        if pid == own:
            continue
        line = install_manifest.command_line(pid)
        if line and _WATCHDOG in line and here in os.path.normcase(line):
            found.append(pid)
    return found


def remove_backup(updates_dir) -> list:
    """Remove the old backup in *updates_dir* (:data:`BACKUP_PARTS`), the venv copies its
    components state names (``updater_components.discard``) and what an earlier removal
    left. The backup folder goes last, renamed aside first, so a removal that stops
    leaves a backup that is whole or none. Returns what could not be removed, one line
    each; nothing of the backup folder is removed while anything else is left."""
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
    try:
        asides = [p for p in updates_dir.iterdir() if p.name.startswith(f"backup{_REMOVING}")]
    except OSError as e:
        return [f"{updates_dir}: {e}"]
    for path in asides:
        left.extend(_remove(path))
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


def _record(root, version: str, identity: dict, *, passed: bool, removed: bool,
            reasons: list, mine: bool = False) -> None:
    """Write :data:`CHECKED` of the install at *root*; *mine* says that the backup was
    provably the install's (:func:`provable`) when the checks ran. Raises OSError."""
    folder = _folder(root)
    folder.mkdir(exist_ok=True)
    tmp = folder / f"{CHECKED}.tmp"
    data = {"version": version, "backup": identity, "passed": passed, "removed": removed,
            "mine": mine,
            "reasons": [list(r) for r in reasons], "at": time.time()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, folder / CHECKED)


def _result(home, *, state: str, old: str, new: str, note: str, notes: list) -> None:
    """Write the result of the last update (``update_mechanism.applier.RESULT``) in the
    data folder *home*. Raises OSError."""
    from localm.update_mechanism import applier
    updates = Path(home) / "updates"
    updates.mkdir(parents=True, exist_ok=True)
    result = {"state": state, "from": old, "to": new,
              "version": new if state == "new" else old, "note": note,
              "notes": [str(n) for n in notes], "lines": [], "at": time.time()}
    tmp = updates / f"{applier.RESULT}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, updates / applier.RESULT)


def tidy(root, updates_dir, *, warn: Say, wait_s: float = 0.0) -> Optional[bool]:
    """Remove the old backup in *updates_dir* when the checks passed with it, it was
    provably the install's at *root* when they ran (:data:`CHECKED` says so; an earlier
    removal that stopped may have removed that proof) or is now, and no update watchdog
    of the install runs; with
    *wait_s*, wait that long for such a watchdog to end. True when it was removed (and
    :data:`CHECKED` says so), False when it stays for a later start, None when there is
    nothing to remove."""
    identity = backup_identity(updates_dir)
    checked = read_checked(root)
    if identity is None or checked.get("passed") is not True or checked.get("backup") != identity:
        return None
    if checked.get("mine") is not True and not provable(root, updates_dir):
        return None
    deadline = time.monotonic() + wait_s
    while True:
        running = watchdogs(root)
        if running == []:
            break
        if time.monotonic() >= deadline:
            return False
        time.sleep(_WATCHDOG_POLL_S)
    left = remove_backup(updates_dir)
    if left:
        warn("The backup of the update to localm " + str(checked.get("version"))
             + " could not be removed completely; a later start tries again: "
             + "; ".join(left[:3]))
        return False
    try:
        _record(root, str(checked.get("version")), identity, passed=True, removed=True,
                reasons=[], mine=True)
    except OSError as e:
        warn(f"Could not record that the backup was removed: {e}")
    return True


# ---------------------------------------------------------------- at a start

def _claim(root) -> Optional[Path]:
    """Take :data:`CHECKING` of the install at *root*; None when another start holds it.
    Raises OSError."""
    folder = _folder(root)
    folder.mkdir(exist_ok=True)
    path = folder / CHECKING
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                if time.time() - os.stat(path).st_mtime < CHECKING_STALE_S:
                    return None
                os.unlink(path)
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return path
    return None


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
    waits for the old watchdog in a thread). Returns the note for the user when this
    process must end because the checks failed and the old backup was put back, or
    putting it back failed; else None. *say* gets each step, *warn* each problem;
    neither may raise."""
    updates_dir = Path(home) / "updates"
    identity = due(root, updates_dir, version)
    if identity is None:
        _tidy_now_or_later(root, updates_dir, warn=warn, server=server)
        return None
    if _steps_pending():
        return None
    try:
        claim = _claim(root)
    except OSError as e:
        warn(f"The checks of the update to localm {version} could not start ({e}); they run "
             "at the next start.")
        return None
    if claim is None:
        return None
    try:
        old = identity["version"]
        say(f"localm {old} installed localm {version}; checking the update ...")
        failed = run_checks(root, home, version, say=say,
                            answers=not (server and watchdogs(root)))
        mine = provable(root, updates_dir)
        try:
            _record(root, version, identity, passed=not failed, removed=False, reasons=failed,
                    mine=mine)
        except OSError as e:
            warn(f"Could not record the checks of the update to localm {version}: {e}")
        if not failed:
            say(f"The update from localm {old} to {version} is complete.")
            _tidy_now_or_later(root, updates_dir, warn=warn, server=server)
            return None
        why = "; ".join(f"{reason}" for _kind, reason in failed)
        if mine:
            return _put_back(home, old=old, new=version, why=f"its checks failed: {why}",
                             say=say)
        note = (f"The update to localm {version} failed its checks: {why}. The backup of "
                f"localm {old} is not this install's, so nothing was put back. "
                + _how_to_repair(failed, version))
        warn(note)
        try:
            _result(home, state="failed", old=old, new=version, note=note, notes=[])
        except OSError as e:
            warn(f"Could not record the result of the checks: {e}")
        return None
    finally:
        try:
            os.unlink(claim)
        except OSError as e:
            warn(f"Could not remove {claim}: {e}")


def _steps_pending() -> bool:
    """Whether a runtime or component step of the last update has not run yet."""
    from localm import updater_components, updater_runtime
    return (updater_runtime.read_pending() is not None
            or updater_components.read_pending() is not None)


def _tidy_now_or_later(root, updates_dir, *, warn: Say, server: bool) -> None:
    if tidy(root, updates_dir, warn=warn) is False and server:
        threading.Thread(target=tidy, args=(root, updates_dir),
                         kwargs={"warn": warn, "wait_s": WATCHDOG_WAIT_S},
                         daemon=True, name="old-update-backup").start()


def _put_back(home, *, old: str, new: str, why: str, say: Say) -> str:
    """Put the old backup back (``updater.rollback_last``) because the update from localm
    *old* to *new* failed with *why*, record the outcome (:func:`_recorded`) and return
    the note for the user."""
    from localm import updater
    say(f"The update to localm {new} failed: {why}. Putting localm {old} back ...")
    try:
        updater.rollback_last(say=say)
    except Exception as e:
        detail = f"{getattr(e, 'summary', e)}: {getattr(e, 'reason', '')}".rstrip(": ")
        return _recorded(home, old=old, new=new, why=why, ok=False, detail=detail)
    return _recorded(home, old=old, new=new, why=why, ok=True)


def _recorded(home, *, old: str, new: str, why: str, ok: bool, detail: str = "") -> str:
    """Record that the update from localm *old* to *new* was put back because *why*
    (*ok*), or that putting it back failed with *detail*; returns the note for the
    user."""
    if ok:
        note = (f"The update to localm {new} failed and was undone: {why}. localm {old} is in "
                "place; start it again.")
        state = "old"
    else:
        note = (f"The update to localm {new} failed ({why}) and could not be undone: {detail}. "
                "localm doctor shows which parts disagree.")
        state = "failed"
    try:
        _result(home, state=state, old=old, new=new, note=note, notes=[])
    except OSError as e:
        note += f" (The result could not be recorded: {e}.)"
    return note
