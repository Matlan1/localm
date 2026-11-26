# SPDX-License-Identifier: AGPL-3.0-or-later
"""The checks of the first start after an older localm (0.3.2 to 0.3.4b4) installed
this build with its own updater (``localm.updater_old_clients``), the start-up hook
that runs them (``cli._core._update_steps_at_start``) and the standalone rollback's
refusal of a backup it must not put back (``scripts/rollback_update.py``).

Every install and data folder is a throwaway one under ``tmp_path``."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import install_manifest, update_package, updater, updater_old_clients as oc
from localm.bugreport import LocalmError
from localm.update_mechanism import applier, disk, restoring

REPO = Path(__file__).resolve().parents[1]
NEW = "0.3.4"
OLD = "0.3.3"
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
_real_remove = oc._remove
_real_watchdogs = oc.watchdogs


def _install(tmp_path, name="install") -> Path:
    """A throwaway install of NEW whose files equal its release statement."""
    root = tmp_path / name
    (root / "localm").mkdir(parents=True)
    (root / "VERSION").write_text(NEW, encoding="utf-8")
    (root / "localm" / "x.py").write_text("print('new')\n", encoding="utf-8")
    (root / "README.md").write_text("new\n", encoding="utf-8")
    statement = {"format": update_package.FORMAT, "version": NEW,
                 "files": update_package.files(root), "components": {}}
    (root / update_package.NAME).write_text(json.dumps(statement), encoding="utf-8")
    return root


def _old_backup(home: Path, root: Path, *, state_root=None, kept_for=OLD) -> Path:
    """What an old client's update leaves in *home*: the backup of OLD (with the record
    0.3.3 and later write last), its runtime and components folders and its swap list;
    the components state names *state_root* (None: no state file)."""
    updates = home / "updates"
    backup = updates / "backup"
    (backup / "localm").mkdir(parents=True)
    (backup / "VERSION").write_text(OLD, encoding="utf-8")
    (backup / "localm" / "x.py").write_text("print('old')\n", encoding="utf-8")
    (backup / oc.BACKUP_RECORD).write_text(json.dumps({"names": ["VERSION", "localm"]}),
                                           encoding="utf-8")
    (updates / "backup-runtime").mkdir()
    (updates / "backup-runtime" / "lib.bin").write_bytes(b"old runtime")
    (updates / "backup-components").mkdir()
    if state_root is not None:
        state = {"root": str(state_root), "components": {}}
        if kept_for:
            state["backup_version"] = kept_for
        (updates / "backup-components" / "state.json").write_text(json.dumps(state),
                                                                   encoding="utf-8")
    (updates / "applied_names.json").write_text(json.dumps(["VERSION", "localm"]),
                                                encoding="utf-8")
    return updates


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = _install(tmp_path)
    home = tmp_path / "home"
    updates = _old_backup(home, root, state_root=root)
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: None)
    monkeypatch.setattr(oc, "_steps_pending", lambda: False)
    monkeypatch.setattr(oc, "watchdogs", lambda r: [])
    put_back = []

    def rollback_last(*, installed=None, say=None):
        put_back.append(installed)
        return {"rolled_back": True}
    monkeypatch.setattr(updater, "rollback_last", rollback_last)
    return root, home, updates, put_back


def _start(root, home, *, server=False):
    said, warned = [], []
    note = oc.at_start(root=root, home=home, version=NEW, say=said.append,
                       warn=warned.append, server=server)
    return note, said, warned


def _result(home) -> dict:
    return json.loads((home / "updates" / applier.RESULT).read_text(encoding="utf-8"))


def _load_rb():
    spec = importlib.util.spec_from_file_location(
        "rollback_update_old_clients", REPO / "scripts" / "rollback_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the trigger

def test_the_checks_are_due_only_for_a_backup_of_another_version(tmp_path):
    root = _install(tmp_path)
    updates = tmp_path / "home" / "updates"
    assert oc.due(root, updates, NEW) is None
    _old_backup(tmp_path / "home", root, state_root=root)
    identity = oc.due(root, updates, NEW)
    assert identity and identity["version"] == OLD
    (updates / "backup" / "VERSION").write_text(NEW, encoding="utf-8")
    assert oc.due(root, updates, NEW) is None


def test_a_backup_the_checks_ran_for_is_not_checked_again(setup):
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()   # not provable: it stays
    assert oc.due(root, updates, NEW) is not None
    assert _start(root, home)[0] is None
    assert oc.read_checked(root)["passed"] is True
    assert (updates / "backup" / "VERSION").is_file()
    assert oc.due(root, updates, NEW) is None
    assert oc.due(root, updates, "0.3.5") is None


def test_a_new_backup_of_the_same_version_is_checked_again(tmp_path):
    """0.3.3 -> this build -> back to 0.3.3 -> this build again by 0.3.3's updater: its
    new backup holds the same VERSION text and even the same modification time."""
    root = _install(tmp_path)
    updates = _old_backup(tmp_path / "home", root, state_root=root)
    first = oc.backup_identity(updates)
    oc._record(root, NEW, first, passed=True, removed=False, reasons=[])
    assert oc.due(root, updates, NEW) is None
    shutil.rmtree(updates / "backup")
    time.sleep(0.05)
    (updates / "backup").mkdir()
    (updates / "backup" / "VERSION").write_text(OLD, encoding="utf-8")
    os.utime(updates / "backup" / "VERSION", ns=(first["mtime_ns"], first["mtime_ns"]))
    assert oc.due(root, updates, NEW) is not None


# ---------------------------------------------------------------- the proof

@pytest.mark.parametrize("state_root, kept_for, mine", [
    ("root", OLD, True), ("root", None, True), ("other", OLD, False),
    ("root", "0.3.1", False), (None, OLD, False)])
def test_a_backup_is_this_install_s_only_when_its_state_names_it(
        tmp_path, state_root, kept_for, mine):
    root = _install(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    named = {"root": root, "other": other, None: None}[state_root]
    updates = _old_backup(tmp_path / "home", root, state_root=named, kept_for=kept_for)
    assert oc.provable(root, updates) is mine


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_a_backup_is_the_update_s_only_when_it_was_made_beside_the_release(tmp_path):
    root = _install(tmp_path)
    updates = _old_backup(tmp_path / "home", root, state_root=root)
    assert oc.tied(root, updates) and oc.mine(root, updates)
    _age(updates / "backup" / oc.BACKUP_RECORD, 8 * 86400)    # an update eight days ago
    assert not oc.tied(root, updates) and not oc.mine(root, updates)
    (updates / "backup" / oc.BACKUP_RECORD).unlink()          # 0.3.2: VERSION's own time
    assert oc.tied(root, updates)


# ---------------------------------------------------------------- passed

def test_passed_checks_remove_the_backup_that_is_this_install_s(setup):
    root, home, updates, put_back = setup
    note, said, warned = _start(root, home)
    assert note is None and not warned and not put_back
    for name in oc.BACKUP_PARTS:
        assert not os.path.lexists(updates / name), name
    checked = oc.read_checked(root)
    assert checked["passed"] is True and checked["removed"] is True
    assert checked["version"] == NEW and checked["backup"]["version"] == OLD
    assert any("is complete" in line for line in said)


def test_passed_checks_remove_the_venv_copy_the_old_backup_keeps_beside_the_venv(setup):
    from localm import updater_components as uc
    root, home, updates, _ = setup
    snap = updates / "backup-components"
    copy = uc._venv_copy_path(root / ".venv", snap, "deps")
    copy.mkdir()
    (copy / uc.COPY_MARKER).write_text(str(snap.resolve()), encoding="utf-8")
    (copy / "pkg.py").write_text("old\n", encoding="utf-8")
    state = json.loads((snap / "state.json").read_text(encoding="utf-8"))
    state["components"] = {"venv": {"venv_copy": str(copy)}}
    (snap / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert _start(root, home)[0] is None
    assert not copy.exists() and not snap.exists() and not (updates / "backup").exists()


def test_a_removal_that_stops_leaves_the_backup_folder_whole(setup, monkeypatch):
    root, home, updates, _ = setup
    monkeypatch.setattr(oc, "_remove", lambda path: [f"{path}: held"]
                        if path.name == "backup-runtime" else [])
    assert _start(root, home)[0] is None
    assert (updates / "backup" / "localm" / "x.py").is_file()
    assert oc.read_checked(root)["removed"] is False
    monkeypatch.setattr(oc, "_remove", _real_remove)
    assert oc.tidy(root, updates, warn=print) is True
    assert not os.path.lexists(updates / "backup")
    assert [p.name for p in updates.iterdir() if p.name.startswith("backup")] == []


def test_what_a_stopped_removal_left_is_swept_later(setup):
    """The backup folder renamed aside, and a venv copy whose components state is gone."""
    from localm import updater_components as uc
    root, home, updates, _ = setup
    aside = updates / f"backup{oc._REMOVING}1-2"
    (aside / "localm").mkdir(parents=True)
    (aside / "localm" / "x.py").write_text("old\n", encoding="utf-8")
    snap = updates / "backup-components"
    copy = uc._venv_copy_path(root / ".venv", snap, "deps")
    copy.mkdir()
    (copy / uc.COPY_MARKER).write_text(str(snap.resolve()), encoding="utf-8")
    foreign = uc._venv_copy_path(root / ".venv", snap)
    foreign.mkdir()
    (foreign / uc.COPY_MARKER).write_text(str(root.parent / "elsewhere"), encoding="utf-8")
    shutil.rmtree(updates / "backup")
    assert oc.tidy(root, updates, warn=print) is None       # nothing due: asides only
    assert not aside.exists() and copy.exists()
    shutil.rmtree(snap)
    assert oc.remove_backup(updates, root) == []
    assert not copy.exists() and foreign.exists()


def test_passed_checks_keep_a_backup_that_is_not_provably_this_install_s(setup):
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()
    assert _start(root, home)[0] is None
    assert (updates / "backup" / "VERSION").is_file()
    assert oc.read_checked(root)["passed"] is True
    assert oc.read_checked(root)["removed"] is False


def test_the_backup_stays_when_the_install_changed_after_its_checks(setup, monkeypatch):
    root, home, updates, _ = setup
    monkeypatch.setattr(oc, "_tidy_now_or_later", lambda *a, **k: None)
    assert _start(root, home)[0] is None and oc.read_checked(root)["passed"] is True
    (root / "README.md").write_text("changed\n", encoding="utf-8")
    warned = []
    assert oc.tidy(root, updates, warn=warned.append) is False
    assert (updates / "backup" / "VERSION").is_file() and warned
    (root / "README.md").write_text("new\n", encoding="utf-8")
    statement = json.loads((root / update_package.NAME).read_text(encoding="utf-8"))
    statement["files"] = update_package.files(root)
    (root / update_package.NAME).write_text(json.dumps(statement), encoding="utf-8")
    (root / "VERSION").write_text(OLD, encoding="utf-8")
    assert oc.tidy(root, updates, warn=warned.append) is False
    assert (updates / "backup" / "VERSION").is_file()


# ---------------------------------------------------------------- old watchdogs

def _fake_watchdog(root: Path) -> subprocess.Popen:
    """A process whose command line is that of an old client's update watchdog of
    *root*."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             str(root / "scripts" / "update_watchdog.py"),
                             "--install-root", str(root)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **_NO_WINDOW)


def _read_error(pid: int) -> str:
    """What the system says when the command line of process *pid* is read."""
    if os.name != "nt":
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                return f"read {len(fh.read())} bytes"
        except OSError as e:
            return repr(e)
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    nt = ctypes.WinDLL("ntdll")
    nt.NtQueryInformationProcess.restype = ctypes.c_long
    nt.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return f"OpenProcess failed, winerror {ctypes.get_last_error()}"
    try:
        buf = ctypes.create_string_buffer(0x10010)
        got = wintypes.ULONG(0)
        status = nt.NtQueryInformationProcess(h, 60, buf, len(buf), ctypes.byref(got))
        return f"NtQueryInformationProcess(60) status {status & 0xFFFFFFFF:#010x}, {got.value} bytes"
    finally:
        k32.CloseHandle(h)


def _wait_listed(root: Path, pid: int) -> None:
    """Wait up to 20 s for *pid* among the watchdogs of *root*; on failure, say what
    watchdogs() returned and, for each process it could not tell, its pid, program
    and the error of reading its command line."""
    deadline = time.monotonic() + 20
    while True:
        untold: list = []
        got = oc.watchdogs(root, untold=untold)
        if pid in (got or []):
            return
        if time.monotonic() >= deadline:
            detail = "; ".join(f"pid {r[0]} program {r[2]}: {_read_error(r[0])}" for r in untold)
            pytest.fail(f"the watchdog (pid {pid}) was not found: watchdogs() returned {got!r}; "
                        f"could not tell {len(untold)} process(es): {detail}")
        time.sleep(0.2)


def test_the_update_stays_open_while_the_old_watchdog_of_this_install_runs(setup, monkeypatch):
    """A real process with a watchdog's command line: the version is left to it, nothing
    is recorded as passed and the backup stays until it has ended; then the checks
    finish and the backup goes."""
    root, home, updates, _ = setup
    monkeypatch.setattr(oc, "watchdogs", _real_watchdogs)
    asked = []
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: asked.append(v))
    watchdog = _fake_watchdog(root)
    try:
        _wait_listed(root, watchdog.pid)
        assert _start(root, home)[0] is None
        assert asked == []
        checked = oc.read_checked(root)
        assert checked["passed"] is None and (updates / "backup" / "VERSION").is_file()
        assert oc.tidy(root, updates, warn=print) is None
        assert oc.finish(root, home, NEW, warn=print) is False
    finally:
        watchdog.kill()
        watchdog.wait(timeout=30)
    assert oc.finish(root, home, NEW, warn=print) is True
    assert asked == [NEW]
    assert not (updates / "backup").exists()
    checked = oc.read_checked(root)
    assert checked["passed"] is True and checked["removed"] is True


def test_a_watchdog_of_a_sibling_folder_is_not_this_install_s(tmp_path):
    root = _install(tmp_path, "localm")
    sibling = _install(tmp_path, "localm-dev")
    watchdog = _fake_watchdog(sibling)
    try:
        _wait_listed(sibling, watchdog.pid)
        assert watchdog.pid not in (oc.watchdogs(root) or [])
    finally:
        watchdog.kill()
        watchdog.wait(timeout=30)


def test_an_unreadable_command_line_means_a_watchdog_may_run(monkeypatch, tmp_path):
    root = tmp_path / "install"
    monkeypatch.setattr(oc, "_by_ps", lambda: False)
    monkeypatch.setattr(install_manifest, "list_processes",
                        lambda: [(41, 1, None, None), (42, 1, "/x/python.exe", None)])
    lines = {41: None, 42: None}
    monkeypatch.setattr(oc, "_argv", lambda pid: lines[pid])
    assert oc.watchdogs(root) is None
    lines[42] = ["python", "other.py"]
    assert oc.watchdogs(root) == []
    lines[42] = ["python", str(root / "scripts" / "update_watchdog.py"), "--port", "1"]
    assert oc.watchdogs(root) is None
    lines[42] = ["python", str(root / "scripts" / "update_watchdog.py"),
                 f"--install-root={root}"]
    assert oc.watchdogs(root) == [42]
    monkeypatch.setattr(install_manifest, "list_processes", lambda: None)
    assert oc.watchdogs(root) is None


def test_a_process_that_ended_before_its_command_line_was_read_is_left_out(monkeypatch,
                                                                          tmp_path):
    """A busy computer starts and ends processes all the time; one that ended between
    the listing and the read says nothing about a watchdog. One that still runs under
    the same start time does."""
    root = tmp_path / "install"
    monkeypatch.setattr(oc, "_by_ps", lambda: False)
    listings = [[(42, 1, "/x/python.exe", 100)], []]
    monkeypatch.setattr(install_manifest, "list_processes", lambda: listings.pop(0))
    monkeypatch.setattr(oc, "_argv", lambda pid: None)
    assert oc.watchdogs(root) == []
    listings[:] = [[(42, 1, "/x/python.exe", 100)], [(42, 1, "/x/python.exe", 200)]]
    assert oc.watchdogs(root) == []
    listings[:] = [[(42, 1, "/x/python.exe", 100)], [(42, 1, "/x/python.exe", 100)]]
    assert oc.watchdogs(root) is None


def test_a_program_that_is_no_interpreter_is_never_an_old_watchdog(monkeypatch, tmp_path):
    """Seen on an elevated Windows runner: the isolated LSA process names its program
    but refuses its command line (status 0x8000000d). An old watchdog runs in a Python
    interpreter or in LocaLM.exe, so only those count when they cannot be read."""
    root = tmp_path / "install"
    monkeypatch.setattr(oc, "_by_ps", lambda: False)
    lsa = str(tmp_path / "System32" / "LsaIso.exe")
    rows = [(956, 4, lsa, 100)]
    monkeypatch.setattr(install_manifest, "list_processes", lambda: list(rows))
    read = []
    monkeypatch.setattr(oc, "_argv", lambda pid: read.append(pid))
    assert oc.watchdogs(root) == [] and read == []
    for name in ("python.exe", "pythonw.exe", "python3.12", "LocaLM.exe"):
        program = str(tmp_path / "bin" / name)
        rows[:] = [(956, 4, lsa, 100), (57, 4, program, 100)]
        untold: list = []
        assert oc.watchdogs(root, untold=untold) is None, program
        assert [r[0] for r in untold] == [57]


def test_the_command_lines_ps_shows_are_read_on_a_system_without_proc(monkeypatch, tmp_path):
    root = tmp_path / "my install"
    monkeypatch.setattr(oc, "_by_ps", lambda: True)
    monkeypatch.setattr(install_manifest, "list_processes", lambda: [
        (7, 1, "/usr/sbin/launchd", None), (8, 1, "/opt/py/bin/Python", None)])
    shown = {8: None}
    monkeypatch.setattr(oc, "_ps_command_line", lambda pid: shown[pid])
    assert oc.watchdogs(root) is None
    shown[8] = (f"/opt/py/bin/Python {root}/scripts/update_watchdog.py --port 5 "
                f"--install-root {root} --code-root /x/backup")
    assert oc.watchdogs(root) == [8]
    assert oc._option_in_line(shown[8], "--code-root") == "/x/backup"


def test_a_watchdog_that_cannot_be_told_counts_only_while_an_old_one_could_run(
        monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "watchdogs", lambda r: None)
    assert oc._may_run(tmp_path, time.time())
    assert not oc._may_run(tmp_path, time.time() - oc.WATCHDOG_WAIT_S - 1)
    monkeypatch.setattr(oc, "watchdogs", lambda r: [5])
    assert oc._may_run(tmp_path, 0.0)


@pytest.mark.skipif(os.name != "nt", reason="the Windows command line split")
def test_a_windows_command_line_is_split_as_the_c_runtime_splits_it():
    assert oc._split_windows('"C:\\a b\\python.exe" x.py --install-root "D:\\my localm"') == [
        "C:\\a b\\python.exe", "x.py", "--install-root", "D:\\my localm"]


def test_open_checks_are_never_put_back_or_refused_while_a_watchdog_may_run(setup,
                                                                            monkeypatch):
    """The H1 case: a server whose old watchdog runs does not record passed before the
    version was confirmed, so the watchdog's own rollback still works; a failed check
    under the watchdog leaves the way back to it."""
    root, home, updates, put_back = setup
    monkeypatch.setattr(oc, "watchdogs", lambda r: [12345])
    monkeypatch.setattr(oc, "_later", lambda *a, **k: None)
    asked = []
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: asked.append(v))
    assert _start(root, home, server=True)[0] is None
    assert asked == [] and oc.read_checked(root)["passed"] is None
    assert oc.refusal(root, updates) is None
    assert oc.due(root, updates, NEW) is not None
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    note = _start(root, home, server=True)[0]
    assert note and "watchdog of localm 0.3.3 puts" in note and put_back == []
    assert oc.read_checked(root)["passed"] is None and oc.refusal(root, updates) is None
    monkeypatch.setattr(oc, "watchdogs", lambda r: [])
    assert _start(root, home)[0] is not None and put_back == [None]


def test_a_server_asks_the_version_once_the_watchdog_ended(setup, monkeypatch):
    root, home, updates, _ = setup
    running = [[7]]
    monkeypatch.setattr(oc, "watchdogs", lambda r: running[0])
    monkeypatch.setattr(oc, "_later", lambda *a, **k: None)
    assert _start(root, home, server=True)[0] is None
    running[0] = []
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: "localm 0.3.4 does not start")
    warned = []
    assert oc.finish(root, home, NEW, warn=warned.append) is False
    assert oc.read_checked(root)["passed"] is None and (updates / "backup").is_dir()
    assert warned and "next start" in warned[0]
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: None)
    assert oc.finish(root, home, NEW, warn=warned.append) is True
    assert oc.read_checked(root)["passed"] is True and not (updates / "backup").exists()


def test_a_server_the_old_watchdog_undid_is_told_to_start_again(setup, monkeypatch):
    """The new server answered too late: the old watchdog put its version back under it."""
    root, home, updates, _ = setup
    monkeypatch.setattr(oc, "watchdogs", lambda r: [7])
    monkeypatch.setattr(oc, "_later", lambda *a, **k: None)
    assert _start(root, home, server=True)[0] is None
    monkeypatch.setattr(oc, "watchdogs", lambda r: [])
    (root / "VERSION").write_text(OLD, encoding="utf-8")
    warned = []
    monkeypatch.setattr(oc, "run_checks", lambda *a, **k: pytest.fail("checked it again"))
    assert oc.finish(root, home, NEW, warn=warned.append) is False
    assert warned and "start it again" in warned[0] and f"localm {OLD} is in place" in warned[0]
    assert (updates / "backup").is_dir()


# ---------------------------------------------------------------- failed

def test_a_changed_file_puts_this_install_s_backup_back_with_the_reason(setup):
    root, home, updates, put_back = setup
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    note, _said, _warned = _start(root, home)
    assert put_back == [None]
    assert note and "localm/x.py" in note and "undone" in note
    result = _result(home)
    assert result["state"] == "old" and result["from"] == OLD and result["to"] == NEW
    assert "localm/x.py" in result["note"] and result["version"] == OLD
    checked = oc.read_checked(root)
    assert checked["passed"] is False and checked["reasons"][0][0] == "files"
    assert checked["putting_back"] is False


def test_the_old_version_is_told_why_it_was_put_back(setup):
    """localm 0.3.3 to 0.3.4b4 show the last runtime and component update notes recorded
    under their own version in Settings."""
    root, home, updates, _ = setup
    (updates / "runtime-update.json").write_text(json.dumps(
        {"state": "rolled_back", "note": "the update was rolled back", "version": NEW,
         "at": 1.0}), encoding="utf-8")
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    note = _start(root, home)[0]
    for name in ("runtime-update.json", "components-update.json"):
        st = json.loads((updates / name).read_text(encoding="utf-8"))
        assert st["note"] == note and st["version"] == OLD, name
        assert time.time() - st["at"] < 60


def test_a_failed_way_back_is_tried_again_at_the_next_start(setup, monkeypatch):
    root, home, _updates, _ = setup
    (root / "README.md").unlink()
    calls = []

    def refuse(**_k):
        calls.append(1)
        raise LocalmError("an update or rollback is already running", reason="lock held")
    monkeypatch.setattr(updater, "rollback_last", refuse)
    note = _start(root, home)[0]
    assert note and "could not be undone yet" in note and "lock held" in note
    assert oc.read_checked(root)["putting_back"] is True
    assert _result(home)["state"] == "failed" and _result(home)["version"] == NEW
    for _ in range(oc.PUT_BACK_TRIES - 1):
        note = _start(root, home)[0]
    assert len(calls) == oc.PUT_BACK_TRIES
    assert "could not be undone:" in note and "unzip the localm" in note
    assert oc.read_checked(root)["putting_back"] is False
    assert _start(root, home)[0] is None and len(calls) == oc.PUT_BACK_TRIES


def test_a_put_back_that_was_stopped_is_resumed_without_checking_again(setup, monkeypatch):
    root, home, updates, put_back = setup
    oc._record(root, NEW, oc.backup_identity(updates), passed=False, removed=False,
               reasons=[["files", "these files differ: README.md"]], mine=True,
               putting_back=True, tries=1)
    monkeypatch.setattr(oc, "run_checks", lambda *a, **k: pytest.fail("checked again"))
    note, said, _ = _start(root, home)
    assert put_back == [None] and "README.md" in note and "undone" in note
    assert any("an earlier start began" in s for s in said)
    assert oc.read_checked(root)["putting_back"] is False


def test_a_failed_check_never_puts_back_a_backup_of_another_install(setup):
    root, home, updates, put_back = setup
    (updates / "backup-components" / "state.json").write_text(
        json.dumps({"root": str(root.parent / "elsewhere"), "components": {}}),
        encoding="utf-8")
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    note, _said, warned = _start(root, home)
    assert note is None and put_back == []
    assert (updates / "backup" / "VERSION").is_file()
    assert any("nothing was put back" in w and "unzip the localm" in w for w in warned)
    result = _result(home)
    assert result["state"] == "failed" and result["version"] == NEW
    assert result["root"] == str(root.resolve())


def test_a_failed_check_never_puts_back_the_backup_of_an_earlier_update(setup):
    """A b3 backup, then this release unzipped by hand and set up: the backup is this
    install's, but not of the update that installed this version."""
    root, home, updates, put_back = setup
    _age(updates / "backup" / oc.BACKUP_RECORD, 8 * 86400)
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    assert _start(root, home)[0] is None and put_back == []
    assert (updates / "backup" / "VERSION").is_file()
    assert oc.refusal(root, updates)


def test_a_venv_without_this_version_s_packages_fails_the_checks(setup, monkeypatch):
    root, home, _updates, put_back = setup
    from localm import updater_parts
    monkeypatch.setattr(updater_parts, "_deps", lambda r: {
        "venv": True, "record": NEW, "problems": ["requests>=2"]})
    note, _said, _warned = _start(root, home)
    assert note and "requests>=2" in note and put_back == [None]
    assert oc.read_checked(root)["reasons"][0][0] == "parts"


def test_a_version_that_does_not_answer_fails_the_checks(setup, monkeypatch):
    root, home, _updates, put_back = setup
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: f"localm {v} does not start: boom")
    note, _said, _warned = _start(root, home)
    assert note and "does not start: boom" in note and put_back == [None]


def test_files_that_cannot_be_read_now_leave_the_checks_to_the_next_start(setup,
                                                                         monkeypatch):
    root, home, updates, put_back = setup
    tries = []

    def held(_root, **_k):
        tries.append(1)
        raise disk.Held(root / "localm" / "x.py", "it is open in another program")
    monkeypatch.setattr(restoring, "stated_differences", held)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    note, _said, warned = _start(root, home)
    assert note is None and put_back == [] and len(tries) == oc._FILES_TRIES
    assert oc.read_checked(root) == {} and any("next start" in w for w in warned)
    assert oc.due(root, updates, NEW) is not None


def test_the_checks_wait_for_the_update_steps_still_to_run(setup, monkeypatch):
    root, home, updates, put_back = setup
    monkeypatch.setattr(oc, "_steps_pending", lambda: True)
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    assert _start(root, home)[0] is None and put_back == []
    assert oc.read_checked(root) == {} and (updates / "backup").is_dir()


def test_a_second_start_does_not_run_the_checks_beside_the_first(setup):
    """The checks hold a store lock; one a running process holds is waited for briefly
    and then left to it, one whose holder ended is taken at once."""
    root, home, _updates, put_back = setup
    (root / ".localm-update").mkdir()
    target = root / ".localm-update" / oc.CHECKING
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from pathlib import Path; from localm import storekit\n"
         "with storekit.store_lock(Path(sys.argv[1]), timeout=30):\n"
         "    print('held', flush=True); time.sleep(60)\n", str(target)],
        cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, **_NO_WINDOW)
    try:
        assert holder.stdout.readline().strip() == "held"
        (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
        assert _start(root, home)[0] is None and put_back == []
    finally:
        holder.kill()
        holder.wait(timeout=30)
    assert _start(root, home)[0] is not None and put_back == [None]
    assert not target.with_name(oc.CHECKING + ".lock").exists()


# ---------------------------------------------------------------- the files

def test_stated_differences_names_each_file_that_differs(tmp_path):
    root = _install(tmp_path)
    assert restoring.stated_differences(root) == (NEW, [])
    assert restoring.release_of(root) == NEW
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    (root / "README.md").unlink()
    version, differ = restoring.stated_differences(root)
    assert version == NEW and sorted(differ) == ["README.md", "localm/x.py"]
    assert len(restoring.stated_differences(root, first=True)[1]) == 1
    assert restoring.release_of(root) == ""


def test_the_version_is_asked_of_the_real_app(tmp_path):
    """The real probe: this checkout started as its own process, no model."""
    from localm import _version
    home = tmp_path / "home"
    home.mkdir()
    version = _version.read_version()
    assert oc._answers(REPO, home, version) is None
    why = oc._answers(REPO, home, "9.9.9")
    assert why and f"localm {version} answered instead of 9.9.9" in why


# ---------------------------------------------------------------- the start-up hook

def test_the_start_runs_the_checks_after_the_update_steps(monkeypatch):
    from localm import _version
    from localm.cli import _core
    order = []
    monkeypatch.setattr(_core, "_finish_pending_runtime_update", lambda c: order.append("rt"))
    monkeypatch.setattr(_core, "_finish_pending_component_update",
                        lambda c: order.append("comp"))
    monkeypatch.setattr(_core, "_old_client_checks", lambda c: order.append(("checks", c)))
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
    monkeypatch.delenv("LOCALM_UPDATE_STEPS_DONE", raising=False)
    _core._update_steps_at_start("run")
    assert order == ["rt", "comp", ("checks", "run")]
    order.clear()
    monkeypatch.setenv("LOCALM_UPDATE_STEPS_DONE", _version.read_version())
    _core._update_steps_at_start("gui")
    assert order == [("checks", "gui")]
    order.clear()
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.setattr(_core, "_defer_update_steps_to_a_restart", lambda: True)
    _core._update_steps_at_start("gui")
    assert order == []
    monkeypatch.setattr(_core, "_defer_update_steps_to_a_restart", lambda: False)
    _core._update_steps_at_start("gui")
    assert order == [("checks", "gui")]


def test_a_start_whose_checks_put_the_old_version_back_exits_1(monkeypatch):
    from localm.cli import _core
    calls = []

    def at_start(**kw):
        calls.append(kw["server"])
        return "The update to localm 0.3.4 failed and was undone: x."
    monkeypatch.setattr(oc, "at_start", at_start)
    monkeypatch.delenv("LOCALM_UPDATE_ID", raising=False)
    _core._old_client_checks("gui")   # this checkout is a git clone
    assert calls == []
    monkeypatch.setattr(updater, "is_git_clone", lambda root=None: False)
    with pytest.raises(SystemExit) as e:
        _core._old_client_checks("gui")
    assert e.value.code == 1 and calls == [True]
    monkeypatch.setattr(oc, "at_start", lambda **kw: None)
    _core._old_client_checks("list")
    monkeypatch.setattr(oc, "at_start", lambda **kw: pytest.fail("ran in an update child"))
    monkeypatch.setenv("LOCALM_UPDATE_ID", "")
    _core._old_client_checks("list")
    monkeypatch.delenv("LOCALM_UPDATE_ID")
    _core._old_client_checks("_update-probe")


def test_what_the_checks_say_never_reaches_stdout_and_mcp_skips_them(monkeypatch, capfd):
    """``localm mcp`` speaks its protocol on stdout, and ``--json`` output is parsed."""
    from localm.cli import _core
    monkeypatch.delenv("LOCALM_UPDATE_ID", raising=False)
    monkeypatch.setattr(updater, "is_git_clone", lambda root=None: False)

    def at_start(*, say, warn, **_kw):
        say("checking the update")
        warn("a problem")
        return "The update failed and was undone."
    monkeypatch.setattr(oc, "at_start", at_start)
    with pytest.raises(SystemExit):
        _core._old_client_checks("list")
    out, err = capfd.readouterr()
    assert out == "" and "checking the update" in err and "was undone" in err
    monkeypatch.setattr(oc, "at_start", lambda **kw: pytest.fail("ran for localm mcp"))
    _core._old_client_checks("mcp")


def test_the_settings_list_only_this_install_s_results(monkeypatch, tmp_path):
    from localm import updater_components, updater_parts, updater_runtime
    now = time.time()
    monkeypatch.setattr(updater, "last_result", lambda: {
        "state": "failed", "version": NEW, "note": "other install", "root": str(tmp_path / "b"),
        "at": now})
    monkeypatch.setattr(updater_runtime, "last_status", lambda: None)
    monkeypatch.setattr(updater_components, "last_status", lambda: {
        "state": "rolled_back", "version": NEW, "note": "mine", "root": str(tmp_path / "a"),
        "at": now})
    assert [s["note"] for s in updater_parts._last_steps(NEW, tmp_path / "a")] == ["mine"]
    assert len(updater_parts._last_steps(NEW)) == 2


# ---------------------------------------------------------------- rollback.*

def _rollback_install(tmp_path, monkeypatch):
    root = _install(tmp_path)
    shutil.copy2(REPO / "localm" / "updater_old_clients.py",
                 root / "localm" / "updater_old_clients.py")
    home = tmp_path / "home"
    updates = _old_backup(home, root, state_root=None)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    return root, updates


@pytest.mark.parametrize("passed, mine, refused", [
    (None, False, False), (False, True, False), (False, False, True), (True, True, True),
    (True, False, True)])
def test_rollback_script_refuses_a_backup_its_checks_left_alone(
        tmp_path, monkeypatch, passed, mine, refused):
    root, updates = _rollback_install(tmp_path, monkeypatch)
    rb = _load_rb()
    assert rb._refusal(root, updates) is None
    oc._record(root, NEW, oc.backup_identity(updates), passed=passed, removed=False,
               reasons=[], mine=mine)
    assert bool(rb._refusal(root, updates)) is refused


def test_rollback_script_changes_nothing_when_it_refuses(tmp_path, monkeypatch, capsys):
    root, updates = _rollback_install(tmp_path, monkeypatch)
    rb = _load_rb()
    oc._record(root, NEW, oc.backup_identity(updates), passed=True, removed=False,
               reasons=[])
    assert rb._refusal(root, updates)    # the first load may write the helper's bytecode
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert rb.main(["--yes"], install_root=root) == 1
    assert "its checks passed" in capsys.readouterr().err
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert (root / "localm" / "x.py").read_text(encoding="utf-8") == "print('new')\n"


# ---------------------------------------------------------------- the version after

_STAND_IN_MAIN = """\
import json, os, sys
from pathlib import Path
here = Path(__file__).resolve().parent.parent
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("STAND_IN_SAYS") or "localm, version " + (here / "VERSION").read_text().strip())
    raise SystemExit(int(os.environ.get("STAND_IN_VERSION_EXIT") or 0))
(here / "started.json").write_text(json.dumps({
    "argv": sys.argv[1:], "cwd": os.getcwd(),
    "relaunched": os.environ.get("LOCALM_OLD_CLIENT_RELAUNCH"),
    "restart": os.environ.get("LOCALM_RESTART_IN_PROGRESS")}))
raise SystemExit(7)
"""


def _stand_in_build(root: Path) -> None:
    """Make *root* a stand-in build that ``python -m localm`` (run in *root*) starts:
    ``--version`` prints its ``VERSION`` (or ``STAND_IN_SAYS``), any other command
    records its arguments and environment in ``started.json`` and exits 7."""
    (root / "localm" / "__init__.py").write_text("", encoding="utf-8")
    (root / "localm" / "__main__.py").write_text(_STAND_IN_MAIN, encoding="utf-8")


def _started(root: Path, wait_s: float = 0.0):
    path = root / "started.json"
    deadline = time.monotonic() + wait_s
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.2)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@pytest.fixture
def put_back_build(setup, monkeypatch):
    """The setup whose put-back writes the backup's VERSION into the install, the
    install a stand-in build, this process's environment that of a restart."""
    root, home, updates, put_back = setup
    _stand_in_build(root)

    def rollback_last(*, installed=None, say=None):
        put_back.append(installed)
        (root / "VERSION").write_text(OLD, encoding="utf-8")
        return {"rolled_back": True}
    monkeypatch.setattr(updater, "rollback_last", rollback_last)
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.delenv(oc.RELAUNCHED, raising=False)
    monkeypatch.chdir(root)
    return root, home, updates, put_back


def test_a_start_whose_checks_put_the_previous_version_back_starts_it(put_back_build,
                                                                     monkeypatch):
    """The old start exited 1 and the user had to start localm again: the previous
    version now starts with the same command and arguments, and its status is the
    status of this start."""
    from localm.cli import _core
    root, home, _updates, put_back = put_back_build
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    monkeypatch.setattr(updater, "is_git_clone", lambda root=None: False)
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_UPDATE_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["localm", "gui", "--port", "8099"])
    with pytest.raises(SystemExit) as e:
        _core._old_client_checks("gui")
    assert e.value.code == 7 and put_back == [None]
    started = _started(root)
    assert started["argv"] == ["gui", "--port", "8099"], started
    from localm import _version
    assert started["relaunched"] == _version.read_version(), started
    assert started["restart"] is None, started
    assert os.path.samefile(started["cwd"], root)


def test_the_console_says_which_version_starts_and_why(put_back_build):
    root, home, _updates, _ = put_back_build
    (root / "README.md").unlink()
    note, said, warned = _start(root, home)
    assert isinstance(note, oc.Ending) and note.then == "previous" and note.old == OLD
    told: list = []
    assert oc.go_on(note, root=root, argv=["run", "m"], say=told.append, warn=told.append) == 7
    assert "README.md" in note and f"localm {OLD} is in place again" in note
    assert told == [f"Starting localm {OLD} with the same command: localm run m"]


@pytest.mark.parametrize("case", ["no-answer", "wrong-version", "not-back"])
def test_a_previous_version_that_cannot_start_is_not_started_and_the_fix_is_named(
        put_back_build, monkeypatch, case):
    root, home, _updates, _ = put_back_build
    (root / "README.md").unlink()
    note = _start(root, home)[0]
    if case == "no-answer":
        monkeypatch.setenv("STAND_IN_VERSION_EXIT", "1")
        monkeypatch.setenv("STAND_IN_SAYS", "ModuleNotFoundError: No module named 'localm.x'")
    elif case == "wrong-version":
        monkeypatch.setenv("STAND_IN_SAYS", "localm, version 0.1.0")
    else:
        (root / "VERSION").write_text("0.2.9", encoding="utf-8")
    told: list = []
    assert oc.go_on(note, root=root, argv=["gui"], say=told.append, warn=told.append) == 1
    assert _started(root) is None
    assert len(told) == 1 and f"unzip the localm {OLD} release" in told[0], told
    if case == "no-answer":
        assert "No module named" in told[0] and "not started" in told[0], told
    elif case == "wrong-version":
        assert "0.1.0" in told[0], told
    else:
        assert "localm 0.2.9 now" in told[0], told


def _watchdog_process(root: Path, *extra: str, code: str = "import time; time.sleep(60)"):
    return subprocess.Popen([sys.executable, "-c", code,
                             str(root / "scripts" / "update_watchdog.py"),
                             "--install-root", str(root), *extra],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **_NO_WINDOW)


@pytest.mark.parametrize("relaunch, server, then", [
    (True, True, "watchdog"), (False, True, "after_watchdog"), (False, False, None)])
def test_a_failed_check_under_an_old_watchdog_says_who_starts_the_previous_version(
        setup, monkeypatch, relaunch, server, then):
    """localm 0.3.4b3 and b4 watchdogs start the previous version after their rollback
    (``--relaunch-argv``); for earlier ones a starter does."""
    root, home, _updates, put_back = setup
    monkeypatch.setattr(oc, "watchdogs", _real_watchdogs)
    monkeypatch.setattr(oc, "_later", lambda *a, **k: None)
    extra = ("--relaunch-argv", json.dumps(["python", "-m", "localm", "gui"])) if relaunch else ()
    watchdog = _watchdog_process(root, *extra)
    try:
        _wait_listed(root, watchdog.pid)
        (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
        note = _start(root, home, server=server)[0]
    finally:
        watchdog.kill()
        watchdog.wait(timeout=30)
    assert isinstance(note, oc.Ending) and note.then == then and put_back == []
    assert "watchdog of localm 0.3.3 puts" in note or "Start it again once" in note
    if then == "after_watchdog":
        assert watchdog.pid in note.pids and "starts again by itself" in note, note.pids
    if then == "watchdog":
        assert "starts it again with the same command" in note


def test_the_starter_waits_for_the_old_watchdog_then_starts_the_previous_version(
        tmp_path, monkeypatch):
    """A real starter process (this module run with only the standard library) and a
    real process standing in for an old watchdog that puts the previous version back
    and ends."""
    root = _install(tmp_path)
    _stand_in_build(root)
    monkeypatch.delenv(oc.RELAUNCHED, raising=False)
    monkeypatch.chdir(root)
    code = ("import sys, time; time.sleep(2); "
            f"open(sys.argv[1], 'w').write({OLD!r})")
    watchdog = subprocess.Popen([sys.executable, "-c", code, str(root / "VERSION")],
                                stdin=subprocess.DEVNULL, **_NO_WINDOW)
    try:
        note = oc.Ending("failed", then="after_watchdog", old=OLD, new=NEW,
                         pids=[watchdog.pid])
        told: list = []
        assert oc.go_on(note, root=root, argv=["serve", "-p", "1"], say=told.append,
                        warn=told.append) == 1
        assert told == [] and _started(root) is None, "started before the watchdog ended"
        started = _started(root, wait_s=90)
    finally:
        watchdog.kill()
        watchdog.wait(timeout=30)
    assert started and started["argv"] == ["serve", "-p", "1"], started
    assert started["relaunched"] == NEW
    assert (root / "VERSION").read_text(encoding="utf-8") == OLD


def test_a_start_by_a_starter_does_not_arrange_another(monkeypatch, tmp_path):
    monkeypatch.setenv(oc.RELAUNCHED, NEW)
    monkeypatch.setattr(oc, "_spawn_starter", lambda *a: pytest.fail("a second starter"))
    told: list = []
    note = oc.Ending("failed", then="after_watchdog", old=OLD, new=NEW, pids=[1])
    assert oc.go_on(note, root=tmp_path, argv=["gui"], say=told.append, warn=told.append) == 1
    assert told and "Start localm again" in told[0]


def test_the_starter_waits_for_a_settled_version_when_the_watchdogs_cannot_be_told(
        tmp_path, monkeypatch):
    root = _install(tmp_path)
    clock = [0.0]
    slept: list = []

    def sleep(s):
        slept.append(s)
        clock[0] += s
        if len(slept) == 3:
            (root / "VERSION").write_text(OLD, encoding="utf-8")
    oc._wait_for_watchdogs(root, OLD, None, 600.0, sleep=sleep, clock=lambda: clock[0])
    assert len(slept) > 3 and sum(slept[3:]) >= oc._SETTLED_S, slept
    assert clock[0] < 600.0, "it waited for the time limit instead"
    slept.clear()
    (root / "VERSION").write_text(NEW, encoding="utf-8")
    clock[0] = 0.0
    oc._wait_for_watchdogs(root, "9.9", None, 30.0, sleep=lambda s: slept.append(s) or
                           clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    assert clock[0] >= 30.0


# ---------------------------------------------------------------- the kept backup

def test_a_backup_not_provably_this_install_s_is_offered_and_removed_on_request(setup):
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()
    assert _start(root, home)[0] is None and (updates / "backup").is_dir()
    found = oc.kept_backup(root, home)
    assert found and found["version"] == OLD and found["bytes"] > 0, found
    assert oc.remove_kept_backup(root, home) == []
    assert not any((updates / name).exists() for name in oc.BACKUP_PARTS)
    assert oc.read_checked(root)["removed"] is True and oc.kept_backup(root, home) is None
    with pytest.raises(oc.Refused, match="no backup"):
        oc.remove_kept_backup(root, home)


def test_the_kept_backup_is_not_offered_while_its_checks_are_open_or_it_is_mine(setup):
    root, home, updates, _ = setup
    assert oc.kept_backup(root, home) is None, "checks not run yet"
    oc._record(root, NEW, oc.backup_identity(updates), passed=None, removed=False, reasons=[])
    assert oc.kept_backup(root, home) is None, "checks open"
    oc._record(root, NEW, oc.backup_identity(updates), passed=False, removed=False,
               reasons=[], mine=False)
    assert oc.kept_backup(root, home) is None, "provably this install's now"
    with pytest.raises(oc.Refused):
        oc.remove_kept_backup(root, home)
    assert (updates / "backup" / "VERSION").is_file()


def test_the_kept_backup_is_not_removed_while_a_watchdog_may_run_or_an_update_runs(
        setup, monkeypatch):
    from localm.update_mechanism import job
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()
    _start(root, home)
    monkeypatch.setattr(oc, "watchdogs", lambda r: [7])
    with pytest.raises(oc.Refused, match="watchdog"):
        oc.remove_kept_backup(root, home)
    monkeypatch.setattr(oc, "watchdogs", lambda r: [])
    monkeypatch.setattr(job, "locked", lambda r: True)
    with pytest.raises(oc.Refused, match="in progress"):
        oc.remove_kept_backup(root, home)
    assert (updates / "backup" / "VERSION").is_file() and oc.kept_backup(root, home)


OWNER_KEY = "owner-admin-key-old-backup-abc123"


@pytest.fixture
def old_backup_client(setup, monkeypatch):
    import localm.config as cfg
    from fastapi.testclient import TestClient

    from localm import auth
    from localm.inference.http_server import create_app
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()
    _start(root, home)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    scoped = auth.create_key("dev", ["config:read", "config:write"],
                             allow_privileged=True)["key"]
    app = create_app(None)
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=lambda name: None,
               active_model=lambda: "model-a")
    with TestClient(app) as c:
        yield c, scoped, root, home, updates


def test_the_route_shows_the_kept_backup_and_removes_it_only_through_its_checks(
        old_backup_client, monkeypatch):
    c, scoped, root, home, updates = old_backup_client
    owner = {"Authorization": f"Bearer {OWNER_KEY}"}
    r = c.get("/api/update/old-backup", headers=owner)
    assert r.status_code == 200 and r.json()["kept"] is True, r.text
    assert r.json()["version"] == OLD and r.json()["bytes"] > 0
    r = c.post("/api/update/old-backup/remove", headers={"Authorization": f"Bearer {scoped}"})
    assert r.status_code == 403 and (updates / "backup").is_dir(), r.text
    monkeypatch.setattr(oc, "watchdogs", lambda r: [7])
    r = c.post("/api/update/old-backup/remove", headers=owner)
    assert r.status_code == 409 and "watchdog" in r.json()["detail"], r.text
    assert (updates / "backup" / "VERSION").is_file()
    monkeypatch.setattr(oc, "watchdogs", lambda r: [])
    r = c.post("/api/update/old-backup/remove", headers=owner)
    assert r.status_code == 200 and r.json() == {"removed": True}, r.text
    assert not (updates / "backup").exists()
    assert c.get("/api/update/old-backup", headers=owner).json() == {"kept": False}


def test_the_command_removes_the_kept_backup_only_when_there_is_one(setup, monkeypatch):
    from click.testing import CliRunner

    from localm.cli import _core, main
    root, home, updates, _ = setup
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    monkeypatch.setattr(_core, "_update_steps_at_start", lambda command: None)
    r = CliRunner().invoke(main, ["update", "--remove-old-backup", "--yes"])
    assert r.exit_code == 1 and "no backup" in r.output, r.output
    (updates / "backup-components" / "state.json").unlink()
    _start(root, home)
    r = CliRunner().invoke(main, ["update", "--remove-old-backup"])
    assert r.exit_code == 1 and "--yes" in r.output and (updates / "backup").is_dir(), r.output
    r = CliRunner().invoke(main, ["update", "--remove-old-backup", "--yes"])
    assert r.exit_code == 0 and "was removed" in r.output, r.output
    assert f"localm {OLD}" in r.output and not (updates / "backup").exists()
    r = CliRunner().invoke(main, ["update", "--remove-old-backup", "--rollback"])
    assert r.exit_code == 2
