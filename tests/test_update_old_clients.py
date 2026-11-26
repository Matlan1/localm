# SPDX-License-Identifier: AGPL-3.0-or-later
"""The checks of the first start after an older localm (0.3.2 to 0.3.4b4) installed
this build with its own updater (``localm.updater_old_clients``), the start-up hook
that runs them (``cli._core._update_steps_at_start``) and the standalone rollback's
refusal of a backup whose update passed them (``scripts/rollback_update.py``).

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

from localm import update_package, updater, updater_old_clients as oc
from localm.update_mechanism import applier, restoring

REPO = Path(__file__).resolve().parents[1]
NEW = "0.3.4"
OLD = "0.3.3"


def _install(tmp_path) -> Path:
    """A throwaway install of NEW whose files equal its release statement."""
    root = tmp_path / "install"
    (root / "localm").mkdir(parents=True)
    (root / "VERSION").write_text(NEW, encoding="utf-8")
    (root / "localm" / "x.py").write_text("print('new')\n", encoding="utf-8")
    (root / "README.md").write_text("new\n", encoding="utf-8")
    statement = {"format": update_package.FORMAT, "version": NEW,
                 "files": update_package.files(root), "components": {}}
    (root / update_package.NAME).write_text(json.dumps(statement), encoding="utf-8")
    return root


def _old_backup(home: Path, root: Path, *, state_root=None, kept_for=OLD) -> Path:
    """What an old client's update leaves in *home*: the backup of OLD, its runtime and
    components folders and its swap list; the components state names *state_root*
    (None: no state file)."""
    updates = home / "updates"
    backup = updates / "backup"
    (backup / "localm").mkdir(parents=True)
    (backup / "VERSION").write_text(OLD, encoding="utf-8")
    (backup / "localm" / "x.py").write_text("print('old')\n", encoding="utf-8")
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


def test_passed_checks_keep_a_backup_that_is_not_provably_this_install_s(setup):
    root, home, updates, _ = setup
    (updates / "backup-components" / "state.json").unlink()
    assert _start(root, home)[0] is None
    assert (updates / "backup" / "VERSION").is_file()
    assert oc.read_checked(root)["passed"] is True
    assert oc.read_checked(root)["removed"] is False


def _fake_watchdog(root: Path) -> subprocess.Popen:
    """A process whose command line is that of an old client's update watchdog of
    *root*."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             str(root / "scripts" / "update_watchdog.py"),
                             "--install-root", str(root)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **flags)


def test_the_backup_stays_while_the_old_watchdog_of_this_install_runs(setup):
    root, home, updates, _ = setup
    watchdog = _fake_watchdog(root)
    try:
        deadline = time.monotonic() + 20
        while watchdog.pid not in (oc.watchdogs(root) or []):
            assert time.monotonic() < deadline, "the watchdog was not found"
            time.sleep(0.2)
        assert _start(root, home)[0] is None
        assert (updates / "backup" / "VERSION").is_file()
        assert oc.read_checked(root)["passed"] is True
        assert oc.tidy(root, updates, warn=print) is False
    finally:
        watchdog.kill()
        watchdog.wait(timeout=30)
    assert oc.tidy(root, updates, warn=print) is True
    assert not (updates / "backup").exists()
    assert oc.read_checked(root)["removed"] is True


def test_a_server_leaves_the_version_to_the_old_watchdog(setup, monkeypatch):
    root, home, _updates, _ = setup
    asked = []
    monkeypatch.setattr(oc, "_answers", lambda r, h, v: asked.append(v))
    monkeypatch.setattr(oc, "watchdogs", lambda r: [12345])
    monkeypatch.setattr(oc, "_tidy_now_or_later", lambda *a, **k: None)
    assert _start(root, home, server=True)[0] is None
    assert asked == []
    assert oc.read_checked(root)["passed"] is True
    (root / ".localm-update" / oc.CHECKED).unlink()
    assert _start(root, home, server=False)[0] is None
    assert asked == [NEW]


# ---------------------------------------------------------------- failed

def test_a_changed_file_puts_this_install_s_backup_back_with_the_reason(setup):
    root, home, updates, put_back = setup
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    note, _said, _warned = _start(root, home)
    assert put_back == [None]
    assert note and "localm/x.py" in note and "undone" in note
    result = _result(home)
    assert result["state"] == "old" and result["from"] == OLD and result["to"] == NEW
    assert "localm/x.py" in result["note"]
    checked = oc.read_checked(root)
    assert checked["passed"] is False and checked["reasons"][0][0] == "files"


def test_a_failed_way_back_is_said_and_recorded(setup, monkeypatch):
    root, home, _updates, _ = setup
    (root / "README.md").unlink()

    def refuse(**_k):
        raise RuntimeError("the backup is incomplete")
    monkeypatch.setattr(updater, "rollback_last", refuse)
    note, _said, _warned = _start(root, home)
    assert note and "could not be undone" in note and "the backup is incomplete" in note
    assert _result(home)["state"] == "failed"


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
    assert _result(home)["state"] == "failed"


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


def test_the_checks_wait_for_the_update_steps_still_to_run(setup, monkeypatch):
    root, home, updates, put_back = setup
    monkeypatch.setattr(oc, "_steps_pending", lambda: True)
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    assert _start(root, home)[0] is None and put_back == []
    assert oc.read_checked(root) == {} and (updates / "backup").is_dir()


def test_a_second_start_does_not_run_the_checks_beside_the_first(setup):
    root, home, _updates, put_back = setup
    (root / ".localm-update").mkdir()
    (root / ".localm-update" / oc.CHECKING).write_text("1", encoding="utf-8")
    (root / "localm" / "x.py").write_text("print('changed')\n", encoding="utf-8")
    assert _start(root, home)[0] is None and put_back == []
    old = time.time() - oc.CHECKING_STALE_S - 5
    os.utime(root / ".localm-update" / oc.CHECKING, (old, old))
    assert _start(root, home)[0] is not None and put_back == [None]
    assert not (root / ".localm-update" / oc.CHECKING).exists()


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


# ---------------------------------------------------------------- rollback.*

def _load_rb():
    spec = importlib.util.spec_from_file_location(
        "rollback_update_old_clients", REPO / "scripts" / "rollback_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rollback_script_refuses_a_backup_whose_update_passed_its_checks(
        tmp_path, monkeypatch, capsys):
    root = _install(tmp_path)
    shutil.copy2(REPO / "localm" / "updater_old_clients.py",
                 root / "localm" / "updater_old_clients.py")
    home = tmp_path / "home"
    updates = _old_backup(home, root, state_root=None)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    rb = _load_rb()
    assert rb._passed_backup(root, updates) is None
    oc._record(root, NEW, oc.backup_identity(updates), passed=False, removed=False,
               reasons=[["files", "x"]])
    assert rb._passed_backup(root, updates) is None
    oc._record(root, NEW, oc.backup_identity(updates), passed=True, removed=False,
               reasons=[])
    assert rb._passed_backup(root, updates) == NEW
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert rb.main(["--yes"], install_root=root) == 1
    assert "its checks passed" in capsys.readouterr().err
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert (root / "localm" / "x.py").read_text(encoding="utf-8") == "print('new')\n"
