# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runtime and component steps of an update are carried out by the installed
(new) version at its own start, never by the process that applied the update:
every ``updater.apply`` call site defers them, a server restart in a process
older than the installed version hands them to the re-executed version, and a
failure of those steps at the new version's start rolls the update back."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from localm import _version, updater, updater_components, updater_runtime
from localm.cli import _core

ROOT = Path(__file__).resolve().parent.parent


def _is_updater_apply(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "apply"
            and isinstance(node.value, ast.Name) and node.value.id in ("updater", "_updater"))


def _apply_calls():
    """Every call of ``updater.apply``, directly or passed as an argument to
    another call (``asyncio.to_thread(_probes_held, updater.apply, ...)``), with
    the call whose keywords reach apply."""
    for path in sorted((ROOT / "localm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_updater_apply(node.func) or any(_is_updater_apply(a) for a in node.args):
                yield path.relative_to(ROOT).as_posix(), node


def test_every_update_path_leaves_the_runtime_and_components_to_the_new_version():
    calls = list(_apply_calls())
    where = sorted({w for w, _ in calls})
    assert len(calls) >= 4 and "localm/inference/routes/admin.py" in where \
        and "localm/cli/maintenance.py" in where, [(w, n.lineno) for w, n in calls]
    for where, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert "defer_runtime" in kw, f"{where}:{node.lineno} does not defer the update steps"
        value = kw["defer_runtime"]
        assert (isinstance(value, ast.Constant) and value.value is True) or (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "_defer_update_steps"), f"{where}:{node.lineno}"


@pytest.mark.parametrize("target, running, deferred", [
    ("0.3.5", "0.3.4", True),
    ("0.3.4", "0.3.4", True),
    ("0.3.3", "0.3.4", False),
    ("nightly", "0.3.4", True),
])
def test_a_terminal_downgrade_carries_its_steps_out_itself(monkeypatch, target, running,
                                                          deferred):
    from localm.cli import maintenance
    monkeypatch.setattr(_version, "read_version", lambda: running)
    assert maintenance._defer_update_steps(target) is deferred


def test_the_scan_sees_an_apply_passed_as_an_argument():
    tree = ast.parse("asyncio.to_thread(_probes_held, updater.apply, 1, defer_runtime=True)\n"
                     "updater.apply(2)\n"
                     "updater.check()\n")
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        _is_updater_apply(n.func) or any(_is_updater_apply(a) for a in n.args))]
    assert [n.lineno for n in found] == [1, 2]


@pytest.mark.parametrize("installed, running, newer", [
    ("0.3.5", "0.3.4", True),
    ("0.3.4", "0.3.4", False),
    ("0.3.3", "0.3.4", False),
    ("nightly", "0.3.4", True),
])
def test_a_newer_installed_version_than_the_running_code_is_seen(monkeypatch, installed,
                                                                  running, newer):
    monkeypatch.setattr(_version, "AT_START", running)
    monkeypatch.setattr(_version, "read_version", lambda: installed)
    assert updater.newer_version_installed() is newer


@pytest.fixture
def updates_dir(tmp_path, monkeypatch):
    d = tmp_path / "updates"
    d.mkdir()
    monkeypatch.setattr(updater, "_updates_dir", lambda create=True: d)
    return d


@pytest.fixture
def restart(tmp_path, monkeypatch, updates_dir):
    """_do_restart with nothing loaded, capturing what it hands to the re-exec."""
    import localm.config as cfg
    import localm.inference.http_server as hs
    from localm.inference import embedder as emb
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(hs, "_engines", {})
    monkeypatch.setattr(emb, "active_requests", lambda: 1)
    seen = {"finished": 0}

    def finish():
        seen["finished"] += 1
        return False
    monkeypatch.setattr(hs, "_finish_pending_runtime_update", finish)

    def execv_restart(port, instance_id, *, update_steps_done=False):
        seen["steps_done"] = update_steps_done
        raise SystemExit(0)
    monkeypatch.setattr(hs, "_execv_restart", execv_restart)
    monkeypatch.setattr(_version, "AT_START", "0.3.4")
    return hs, seen


def test_a_restart_after_a_newer_version_was_installed_leaves_the_steps_to_it(
        restart, monkeypatch):
    hs, seen = restart
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.5")
    with pytest.raises(SystemExit):
        hs._do_restart()
    assert seen == {"finished": 0, "steps_done": False}


@pytest.mark.parametrize("installed", ["0.3.4", "0.3.3"])
def test_a_restart_of_the_installed_or_a_newer_running_version_carries_the_steps_out(
        restart, monkeypatch, installed):
    hs, seen = restart
    monkeypatch.setattr(_version, "read_version", lambda: installed)
    with pytest.raises(SystemExit):
        hs._do_restart()
    assert seen == {"finished": 1, "steps_done": True}


@pytest.fixture
def at_start(monkeypatch, updates_dir):
    """The start-time steps of cli._core with the finish and rollback recorded."""
    seen = {"rolled_back": 0}

    def rollback_last(*, installed=None, say=None):
        seen["rolled_back"] += 1
        return {"rolled_back": True, "backup": "b"}
    monkeypatch.setattr(updater, "rollback_last", rollback_last)
    return seen


def _runtime_pending(repair=False):
    updater_runtime.write_pending({"backend": "vulkan", "current": "b1", "target": "b2",
                                   "pins": {}}, root=".", version="9.9.9", repair=repair)


@pytest.mark.parametrize("state, repair, rolls_back", [
    ("failed", False, True),
    ("failed", True, False),
    ("kept", False, False),
    ("waiting", False, False),
    ("updated", False, False),
])
def test_a_runtime_step_that_fails_at_the_new_versions_start_rolls_the_update_back(
        at_start, monkeypatch, state, repair, rolls_back):
    _runtime_pending(repair=repair)
    updater_components.write_pending([{"name": "sdcpp", "update": True}], root=".",
                                     version="9.9.9")
    monkeypatch.setattr(updater_runtime, "finish_pending",
                        lambda **kw: {"state": state, "note": f"the runtime {state}"})
    if rolls_back:
        with pytest.raises(SystemExit) as exc:
            _core._finish_pending_runtime_update("run")
        assert exc.value.code == 1
        assert updater_components.read_pending() is None
    else:
        _core._finish_pending_runtime_update("run")
        assert updater_components.read_pending() is not None
    assert at_start["rolled_back"] == (1 if rolls_back else 0)


@pytest.mark.parametrize("state, repair, rolls_back", [
    ("failed", False, True),
    ("failed", True, False),
    ("waiting", False, False),
    ("updated", False, False),
])
def test_a_component_step_that_fails_at_the_new_versions_start_rolls_the_update_back(
        at_start, monkeypatch, state, repair, rolls_back):
    updater_components.write_pending([{"name": "sdcpp", "update": True}], root=".",
                                     version="9.9.9", repair=repair)
    monkeypatch.setattr(updater_components, "finish_pending",
                        lambda **kw: {"state": state, "note": f"components {state}"})
    command = sorted(updater_components.COMPONENT_COMMANDS)[0]
    if rolls_back:
        with pytest.raises(SystemExit) as exc:
            _core._finish_pending_component_update(command)
        assert exc.value.code == 1
    else:
        _core._finish_pending_component_update(command)
    assert at_start["rolled_back"] == (1 if rolls_back else 0)


def test_a_rollback_that_cannot_be_done_still_stops_the_command(at_start, monkeypatch, capsys):
    from localm.bugreport import LocalmError

    def refuse(*, installed=None, say=None):
        raise LocalmError("no backup", reason="nothing to restore")
    monkeypatch.setattr(updater, "rollback_last", refuse)
    with pytest.raises(SystemExit) as exc:
        _core._roll_back_update_at_start("the runtime failed")
    assert exc.value.code == 1


def test_a_deferred_component_update_says_when_it_happens(updates_dir):
    status, note = updater._apply_components(
        [{"name": "sdcpp", "update": True, "note": "sd.cpp master-1 -> master-2"}], True,
        klass="code", target=Path("."), version="9.9.9", defer=True, runner=None,
        say=None, roll_back=None, when="at the next start of localm")
    assert status is None and note.endswith("at the next start of localm")
    assert updater_components.read_pending() is not None


def test_a_catch_up_at_start_is_a_repair_and_never_rolls_an_update_back(at_start,
                                                                       monkeypatch):
    monkeypatch.setattr(updater_components, "catch_up", lambda root=None, *, repair=False: (
        updater_components.write_pending([{"name": "sdcpp", "update": True}], root=".",
                                         version="9.9.9", repair=repair)
        or [{"name": "sdcpp", "update": True, "note": "sd.cpp"}]))
    monkeypatch.setattr(updater_components, "finish_pending",
                        lambda **kw: {"state": "failed", "note": "components failed"})
    _core._finish_pending_component_update("gui")
    assert updater_components.read_pending()["repair"] is True
    assert at_start["rolled_back"] == 0


def test_a_preload_waits_for_the_restart_that_installs_the_update(monkeypatch):
    from localm.plugins.gui import cli as gui_cli
    import threading
    loaded = threading.Event()

    class Engine:
        def load(self):
            loaded.set()
    monkeypatch.setenv("LOCALM_FINISH_UPDATE_AFTER_START", "1")
    gui_cli._start_preload(None, Engine())
    assert not loaded.wait(1.0)
    monkeypatch.delenv("LOCALM_FINISH_UPDATE_AFTER_START")
    gui_cli._start_preload(None, Engine())
    assert loaded.wait(10.0)
