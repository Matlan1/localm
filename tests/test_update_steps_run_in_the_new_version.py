# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runtime and component steps of an update are carried out by the installed
(new) version at its own start, never by the process that applied the update:
every ``updater.apply`` call site defers them, and a server restart after an update
this process applied hands them to the re-executed version."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from localm import updater, updater_components, updater_runtime

ROOT = Path(__file__).resolve().parent.parent


def _apply_calls():
    for path in sorted((ROOT / "localm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "apply" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("updater", "_updater")):
                yield path.relative_to(ROOT).as_posix(), node


def test_every_update_path_leaves_the_runtime_and_components_to_the_new_version():
    calls = list(_apply_calls())
    assert len(calls) >= 4, calls
    for where, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        assert "defer_runtime" in kw, f"{where}:{node.lineno} does not defer the update steps"
        assert isinstance(kw["defer_runtime"], ast.Constant) and kw["defer_runtime"].value is True, \
            f"{where}:{node.lineno}"


@pytest.fixture
def updates_dir(tmp_path, monkeypatch):
    d = tmp_path / "updates"
    d.mkdir()
    monkeypatch.setattr(updater, "_updates_dir", lambda create=True: d)
    monkeypatch.setattr(updater_components, "_updates_dir", lambda create=True: d)
    monkeypatch.setattr(updater, "_APPLY_TOKEN", None)
    return d


def _pending_runtime(applied_by=None):
    p = {"backend": "vulkan", "current": "b1", "target": "b2", "pins": {}}
    updater_runtime.write_pending(p, root=".", version="9.9.9", applied_by=applied_by)
    return updater_runtime.read_pending()


def test_a_pending_record_names_the_process_that_applied_the_update(updates_dir):
    assert updater.applied_here(_pending_runtime()) is False
    token = updater._apply_token()
    assert updater._apply_token() == token
    assert updater.applied_here(_pending_runtime(applied_by=token)) is True
    assert updater.applied_here(_pending_runtime(applied_by="another process")) is False
    updater_components.write_pending([{"name": "sdcpp", "update": True}], root=".",
                                     version="9.9.9", applied_by=token)
    assert json.loads((updates_dir / updater_components.PENDING_FILE).read_text(
        encoding="utf-8"))["applied_by"] == token
    assert updater.applied_here(None, updater_components.read_pending()) is True
    assert updater.applied_here(None, None) is False


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
    return hs, seen


def test_a_restart_after_an_update_this_process_applied_leaves_the_steps_to_the_new_version(
        restart):
    hs, seen = restart
    _pending_runtime(applied_by=updater._apply_token())
    with pytest.raises(SystemExit):
        hs._do_restart()
    assert seen == {"finished": 0, "steps_done": False}


def test_a_restart_of_the_installed_version_carries_the_steps_out(restart):
    hs, seen = restart
    _pending_runtime(applied_by="the process that applied it")
    with pytest.raises(SystemExit):
        hs._do_restart()
    assert seen == {"finished": 1, "steps_done": True}
