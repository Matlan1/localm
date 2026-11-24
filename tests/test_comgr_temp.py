# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm.comgr_temp``: the empty ``comgr-<pid>-...`` folders of localm's own
ended GPU processes are removed, and nothing else in the temp folder is."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm import comgr_temp

_REPO = Path(__file__).resolve().parents[1]
_LOCALM = _REPO / "localm"


@pytest.fixture
def temp(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(comgr_temp, "_home", lambda: home)
    t = tmp_path / "temp"
    t.mkdir()
    monkeypatch.setattr(comgr_temp, "_temp_folder", lambda: str(t))
    return t


@pytest.fixture
def process(temp):
    """A recorded process: (pid, end) where end() kills it and returns."""
    procs = []

    def start():
        proc = subprocess.Popen(
            [sys.executable, "-c", "import os,sys; print(os.getpid(), flush=True); sys.stdin.read()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        procs.append(proc)
        pid = int(proc.stdout.readline())
        comgr_temp._record(pid)

        def end():
            import psutil
            for p in psutil.Process(proc.pid).children(recursive=True):
                p.kill()
            proc.kill()
            proc.wait(30)
            deadline = time.monotonic() + 10
            while psutil.pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        return pid, end
    yield start
    for proc in procs:
        proc.kill()
        proc.wait(30)


def _folder(temp, pid, tail="2-abcdef"):
    d = temp / f"comgr-{pid}-{tail}"
    (d / "output").mkdir(parents=True)
    return d


def _records():
    d = comgr_temp.records_dir()
    return list(d.glob("*.json")) if d.is_dir() else []


def test_the_folders_of_a_running_process_stay_and_go_when_it_ends(temp, process):
    pid, end = process()
    d = _folder(temp, pid)
    assert comgr_temp.sweep() == 1 and d.is_dir()
    end()
    assert comgr_temp.sweep() == 0
    assert not d.exists() and _records() == []


def test_folders_of_processes_localm_did_not_record_are_never_touched(temp, process):
    pid, end = process()
    others = [_folder(temp, os.getpid()), _folder(temp, int(f"{pid}7")), _folder(temp, 2 ** 30)]
    end()
    comgr_temp.sweep()
    assert all((d / "output").is_dir() for d in others)


def test_a_folder_that_holds_a_file_stays(temp, process):
    pid, end = process()
    d = _folder(temp, pid)
    (d / "output" / "kernel.o").write_bytes(b"x")
    end()
    comgr_temp.sweep()
    assert (d / "output" / "kernel.o").is_file()


def _link(link: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_nothing_is_followed_or_removed_through_a_link(temp, process, tmp_path):
    pid, end = process()
    target = tmp_path / "target"
    (target / "output").mkdir(parents=True)
    _link(temp / f"comgr-{pid}-9-link", target)
    inner = temp / f"comgr-{pid}-8-inner"
    inner.mkdir()
    _link(inner / "output", target)
    end()
    comgr_temp.sweep()
    assert (target / "output").is_dir()
    assert os.path.lexists(temp / f"comgr-{pid}-9-link") and os.path.lexists(inner / "output")


def test_a_folder_older_or_younger_than_the_process_is_not_its_own(temp, process):
    pid, end = process()
    old, young, own = _folder(temp, pid, "1-old"), _folder(temp, pid, "1-new"), _folder(temp, pid)
    now = time.time()
    os.utime(old, (now - 3600, now - 3600))
    os.utime(young, (now + 3600, now + 3600))
    end()
    comgr_temp.sweep()
    assert old.is_dir() and young.is_dir() and not own.exists()


def test_a_pid_that_another_process_has_by_now_keeps_that_ones_folders(temp, monkeypatch):
    pid = os.getpid()
    mine = _folder(temp, pid)
    comgr_temp._record(pid)
    path = _records()[0]
    rec = json.loads(path.read_text())
    rec["start"] = {"created": 1.0} if "created" in (rec["start"] or {}) else {
        "boot": "another-boot", "ticks": 1}
    rec["created"] = 1.0
    path.write_text(json.dumps(rec))
    assert comgr_temp.sweep() == 0
    assert mine.is_dir()


def test_a_record_of_another_machine_is_left_alone(temp, process):
    pid, end = process()
    d = _folder(temp, pid)
    path = _records()[0]
    path.write_text(json.dumps({**json.loads(path.read_text()), "space": "elsewhere"}))
    end()
    comgr_temp.sweep()
    assert d.is_dir() and path.is_file()


def test_no_records_folder_is_nothing_to_do(temp):
    assert comgr_temp.sweep() == 0 and not comgr_temp.records_dir().exists()


@pytest.mark.parametrize("pid", ["1", 0, -5, True, 2 ** 70, 1.5, None, [1], {"a": 1},
                                 "1-*", "../1"])
def test_a_record_without_a_usable_pid_is_never_acted_on(temp, monkeypatch, pid):
    monkeypatch.setattr(comgr_temp, "_remove_folders",
                        lambda *a: pytest.fail("acted on a bad record"))
    rec = {"pid": pid, "start": None, "space": comgr_temp.procid.pid_space_id(),
           "created": 1.0}
    comgr_temp.records_dir().mkdir(parents=True)
    (comgr_temp.records_dir() / "5-a.json").write_text(json.dumps(rec), encoding="utf-8")
    assert comgr_temp.sweep() == 0


@pytest.mark.parametrize("raw", [
    b"", b"[]", b"null", b"\xff\xfe", b"[" * 4000, b'{"pid": ' + b"9" * 4000 + b"}",
    b" " * 100_000])
def test_a_file_that_is_no_record_is_ignored(temp, raw):
    comgr_temp.records_dir().mkdir(parents=True)
    (comgr_temp.records_dir() / "5-a.json").write_bytes(raw)
    (comgr_temp.records_dir() / "6-folder.json").mkdir()
    assert comgr_temp.sweep() == 0


@pytest.mark.parametrize("field,value", [
    ("created", float("nan")), ("created", "x"), ("created", [1]), ("start", "x"),
    ("start", {"created": float("inf")}), ("start", {"boot": 1, "ticks": "a"}), ("space", 7)])
def test_a_wrong_optional_field_never_raises(temp, process, field, value):
    pid, end = process()
    path = _records()[0]
    path.write_text(json.dumps({**json.loads(path.read_text()), field: value}))
    end()
    comgr_temp.sweep()


def test_a_path_in_a_record_is_not_used(temp, process, tmp_path):
    pid, end = process()
    elsewhere = tmp_path / "elsewhere"
    other, own = _folder(elsewhere, pid), _folder(temp, pid)
    path = _records()[0]
    path.write_text(json.dumps({**json.loads(path.read_text()), "temp": str(elsewhere)}))
    end()
    comgr_temp.sweep()
    assert other.is_dir() and not own.exists()


def test_one_bad_record_does_not_stop_the_others(temp, process, monkeypatch):
    pid, end = process()
    d = _folder(temp, pid)
    real = comgr_temp._is_gone

    def odd(rec):
        if rec["pid"] == 1:
            raise RuntimeError("odd record")
        return real(rec)
    monkeypatch.setattr(comgr_temp, "_is_gone", odd)
    rec = {**json.loads(_records()[0].read_text()), "pid": 1}
    (comgr_temp.records_dir() / "0-bad.json").write_text(json.dumps(rec))
    end()
    comgr_temp.sweep()
    assert not d.exists()


def test_a_flood_of_records_is_cut_off(temp, monkeypatch):
    monkeypatch.setattr(comgr_temp, "_MAX_RECORDS", 5)
    comgr_temp.records_dir().mkdir(parents=True)
    for i in range(20):
        (comgr_temp.records_dir() / f"{i}-a.json").write_text("{}")
    assert len(comgr_temp._record_paths()) == 5


def test_the_watcher_removes_them_without_being_asked(temp, process, monkeypatch):
    monkeypatch.setattr(comgr_temp, "_POLL_SECONDS", 0.05)
    monkeypatch.setattr(comgr_temp, "_IDLE_POLLS", 3)
    monkeypatch.setattr(comgr_temp, "_watcher", None)
    pid, end = process()
    d = _folder(temp, pid)
    comgr_temp.watch()
    thread = comgr_temp._watcher
    end()
    deadline = time.monotonic() + 15
    while d.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not d.exists()
    thread.join(15)
    assert not thread.is_alive()


def test_finish_waits_for_the_processes_below_this_one(temp, process):
    pid, end = process()
    d = _folder(temp, pid)
    import threading
    threading.Timer(0.5, end).start()
    comgr_temp.finish(wait=10)
    assert not d.exists()


def test_a_process_registers_once(temp, monkeypatch):
    monkeypatch.setattr(comgr_temp, "_registered_pid", None)
    comgr_temp.register()
    comgr_temp.register()
    assert [json.loads(p.read_text())["pid"] for p in _records()] == [os.getpid()]


@pytest.mark.parametrize("name,files,expected", [
    ("master-1-rocm", [], 1), ("lib", ["ggml-hip.dll"], 1), ("lib", ["amdhip64_7.dll"], 1),
    ("master-1-vulkan", ["ggml-vulkan.dll"], 0)])
def test_only_a_hip_runtime_is_recorded(temp, tmp_path, monkeypatch, name, files, expected):
    monkeypatch.setattr(comgr_temp, "_registered_pid", None)
    d = tmp_path / name
    d.mkdir()
    for f in files:
        (d / f).write_bytes(b"")
    comgr_temp.register_if_hip(d)
    assert len(_records()) == expected


def _calls(path: Path, func: str | None) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if func is not None:
        tree = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func)
    return {n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "comgr_temp"}


@pytest.mark.parametrize("file,func,call", [
    ("inference/backends/llamacpp/_loader.py", "load_lib", "register_if_hip"),
    ("media/sdcpp/_binding.py", "load_library", "register_if_hip"),
    ("inference/backends/_hf_runner.py", "prepare_worker_process", "register"),
    ("_torch_gpu_probe.py", "main", "register"),
    ("_mp_spawn.py", "ensure_spawn_uses_venv_python", "watch"),
    ("inference/backends/llamacpp/_loader.py", "_spawn_probe_daemon", "watch"),
    ("discover.py", "_ask_torch_probe_child", "watch"),
    ("media/comfy_client.py", "_kill_process_tree", "adopt_tree"),
    ("inference/http_server.py", "_finish_comgr_temp", "finish"),
])
def test_every_place_that_loads_or_starts_a_hip_runtime_takes_part(file, func, call):
    assert call in _calls(_LOCALM / file, func)


def test_starting_the_server_removes_what_a_killed_one_left(temp, process):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    pid, end = process()
    d = _folder(temp, pid)
    end()
    engine = MagicMock()
    engine.display_name = "test-model"
    with TestClient(create_app(engine)):
        assert not d.exists()
