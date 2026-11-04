# SPDX-License-Identifier: AGPL-3.0-or-later
"""A runtime replace against the holders of the runtime: this install's probe
daemon is stopped, even while a running localm keeps starting new ones; any other
holder keeps the refusal and is left running. Real processes hold a real DLL."""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from localm import install_manifest, updater, updater_runtime
from localm.inference.backends.llamacpp import _loader
from localm.setup_llama import runtime_dir

PROBE_MODULE = "localm.inference.backends.llamacpp._vram_probe"
win_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows file locks")

_HOLDER = """
import ctypes, sys
ctypes.WinDLL(sys.argv[-1] if sys.argv[-1].endswith('.dll') else {dll!r})
print('ready', flush=True)
sys.stdin.read()
"""


def _copy_python(dest: Path) -> Path:
    """A real python.exe at *dest*: its image path is what the identification reads."""
    dest.mkdir(parents=True)
    base = Path(sys.base_prefix)
    for f in base.glob("python*"):
        if f.suffix.lower() in (".exe", ".dll"):
            shutil.copy2(f, dest / f.name)
    for f in base.glob("vcruntime*.dll"):
        shutil.copy2(f, dest / f.name)
    return dest / "python.exe"


def _start(python: Path, code_dir: Path, args: list, dll: Path) -> subprocess.Popen:
    """Start *python* with *args* with a stub module tree holding *dll*, and
    wait until it has loaded it."""
    env = {"PYTHONHOME": sys.base_prefix, "PYTHONPATH": str(code_dir),
           "SystemRoot": os.environ.get("SystemRoot", "")}
    proc = subprocess.Popen([str(python), *args], cwd=str(code_dir),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    line = proc.stdout.readline().strip()
    if line != "ready":
        rest = proc.stdout.read() if proc.poll() is not None or proc.wait(timeout=10) is not None else ""
        pytest.fail(f"the holder did not load the DLL: {line} {rest}")
    return proc


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "install"
    lib = updater_runtime.runtime_lib(root)
    lib.mkdir(parents=True)
    dll_src = next(Path(sys.base_prefix, "DLLs").glob("libffi*.dll"))
    dll = lib / "ggml-base.dll"
    shutil.copy2(dll_src, dll)
    (lib / "marker.txt").write_text("old", encoding="utf-8")
    code = tmp_path / "code"
    mod = code / Path(*PROBE_MODULE.split("."))
    mod.parent.mkdir(parents=True)
    for d in [code / "localm", *mod.parents]:
        if code in d.parents:
            (d / "__init__.py").write_text("", encoding="utf-8")
    src = _HOLDER.format(dll=str(dll))
    mod.with_suffix(".py").write_text(src, encoding="utf-8")
    (code / "other_holder.py").write_text(src, encoding="utf-8")
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    procs: list = []
    yield {"root": root, "lib": lib, "dll": dll, "code": code, "procs": procs,
           "python": _copy_python(root / ".venv" / "localm-app"), "tmp": tmp_path}
    for p in procs:
        p.kill()
        p.wait(timeout=10)


def _probe_args():
    return ["-u", "-m", PROBE_MODULE]


def _fill(staging: Path) -> str:
    (staging / "ggml-base.dll").write_bytes(b"new build")
    (staging / "marker.txt").write_text("new", encoding="utf-8")
    return "filled"


@win_only
def test_command_line_reads_a_real_process(world):
    proc = _start(world["python"], world["code"], _probe_args(), world["dll"])
    world["procs"].append(proc)
    assert install_manifest.command_line(proc.pid).split()[-3:] == _probe_args()
    assert install_manifest.command_line(2 ** 30) is None


@win_only
def test_replace_stops_the_own_probe_daemon_while_it_keeps_respawning(world):
    stop = threading.Event()
    pids: list = []

    def parent_polling_stats():
        while not stop.is_set():
            proc = _start(world["python"], world["code"], _probe_args(), world["dll"])
            world["procs"].append(proc)
            pids.append(proc.pid)
            proc.wait()
            time.sleep(0.05)

    t = threading.Thread(target=parent_polling_stats, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 60
        while not pids and t.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pids, "the first daemon did not start"
        assert runtime_dir._files_in_use(world["lib"]), "the daemon must really hold the DLL"

        def fill_during_which_a_new_daemon_starts(staging):
            first = pids[-1]
            deadline = time.monotonic() + 60
            while (pids[-1] == first or not runtime_dir._files_in_use(world["lib"])) and (
                    time.monotonic() < deadline):
                time.sleep(0.02)
            assert pids[-1] != first, "the first daemon was stopped and a new one started"
            stop.set()
            return _fill(staging)

        assert runtime_dir._replace_target(world["lib"],
                                           fill_during_which_a_new_daemon_starts) == "filled"
        assert (world["lib"] / "marker.txt").read_text(encoding="utf-8") == "new"
        assert world["dll"].read_bytes() == b"new build"
    finally:
        stop.set()


@win_only
def test_a_probe_daemon_of_another_install_is_refused_and_left_running(world):
    python = _copy_python(world["tmp"] / "other-install" / ".venv" / "localm-app")
    proc = _start(python, world["code"], _probe_args(), world["dll"])
    world["procs"].append(proc)
    with pytest.raises(runtime_dir.RuntimeInUseError) as e:
        runtime_dir._replace_target(world["lib"], _fill)
    assert e.value.partial is False
    assert proc.poll() is None, "a process of another install is never stopped"
    assert (world["lib"] / "marker.txt").read_text(encoding="utf-8") == "old"


@win_only
def test_another_process_of_this_install_is_refused_and_left_running(world):
    proc = _start(world["python"], world["code"], ["-u", "-m", "other_holder"], world["dll"])
    world["procs"].append(proc)
    with pytest.raises(runtime_dir.RuntimeInUseError) as e:
        runtime_dir._replace_target(world["lib"], _fill)
    assert e.value.partial is False
    assert proc.poll() is None, "only the disposable probe daemon is stopped"
    assert (world["lib"] / "marker.txt").read_text(encoding="utf-8") == "old"


@win_only
def test_a_probe_daemon_next_to_a_foreign_holder_is_not_stopped(world):
    daemon = _start(world["python"], world["code"], _probe_args(), world["dll"])
    world["procs"].append(daemon)
    other = _start(world["python"], world["code"], ["-u", "-m", "other_holder"], world["dll"])
    world["procs"].append(other)
    with pytest.raises(runtime_dir.RuntimeInUseError):
        runtime_dir._replace_target(world["lib"], _fill)
    assert daemon.poll() is None and other.poll() is None


@win_only
def test_a_replace_of_another_runtime_folder_stops_nothing(world, tmp_path, monkeypatch):
    proc = _start(world["python"], world["code"], _probe_args(), world["dll"])
    world["procs"].append(proc)
    elsewhere = tmp_path / "install2"
    elsewhere.mkdir()
    monkeypatch.setattr(updater, "repo_root", lambda: elsewhere)
    with pytest.raises(runtime_dir.RuntimeInUseError):
        runtime_dir._replace_target(world["lib"], _fill)
    assert proc.poll() is None


def _procs(*rows):
    return [(pid, ppid, exe, 1000 + i) for i, (pid, ppid, exe) in enumerate(rows)]


def test_identification_follows_a_venv_launcher_to_a_base_interpreter_outside(tmp_path):
    root = tmp_path / "inst"
    launcher = str(root / ".venv" / "Scripts" / "python.exe")
    base = str(tmp_path / "shared" / "python.exe")
    probe = f"x -u -m {PROBE_MODULE}"
    procs = _procs((10, 1, launcher), (11, 10, base))
    commands = {10: probe, 11: probe}
    assert updater_runtime.own_probe_daemons(
        [{"pid": 11, "name": "python"}], root, procs=procs,
        read_command=commands.get) == [11, 10]


def test_identification_refuses_a_look_alike(tmp_path):
    root = tmp_path / "inst"
    inside = str(root / ".venv" / "Scripts" / "python.exe")
    outside = str(tmp_path / "other" / "python.exe")
    probe = f"x -u -m {PROBE_MODULE}"
    holder = [{"pid": 11, "name": "python"}]
    for procs, commands in (
            # Same command line, program and parent outside the install.
            (_procs((10, 1, outside), (11, 10, outside)), {10: probe, 11: probe}),
            # Parent inside the install, but it is not the launcher of this daemon.
            (_procs((10, 1, inside), (11, 10, outside)), {10: "x -m localm gui", 11: probe}),
            # Program inside the install, other command line.
            (_procs((11, 1, inside)), {11: "x -m localm gui"}),
            # The module name somewhere else on the command line.
            (_procs((11, 1, inside)), {11: f"x -u -m {PROBE_MODULE} --extra"}),
            # A command line that cannot be read.
            (_procs((11, 1, inside)), {})):
        assert updater_runtime.own_probe_daemons(holder, root, procs=procs,
                                                 read_command=commands.get) is None


def test_a_holder_with_no_pid_or_already_gone_needs_no_stop(tmp_path):
    assert updater_runtime.own_probe_daemons(
        [{"pid": None, "name": "?"}, {"pid": 99, "name": "gone"}], tmp_path,
        procs=[], read_command=lambda pid: pytest.fail("not asked")) == []


# ------------------------- the running app holds its daemon -------------------

class _Daemon:
    def __init__(self):
        self.killed = False

    def poll(self):
        return 1 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 1


def test_probe_daemon_held_stops_the_daemon_and_spawns_none_until_it_ends(monkeypatch):
    spawned = []
    monkeypatch.setattr(_loader, "_PROBE_PROC", None)
    monkeypatch.setattr(_loader, "_spawn_probe_daemon", lambda: spawned.append(1) or None)
    daemon = _Daemon()
    monkeypatch.setattr(_loader, "_PROBE_PROC", daemon)
    with _loader.probe_daemon_held():
        with _loader.probe_daemon_held():
            assert _loader.gpu_memory_isolated() is None
        assert _loader.gpu_devices_isolated() is None
    assert daemon.killed and spawned == []
    assert _loader._PROBE_HOLDS == 0
    assert _loader.gpu_memory_isolated() is None
    assert spawned == [1], "after the hold a query starts a daemon again"


def test_the_restart_runs_its_update_steps_with_the_daemon_held(monkeypatch):
    import localm.inference.http_server as hs
    seen = []
    monkeypatch.setattr(hs, "_finish_pending_update_steps",
                        lambda: seen.append(_loader._PROBE_HOLDS) or False)
    assert hs._finish_pending_runtime_update() is False
    assert seen == [1] and _loader._PROBE_HOLDS == 0


@pytest.mark.parametrize("spec", ["-e .[gpu]", "torch==2.6.0 --torch-backend=cu126"])
def test_the_pytorch_step_leaves_the_project_alone(tmp_path, spec):
    from localm import updater_components as uc
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1"\n', encoding="utf-8")
    _reqs, commands, _line = uc.torch_step(tmp_path, spec)
    assert commands
    for cmd in commands:
        assert "-e" not in cmd and "--editable" not in cmd
        assert not any(a.startswith(".[") or a == "." for a in cmd)
    assert commands[-1][-5:-4] == ["pyproject.toml"] and commands[-1].count("--extra") == 2
