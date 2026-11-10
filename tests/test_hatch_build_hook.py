"""The console-scripts metadata hook (hatch_build.py): real files, real locks."""
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    import hatchling.metadata.plugin.interface  # noqa: F401
except ImportError:
    # The build backend is not a test dependency; the hook only subclasses this.
    for _name in ("hatchling", "hatchling.metadata", "hatchling.metadata.plugin",
                  "hatchling.metadata.plugin.interface"):
        sys.modules[_name] = types.ModuleType(_name)
    sys.modules["hatchling.metadata.plugin.interface"].MetadataHookInterface = object
import hatch_build  # noqa: E402
from tests._exe_stub import write_exe_stub  # noqa: E402

windows_only = pytest.mark.skipif(os.name != "nt", reason="locks by share mode")
SCRIPTS = {"localm": "localm.cli:console_main", "localcoder": "x:y"}
RECORD = ("../../Scripts/localcoder.exe,sha256=a,1\r\n../../Scripts/localm.exe,sha256=b,2\r\n"
          "localm-0.3.2.dist-info/METADATA,sha256=c,3\r\n")


@pytest.fixture
def install(tmp_path):
    (tmp_path / ".venv/Scripts").mkdir(parents=True)
    dist = tmp_path / ".venv/Lib/site-packages/localm-0.3.2.dist-info"
    dist.mkdir(parents=True)
    (dist / "RECORD").write_bytes(RECORD.encode())
    for n in ("localm.exe", "localcoder.exe"):
        (tmp_path / ".venv/Scripts" / n).write_bytes(b"launcher")
    return tmp_path


def _record(root):
    return (root / ".venv/Lib/site-packages/localm-0.3.2.dist-info/RECORD").read_bytes().decode()


def _hold(path):
    """A handle that refuses writers, as a running launcher does. Windows only."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    h = k32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert h not in (None, wintypes.HANDLE(-1).value)
    return lambda: k32.CloseHandle(wintypes.HANDLE(h))


def test_nothing_locked_changes_nothing(install, monkeypatch):
    told = []
    monkeypatch.setattr(hatch_build, "_tell_console", told.append)
    assert hatch_build.scripts_for_this_build(install, SCRIPTS) == SCRIPTS
    assert _record(install) == RECORD
    assert told == []


def test_no_venv_changes_nothing(tmp_path):
    assert hatch_build.scripts_for_this_build(tmp_path, SCRIPTS) == SCRIPTS
    assert list(tmp_path.iterdir()) == []


@windows_only
@pytest.mark.parametrize("held, kept_line", [("localm", "localcoder.exe"), ("localcoder", "localm.exe")])
def test_a_locked_launcher_leaves_the_build_and_its_record(install, monkeypatch, held, kept_line):
    told = []
    monkeypatch.setattr(hatch_build, "_tell_console", told.append)
    exe = install / ".venv/Scripts" / f"{held}.exe"
    release = _hold(exe)
    try:
        got = hatch_build.scripts_for_this_build(install, SCRIPTS)
    finally:
        release()
    assert held not in got and len(got) == 1
    text = _record(install)
    assert f"Scripts/{held}.exe" not in text and f"Scripts/{kept_line}" in text and "METADATA" in text
    assert "\r\n" in text
    assert exe.read_bytes() == b"launcher"
    assert len(told) == 1 and f"{held}.exe" in told[0]


@windows_only
def test_a_running_launcher_is_left_out_of_the_build(install, monkeypatch):
    """``Scripts/localm.exe`` running as a program, as when ``localm update`` or the
    global command starts the update."""
    told = []
    monkeypatch.setattr(hatch_build, "_tell_console", told.append)
    venv = install / ".venv"
    exe = venv / "Scripts" / "localm.exe"
    write_exe_stub(exe)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys._base_executable).parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8")
    proc = subprocess.Popen([str(exe), "-c", "import sys, time; print('up', flush=True); "
                             "time.sleep(120)"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        assert proc.stdout.readline().strip() == b"up"
        got = hatch_build.scripts_for_this_build(install, SCRIPTS)
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert got == {"localcoder": "x:y"}
    text = _record(install)
    assert "Scripts/localm.exe" not in text and "Scripts/localcoder.exe" in text
    assert len(told) == 1 and "localm.exe" in told[0]


@windows_only
def test_both_locked_and_a_second_pass_says_nothing_more(install, monkeypatch):
    told = []
    monkeypatch.setattr(hatch_build, "_tell_console", told.append)
    releases = [_hold(install / ".venv/Scripts" / f"{n}.exe") for n in ("localm", "localcoder")]
    try:
        assert hatch_build.scripts_for_this_build(install, SCRIPTS) == {}
        assert hatch_build.scripts_for_this_build(install, SCRIPTS) == {}
    finally:
        for r in releases:
            r()
    assert len(told) == 1


def _hook(root):
    install_root = str(root)

    class Hook(hatch_build.LocalmScriptsHook):
        def __init__(self):
            pass
        root = install_root
        config = {"console-scripts": dict(SCRIPTS)}
    return Hook()


def test_the_hook_sets_the_scripts(install):
    metadata = {}
    _hook(install).update(metadata)
    assert metadata["scripts"] == SCRIPTS


@windows_only
def test_any_failure_in_the_hook_leaves_things_as_they_are(install, monkeypatch, capsys):
    def broken(*a):
        raise PermissionError("no")
    monkeypatch.setattr(hatch_build, "_drop_from_record", broken)
    release = _hold(install / ".venv/Scripts/localm.exe")
    metadata = {}
    try:
        _hook(install).update(metadata)
    finally:
        release()
    assert metadata["scripts"] == SCRIPTS
    assert _record(install) == RECORD
    assert "leaving the launchers as they are" in capsys.readouterr().out


@pytest.mark.parametrize("fails", [subprocess.TimeoutExpired("python", 20), OSError("cannot start")])
def test_the_daemon_stop_failing_never_fails_the_build(install, monkeypatch, capsys, fails):
    (install / ".venv/Scripts/python.exe").write_bytes(b"x")
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        raise fails
    monkeypatch.setattr(hatch_build.subprocess, "run", run)
    hatch_build.stop_own_probe_daemon(install)
    assert len(calls) == 1 and calls[0][1]["timeout"] == hatch_build.STOP_TIMEOUT_S
    assert "could not stop the runtime probe" in capsys.readouterr().out


def test_the_daemon_stop_leaves_the_rebuild_marker_before_it_stops_anything(install, monkeypatch):
    (install / ".venv/Scripts/python.exe").write_bytes(b"x")
    seen = []
    monkeypatch.setattr(hatch_build.subprocess, "run", lambda cmd, **kw: seen.append(
        (install / ".venv" / hatch_build.REBUILD_MARKER).is_file()))
    hatch_build.stop_own_probe_daemon(install)
    assert seen == [True]


def test_no_existing_install_starts_no_process(tmp_path, monkeypatch):
    monkeypatch.setattr(hatch_build.subprocess, "run", lambda *a, **k: pytest.fail("started a process"))
    hatch_build.stop_own_probe_daemon(tmp_path)
    (tmp_path / ".venv/Scripts").mkdir(parents=True)
    (tmp_path / ".venv/Scripts/python.exe").write_bytes(b"x")
    (tmp_path / ".venv/Lib/site-packages").mkdir(parents=True)
    hatch_build.stop_own_probe_daemon(tmp_path)


def test_tell_console_with_no_console_writes_nothing(monkeypatch):
    def no_console(*a, **k):
        raise OSError(6, "no console")
    monkeypatch.setattr("builtins.open", no_console)
    hatch_build._tell_console("x")


def test_pyproject_declares_the_hook_and_ships_it():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" in data["project"]["dynamic"] and "scripts" not in data["project"]
    assert "hatch_build.py" in data["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
