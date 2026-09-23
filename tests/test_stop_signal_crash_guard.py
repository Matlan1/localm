# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stop signal ends a crash-guarded server the way Ctrl+C does: serving winds
down and run_server() disarms the crash guard, so the crash-recovery watchdog
reads a clean stop and does not relaunch the server.

Real processes, real signals: closing a terminal sends SIGHUP, ``kill`` and
service managers send SIGTERM, and on Windows uvicorn re-raises a captured
Ctrl+Break (SIGBREAK) after its graceful shutdown. Each child runs with the
watchdog switched off (LOCALM_CRASH_WATCHDOG=off) in a throwaway LOCALM_HOME.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SERVER = HERE / "_portmux_stop_signal_server.py"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _child_env(home: Path) -> dict:
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home)
    env["LOCALM_CRASH_WATCHDOG"] = "off"
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    for var in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(var, None)
    return env


def _markers(home: Path) -> list:
    return sorted((home / "run").glob("server-crash.*.marker"))


def _wait_until_serving(proc, home: Path, port: int, timeout: float) -> Path:
    """The armed crash marker once the process is serving on *port*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited early with {proc.returncode}")
        found = _markers(home)
        if found:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    return found[0]
            except OSError:
                pass
        time.sleep(0.1)
    raise AssertionError(f"server never armed its crash guard and served on {port}")


def _stop(proc, log) -> str:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=30)
    log.close()
    return Path(log.name).read_text(encoding="utf-8", errors="replace")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("signame", ["SIGHUP", "SIGTERM"])
def test_a_posix_stop_signal_during_serving_disarms_the_crash_guard(tmp_path, signame):
    home = tmp_path / "home"
    home.mkdir()
    port = _free_port()
    log = open(tmp_path / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), str(port), "posix-stop"],
        env=_child_env(home), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    rc = None
    try:
        marker = _wait_until_serving(proc, home, port, timeout=60)
        os.kill(proc.pid, getattr(signal, signame))
        rc = proc.wait(timeout=60)
    finally:
        output = _stop(proc, log)

    assert not marker.exists(), (
        f"{signame} left {marker.name} behind, so the crash-recovery watchdog "
        f"would relaunch a server that was stopped on purpose (exit {rc}):\n{output}")
    assert rc == 0, output
    assert "run_server returned" in output


@pytest.mark.skipif(sys.platform != "win32", reason="Ctrl+Break is Windows-only")
def test_ctrl_break_during_serving_disarms_the_crash_guard(tmp_path):
    """uvicorn stops gracefully on SIGBREAK and then re-raises it; at the default
    disposition that exits the process with code 3 before run_server() disarms."""
    home = tmp_path / "home"
    home.mkdir()
    port = _free_port()
    marker = home / "run" / "server-crash.ctrl-break.marker"
    log = open(tmp_path / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), str(port), "ctrl-break", "SIGBREAK"],
        env=_child_env(home), stdout=log, stderr=subprocess.STDOUT)
    rc = None
    try:
        rc = proc.wait(timeout=90)
    finally:
        output = _stop(proc, log)

    assert "raising SIGBREAK while serving (marker armed: True)" in output, output
    assert not marker.exists(), (
        f"Ctrl+Break left {marker.name} behind, so the crash-recovery watchdog "
        f"would relaunch a server that was stopped on purpose (exit {rc}):\n{output}")
    assert rc == 0, output
    assert "run_server returned" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("signame", ["SIGHUP", "SIGTERM"])
def test_localm_gui_stopped_by_a_posix_signal_disarms_its_crash_guard(tmp_path, signame):
    """The real `localm gui` entry: closing its terminal (SIGHUP) or `kill`
    (SIGTERM) is a clean stop, not a crash."""
    home = tmp_path / "home"
    home.mkdir()
    port = _free_port()
    log = open(tmp_path / "gui.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "localm", "gui", "--no-model", "--no-browser",
         "--isolated", "-p", str(port)],
        env=_child_env(home), cwd=str(tmp_path), stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    rc = None
    try:
        marker = _wait_until_serving(proc, home, port, timeout=180)
        os.kill(proc.pid, getattr(signal, signame))
        rc = proc.wait(timeout=90)
    finally:
        output = _stop(proc, log)

    assert not marker.exists(), (
        f"{signame} left {marker.name} behind, so the crash-recovery watchdog "
        f"would relaunch a `localm gui` that was stopped on purpose "
        f"(exit {rc}):\n{output}")
    assert _markers(home) == []
    assert rc == 0, output
