# SPDX-License-Identifier: AGPL-3.0-or-later
"""Walking a process tree from parent pids: a pid is only a number, so a process
counts as a child only when its parent provably started no later than it. A
process whose start time cannot be read (Windows gives none for csrss.exe,
winlogon.exe, dwm.exe) is never a child, whichever walk asks; a stop ends a pid
only while it is still the process that was listed; a background server on
POSIX is asked to stop before the rest of its tree is ended."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import pytest

from localm import install_manifest, instances, updater_runtime

REPO = Path(__file__).resolve().parents[1]

SERVER, WORKER, ROOT_LAUNCHER = 9580, 9600, 9500
CSRSS, WINLOGON, DWM, FONTDRV = 3868, 1464, 1728, 1364


def _table(root: Path) -> list:
    """The process table of LAUNCH-009: a server whose pid an exited session
    manager once had, so protected system processes (no start time) carry it as
    their parent pid."""
    inside = str(root / ".venv" / "Scripts" / "python.exe")
    return [
        (4, 0, None, None),
        (ROOT_LAUNCHER, 4, inside, 100),
        (SERVER, ROOT_LAUNCHER, inside, 200),
        (WORKER, SERVER, inside, 300),
        (CSRSS, SERVER, None, None),
        (WINLOGON, SERVER, None, None),
        (DWM, WINLOGON, None, None),
        (FONTDRV, WINLOGON, None, None),
        (5000, SERVER, r"C:\Other\app.exe", 150),   # started before the server: reused pid
    ]


def test_a_process_with_no_start_time_is_never_a_child():
    by_pid = {p[0]: p for p in _table(Path("inst"))}
    assert install_manifest.process_parent(by_pid, WORKER) == SERVER
    for pid in (CSRSS, WINLOGON, DWM, FONTDRV, 5000):
        assert install_manifest.process_parent(by_pid, pid) is None, pid
    assert install_manifest.descendants(by_pid, SERVER) == [WORKER, SERVER]
    assert install_manifest.descendants(by_pid, ROOT_LAUNCHER) == [WORKER, SERVER, ROOT_LAUNCHER]
    assert updater_runtime._subtree(by_pid, SERVER) == [WORKER, SERVER]


def test_the_ancestor_walk_keeps_an_unknown_parent_and_drops_a_later_one():
    unknown = [(1, 0, None, None), (2, 1, None, 10), (3, 2, None, 20)]
    assert install_manifest._ancestors(unknown, 3) == {1, 2, 3}
    reused = [(1, 0, None, 5), (2, 1, None, 30), (3, 2, None, 20)]
    assert install_manifest._ancestors(reused, 3) == {3}


def test_the_update_never_counts_a_protected_process_into_the_servers_tree(tmp_path):
    root = tmp_path / "inst"
    procs = _table(root)
    tops, foreign = updater_runtime.classify_holders(
        [{"pid": WORKER, "name": "python.exe"}], root, home=tmp_path / "home",
        procs=procs, self_pid=7)
    assert foreign == []
    assert list(tops) == [ROOT_LAUNCHER]
    tree = tops[ROOT_LAUNCHER]["tree"]
    assert tree == [WORKER, SERVER, ROOT_LAUNCHER]
    assert not {CSRSS, WINLOGON, DWM, FONTDRV, 5000} & set(tree)
    assert tops[ROOT_LAUNCHER]["starts"] == {WORKER: 300, SERVER: 200, ROOT_LAUNCHER: 100}


def test_a_protected_process_is_never_taken_for_a_probe_daemons_launcher(tmp_path):
    root = tmp_path / "inst"
    inside = str(root / ".venv" / "Scripts" / "python.exe")
    outside = str(tmp_path / "shared" / "python.exe")
    probe = "x -u -m localm.inference.backends.llamacpp._vram_probe"
    procs = [(10, 1, inside, None), (11, 10, outside, 200)]
    commands = {10: probe, 11: probe}
    assert updater_runtime.own_probe_daemons(
        [{"pid": 11, "name": "python"}], root, procs=procs, read_command=commands.get) is None
    assert updater_runtime.running_own_probe_daemons(
        root, procs=procs, read_command=commands.get) == []


def test_the_comfy_stop_walks_the_same_guarded_tree(tmp_path, monkeypatch):
    from localm import updater_components
    root = tmp_path / "comfyui"
    (root / "venv").mkdir(parents=True)
    py = str(root / "venv" / "python.exe")
    git = str(tmp_path / "git" / "git.exe")
    monkeypatch.setattr(install_manifest, "list_processes", lambda: [
        (50, 1, py, 100), (51, 50, git, 110), (CSRSS, 50, None, None)])
    stopped = []
    monkeypatch.setattr(install_manifest, "_stop",
                        lambda pid, timeout=0, start=None: stopped.append((pid, start)) or True)
    assert updater_components._stop_comfy(root, lambda m: None) == []
    assert stopped == [(51, 110), (50, 100)]


def test_ps_elapsed_times_become_start_times():
    assert install_manifest._etime_seconds("05:07") == 307
    assert install_manifest._etime_seconds("01:02:03") == 3723
    assert install_manifest._etime_seconds("2-01:02:03") == 2 * 86400 + 3723
    for bad in ("", "x", "1:2:3:4", "a-01:02", "--1:00"):
        assert install_manifest._etime_seconds(bad) is None, bad
    rows = install_manifest._ps_processes(
        ["  1     0  3-00:00:00 /sbin/launchd",
         " 40     1       00:10 /Applications/LocaLM/.venv/bin/python3",
         " 41    40       00:05 python3 -m something",
         " 42    40        ?? /usr/bin/x"], now=1_000_000)

    def exe(p):
        return p if os.path.isabs(p) else None
    assert rows == [(1, 0, exe("/sbin/launchd"), 1_000_000 - 3 * 86400),
                    (40, 1, exe("/Applications/LocaLM/.venv/bin/python3"), 1_000_000 - 10),
                    (41, 40, None, 1_000_000 - 5),
                    (42, 40, exe("/usr/bin/x"), None)]


def _sleeper() -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"],
                            creationflags=flags)


@pytest.mark.skipif(not (sys.platform == "win32" or os.path.isdir("/proc")),
                    reason="start times are read on Windows and Linux")
def test_a_stop_leaves_a_process_with_another_start_time_alone():
    proc = _sleeper()
    try:
        deadline = time.monotonic() + 30
        start = None
        while start is None and time.monotonic() < deadline:
            start = next((p[3] for p in install_manifest.list_processes() or []
                          if p[0] == proc.pid), None)
            time.sleep(0.05)
        assert isinstance(start, int), "the start time of a process of ours is read"
        assert install_manifest._stop(proc.pid, timeout=2, start=start + 1) is True
        time.sleep(0.3)
        assert proc.poll() is None, "a pid with another start time is not ended"
        assert install_manifest._stop(proc.pid, timeout=10, start=start) is True
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# A stand-in background server: it starts a worker child, and on SIGTERM writes
# whether that worker was still running, then exits.
_SERVER = (
    "import os, signal, subprocess, sys, time\n"
    "w = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'])\n"
    "def bye(*_):\n"
    "    try:\n"
    "        os.kill(w.pid, 0)\n"
    "        state = 'running' if w.poll() is None else 'gone'\n"
    "    except OSError:\n"
    "        state = 'gone'\n"
    "    open(sys.argv[1], 'w').write(state)\n"
    "    w.kill(); w.wait()\n"
    "    os._exit(0)\n"
    "signal.signal(signal.SIGTERM, bye)\n"
    "print(w.pid, flush=True)\n"
    "while True: time.sleep(0.1)\n")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX asks a server to stop by SIGTERM")
def test_a_background_server_is_asked_to_stop_before_its_workers(tmp_path, monkeypatch):
    home, root = tmp_path / "home", tmp_path / "inst"
    root.mkdir()
    said = tmp_path / "said"
    server = subprocess.Popen([sys.executable, "-I", "-c", _SERVER, str(said)],
                              stdout=subprocess.PIPE, text=True)
    worker = int(server.stdout.readline())
    try:
        run = instances.run_dir(home)
        run.mkdir(parents=True)
        (run / "x.json").write_text(json.dumps({
            "instance_id": "x", "pid": server.pid, "port": 1, "host": "127.0.0.1",
            "root_dir": str(root), "mode": "api", "token": "t",
            "started": datetime.now(UTC).isoformat()}), encoding="utf-8")
        calls = []
        monkeypatch.setattr(updater_runtime, "holders",
                            lambda lib: calls.append(1) or (
                                [{"pid": worker, "name": "python"}] if len(calls) == 1 else []))
        assert updater_runtime.stop_own_holders(tmp_path, root, home=home, wait=5.0) == []
        assert server.wait(timeout=10) == 0
        assert said.read_text() == "running", "the server went first, its worker after"
        assert not (run / "x.json").exists()
    finally:
        if server.poll() is None:
            server.kill()
            server.wait(timeout=10)
        try:
            os.kill(worker, signal.SIGKILL)
        except OSError:
            pass


def test_only_install_manifest_reads_a_parent_pid():
    """Every walk over parent pids goes through install_manifest.process_parent:
    no other module reads the parent column of a process row."""
    offenders = []
    for path in (REPO / "localm").rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel == "localm/install_manifest.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "list_processes" not in text:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if re.search(r"\]\[1\]|\b(p|row|proc)\[1\]|\bppid\b", line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == []
