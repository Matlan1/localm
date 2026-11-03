# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native-fault trace localm arms must not itself crash the process.

On Windows, faulthandler's exception handler also runs for first-chance
exceptions that the raising code goes on to handle (WebView2 raises and handles
0x8001010D on the app window's UI thread whenever a UI Automation client such
as the on-screen keyboard queries it). That thread does not hold the GIL, so
an all-thread dump walks other threads' frames while they run and can fault.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

from localm import debuglog

REPO = pathlib.Path(__file__).resolve().parent.parent

_CHILD = r'''
import ctypes, faulthandler, os, sys, threading, time
from localm import bugreport

home = sys.argv[1]
ctypes.WinDLL("kernel32").SetErrorMode(0x0001 | 0x0002 | 0x8000)
assert bugreport.arm_crash_guard(home=home, instance_id="fault-trace-test")
assert faulthandler.is_enabled()

stop = time.monotonic() + float(sys.argv[2])

def deep(n):
    return deep(n - 1) + 1 if n else 0

def busy():
    while time.monotonic() < stop:
        sum(x for x in range(50))
        deep(40)

def churn():
    while time.monotonic() < stop:
        ts = [threading.Thread(target=deep, args=(10,)) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

workers = [threading.Thread(target=busy, daemon=True) for _ in range(2)]
workers.append(threading.Thread(target=churn, daemon=True))
for w in workers:
    w.start()

raise_exc = ctypes.WinDLL("kernel32").RaiseException
raise_exc.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
raise_exc.restype = None
raised = 0
while time.monotonic() < stop:
    try:
        raise_exc(0x8001010D, 0, 0, None)
    except OSError as e:
        raised += 1
        if "access violation" in str(e).lower():
            print(f"RAISED={raised} ACCESS_VIOLATION={e}")
            sys.exit(3)
for w in workers:
    w.join(timeout=5)
print(f"RAISED={raised}")
'''


@pytest.mark.skipif(sys.platform != "win32", reason="Windows exception handler only")
def test_windows_fault_trace_survives_handled_exceptions(tmp_path):
    """Handled first-chance exceptions raised on a thread without the GIL, while
    other threads run Python and start and end, never fault the trace that
    arm_crash_guard armed."""
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONFAULTHANDLER", "LOCALM_DEBUG")}
    env.update({"LOCALM_HOME": str(home), "LOCALM_KEEP_DIAGNOSTICS": "1",
                "LOCALM_MODE": "log", "PYTHONPATH": str(REPO)})
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(home), "4"], cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW)
    traces = list((home / "run").glob("server-crash-trace.*.txt"))
    trace = "".join(t.read_text(encoding="utf-8", errors="replace") for t in traces)

    assert "ACCESS_VIOLATION" not in proc.stdout, proc.stdout
    assert "access violation" not in trace.lower(), trace[-2000:]
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-2000:])
    raised = int(proc.stdout.strip().rsplit("RAISED=", 1)[1].split()[0])
    assert raised >= 20, proc.stdout
    assert trace.count("Windows fatal exception: code 0x8001010d") == raised


def test_fault_trace_all_threads_is_false_only_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert debuglog.fault_trace_all_threads() is False
    for other in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", other)
        assert debuglog.fault_trace_all_threads() is True


def _faulthandler_enable_calls():
    for path in sorted((REPO / "localm").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "enable"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "faulthandler"):
                yield path.relative_to(REPO).as_posix(), node


def test_every_faulthandler_enable_takes_all_threads_from_the_helper():
    """Every faulthandler.enable call in localm passes
    all_threads=fault_trace_all_threads(), so no armed trace dumps every thread
    from Windows' exception handler."""
    calls = list(_faulthandler_enable_calls())
    assert len(calls) >= 6, calls
    bad = []
    for where, node in calls:
        kw = {k.arg: k.value for k in node.keywords}
        value = kw.get("all_threads")
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "fault_trace_all_threads"):
            bad.append(f"{where}:{node.lineno}")
    assert bad == []
