"""No test opens a console window on the desktop (Windows).

tests/_no_console_window.py wraps process creation so a spawn from a console-less pytest process
carries CREATE_NO_WINDOW; tests/__init__.py arms it. This file pins the guard's decision, proves it
end to end from a process that really has no console, and fails when a test file adds a way to
open a window that the guard cannot see.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _no_console_window as guard

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows behaviour")


class TestCreationFlags:
    def test_a_console_less_parent_gets_no_window_added(self):
        assert guard.creation_flags_for(0, console_attached=False) == guard.CREATE_NO_WINDOW

    def test_other_flags_are_kept_and_no_window_is_added_beside_them(self):
        new_group = 0x00000200
        assert guard.creation_flags_for(new_group, console_attached=False) == new_group | guard.CREATE_NO_WINDOW

    @pytest.mark.parametrize("chosen", [guard.CREATE_NO_WINDOW, guard.DETACHED_PROCESS, guard.CREATE_NEW_CONSOLE])
    def test_a_flag_that_already_chooses_a_console_behaviour_is_left_alone(self, chosen):
        assert guard.creation_flags_for(chosen, console_attached=False) == chosen

    def test_a_parent_with_a_console_leaves_the_flags_alone(self):
        assert guard.creation_flags_for(0, console_attached=True) == 0


@windows_only
def test_the_tests_package_armed_the_guard_in_this_process():
    assert guard.is_armed()


_DETACHED_PARENT = textwrap.dedent(
    """
    import _winapi, json, os, subprocess, sys
    seen = []
    real = _winapi.CreateProcess

    def recorder(app, cmd, pa, ta, inherit, flags, *rest):
        seen.append(flags)
        if not flags & 0x08000000:
            raise RuntimeError("spawn without CREATE_NO_WINDOW from a console-less process")
        return real(app, cmd, pa, ta, inherit, flags, *rest)

    _winapi.CreateProcess = recorder
    from tests import _no_console_window as guard
    guard.arm()
    probe = "import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())"
    out = {"console_before": guard.has_console()}
    try:
        child = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        out["child_console_window"] = child.stdout.strip()
        os.system("exit 0")
        out["flags"] = seen
    except RuntimeError as e:
        out["error"] = str(e)
        out["flags"] = seen
    open(sys.argv[1], "w").write(json.dumps(out))
    """
)


@windows_only
def test_a_console_less_process_starts_children_without_a_window(tmp_path):
    """The parent is the base interpreter started DETACHED_PROCESS, so it has no console, like a harness
    launched without a terminal (the venv launcher shim would be given a console of its own)."""
    script = tmp_path / "parent.py"
    script.write_text(_DETACHED_PARENT, encoding="utf-8")
    result = tmp_path / "result.json"
    done = subprocess.run(
        [getattr(sys, "_base_executable", sys.executable), str(script), str(result)],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)},
        creationflags=guard.DETACHED_PROCESS,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=120,
    )
    out = json.loads(result.read_text(encoding="utf-8"))
    assert "error" not in out, out
    assert out["console_before"] is False, out
    assert out["flags"] and all(f & guard.CREATE_NO_WINDOW for f in out["flags"]), out
    assert out["child_console_window"] == "0", out
    assert done.returncode == 0


def _window_openers(source: str) -> list[str]:
    """What in *source* opens a window by a route that process-creation wrapping cannot see."""
    found = []
    tree = ast.parse(source)
    docstrings = {
        id(n.body[0].value) for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "os.startfile" or name.rsplit(".", 1)[-1] in {
                "ShellExecute", "ShellExecuteW", "ShellExecuteExW", "CreateProcessW", "CreateProcessA", "WinExec",
            }:
                found.append(f"line {node.lineno}: {name}()")
            for kw in node.keywords:
                if kw.arg == "creationflags":
                    text = ast.unparse(kw.value)
                    for bad in ("CREATE_NEW_CONSOLE", "DETACHED_PROCESS"):
                        if bad in text:
                            found.append(f"line {node.lineno}: creationflags uses {bad}")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
              and "creationflags" in node.value and re.search(r"\bimport\b|Popen\(", node.value)):
            text = node.value
            if re.search(r"creationflags\W*=\s*(0x0*8|0x0*10)\b|creationflags\W*=[^\n]*(DETACHED_PROCESS|CREATE_NEW_CONSOLE)", text) \
                    and "_base_executable" not in text:
                found.append(f"line {node.lineno}: script text sets creationflags to DETACHED_PROCESS or CREATE_NEW_CONSOLE")
        elif isinstance(node, (ast.List, ast.Tuple)):
            items = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            lowered = [i.lower() for i in items]
            if lowered[:2] in (["cmd", "/c"], ["cmd.exe", "/c"]) and len(lowered) > 2 and lowered[2] == "start":
                found.append(f"line {node.lineno}: cmd /c start")
    return found


class TestWindowOpenersSentinel:
    @pytest.mark.parametrize("snippet", [
        "import os\nos.startfile('x.bat')\n",
        "import ctypes\nctypes.windll.shell32.ShellExecuteW(0, 'open', 'x', None, None, 1)\n",
        "import subprocess\nsubprocess.Popen(['x'], creationflags=subprocess.CREATE_NEW_CONSOLE)\n",
        "import subprocess\nsubprocess.Popen(['x'], creationflags=subprocess.DETACHED_PROCESS | 8)\n",
        "import subprocess\nsubprocess.run(['cmd', '/c', 'start', 'x'])\n",
        "SCRIPT = \"kw['creationflags'] = 0x00000008\\nsubprocess.Popen([sys.executable, '-c', code], **kw)\"\n",
    ])
    def test_the_scan_sees_each_way_to_open_a_window(self, snippet):
        assert _window_openers(snippet)

    def test_the_scan_passes_a_detached_spawn_of_the_base_interpreter(self):
        script = "kw['creationflags'] = 0x00000008\\nsubprocess.Popen([sys._base_executable, '-c', code], **kw)"
        assert not _window_openers(f"SCRIPT = {script!r}\n")

    def test_the_scan_passes_an_ordinary_spawn(self):
        assert not _window_openers("import subprocess\nsubprocess.run(['git', 'init'], creationflags=0x200)\n")

    def test_no_test_file_opens_a_window_the_guard_cannot_see(self):
        offenders = {}
        for path in sorted(TESTS.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            hits = _window_openers(path.read_text(encoding="utf-8", errors="replace"))
            if hits:
                offenders[path.relative_to(REPO).as_posix()] = hits
        assert not offenders, offenders


def _spawns_without_creationflags(source: str) -> list[int]:
    """Line numbers of subprocess/os.system calls in *source* that pass neither ``creationflags`` nor ``**kwargs``."""
    spawns = {"subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call",
              "subprocess.check_output", "os.system"}
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in spawns:
            names = {kw.arg for kw in node.keywords}
            if "creationflags" not in names and None not in names:
                lines.append(node.lineno)
    return lines


class TestProductSpawnsFromAConsoleLessProcess:
    """launcher.pyw runs under pythonw.exe, which has no console; a child it starts without the flag opens a window."""

    def test_the_scan_sees_a_spawn_without_the_flag(self):
        assert _spawns_without_creationflags("import subprocess\nsubprocess.run(['x'])\n") == [2]
        assert not _spawns_without_creationflags("import subprocess\nsubprocess.run(['x'], creationflags=1)\n")

    def test_every_spawn_in_launcher_pyw_carries_creationflags(self):
        source = (REPO / "launcher.pyw").read_text(encoding="utf-8")
        assert _spawns_without_creationflags(source) == []
