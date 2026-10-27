# SPDX-License-Identifier: AGPL-3.0-or-later
"""No code under localm/ starts a command with a bare ``uv``.

A bare ``uv`` is looked up on PATH, which is exactly where the uv an install
uses may not be (a contained install keeps it in ``<root>/.uv``). Every command
that runs uv takes the path :func:`localm.uv_tool.find_uv` returns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_BARE = {"uv", "uv.exe"}
_EXEC_CALLS = {"run", "call", "check_call", "check_output", "Popen", "system",
               "create_subprocess_exec", "create_subprocess_shell",
               "execv", "execvp", "execvpe", "execl", "execlp", "execlpe"}


def _call_name(func) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _is_which_uv(node) -> bool:
    """``which("uv")`` / ``shutil.which("uv")``: a PATH lookup used as argv[0]."""
    return (isinstance(node, ast.Call) and _call_name(node.func) == "which" and bool(node.args)
            and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).lower() in _BARE)


def _is_bare_program(node) -> bool:
    """A program name that is a bare uv: the string, any letter case, or which("uv")."""
    return ((isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.strip().lower() in _BARE) or _is_which_uv(node))


def _starts_with_bare_uv(node) -> bool:
    """A command (argv sequence, shell string, f-string or concatenation) whose
    first word is a bare uv."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return bool(node.elts) and _is_bare_program(node.elts[0])
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        words = node.value.split()
        return bool(words) and words[0].lower() in _BARE
    if isinstance(node, ast.JoinedStr):
        first = node.values[0] if node.values else None
        return isinstance(first, ast.Constant) and _starts_with_bare_uv(first)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _starts_with_bare_uv(node.left)
    return _is_which_uv(node)


def bare_uv_commands(source: str) -> list[int]:
    """Line numbers of a command that starts with a bare uv: an argv list or tuple
    whose first element is ``uv`` (any letter case) or ``which("uv")`` wherever it
    appears, and the command argument (first positional, ``args=`` or ``program=``)
    of a process-starting call (subprocess, os.system, os.exec*, asyncio
    create_subprocess_*, under any import alias) given as a string, f-string or
    concatenation whose first word is ``uv``."""
    hits = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.List, ast.Tuple)) and _starts_with_bare_uv(node):
            hits.add(node.lineno)
        elif isinstance(node, ast.Call) and _call_name(node.func) in _EXEC_CALLS:
            cands = node.args[:1] + [k.value for k in node.keywords if k.arg in ("args", "program")]
            if any(_starts_with_bare_uv(c) for c in cands):
                hits.add(node.lineno)
    return sorted(hits)


@pytest.mark.parametrize("source, expected", [
    ('subprocess.run(["uv", "pip", "install", "x"])', [1]),
    ('cmd = ("uv", "venv")', [1]),
    ('attempts = [["uv.exe", "pip"], [sys.executable, "-m", "pip"]]', [1]),
    ('os.system("uv pip install x")', [1]),
    ('subprocess.run("uv pip install x", shell=True)', [1]),
    ('subprocess.run(f"uv pip install {pkg}", shell=True)', [1]),
    ('subprocess.run("uv " + " ".join(args), shell=True)', [1]),
    ('cmd = ["UV.EXE", "venv"]', [1]),
    ('cmd = [shutil.which("uv"), "pip", "install"]', [1]),
    ('import subprocess as sp\nsp.check_call(args=["uv", "sync"])', [2]),
    ('from subprocess import run\nrun("uv lock", shell=True)', [2]),
    ('await asyncio.create_subprocess_exec("uv", "pip", "install")', [1]),
    ('await asyncio.create_subprocess_shell("uv pip install x")', [1]),
    ('os.execvp("uv", ["uv", "run"])', [1]),
    ('subprocess.run([uv, "pip", "install", "x"])', []),
    ('subprocess.run([sys.executable, "-m", "pip", "install"])', []),
    ('if name == "uv": pass', []),
    ('label = ["pip", "uv"]', []),
    ('msg = "run uv pip install -e .[gpu] to add it"', []),
    ('hint = "uv pip install -e .[gpu]"', []),
    ('where = shutil.which("uv") or "your profile"', []),
])
def test_the_scanner_finds_what_it_claims_to(source, expected):
    assert bare_uv_commands(source) == expected


def test_no_module_under_localm_starts_a_command_with_a_bare_uv():
    found = {}
    for path in sorted((ROOT / "localm").rglob("*.py")):
        hits = bare_uv_commands(path.read_text(encoding="utf-8", errors="replace"))
        if hits:
            found[path.relative_to(ROOT).as_posix()] = hits
    assert found == {}, f"commands that search PATH for uv: {found}"
