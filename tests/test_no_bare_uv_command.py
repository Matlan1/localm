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
_BARE = ("uv", "uv.exe")
_SHELL_CALLS = {("subprocess", name) for name in
                ("run", "call", "check_call", "check_output", "Popen")} | {("os", "system")}


def bare_uv_commands(source: str) -> list[int]:
    """Line numbers of a command argv that begins with a bare uv: a list or tuple
    whose first element is the string ``uv``, or a shell command string passed to
    subprocess / os.system that begins with ``uv ``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            first = node.elts[0]
            if isinstance(first, ast.Constant) and first.value in _BARE:
                hits.append(node.lineno)
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and (func.value.id, func.attr) in _SHELL_CALLS):
                arg = node.args[0]
                if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and arg.value.split(" ", 1)[0] in _BARE):
                    hits.append(node.lineno)
    return sorted(hits)


@pytest.mark.parametrize("source, expected", [
    ('subprocess.run(["uv", "pip", "install", "x"])', [1]),
    ('cmd = ("uv", "venv")', [1]),
    ('attempts = [["uv.exe", "pip"], [sys.executable, "-m", "pip"]]', [1]),
    ('os.system("uv pip install x")', [1]),
    ('subprocess.run("uv pip install x", shell=True)', [1]),
    ('subprocess.run([uv, "pip", "install", "x"])', []),
    ('subprocess.run([sys.executable, "-m", "pip", "install"])', []),
    ('if name == "uv": pass', []),
    ('label = ["pip", "uv"]', []),
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
