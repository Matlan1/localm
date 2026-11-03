# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nothing in localm starts a program it looks up on PATH.

Scans every function in every ``localm`` module (the scanner and its call graph
come from test_update_path_runs_no_bare_program) and fails on a process start
whose program is a bare name (``"git"``, ``"taskkill"``) or a ``which()``
result instead of ``sys.executable``, a program localm brought, or a full path
to an operating-system program. The sites in ALLOWED run the user's own tools
by design or are argument lists, not commands; each carries the reason.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_update_path_runs_no_bare_program import (
    _call_sinks, _Graph, _sinks, bare_program_lines)

ROOT = Path(__file__).resolve().parents[1]

_CODER_USER_TOOLS = "the coder runs the user's own tools in the user's project"
_ARGUMENTS = "an argument list for a program chosen elsewhere, not a command"

# ``module:qualname`` -> why a bare program there is right.
ALLOWED = {
    "localm.plugins.coder.doctor:_git": _CODER_USER_TOOLS,
    "localm.plugins.coder.tools.git:tool_git_diff": _ARGUMENTS,
    "localm.plugins.coder.tools.git:tool_git_log": _ARGUMENTS,
    "localm.plugins.coder.tools.git:tool_git_push": _ARGUMENTS,
    "localm.plugins.coder.shell_guard:classify_git_push": _ARGUMENTS,
    "localm.plugins.coder.tools.shell:_detect_test_runner": _CODER_USER_TOOLS,
    "localm.plugins.coder.tools.shell:tool_run_tests": _CODER_USER_TOOLS,
    "localm.plugins.gui.routes.comfy:register": _ARGUMENTS,
    "localm.plugins.gui.routes.models.acquisition:register": _ARGUMENTS,
    "localm.plugins.gui.routes.runtime:register": _ARGUMENTS,
    "localm.plugins.mcpserver.cli:main": _ARGUMENTS,
}


def _module_names(root: Path) -> list:
    out = []
    for p in sorted((root / "localm").rglob("*.py")):
        if "__pycache__" in p.parts or "static" in p.parts:
            continue
        parts = list(p.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        out.append(".".join(parts))
    return out


def bare_program_sites(root: Path = ROOT) -> list:
    """``module:qualname:line`` for every bare-program process start in localm."""
    graph = _Graph(root)
    reached = {}
    for name in _module_names(root):
        mod = graph.module(name)
        if mod is not None:
            reached.update({(name, q): fn for q, fn in mod.functions.items()})
    sinks = _sinks(graph, reached)
    sites = []
    for (mod_name, qual), fn in sorted(reached.items(), key=lambda kv: kv[0]):
        calls = _call_sinks(graph, sinks, graph.module(mod_name), fn, qual)
        sites += [f"{mod_name}:{qual}:{line}" for line in bare_program_lines(fn, calls)]
    return sites


def _fake_tree(tmp_path, files: dict) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_the_scan_covers_every_module_and_follows_wrappers(tmp_path):
    root = _fake_tree(tmp_path, {
        "localm/__init__.py": "",
        "localm/a.py": "import subprocess\ndef f():\n    subprocess.run(['git', 'pull'])\n",
        "localm/sub/__init__.py": ("from localm.sub import b\n"
                                   "def g():\n    b._wrap(['taskkill', '/PID', '1'])\n"),
        "localm/sub/b.py": ("import subprocess, sys\n"
                            "def _wrap(cmd):\n    return subprocess.run(cmd)\n"
                            "def ok():\n    subprocess.run([sys.executable, '-m', 'x'])\n"),
    })
    assert bare_program_sites(root) == ["localm.a:f:3", "localm.sub:g:3"]


def test_nothing_in_localm_starts_a_program_from_path():
    unexpected = [s for s in bare_program_sites()
                  if s.rpartition(":")[0] not in ALLOWED]
    assert unexpected == []


def test_every_allowed_site_still_exists():
    seen = {s.rpartition(":")[0] for s in bare_program_sites()}
    assert set(ALLOWED) <= seen
