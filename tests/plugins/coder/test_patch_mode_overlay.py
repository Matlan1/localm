# SPDX-License-Identifier: AGPL-3.0-or-later
"""Patch mode keeps a session overlay of the writes it captured: later reads,
greps and edits see them, and the patch is one diff per file from the original
to the final content, so ``git apply`` accepts it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

CALC = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n\n\nprint(mul(2, 3))\n"


def _agent(tmp_path: Path) -> Agent:
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, auto_approve=True)
    agent.patch_mode = True
    return agent


def _call(name: str, **args):
    call = MagicMock()
    call.name = name
    call.args = args
    return call


def _run(agent, name, **args):
    return agent._execute_tool(_call(name, **args), interactive=False)


def _git_apply_check(root: Path, patch_text: str) -> subprocess.CompletedProcess:
    patch_file = root.parent / (root.name + ".patch")
    patch_file.write_text(patch_text, encoding="utf-8", newline="")
    try:
        return subprocess.run(
            ["git", "apply", "--check", str(patch_file)], cwd=root,
            capture_output=True, text=True, creationflags=_NO_WINDOW)
    finally:
        patch_file.unlink(missing_ok=True)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(CALC, encoding="utf-8", newline="")
    return root


def test_two_edits_of_one_file_give_one_applicable_diff(project):
    agent = _agent(project)
    assert _run(agent, "edit_file", path="calc.py",
                old="def mul(a, b):", new="def multiply(a, b):").ok
    assert _run(agent, "edit_file", path="calc.py",
                old="print(mul(2, 3))", new="print(multiply(2, 3))").ok
    diff = agent.flush_patch()
    assert diff.count("--- a/calc.py") == 1
    assert "+def multiply(a, b):" in diff and "+print(multiply(2, 3))" in diff
    assert (project / "calc.py").read_text(encoding="utf-8") == CALC
    check = _git_apply_check(project, diff)
    assert check.returncode == 0, check.stderr


def test_the_same_edit_repeated_does_not_duplicate_the_hunk(project):
    agent = _agent(project)
    for _ in range(2):
        _run(agent, "edit_file", path="calc.py",
             old="def mul(a, b):", new="def multiply(a, b):")
    diff = agent.flush_patch()
    assert diff.count("+def multiply(a, b):") == 1
    assert _git_apply_check(project, diff).returncode == 0


def test_read_file_sees_the_captured_edit(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py",
         old="def mul(a, b):", new="def multiply(a, b):")
    out = _run(agent, "read_file", path="calc.py").output
    assert "def multiply(a, b):" in out
    assert "def mul(a, b):" not in out


def test_read_file_sees_a_file_written_in_the_session(project):
    agent = _agent(project)
    _run(agent, "write_file", path="notes.txt", content="alpha\nbeta\n")
    assert not (project / "notes.txt").exists()
    assert "beta" in _run(agent, "read_file", path="notes.txt").output


def test_grep_sees_the_captured_edit_and_not_the_replaced_text(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py",
         old="def mul(a, b):", new="def multiply(a, b):")
    assert "multiply" in _run(agent, "grep", pattern="def multiply").output
    assert "def mul(" not in _run(agent, "grep", pattern=r"def mul\(").output


def test_search_replace_after_an_edit_builds_on_the_overlay(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py",
         old="def mul(a, b):", new="def multiply(a, b):")
    assert _run(agent, "search_replace", pattern=r"\bmul\(", replacement="multiply(",
                glob="calc.py").ok
    diff = agent.flush_patch()
    assert diff.count("--- a/calc.py") == 1
    assert "print(multiply(2, 3))" in diff
    assert _git_apply_check(project, diff).returncode == 0


def test_edit_files_after_an_edit_builds_on_the_overlay(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py", old="return a + b", new="return b + a")
    assert _run(agent, "edit_files", edits=[
        {"path": "calc.py", "old": "return b + a", "new": "return a - b"}]).ok
    diff = agent.flush_patch()
    assert "+    return a - b" in diff and "b + a" not in diff
    assert _git_apply_check(project, diff).returncode == 0


def test_patch_file_after_an_edit_builds_on_the_overlay(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py", old="return a + b", new="return b + a")
    hunk = ("--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n-    return b + a\n+    return a - b\n")
    assert _run(agent, "patch_file", path="calc.py", diff=hunk).ok
    diff = agent.flush_patch()
    assert "+    return a - b" in diff and "b + a" not in diff
    assert _git_apply_check(project, diff).returncode == 0


def test_an_edit_that_restores_the_original_leaves_no_patch(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py", old="return a + b", new="return b + a")
    _run(agent, "edit_file", path="calc.py", old="return b + a", new="return a + b")
    assert agent.has_patch() is False
    assert agent.current_patch() == ""


def test_several_files_each_get_one_diff(project):
    (project / "other.py").write_text("x = 1\n", encoding="utf-8")
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py", old="return a + b", new="return b + a")
    _run(agent, "edit_file", path="other.py", old="x = 1", new="x = 2")
    _run(agent, "edit_file", path="calc.py", old="return a * b", new="return b * a")
    diff = agent.flush_patch()
    assert diff.count("--- a/calc.py") == 1 and diff.count("--- a/other.py") == 1
    assert _git_apply_check(project, diff).returncode == 0


def test_flushing_clears_the_overlay(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py",
         old="def mul(a, b):", new="def multiply(a, b):")
    agent.flush_patch()
    assert "def mul(a, b):" in _run(agent, "read_file", path="calc.py").output
    assert agent.has_patch() is False


def test_current_patch_does_not_clear_the_overlay(project):
    agent = _agent(project)
    _run(agent, "edit_file", path="calc.py",
         old="def mul(a, b):", new="def multiply(a, b):")
    first = agent.current_patch()
    assert first and agent.current_patch() == first
    assert agent.has_patch() is True


def test_a_call_that_changes_nothing_is_reported_as_skipped(project):
    agent = _agent(project)
    result = _run(agent, "edit_file", path="calc.py", old="no such text", new="x")
    assert result.ok is False
    assert agent.has_patch() is False


def test_files_without_a_final_newline_still_give_an_applicable_multifile_patch(project):
    (project / "nonl.txt").write_bytes(b"alpha\nbeta")
    (project / "other.txt").write_bytes(b"one\ntwo\n")
    agent = _agent(project)
    _run(agent, "edit_file", path="nonl.txt", old="beta", new="gamma")
    _run(agent, "edit_file", path="other.txt", old="two", new="three")
    diff = agent.flush_patch()
    assert "\\ No newline at end of file" in diff
    assert "---" not in "".join(
        line for line in diff.splitlines() if line.startswith(("-", "+")) and "a/" not in line and "b/" not in line and line[1:2] == "-")
    check = _git_apply_check(project, diff)
    assert check.returncode == 0, check.stderr


def test_the_overlay_is_safe_to_read_while_writes_land(project):
    import threading
    agent = _agent(project)
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                agent.current_patch()
                agent.has_patch()
            except Exception as e:
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(200):
            agent._patch_overlay.record(f"f{i}.txt", "x\n")
    finally:
        stop.set()
        t.join()
    assert errors == []


def test_grep_and_search_replace_see_a_file_created_in_the_session(project):
    agent = _agent(project)
    _run(agent, "write_file", path="newmod.py", content="def mul(a, b):\n    return a * b\n")
    assert "newmod.py" in _run(agent, "grep", pattern="def mul").output
    assert _run(agent, "search_replace", pattern=r"\bmul\b", replacement="multiply",
                glob="**/*.py").ok
    diff = agent.flush_patch()
    assert "+def multiply(a, b):" in diff
    assert "--- /dev/null" in diff or "--- a/newmod.py" in diff
