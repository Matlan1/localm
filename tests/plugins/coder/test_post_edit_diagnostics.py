# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-edit diagnostics: the ruff pass appended to write/edit/patch results.

Runs the real ruff found next to the test interpreter. What the tests pin is the
harness around it: when a process may be spawned at all (never for a restricted
session, never with ruff absent or the project opted out), where ruff is allowed to
come from, and that the result stays honest on a timeout or a ruff failure.
Spawns are counted on the real ``subprocess.run``, so "no process" is asserted, not
inferred from a missing block."""

import os
import stat
import sys
from pathlib import Path

import pytest

from localm.plugins.coder.project_config import load_project_config
from localm.plugins.coder.tools import (
    diagnostics, tool_edit_file, tool_edit_files, tool_patch_file, tool_write_file,
)

_EXE = "ruff.exe" if sys.platform == "win32" else "ruff"
_RUFF = Path(sys.executable).parent / _EXE

needs_ruff = pytest.mark.skipif(not _RUFF.is_file(),
                                reason="ruff is not installed beside the interpreter")

UNUSED_IMPORT = "import os\nx = 1\n"


@pytest.fixture
def ruff_on_path(monkeypatch):
    monkeypatch.setenv("PATH", str(_RUFF.parent) + os.pathsep + os.environ.get("PATH", ""))


@pytest.fixture
def no_ruff_on_path(monkeypatch):
    monkeypatch.setenv("PATH", os.fspath(Path(__file__).parent))


@pytest.fixture
def spawns(monkeypatch):
    calls: list = []
    real = diagnostics.subprocess.run

    def counting(argv, *a, **k):
        calls.append(list(argv))
        return real(argv, *a, **k)

    monkeypatch.setattr(diagnostics.subprocess, "run", counting)
    return calls


def _write(tmp_path, name="m.py", content=UNUSED_IMPORT, restricted=False):
    return tool_write_file(tmp_path, name, content, _restricted=restricted)


@needs_ruff
@pytest.mark.usefixtures("ruff_on_path")
class TestWriteFile:
    def test_unused_import_is_reported(self, tmp_path, spawns):
        r = _write(tmp_path)
        assert r.ok is True
        assert "[ruff F,E9] 1 issue(s):" in r.output
        assert "F401" in r.output and "m.py:1:" in r.output
        assert "ruff" in r.summary
        assert len(spawns) == 1

    def test_clean_file_adds_nothing(self, tmp_path, spawns):
        r = _write(tmp_path, content="x = 1\n")
        assert "[ruff" not in r.output
        assert "ruff" not in r.summary
        assert len(spawns) == 1

    def test_output_is_capped_with_a_remainder_count(self, tmp_path):
        body = "".join(f"import mod{i}\n" for i in range(15))
        r = _write(tmp_path, content=body)
        assert "15 issue(s)" in r.output
        assert r.output.count("F401") == diagnostics.MAX_DIAGNOSTICS
        assert "... and 5 more" in r.output

    def test_syntax_error_is_left_to_the_syntax_check(self, tmp_path, spawns):
        r = _write(tmp_path, content="def broken(:\n")
        assert "[syntax check]" in r.output
        assert "[ruff" not in r.output
        assert spawns == []

    def test_non_python_file_spawns_nothing(self, tmp_path, spawns):
        r = _write(tmp_path, name="notes.txt", content="import os\n")
        assert "[ruff" not in r.output
        assert spawns == []

    def test_no_cache_directory_is_left_in_the_project(self, tmp_path):
        _write(tmp_path)
        assert not (tmp_path / ".ruff_cache").exists()

    def test_oversized_file_is_skipped(self, tmp_path, spawns, monkeypatch):
        monkeypatch.setattr(diagnostics, "MAX_FILE_BYTES", 10)
        r = _write(tmp_path, content=UNUSED_IMPORT * 5)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_project_ruff_excludes_are_honoured(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.ruff]\nextend-exclude = ["gen.py"]\n', encoding="utf-8")
        r = _write(tmp_path, name="gen.py")
        assert "[ruff" not in r.output


@needs_ruff
@pytest.mark.usefixtures("ruff_on_path")
class TestNoProcessWhenNotAllowed:
    def test_restricted_session_never_spawns(self, tmp_path, spawns):
        r = _write(tmp_path, restricted=True)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_a_caller_that_says_nothing_is_treated_as_restricted(self, tmp_path, spawns):
        r = tool_write_file(tmp_path, "m.py", UNUSED_IMPORT)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_opt_out_in_project_config(self, tmp_path, spawns):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            "diagnostics = false\n", encoding="utf-8")
        r = _write(tmp_path)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_explicit_opt_in_still_runs(self, tmp_path, spawns):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            "diagnostics = true\n", encoding="utf-8")
        assert "F401" in _write(tmp_path).output
        assert len(spawns) == 1

    def test_unreadable_config_cannot_be_overridden_so_nothing_runs(self, tmp_path, spawns):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            "diagnostics = [unterminated\n", encoding="utf-8")
        r = _write(tmp_path)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_the_config_key_survives_loading(self, tmp_path):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text(
            "diagnostics = false\n", encoding="utf-8")
        assert load_project_config(tmp_path)["diagnostics"] is False


class TestRuffAbsent:
    def test_absent_ruff_changes_nothing(self, tmp_path, no_ruff_on_path, spawns):
        r = _write(tmp_path)
        assert r.ok is True
        assert "ruff" not in r.output
        assert "ruff" not in r.summary
        assert spawns == []


class TestResolution:
    def _exe(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        exe = directory / _EXE
        exe.write_bytes(b"")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        return exe

    def test_absolute_path_entry_resolves(self, tmp_path):
        exe = self._exe(tmp_path / "bin")
        assert diagnostics._resolve_ruff(str(tmp_path / "bin"), ".EXE") == str(exe)

    def test_relative_and_empty_entries_are_ignored(self, tmp_path, monkeypatch):
        self._exe(tmp_path)
        monkeypatch.chdir(tmp_path)
        joined = os.pathsep.join(["", ".", "bin", os.curdir])
        assert diagnostics._resolve_ruff(joined, ".EXE") is None

    def test_a_ruff_in_the_working_directory_is_never_found(self, tmp_path, monkeypatch):
        self._exe(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "")
        assert diagnostics.find_ruff() is None

    @pytest.mark.skipif(sys.platform != "win32", reason="PATHEXT is a Windows rule")
    def test_only_real_executables_count_on_windows(self, tmp_path):
        (tmp_path / "ruff.bat").write_bytes(b"")
        (tmp_path / "ruff.cmd").write_bytes(b"")
        assert diagnostics._resolve_ruff(str(tmp_path), ".COM;.EXE;.BAT;.CMD") is None


@needs_ruff
@pytest.mark.usefixtures("ruff_on_path")
class TestHonestFailures:
    def test_timeout_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diagnostics, "TIMEOUT_SECONDS", 0.001)
        r = _write(tmp_path)
        assert r.ok is True
        assert "diagnostics skipped: timed out" in r.output

    def test_a_ruff_failure_says_so(self, tmp_path):
        (tmp_path / "ruff.toml").write_text("this is [not valid\n", encoding="utf-8")
        r = _write(tmp_path)
        assert r.ok is True
        assert "diagnostics skipped: ruff failed" in r.output
        assert "F401" not in r.output

    def test_a_ruff_that_cannot_start_says_so(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise PermissionError("denied")

        monkeypatch.setattr(diagnostics.subprocess, "run", boom)
        r = _write(tmp_path)
        assert r.ok is True
        assert "could not start ruff" in r.output


@needs_ruff
@pytest.mark.usefixtures("ruff_on_path")
class TestOtherWriteTools:
    def test_edit_file(self, tmp_path, spawns):
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "m.py", "x = 1", "import os\nx = 1",
                           _restricted=False)
        assert "F401" in r.output and len(spawns) == 1

    def test_edit_file_restricted(self, tmp_path, spawns):
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "m.py", "x = 1", "import os\nx = 1",
                           _restricted=True)
        assert "[ruff" not in r.output and spawns == []

    def test_edit_file_syntax_error_is_left_to_the_syntax_check(self, tmp_path, spawns):
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "m.py", "x = 1", "def f(:", _restricted=False)
        assert "[syntax check]" in r.output and "[ruff" not in r.output
        assert spawns == []

    def test_patch_file(self, tmp_path, spawns):
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        diff = "--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n+import os\n x = 1\n"
        r = tool_patch_file(tmp_path, "m.py", diff, _restricted=False)
        assert r.ok is True
        assert "F401" in r.output and len(spawns) == 1

    def test_patch_file_syntax_error_is_left_to_the_syntax_check(self, tmp_path, spawns):
        (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        diff = "--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n+def f(:\n x = 1\n"
        r = tool_patch_file(tmp_path, "m.py", diff, _restricted=False)
        assert "[syntax check]" in r.output and "[ruff" not in r.output
        assert spawns == []

    def test_edit_files_runs_one_process_for_the_batch(self, tmp_path, spawns):
        for name in ("a.py", "b.py", "c.txt"):
            (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
        edits = [{"path": n, "old": "x = 1", "new": "import os\nx = 1"}
                 for n in ("a.py", "b.py", "c.txt")]
        r = tool_edit_files(tmp_path, edits, _restricted=False)
        assert r.ok is True
        assert "2 issue(s)" in r.output
        assert "a.py:1:" in r.output and "b.py:1:" in r.output
        assert "c.txt" not in r.output.split("[ruff", 1)[1]
        assert len(spawns) == 1

    def test_edit_files_skips_a_file_with_a_syntax_error(self, tmp_path, spawns):
        (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "bad.py").write_text("x = 1\n", encoding="utf-8")
        edits = [{"path": "ok.py", "old": "x = 1", "new": "import os\nx = 1"},
                 {"path": "bad.py", "old": "x = 1", "new": "def f(:"}]
        r = tool_edit_files(tmp_path, edits, _restricted=False)
        assert "[syntax check]" in r.output
        assert "ok.py:1:" in r.output.split("[ruff", 1)[1]
        assert "bad.py" not in r.output.split("[ruff", 1)[1]


def _agent(cwd, **kw):
    from unittest.mock import MagicMock, patch

    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=cwd, auto_approve=True, self_verify=False, **kw)


def _tool_call(name, **args):
    from unittest.mock import MagicMock
    call = MagicMock()
    call.name = name
    call.args = args
    return call


@needs_ruff
@pytest.mark.usefixtures("ruff_on_path")
class TestThroughTheAgent:
    """The real dispatch path: the session, not the tool call, decides."""

    def test_an_owner_session_gets_diagnostics(self, tmp_path, spawns):
        agent = _agent(tmp_path)
        r = agent._execute_tool(
            _tool_call("write_file", path="m.py", content=UNUSED_IMPORT), interactive=False)
        assert "F401" in r.output
        assert len(spawns) == 1

    def test_a_restricted_session_spawns_nothing(self, tmp_path, spawns):
        agent = _agent(tmp_path, restricted=True)
        r = agent._execute_tool(
            _tool_call("write_file", path="m.py", content=UNUSED_IMPORT), interactive=False)
        assert r.ok is True
        assert (tmp_path / "m.py").read_text(encoding="utf-8") == UNUSED_IMPORT
        assert "[ruff" not in r.output
        assert spawns == []

    def test_a_model_supplied_flag_cannot_lift_the_restriction(self, tmp_path, spawns):
        agent = _agent(tmp_path, restricted=True)
        r = agent._execute_tool(
            _tool_call("write_file", path="m.py", content=UNUSED_IMPORT, _restricted=False),
            interactive=False)
        assert "[ruff" not in r.output
        assert spawns == []

    def test_every_diagnosed_tool_is_injected(self):
        import inspect

        from localm.plugins.coder.tools import TOOL_REGISTRY
        for name in diagnostics.DIAGNOSED_TOOLS:
            assert "_restricted" in inspect.signature(TOOL_REGISTRY[name].fn).parameters
