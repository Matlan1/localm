# SPDX-License-Identifier: AGPL-3.0-or-later
"""edit_files must be wired into the agent like every other write tool.

A multi-file tool is the easy one to half-integrate, because every path-consuming
site in the agent (scope check, undo snapshot, changed-file tracker, project-map
refresh, patch-mode intercept) was written against a single top-level `path` arg.
edit_files has none - its paths live inside `edits=[{path, ...}]` - so each site
resolves them through the shared _call_target_paths() helper.

The scope case is the one that MUST NOT regress: a tool in _SCOPED_TOOLS whose
paths the checker cannot see passes the check by having nothing to check, which
is a silent fail-OPEN. That is the opposite of the default-deny the set exists
to provide, so it is tested directly, on disk.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.agent.constants import (
    _MUTATING_TOOLS, _UNDOABLE_TOOLS, _call_target_paths,
)


def _make_agent(tmp_path: Path, scope: str | None = None, **kw) -> Agent:
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, scope=scope,
                      auto_approve=True, **kw)
    return agent


def _call(name: str, **args):
    call = MagicMock()
    call.name = name
    call.args = args
    return call


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("import old\n", encoding="utf-8")
    (tmp_path / "secrets.txt").write_text("import old\n", encoding="utf-8")
    return tmp_path


def _swap(path):
    return {"path": path, "old": "import old", "new": "import new"}


class TestTargetPathResolution:
    def test_nested_paths_are_resolved(self):
        assert _call_target_paths(
            "edit_files", {"edits": [{"path": "a.py"}, {"path": "b.py"}]}
        ) == ["a.py", "b.py"]

    def test_single_path_tools_unchanged(self):
        assert _call_target_paths("edit_file", {"path": "a.py"}) == ["a.py"]
        assert _call_target_paths("write_file", {}) == []

    @pytest.mark.parametrize("edits", ["garbage", None, [{}], ["x"], [{"path": ""}]])
    def test_malformed_edits_resolve_to_no_paths(self, edits):
        # No paths means the scope check has nothing to allow; the tool itself
        # rejects the malformed call. This must not raise.
        assert _call_target_paths("edit_files", {"edits": edits}) == []


class TestScopeConfinement:
    def test_out_of_scope_edit_is_rejected_and_nothing_is_written(self, project):
        agent = _make_agent(project, scope="src/**")
        before = (project / "secrets.txt").read_bytes()
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("secrets.txt")]), interactive=False)
        assert result.ok is False
        assert "outside the active scope" in result.output
        assert (project / "secrets.txt").read_bytes() == before

    def test_one_out_of_scope_path_rejects_the_whole_batch(self, project):
        """The in-scope file must not be edited either - the call never runs."""
        agent = _make_agent(project, scope="src/**")
        before_in = (project / "src" / "main.py").read_bytes()
        before_out = (project / "secrets.txt").read_bytes()
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        assert result.ok is False
        assert "outside the active scope" in result.output
        assert (project / "src" / "main.py").read_bytes() == before_in
        assert (project / "secrets.txt").read_bytes() == before_out

    def test_in_scope_batch_is_allowed(self, project):
        agent = _make_agent(project, scope="src/**")
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py")]), interactive=False)
        assert result.ok is True
        assert (project / "src" / "main.py").read_text(encoding="utf-8") == "import new\n"

    def test_edit_files_is_in_the_scoped_set(self):
        assert "edit_files" in _SCOPED_TOOLS


class TestUndoAndTracking:
    def test_one_undo_entry_per_file(self, project):
        agent = _make_agent(project)
        agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        undone = [e["path"].name for e in agent._undo_stack if e["tool"] == "edit_files"]
        assert sorted(undone) == ["main.py", "secrets.txt"]

    def test_undo_snapshots_hold_the_pre_edit_bytes(self, project):
        agent = _make_agent(project)
        agent._execute_tool(_call("edit_files", edits=[_swap("src/main.py")]),
                            interactive=False)
        entry = [e for e in agent._undo_stack if e["tool"] == "edit_files"][0]
        assert entry["old_content"] == b"import old\n"

    def test_every_edited_file_is_tracked_as_changed(self, project):
        agent = _make_agent(project)
        agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        # main.py is a code file, so it also lands in the unverified-writes set.
        assert any("main.py" in p for p in agent._unverified_writes)

    def test_registered_as_undoable_and_mutating(self):
        assert "edit_files" in _UNDOABLE_TOOLS
        assert "edit_files" in _MUTATING_TOOLS


class TestPatchMode:
    def test_patch_mode_writes_nothing_to_disk(self, project):
        """patch_mode promises no changes. A write tool the interceptor cannot
        express as a diff would fall through to a REAL write."""
        agent = _make_agent(project, patch_mode=True)
        before = (project / "src" / "main.py").read_bytes()
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        assert result.ok is True
        assert "[patch-mode]" in result.output
        assert (project / "src" / "main.py").read_bytes() == before
        assert (project / "secrets.txt").read_bytes() == before

    def test_captured_diff_covers_every_file(self, project):
        agent = _make_agent(project, patch_mode=True)
        agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        diff = agent.flush_patch()
        assert "main.py" in diff and "secrets.txt" in diff
        assert diff.count("-import old") == 2
        assert diff.count("+import new") == 2

    def test_patch_mode_names_the_files_it_captured(self, project):
        agent = _make_agent(project, patch_mode=True)
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py")]), interactive=False)
        assert "src/main.py" in result.output

    def test_multifile_diff_composes_repeated_edits_to_one_file(self, project):
        from localm.plugins.coder.diffutil import compute_multifile_diff
        diff = compute_multifile_diff(project, [
            {"path": "src/main.py", "old": "import old", "new": "import mid"},
            {"path": "src/main.py", "old": "import mid", "new": "import new"},
        ])
        assert "+import new" in diff
        assert "import mid" not in diff      # the intermediate state never appears
        assert diff.count("--- a/src/main.py") == 1   # one hunk header per file
