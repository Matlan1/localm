# SPDX-License-Identifier: AGPL-3.0-or-later
"""edit_files is wired into the agent like every other write tool.

Every path-consuming site in the agent (scope check, undo snapshot, changed-file
tracker, project-map refresh, patch-mode intercept) reads a single top-level
`path` arg. edit_files has none - its paths live inside `edits=[{path, ...}]` -
so each site resolves them through the shared _call_target_paths() helper.

The scope case is checked directly, on disk: a tool in _SCOPED_TOOLS whose paths
the checker cannot see would pass the check by having nothing to check, which is
a silent fail-OPEN rather than the set's default-deny.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.agent.constants import (
    _MUTATING_TOOLS, _UNDOABLE_TOOLS, _call_target_paths,
)


def _make_agent(tmp_path: Path, scope: str | None = None,
                patch_mode: bool = False, **kw) -> Agent:
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, scope=scope,
                      auto_approve=True, **kw)
    # patch_mode is not a constructor arg - the CLI sets the attribute directly
    # (cli/_main.py). Passing it to Agent() is swallowed by **gen_kwargs.
    agent.patch_mode = patch_mode
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
        # rejects the malformed call. Resolution does not raise.
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
        """The in-scope file is not edited either: the call never runs."""
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
        # Compare against the bytes actually on disk: write_text translates the
        # newline to the platform line ending.
        before = (project / "src" / "main.py").read_bytes()
        agent = _make_agent(project)
        agent._execute_tool(_call("edit_files", edits=[_swap("src/main.py")]),
                            interactive=False)
        entry = [e for e in agent._undo_stack if e["tool"] == "edit_files"][0]
        assert entry["old_content"] == before

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

    def test_undo_reverts_the_WHOLE_batch_not_just_one_file(self, project):
        """A multi-file call pushes one entry per file but is ONE operation, so
        a single /undo restores every file in the batch."""
        agent = _make_agent(project)
        before = {n: (project / n).read_bytes()
                  for n in ("src/main.py", "secrets.txt")}
        agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)
        assert (project / "src" / "main.py").read_bytes() != before["src/main.py"]

        msg = agent.undo()
        for name, original in before.items():
            assert (project / name).read_bytes() == original, \
                f"{name} was not restored by a single /undo"
        assert "2 of 2" in msg

    def test_undo_of_a_single_file_tool_is_unchanged(self, project):
        """A one-entry-per-call tool undoes exactly its own entry."""
        agent = _make_agent(project)
        before = (project / "src" / "main.py").read_bytes()
        agent._execute_tool(
            _call("edit_file", path="src/main.py", old="import old",
                  new="import new"), interactive=False)
        msg = agent.undo()
        assert (project / "src" / "main.py").read_bytes() == before
        assert msg.startswith("Undid edit_file: restored")

    def test_two_batches_undo_one_batch_at_a_time(self, project):
        agent = _make_agent(project)
        agent._execute_tool(_call("edit_files", edits=[_swap("src/main.py")]),
                            interactive=False)
        agent._execute_tool(
            _call("edit_files",
                  edits=[{"path": "secrets.txt", "old": "import old",
                          "new": "import other"}]), interactive=False)
        agent.undo()          # undoes only the SECOND batch
        assert (project / "secrets.txt").read_text(encoding="utf-8") == "import old\n"
        assert (project / "src" / "main.py").read_text(encoding="utf-8") == "import new\n"
        agent.undo()          # now the first
        assert (project / "src" / "main.py").read_text(encoding="utf-8") == "import old\n"

    def test_a_failed_undo_is_reported_not_claimed_complete(self, project,
                                                            monkeypatch):
        agent = _make_agent(project)
        agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"), _swap("secrets.txt")]),
            interactive=False)

        real = Path.write_bytes

        def failing(self, data):
            if self.name == "secrets.txt":
                raise OSError("read-only filesystem")
            return real(self, data)

        monkeypatch.setattr(Path, "write_bytes", failing)
        msg = agent.undo()
        assert "did NOT fully succeed" in msg
        assert "secrets.txt" in msg


class TestPatchMode:
    def test_patch_mode_writes_nothing_to_disk(self, project):
        """Under patch_mode the interceptor captures the batch as a diff and
        nothing reaches disk."""
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
        assert "[patch-mode]" in result.output
        assert "src/main.py" in result.output.replace("\\", "/")

    def test_multifile_diff_composes_repeated_edits_to_one_file(self, project):
        from localm.plugins.coder.diffutil import compute_multifile_diff
        diff = compute_multifile_diff(project, [
            {"path": "src/main.py", "old": "import old", "new": "import mid"},
            {"path": "src/main.py", "old": "import mid", "new": "import new"},
        ])
        assert "+import new" in diff
        assert "import mid" not in diff      # the intermediate state never appears
        assert diff.count("--- a/src/main.py") == 1   # one hunk header per file

    def test_diff_composes_across_equivalent_spellings_of_one_path(self, project):
        """"a.py" and "./a.py" are the same file to the tool (it resolves), so
        the preview composes them into one hunk."""
        from localm.plugins.coder.diffutil import compute_multifile_diff
        diff = compute_multifile_diff(project, [
            {"path": "src/main.py", "old": "import old", "new": "import mid"},
            {"path": "./src/main.py", "old": "import mid", "new": "import new"},
        ])
        assert diff.count("+import new") == 1
        assert diff.count("--- a/") == 1

    def test_a_batch_the_tool_would_reject_produces_no_diff(self, project):
        """A batch edit_files would refuse produces no diff at all, rather than
        a partial one."""
        from localm.plugins.coder.diffutil import compute_multifile_diff
        assert compute_multifile_diff(project, [
            {"path": "src/main.py", "old": "import old", "new": "import new"},
            {"path": "secrets.txt", "old": "NOT IN THE FILE", "new": "x"},
        ]) is None

    def test_patch_mode_rejects_a_batch_with_a_miss_and_writes_nothing(self, project):
        agent = _make_agent(project, patch_mode=True)
        before = (project / "src" / "main.py").read_bytes()
        result = agent._execute_tool(
            _call("edit_files", edits=[_swap("src/main.py"),
                                       {"path": "secrets.txt",
                                        "old": "NOT IN THE FILE", "new": "x"}]),
            interactive=False)
        assert result.ok is False
        assert (project / "src" / "main.py").read_bytes() == before
        assert agent.flush_patch() == ""
