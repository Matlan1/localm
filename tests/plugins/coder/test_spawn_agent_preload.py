# SPDX-License-Identifier: AGPL-3.0-or-later
"""spawn_agent file preloading fails loudly instead of poisoning the child.

The preload loop checks tool_read_file's r.ok, so a missing file's ERROR TEXT
is never fed to the child agent as "file content". An unreadable preload fails
the call, naming every unreadable file, before any child is spawned."""

from unittest.mock import MagicMock, patch

from localm.plugins.coder.tools import tool_spawn_agent

_AGENT = "localm.plugins.coder.agent.Agent"


def _parent(tmp_path):
    parent = MagicMock()
    parent.backend.model_id = "test"
    parent.cwd = tmp_path
    return parent


class TestSpawnPreloadFailure:
    def test_missing_file_fails_before_spawn(self, tmp_path):
        with patch(_AGENT) as MockAgent:
            r = tool_spawn_agent(tmp_path, "do work", files=["missing.txt"],
                                 _parent_agent=_parent(tmp_path))
        assert r.ok is False
        assert "could not pre-load" in r.output
        assert "missing.txt" in r.output
        MockAgent.assert_not_called()          # no child was spawned

    def test_all_failures_listed(self, tmp_path):
        (tmp_path / "real.txt").write_text("content\n", encoding="utf-8")
        with patch(_AGENT) as MockAgent:
            r = tool_spawn_agent(
                tmp_path, "do work",
                files=["gone-a.txt", "real.txt", "gone-b.txt"],
                _parent_agent=_parent(tmp_path))
        assert r.ok is False
        assert "gone-a.txt" in r.output and "gone-b.txt" in r.output
        MockAgent.assert_not_called()

    def test_readable_files_still_preload(self, tmp_path):
        (tmp_path / "ctx.txt").write_text("the context\n", encoding="utf-8")
        with patch(_AGENT) as MockAgent:
            MockAgent.return_value.run_task.return_value = "done"
            MockAgent.return_value.turns = 1
            r = tool_spawn_agent(tmp_path, "do work", files=["ctx.txt"],
                                 _parent_agent=_parent(tmp_path))
        assert r.ok is True
        MockAgent.assert_called_once()
        task_sent = MockAgent.return_value.run_task.call_args[0][0]
        assert "the context" in task_sent
        assert "Task:\ndo work" in task_sent

    def test_no_files_unchanged(self, tmp_path):
        with patch(_AGENT) as MockAgent:
            MockAgent.return_value.run_task.return_value = "done"
            MockAgent.return_value.turns = 1
            r = tool_spawn_agent(tmp_path, "plain task",
                                 _parent_agent=_parent(tmp_path))
        assert r.ok is True
        assert MockAgent.return_value.run_task.call_args[0][0] == "plain task"
