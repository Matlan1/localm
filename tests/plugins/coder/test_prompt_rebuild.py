# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder system prompt is rebuilt at several points in a session (set_cwd,
reindex, reload_memory, and the per-write project-map refresh). A rebuild that
carries only the MCP tool docs makes plugin tools and agent skills silently
vanish from the prompt mid-session, so the model "forgets" they exist. These
tests pin that every rebuild path keeps the COMBINED mcp + plugin + skill docs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.tools import ToolResult


def _make_agent(tmp_path: Path, **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _make_call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


_MCP = "MCP_DOC_MARKER_xyz"
_PLUGIN = "PLUGIN_DOC_MARKER_xyz"
_SKILL = "SKILL_DOC_MARKER_xyz"


def _seed_docs(agent):
    agent._mcp_docs = _MCP
    agent._plugin_docs = _PLUGIN
    agent._skill_docs = _SKILL
    agent._rebuild_system_prompt()


def _assert_all_present(agent):
    for marker in (_MCP, _PLUGIN, _SKILL):
        assert marker in agent._system_prompt, f"{marker} missing after rebuild"


def _noop(agent, tmp_path):
    pass


def _reindex(agent, tmp_path):
    agent.reindex()


def _reload_memory(agent, tmp_path):
    agent.reload_memory()


def _set_cwd(agent, tmp_path):
    agent.set_cwd(tmp_path)


def _compact(agent, tmp_path):
    with patch.object(agent, "_compact_history", return_value=True):
        assert agent.compact() is True


def _compact_to(agent, tmp_path):
    with patch.object(agent, "_compact_history", return_value=True), \
         patch.object(agent, "_drop_old_tool_output", return_value=0), \
         patch.object(agent, "_prompt_tokens_estimate", return_value=10_000):
        assert agent._compact_to(100, 1000) is True


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(_noop, id="initial_rebuild"),
        pytest.param(_reindex, id="reindex"),
        pytest.param(_reload_memory, id="reload_memory"),
        pytest.param(_set_cwd, id="set_cwd"),
        pytest.param(_compact, id="compact"),
        pytest.param(_compact_to, id="compact_to"),
    ],
)
def test_rebuild_keeps_plugin_and_skill_docs(tmp_path, action):
    # Each row pins a distinct rebuild call site.
    agent = _make_agent(tmp_path)
    _seed_docs(agent)
    action(agent, tmp_path)
    _assert_all_present(agent)


def test_empty_docs_do_not_crash_rebuild(tmp_path):
    # The default (no mcp/plugin/skill tools) rebuilds fine - the join filters ""s.
    agent = _make_agent(tmp_path)
    agent._mcp_docs = agent._plugin_docs = agent._skill_docs = ""
    agent._rebuild_system_prompt()
    assert "AVAILABLE TOOLS" in agent._system_prompt


# --------------------------------------------------------------------------- #
#  Which tools actually trigger a refresh (_MUTATING_TOOLS coverage)          #
#                                                                              #
#  Whether _refresh_map_for_tool calls refresh_file for a given tool at all.   #
#  search_replace has no `path` arg, so it is driven from ToolResult.changes.  #
# --------------------------------------------------------------------------- #

class TestIncrementalMapRefreshCoverage:
    def test_patch_file_refreshes_the_map(self, tmp_path):
        agent = _make_agent(tmp_path)
        call = _make_call("patch_file", path="a.py", diff="unused")
        agent._refresh_map_for_tool(call)
        agent._project_map.refresh_file.assert_called_once_with(
            (tmp_path / "a.py").resolve())

    def test_edit_notebook_cell_refreshes_the_map(self, tmp_path):
        agent = _make_agent(tmp_path)
        call = _make_call("edit_notebook_cell", path="n.ipynb",
                          cell_index=0, source="x = 1")
        agent._refresh_map_for_tool(call)
        agent._project_map.refresh_file.assert_called_once_with(
            (tmp_path / "n.ipynb").resolve())

    def test_search_replace_refreshes_the_map_via_result_changes(self, tmp_path):
        """search_replace has no `path` arg at all - _call_target_paths()
        alone finds nothing for it. The optional *result* parameter, reading
        ToolResult.changes, is what makes this work."""
        agent = _make_agent(tmp_path)
        call = _make_call("search_replace", pattern="x", replacement="y",
                          glob="*.py")
        result = ToolResult.success(
            "ok", changes=[("a.py", b"x = 1\n", "y = 1\n")])
        agent._refresh_map_for_tool(call, result)
        agent._project_map.refresh_file.assert_called_once_with(
            (tmp_path / "a.py").resolve())

    def test_omitting_result_does_not_crash_an_existing_caller(self, tmp_path):
        """Existing callers (including this file's own _per_write_refresh
        above) call _refresh_map_for_tool with just *call* - the new
        parameter must not become mandatory."""
        agent = _make_agent(tmp_path)
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        agent._refresh_map_for_tool(_make_call("write_file", path="a.py"))
        agent._project_map.refresh_file.assert_called_once_with(
            (tmp_path / "a.py").resolve())

    def test_run_shell_marks_the_map_dirty_instead_of_refreshing(self, tmp_path):
        """run_shell has no `path`-shaped arg at all (only `command`), so a
        per-file refresh_file() call is not possible ahead of time - see
        _MUTATING_TOOLS's own comment. Marking the whole map dirty is what
        covers it; a real ProjectMap's resulting stat-diff rescan is exercised
        by the indexer tests."""
        agent = _make_agent(tmp_path)
        call = _make_call("run_shell", command="echo hi")
        agent._refresh_map_for_tool(call)
        agent._project_map.mark_dirty.assert_called_once()
        agent._project_map.refresh_file.assert_not_called()

    def test_run_shell_does_not_eagerly_rebuild_the_prompt(self, tmp_path):
        """Marking dirty is not followed by a rebuild: the map reconciles at
        its next read."""
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            call = _make_call("run_shell", command="echo hi")
            agent._refresh_map_for_tool(call)
            mock_rebuild.assert_not_called()


class TestPromptPrefixStaysStable:
    """Tool calls update the project map, never the system prompt: the
    prompt every turn starts with stays identical until a compaction, which
    rebuilds it with the current map."""

    def test_a_write_tool_does_not_rebuild_the_prompt(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            agent._refresh_map_for_tool(_make_call("write_file", path="a.py"))
            mock_rebuild.assert_not_called()
        agent._project_map.refresh_file.assert_called_once_with(
            (tmp_path / "a.py").resolve())

    def test_build_messages_does_not_rebuild_a_dirty_map(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._project_map.dirty = True
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            agent._build_messages()
            mock_rebuild.assert_not_called()

    def test_a_compaction_that_changes_nothing_does_not_rebuild(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_compact_history", return_value=False), \
             patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            assert agent.compact() is False
            mock_rebuild.assert_not_called()


class TestToolChangesEndToEnd:
    """A REAL Agent with a REAL ProjectMap driven through the actual tool
    dispatch path; only the LLM backend is a mock, never called."""

    def _make_real_agent(self, tmp_path: Path):
        from localm.plugins.coder.agent import Agent
        backend = MagicMock()
        backend.model_id = "test-model"
        backend.native_tools = False
        return Agent(backend=backend, cwd=tmp_path, auto_approve=True)

    def test_files_changed_by_tools_reach_the_prompt_at_compaction_not_before(self, tmp_path):
        from localm.plugins.coder.parser import ToolCall

        (tmp_path / "a.py").write_text("def existing():\n    pass\n", encoding="utf-8")
        agent = self._make_real_agent(tmp_path)
        assert agent._project_map.file_count() == 1
        before = agent._system_prompt
        assert "existing" in before

        # What a real `run_shell(command="echo ... > b.py")` would leave behind.
        (tmp_path / "b.py").write_text("def brand_new():\n    pass\n", encoding="utf-8")
        shell = ToolCall(name="run_shell", args={"command": "echo done"},
                         raw="", start=0, end=0)
        assert agent._execute_tool(shell, interactive=False).ok
        write = ToolCall(name="write_file",
                         args={"path": "c.py", "content": "def made_by_write():\n    pass\n"},
                         raw="", start=0, end=0)
        assert agent._execute_tool(write, interactive=False).ok
        assert (tmp_path / "c.py").is_file()

        messages = agent._build_messages()
        assert messages[0] == {"role": "system", "content": before}
        assert agent._system_prompt == before
        assert agent._project_map.file_count() == 3

        with patch.object(agent, "_compact_history", return_value=True):
            assert agent.compact() is True
        assert "brand_new" in agent._system_prompt
        assert "made_by_write" in agent._system_prompt

    def test_an_automatic_compaction_refreshes_the_map_in_the_prompt(self, tmp_path):
        (tmp_path / "a.py").write_text("def existing():\n    pass\n", encoding="utf-8")
        agent = self._make_real_agent(tmp_path)
        (tmp_path / "b.py").write_text("def brand_new():\n    pass\n", encoding="utf-8")
        agent._project_map.mark_dirty()
        agent._build_messages()
        assert "brand_new" not in agent._system_prompt

        with patch.object(agent, "_compact_history", return_value=True), \
             patch.object(agent, "_drop_old_tool_output", return_value=0), \
             patch.object(agent, "_prompt_tokens_estimate", return_value=10_000):
            assert agent._compact_to(100, 1000) is True
        assert "brand_new" in agent._system_prompt
