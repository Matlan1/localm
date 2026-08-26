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


def _per_write_refresh(agent, tmp_path):
    (tmp_path / "x.py").write_text("x = 1", encoding="utf-8")
    agent._refresh_map_for_tool(_make_call("write_file", path="x.py"))


def _build_messages_when_dirty(agent, tmp_path):
    # This rebuild site is reached only through _build_messages's own dirty
    # check.
    agent._project_map.dirty = True
    agent._build_messages()


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(_noop, id="initial_rebuild"),
        pytest.param(_reindex, id="reindex"),
        pytest.param(_reload_memory, id="reload_memory"),
        pytest.param(_set_cwd, id="set_cwd"),
        pytest.param(_per_write_refresh, id="per_write_refresh"),
        pytest.param(_build_messages_when_dirty, id="build_messages_when_dirty"),
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
        """Marking dirty must NOT be immediately followed by a rebuild here,
        unlike every other tool this method handles - see this method's own
        docstring for why: an eager rebuild would scan on every run_shell call
        instead of once for however many happen before the map is next
        actually read (context._build_messages, once per turn)."""
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            call = _make_call("run_shell", command="echo hi")
            agent._refresh_map_for_tool(call)
            mock_rebuild.assert_not_called()


class TestBuildMessagesDeferredRescan:
    """_build_messages (context.py) is where a run_shell-dirtied map actually
    gets reconciled - the one place guaranteed to run before the model's next
    turn. These pin the gate itself, separately from the "docs preserved"
    property already covered by build_messages_when_dirty above."""

    def test_rebuilds_when_dirty(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._project_map.dirty = True
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            agent._build_messages()
            mock_rebuild.assert_called_once()

    def test_does_not_rebuild_when_clean(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._project_map.dirty = False
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            agent._build_messages()
            mock_rebuild.assert_not_called()

    def test_a_mocked_project_map_never_spuriously_rebuilds(self, tmp_path):
        """A MagicMock's un-configured `.dirty` attribute is itself a
        MagicMock, which is truthy - if the gate used plain truthiness instead
        of `is True`, every test in this file (and any other that mocks
        ProjectMap wholesale) would rebuild on every _build_messages call."""
        agent = _make_agent(tmp_path)
        assert not isinstance(agent._project_map.dirty, bool)   # sanity: it's a MagicMock
        with patch.object(agent, "_rebuild_system_prompt") as mock_rebuild:
            agent._build_messages()
            mock_rebuild.assert_not_called()


class TestRunShellEndToEnd:
    """A REAL Agent with a REAL ProjectMap (unlike _make_agent above, which
    mocks ProjectMap entirely) driven through the actual dispatch path -
    the integration-level counterpart to the unit-level tests above. Runs a
    real `echo` via the real run_shell tool; nothing here is mocked except
    the LLM backend, which is never called."""

    def _make_real_agent(self, tmp_path: Path):
        from localm.plugins.coder.agent import Agent
        backend = MagicMock()
        backend.model_id = "test-model"
        backend.native_tools = False
        return Agent(backend=backend, cwd=tmp_path, auto_approve=True)

    def test_file_created_by_run_shell_is_stale_mid_turn_fresh_next_turn(self, tmp_path):
        from localm.plugins.coder.parser import ToolCall

        (tmp_path / "a.py").write_text("def existing():\n    pass\n", encoding="utf-8")
        agent = self._make_real_agent(tmp_path)
        assert agent._project_map.file_count() == 1

        # What a real `run_shell(command="echo ... > b.py")` would leave behind.
        (tmp_path / "b.py").write_text("def brand_new():\n    pass\n", encoding="utf-8")
        call = ToolCall(name="run_shell", args={"command": "echo done"},
                         raw="", start=0, end=0)
        result = agent._execute_tool(call, interactive=False)
        assert result.ok

        # Same turn: still stale by design (mark_dirty defers the rescan).
        assert agent._project_map.dirty is True
        assert "brand_new" not in agent._system_prompt

        # The next turn's _build_messages call is where it catches up.
        agent._build_messages()
        assert agent._project_map.dirty is False
        assert agent._project_map.file_count() == 2
        assert "brand_new" in agent._system_prompt
