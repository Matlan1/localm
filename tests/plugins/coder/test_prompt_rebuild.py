# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder system prompt is rebuilt at several points in a session (set_cwd,
reindex, reload_memory, and the per-write project-map refresh). A prior bug
rebuilt with only the MCP tool docs, so plugin tools and agent skills silently
vanished from the prompt mid-session and the model "forgot" they existed. These
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


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(_noop, id="initial_rebuild"),
        pytest.param(_reindex, id="reindex"),
        pytest.param(_reload_memory, id="reload_memory"),
        pytest.param(_set_cwd, id="set_cwd"),
        pytest.param(_per_write_refresh, id="per_write_refresh"),
    ],
)
def test_rebuild_keeps_plugin_and_skill_docs(tmp_path, action):
    # Each row pins a distinct rebuild call site as its own regression guard -
    # keep all 5 separate so a bug in any single path still fails its own case.
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
#  A different property from the parametrize above: THAT test only checks    #
#  that a refresh which DOES fire preserves the combined docs. These check   #
#  whether _refresh_map_for_tool actually calls refresh_file for a given     #
#  tool at all - patch_file and edit_notebook_cell were silently missing     #
#  despite having a real resolvable path (a pre-existing gap, unrelated to   #
#  search_replace - found auditing _MUTATING_TOOLS for the same root cause   #
#  search_replace's own gap had), and search_replace needed a second data    #
#  source entirely (ToolResult.changes) since it has no `path` arg at all.   #
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

    def test_run_shell_does_not_refresh_the_map_pinned_as_a_known_gap(self, tmp_path):
        """PINNED, not silently left unverified: run_shell has no
        `path`-shaped arg at all (only `command`), so despite its
        _MUTATING_TOOLS membership this is currently a no-op - see that
        constant's own comment and dev-notes/coder-changed-files-tracking-
        gaps-2026-07-29.md item 2 (deliberately deferred: needs the same
        git-diff detection already gated off the live path elsewhere for a
        blocking-subprocess-on-the-event-loop risk, not a quick fix here).
        If this ever starts passing, it should be because that item was
        deliberately closed, not because something else changed underneath
        it unnoticed."""
        agent = _make_agent(tmp_path)
        call = _make_call("run_shell", command="echo hi")
        agent._refresh_map_for_tool(call)
        agent._project_map.refresh_file.assert_not_called()
