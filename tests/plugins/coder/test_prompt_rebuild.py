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
