# SPDX-License-Identifier: AGPL-3.0-or-later
"""A terminal session that confirms tools is never asked to approve a
``spawn_agent_background`` call that would then refuse: the refusal comes first,
before any hook or confirmation. A session that can start one still asks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools.agents import background_spawn_refusal
from localm.plugins.coder.tools.registry import TOOL_REGISTRY


def _make_agent(tmp_path: Path, *, auto_approve: bool):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    backend.supports_grammar = False
    backend.last_usage = {}
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, auto_approve=auto_approve)
    agent._audit = MagicMock()
    return agent


def _call():
    return ToolCall(name="spawn_agent_background", args={"task": "look around"},
                    raw="", start=0, end=0)


def test_the_registry_entry_carries_the_precheck():
    assert TOOL_REGISTRY["spawn_agent_background"].precheck is background_spawn_refusal


def test_a_terminal_session_is_refused_without_being_asked(tmp_path):
    agent = _make_agent(tmp_path, auto_approve=False)
    asked = []
    agent._confirm_tool = lambda call: asked.append(call.name) or True
    result = agent._execute_tool(_call(), interactive=True)
    assert asked == []
    assert not result.ok
    assert "cannot prompt there" in result.output


def test_a_session_with_an_approval_channel_still_asks(tmp_path):
    agent = _make_agent(tmp_path, auto_approve=False)
    asked = []
    agent.confirm_handler = lambda *a, **k: asked.append("asked") or False
    agent._execute_tool(_call(), interactive=False)
    assert asked == ["asked"]
