# SPDX-License-Identifier: AGPL-3.0-or-later
"""``--verbose`` on a one-shot TASK run prints each tool's result and output
on stderr, under the tool call line; without it only the call line is printed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import ToolResult


def _make_agent(tmp_path: Path, *, verbose: bool):
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
        agent = Agent(backend=backend, cwd=tmp_path, verbose=verbose)
    agent._audit = MagicMock()
    agent.report_progress = True
    return agent


def _run_read(agent, output="line one\nline two"):
    tool_def = MagicMock()
    tool_def.destructive = False
    tool_def.fn = MagicMock(return_value=ToolResult.success(output, summary="read a.py"))
    call = ToolCall(name="read_file", args={"path": "a.py"}, raw="", start=0, end=0)
    with patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY", {"read_file": tool_def}):
        return agent._execute_tool(call, interactive=False)


@pytest.fixture
def wide_consoles(monkeypatch):
    from localm.plugins.coder import display
    from localm.console import SafeConsole
    monkeypatch.setattr(display, "err_console",
                        SafeConsole(stderr=True, highlight=False, width=200,
                                    no_color=True))
    monkeypatch.setattr(display, "console",
                        SafeConsole(highlight=False, width=200, no_color=True))


def test_verbose_one_shot_prints_the_tool_output_on_stderr(tmp_path, capsys, wide_consoles):
    _run_read(_make_agent(tmp_path, verbose=True))
    out, err = capsys.readouterr()
    assert "read_file" in err
    assert "read a.py" in err
    assert "line one" in err and "line two" in err
    assert "line one" not in out


def test_a_one_shot_without_verbose_prints_only_the_call(tmp_path, capsys, wide_consoles):
    _run_read(_make_agent(tmp_path, verbose=False))
    out, err = capsys.readouterr()
    assert "read_file" in err
    assert "line one" not in err and "line one" not in out


def test_a_gui_session_prints_nothing_even_when_verbose(tmp_path, capsys, wide_consoles):
    agent = _make_agent(tmp_path, verbose=True)
    agent.on_event = lambda ev: None
    _run_read(agent)
    out, err = capsys.readouterr()
    assert "line one" not in err and "line one" not in out
