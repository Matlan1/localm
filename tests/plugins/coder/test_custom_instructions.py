# SPDX-License-Identifier: AGPL-3.0-or-later
"""Custom coder system-instructions: a `.localcoder/system.md` file and the
`--system` CLI flag inject user-authored guidance under "## User Instructions"
in the agent system prompt, distinct from LOCALCODER.md project memory.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


# --------------------------------------------------------------------------- #
#  Loader + prompt assembly                                                    #
# --------------------------------------------------------------------------- #

def test_load_custom_instructions_reads_system_md(tmp_path):
    from localm.plugins.coder.memory import load_custom_instructions

    assert load_custom_instructions(tmp_path) == ""  # absent -> empty
    d = tmp_path / ".localcoder"
    d.mkdir()
    (d / "system.md").write_text("  Always write guard clauses.\n", encoding="utf-8")
    assert load_custom_instructions(tmp_path) == "Always write guard clauses."


def test_build_system_prompt_injects_user_instructions(tmp_path):
    from localm.plugins.coder.prompts import build_system_prompt

    out = build_system_prompt(tmp_path, custom_instructions="Use British spelling.")
    assert "## User Instructions" in out
    assert "Use British spelling." in out
    # No custom instructions -> no section, no empty heading.
    assert "## User Instructions" not in build_system_prompt(tmp_path)


# --------------------------------------------------------------------------- #
#  Agent wiring                                                                #
# --------------------------------------------------------------------------- #

def _make_agent(tmp_path: Path, **kwargs):
    """Build an Agent with the heavy bits stubbed but the real memory /
    custom-instructions loaders running against tmp_path."""
    from localm.plugins.coder.agent import Agent

    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def test_agent_reads_system_md_into_prompt(tmp_path):
    d = tmp_path / ".localcoder"
    d.mkdir()
    (d / "system.md").write_text("PREFER_TABS_OVER_SPACES", encoding="utf-8")
    agent = _make_agent(tmp_path)
    assert "## User Instructions" in agent._system_prompt
    assert "PREFER_TABS_OVER_SPACES" in agent._system_prompt


def test_agent_system_flag_overrides_file(tmp_path):
    d = tmp_path / ".localcoder"
    d.mkdir()
    (d / "system.md").write_text("FROM_FILE", encoding="utf-8")
    agent = _make_agent(tmp_path, custom_instructions="FROM_FLAG")
    assert "FROM_FLAG" in agent._system_prompt
    assert "FROM_FILE" not in agent._system_prompt


def test_custom_instructions_are_distinct_from_project_memory(tmp_path):
    (tmp_path / "LOCALCODER.md").write_text(
        "- the build is slow", encoding="utf-8")
    d = tmp_path / ".localcoder"
    d.mkdir()
    (d / "system.md").write_text("Be terse.", encoding="utf-8")
    agent = _make_agent(tmp_path)
    p = agent._system_prompt
    assert "## Project Memory" in p and "the build is slow" in p
    assert "## User Instructions" in p and "Be terse." in p


def test_set_cwd_reloads_system_md_without_override(tmp_path):
    a = tmp_path / "a"
    (a / ".localcoder").mkdir(parents=True)
    (a / ".localcoder" / "system.md").write_text("A_RULES", encoding="utf-8")
    b = tmp_path / "b"
    (b / ".localcoder").mkdir(parents=True)
    (b / ".localcoder" / "system.md").write_text("B_RULES", encoding="utf-8")

    agent = _make_agent(a)
    assert "A_RULES" in agent._system_prompt
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM:
        MockPM.build.return_value.file_count.return_value = 0
        agent.set_cwd(b)
    assert "B_RULES" in agent._system_prompt
    assert "A_RULES" not in agent._system_prompt


def test_set_cwd_keeps_explicit_override(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    (b / ".localcoder").mkdir(parents=True)
    (b / ".localcoder" / "system.md").write_text("B_FILE", encoding="utf-8")

    agent = _make_agent(a, custom_instructions="STICKY_OVERRIDE")
    assert "STICKY_OVERRIDE" in agent._system_prompt
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM:
        MockPM.build.return_value.file_count.return_value = 0
        agent.set_cwd(b)
    # The flag override wins over the new cwd's file.
    assert "STICKY_OVERRIDE" in agent._system_prompt
    assert "B_FILE" not in agent._system_prompt


# --------------------------------------------------------------------------- #
#  CLI flag wiring                                                             #
# --------------------------------------------------------------------------- #

def test_cli_exposes_system_option_bound_to_custom_instructions():
    from localm.plugins.coder.cli._main import main

    opt = next((p for p in main.params if p.name == "system_instructions"), None)
    assert opt is not None, "--system option missing"
    assert "--system" in opt.opts
