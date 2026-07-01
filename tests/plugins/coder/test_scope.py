# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the scope-filtering logic in Agent._execute_tool.

When an agent has a scope glob set, file-access tools that target a path
outside the glob pattern must be rejected without reaching the tool function.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_agent(tmp_path: Path, scope: str | None = None) -> Agent:
    """Return an Agent with a mock backend and optional scope."""
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, scope=scope)
    return agent


def _make_tool_call(name: str, **args):
    call = MagicMock()
    call.name = name
    call.args = args
    return call


# ---------------------------------------------------------------------------
#  Scope enforcement
# ---------------------------------------------------------------------------

class TestScopeEnforcement:
    def test_no_scope_allows_any_path(self, tmp_path):
        agent = _make_agent(tmp_path, scope=None)
        (tmp_path / "src.py").write_text("x = 1\n")

        call = _make_tool_call("read_file", path="src.py")
        with patch("localm.plugins.coder.tools.tool_read_file",
                   return_value=MagicMock(ok=True, output="ok", summary="ok")):
            pass
        # No scope → execute_tool reaches the actual tool
        result = agent._execute_tool(call, interactive=False)
        # read_file will fail (file has no content mock) but shouldn't be scope-rejected
        assert "outside the active scope" not in result.output

    def test_mcp_tool_path_confined_by_scope(self, tmp_path):
        """CHK-MCP-SCOPE: a REGISTERED mcp_* tool's path arg is confined by the
        active scope, even though MCP tools are not in _SCOPED_TOOLS. (MCP tools are
        registered dynamically in production; here a stub reaches the scope gate.)"""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("mcp_fs_read_file", path="/etc/passwd")   # outside scope
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"mcp_fs_read_file": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_mcp_tool_uncommon_path_arg_confined(self, tmp_path):
        """A path under an uncommon MCP arg name (source_path) is still scoped."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("mcp_fs_copy", source_path="/etc/shadow")  # outside scope
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"mcp_fs_copy": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_scope_allows_matching_path(self, tmp_path):
        agent = _make_agent(tmp_path, scope="src/*.py")
        call = _make_tool_call("read_file", path="src/main.py")

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")

        result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" not in result.output

    def test_scope_rejects_non_matching_path(self, tmp_path):
        agent = _make_agent(tmp_path, scope="src/*.py")
        call = _make_tool_call("read_file", path="README.md")

        result = agent._execute_tool(call, interactive=False)
        assert not result.ok
        assert "outside the active scope" in result.output

    def test_scope_rejects_different_subdir(self, tmp_path):
        agent = _make_agent(tmp_path, scope="src/**/*.py")
        call = _make_tool_call("write_file", path="tests/test_x.py", content="pass\n")

        result = agent._execute_tool(call, interactive=False)
        assert not result.ok
        assert "outside the active scope" in result.output

    def test_scope_not_applied_to_non_scoped_tools(self, tmp_path):
        """run_shell and git tools should NOT be filtered by scope."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        call = _make_tool_call("git_status")

        result = agent._execute_tool(call, interactive=False)
        # Should execute git_status (may succeed or fail based on git in PATH),
        # but must NOT return a scope-rejection error
        assert "outside the active scope" not in result.output

    def test_scope_check_uses_forward_slashes(self, tmp_path):
        """Path normalisation: backslash paths must match forward-slash patterns."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        # Simulate Windows-style path arg
        call = _make_tool_call("read_file", path="src\\main.py")

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")

        result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" not in result.output


# ---------------------------------------------------------------------------
#  _SCOPED_TOOLS registry contents
# ---------------------------------------------------------------------------

class TestScopedToolsSet:
    def test_read_file_in_scoped_tools(self):
        assert "read_file" in _SCOPED_TOOLS

    def test_write_file_in_scoped_tools(self):
        assert "write_file" in _SCOPED_TOOLS

    def test_edit_file_in_scoped_tools(self):
        assert "edit_file" in _SCOPED_TOOLS

    def test_patch_file_in_scoped_tools(self):
        assert "patch_file" in _SCOPED_TOOLS

    def test_run_shell_not_in_scoped_tools(self):
        assert "run_shell" not in _SCOPED_TOOLS

    def test_fetch_url_not_in_scoped_tools(self):
        assert "fetch_url" not in _SCOPED_TOOLS
