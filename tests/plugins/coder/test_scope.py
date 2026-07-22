# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the scope-filtering logic in Agent._execute_tool.

When an agent has a scope glob set, file-access tools that target a path
outside the glob pattern must be rejected without reaching the tool function.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_agent_with(tmp_path: Path, **kwargs) -> Agent:
    """Return an Agent with a mock backend and any extra constructor kwargs."""
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, **kwargs)
    return agent


def _make_agent(tmp_path: Path, scope: str | None = None) -> Agent:
    """Return an Agent with a mock backend and optional scope."""
    return _make_agent_with(tmp_path, scope=scope)


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

    def test_plugin_tool_path_confined_by_scope(self, tmp_path):
        """CHK-SCOPE-PLUGIN: a plugin_* tool's path arg is confined by the active
        scope, same as an mcp_* tool. Plugin tools are dynamically registered (not
        in _SCOPED_TOOLS), so, like MCP tools, they are gated by their name prefix -
        previously they were the one dynamic file-tool family the scope check missed."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("plugin_fs_read_file", path="/etc/passwd")   # outside scope
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"plugin_fs_read_file": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_plugin_tool_uncommon_path_arg_confined(self, tmp_path):
        """A path under an uncommon plugin arg name (source_path) is still scoped."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("plugin_disk_copy", source_path="/etc/shadow")
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"plugin_disk_copy": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_plugin_tool_in_scope_path_allowed(self, tmp_path):
        """A plugin_* tool whose path IS inside the scope is not scope-rejected."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")
        call = _make_tool_call("plugin_fs_read_file", path="src/main.py")
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"plugin_fs_read_file": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" not in result.output

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

# ---------------------------------------------------------------------------
#  A scope that does not confine the shell must SAY so (work item A3)
# ---------------------------------------------------------------------------

class TestScopeShellNotice:
    """run_shell / run_tests are deliberately left out of _SCOPED_TOOLS: they
    execute a process, and no path-arg check can confine arbitrary code. That
    trade-off is correct and stays. What was wrong is that it was documented ONLY
    in a source comment, so a user who set --scope got no runtime signal at all
    and could reasonably believe the session was confined. These tests pin the
    signal, not a sandbox."""

    def _warnings(self, tmp_path, scope, **kwargs):
        with patch("localm.plugins.coder.agent.print_warning") as warn:
            agent = _make_agent_with(tmp_path, scope=scope, **kwargs)
        return agent, [str(c.args[0]) for c in warn.call_args_list]

    def test_notice_fires_once_for_a_scoped_session_with_shell(self, tmp_path):
        _, warnings = self._warnings(tmp_path, "src/**")
        hits = [w for w in warnings if "confines the file tools only" in w]
        assert len(hits) == 1, f"expected exactly one notice, got {warnings}"
        assert "run_shell" in hits[0] and "run_tests" in hits[0]

    def test_no_notice_without_a_scope(self, tmp_path):
        """Control: an unscoped session has no false belief to correct."""
        _, warnings = self._warnings(tmp_path, None)
        assert not [w for w in warnings if "confines the file tools only" in w]

    def test_no_notice_when_the_shell_tools_are_disabled(self, tmp_path):
        """Nothing to warn about: with the shell gone the scope really is the
        boundary."""
        _, warnings = self._warnings(
            tmp_path, "src/**",
            disabled_tools=frozenset({"run_shell", "run_tests"}))
        assert not [w for w in warnings if "confines the file tools only" in w]

    def test_a_sub_agent_does_not_repeat_the_notice(self, tmp_path):
        """A spawned child is a new Agent but not a new session: the user already
        saw the notice from the parent, and a delegation-heavy run would otherwise
        repeat it once per spawn until it reads as noise."""
        parent = _make_agent_with(tmp_path, scope="src/**")
        with patch("localm.plugins.coder.agent.print_warning") as warn:
            _make_agent_with(tmp_path, scope="src/**", parent=parent)
        warnings = [str(c.args[0]) for c in warn.call_args_list]
        assert not [w for w in warnings if "confines the file tools only" in w]

    def test_notice_reaches_a_gui_session_over_on_event(self, tmp_path):
        events: list = []
        with patch("localm.plugins.coder.agent.print_warning"):
            _make_agent_with(tmp_path, scope="src/**", on_event=events.append)
        texts = [str(e.get("text", "")) for e in events if e.get("type") == "info"]
        assert any("confines the file tools only" in t for t in texts), texts


class TestShellArgvScopeCheck:
    """Best-effort argv check: flag a shell command that references paths outside
    the scope. A WARNING, never a block - the command still runs."""

    def _run_shell(self, tmp_path, scope, command):
        agent = _make_agent_with(tmp_path, scope=scope)
        agent._audit = MagicMock()
        call = _make_tool_call("run_shell", command=command)
        with patch("localm.plugins.coder.agent.print_warning") as warn:
            result = agent._execute_tool(call, interactive=False)
        return result, [str(c.args[0]) for c in warn.call_args_list], agent._audit

    def test_out_of_scope_path_is_flagged(self, tmp_path):
        (tmp_path / "secrets.txt").write_text("token\n")
        _, warnings, audit = self._run_shell(
            tmp_path, "src/**", "cat secrets.txt")
        hits = [w for w in warnings if "outside the active scope" in w]
        assert hits, f"no warning for an out-of-scope path; got {warnings}"
        assert "secrets.txt" in hits[0]
        assert "not" in hits[0].lower() and "confined" in hits[0].lower()
        assert "scope_shell_path" in [c.args[0] for c in audit.notice.call_args_list]

    def test_the_command_still_runs(self, tmp_path):
        """Warn, do not block: escalating a legitimate command into a hard failure
        would break working setups for a heuristic's benefit."""
        (tmp_path / "secrets.txt").write_text("token\n")
        # This is the one test here that asserts the command really EXECUTED, so
        # it needs a command that exists on the platform. `cat` is not a cmd.exe
        # builtin and is not on a stock Windows PATH: it resolves only when
        # Git-for-Windows' usr/bin happens to be there, so this passed under Git
        # Bash and redded under PowerShell on the same machine. `type` is the
        # cmd equivalent and needs nothing installed.
        read_file = ("type secrets.txt" if sys.platform == "win32"
                     else "cat secrets.txt")
        result, warnings, _ = self._run_shell(tmp_path, "src/**", read_file)
        assert [w for w in warnings if "outside the active scope" in w]
        assert result.ok, result.output           # it executed
        assert "token" in result.output           # and really did read the file

    def test_absolute_path_outside_cwd_is_flagged(self, tmp_path):
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", "cat /etc/passwd")
        assert [w for w in warnings if "/etc/passwd" in w], warnings

    def test_parent_traversal_is_flagged_even_if_absent(self, tmp_path):
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", "cat ../../elsewhere.txt")
        assert [w for w in warnings if "outside the active scope" in w], warnings

    def test_in_scope_path_is_not_flagged(self, tmp_path):
        """Control: the check must be quiet when the command stays in scope, or it
        is noise the user learns to ignore."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", "cat src/main.py")
        assert not [w for w in warnings if "outside the active scope" in w], warnings

    def test_no_check_without_a_scope(self, tmp_path):
        (tmp_path / "secrets.txt").write_text("token\n")
        _, warnings, _ = self._run_shell(tmp_path, None, "cat secrets.txt")
        assert not [w for w in warnings if "outside the active scope" in w]

    def test_flags_and_non_path_tokens_are_not_mistaken_for_paths(self, tmp_path):
        """A regex, a flag, and a quoted message all contain slashes but none is a
        path; flagging them would drown the real signal."""
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**",
            "git commit --allow-empty -m 'fix a/b handling' && sed s/foo/bar/")
        assert not [w for w in warnings if "outside the active scope" in w], warnings

    def test_run_tests_path_arg_is_checked_too(self, tmp_path):
        """run_shell is not the only unscoped executor; run_tests takes a target
        path that can point anywhere too."""
        from localm.plugins.coder import agent as _agent
        from localm.plugins.coder.tools import ToolResult
        (tmp_path / "tests").mkdir()
        agent = _make_agent_with(tmp_path, scope="src/**")
        agent._audit = MagicMock()
        call = _make_tool_call("run_tests", path="tests")
        stub = MagicMock(destructive=False,
                         fn=lambda cwd, **kw: ToolResult.success("0 passed"))
        with patch("localm.plugins.coder.agent.print_warning") as warn, \
             patch.dict(_agent.TOOL_REGISTRY, {"run_tests": stub}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        warnings = [str(c.args[0]) for c in warn.call_args_list]
        assert [w for w in warnings if "outside the active scope" in w], warnings
        assert result.ok                # still ran; the check only warns


class TestShellArgvScopeHeuristicPrecision:
    """A warning nobody believes is worse than no warning. Two token shapes were
    reported as out-of-scope paths when they are not paths at all: anything with
    a colon in position 1 (``5:30``, ``s:old:new:``) and a command verb that
    happens to match a directory name (``npm test`` in a repo with ``test/``).
    Both must go quiet without the real findings going quiet with them."""

    @pytest.fixture
    def repo(self, tmp_path):
        """An utterly ordinary layout: these directory names are exactly the ones
        common command verbs collide with."""
        for name in ("src", "test", "tests", "docs", "build"):
            (tmp_path / name).mkdir()
        (tmp_path / "secrets.txt").write_text("token\n")
        return tmp_path

    def _flagged(self, cwd, tool="run_shell", **args):
        with patch("localm.plugins.coder.agent.print_warning"):
            agent = _make_agent_with(cwd, scope="src/**")
        return agent._shell_paths_outside_scope(_make_tool_call(tool, **args))

    @pytest.mark.parametrize("command", [
        "ffmpeg -ss 5:30 -i in.mp4 out.mp4",       # a timestamp offset
        "ffmpeg -aspect 4:3 -i in.mp4 out.mp4",    # an aspect ratio
        "sed s:old:new: notes.txt",                # a sed delimiter
        "prog a:b",                                # a generic key:value argument
    ])
    def test_a_colon_token_is_not_read_as_a_drive_path(self, repo, command):
        assert self._flagged(repo, command=command) == []

    def test_a_real_drive_path_is_still_flagged(self, repo):
        """Fires-control for the case above: the same colon check, still loud on a
        genuinely drive-qualified path."""
        assert self._flagged(repo, command=r"cat C:\Windows\win.ini") == [
            r"C:\Windows\win.ini"]
        assert self._flagged(repo, command="cat D:/other/file.txt") == [
            "D:/other/file.txt"]
        assert self._flagged(repo, command="cat E:") == ["E:"]

    @pytest.mark.parametrize("command", [
        "npm test",                  # test/ exists
        "make docs",                 # docs/ exists
        "cargo build",               # build/ exists
        "npm run build",             # a dispatch subcommand, then a script name
        "uv run pytest",
        "npm --silent test",         # a flag does not take the subcommand slot
        "echo start && make docs",   # && starts a new command, so make is a program
    ])
    def test_a_command_verb_is_not_read_as_a_path(self, repo, command):
        assert self._flagged(repo, command=command) == []

    def test_the_same_word_is_still_flagged_in_argument_position(self, repo):
        """Fires-control for the case above: ``docs`` is quiet as a make target and
        loud as a real argument, so the check became precise rather than mute."""
        assert self._flagged(repo, command="make docs") == []
        assert self._flagged(repo, command="cp -r docs backup") == ["docs"]
        assert self._flagged(repo, command="git add docs") == ["docs"]

    def test_an_explicit_path_in_command_position_is_still_flagged(self, repo):
        """Only the exists-under-cwd guess is skipped for a command word. Running a
        script written out as an out-of-scope path is what the warning is for."""
        assert self._flagged(repo, command="./build/run.sh --fast") == [
            "./build/run.sh"]
        assert self._flagged(repo, command="/usr/local/bin/deploy") == [
            "/usr/local/bin/deploy"]
        assert self._flagged(repo, command="make ../other/target") == [
            "../other/target"]

    def test_a_real_out_of_scope_reference_still_warns(self, repo):
        """The #781 behaviour restated against this layout: an existing relative
        path and an absolute path, both outside the scope, both still reported."""
        assert self._flagged(repo, command="cat secrets.txt") == ["secrets.txt"]
        assert self._flagged(repo, command="cat /etc/passwd") == ["/etc/passwd"]
        assert self._flagged(repo, command="cat ../../elsewhere.txt") == [
            "../../elsewhere.txt"]

    def test_run_tests_args_are_not_treated_as_a_command_line(self, repo):
        """run_tests passes a target PATH plus extra args, not a command line, so
        the program-word suppression must not reach either of them."""
        assert self._flagged(repo, tool="run_tests", path="tests") == ["tests"]
        assert self._flagged(repo, tool="run_tests",
                             path="tests", extra_args="docs") == ["tests", "docs"]


class TestScopedToolsSet:
    @pytest.mark.parametrize(
        "tool,expected",
        [
            ("read_file", True),
            ("write_file", True),
            ("edit_file", True),
            ("patch_file", True),
            ("run_shell", False),
            ("fetch_url", False),
        ],
    )
    def test_tool_in_scoped_tools(self, tool, expected):
        assert (tool in _SCOPED_TOOLS) == expected
