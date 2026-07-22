# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the scope-filtering logic in Agent._execute_tool.

When an agent has a scope glob set, file-access tools that target a path
outside the glob pattern must be rejected without reaching the tool function.
"""

import contextlib
import os
import os.path
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.agent.execution import (
    _is_path_like, _looks_like_drive_path,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _outside_scope(tmp_path: Path, name: str = "x.txt") -> str:
    """An absolute path OUTSIDE the ``src/**`` scope, owned and created by the
    test in its own tmp_path.

    Every "out of scope" target in this file is a real but disposable file the
    test made. A system path (a real ``/etc/...`` or drive-anchored OS file) must
    never be used as one: the code under test path-processes what it is handed,
    so such a target makes the suite itself reach out and touch a real OS file,
    and at the access point a legitimate test, a command gone wrong, and a live
    injection are indistinguishable.
    """
    outside = tmp_path / "outside" / name
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("disposable\n", encoding="utf-8")
    return str(outside)


@contextlib.contextmanager
def _no_filesystem():
    """Turn any filesystem access inside the block into an immediate failure.

    Asserting that a check is lexical means asserting the ABSENCE of a
    capability, so the guard has to see the syscalls. Audit hooks cannot:
    ``Path.exists()``, ``Path.resolve()`` and ``os.stat()`` emit no audit event
    (measured on py3.12) - only ``open`` does. Patching these does see them,
    because ``Path.exists`` routes to ``os.stat`` and ``Path.resolve`` to
    ``os.path.realpath``.
    """
    targets = [(os, "stat"), (os, "lstat"), (os, "open"), (os, "scandir"),
               (os, "listdir"), (os.path, "realpath")]

    def _forbid(label):
        def _boom(path, *a, **kw):
            raise AssertionError(
                f"filesystem access inside a purely lexical check: "
                f"{label}({path!r})")
        return _boom

    saved = [(mod, name, getattr(mod, name)) for mod, name in targets]
    for mod, name in targets:
        setattr(mod, name, _forbid(f"{mod.__name__}.{name}"))
    try:
        yield
    finally:
        for mod, name, original in saved:
            setattr(mod, name, original)


@contextlib.contextmanager
def _records_touches_outside(root: Path):
    """Record filesystem calls made against paths NOT under *root*.

    The recording twin of :func:`_no_filesystem`, and the enforcement path needs
    it rather than the raising one. ``_scope_rel`` legitimately resolves its cwd
    ANCHOR, so a blanket "any syscall fails" guard would fire on the anchor and
    could never tell that apart from the thing actually under test: a stat of the
    model-supplied VALUE. Scoping the record to paths outside cwd separates them,
    and counting rather than raising means the assertion can be an exact number
    (0) instead of "did not blow up".
    """
    touches: list = []
    root_s = str(root).replace("\\", "/").lower()
    targets = [(os, "stat"), (os, "lstat"), (os, "open"), (os, "scandir"),
               (os, "listdir"), (os.path, "realpath")]

    def _record(label, original):
        def _spy(path, *a, **kw):
            try:
                s = str(path).replace("\\", "/").lower()
            except Exception:
                s = ""
            if s and not s.startswith(root_s):
                touches.append(f"{label}({path!r})")
            return original(path, *a, **kw)
        return _spy

    saved = [(mod, name, getattr(mod, name)) for mod, name in targets]
    for mod, name in targets:
        setattr(mod, name, _record(f"{mod.__name__}.{name}", getattr(mod, name)))
    try:
        yield touches
    finally:
        for mod, name, original in saved:
            setattr(mod, name, original)


def _link_dir_or_skip(link: Path, target: Path) -> None:
    """Create *link* as a directory link to *target*, or skip the test.

    Tries a real symlink first, then a Windows junction (``mklink /J``), which
    needs no elevation and so works on an ordinary contributor box where
    ``os.symlink`` raises. Only if BOTH are unavailable is the platform genuinely
    unable to express the case, which is a documented platform block, not a pass.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        import subprocess
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True).returncode == 0:
            return
    pytest.skip("this platform/account cannot create a directory link "
                "(no symlink privilege and no junction support)")


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
        call = _make_tool_call("mcp_fs_read_file", path=_outside_scope(tmp_path))
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"mcp_fs_read_file": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_mcp_tool_uncommon_path_arg_confined(self, tmp_path):
        """A path under an uncommon MCP arg name (source_path) is still scoped."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("mcp_fs_copy", source_path=_outside_scope(tmp_path))
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
        call = _make_tool_call("plugin_fs_read_file", path=_outside_scope(tmp_path))
        with patch.dict(_agent.TOOL_REGISTRY,
                        {"plugin_fs_read_file": MagicMock(destructive=False)}, clear=False):
            result = agent._execute_tool(call, interactive=False)
        assert "outside the active scope" in result.output

    def test_plugin_tool_uncommon_path_arg_confined(self, tmp_path):
        """A path under an uncommon plugin arg name (source_path) is still scoped."""
        from localm.plugins.coder import agent as _agent
        agent = _make_agent(tmp_path, scope="src/**")
        call = _make_tool_call("plugin_disk_copy", source_path=_outside_scope(tmp_path))
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

class TestScopeEnforcementNeverStatsTheValue:
    """The hard gate must decide WITHOUT touching the path it is deciding about.

    #802 made the shell WARNING path filesystem-free and deliberately left the
    ENFORCEMENT path alone; it carried the same defect. ``_scope_rel`` fell back
    to ``Path(raw).resolve()`` for an absolute path that was not lexically under
    cwd, so a ``--scope`` session where the model emitted an absolute path
    anywhere on the machine stat-ed exactly that path in order to REFUSE it
    (measured: one ``os.path.realpath`` plus one ``os.stat``).

    A stat is an access. At the access point a legitimate gate-check, a command
    gone wrong and a live injection attempt are indistinguishable, so the gate
    must not have the capability rather than try to use it carefully.

    These assert an exact ZERO rather than "fewer": the whole property is the
    ABSENCE of a capability, and any nonzero count is the capability still being
    there. Every target is a real but disposable file the test created.
    """

    @pytest.mark.parametrize("tool", ["read_file", "mcp_fs_read_file"])
    def test_refusing_an_absolute_path_outside_cwd_touches_nothing(
            self, tmp_path, tmp_path_factory, tool):
        outside = tmp_path_factory.mktemp("outside-cwd")
        target = outside / "disposable-target.txt"
        target.write_text("disposable\n", encoding="utf-8")
        agent = _make_agent(tmp_path, scope="src/**")

        with _records_touches_outside(tmp_path) as touches:
            offending = agent._scope_violation(
                _make_tool_call(tool, path=str(target)))

        assert offending == str(target), "the call must still be refused"
        assert touches == [], (
            "the scope gate reached out and touched the path it was refusing: "
            + "; ".join(touches))

    def test_an_in_cwd_absolute_path_in_scope_is_still_allowed(self, tmp_path):
        """BUG-6, unchanged: an absolute path INSIDE cwd that matches the scope
        must pass. It is lexically under cwd, so it never needed the fallback."""
        (tmp_path / "src").mkdir()
        target = tmp_path / "src" / "a.py"
        target.write_text("# in scope\n", encoding="utf-8")
        agent = _make_agent(tmp_path, scope="src/**")
        assert agent._scope_violation(
            _make_tool_call("read_file", path=str(target))) is None

    def test_a_relative_in_scope_path_is_still_allowed(self, tmp_path):
        agent = _make_agent(tmp_path, scope="src/**")
        assert agent._scope_violation(
            _make_tool_call("read_file", path="src/a.py")) is None

    def test_a_path_reaching_into_cwd_through_a_link_is_refused(
            self, tmp_path, tmp_path_factory):
        """The accepted COST of the fix, pinned so it cannot be reintroduced by
        accident or silently regress back to statting.

        An absolute path lexically OUTSIDE cwd that reaches INSIDE through a
        directory link used to be ALLOWED, because the resolve() fallback looked
        through the link. Measured both ways before this test was written: with
        the fallback the verdict is ALLOWED and two out-of-cwd touches are made;
        without it the verdict is REFUSED and zero. That delta is the whole point
        - the fallback had no other regression coverage, so a green suite alone
        would prove nothing about removing it.

        Refusing is the fail-CLOSED direction, which is the only direction a
        confinement gate may move. The real escape direction (a path lexically
        INSIDE cwd that links OUT) never used this fallback at all: it satisfies
        the first ``relative_to``. Escapes are caught by ``tools/base.py::
        _confine`` at execution time, which is untouched by this.
        """
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("# in scope\n", encoding="utf-8")
        outside = tmp_path_factory.mktemp("outside-cwd")
        _link_dir_or_skip(outside / "link", tmp_path / "src")
        via_link = outside / "link" / "a.py"
        assert via_link.exists(), "the link must really reach the in-cwd file"

        agent = _make_agent(tmp_path, scope="src/**")
        with _records_touches_outside(tmp_path) as touches:
            offending = agent._scope_violation(
                _make_tool_call("read_file", path=str(via_link)))

        assert offending == str(via_link), (
            "a path that is lexically outside cwd must be refused even when it "
            "would resolve back inside - looking through the link is exactly the "
            "stat this gate may not make")
        assert touches == [], "; ".join(touches)


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
        secrets = _outside_scope(tmp_path, "secrets.txt")
        _, warnings, audit = self._run_shell(
            tmp_path, "src/**", f'cat "{secrets}"')
        hits = [w for w in warnings if "outside the active scope" in w]
        assert hits, f"no warning for an out-of-scope path; got {warnings}"
        assert "secrets.txt" in hits[0]
        assert "not" in hits[0].lower() and "confined" in hits[0].lower()
        assert "scope_shell_path" in [c.args[0] for c in audit.notice.call_args_list]

    def test_the_command_still_runs(self, tmp_path):
        """Warn, do not block: escalating a legitimate command into a hard failure
        would break working setups for a heuristic's benefit."""
        Path(_outside_scope(tmp_path, "secrets.txt")).write_text(
            "token\n", encoding="utf-8")
        # This is the one test here that asserts the command really EXECUTED, so
        # it needs a command that exists on the platform. `cat` is not a cmd.exe
        # builtin and is not on a stock Windows PATH: it resolves only when
        # Git-for-Windows' usr/bin happens to be there, so this passed under Git
        # Bash and redded under PowerShell on the same machine. `type` is the
        # cmd equivalent and needs nothing installed.
        #
        # Written explicitly-relative and UNQUOTED. Explicitly-relative so the
        # lexical check sees a path at all; unquoted because _shell_argv splits
        # with posix=False on Windows, which KEEPS the quotes in the token, and a
        # quoted path then goes through `cmd /C` with quotes cmd strips wrongly.
        # The other tests here never execute, so only this one is exposed to it.
        read_file = (r"type .\outside\secrets.txt" if sys.platform == "win32"
                     else "cat ./outside/secrets.txt")
        result, warnings, _ = self._run_shell(tmp_path, "src/**", read_file)
        assert [w for w in warnings if "outside the active scope" in w]
        assert result.ok, result.output           # it executed
        assert "token" in result.output           # and really did read the file

    def test_absolute_path_outside_cwd_is_flagged(self, tmp_path, tmp_path_factory):
        # A disposable file the test owns, in a temp dir that is genuinely outside
        # the agent's cwd - never a real OS path.
        outside = tmp_path_factory.mktemp("outside_cwd") / "elsewhere.txt"
        outside.write_text("disposable\n", encoding="utf-8")
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", f'cat "{outside}"')
        assert [w for w in warnings if "elsewhere.txt" in w], warnings

    def test_parent_traversal_is_flagged_even_if_absent(self, tmp_path):
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", "cat ../../elsewhere.txt")
        assert [w for w in warnings if "outside the active scope" in w], warnings

    def test_in_scope_path_is_not_flagged(self, tmp_path):
        """Control: the check must be quiet when the command stays in scope, or it
        is noise the user learns to ignore."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")
        # Written explicitly-relative so it IS path-like under the lexical rule:
        # a bare `src/main.py` would now be skipped as not-a-path, which would
        # make this control pass without testing anything.
        _, warnings, _ = self._run_shell(
            tmp_path, "src/**", "cat ./src/main.py")
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
        for name in ("src", "test", "tests", "docs", "build", "scripts", "tools"):
            (tmp_path / name).mkdir()
        (tmp_path / "secrets.txt").write_text("token\n")
        (tmp_path / "build" / "run.sh").write_text("#!/bin/sh\n")
        (tmp_path / "tools" / "gen.py").write_text("pass\n")
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

    def test_a_real_drive_path_is_still_flagged(self, repo, tmp_path_factory):
        """Fires-control for the case above: the same colon check, still loud on a
        genuinely drive-qualified path. The target is a disposable file the test
        owns (drive-qualified on Windows, absolute on POSIX), never a real OS
        file: the check must be provable without the suite itself reaching out."""
        outside = tmp_path_factory.mktemp("drive_qualified") / "file.txt"
        outside.write_text("disposable\n", encoding="utf-8")
        assert self._flagged(repo, command=f'cat "{outside}"') == [str(outside)]
        # A bare drive letter is drive-qualified too, and names nothing.
        assert self._flagged(repo, command="cat E:") == ["E:"]

    @pytest.mark.parametrize("command", [
        "npm test",                  # test/ exists
        "make docs",                 # docs/ exists
        "cargo build",               # build/ exists
        "npm run build",             # a dispatch subcommand, then a script name
        "uv run build",              # the same, for a non-npm runner
        "npm --silent test",         # a flag does not take the subcommand slot
        "echo start && make docs",   # && starts a new command, so make is a program
        "CI=1 npm test",             # an env assignment is not the program word
        "NODE_ENV=test make docs",
        "sudo npm test",             # nor is a transparent wrapper
        "time make docs",
        "env CI=1 npm test",
    ])
    def test_a_command_verb_is_not_read_as_a_path(self, repo, command):
        assert self._flagged(repo, command=command) == []

    def test_a_bare_relative_path_is_no_longer_flagged(self, repo):
        """The documented cost of dropping the filesystem probe, made executable.

        `build/run.sh` and `tools/gen.py` exist here and are outside the scope,
        and they are no longer reported. Telling them apart from `npm test` in a
        repo that has a `test/` folder is exactly what the exists-under-cwd stat
        did, and that same stat read whatever a drive-anchored token named,
        anywhere on the machine. A separator rule is no substitute: it re-flags
        `sed s/foo/bar/` and a quoted `-m 'fix a/b handling'`. Write the path
        explicitly and it is loud again, and reaching OUT of the workspace - the
        case this warning exists for - is unaffected."""
        assert self._flagged(repo, command="build/run.sh --fast") == []
        assert self._flagged(repo, command="uv run tools/gen.py") == []
        assert self._flagged(repo, command="./build/run.sh --fast") == [
            "./build/run.sh"]

    def test_a_path_valued_flag_is_caught_when_written_as_a_path(self, repo):
        """`git -C dir` moves the process's working directory, the strongest
        out-of-scope signal there is. A bare `docs` cannot be told from a
        subcommand without asking the filesystem, so it goes unreported like any
        other bare word; written as a path - which is the form that can actually
        leave the workspace - it is still caught, in the flag's value position
        like anywhere else."""
        assert self._flagged(repo, command="git -C docs status") == []
        assert self._flagged(repo, command="git -C ./docs status") == ["./docs"]
        assert self._flagged(repo, command="make -C ../other all") == ["../other"]

    def test_a_colon_delimiter_holding_slashes_is_not_a_drive_path(self, repo):
        """A colon is chosen as the sed delimiter precisely BECAUSE the pattern
        contains slashes, so this is the common form. A real drive-qualified path
        carries exactly one colon."""
        assert self._flagged(
            repo, command="sed s:/usr/local:/opt:g notes.txt") == []
        assert self._flagged(
            repo, command="sed -i s:/old/path:/new/path: notes.txt") == []

    def test_position_no_longer_changes_the_answer(self, repo):
        """The old check answered differently depending on where a word sat in the
        command line (program slot, subcommand slot, argument), which is what all
        the runner/wrapper/flag tables were for. Classification is by syntax now:
        `docs` is quiet everywhere and `./docs` is loud everywhere. One rule, with
        no positional exceptions left to get wrong."""
        for command in ("make docs", "cp -r docs backup", "git add docs"):
            assert self._flagged(repo, command=command) == [], command
        for command in ("cp -r ./docs backup", "git add ./docs", "./docs/gen.sh"):
            assert self._flagged(repo, command=command) != [], command

    def test_an_explicit_path_in_command_position_is_still_flagged(
            self, repo, tmp_path_factory):
        """Running a script written out as an out-of-scope path is what the
        warning is for, wherever it sits in the command line. The absolute target
        is a disposable file the test owns, never a real OS binary."""
        deploy = tmp_path_factory.mktemp("deploy_bin") / "deploy.sh"
        deploy.write_text("#!/bin/sh\n", encoding="utf-8")
        assert self._flagged(repo, command="./build/run.sh --fast") == [
            "./build/run.sh"]
        assert self._flagged(repo, command=f'"{deploy}"') == [str(deploy)]
        assert self._flagged(repo, command="make ../other/target") == [
            "../other/target"]

    def test_a_real_out_of_scope_reference_still_warns(self, repo, tmp_path_factory):
        """The #781 behaviour restated for the lexical rule: an explicitly written
        relative path, a parent traversal, and an absolute path outside cwd are all
        still reported. The absolute target is a disposable file the test owns."""
        secrets = tmp_path_factory.mktemp("absolute_target") / "secrets.txt"
        secrets.write_text("disposable\n", encoding="utf-8")
        assert self._flagged(repo, command="cat ./secrets.txt") == ["./secrets.txt"]
        assert self._flagged(repo, command=f'cat "{secrets}"') == [str(secrets)]
        assert self._flagged(repo, command="cat ../../elsewhere.txt") == [
            "../../elsewhere.txt"]

    def test_run_tests_path_is_a_declared_path_arg(self, repo):
        """run_tests' `path` is declared a path by the tool's own schema, so it
        needs no path-likeness guess at all: it is checked WHOLE (spaces and all)
        and a bare `tests` is still reported there, unlike the same word inside a
        command line. `extra_args` is free-form (a -k expression, a marker name),
        so it follows the command-line rule."""
        assert self._flagged(repo, tool="run_tests", path="tests") == ["tests"]
        assert self._flagged(repo, tool="run_tests", path="my tests") == ["my tests"]
        assert self._flagged(repo, tool="run_tests",
                             path="tests", extra_args="docs") == ["tests"]
        assert self._flagged(repo, tool="run_tests", path="tests",
                             extra_args="./docs") == ["tests", "./docs"]


class TestPathLikeSyntax:
    """`_is_path_like` and `_looks_like_drive_path` are pure STRING classifiers,
    and they are the whole reason the scope check no longer needs the filesystem,
    so they are tested as strings. The drive letters below are tokens: never used
    as paths, never joined, never resolved."""

    @pytest.mark.parametrize("tok", [
        "./a", "../a", "~/a", "/a/b", "a/../b",          # explicit relative / absolute
        "Q:", "Q:/a", r"Q:\a", "q:/a",                   # drive-qualified
    ])
    def test_path_like_by_syntax(self, tok):
        assert _is_path_like(tok)

    @pytest.mark.parametrize("tok", [
        "test", "docs",              # bare words (the accepted trade-off)
        "build/run.sh",              # a bare relative path, likewise
        "a:b", "5:30", "4:3",        # colon-bearing non-paths
        "s:old:new:", "s/foo/bar/",  # sed forms
        "Q:a",                       # drive-RELATIVE, deliberately given up
    ])
    def test_not_path_like_by_syntax(self, tok):
        assert not _is_path_like(tok)

    def test_drive_qualification_needs_a_letter_and_one_colon(self):
        assert _looks_like_drive_path("Q:/a") and _looks_like_drive_path("Q:")
        assert not _looks_like_drive_path("5:30")
        assert not _looks_like_drive_path("s:/usr/local:/opt:g")


class TestShellScopeCheckIsPurelyLexical:
    """The scope warning evaluates a command the model has only PROPOSED: before
    any confirmation, before anything executes. Doing that must not touch the
    filesystem. `(self.cwd / tok).exists()` discards cwd entirely for a
    drive-anchored token, so merely READING a proposed command stat-ed whatever
    the model named, anywhere on the machine - and at the access point a
    legitimate probe, a command gone wrong, and a live injection attempt are
    indistinguishable, so the capability has to be absent rather than careful.

    Absence of a capability is the property under test, so these assert it
    directly instead of inferring it from a correct answer. Every target is a
    disposable file the test created in its own temp dir."""

    @pytest.fixture
    def agent(self, tmp_path):
        (tmp_path / "src").mkdir()
        with patch("localm.plugins.coder.agent.print_warning"):
            return _make_agent_with(tmp_path, scope="src/**")

    def test_the_guard_itself_fires(self, tmp_path):
        """Fires-control for `_no_filesystem`. Without it, every test below could
        be green because the guard never armed."""
        with pytest.raises(AssertionError, match="filesystem access"):
            with _no_filesystem():
                (tmp_path / "anything.txt").exists()
        with pytest.raises(AssertionError, match="filesystem access"):
            with _no_filesystem():
                (tmp_path / "anything.txt").resolve()
        # ...and it puts the real functions back.
        assert (tmp_path).exists()

    def test_classifying_a_command_touches_no_files(self, agent, tmp_path_factory):
        outside = tmp_path_factory.mktemp("lexical_target") / "secrets.txt"
        outside.write_text("disposable\n", encoding="utf-8")
        call = _make_tool_call(
            "run_shell", command=f'cat "{outside}" ./notes.txt ../up.txt')
        with _no_filesystem():
            flagged = agent._shell_paths_outside_scope(call)
        assert flagged == [str(outside), "./notes.txt", "../up.txt"]

    def test_an_in_scope_command_touches_no_files(self, agent):
        with _no_filesystem():
            assert agent._shell_paths_outside_scope(
                _make_tool_call("run_shell", command="cat ./src/main.py")) == []

    def test_a_declared_path_arg_touches_no_files(self, agent):
        with _no_filesystem():
            assert agent._shell_paths_outside_scope(
                _make_tool_call("run_tests", path="tests")) == ["tests"]

    def test_the_whole_warning_path_touches_no_files(self, agent, tmp_path_factory):
        """Not only the classifier: formatting the warning and recording the audit
        notice run on the same proposed command, so they must stay clean too."""
        outside = tmp_path_factory.mktemp("warn_target") / "s.txt"
        outside.write_text("disposable\n", encoding="utf-8")
        agent._audit = MagicMock()
        call = _make_tool_call("run_shell", command=f'cat "{outside}"')
        with patch("localm.plugins.coder.agent.print_warning") as warn, \
             _no_filesystem():
            agent._warn_shell_outside_scope(call)
        assert [w for w in (str(c.args[0]) for c in warn.call_args_list)
                if "outside the active scope" in w]
        assert "scope_shell_path" in [
            c.args[0] for c in agent._audit.notice.call_args_list]


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
