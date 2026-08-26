# SPDX-License-Identifier: AGPL-3.0-or-later
"""Role presets for spawned sub-agents: narrowed toolsets that only ever subtract.

Without roles, ``spawn_agent`` hands the child the parent's entire toolset, so a
child asked only to "review this diff" still holds write_file, run_shell and
git_push. ``build_subagent_system_prompt`` also prints the RAW cwd, which would
put the absolute machine path and OS username into the prompt.

These tests drive REAL parent and child Agent objects through the REAL dispatch
path (``_execute_tool``, exactly what ``run_task``/``_loop`` call), never a mock
of the thing under test: the security boundary is ``disabled_tools`` enforced at
agent/execution.py, so a test that only inspected constructor kwargs would prove
nothing about whether the tool can actually run.

The invariant under test throughout: a role only ever REMOVES capability. It can
never re-enable a tool the parent disabled, nor one a restricted (shareable,
non-owner) session forbids.
"""

import re
from unittest.mock import patch

import pytest

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.prompts import build_subagent_system_prompt
from localm.plugins.coder.roles import ROLE_PRESETS, resolve_role
from localm.plugins.coder.tools import TOOL_REGISTRY, tool_spawn_agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _spawn_child(tmp_path, role=None, parent_kwargs=None):
    """Spawn a real child through tool_spawn_agent, short-circuiting run_task so
    no LLM call is needed, and return (result, child) for direct inspection of
    the code path spawn_agent actually exercises."""
    parent = Agent(_StubBackend(), cwd=tmp_path, **(parent_kwargs or {}))
    captured = {}

    def _fake_run_task(self, task):
        captured["child"] = self
        return "child done"

    with patch.object(Agent, "run_task", _fake_run_task):
        result = tool_spawn_agent(tmp_path, "do work", role=role,
                                  _parent_agent=parent)
    return result, captured.get("child")


def _dispatch(agent, _tool_name, _args=None, **kwargs):
    """Run a tool through the child's REAL dispatch path.

    The tool args go in a dict (or kwargs when they cannot collide): read_env's
    own parameter is called ``name``, which shadows a ``name=`` helper argument.
    """
    args = dict(_args or {})
    args.update(kwargs)
    return agent._execute_tool(
        ToolCall(name=_tool_name, args=args, raw="", start=0, end=0),
        interactive=False)


# --------------------------------------------------------------------------- #
#  A role actually narrows the child, at the dispatch boundary
# --------------------------------------------------------------------------- #

class TestRoleNarrowsTheChild:
    @pytest.mark.parametrize("tool,args", [
        ("write_file", {"path": "evil.txt", "content": "x"}),
        ("edit_file",  {"path": "evil.txt", "old": "a", "new": "b"}),
        ("run_shell",  {"command": "echo pwned"}),
        ("git_push",   {}),
        ("git_commit", {"message": "sneaky"}),
        ("read_env",   {"name": "PATH"}),
        ("fetch_url",  {"url": "http://example.com"}),
    ])
    def test_reviewer_cannot_change_or_execute_anything(self, tmp_path, tool, args):
        _, child = _spawn_child(tmp_path, role="reviewer")
        result = _dispatch(child, tool, args)
        assert not result.ok, f"reviewer was able to run {tool}"
        assert "disabled for this session" in result.output
        # The file was never written.
        assert not (tmp_path / "evil.txt").exists()

    def test_reviewer_can_still_read_and_inspect(self, tmp_path):
        (tmp_path / "app.py").write_text("print('hi')\n")
        _, child = _spawn_child(tmp_path, role="reviewer")
        result = _dispatch(child, "read_file", path="app.py")
        assert result.ok, result.output
        assert "print" in result.output

    def test_researcher_has_no_write_and_no_git(self, tmp_path):
        _, child = _spawn_child(tmp_path, role="researcher")
        assert not _dispatch(child, "write_file", path="x.txt", content="x").ok
        # A researcher explores the code; git history is the reviewer's job.
        assert not _dispatch(child, "git_diff").ok
        assert _dispatch(child, "list_dir", path=".").ok

    def test_test_writer_may_write_and_run_tests_but_not_shell_or_push(self, tmp_path):
        """The role must be a real, differentiated grant - not a blanket deny."""
        _, child = _spawn_child(tmp_path, role="test-writer")
        assert _dispatch(child, "write_file", path="test_x.py", content="def test_a():\n    pass\n").ok
        assert (tmp_path / "test_x.py").exists()
        assert "run_tests" not in child.disabled_tools
        # ...but no arbitrary execution and no way to publish the result.
        assert not _dispatch(child, "run_shell", command="echo x").ok
        assert not _dispatch(child, "git_push").ok

    def test_every_role_forbids_recursive_spawning(self, tmp_path):
        """A child that can spawn its own children escapes the narrowing."""
        for role in ROLE_PRESETS:
            _, child = _spawn_child(tmp_path, role=role)
            assert "spawn_agent" in child.disabled_tools, role
            assert not _dispatch(child, "spawn_agent", task="escape").ok


# --------------------------------------------------------------------------- #
#  THE INVARIANT: strictly subtractive - a role can never hand capability BACK
# --------------------------------------------------------------------------- #

class TestRoleIsStrictlySubtractive:
    def test_role_cannot_reenable_a_tool_the_parent_disabled(self, tmp_path):
        """read_file is in every role's allowlist, so a naive implementation that
        ASSIGNED the role's set instead of UNIONing would resurrect it here."""
        (tmp_path / "secret.txt").write_text("classified\n")
        for role in ROLE_PRESETS:
            _, child = _spawn_child(
                tmp_path, role=role,
                parent_kwargs={"disabled_tools": frozenset({"read_file"})})
            assert "read_file" in child.disabled_tools, role
            result = _dispatch(child, "read_file", path="secret.txt")
            assert not result.ok, f"role {role} re-enabled read_file"
            assert "classified" not in result.output

    def test_role_cannot_reenable_a_tool_a_restricted_session_forbids(self, tmp_path):
        """A restricted (shareable, non-owner) key forbids run_tests as RCE. The
        test-writer role's allowlist DOES include run_tests, so this is the exact
        case where a role could smuggle execution back into a shared session."""
        _, child = _spawn_child(tmp_path, role="test-writer",
                                parent_kwargs={"restricted": True})
        assert "run_tests" in child.disabled_tools
        assert not _dispatch(child, "run_tests").ok
        assert not _dispatch(child, "run_shell", command="echo x").ok

    def test_child_toolset_is_a_subset_of_the_parents(self, tmp_path):
        """The general statement of the invariant, over every role."""
        all_tools = frozenset(TOOL_REGISTRY)
        for parent_disabled in (frozenset(), frozenset({"grep", "git_log"})):
            parent_enabled = all_tools - parent_disabled
            for role in ROLE_PRESETS:
                _, child = _spawn_child(
                    tmp_path, role=role,
                    parent_kwargs={"disabled_tools": parent_disabled})
                child_enabled = all_tools - child.disabled_tools
                assert child_enabled <= parent_enabled, (
                    f"role {role} gained {sorted(child_enabled - parent_enabled)}")

    def test_role_denies_a_dynamically_registered_tool_by_default(self, tmp_path):
        """Roles are ALLOWLISTS: an MCP/plugin/skill tool registered at runtime
        must be denied to a role it was never listed in, not silently inherited."""
        from localm.plugins.coder.tools.registry import ToolDef
        TOOL_REGISTRY["mcp_exfiltrate"] = ToolDef(
            name="mcp_exfiltrate", fn=lambda cwd, **kw: None,
            description="hostile external tool", params={})
        try:
            _, child = _spawn_child(tmp_path, role="reviewer")
            assert "mcp_exfiltrate" in child.disabled_tools
            assert not _dispatch(child, "mcp_exfiltrate").ok
        finally:
            TOOL_REGISTRY.pop("mcp_exfiltrate", None)


# --------------------------------------------------------------------------- #
#  An unknown role must fail loudly, never fall back to full capability
# --------------------------------------------------------------------------- #

class TestUnknownRoleFailsClosed:
    def test_spawn_agent_rejects_an_unknown_role(self, tmp_path):
        result, child = _spawn_child(tmp_path, role="admin")
        assert not result.ok
        assert "unknown role" in result.output
        # It must not have silently spawned a full-capability child instead.
        assert child is None

    def test_resolve_role_raises_rather_than_returning_none(self):
        with pytest.raises(ValueError, match="unknown role"):
            resolve_role("superuser")
        # The message names the real options so the model can retry.
        try:
            resolve_role("superuser")
        except ValueError as exc:
            for name in ROLE_PRESETS:
                assert name in str(exc)

    @pytest.mark.parametrize("bad", [123, ["reviewer"], {"name": "reviewer"}, True])
    def test_a_non_string_role_fails_closed(self, tmp_path, bad):
        """A model can emit anything for an argument; it must not become a
        full-capability child or an obscure AttributeError."""
        result, child = _spawn_child(tmp_path, role=bad)
        assert not result.ok
        assert "unknown role" in result.output
        assert child is None

    def test_no_role_keeps_the_previous_full_toolset(self, tmp_path):
        """Regression guard: roles are opt-in, an ordinary spawn is unchanged."""
        result, child = _spawn_child(tmp_path, role=None)
        assert result.ok
        assert child.role is None
        assert _dispatch(child, "write_file", path="ok.txt", content="x").ok

    def test_underscore_spelling_is_accepted(self, tmp_path):
        _, child = _spawn_child(tmp_path, role="test_writer")
        assert child.role == "test-writer"


# --------------------------------------------------------------------------- #
#  End to end: a role selected the way a MODEL actually selects one
# --------------------------------------------------------------------------- #

class TestModelEmittedRoleReachesTheChild:
    """The tests above call tool_spawn_agent directly. This one goes through the
    parent's own dispatch, the path a model-emitted tool call really takes, so the
    `role` argument is proven to plumb from the tool schema to the narrowing."""

    def test_role_from_a_tool_call_narrows_the_child(self, tmp_path):
        parent = Agent(_StubBackend(), cwd=tmp_path)
        captured = {}

        def _fake_run_task(self, task):
            captured["child"] = self
            return "reviewed"

        with patch.object(Agent, "run_task", _fake_run_task):
            result = _dispatch(parent, "spawn_agent",
                               {"task": "review the diff", "role": "reviewer"})

        assert result.ok, result.output
        child = captured["child"]
        assert child.role == "reviewer"
        assert not _dispatch(child, "write_file",
                             {"path": "evil.txt", "content": "x"}).ok
        assert not (tmp_path / "evil.txt").exists()

    def test_role_is_advertised_in_the_tool_schema(self):
        """A role the model is never told about is a role it will never use."""
        params = TOOL_REGISTRY["spawn_agent"].params
        assert "role" in params
        for name in ROLE_PRESETS:
            assert name in params["role"]["description"]


# --------------------------------------------------------------------------- #
#  The child's prompt: home-anchored cwd, real role signal, no dead advice
# --------------------------------------------------------------------------- #

class TestChildPromptHygiene:
    def test_child_prompt_never_contains_the_raw_absolute_cwd(self, tmp_path):
        """A builder that interpolates {cwd} directly puts the absolute machine
        path and OS username into the prompt, and thus into anything the model
        echoes back."""
        for role in ROLE_PRESETS:
            _, child = _spawn_child(tmp_path, role=role)
            prompt = child._system_prompt
            assert str(tmp_path) not in prompt, role
            assert str(tmp_path.resolve()) not in prompt, role
            # The home-anchored display is what appears instead.
            from localm.plugins.coder.prompts import _display_cwd
            assert _display_cwd(tmp_path) in prompt, role

    def test_codebase_map_header_is_home_anchored_too(self, tmp_path):
        """The map is only emitted for a NON-EMPTY project, so an empty tmp_path
        would not exercise it. Its header must not print the raw absolute root
        into the same prompt, undoing the anchoring three lines above it."""
        (tmp_path / "app.py").write_text("def main():\n    return 1\n")
        _, child = _spawn_child(tmp_path, role="reviewer")
        prompt = child._system_prompt
        assert "Codebase map" in prompt, "map not in prompt - test proves nothing"
        assert str(tmp_path) not in prompt
        assert str(tmp_path.resolve()) not in prompt

    def test_main_agent_prompt_is_also_free_of_the_raw_root(self, tmp_path):
        """The map leak was never role-specific: it hit every coder session."""
        (tmp_path / "app.py").write_text("def main():\n    return 1\n")
        agent = Agent(_StubBackend(), cwd=tmp_path)
        assert "Codebase map" in agent._system_prompt
        assert str(tmp_path) not in agent._system_prompt

    def test_role_brief_itself_is_home_anchored(self, tmp_path):
        brief = build_subagent_system_prompt(tmp_path, "reviewer")
        assert str(tmp_path) not in brief
        from localm.plugins.coder.prompts import _display_cwd
        assert _display_cwd(tmp_path) in brief

    def test_child_prompt_carries_the_role_mission(self, tmp_path):
        """The role must add real signal. Before this, parent and child prompts
        differed by exactly ONE line (the agent name)."""
        _, plain = _spawn_child(tmp_path, role=None)
        _, reviewer = _spawn_child(tmp_path, role="reviewer")
        assert "YOUR ROLE: reviewer" in reviewer._system_prompt
        assert "YOUR ROLE" not in plain._system_prompt
        # The mission text is present.
        assert ROLE_PRESETS["reviewer"].mission[:40] in reviewer._system_prompt

    def test_child_prompt_keeps_the_full_agent_safety_sections(self, tmp_path):
        """A role must not DOWNGRADE the child. A lean builder producing a
        ~500-char prompt with no RULES and no untrusted-content framing is the
        failure; the child keeps the full prompt and gains the role brief on top."""
        _, child = _spawn_child(tmp_path, role="reviewer")
        prompt = child._system_prompt
        assert "RULES" in prompt
        assert "GROUNDING" in prompt
        assert "AVAILABLE TOOLS" in prompt

    def test_narrowed_child_is_not_told_to_use_tools_it_cannot_call(self, tmp_path):
        """Advertising a disabled capability wastes turns and is a confusing
        info-leak. Every role disables spawn_agent and run_shell."""
        _, child = _spawn_child(tmp_path, role="reviewer")
        prompt = child._system_prompt
        assert "use spawn_agent to delegate" not in prompt
        assert "run_shell" not in prompt

    def test_read_only_role_is_not_told_to_edit_or_run_tests(self, tmp_path):
        """A reviewer told to "prefer edit_file" three lines above a brief saying
        it cannot edit contradicts itself and wastes turns on refusals."""
        _, child = _spawn_child(tmp_path, role="reviewer")
        rules = child._system_prompt.split("RULES")[-1].split("YOUR ROLE")[0]
        for gone in ("edit_file", "patch_file", "write_file", "run_tests", "run_shell"):
            assert gone not in rules, f"reviewer RULES still advertise {gone}"

    def test_test_writer_keeps_the_edit_and_test_rules_it_can_use(self, tmp_path):
        """The converse: the prose must not be stripped for a role that CAN edit."""
        _, child = _spawn_child(tmp_path, role="test-writer")
        rules = child._system_prompt.split("RULES")[-1].split("YOUR ROLE")[0]
        assert "edit_file" in rules
        assert "run_tests" in rules
        assert "run_shell" not in rules   # test-writer has no shell

    def test_rules_are_contiguously_numbered_whatever_is_dropped(self, tmp_path):
        """A gap in the numbering reads as a prompt-assembly bug."""
        for role in list(ROLE_PRESETS) + [None]:
            _, child = _spawn_child(tmp_path, role=role)
            rules = child._system_prompt.split("RULES")[-1].split("YOUR ROLE")[0]
            numbers = [int(m) for m in re.findall(r"^(\d+)\. ", rules, re.MULTILINE)]
            assert numbers == list(range(len(numbers))), (role, numbers)
            assert len(numbers) >= 5, (role, numbers)
