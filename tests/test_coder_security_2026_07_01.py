# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the coder-security backlog cluster.

  scope allowlist is default-deny (contract test)
  RULES prose + subagent prompt omit run_shell when disabled
  unattended one-shot gates run_shell (fail-closed, opt-out --yes)
"""

from pathlib import Path

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.prompts import (
    _rules_section, build_subagent_system_prompt, build_system_prompt,
)


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


# --------------------------------------------------------------------------- #
#  Every file-touching tool is scoped or explicitly exempt
# --------------------------------------------------------------------------- #

_PATHY_NAMES = {"path", "glob", "output_path", "input_image",
                "file", "dir", "pattern", "dest", "src"}


def _is_pathy(param_name: str) -> bool:
    return (param_name in _PATHY_NAMES
            or param_name.endswith(("_path", "_file", "_dir")))


def _has_path_arg(tool) -> bool:
    """True if the tool takes a filesystem path in ANY of its params.

    Covers both shapes: a top-level path-like arg, and a path NESTED inside a
    collection arg (edit_files' ``edits=[{path, old, new}]``). The nested shape
    is the one that hides: its own param name ("edits") is not path-like, so a
    name-only check would clear the tool while its real targets go unconfined.
    """
    for param_name, meta in tool.params.items():
        if _is_pathy(param_name):
            return True
        items = (meta or {}).get("items") or {}
        if any(_is_pathy(k) for k in (items.get("properties") or {})):
            return True
    return False


def test_scope_allowlist_is_default_deny():
    from localm.plugins.coder.tools.registry import TOOL_REGISTRY
    from localm.plugins.coder.agent.constants import (
        _SCOPED_TOOLS, _INTENTIONALLY_UNSCOPED,
    )
    offenders = []
    for name, tool in TOOL_REGISTRY.items():
        if _has_path_arg(tool) and name not in _SCOPED_TOOLS \
                and name not in _INTENTIONALLY_UNSCOPED:
            offenders.append(name)
    assert not offenders, (
        f"tools with a path-like arg but no scope decision (add to _SCOPED_TOOLS "
        f"or _INTENTIONALLY_UNSCOPED): {offenders} - unconfined by omission")


def test_default_deny_check_sees_a_nested_path_arg():
    """Negative test for the contract test above: a tool whose paths live inside
    a collection arg must be DETECTED as path-taking. Without this, a nested-path
    tool passes the scope check by having no `path` arg to check - a fail-OPEN,
    which is the exact opposite of what the allowlist is for."""
    from localm.plugins.coder.tools.registry import TOOL_REGISTRY

    class _NestedPathTool:
        params = {"edits": {"type": "array", "items": {
            "type": "object",
            "properties": {"path": {}, "old": {}, "new": {}}}}}

    assert _has_path_arg(_NestedPathTool()) is True
    # And the real registry entry is detected the same way.
    assert _has_path_arg(TOOL_REGISTRY["edit_files"]) is True

    class _NoPathTool:
        params = {"query": {"type": "string"}, "n": {"type": "int"}}

    assert _has_path_arg(_NoPathTool()) is False


def test_nested_path_tool_scope_check_reads_the_nested_paths():
    """The resolver the scope check depends on must actually see nested paths -
    an empty result would silently allow everything."""
    from localm.plugins.coder.agent.constants import (
        _SCOPED_TOOLS, _call_target_paths,
    )
    assert "edit_files" in _SCOPED_TOOLS
    assert _call_target_paths(
        "edit_files",
        {"edits": [{"path": "a.py", "old": "x", "new": "y"},
                   {"path": "../escape.py", "old": "x", "new": "y"}]},
    ) == ["a.py", "../escape.py"]


# --------------------------------------------------------------------------- #
#  The prose stops advertising run_shell when it is disabled
# --------------------------------------------------------------------------- #

def test_rules_prose_omits_run_shell_when_disabled():
    on = _rules_section("default", disabled_tools=frozenset())
    off = _rules_section("default", disabled_tools=frozenset({"run_shell"}))
    assert "run_shell" in on
    assert "run_shell" not in off


def test_subagent_prompt_omits_run_shell_when_disabled():
    on = build_subagent_system_prompt(Path("."), "helper")
    off = build_subagent_system_prompt(Path("."), "helper",
                                       disabled_tools=frozenset({"run_shell"}))
    assert "run_shell" in on
    assert "run_shell" not in off


def test_full_system_prompt_run_shell_free_when_disabled(tmp_path):
    # With run_shell disabled, the tool docs AND the RULES prose drop it, so a
    # restricted coder is never told about a capability it cannot use.
    restricted = build_system_prompt(tmp_path, model_name="generic",
                                     disabled_tools=frozenset({"run_shell"}))
    assert "run_shell" not in restricted


# --------------------------------------------------------------------------- #
#  The one-shot shell gate (auto_approve on, but run_shell confirmed)
# --------------------------------------------------------------------------- #

def _shell_call():
    return ToolCall(name="run_shell", args={"command": "echo hi"},
                    raw="", start=0, end=0)


def test_oneshot_shell_denied_without_yes(tmp_path):
    # The config the one-shot CLI builds when --yes is NOT passed: auto_approve=True
    # (file writes proceed unattended) but run_shell in always_confirm, so a
    # non-interactive run has no confirmer and fails closed.
    agent = Agent(_StubBackend(), cwd=tmp_path,
                  auto_approve=True, always_confirm={"run_shell"})
    res = agent._execute_tool(_shell_call(), interactive=False)
    assert not res.ok
    assert "confirmation" in res.output.lower() or "denied" in res.output.lower()


def test_oneshot_shell_runs_with_yes(tmp_path):
    # With --yes the one-shot does NOT add run_shell to always_confirm, so under
    # auto_approve the shell runs.
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    res = agent._execute_tool(_shell_call(), interactive=False)
    assert res.ok, res.output
