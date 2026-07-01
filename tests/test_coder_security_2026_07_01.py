# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-07-01 coder-security backlog cluster.

  AUD-CODERTOOLS - scope allowlist is default-deny (contract test)
  REC-N1-PROSE   - RULES prose + subagent prompt omit run_shell when disabled
  R19a           - unattended one-shot gates run_shell (fail-closed, opt-out --yes)
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
#  AUD-CODERTOOLS - every file-touching tool is scoped or explicitly exempt
# --------------------------------------------------------------------------- #

def test_scope_allowlist_is_default_deny():
    from localm.plugins.coder.tools.registry import TOOL_REGISTRY
    from localm.plugins.coder.agent.constants import (
        _SCOPED_TOOLS, _INTENTIONALLY_UNSCOPED,
    )
    pathy_names = {"path", "glob", "output_path", "input_image",
                   "file", "dir", "pattern", "dest", "src"}
    offenders = []
    for name, tool in TOOL_REGISTRY.items():
        has_path_arg = any(
            p in pathy_names or p.endswith(("_path", "_file", "_dir"))
            for p in tool.params
        )
        if has_path_arg and name not in _SCOPED_TOOLS \
                and name not in _INTENTIONALLY_UNSCOPED:
            offenders.append(name)
    assert not offenders, (
        f"tools with a path-like arg but no scope decision (add to _SCOPED_TOOLS "
        f"or _INTENTIONALLY_UNSCOPED): {offenders} - unconfined by omission")


# --------------------------------------------------------------------------- #
#  REC-N1-PROSE - the prose stops advertising run_shell when it is disabled
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
#  R19a - the one-shot shell gate (auto_approve on, but run_shell confirmed)
# --------------------------------------------------------------------------- #

def _shell_call():
    return ToolCall(name="run_shell", args={"command": "echo hi"},
                    raw="", start=0, end=0)


def test_oneshot_shell_denied_without_yes(tmp_path):
    # This is the config the one-shot CLI now builds when --yes is NOT passed:
    # auto_approve=True (so file writes proceed unattended) but run_shell in
    # always_confirm -> a non-interactive run has no confirmer -> fail closed.
    agent = Agent(_StubBackend(), cwd=tmp_path,
                  auto_approve=True, always_confirm={"run_shell"})
    res = agent._execute_tool(_shell_call(), interactive=False)
    assert not res.ok
    assert "confirmation" in res.output.lower() or "denied" in res.output.lower()


def test_oneshot_shell_runs_with_yes(tmp_path):
    # With --yes the one-shot does NOT add run_shell to always_confirm, so under
    # auto_approve the shell runs (proving the gate is opt-out, not always-on).
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    res = agent._execute_tool(_shell_call(), interactive=False)
    assert res.ok, res.output
