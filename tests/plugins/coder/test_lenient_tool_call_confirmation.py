# SPDX-License-Identifier: AGPL-3.0-or-later
"""A tool call recovered ONLY via parser.py's name-gated lenient fallback (no
<tool_call> wrapper, no explicit fence, no marker of any kind - just a JSON
blob whose shape happens to match a real tool name) must not be allowed to
silently execute a destructive tool under auto_approve, even though an
ordinary, marker-carrying call still does.
"""

from __future__ import annotations

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _write_call(lenient: bool, rel: str = "out.txt") -> ToolCall:
    return ToolCall(name="write_file", args={"path": rel, "content": "hi"},
                    raw="", start=0, end=0, lenient=lenient)


def test_a_lenient_recovered_destructive_call_is_denied_under_auto_approve(tmp_path):
    """auto_approve=True normally skips confirmation for a destructive tool, but
    a call recovered with no marker of tool-call intent at all is still denied
    when nobody is available to confirm it (fail-closed, the same path an
    explicit always_confirm entry takes)."""
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    result = agent._execute_tool(_write_call(lenient=True), interactive=False)
    assert not result.ok
    assert "requires confirmation" in result.output
    assert not (tmp_path / "out.txt").exists()


def test_fires_control_the_same_call_not_flagged_lenient_still_auto_approves(tmp_path):
    """Identical call, identical auto_approve=True, only the lenient flag
    differs: the ordinary trusted path is unaffected."""
    agent = Agent(_StubBackend(), cwd=tmp_path, auto_approve=True)
    result = agent._execute_tool(_write_call(lenient=False), interactive=False)
    assert result.ok, result.output
    assert (tmp_path / "out.txt").read_text() == "hi"
