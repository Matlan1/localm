# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder GUI's tool-call display leak.

context.py's live-streaming hider (_stream_hiding_tool_calls) only recognises
the UNCONDITIONAL <tool_call>/<|tool_call> wrapper dialects, because those are
the only shapes it can safely hide with no lookahead. parser.py's
parse_tool_calls recognises several MORE shapes once the full response has
arrived and the tool registry is known: an explicit ```tool_call/```tool_code
fence, a name-gated ```json/bare fence, and a bare top-level JSON object. A
call written in one of those extra shapes streams to the GUI as plain visible
text (the live hider has no idea it will turn out to be a real call) and is
THEN executed for real once parse_tool_calls runs - so the chat bubble is left
showing the executed call's own raw JSON, indistinguishable from prose.

_LEAKED_RESPONSE below is a real captured shape: narration, a ```json fence for
run_tests, then a block of pytest-looking text the model wrote as its own prose.
That trailing block is NOT itself tool-call-shaped and must survive uncorrected:
only EXECUTED spans are stripped, never anything that merely resembles tool
output.

Once parse_tool_calls/split_response know which spans of the response were REAL
calls, loop.py emits an "assistant_text" event carrying the authoritative
leftover text, so the event-sink (GUI) can fix up whatever it already streamed
for every shape parser.py recognises.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.tools import ToolResult


def _make_agent(tmp_path: Path, **kwargs) -> object:
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    events = []
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, on_event=events.append, **kwargs)
    agent._events = events
    return agent


def _stub_run_tests(result=None):
    """Patch TOOL_REGISTRY's run_tests so the real pytest suite never runs -
    same pattern as TestUnverifiedWriteTracking in test_agent_loop_guards.py."""
    tool_def = MagicMock()
    tool_def.destructive = False
    tool_def.fn = MagicMock(return_value=result or ToolResult.success(
        "2 passed in 0.03s", summary="2 passed"))
    return patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY",
                      {"run_tests": tool_def})


_NARRATION = ("Now I'll run the tests using `pytest` to ensure everything is "
             "working correctly.")
_FENCE_CALL = '```json\n{"name": "run_tests", "args": {}}\n```'
_HALLUCINATED_OUTPUT = (
    "==================== test session starts ====================\n"
    "collected 2 items\n"
    "==================== 2 passed in 0.03s ====================")
_LEAKED_RESPONSE = f"{_NARRATION}\n\n{_FENCE_CALL}\n\n{_HALLUCINATED_OUTPUT}"


class TestAssistantTextCorrection:
    def test_executed_json_fence_call_is_removed_from_the_correction_event(self, tmp_path):
        agent = _make_agent(tmp_path)
        responses = iter([_LEAKED_RESPONSE, "Done."])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             _stub_run_tests():
            agent.run_task("add whisper() and test it")

        # The call was executed.
        tool_calls = [e for e in agent._events if e["type"] == "tool_call"]
        assert [c["tool"] for c in tool_calls] == ["run_tests"]

        # Exactly one correction event for this turn.
        corrections = [e for e in agent._events if e["type"] == "assistant_text"]
        assert len(corrections) == 1
        corrected = corrections[0]["text"]

        # The executed call's raw JSON and fence markers are gone from it.
        assert '"name": "run_tests"' not in corrected
        assert "```" not in corrected

        # The narration and the hallucinated prose survive verbatim.
        assert _NARRATION in corrected
        assert _HALLUCINATED_OUTPUT in corrected

    def test_no_correction_event_when_nothing_was_called(self, tmp_path):
        """A plain final answer (no tool calls) needs no fix-up: whatever
        streamed IS the real answer, so firing a correction here would be
        pure noise on every ordinary turn."""
        agent = _make_agent(tmp_path)
        with patch.object(agent, "_call_llm", return_value="Just a plain answer."):
            agent.run_task("explain this function")

        assert not [e for e in agent._events if e["type"] == "assistant_text"]

    def test_correction_matches_what_already_streamed_for_the_canonical_xml_form(self, tmp_path):
        """Control: the ALREADY-working case (a canonical <tool_call> wrapper,
        which the live hider already hides correctly) must not regress - the
        correction should just restate the same leftover text, not something
        different or extra."""
        agent = _make_agent(tmp_path)
        responses = iter([
            'Sure, reading it now.\n\n'
            '<tool_call>\n{"name": "run_tests", "args": {}}\n</tool_call>',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             _stub_run_tests():
            agent.run_task("run the tests")

        corrections = [e for e in agent._events if e["type"] == "assistant_text"]
        assert len(corrections) == 1
        assert corrections[0]["text"].strip() == "Sure, reading it now."

    def test_bare_json_call_with_no_fence_is_also_removed(self, tmp_path):
        """Generality check: the fix is not fence-specific. A bare top-level
        JSON object (parser.py's other name-gated shape, no wrapper at all) is
        just as invisible to the live hider and must be corrected the same way."""
        agent = _make_agent(tmp_path)
        responses = iter([
            'Running the suite now.\n\n{"name": "run_tests", "args": {}}',
            "Done.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)), \
             _stub_run_tests():
            agent.run_task("run the tests")

        tool_calls = [e for e in agent._events if e["type"] == "tool_call"]
        assert [c["tool"] for c in tool_calls] == ["run_tests"]
        corrections = [e for e in agent._events if e["type"] == "assistant_text"]
        assert len(corrections) == 1
        assert '"name": "run_tests"' not in corrections[0]["text"]
        assert "Running the suite now." in corrections[0]["text"]
