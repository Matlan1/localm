# SPDX-License-Identifier: AGPL-3.0-or-later
"""_stream_hiding_tool_calls (context.py) - the live streaming hider shared by
the CLI's interactive terminal and the GUI's event-sink "token" events
(CODER-3: one implementation, both surfaces).

Extended 2026-07-30 to also hide a call written in one of parser.py's OTHER
recognised shapes (an explicit ```tool_call/```tool_code fence, or a
name-gated ```json/bare fence) - previously only the unconditional
<tool_call>/<|tool_call> wrapper family was hidden here, so a fenced call
streamed to BOTH the terminal and the GUI as plain visible text and then
executed for real once the full response arrived (the leak this file's
sibling, test_assistant_text_correction.py, covers from the GUI-correction
side). This file is also the first DIRECT coverage of the hider itself - the
existing <tool_call> marker path had no dedicated test before this; the
canonical-XML cases below are a regression baseline for it, not just for the
new fence handling.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent.context import _ContextMixin


def _hide(pieces, tool_names=None):
    """Drive _stream_hiding_tool_calls directly and split shown vs hidden."""
    shown, hidden = [], []
    for text, is_hidden in _ContextMixin._stream_hiding_tool_calls(
            iter(pieces), tool_names=tool_names):
        (hidden if is_hidden else shown).append(text)
    return "".join(shown), "".join(hidden)


# ---------------------------------------------------------------------------
#  Direct generator tests - the pre-existing <tool_call>/marker path
# ---------------------------------------------------------------------------

class TestCanonicalMarkerBaseline:
    """No dedicated test covered this before; a regression guard for the
    already-shipped behaviour this change must not disturb."""

    def test_xml_tool_call_fully_hidden(self):
        shown, hidden = _hide([
            "Reading.\n\n<tool_call>\n",
            '{"name": "read_file", "args": {"path": "a.py"}}\n',
            "</tool_call>",
        ])
        assert shown == "Reading.\n\n"
        assert '"name": "read_file"' in hidden

    def test_marker_split_across_chunk_boundary(self):
        """A chunk boundary landing mid-marker must not leak a fragment."""
        shown, hidden = _hide(["Hi <tool", "_call>x</tool_call>", " bye"])
        assert shown == "Hi  bye"
        assert hidden == "<tool_call>x</tool_call>"

    def test_unclosed_marker_at_stream_end_shown_hidden(self):
        shown, hidden = _hide(["before <tool_call>never closes"])
        assert shown == "before "
        assert hidden == "<tool_call>never closes"

    def test_plain_text_with_no_marker_streams_untouched(self):
        shown, hidden = _hide(["just ", "ordinary ", "prose, no markers here"])
        assert shown == "just ordinary prose, no markers here"
        assert hidden == ""


# ---------------------------------------------------------------------------
#  Direct generator tests - the new fence handling
# ---------------------------------------------------------------------------

class TestFenceHiding:
    def test_explicit_tool_call_fence_hidden_unconditionally(self):
        """No tool_names needed - the wrapper itself signals intent, same as
        parse_tool_calls treats it."""
        shown, hidden = _hide(
            ['```tool_call\n{"name": "run_tests", "args": {}}\n```'],
            tool_names=None,
        )
        assert shown == ""
        assert '"name": "run_tests"' in hidden

    def test_explicit_tool_code_fence_hidden(self):
        shown, hidden = _hide(
            ['```tool_code\n{"name": "run_tests", "args": {}}\n```'])
        assert shown == ""
        assert '"name": "run_tests"' in hidden

    def test_json_fence_for_a_real_tool_hidden(self):
        """The exact reported leak shape: narration, a ```json fence for a
        registered tool, trailing prose."""
        shown, hidden = _hide([
            "Now I'll run the tests.\n\n"
            '```json\n{"name": "run_tests", "args": {}}\n```\n\n'
            "fake output here"
        ], tool_names={"run_tests"})
        assert '"name": "run_tests"' not in shown
        assert "```" not in shown
        assert "Now I'll run the tests." in shown
        assert "fake output here" in shown
        assert '"name": "run_tests"' in hidden

    def test_bare_fence_no_lang_for_a_real_tool_hidden(self):
        shown, hidden = _hide(
            ['```\n{"name": "run_tests", "args": {}}\n```'],
            tool_names={"run_tests"})
        assert shown == ""
        assert '"name": "run_tests"' in hidden

    def test_json_fence_with_unregistered_name_not_hidden(self):
        """A prose example naming a tool that does not exist must display -
        exactly mirrors parse_tool_calls' own name-gate."""
        shown, hidden = _hide([
            'example: ```json\n{"name": "made_up_tool", "args": {}}\n``` ok'
        ], tool_names={"run_tests"})
        assert '"name": "made_up_tool"' in shown
        assert hidden == ""

    def test_no_tool_names_supplied_never_hides_ambiguous_fence(self):
        """Safe default: with nothing to gate against, an ambiguous fence
        always displays - matches pre-extension behaviour exactly."""
        shown, hidden = _hide(
            ['```json\n{"name": "run_tests", "args": {}}\n```'],
            tool_names=None)
        assert '"name": "run_tests"' in shown
        assert hidden == ""

    def test_ordinary_code_fence_not_delayed_or_hidden(self):
        shown, hidden = _hide(
            ["```python\ndef f():\n    return 1\n```\nmore text"],
            tool_names={"run_tests"})
        assert "def f()" in shown
        assert hidden == ""

    def test_fence_json_body_but_no_name_key_not_hidden(self):
        """A fence whose body is valid JSON but not {"name": ..., "args": ...}
        shaped must display - it never gets treated as call-shaped."""
        shown, hidden = _hide(
            ['```json\n{"just": "data", "nothing": "special"}\n```'],
            tool_names={"run_tests"})
        assert '"just": "data"' in shown
        assert hidden == ""

    def test_unclosed_ambiguous_fence_released_not_lost_forever(self):
        """Unlike the marker path, an unclosed name-gated fence must be
        RELEASED at stream end, not hidden - parse_tool_calls can never treat
        an unclosed fence as a real call either, so hiding it here could hide
        genuine truncated prose with nothing downstream to ever reveal it."""
        shown, hidden = _hide(
            ['Text before ```json\n{"name": "run_tests"'],
            tool_names={"run_tests"})
        assert '"name": "run_tests"' in shown
        assert hidden == ""

    def test_unclosed_explicit_fence_stays_hidden_like_a_marker(self):
        """The explicit wrapper is unconditional, same semantics as an
        unclosed <tool_call> marker - and looks_like_tool_attempt() already
        recognises this shape, so the repair-turn machinery still fires."""
        shown, hidden = _hide(['before ```tool_call\nnever closes'])
        assert shown == "before "
        assert "never closes" in hidden

    def test_fence_split_across_many_small_chunks(self):
        """Same call as the reported leak, fed one small chunk at a time -
        proves the header/body scanning survives arbitrary chunk boundaries,
        not just whole-piece delivery."""
        text = ('Sure.\n\n```json\n{"name": "run_tests", "args": {}}\n```\n\ndone')
        pieces = [text[i:i + 3] for i in range(0, len(text), 3)]
        shown, hidden = _hide(pieces, tool_names={"run_tests"})
        assert shown == "Sure.\n\n\n\ndone"
        assert '"name": "run_tests"' in hidden

    def test_tc_marker_and_fence_both_present_in_one_response(self):
        """A marker-style call followed by a fence-style call in the SAME
        response - both recognised independently."""
        shown, hidden = _hide([
            '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n'
            '</tool_call>\n\nthen ```json\n{"name": "run_tests", "args": {}}\n```'
        ], tool_names={"run_tests"})
        assert shown == "\n\nthen "
        assert '"name": "read_file"' in hidden
        assert '"name": "run_tests"' in hidden


class TestChunkBoundarySweep:
    """Every case below is fed through EVERY chunk size in the sweep, not just
    one - a single fixed split point is exactly how the real bug here shipped
    unnoticed: whole-string and 1-char-at-a-time delivery both passed, but
    3-char chunks did not, because a fence header ending precisely at a chunk
    boundary (the header line complete, but the body's first byte not yet in
    the buffer) was wrongly judged "not gate-able" and released instead of
    waiting one more piece. Fixed in context.py; this sweep is what should
    have caught it, and is what must keep catching its like."""

    _CASES = [
        pytest.param(
            "Sure.\n\n```json\n{\"name\": \"run_tests\", \"args\": {}}\n```\n\ndone",
            {"run_tests"}, "Sure.\n\n\n\ndone", '"name": "run_tests"',
            id="json-fence-real-tool"),
        pytest.param(
            'Ok ```tool_call\n{"name": "run_tests", "args": {}}\n``` end',
            None, "Ok  end", '"name": "run_tests"',
            id="explicit-tool-call-fence"),
        pytest.param(
            'x```\n{"name": "run_tests", "args": {}}\n```y',
            {"run_tests"}, "xy", '"name": "run_tests"',
            id="bare-fence-real-tool"),
        pytest.param(
            'a ```json\n{"name": "nope", "args": {}}\n``` b',
            {"run_tests"}, 'a ```json\n{"name": "nope", "args": {}}\n``` b', None,
            id="unregistered-name-stays-visible"),
        pytest.param(
            'Reading.\n\n<tool_call>\n{"name": "read_file", '
            '"args": {"path": "a.py"}}\n</tool_call>',
            {"run_tests"}, "Reading.\n\n", '"name": "read_file"',
            id="xml-tool-call-unaffected"),
        pytest.param(
            "```python\ndef f():\n    return 1\n```\nmore",
            {"run_tests"}, "```python\ndef f():\n    return 1\n```\nmore", None,
            id="ordinary-python-fence-never-hidden"),
    ]

    @pytest.mark.parametrize("text,tool_names,expect_shown,expect_hidden_substr", _CASES)
    @pytest.mark.parametrize("chunk", list(range(1, 15)) + [20, 33, 10_000])
    def test_result_identical_at_every_chunk_size(
            self, text, tool_names, expect_shown, expect_hidden_substr, chunk):
        pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)] or [""]
        shown, hidden = _hide(pieces, tool_names=tool_names)
        assert shown == expect_shown, f"chunk={chunk}: {shown!r} != {expect_shown!r}"
        if expect_hidden_substr is not None:
            assert expect_hidden_substr in hidden, f"chunk={chunk}: not hidden"
            assert expect_hidden_substr not in shown, f"chunk={chunk}: leaked into shown"


class TestFenceBodyCapGivesUp:
    """The bound that stops a never-closing ambiguous fence from being held
    back forever (mirrors parser.py's _MAX_EXPENSIVE_MARKER_RESCANS reasoning:
    cap the pathological case, release rather than hang or hide forever)."""

    def test_body_exceeding_the_cap_is_released_not_lost(self, monkeypatch):
        monkeypatch.setattr(_ContextMixin, "_MAX_PENDING_FENCE_BODY", 50)
        pieces = ['```json\n{'] + ['"a": 1,'] * 20   # grows well past 50 chars
        shown, hidden = _hide(pieces, tool_names={"run_tests"})
        assert hidden == "", "must not stay hidden forever once the cap is exceeded"
        assert '"a": 1' in shown, "buffered content must be released, not dropped"

    def test_adversarial_never_closing_fence_body_stays_fast(self):
        """No artificially lowered cap here - the REAL 2,000,000-char default,
        proving the whole-stream cost stays low even under many small pieces
        (bounded, not the O(pieces x buffer length) blowup this codebase has
        been bitten by before in exactly this class of code)."""
        pieces = ['```json\n{'] + ['"a": 1,'] * 3000
        t0 = time.monotonic()
        shown, hidden = _hide(pieces, tool_names={"run_tests"})
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"took {elapsed:.3f}s - too slow"
        assert '"a": 1' in shown

    def test_real_cap_fires_on_a_realistic_truncated_write_file_body(self):
        """Not a synthetic filler pattern: a body shaped like a real, large
        write_file call whose model turn got cut off mid-generation (hit
        max_tokens) before the closing fence ever arrived - the release valve
        must fire on ordinary large content, not only on adversarial
        repetition, and must do so at the REAL 2,000,000-char default, not
        an artificially lowered one."""
        # A source file's worth of plausible-looking Python lines as the
        # "content" value, JSON-escaped, comfortably past the real cap.
        line = '    result = compute(x, y, z)  # step %d\\n'
        content_value = "".join(line % i for i in range(60_000))  # ~2.1M chars
        body_open = '```json\n{"name": "write_file", "args": {"path": "big.py", "content": "'
        pieces = [body_open] + [content_value[i:i + 4000]
                                for i in range(0, len(content_value), 4000)]
        # No closing fence at all - the truncated-turn scenario.
        t0 = time.monotonic()
        shown, hidden = _hide(pieces, tool_names={"write_file"})
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"took {elapsed:.3f}s - too slow"
        assert hidden == "", "a truncated call must never vanish - nothing to correct it later"
        assert "result = compute(x, y, z)" in shown, "released content must be intact, not dropped"
        assert body_open in shown

    def test_adversarial_fence_header_never_completes(self):
        """A ``` with a long run of characters and no newline ever - must not
        hang waiting for a header that will never arrive."""
        pieces = ["```"] + ["a"] * 5000
        t0 = time.monotonic()
        shown, hidden = _hide(pieces)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"took {elapsed:.3f}s - too slow"
        assert shown.startswith("```aaa")


# ---------------------------------------------------------------------------
#  Integration: through the real _call_llm, both dispatch branches
# ---------------------------------------------------------------------------

def _make_agent(tmp_path: Path, on_event=None) -> object:
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
        agent = Agent(backend=backend, cwd=tmp_path, on_event=on_event)
    agent._audit = MagicMock()
    return agent


def _stream_backend(agent, full_text, chunk_size=7):
    pieces = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    def fake_chat_stream(messages, on_reasoning=None, **kw):
        yield from pieces
    agent.backend.chat_stream.side_effect = fake_chat_stream


_LEAKED = ("Now I'll run the tests using `pytest`.\n\n"
          '```json\n{"name": "run_tests", "args": {}}\n```\n\n'
          "fake output here")


class TestCallLLMInteractiveHidesFence:
    """The CLI terminal path: this is the surface the retroactive GUI
    correction (loop.py's assistant_text event) cannot reach - a terminal
    cannot un-print - so it must be fixed at the SOURCE, here."""

    def test_json_fence_call_never_printed_to_terminal(self, tmp_path):
        agent = _make_agent(tmp_path)
        _stream_backend(agent, _LEAKED)
        printed = []
        with patch("localm.plugins.coder.agent.context.print_streaming_token",
                   side_effect=lambda t, **kw: printed.append(t)), \
             patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_thinking"), \
             patch("localm.plugins.coder.agent.context.print_assistant_label"), \
             patch("localm.plugins.coder.agent.context.print_streaming_done"), \
             patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY",
                        {"run_tests": MagicMock(destructive=False)}):
            result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        shown = "".join(printed)
        assert result == _LEAKED, "the full raw response is still captured/returned"
        assert '"name": "run_tests"' not in shown
        assert "```" not in shown
        assert "Now I'll run the tests" in shown
        assert "fake output here" in shown

    def test_unregistered_tool_name_still_prints(self, tmp_path):
        agent = _make_agent(tmp_path)
        _stream_backend(agent, 'See: ```json\n{"name": "not_a_real_tool", '
                                '"args": {}}\n``` example')
        printed = []
        with patch("localm.plugins.coder.agent.context.print_streaming_token",
                   side_effect=lambda t, **kw: printed.append(t)), \
             patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_thinking"), \
             patch("localm.plugins.coder.agent.context.print_assistant_label"), \
             patch("localm.plugins.coder.agent.context.print_streaming_done"):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)
        assert '"name": "not_a_real_tool"' in "".join(printed)

    def test_canonical_xml_tool_call_regression_guard(self, tmp_path):
        agent = _make_agent(tmp_path)
        _stream_backend(agent, 'Reading.\n\n<tool_call>\n{"name": "read_file", '
                                '"args": {"path": "a.py"}}\n</tool_call>')
        printed = []
        with patch("localm.plugins.coder.agent.context.print_streaming_token",
                   side_effect=lambda t, **kw: printed.append(t)), \
             patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_thinking"), \
             patch("localm.plugins.coder.agent.context.print_assistant_label"), \
             patch("localm.plugins.coder.agent.context.print_streaming_done"):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)
        assert "".join(printed) == "Reading.\n\n"


class TestCallLLMEventSinkHidesFence:
    """The GUI path: this alone used to be the only place the (incomplete)
    hiding applied. Kept as a companion to the interactive test above so a
    future change to one branch cannot silently stop covering the other."""

    def test_json_fence_call_never_emitted_as_a_token_event(self, tmp_path):
        events = []
        agent = _make_agent(tmp_path, on_event=events.append)
        _stream_backend(agent, _LEAKED)
        with patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY",
                        {"run_tests": MagicMock(destructive=False)}):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
        token_text = "".join(e["text"] for e in events if e["type"] == "token")
        assert '"name": "run_tests"' not in token_text
        assert "Now I'll run the tests" in token_text
