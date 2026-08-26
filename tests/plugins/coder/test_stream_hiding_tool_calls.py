# SPDX-License-Identifier: AGPL-3.0-or-later
"""_stream_hiding_tool_calls (context.py) - the live streaming hider shared by
the CLI's interactive terminal and the GUI's event-sink "token" events (one
implementation, both surfaces).

It hides a call written in any of parser.py's recognised shapes: the
unconditional <tool_call>/<|tool_call> wrapper family, an explicit
```tool_call/```tool_code fence, and a name-gated ```json/bare fence. A fenced
call that is not hidden streams to BOTH the terminal and the GUI as plain
visible text and then executes for real once the full response arrives.

Name-gated fences are decided INCREMENTALLY, not by buffering the whole body to
the fence close: a legitimate large ```json example (the model showing what a
call looks like, or just unrelated JSON data) would otherwise pay for that
buffering even though it is never going to be hidden, and streaming would
visibly "freeze then burst" for anything JSON-shaped. _NameKeyGate (context.py)
scans the object's top-level keys as they arrive - correctly skipping
non-"name" keys of any JSON type (string, number, bool, null, nested
object/array), so "name" is found wherever it appears, not only as the first
key - and releases the moment the answer is knowable: as soon as a "name"
value's accumulated prefix can no longer match any registered tool, or once the
object closes having found no "name" key at all. TestReleaseLatency below turns
"streams normally" into a checkable number. A huge object with NO "name" key at
all is inherently NOT fast (a key cannot be known absent without scanning past
every key that IS present); only the common "name present, wrong value" and
"name absent from an otherwise short object" shapes are.

TestChunkBoundarySweep exists because an incremental design is easy to get
wrong at a chunk boundary: a 4-step sequence (skip whitespace, expect ':', skip
whitespace, look at the value) written as ONE state assumes all 4 steps
complete before it can ever return "need more data", so a call that pauses
right after consuming the colon resumes by re-checking "is the next character a
colon" against a character that came AFTER the colon already consumed -
invisible at whole-string or 1-char-at-a-time delivery, real at other chunk
sizes. Each of those 4 steps is its own persistent state (see
_advance_name_key_gate's docstring). Sweep chunk sizes when testing anything
that decides incrementally; never trust one lucky delivery pattern.
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


def _release_latency(text, tool_names=None):
    """Feed `text` one character at a time; return how many characters had
    been PULLED FROM THE PIECE ITERATOR by the moment the first non-hidden
    output is yielded (None if the whole text stays hidden - e.g. a real
    call with no surrounding narration)."""
    consumed = [0]

    def counting_pieces():
        for c in text:
            consumed[0] += 1
            yield c

    for chunk, hidden in _ContextMixin._stream_hiding_tool_calls(
            counting_pieces(), tool_names=tool_names):
        if not hidden:
            return consumed[0]
    return None


# ---------------------------------------------------------------------------
#  Direct generator tests - the pre-existing <tool_call>/marker path
# ---------------------------------------------------------------------------

class TestCanonicalMarkerBaseline:
    """Hiding of the canonical ``<tool_call>`` marker path."""

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


class TestKeyOrderIndependence:
    """_NameKeyGate finds "name" wherever it appears among an object's
    top-level keys, not only as the first one - the general case a purely
    "check the first key" design would get wrong. Each of these correctly
    SKIPS a non-"name" key's value (string/number/bool/null/nested
    object/array) to keep looking, exactly matching _try_parse_body's own
    order-independent `"name" in obj` check."""

    def test_name_is_the_last_key(self):
        shown, hidden = _hide(
            ['x ```json\n{"args": {"a": 1}, "name": "run_tests"}\n``` y'],
            tool_names={"run_tests"})
        assert shown == "x  y"
        assert '"name": "run_tests"' in hidden

    def test_name_is_in_the_middle(self):
        shown, hidden = _hide(
            ['x ```json\n{"id": 5, "name": "write_file", "args": {}}\n``` y'],
            tool_names={"write_file"})
        assert shown == "x  y"
        assert '"name": "write_file"' in hidden

    def test_nested_name_key_does_not_misfire(self):
        """A "name" key INSIDE a nested object (not top-level) must not be
        mistaken for the call's own name - _try_parse_body would not find it
        either (json.loads gives a top-level dict; "name" in obj checks only
        that dict's own keys)."""
        shown, hidden = _hide(
            ['x ```json\n{"args": {"name": "nested"}, "id": 1}\n``` y'],
            tool_names={"run_tests"})
        assert shown == 'x ```json\n{"args": {"name": "nested"}, "id": 1}\n``` y'
        assert hidden == ""

    def test_nested_array_and_object_before_name(self):
        shown, hidden = _hide([
            'x ```json\n{"items": [1, [2, 3], {"a": 1}], '
            '"name": "run_tests", "args": {}}\n``` y'
        ], tool_names={"run_tests"})
        assert shown == "x  y"
        assert '"name": "run_tests"' in hidden

    def test_bool_null_number_keys_before_name(self):
        shown, hidden = _hide([
            'x ```json\n{"a": true, "b": null, "c": -3.5, '
            '"name": "read_file", "args": {}}\n``` y'
        ], tool_names={"read_file"})
        assert shown == "x  y"
        assert '"name": "read_file"' in hidden

    def test_empty_object_releases(self):
        shown, hidden = _hide(['x ```json\n{}\n``` y'], tool_names={"run_tests"})
        assert shown == 'x ```json\n{}\n``` y'
        assert hidden == ""

    def test_name_value_is_not_a_string(self):
        """"name" present but its value is a nested object, not a string -
        _try_parse_body's isinstance(name, str) check would reject this too."""
        shown, hidden = _hide(['x ```json\n{"name": {"x": 1}}\n``` y'],
                              tool_names={"run_tests"})
        assert shown == 'x ```json\n{"name": {"x": 1}}\n``` y'
        assert hidden == ""


class TestChunkBoundarySweep:
    """Every case below is fed through EVERY chunk size in the sweep, not just
    one. A single fixed split point hides real defects: whole-string and
    1-char-at-a-time delivery can both pass while 3-char chunks do not,
    because a fence header ending precisely at a chunk boundary (the header
    line complete, but the body's first byte not yet in the buffer) is wrongly
    judged "not gate-able" and released instead of waiting one more piece."""

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

    def test_cap_fires_mid_stream_against_a_genuinely_infinite_generator(
            self, monkeypatch):
        """The earlier adversarial tests all feed a FINITE (if huge) piece
        list, so a cap that never actually fires and instead relies on "the
        generator eventually runs out" would still pass them - the trailing
        "if buf: yield ..." at end-of-stream would release it anyway. Feed a
        generator that NEVER terminates and pull only ONE value from the
        gate's output: if the cap works, that first value arrives quickly
        and is bounded in size; if it does not, this hangs forever (proving
        the difference the other tests could not)."""
        def infinite_filler():
            yield "```json\n{"
            while True:
                yield '"a": 1,'

        # Shrink the cap rather than accumulate a real 2,000,000 chars: the
        # property under test is that the cap fires against a generator that
        # never ends. The wall-clock bound below is a coarse slowness guard; the
        # len(chunk) check is what proves the release was bounded.
        monkeypatch.setattr(_ContextMixin, "_MAX_PENDING_FENCE_BODY", 50)
        gen = _ContextMixin._stream_hiding_tool_calls(
            infinite_filler(), tool_names={"run_tests"})
        t0 = time.monotonic()
        chunk, hidden = next(gen)
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0, f"took {elapsed:.3f}s - pathologically slow"
        assert not hidden
        assert len(chunk) < _ContextMixin._MAX_PENDING_FENCE_BODY + 100, (
            f"released chunk is {len(chunk)} chars - cap did not bound it")


class TestReleaseLatency:
    """Upper bounds on how many characters buffer before the gate releases.

    A real tool call's narration is hidden entirely - no separate "release"
    happens for the call span itself, and _release_latency returns None for
    a body with nothing else around it - so these all use a body that is NOT
    a call, where "release" is the observable event.
    """

    TOOLS = {"run_tests", "read_file", "write_file"}

    def test_name_first_wrong_value_releases_within_the_name(self):
        """The common "JSON example that happens to have a name field"
        shape (a person/product/file description) - releases as soon as
        the value diverges from every registered tool's prefix, typically
        well under the length of the object itself."""
        text = '```json\n{"name": "John Smith", "age": 30}\n```'
        lat = _release_latency(text, tool_names=self.TOOLS)
        assert lat is not None
        assert lat <= 25, f"released after {lat} of {len(text)} chars - too slow"

    def test_no_name_key_in_a_short_object_releases_at_its_close(self):
        """No "name" key anywhere: the absence is only known once the object
        itself closes, since every key present has to be scanned past first.
        Release still happens before the fence's own closing backticks."""
        obj = '{"users": [1, 2, 3], "count": 3}'
        text = "```json\n" + obj + "\n```"
        lat = _release_latency(text, tool_names=self.TOOLS)
        assert lat is not None
        assert lat <= len(obj) + 10, f"released after {lat}, object is {len(obj)} chars"
        assert lat < len(text), "must release before the fence's own close, not just at it"

    def test_large_json_with_no_name_key_is_bounded_by_the_object_not_the_whole_response(self):
        """For an object with NO "name" key, release happens once that object
        closes - proportional to the OBJECT's size, not to however much
        unrelated text follows it in the same response."""
        big_obj = '{"items": ' + str(list(range(2000))) + "}"
        text = "```json\n" + big_obj + "\n```" + ("\nmore narration after. " * 50)
        lat = _release_latency(text, tool_names=self.TOOLS)
        assert lat is not None
        assert lat <= len(big_obj) + 20, (
            f"released after {lat}, object alone is {len(big_obj)} chars")
        assert lat < len(big_obj) + 20 < len(text), (
            "must not have waited for the trailing narration too")

    def test_unregistered_tool_name_releases_fast_even_with_a_perfect_shape(self):
        """A body shaped EXACTLY like a real call, differing only in the
        name - still releases fast, proving the gate checks the value
        against the registry rather than just "looks call-shaped"."""
        text = '```json\n{"name": "delete_everything", "args": {}}\n```'
        lat = _release_latency(text, tool_names=self.TOOLS)
        assert lat is not None
        assert lat <= 25, f"released after {lat} of {len(text)} chars - too slow"

    def test_a_real_call_is_never_released_early(self):
        """Control: the positive case must NOT show this fast-release
        behaviour - a genuine call stays fully hidden, with no separate
        visible chunk at all."""
        text = '```json\n{"name": "run_tests", "args": {}}\n```'
        lat = _release_latency(text, tool_names=self.TOOLS)
        assert lat is None, f"a real call must never release early, got {lat}"


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
    """The GUI path. Kept as a companion to the interactive test above so a
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
