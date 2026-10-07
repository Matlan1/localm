# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the shared control-marker scrubber and its application at the engine
layer, so channel/harmony tokens never leak to the GUI regardless of backend.
"""

from localm.inference.engine import Engine
from localm.textnorm import (
    ThinkSplitter, scrub_stream, scrub_text, split_think,
)


def _scrub(pieces):
    return "".join(scrub_stream(iter(pieces)))


def _stream_split(pieces):
    """Drive ThinkSplitter over *pieces* and return (content, reasoning)."""
    sp = ThinkSplitter()
    cs, rs = [], []
    for p in pieces:
        c, r = sp.feed(p)
        cs.append(c)
        rs.append(r)
    c, r = sp.flush()
    cs.append(c)
    rs.append(r)
    return "".join(cs), "".join(rs)


class TestSplitThink:
    def test_one_shot_separates_block(self):
        c, r = split_think("<think>\nreasoning here\n</think>\nThe answer.")
        assert r.strip() == "reasoning here"
        assert c.strip() == "The answer."

    def test_no_think_is_all_content(self):
        c, r = split_think("Just an answer, no reasoning.")
        assert c == "Just an answer, no reasoning."
        assert r == ""

    def test_unclosed_think_runs_to_end(self):
        c, r = split_think("<think>still thinking and never closed")
        assert c == ""
        assert "still thinking" in r

    def test_content_before_and_after_block(self):
        c, r = split_think("intro <think>mid</think> outro")
        assert c == "intro  outro"
        assert r == "mid"

    def test_streaming_matches_one_shot(self):
        full = "Pre <think>because reasons</think> Post."
        # A naive concatenation of content deltas must equal the one-shot
        # content, and the tags must never appear in either channel.
        c_stream, r_stream = _stream_split(list(full))   # one char at a time
        c_once, r_once = split_think(full)
        assert c_stream == c_once
        assert r_stream == r_once
        assert "<think>" not in c_stream and "</think>" not in c_stream
        assert "<think>" not in r_stream and "</think>" not in r_stream

    def test_tag_split_across_pieces(self):
        # The open tag is fragmented across three feeds; it must still be removed
        # and the reasoning routed correctly (no leaked "<thi" / "nk>").
        c, r = _stream_split(["answer <thi", "nk>secret rea", "soning</thi", "nk> done"])
        assert c == "answer  done"
        assert r == "secret reasoning"
        assert "<thi" not in c and "nk>" not in c

    def test_multiple_blocks_concatenate(self):
        c, r = split_think("<think>a</think>X<think>b</think>Y")
        assert c == "XY"
        assert "a" in r and "b" in r

    def test_lt_in_prose_is_not_a_tag(self):
        c, r = split_think("if x < 3 and y > 2 then ok")
        assert c == "if x < 3 and y > 2 then ok"
        assert r == ""


class TestNativeReasoningTags:
    """Native reasoning tags emitted WITHOUT a channel wrapper (<reasoning>,
    <thinking>, <thought>, <reflection>) must normalise to canonical <think>, so
    the reasoning/content split routes them instead of letting them leak."""

    def test_bare_reasoning_becomes_think(self):
        assert scrub_text("<reasoning>why</reasoning>The answer.") == \
            "<think>why</think>The answer."

    def test_routed_into_reasoning_not_content(self):
        c, r = split_think(scrub_text("<reasoning>secret</reasoning>Hello."))
        assert c.strip() == "Hello."
        assert r.strip() == "secret"

    def test_all_aliases(self):
        for tag in ("thinking", "thought", "reflection"):
            assert scrub_text(f"<{tag}>r</{tag}>A") == "<think>r</think>A", tag

    def test_case_insensitive(self):
        assert scrub_text("<Reasoning>r</REASONING>A") == "<think>r</think>A"

    def test_canonical_think_and_other_tags_untouched(self):
        assert scrub_text("<think>r</think>A") == "<think>r</think>A"
        assert scrub_text("<random>x</random>A") == "<random>x</random>A"

    def test_streaming_split_tag_normalised(self):
        # The bare tag fragmented across stream pieces still normalises + routes.
        out = _scrub(["pre <reaso", "ning>why</reason", "ing> post"])
        c, r = split_think(out)
        assert r.strip() == "why"
        assert "reaso" not in c and "reason" not in c
        assert "pre" in c and "post" in c


class TestSharedScrub:
    def test_gemma_channel_pair_becomes_think(self):
        text = "<|channel>thought\nhmm<channel|>Good morning!"
        assert _scrub([text]) == "<think>\nhmm\n</think>\nGood morning!"

    def test_harmony_channels_become_think(self):
        text = ("<|channel|>analysis<|message|>Reasoning."
                "<|channel|>final<|message|>The answer.")
        assert _scrub([text]) == "<think>\nReasoning.\n</think>\nThe answer."

    def test_empty_thought_does_not_leak_tokens(self):
        """An empty thought block."""
        out = _scrub(["<|channel>thought\n<channel|>"])
        assert "<|channel" not in out and "channel|>" not in out

    def test_whitespace_inside_tag_tolerated(self):
        out = _scrub(["<| channel |>thought\nhi<channel|>done"])
        assert "channel" not in out
        assert out == "<think>\nhi\n</think>\ndone"

    def test_extra_channel_names_open_think(self):
        for kind in ("thinking", "reasoning", "reflection"):
            out = _scrub([f"<|channel>{kind}\nx<channel|>y"])
            assert out == "<think>\nx\n</think>\ny", kind

    def test_plain_text_untouched(self):
        assert _scrub(["just a normal reply, no markers."]) == \
            "just a normal reply, no markers."

    def test_idempotent(self):
        text = "<|channel>thought\nr\n<channel|>answer"
        once = scrub_text(text)
        assert scrub_text(once) == once

    def test_marker_straddling_chunks(self):
        pieces = ["safe text <|chan", "nel|>thought", " more text"]
        assert _scrub(pieces) == "safe text <think>\n more text"


class _FakeBackend:
    """Stand-in for a backend (e.g. HF) that does NOT scrub on its own."""

    loaded = True

    def chat_stream(self, messages, **kwargs):
        yield "<|channel>thought\n"
        yield "internal reasoning<channel|>"
        yield "Hello there!"


class TestEngineLayerScrub:
    def test_engine_scrubs_unscrubbing_backend(self):
        """For an HF-style backend with no scrub of its own, the engine layer
        normalises the stream so raw channel tokens never reach the caller."""
        eng = Engine.__new__(Engine)          # skip real model loading
        eng._backend = _FakeBackend()
        eng.display_name = "fake"
        out = "".join(eng.chat_stream([{"role": "user", "content": "hi"}]))
        assert "<|channel" not in out and "channel|>" not in out
        assert out == "<think>\ninternal reasoning\n</think>\nHello there!"

    def test_engine_releases_each_piece_before_the_backend_makes_the_next(self):
        """A reply with no marker characters reaches the caller piece by piece,
        not after the backend has produced dozens more characters (or, for a
        short reply, after the whole generation)."""
        produced = []

        class _Backend:
            loaded = True

            def chat_stream(self, messages, **kwargs):
                for piece in ("The circle", " is the", " largest shape."):
                    produced.append(piece)
                    yield piece

        eng = Engine.__new__(Engine)
        eng._backend = _Backend()
        eng.display_name = "fake"
        seen = [(len(produced), piece)
                for piece in eng.chat_stream([{"role": "user", "content": "hi"}])]
        assert seen == [(1, "The circle"), (2, " is the"), (3, " largest shape.")]


#  Turn-open markers emitted as plain text.
#
#  A turn-open marker carries the role word after it, so a scrubber that removes
#  the marker alone leaves a bare "model" / "assistant" at the head of the reply.
#  The turn-CLOSE counterparts are absent from _MARKER_RE: the backend treats
#  those as stop strings and ends the turn instead of editing the text.

#  Every marker string _MARKER_RE is meant to remove whole, longest first. Used
#  both as the coverage list and as the bound the stream buffer has to clear.
_TURN_MARKERS = [
    "<|start_header_id|>assistant<|end_header_id|>\n",
    "<|start_header_id|>ipython<|end_header_id|>\n",
    "<start_of_turn>assistant\n",
    "<|im_start|>assistant\n",
    "<start_of_turn>model\n",
    "<|im_start|>system\n",
    "<start_of_turn>user\n",
    "<|turn>assistant\n",
    "<|start|>assistant",
    "<|channel|>",
    "<unused7>",
    "[TOOL_CALLS]",
]


class TestTurnOpenMarkers:
    def test_turn_open_marker_takes_its_role_word_with_it(self):
        """The role word is part of the marker and is removed with it, so no
        bare 'model' or 'assistant' is left in the visible reply."""
        for marker in _TURN_MARKERS:
            out = scrub_text(f"{marker}Hello")
            assert out == "Hello", f"{marker!r} left {out!r}"

    def test_role_words_in_ordinary_prose_are_untouched(self):
        """Only the structural marker is removed. A bare role word the model
        writes in prose survives untouched."""
        for prose in ("Model: here is the answer",
                      "the Model card says otherwise",
                      "she asked him about the assistant role",
                      "a system prompt names the user"):
            assert scrub_text(prose) == prose

    def test_near_miss_markers_are_not_stripped(self):
        """The match is anchored on the whole delimiter, so text that merely
        starts the same way survives."""
        for safe in ("<started>", "x<start_of_turnip>", "a < b", "<|imagine|>",
                     "<start_of_turn", "<|start_header_id|>nobody"):
            assert scrub_text(f"keep {safe} keep") == f"keep {safe} keep"

    def test_marker_hold_covers_every_marker_at_every_stream_split(self):
        """_MARKER_HOLD bounds how much text scrub_stream keeps buffered, so it
        has to stay at or above the longest marker _MARKER_RE can match. Driven
        through the real streaming path at every split point."""
        from localm.textnorm import _MARKER_HOLD

        for marker in _TURN_MARKERS:
            assert len(marker) <= _MARKER_HOLD, (
                f"{marker!r} is {len(marker)} chars, longer than the "
                f"_MARKER_HOLD={_MARKER_HOLD} stream buffer")
            text = f"before {marker}after"
            for i in range(len(text) + 1):
                out = _scrub([text[:i], text[i:]])
                assert out == "before after", (
                    f"{marker!r} split at {i} produced {out!r}")

    def test_streaming_and_one_shot_agree_on_turn_open_markers(self):
        for marker in _TURN_MARKERS:
            text = f"a {marker}b"
            assert _scrub([text]) == scrub_text(text)

    def test_turn_open_scrub_is_idempotent(self):
        for marker in _TURN_MARKERS:
            once = scrub_text(f"{marker}reply")
            assert scrub_text(once) == once


_PROSE = "The quick brown fox jumps over the lazy dog, twice over. "

#  Marker strings scrub_text rewrites or removes, beyond _TURN_MARKERS.
_OTHER_MARKERS = [
    "<|channel|>analysis<|message|>",
    "<|channel|>final<|message|>",
    "<|channel>thought\n",
    "<channel|>",
    "<reasoning>",
    "</ reasoning >",
    '<|"|>',
    "<|turn>model\n",
    "<turn|>",
    "<|return|>",
    "<unused12>",
]


def _chunked(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _first_chars(sub):
    """Characters a parsed regex can start matching on."""
    from re import _constants as c

    op, av = sub.data[0]
    if op is c.LITERAL:
        return {chr(av)}
    if op is c.BRANCH:
        return set().union(*(_first_chars(alt) for alt in av[1]))
    if op is c.SUBPATTERN:
        return _first_chars(av[3])
    if op is c.IN and all(o is c.LITERAL for o, _ in av):
        return {chr(v) for _, v in av}
    raise AssertionError(f"cannot tell what {sub!r} starts with ({op})")


class TestStreamRelease:
    """scrub_stream releases text as it arrives and holds back only what could
    still turn out to be a marker."""

    def test_every_scrub_pattern_starts_with_a_marker_start_character(self):
        from re import _parser

        from localm.textnorm import _MARKER_START, _SCRUB_SUBS

        for rx, _ in _SCRUB_SUBS:
            starts = _first_chars(_parser.parse(rx.pattern, rx.flags))
            assert starts <= set(_MARKER_START), (rx.pattern, starts)

    def test_plain_text_is_released_piece_by_piece(self):
        pulled = []

        def source():
            for piece in ("The circle", " is the", " largest shape."):
                pulled.append(piece)
                yield piece

        seen = [(len(pulled), out) for out in scrub_stream(source())]
        assert seen == [(1, "The circle"), (2, " is the"), (3, " largest shape.")]

    def test_a_possible_marker_tail_is_held_until_it_resolves(self):
        out = list(scrub_stream(iter(["Answer: <start_of", "_turn>model\nHi"])))
        assert out == ["Answer: ", "Hi"]

    def test_marker_characters_that_cannot_start_a_marker_are_not_held(self):
        pieces = ("if a < b then", " see [the docs](u)", " or <br> and", " [1]")
        pulled = []

        def source():
            for piece in pieces:
                pulled.append(piece)
                yield piece

        seen = [(len(pulled), out) for out in scrub_stream(source())]
        assert seen == [(i + 1, piece) for i, piece in enumerate(pieces)]

    def test_a_reasoning_reply_streams_its_answer_as_it_arrives(self):
        tokens = ["<think>", "\n", "The", " user", " asks", " for", " the", " capital",
                  ".", "\n", "</think>", "\n\n", "The", " capital", " of", " France",
                  " is", " Paris", "."]
        pulled = []

        def source():
            for token in tokens:
                pulled.append(token)
                yield token

        seen = [(len(pulled), out) for out in scrub_stream(source())]
        assert seen == [(i + 1, token) for i, token in enumerate(tokens)]

    def test_a_tag_is_held_only_while_it_could_become_a_marker(self):
        pulled = []

        def source():
            for ch in "<think>ok":
                pulled.append(ch)
                yield ch

        seen = [(len(pulled), out) for out in scrub_stream(source())]
        assert seen[0] == (7, "<think>")
        assert "".join(out for _, out in seen) == "<think>ok"

    def test_the_cut_backs_up_past_a_complete_marker_that_straddles_it(self, monkeypatch):
        """With a pattern whose match holds another possible marker inside
        it, the cut moves to the start of the outer match."""
        import re
        from re import _parser

        import localm.textnorm as tn

        rx = re.compile(r"<ab<cd>|<cd>XY")
        monkeypatch.setattr(tn, "_SCRUB_RE", rx)
        monkeypatch.setattr(tn, "_SCRUB_TREE",
                            tn._tagged(_parser.parse(rx.pattern, rx.flags), False))
        tn._could_become_marker.cache_clear()
        try:
            assert tn._commit_point("<ab<cd>X") == 0
        finally:
            tn._could_become_marker.cache_clear()

    def test_every_prefix_of_every_match_counts_as_a_possible_marker(self):
        import random

        from localm.textnorm import _SCRUB_RE, _SCRUB_TREE, _prefix_fits

        rng = random.Random(3)
        frags = _TURN_MARKERS + _OTHER_MARKERS + [
            "<|", "<", "[", " ", "  ", "\n", "x", "|", ">", "1234", "model",
            "<| channel |>", "< reasoning>", "</thinking >", "<Thinking>", "</ THOUGHT>",
            "<unused"]
        found = 0
        for _ in range(3000):
            text = "".join(rng.choice(frags) for _ in range(rng.randint(1, 6)))
            for m in _SCRUB_RE.finditer(text):
                found += 1
                whole = m.group(0)
                for end in range(1, len(whole) + 1):
                    assert _prefix_fits(_SCRUB_TREE, whole[:end], 0), (whole, end)
        assert found > 3000

    def test_a_llama3_role_header_streamed_as_tokens_is_removed(self):
        """The header holds a second ``<`` inside it; a cut there would release
        its first half as text."""
        tokens = ["<|", "start", "_header", "_id", "|>", "assistant", "<|", "end",
                  "_header", "_id", "|>", "\n", "Hello", " there"] + [" word"] * 12
        assert _scrub(tokens) == "Hello there" + " word" * 12

    def test_streaming_matches_one_shot_for_every_marker_and_chunking(self):
        """Markers placed before, inside and after the hold window, next to
        prose that contains marker characters, cut into pieces of many sizes."""
        for marker in _TURN_MARKERS + _OTHER_MARKERS:
            for lead in (0, 1, 30, 47, 48, 49, 55, 56, 57, 95):
                text = (_PROSE * 2)[:lead] + marker + "Hello [1] a<b " + _PROSE \
                    + marker + marker + "bye"
                want = scrub_text(text)
                for size in (1, 2, 3, 5, 7, 11, 13, 29, 64, len(text)):
                    got = _scrub(_chunked(text, size))
                    assert got == want, (marker, lead, size, got)

    def test_streaming_matches_one_shot_at_every_two_piece_split(self):
        for marker in _TURN_MARKERS + _OTHER_MARKERS:
            text = _PROSE + marker + "after " + _PROSE
            want = scrub_text(text)
            for i in range(len(text) + 1):
                assert _scrub([text[:i], text[i:]]) == want, (marker, i)

    def test_adjacent_markers_stream_like_one_shot(self):
        """Two markers side by side are each rewritten on their own: a
        turn-open marker's optional trailing newline never takes the newline a
        think-close rewrite puts in front of the next one."""
        want = "\n</think>\n" + "x" * 33
        assert scrub_text("<|turn><channel|>" + "x" * 33) == want
        assert _scrub(["<|turn><channel|>" + "x" * 33]) == want
        for first in _TURN_MARKERS + _OTHER_MARKERS:
            for second in _TURN_MARKERS + _OTHER_MARKERS:
                text = _PROSE + first + second + "The answer. " + _PROSE
                want = scrub_text(text)
                for size in (1, 3, 7, len(text)):
                    got = _scrub(_chunked(text, size))
                    assert got == want, (first, second, size, got)

    def test_marker_hold_covers_the_longest_possible_match(self):
        from re import _parser

        from localm.textnorm import _MARKER_HOLD, _SCRUB_SUBS

        for rx, _ in _SCRUB_SUBS:
            longest = _parser.parse(rx.pattern, rx.flags).getwidth()[1]
            assert longest <= _MARKER_HOLD, (rx.pattern, longest)

    def test_whitespace_padded_tags_stream_like_one_shot(self):
        for text in ("<|" + " " * 4 + "channel" + " " * 4 + "|>analysis "
                     + " " * 60 + "Hello",
                     "<|" + " " * 39 + "channel|>Hello",
                     "<unused" + "7" * 42 + ">Hello",
                     "<" + " " * 39 + "reasoning>Hello"):
            for size in (1, 2, 5):
                assert _scrub(_chunked(text, size)) == scrub_text(text), (text, size)

class TestToolCallsToken:
    """Mistral's ``[TOOL_CALLS]`` token written out as plain text."""

    def test_token_is_removed_from_text(self):
        assert scrub_text("[TOOL_CALLS] The grep found no matches") == " The grep found no matches"
        assert scrub_text("[TOOL_CALLS][TOOL_CALLS]x") == "x"

    def test_other_bracketed_text_survives(self):
        for safe in ("[tool_calls]", "[TOOL_CALL]", "[TOOL_CALLS", "[1] [link](u)"):
            assert scrub_text(f"keep {safe} keep") == f"keep {safe} keep"

    def test_a_flood_is_cut_and_the_source_is_closed(self):
        from localm.textnorm import _MARKER_FLOOD_LIMIT, scrub_stream

        pulled = []

        def source():
            try:
                for i in range(300):
                    pulled.append(i)
                    yield "[TOOL_CALLS]"
            finally:
                pulled.append("closed")

        held = source()
        out = "".join(scrub_stream(held))
        assert out == ""
        assert pulled[-1] == "closed"
        assert len(pulled) < 40, len(pulled)
        assert _MARKER_FLOOD_LIMIT < 40

    def test_text_before_a_flood_is_kept(self):
        text = "Here you go." + "[TOOL_CALLS] " * 300
        out = _scrub([text[i:i + 7] for i in range(0, len(text), 7)])
        assert out.startswith("Here you go.")
        assert "[TOOL_CALLS]" not in out
        assert len(out) < 100

    def test_a_few_markers_are_not_a_flood(self):
        text = "a[TOOL_CALLS]b[TOOL_CALLS]c"
        assert _scrub([text]) == "abc"


class TestThinkExitMarker:
    """With a lazy <tool_call> grammar, llama.cpp matches the trigger inside a
    think block too, and from there the grammar allows only the call and the
    end of generation, so </think> never comes. Given the marker, a think block
    still open at the end hands everything from the marker on to the content;
    a block that does close stays reasoning, marker included."""

    CALL = '<tool_call>{"name": "web_search", "args": {"query": "q"}}</tool_call>'
    MARK = "<tool_call>"

    def _texts(self):
        return [
            "<think>I should search. " + self.CALL,
            "<think>plan " + self.CALL + " no, wrong.</think>Answer.",
            "<think>plan</think>\n" + self.CALL,
            "Pre <think>a " + self.CALL,
            "<think>" + self.MARK[:5],
            "no think at all " + self.CALL,
        ]

    def test_an_open_block_ends_at_the_marker(self):
        c, r = split_think("<think>I should search. " + self.CALL, exit_marker=self.MARK)
        assert (c, r) == (self.CALL, "I should search. ")

    def test_without_the_marker_an_open_block_is_all_reasoning(self):
        c, r = split_think("<think>I should search. " + self.CALL)
        assert c == "" and r.endswith(self.CALL)

    def test_a_block_that_closes_stays_reasoning(self):
        text = "<think>plan " + self.CALL + " no, wrong.</think>Answer."
        assert split_think(text, exit_marker=self.MARK) == split_think(text)
        assert split_think(text, exit_marker=self.MARK)[0] == "Answer."

    def test_a_marker_outside_a_think_block_is_untouched(self):
        text = "<think>plan</think>\n" + self.CALL
        assert split_think(text, exit_marker=self.MARK) == ("\n" + self.CALL, "plan")

    def test_streaming_matches_the_one_shot_split_at_every_cut(self):
        for text in self._texts():
            want = split_think(text, exit_marker=self.MARK)
            for cut in range(len(text) + 1):
                sp = ThinkSplitter(exit_marker=self.MARK)
                parts = [sp.feed(text[:cut]), sp.feed(text[cut:]), sp.flush()]
                got = ("".join(p[0] for p in parts), "".join(p[1] for p in parts))
                assert got == want, (text, cut)

    def test_streaming_one_character_at_a_time(self):
        for text in self._texts():
            sp = ThinkSplitter(exit_marker=self.MARK)
            parts = [sp.feed(ch) for ch in text] + [sp.flush()]
            got = ("".join(p[0] for p in parts), "".join(p[1] for p in parts))
            assert got == split_think(text, exit_marker=self.MARK), text
