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
        """The exact shape from chat_and_queue.png: an empty thought block."""
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
        """The leak was an HF-style backend with no scrub; the engine layer must
        normalise it so raw channel tokens never reach the caller/GUI."""
        eng = Engine.__new__(Engine)          # skip real model loading
        eng._backend = _FakeBackend()
        eng.display_name = "fake"
        out = "".join(eng.chat_stream([{"role": "user", "content": "hi"}]))
        assert "<|channel" not in out and "channel|>" not in out
        assert out == "<think>\ninternal reasoning\n</think>\nHello there!"
