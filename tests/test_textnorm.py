"""
Tests for the shared control-marker scrubber and its application at the engine
layer, so channel/harmony tokens never leak to the GUI regardless of backend.
"""

from localm.inference.engine import Engine
from localm.inference.textnorm import scrub_stream, scrub_text


def _scrub(pieces):
    return "".join(scrub_stream(iter(pieces)))


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
