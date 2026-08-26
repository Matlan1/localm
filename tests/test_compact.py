# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for conversation compaction (localm.inference.compact)."""

from localm.inference.compact import (
    KEEP_RECENT,
    compact_messages,
    estimate_tokens,
    maybe_compact,
)


def _history(n_turns: int, chars: int = 400) -> list:
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * chars})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * chars})
    return msgs


def _summariser(messages, max_tokens):
    return "A concise summary of the earlier conversation."


class TestEstimateTokens:
    def test_plain_text(self):
        msgs = [{"role": "user", "content": "x" * 400}]
        assert estimate_tokens(msgs) == 100

    def test_uses_real_counter_when_given(self):
        msgs = [{"role": "user", "content": "hello world"}]
        assert estimate_tokens(msgs, count_tokens=lambda t: 42) == 42

    def test_images_add_flat_cost(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]}]
        assert estimate_tokens(msgs) >= 750

    def test_broken_counter_falls_back(self):
        def boom(_):
            raise RuntimeError("tokenizer gone")
        msgs = [{"role": "user", "content": "x" * 80}]
        assert estimate_tokens(msgs, count_tokens=boom) == 20


class TestCompactMessages:
    def test_short_history_untouched(self):
        msgs = _history(2)   # 4 messages == KEEP_RECENT
        out, changed = compact_messages(msgs, _summariser)
        assert changed is False
        assert out == msgs

    def test_older_turns_replaced_by_summary(self):
        msgs = _history(6)   # 12 messages
        out, changed = compact_messages(msgs, _summariser)
        assert changed is True
        # bridge pair + last KEEP_RECENT verbatim
        assert len(out) == 2 + KEEP_RECENT
        assert "[Conversation summary]" in out[0]["content"]
        assert "concise summary" in out[0]["content"]
        assert out[-KEEP_RECENT:] == msgs[-KEEP_RECENT:]

    def test_system_prompt_preserved_first(self):
        msgs = [{"role": "system", "content": "be terse"}, *_history(6)]
        out, changed = compact_messages(msgs, _summariser)
        assert changed is True
        assert out[0] == {"role": "system", "content": "be terse"}
        assert "[Conversation summary]" in out[1]["content"]

    def test_summariser_failure_falls_back_to_trim(self):
        def broken(messages, max_tokens):
            raise RuntimeError("model died")
        msgs = _history(6)
        out, changed = compact_messages(msgs, broken)
        assert changed is True
        assert "removed to fit the context window" in out[0]["content"]
        assert out[-KEEP_RECENT:] == msgs[-KEEP_RECENT:]

    def test_empty_summary_falls_back_to_trim(self):
        out, changed = compact_messages(_history(6), lambda m, t: "   ")
        assert changed is True
        assert "removed to fit" in out[0]["content"]

    def test_never_raises(self):
        # A summariser that returns None instead of a string.
        out, changed = compact_messages(_history(6), lambda m, t: None)
        assert changed is True


class TestMaybeCompact:
    def test_below_threshold_untouched(self):
        msgs = _history(3, chars=100)
        out, compacted = maybe_compact(
            msgs, limit_tokens=16384, generate=_summariser)
        assert compacted is False
        assert out == msgs

    def test_above_threshold_compacts(self):
        # ~8 turns * 2 * 400 chars / 4 ≈ 1600 tokens; limit 2000 → ratio hit
        msgs = _history(8)
        out, compacted = maybe_compact(
            msgs, limit_tokens=2000, generate=_summariser)
        assert compacted is True
        assert len(out) < len(msgs)

    def test_zero_limit_disables(self):
        msgs = _history(50)
        out, compacted = maybe_compact(
            msgs, limit_tokens=0, generate=_summariser)
        assert compacted is False
