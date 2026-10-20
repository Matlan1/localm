# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server's compaction is reused across requests: a client that resends
the whole conversation every turn gets the same bridge back with no new
summariser call while the compacted conversation still fits, and only the
newly old turns are summarised once it no longer does."""

from __future__ import annotations

import asyncio

import pytest

import localm.inference.http_server as hs
from localm.inference import compact


class _Engine:
    """A *capacity*-token engine counting one token per three characters,
    whose summariser replies with a numbered summary naming every FACT-n
    marker its prompt carries."""

    def __init__(self, capacity=8192, name="reuse-model"):
        self.capacity = capacity
        self.display_name = name
        self.prompts: list[str] = []
        self.fail = False

    def context_capacity(self):
        return self.capacity

    def count_tokens(self, text):
        return len(str(text)) // 3

    def count_messages_tokens(self, messages):
        return sum(len(str(m.get("content", ""))) // 3 + 4 for m in messages)

    def chat_stream(self, messages, **kw):
        prompt = str(messages[0]["content"])
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("summariser down")
        facts = sorted({w.strip(".,") for w in prompt.split() if w.startswith("FACT-")})
        yield f"Summary {len(self.prompts)}: " + " ".join(facts)


def _turn(i: int, chars: int = 900) -> list[dict]:
    text = f"FACT-{i} "
    while len(text) < chars:
        text += f"word{i} "
    return [{"role": "user", "content": f"question {i}: " + text},
            {"role": "assistant", "content": f"answer {i}: " + text}]


def _conversation(turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(turns):
        msgs.extend(_turn(i))
    return msgs


def _compact(engine, messages):
    out, changed, gone = asyncio.run(hs._compact_for_capacity(engine, messages))
    assert gone is False
    return out, changed


def _bridge_text(out):
    return str(out[1]["content"])


def test_a_resent_conversation_reuses_the_bridge_without_a_summariser_call():
    eng = _Engine()
    first, changed = _compact(eng, _conversation(14))
    assert changed is True
    calls = len(eng.prompts)
    assert calls >= 1

    grown = _conversation(15)
    second, changed = _compact(eng, grown)
    assert changed is True
    assert len(eng.prompts) == calls
    assert _bridge_text(second) == _bridge_text(first)
    assert second[-2:] == grown[-2:]


def test_the_reused_conversation_is_the_resent_tail_after_the_same_bridge():
    eng = _Engine()
    first, _ = _compact(eng, _conversation(14))
    second, _ = _compact(eng, _conversation(15))
    assert second[:len(first)] == first


def test_only_the_newly_old_turns_are_summarised_once_the_reuse_no_longer_fits():
    eng = _Engine()
    _compact(eng, _conversation(14))
    calls = len(eng.prompts)
    out, changed = _compact(eng, _conversation(30))
    assert changed is True
    new_prompts = eng.prompts[calls:]
    assert new_prompts
    assert "Summary of the earlier part:\nSummary " in new_prompts[0]
    assert "question 0:" not in "".join(new_prompts)
    assert "FACT-0" in _bridge_text(out)
    assert not hs._needs_compaction(eng.capacity, eng.count_messages_tokens(out), out)


def test_compaction_aims_for_half_the_window_not_the_last_four_messages():
    eng = _Engine()
    out, _ = _compact(eng, _conversation(14))
    kept = [m for m in out[3:]]
    assert len(kept) > compact.KEEP_RECENT
    assert eng.count_messages_tokens(out) <= eng.capacity * compact.COMPACT_TARGET


def test_another_engine_does_not_reuse_the_bridge():
    a, b = _Engine(), _Engine()
    _compact(a, _conversation(14))
    _compact(b, _conversation(15))
    assert b.prompts


def test_a_changed_early_message_is_summarised_again():
    eng = _Engine()
    _compact(eng, _conversation(14))
    calls = len(eng.prompts)
    edited = _conversation(15)
    edited[1] = {"role": "user", "content": "question 0: FACT-99 edited"}
    out, _ = _compact(eng, edited)
    assert len(eng.prompts) > calls
    assert "FACT-99" in _bridge_text(out)


def test_a_digest_fallback_is_not_reused():
    eng = _Engine()
    eng.fail = True
    first, changed = _compact(eng, _conversation(14))
    assert changed is True
    assert "condensed to fit the context window" in _bridge_text(first)
    eng.fail = False
    calls = len(eng.prompts)
    second, _ = _compact(eng, _conversation(15))
    assert len(eng.prompts) > calls
    assert _bridge_text(second).startswith("[Conversation summary]\nSummary")


def test_a_failed_continuation_keeps_the_earlier_summary_and_digests_the_rest():
    eng = _Engine()
    _compact(eng, _conversation(14))
    eng.fail = True
    out, _ = _compact(eng, _conversation(30))
    text = _bridge_text(out)
    assert text.startswith("[Conversation summary]\nSummary 1:")
    assert "condensed to fit the context window" in text


def test_prefix_digests_match_exactly_on_equal_prefixes():
    a = _conversation(3)
    b = _conversation(3)
    b[-1] = {"role": "assistant", "content": "different"}
    da, db = compact.prefix_digests(a), compact.prefix_digests(b)
    assert len(da) == len(a) + 1
    assert da[:-1] == db[:-1]
    assert da[-1] != db[-1]
    assert compact.prefix_digests([{"role": "user", "content": "x"}])[1] != \
        compact.prefix_digests([{"role": "assistant", "content": "x"}])[1]


def test_the_cache_drops_the_least_recently_used_entry():
    cache = compact.CompactionCache(max_entries=2)
    cache.put("m", "a", "A")
    cache.put("m", "b", "B")
    assert cache.get("m", "a") == "A"
    cache.put("m", "c", "C")
    assert cache.get("m", "b") is None
    assert cache.get("m", "a") == "A" and cache.get("m", "c") == "C"
    assert cache.get("other", "a") is None
    assert len(cache) == 2


@pytest.mark.parametrize("capacity,reserve", [(4096, 2048), (8192, 2048),
                                              (32768, 3276), (0, 0)])
def test_reply_reserve(capacity, reserve):
    assert compact.reply_reserve(capacity) == reserve
