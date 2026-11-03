# SPDX-License-Identifier: AGPL-3.0-or-later
"""Raw gpt-oss (harmony) replies of a multi-turn chat go through the real
reply pipeline: the GGUF text filters, the engine's scrub and the server's
reply router.

tests/fixtures/harmony_raw_streams.json holds the raw generated text of replies
captured from gpt-oss-20b (MXFP4) for the prompt that
``_chat_formats.format_harmony`` renders: the first turn, the second turn (with
and without a tool offered) and a reply that calls a tool. Every reply must put
its reasoning in the reasoning field and leave no channel marker in the
answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from localm.inference.backends.llamacpp.llama import (
    _STOP_STRINGS, _filtered_stream, _scrub_stream)
from localm.inference.http_server import _ReplyRouter
from localm.textnorm import scrub_stream, scrub_text

_RAW = json.loads((Path(__file__).parent / "fixtures" / "harmony_raw_streams.json")
                  .read_text(encoding="utf-8"))
_STOPS = _STOP_STRINGS - {"<|end|>"}
_MARKER = re.compile(r"<\|[a-z_]+\|>|<\|?channel|</?think>")
_ANSWERS = ["first_turn", "first_turn_tools_offered", "second_turn", "second_turn_tools_offered"]


def _pieces(raw, size):
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def _route(raw, size, tools=None):
    router = _ReplyRouter(None, tools, {})
    events = []
    for piece in scrub_stream(_scrub_stream(_filtered_stream(iter(_pieces(raw, size)), _STOPS))):
        events += router.feed(piece)
    events += router.flush()
    return events


def _joined(events, kind):
    return "".join(v for k, v in events if k == kind)


def _expected_split(raw):
    head, answer = raw.split("<|end|><|start|>assistant<|channel|>final<|message|>")
    return head.removeprefix("<|channel|>analysis<|message|>"), answer


@pytest.mark.parametrize("name", _ANSWERS)
@pytest.mark.parametrize("size", [1, 5, 64, 100000])
@pytest.mark.parametrize("tools", [None, {"web_search"}])
def test_reasoning_goes_to_the_reasoning_field_and_the_answer_is_clean(name, size, tools):
    reasoning, answer = _expected_split(_RAW[name])
    events = _route(_RAW[name], size, tools)
    assert _joined(events, "reasoning").strip() == reasoning.strip()
    assert _joined(events, "text").strip() == answer.strip()
    assert not _MARKER.search(_joined(events, "text"))
    assert not [v for k, v in events if k == "call"]


@pytest.mark.parametrize("name", _ANSWERS)
def test_one_shot_scrub_is_one_think_block_then_the_answer(name):
    reasoning, answer = _expected_split(_RAW[name])
    assert scrub_text(_RAW[name]) == f"<think>\n{reasoning}\n</think>\n{answer}"


@pytest.mark.parametrize("size", [1, 7, 100000])
def test_a_tool_call_reply_routes_the_reasoning_and_the_call_apart(size):
    events = _route(_RAW["tool_called"], size, {"web_search"})
    assert _joined(events, "reasoning").strip() == "We need to use web_search."
    assert not _joined(events, "text").strip()
    [call] = [v for k, v in events if k == "call"]
    assert (call.name, call.arguments) == ("web_search", {"query": "current weather in Paris"})


def test_a_tool_call_reply_not_offered_a_tool_leaves_the_reasoning_apart():
    events = _route(_RAW["tool_called"], 9, None)
    assert _joined(events, "reasoning").strip() == "We need to use web_search."
    assert not [v for k, v in events if k == "call"]
