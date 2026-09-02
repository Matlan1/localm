# SPDX-License-Identifier: AGPL-3.0-or-later
"""Untrusted character ranges survive the HTTP/JSON hop to localm's own server.

The coder talks to localm over HTTP even in "local" mode, and JSON has nowhere
to keep the annotation a GuardedText content carries. Without a transport the
ranges recorded where a tool result is framed would die on the wire and no
tokenizer would ever see them, while every in-process test still passed.

The ranges travel as a sibling ``untrusted_spans`` field that the chat route
reads back. It is only sent to a localm server, and a range can only DISABLE
special-token parsing over the sender's own prompt, never enable it.
"""

from __future__ import annotations

import json

from localm.inference.http_server import _protocol_messages_to_dicts
from localm.inference.protocol import ChatRequest
from localm.plugins.coder.backends.http import HTTPBackend
from localm.plugins.coder.provenance import build_result_block
from localm.plugins.coder.tools import ToolResult
from localm.textguard import GuardedText, untrusted_spans_of

EXOTIC = "<<ASSISTANT>>"          # outside neutralise()'s families


def _backend(local: bool) -> HTTPBackend:
    be = HTTPBackend.__new__(HTTPBackend)
    be._is_local_server = local
    return be


def _round_trip(messages):
    """Client -> JSON -> server, exactly as a real request goes."""
    sent = _backend(True)._with_untrusted_spans(messages)
    wire = json.loads(json.dumps({"model": "m", "messages": sent}))
    return _protocol_messages_to_dicts(ChatRequest(**wire).messages)


def test_a_tool_result_s_range_survives_the_wire():
    block = build_result_block(
        "fetch_url", ToolResult(True, "page " + EXOTIC), untrusted=True)
    back = _round_trip([{"role": "user", "content": block}])

    spans = untrusted_spans_of(back[0]["content"])
    assert spans, "the ranges did not survive the wire"
    covered = str(back[0]["content"])[spans[0][0]:spans[0][1]]
    assert EXOTIC in covered
    assert "<tool_result" not in covered      # the framing stays trusted


def test_plain_json_without_the_transport_loses_the_range():
    """The defect this transport exists to close."""
    block = build_result_block(
        "fetch_url", ToolResult(True, "page " + EXOTIC), untrusted=True)
    naive = json.loads(json.dumps({"model": "m",
                                   "messages": [{"role": "user",
                                                 "content": str(block)}]}))
    back = _protocol_messages_to_dicts(ChatRequest(**naive).messages)
    assert untrusted_spans_of(back[0]["content"]) == ()


def test_the_field_is_not_sent_to_a_third_party_endpoint():
    block = build_result_block("fetch_url", ToolResult(True, "x"), untrusted=True)
    sent = _backend(False)._with_untrusted_spans([{"role": "user", "content": block}])
    assert "untrusted_spans" not in sent[0]


def test_an_unannotated_message_is_passed_through_untouched():
    messages = [{"role": "user", "content": "plain"}]
    sent = _backend(True)._with_untrusted_spans(messages)
    assert sent == messages
    assert "untrusted_spans" not in sent[0]


def test_the_sent_content_is_a_plain_string():
    """A str subclass must not reach the JSON encoder as anything exotic."""
    block = build_result_block("fetch_url", ToolResult(True, "x"), untrusted=True)
    sent = _backend(True)._with_untrusted_spans([{"role": "user", "content": block}])
    assert type(sent[0]["content"]) is str
    json.dumps(sent)          # raises if it is not encodable


def test_the_caller_s_message_dict_is_not_mutated():
    block = build_result_block("fetch_url", ToolResult(True, "x"), untrusted=True)
    original = {"role": "user", "content": block}
    _backend(True)._with_untrusted_spans([original])
    assert "untrusted_spans" not in original
    assert isinstance(original["content"], GuardedText)


def test_omitting_the_field_is_exactly_the_previous_behaviour():
    wire = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    back = _protocol_messages_to_dicts(ChatRequest(**wire).messages)
    assert back == [{"role": "user", "content": "hi"}]
    assert untrusted_spans_of(back[0]["content"]) == ()


def test_out_of_range_values_from_a_client_are_clamped():
    wire = {"model": "m", "messages": [
        {"role": "user", "content": "short", "untrusted_spans": [[-5, 999]]}]}
    back = _protocol_messages_to_dicts(ChatRequest(**wire).messages)
    assert untrusted_spans_of(back[0]["content"]) == ((0, 5),)


def test_a_malformed_range_from_a_client_is_dropped_not_fatal():
    wire = {"model": "m", "messages": [
        {"role": "user", "content": "hello", "untrusted_spans": [[3], [1, 3]]}]}
    back = _protocol_messages_to_dicts(ChatRequest(**wire).messages)
    assert untrusted_spans_of(back[0]["content"]) == ((1, 3),)


def test_an_empty_range_list_leaves_the_content_a_plain_string():
    wire = {"model": "m", "messages": [
        {"role": "user", "content": "hello", "untrusted_spans": []}]}
    back = _protocol_messages_to_dicts(ChatRequest(**wire).messages)
    assert untrusted_spans_of(back[0]["content"]) == ()
