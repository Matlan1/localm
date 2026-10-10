# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON a model, a peer or a stream controls: over-nested and over-long-integer
documents reach the same fallback as any other unparseable text, not an
uncaught ``RecursionError`` / digit-limit ``ValueError``."""

from __future__ import annotations

import json
import sys

import pytest

from localm.inference import ollama_protocol as P
from localm.inference import tool_calling as TC
from localm.inference.http_engine import HttpEngine
from localm.memory import consolidate
from localm.plugins.coder import episodes, parser, reviewer
from localm.plugins.coder.backends.http import HTTPBackend
from localm.plugins.coder.mcp import MCPServer
from localm.plugins.coder.tools.files import _verify_syntax
from tests._hostile_json import (
    BIG_INT, BIG_INT_LIST, DEEP, DEEP_CLOSED, HOSTILE, serve)

EXTRACT_DOCS = pytest.mark.parametrize("doc", [
    DEEP, BIG_INT, "{" + DEEP + "}", '{"a": ' + BIG_INT + "}",
    "```json\n" + DEEP + "\n```"],
    ids=["deep", "bigint", "braced-deep", "braced-bigint", "fenced-deep"])


# ------------------------------------------------------------ tool_calling

@HOSTILE
def test_call_text_keeps_hostile_string_arguments_as_input(doc):
    out = TC._call_text({"function": {"name": "f", "arguments": doc}})
    payload = json.loads(out.split("\n")[1])
    assert payload == {"name": "f", "arguments": {"input": doc}}


@HOSTILE
def test_as_call_refuses_hostile_string_arguments(doc):
    assert TC._as_call({"name": "f", "arguments": doc}, None) is None


@HOSTILE
def test_stream_returns_a_hostile_tagged_body_as_text(doc):
    stream = TC.ToolCallStream()
    events = stream.feed(TC.OPEN_TAG + doc + TC.CLOSE_TAG)
    events += stream.finish()
    assert [kind for kind, _ in events] == ["text"]
    assert doc in events[0][1]


@pytest.mark.parametrize("doc", [DEEP_CLOSED, BIG_INT_LIST], ids=["deep", "bigint"])
def test_stream_returns_a_hostile_bare_value_as_text(doc):
    stream = TC.ToolCallStream()
    events = stream.feed(doc) + stream.finish()
    assert events and all(kind == "text" for kind, _ in events)
    assert "".join(text for _, text in events) == doc


# --------------------------------------------------------- ollama_protocol

@HOSTILE
def test_tool_calls_to_ollama_replaces_hostile_arguments_with_empty(doc):
    out = P.tool_calls_to_ollama(
        [{"id": "c1", "function": {"name": "f", "arguments": doc}}])
    assert out[0]["function"]["arguments"] == {}
    assert out[0]["function"]["name"] == "f"


@HOSTILE
def test_sse_line_json_ignores_a_hostile_payload(doc):
    assert P._sse_line_json("data: " + doc) is None


# ------------------------------------------------------------- http_engine

@pytest.mark.parametrize("junk", [DEEP, BIG_INT, "[1, 2]"],
                         ids=["deep", "bigint", "not-an-object"])
def test_chat_stream_skips_a_hostile_chunk_and_keeps_streaming(junk):
    good = json.dumps({"choices": [{"delta": {"content": "ok"}}]})
    body = f"data: {junk}\n\ndata: {good}\n\ndata: [DONE]\n\n"
    with serve(body, content_type="text/event-stream") as base:
        out = list(HttpEngine(base + "/v1").chat_stream(
            [{"role": "user", "content": "hi"}]))
    assert out == ["ok"]


# ---------------------------------------------------------- coder backends

@HOSTILE
def test_routing_note_header_that_is_hostile_json_clears_the_note(doc):
    backend = HTTPBackend("http://127.0.0.1:1/v1", "m")
    backend.routing_note = "stale"
    backend._note_routing({"X-Localm-Model-Routing": doc})
    assert backend.routing_note is None


@HOSTILE
def test_lenient_json_gives_up_on_a_hostile_body(doc):
    assert parser._lenient_json(doc) is None


@HOSTILE
def test_exact_call_object_refuses_a_hostile_body(doc):
    assert parser._exact_call_object(doc) is None


@EXTRACT_DOCS
def test_reviewer_extract_json_returns_empty(doc):
    assert reviewer._extract_json(doc) == {}


@EXTRACT_DOCS
def test_episodes_extract_json_returns_empty(doc):
    assert episodes._extract_json(doc) == {}


@EXTRACT_DOCS
def test_consolidate_parse_json_object_returns_empty(doc):
    assert consolidate._parse_json_object(doc) == {}


_MCP_SERVER = r"""
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    sys.stdout.write(("[" * 100000 if sys.argv[1] == "deep" else "9" * 5000) + "\n")
    sys.stdout.write("[1, 2]\n")
    result = {"tools": []} if msg["method"] == "tools/list" else {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


@pytest.mark.parametrize("kind", ["deep", "bigint"])
def test_mcp_client_reader_survives_hostile_lines_from_the_server(kind, monkeypatch):
    monkeypatch.setattr("localm.plugins.coder.mcp._INIT_TIMEOUT", 8)
    server = MCPServer("hostile", sys.executable, ["-c", _MCP_SERVER, kind])
    try:
        server.start()
        assert server.tools == []
        assert server.alive
    finally:
        server.stop()


def test_json_write_check_still_flags_a_real_syntax_error(tmp_path):
    assert "JSON syntax error" in _verify_syntax(tmp_path / "x.json", "{not json")


def test_json_write_check_accepts_extreme_but_well_formed_json(tmp_path):
    assert _verify_syntax(tmp_path / "x.json", DEEP_CLOSED) is None
    assert _verify_syntax(tmp_path / "x.json", BIG_INT_LIST) is None
