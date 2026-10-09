# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the MCP stdio server's JSON-RPC handling.

Every line an MCP client writes is attacker-shaped input to a long-running
loop. The contract: whatever the line holds, the loop survives, answers
well-formed JSON-RPC, and keeps serving the next line."""
from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given, strategies as st  # noqa: E402

from localm.plugins.mcpserver.server import MCPStdioServer  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402

_leaf = (st.none() | st.booleans() | st.integers(-(2 ** 70), 2 ** 70) | st.floats()
         | st.text(max_size=10))
_json = st.recursive(
    _leaf,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=8), inner,
                                                                  max_size=3),
    max_leaves=12)


def _server() -> MCPStdioServer:
    def boom(_args):
        raise RuntimeError("tool failure")

    return MCPStdioServer({
        "echo": {"description": "echo", "inputSchema": {"type": "object"},
                 "handler": lambda args: {"content": [{"type": "text", "text": str(args)}],
                                          "isError": False}},
        "boom": {"description": "boom", "inputSchema": {"type": "object"}, "handler": boom},
    })


_messages = st.one_of(
    _json,
    st.fixed_dictionaries({
        "jsonrpc": st.just("2.0"),
        "id": _json,
        "method": st.sampled_from(["initialize", "ping", "tools/list", "tools/call",
                                   "tools/call", "tools/call", "bogus"]) | _json,
        "params": st.fixed_dictionaries(
            {"name": st.sampled_from(["echo", "boom", "nope"]) | _json,
             "arguments": _json, "_meta": _json}, optional={}) | _json,
    }),
)


@given(msg=_messages)
@example(msg={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]})
@example(msg={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ["echo"]}})
def test_handle_answers_with_a_json_rpc_object_or_nothing(msg):
    out = _bounds.returns_within(_server().handle, msg)
    assert out is None or (isinstance(out, dict) and out.get("jsonrpc") == "2.0")
    if out is not None:
        json.dumps(out)


_lines = st.one_of(
    _messages.map(lambda m: json.dumps(m, allow_nan=True)),
    st.text(max_size=60),
    st.sampled_from(["9" * 5000, "[" * 100_000, '{"id": ' + "9" * 5000 + "}", "\x00", "{",
                     '{"id": 1, "method": "ping"}']),
)


@given(lines=st.lists(_lines, max_size=6))
def test_run_stdio_survives_any_input_and_keeps_serving(lines):
    server = _server()
    out = io.StringIO()
    stdin = io.StringIO("\n".join(lines) + '\n{"jsonrpc": "2.0", "id": "last", "method": "ping"}\n')
    _bounds.returns_within(server.run_stdio, stdin, out)
    replies = [json.loads(line) for line in out.getvalue().split(chr(10)) if line.strip()]
    assert replies and replies[-1] == {"jsonrpc": "2.0", "id": "last", "result": {}}
