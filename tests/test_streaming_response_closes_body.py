# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm's StreamingResponse closes its body iterator whenever streaming
stops, and every route returns it."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from localm.streaming_response import StreamingResponse

_REPO = Path(__file__).resolve().parents[1]


class _SendGone(Exception):
    pass


def _body(events):
    async def gen():
        try:
            for i in range(100):
                yield f"data: {i}\n\n"
        finally:
            events.append("closed")
    return gen()


def test_a_send_that_raises_closes_the_body_where_it_waits():
    events = []
    sends = []

    async def send(message):
        sends.append(message)
        if message["type"] == "http.response.body" and len(sends) >= 3:
            raise _SendGone()

    async def scenario():
        resp = StreamingResponse(_body(events), media_type="text/event-stream")
        with pytest.raises(_SendGone):
            await resp.stream_response(send)
        assert events == ["closed"]

    asyncio.run(scenario())


def test_a_cancel_inside_send_closes_the_body_where_it_waits():
    events = []
    entered = None

    async def send(message):
        if message["type"] == "http.response.body":
            entered.set()
            await asyncio.Event().wait()

    async def scenario():
        nonlocal entered
        entered = asyncio.Event()
        resp = StreamingResponse(_body(events), media_type="text/event-stream")
        task = asyncio.ensure_future(resp.stream_response(send))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["closed"]

    asyncio.run(scenario())


def test_a_finished_stream_sends_everything_and_closes_once():
    events = []
    bodies = []

    async def send(message):
        if message["type"] == "http.response.body":
            bodies.append(message.get("body", b""))

    async def scenario():
        resp = StreamingResponse(_body(events), media_type="text/event-stream")
        await resp.stream_response(send)

    asyncio.run(scenario())
    assert events == ["closed"]
    assert bodies[0] == b"data: 0\n\n"
    assert bodies[99] == b"data: 99\n\n"
    assert bodies[-1] == b""


def test_a_body_that_fails_to_close_is_logged_not_raised(caplog):
    class _Body:
        def __init__(self):
            self.items = iter(["a", "b"])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self):
            raise RuntimeError("close failed")

    async def send(message):
        pass

    async def scenario():
        await StreamingResponse(_Body()).stream_response(send)

    with caplog.at_level("WARNING", logger="localm"):
        asyncio.run(scenario())
    assert any("closing a streaming response's body failed" in r.getMessage()
               for r in caplog.records)


def _starlette_streaming_uses(source: str) -> list[int]:
    """The line of each place in *source* that takes ``StreamingResponse`` from
    fastapi or starlette rather than from ``localm.streaming_response``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.split(".")[0] in ("fastapi", "starlette")):
            if any(alias.name == "StreamingResponse" for alias in node.names):
                found.append(node.lineno)
        if (isinstance(node, ast.Attribute) and node.attr == "StreamingResponse"
                and isinstance(node.value, (ast.Name, ast.Attribute))):
            base = ast.unparse(node.value)
            if base.split(".")[0] in ("fastapi", "starlette", "responses"):
                found.append(node.lineno)
    return found


def test_every_localm_module_uses_the_closing_streaming_response():
    own = _REPO / "localm" / "streaming_response.py"
    offenders = []
    files = sorted((_REPO / "localm").rglob("*.py"))
    assert len(files) > 100
    for path in files:
        if path == own:
            continue
        for line in _starlette_streaming_uses(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_REPO)}:{line}")
    assert offenders == []


def test_the_sentinel_sees_a_starlette_streaming_response():
    source = ("from fastapi.responses import JSONResponse, StreamingResponse\n"
              "import starlette.responses\n"
              "x = starlette.responses.StreamingResponse\n"
              "from localm.streaming_response import StreamingResponse\n")
    assert sorted(_starlette_streaming_uses(source)) == [1, 3]
