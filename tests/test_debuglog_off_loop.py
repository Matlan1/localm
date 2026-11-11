# SPDX-License-Identifier: AGPL-3.0-or-later
"""The debug log's file and console handlers write on the log writer thread, so
a stream that blocks (a disk saturated by a model load) never blocks the thread
that logged, the server's event loop included."""

from __future__ import annotations

import io
import logging
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import debuglog


class _BlockingStream(io.StringIO):
    """A stream whose write blocks until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    def write(self, s: str) -> int:
        self.entered.set()
        self.release.wait(30)
        return super().write(s)


@pytest.fixture
def blocked_handler():
    stream = _BlockingStream()
    handler = debuglog._OffThreadStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    debuglog.logger.addHandler(handler)
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        stream.release.set()
        debuglog._log_writer.drain(10)
        debuglog.logger.removeHandler(handler)
        debuglog.logger.setLevel(old_level)


def test_a_blocked_log_stream_does_not_stall_a_request(blocked_handler):
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        debuglog.logger.debug("request log line")
        return {"ok": True}

    with TestClient(app) as client:
        start = time.monotonic()
        first = client.get("/ping")
        assert blocked_handler.entered.wait(5), "the writer never reached the stream"
        second = client.get("/ping")
        elapsed = time.monotonic() - start
    assert first.status_code == 200 and second.status_code == 200
    assert elapsed < 3, f"requests took {elapsed:.1f}s while the log stream was blocked"
    assert blocked_handler.getvalue() == ""
    blocked_handler.release.set()
    assert debuglog._log_writer.drain(10)
    assert blocked_handler.getvalue().count("request log line") == 2


def test_records_are_written_in_order_after_drain():
    stream = io.StringIO()
    handler = debuglog._OffThreadStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    debuglog.logger.addHandler(handler)
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)
    try:
        for i in range(200):
            debuglog.logger.debug("line %d", i)
        debuglog.flush_log_handlers()
    finally:
        debuglog.logger.removeHandler(handler)
        debuglog.logger.setLevel(old_level)
    assert stream.getvalue().splitlines() == [f"line {i}" for i in range(200)]


def test_a_failing_stream_is_reported_not_raised(monkeypatch):
    class _Broken(io.StringIO):
        def write(self, s):
            raise OSError("disk gone")

    reported = []
    handler = debuglog._OffThreadStreamHandler(_Broken())
    monkeypatch.setattr(handler, "handleError", lambda record: reported.append(record))
    debuglog.logger.addHandler(handler)
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)
    try:
        debuglog.logger.debug("lost line")
        assert debuglog._log_writer.drain(10)
    finally:
        debuglog.logger.removeHandler(handler)
        debuglog.logger.setLevel(old_level)
    assert [r.getMessage() for r in reported] == ["lost line"]
