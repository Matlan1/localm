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


def test_a_closed_file_handler_reopens_its_file(tmp_path):
    """logging.config.dictConfig (uvicorn configures its loggers with it) closes
    every existing handler; the next record must still reach the file, as with
    the stdlib FileHandler."""
    import logging.config
    path = tmp_path / "debug.log"
    handler = debuglog._OffThreadFileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    debuglog.logger.addHandler(handler)
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)
    try:
        debuglog.logger.debug("before")
        assert debuglog._log_writer.drain(10)
        logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})
        assert handler.stream is None
        debuglog.logger.debug("after")
        assert debuglog._log_writer.drain(10)
    finally:
        debuglog.logger.removeHandler(handler)
        debuglog.logger.setLevel(old_level)
        handler.close()
    assert path.read_text(encoding="utf-8").splitlines() == ["before", "after"]


def _off_thread(stream) -> debuglog._OffThreadStreamHandler:
    handler = debuglog._OffThreadStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


@pytest.fixture
def debug_logger():
    added = []
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)

    def add(handler):
        debuglog.logger.addHandler(handler)
        added.append(handler)
        return handler
    try:
        yield add
    finally:
        for h in added:
            stream = getattr(h, "stream", None)
            if isinstance(stream, _BlockingStream):
                stream.release.set()
            h.writer().drain(10)
            debuglog.logger.removeHandler(h)
        debuglog.logger.setLevel(old_level)


def test_a_blocked_console_does_not_hold_back_the_file(debug_logger):
    console = _BlockingStream()
    debug_logger(_off_thread(console))
    file_stream = io.StringIO()
    file_handler = debug_logger(_off_thread(file_stream))
    debuglog.logger.debug("first")
    assert console.entered.wait(5)
    debuglog.logger.debug("second")
    assert file_handler.writer().drain(5)
    assert file_stream.getvalue().splitlines() == ["first", "second"]
    assert console.getvalue() == ""


def test_flush_writes_what_was_queued(debug_logger):
    class _Slow(io.StringIO):
        def write(self, s):
            time.sleep(0.05)
            return super().write(s)
    stream = _Slow()
    handler = debug_logger(_off_thread(stream))
    for i in range(5):
        debuglog.logger.debug("line %d", i)
    handler.flush()
    assert stream.getvalue().splitlines() == [f"line {i}" for i in range(5)]


def test_a_handler_whose_records_cannot_be_written_is_named_on_the_others(debug_logger):
    console = _BlockingStream()
    debug_logger(_off_thread(console))
    file_stream = io.StringIO()
    debug_logger(_off_thread(file_stream))
    debuglog.logger.debug("first")
    assert console.entered.wait(5)
    debuglog.logger.debug("second")
    start = time.monotonic()
    debuglog.flush_log_handlers(timeout=0.5)
    assert time.monotonic() - start < 2, "a stuck handler was waited on twice"
    lines = file_stream.getvalue().splitlines()
    assert lines[:2] == ["first", "second"]
    assert len(lines) == 3 and "the console" in lines[2] and "not written" in lines[2]


def test_a_failure_to_queue_a_record_is_reported_not_raised(debug_logger, monkeypatch):
    handler = debug_logger(_off_thread(io.StringIO()))
    reported = []
    monkeypatch.setattr(handler, "handleError", lambda record: reported.append(record))

    def broken():
        raise RuntimeError("cannot start a thread")
    monkeypatch.setattr(handler, "writer", broken)
    debuglog.logger.debug("unqueued")
    assert [r.getMessage() for r in reported] == ["unqueued"]


_CHILD_EXITS = r"""
import os
from localm import debuglog
debuglog.attach_child_logging()
print("TYPE " + ",".join(sorted(type(h).__name__ for h in debuglog.logger.handlers)))
debuglog.logger.critical("child-line-before-exit")
os._exit(1)
"""

_INHERITED_SERVER = r"""
from localm import debuglog
debuglog.enable_debug()
print("TYPE " + ",".join(sorted(type(h).__name__ for h in debuglog.logger.handlers)))
"""


def _run_with_log(code: str, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    log = tmp_path / "shared.log"
    log.write_text("", encoding="utf-8")
    env = dict(os.environ)
    env["LOCALM_DEBUG"] = str(log)
    env["LOCALM_HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120, creationflags=flags)
    kinds = next(x for x in proc.stdout.splitlines() if x.startswith("TYPE "))[5:].split(",")
    return proc, kinds, log.read_text(encoding="utf-8")


def test_a_child_writes_its_file_records_before_the_logging_call_returns(tmp_path):
    proc, kinds, text = _run_with_log(_CHILD_EXITS, tmp_path)
    assert proc.returncode == 1
    assert "FileHandler" in kinds and "_OffThreadFileHandler" not in kinds
    assert "child-line-before-exit" in text


def test_a_server_that_inherits_the_log_file_writes_it_off_thread(tmp_path):
    proc, kinds, _ = _run_with_log(_INHERITED_SERVER, tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "_OffThreadFileHandler" in kinds and "FileHandler" not in kinds


def test_the_stop_writes_the_debug_log_before_the_process_ends(monkeypatch):
    import localm.inference.http_server as hs
    events = []
    monkeypatch.setattr(hs, "_shutdown_teardown", lambda **kw: events.append("teardown"))
    monkeypatch.setattr(debuglog, "flush_log_handlers", lambda *a, **k: events.append("flush"))

    def fake_exit(code):
        events.append("exit")
        raise SystemExit(code)
    monkeypatch.setattr("os._exit", fake_exit)
    with pytest.raises(SystemExit):
        hs._do_shutdown()
    assert events == ["teardown", "flush", "exit"]
