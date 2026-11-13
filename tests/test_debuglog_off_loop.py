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
    monkeypatch.undo()
    assert [r.getMessage() for r in reported] == ["unqueued"]


_CHILD_EXITS = r"""
import os
from localm import debuglog
debuglog.attach_child_logging()
print("TYPE " + ",".join(sorted(type(h).__name__ for h in debuglog.logger.handlers)),
      flush=True)
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


def test_the_queue_bound_is_ten_thousand_records():
    assert debuglog._MAX_QUEUED_RECORDS == 10_000


def test_a_blocked_stream_holds_at_most_the_bound(debug_logger, monkeypatch):
    """A stream that stops accepting writes keeps at most the bound queued:
    logging stays fast, later records are dropped and counted, and once the
    stream accepts writes again one line says how many were dropped."""
    monkeypatch.setattr(debuglog, "_MAX_QUEUED_RECORDS", 20)
    stream = _BlockingStream()
    handler = debug_logger(_off_thread(stream))
    debuglog.logger.debug("first")
    assert stream.entered.wait(5)
    start = time.monotonic()
    for i in range(100):
        debuglog.logger.debug("queued %d", i)
    assert time.monotonic() - start < 2
    writer = handler.writer()
    assert writer.pending() == 20
    assert writer.dropped() == 80
    stream.release.set()
    assert writer.drain(10)
    lines = stream.getvalue().splitlines()
    assert lines[0] == "first"
    assert lines[1:21] == [f"queued {i}" for i in range(20)]
    assert lines[21:] == ["debug log: 80 records were dropped while the console did not "
                          "accept writes"]
    assert writer.dropped() == 0 and writer.pending() == 0


@pytest.fixture
def slow_debug_file(tmp_path, monkeypatch):
    """A debug log file in tmp_path/logs named for this process, written by an
    off-thread file handler whose writes take 0.2 s each."""
    import os
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / f"localm_2026-01-01_000000_{os.getpid()}.log"
    monkeypatch.setenv("LOCALM_DEBUG", str(path))
    handler = debuglog._OffThreadFileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    real_write = handler._write

    def slow_write(text, record):
        time.sleep(0.2)
        real_write(text, record)
    handler._write = slow_write
    debuglog.logger.addHandler(handler)
    old_level = debuglog.logger.level
    debuglog.logger.setLevel(logging.DEBUG)
    try:
        yield tmp_path, path
    finally:
        handler.writer().drain(10)
        debuglog.logger.removeHandler(handler)
        debuglog.logger.setLevel(old_level)
        handler.close()


def test_a_line_logged_just_before_a_bug_report_is_in_its_log_tail(slow_debug_file):
    import os

    from localm.bugreport.logs import _recent_log_tail_result
    home, _ = slow_debug_file
    debuglog.logger.warning("first line")
    debuglog.logger.warning("line just before the report")
    tail, unavailable = _recent_log_tail_result(home, pid=os.getpid())
    assert unavailable == ""
    assert "line just before the report" in tail


def test_native_output_lands_after_the_lines_logged_before_it(slow_debug_file):
    import os
    _, path = slow_debug_file
    debuglog.logger.warning("python line")
    fd = debuglog.native_stderr_target()
    assert fd is not None
    try:
        os.write(fd, b"native line\n")
    finally:
        os.close(fd)
    text = path.read_text(encoding="utf-8")
    assert text.index("python line") < text.index("native line")


def test_a_debug_log_file_that_does_not_drain_is_named_and_the_reader_goes_on(
        tmp_path, debug_logger):
    release = threading.Event()
    path = tmp_path / "debug.log"
    file_handler = debuglog._OffThreadFileHandler(path, mode="a", encoding="utf-8", delay=True)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    real_write = file_handler._write
    file_handler._write = lambda text, record: (release.wait(30), real_write(text, record))
    debug_logger(file_handler)
    console = io.StringIO()
    console_handler = debug_logger(_off_thread(console))
    debuglog.logger.warning("held line")
    start = time.monotonic()
    assert debuglog.drain_debug_log(0.3) is False
    assert time.monotonic() - start < 2
    release.set()
    assert file_handler.writer().drain(10)
    assert console_handler.writer().drain(10)
    warned = [line for line in console.getvalue().splitlines() if "not written" in line]
    assert len(warned) == 1
    assert "before the debug log file was read" in warned[0]
    assert str(path) in warned[0]
