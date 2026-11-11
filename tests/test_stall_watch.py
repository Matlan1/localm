# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stall watch sees a stall a Python watchdog thread cannot, and keeps an
in-memory record that names where it was.

Three injected stalls, each through the real mechanism (no mocks of the
watch): a long ``json.dumps`` (a C call that holds the GIL and burns CPU), a
native call made through ``ctypes.PyDLL`` that sleeps without releasing the
GIL, and a blocking call made ON a running asyncio event loop while other
threads keep running.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import sys
import threading
import time

import pytest

from localm.inference import _stall_watch as sw

_THRESHOLD = 0.5
_POLL = 0.1


def _wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


@pytest.fixture
def watch_factory():
    started = []

    def make(**kw):
        kw.setdefault("heartbeat_gap", lambda: None)
        kw.setdefault("loop_thread", None)
        w = sw.StallWatch(threshold=_THRESHOLD, poll=_POLL, **kw).start()
        started.append(w)
        time.sleep(3 * _POLL)
        return w
    yield make
    for w in started:
        w.stop()


def _hold_gil_in_json_dumps():
    rows = [[0.123456789] * 384 for _ in range(12000)]
    t0 = time.perf_counter()
    json.dumps(rows)
    return time.perf_counter() - t0


def _native_sleep_holding_the_gil(ms: int) -> None:
    if sys.platform == "win32":
        ctypes.PyDLL("kernel32").Sleep(ms)
    else:
        ctypes.PyDLL(None).nanosleep(ctypes.byref(_timespec(ms)), None)


class _timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    def __init__(self, ms: int) -> None:
        super().__init__(ms // 1000, (ms % 1000) * 1_000_000)


def test_a_gil_held_in_a_c_call_is_recorded_with_its_frame(watch_factory):
    w = watch_factory()
    took = []
    t = threading.Thread(target=lambda: took.append(_hold_gil_in_json_dumps()))
    t.start()
    t.join()
    assert took[0] > 1.5 * _THRESHOLD, f"json.dumps took only {took[0]:.2f} s"
    assert _wait_for(lambda: w.records())
    rec = w.records()[0]
    assert rec["kind"] == "gil" and not rec["ongoing"]
    assert _THRESHOLD <= rec["seconds"] <= took[0] + 2.5
    assert rec["where"].startswith("json/encoder.py:")
    assert any("in _hold_gil_in_json_dumps" in f for f in rec["frames"])
    assert all(":\\" not in f and not f.startswith("/") for f in rec["frames"])


def test_a_native_call_holding_the_gil_without_cpu_is_recorded(watch_factory):
    w = watch_factory()
    t = threading.Thread(target=_native_sleep_holding_the_gil, args=(2000,))
    t.start()
    t.join()
    assert _wait_for(lambda: w.records())
    rec = w.records()[0]
    assert rec["kind"] == "gil"
    assert "in _native_sleep_holding_the_gil" in rec["frames"][0]


def test_a_short_hold_and_an_idle_server_record_nothing(watch_factory):
    w = watch_factory()
    _native_sleep_holding_the_gil(200)
    time.sleep(_THRESHOLD + 5 * _POLL)
    assert w.records() == []


def test_a_blocked_event_loop_is_recorded_with_the_loop_frame(watch_factory):
    loop = asyncio.new_event_loop()
    beat = {"t": None}

    async def heartbeat():
        while True:
            beat["t"] = time.monotonic()
            await asyncio.sleep(0.05)

    def run():
        asyncio.set_event_loop(loop)
        loop.create_task(heartbeat())
        loop.run_forever()

    lt = threading.Thread(target=run, daemon=True)
    lt.start()
    assert _wait_for(lambda: beat["t"] is not None)

    def _blocking_call_on_the_loop():
        time.sleep(2.5)

    try:
        w = watch_factory(heartbeat_gap=lambda: time.monotonic() - beat["t"],
                          loop_thread=lt.ident)
        loop.call_soon_threadsafe(_blocking_call_on_the_loop)
        assert _wait_for(lambda: w.records() and w.records()[0]["ongoing"], 5)
        assert _wait_for(lambda: not w.records()[0]["ongoing"], 5)
        rec = w.records()[0]
        assert rec["kind"] == "loop"
        assert "in _blocking_call_on_the_loop" in rec["frames"][0]
        assert 2.0 <= rec["seconds"] <= 4.0
        assert len(w.records()) == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        lt.join(5)
        loop.close()


def test_a_single_stale_heartbeat_sample_records_nothing(watch_factory):
    gaps = iter([0.0, 50.0])
    w = watch_factory(heartbeat_gap=lambda: next(gaps, 0.0))
    time.sleep(10 * _POLL)
    assert w.records() == []


def test_stop_cancels_the_armed_dump(watch_factory, tmp_path):
    w = watch_factory()
    w.stop()
    out = tmp_path / "late.txt"
    with open(out, "w") as fh:
        _native_sleep_holding_the_gil(int(_THRESHOLD * 1000) + 700)
        fh.flush()
    assert out.read_text() == ""
    assert w.records() == []


def test_the_trace_sink_receives_the_raw_dump(watch_factory):
    got = []
    w = watch_factory(trace_sink=got.append)
    t = threading.Thread(target=_hold_gil_in_json_dumps)
    t.start()
    t.join()
    assert _wait_for(lambda: w.records())
    assert got and "Timeout" in got[0] and "_hold_gil_in_json_dumps" in got[0]


def test_recent_stalls_reads_the_active_watch(watch_factory):
    assert sw.recent_stalls() == []
    w = watch_factory()
    sw.set_active(w)
    try:
        w._add({"id": 1, "at": 0.0, "seconds": 12.0, "kind": "gil", "where": "x",
                "frames": ["x"], "ongoing": False})
        assert sw.recent_stalls()[0]["seconds"] == 12.0
    finally:
        sw.set_active(None)
    assert sw.recent_stalls() == []


@pytest.mark.parametrize("path, short", [
    (r"C:\Users\someone\localm\localm\rag\_store\files.py", "localm/rag/_store/files.py"),
    ("/home/someone/.local/lib/python3.12/json/encoder.py", "json/encoder.py"),
    (r"D:\x\.python\cpython-3.12\Lib\json\encoder.py", "json/encoder.py"),
    ("/opt/venv/lib/python3.12/site-packages/anyio/_backends/_asyncio.py",
     "anyio/_backends/_asyncio.py"),
    (r"C:\Users\someone\localm-home\plugins\rag\plug.py", "plugins/rag/plug.py"),
    ("/srv/somewhere/thing.py", "thing.py"),
    ("<string>", "<string>"),
])
def test_short_path_keeps_no_machine_part(path, short):
    assert sw.short_path(path) == short


@pytest.mark.parametrize("line", [
    "", "  ", 'File "', '  File "x"', '  File "x", line', '  File "x", line 3', '  File "x", line y in f',
    "  <unknown>", "Timeout (0:00:01)!",
])
def test_a_line_that_is_not_a_frame_is_ignored(line):
    assert sw._parse_frame(line) is None


def test_parsing_a_hostile_frame_line_is_linear():
    line = '  File "' + '", line 0 in a' * 8000 + "\n\n"
    t0 = time.perf_counter()
    sw._parse_frame(line)
    assert time.perf_counter() - t0 < 0.5


def test_parse_dump_reads_threads_and_frames():
    text = ("Timeout (0:00:01)!\nThread 0x0000abcd (most recent call first):\n"
            '  File "/x/localm/a.py", line 3 in f\n  File "/x/localm/b.py", line 9 in g\n\n'
            "Current thread 0x00000010 (most recent call first):\n"
            '  File "/x/lib/python3.12/threading.py", line 359 in wait\n')
    threads = sw.parse_dump(text)
    assert threads == {0xabcd: [("/x/localm/a.py", 3, "f"), ("/x/localm/b.py", 9, "g")],
                       0x10: [("/x/lib/python3.12/threading.py", 359, "wait")]}
    assert sw.describe(threads[0xabcd]) == "localm/a.py:3 in f"
    assert sw.describe([("/x/lib/python3.12/json/encoder.py", 258, "iterencode"),
                        ("/x/localm/rag/f.py", 320, "_write")]) == (
        "json/encoder.py:258 in iterencode, called from localm/rag/f.py:320 in _write")


def _loop_stand_in(stop: threading.Event) -> None:
    while not stop.is_set():
        time.sleep(0.01)


def test_a_gil_stall_names_the_loop_thread_first_not_the_busiest_guess(watch_factory):
    """The loop thread is known: ``where`` and ``frames`` are the loop thread's,
    the CPU-proven holder is added after it, and the record and the trace sink
    carry every thread's stack, the loop thread first."""
    stop = threading.Event()
    loop = threading.Thread(target=_loop_stand_in, args=(stop,), name="loop-stand-in",
                            daemon=True)
    loop.start()
    got = []
    try:
        w = watch_factory(loop_thread=loop.ident, trace_sink=got.append)
        t = threading.Thread(target=_hold_gil_in_json_dumps, name="gil-holder")
        t.start()
        t.join()
        assert _wait_for(lambda: w.records())
    finally:
        stop.set()
        loop.join(5)
    rec = w.records(with_threads=True)[0]
    assert rec["kind"] == "gil"
    assert "in _loop_stand_in on the event loop thread" in rec["where"], rec["where"]
    assert "in _loop_stand_in" in rec["frames"][0], rec["frames"]
    holder = rec["where"].split("; ", 1)[1]
    assert "json/encoder.py" in holder and "_hold_gil_in_json_dumps" in holder, holder
    assert rec["threads"][0]["loop"] and rec["threads"][0]["ident"] == loop.ident
    assert rec["threads"][0]["name"] == "loop-stand-in"
    assert any(e["ident"] == t.ident and any("_hold_gil_in_json_dumps" in f for f in e["frames"])
               for e in rec["threads"]), rec["threads"]
    assert all(":\\" not in f and not f.startswith("/")
               for e in rec["threads"] for f in e["frames"])
    assert "threads" not in w.records()[0]
    first = next(ln for ln in got[0].splitlines() if ln.startswith("Thread 0x"))
    assert first.startswith(f"Thread 0x{loop.ident:08x} "), first
    assert "[event loop thread]" in first, first
    lines = got[0].splitlines()
    assert lines[0].startswith("Timeout"), lines[:2]
    assert lines[1].startswith("the event loop thread was in "), lines[:2]
    assert "in _loop_stand_in" in lines[1], lines[:2]


def test_a_blocked_loop_record_carries_every_thread_loop_first(watch_factory):
    loop = asyncio.new_event_loop()
    beat = {"t": None}
    release = threading.Event()
    bystander = threading.Thread(target=release.wait, name="sw-bystander", daemon=True)
    bystander.start()

    async def heartbeat():
        while True:
            beat["t"] = time.monotonic()
            await asyncio.sleep(0.05)

    def run():
        asyncio.set_event_loop(loop)
        loop.create_task(heartbeat())
        loop.run_forever()

    lt = threading.Thread(target=run, name="sw-loop", daemon=True)
    lt.start()
    assert _wait_for(lambda: beat["t"] is not None)

    def _blocking_call_on_the_loop():
        time.sleep(2.0)

    try:
        w = watch_factory(heartbeat_gap=lambda: time.monotonic() - beat["t"],
                          loop_thread=lt.ident)
        sw.set_active(w)
        loop.call_soon_threadsafe(_blocking_call_on_the_loop)
        assert _wait_for(lambda: w.records(), 5)
        rec = w.records(with_threads=True)[0]
        text = sw.recent_stall_stacks()
    finally:
        sw.set_active(None)
        loop.call_soon_threadsafe(loop.stop)
        lt.join(5)
        loop.close()
        release.set()
        bystander.join(5)
    assert rec["kind"] == "loop"
    assert rec["threads"][0]["loop"] and rec["threads"][0]["name"] == "sw-loop"
    assert "in _blocking_call_on_the_loop" in rec["threads"][0]["frames"][0]
    assert {"sw-bystander", "MainThread"} <= {e["name"] for e in rec["threads"]}
    headers = [ln for ln in text.splitlines() if ln.startswith("Thread 0x")]
    assert "[event loop thread]" in headers[0] and '"sw-loop"' in headers[0], headers
    assert any('"sw-bystander" daemon=True' in h for h in headers), headers
    assert "in _blocking_call_on_the_loop" in text.splitlines()[1], text


_RUNNER_WAITING_DUMP = (
    "Timeout (0:00:10)!\n"
    "Thread 0x0000202c (most recent call first):\n"
    '  File "D:\\x\\Lib\\multiprocessing\\connection.py", line 1016 in _exhaustive_wait\n'
    '  File "D:\\x\\Lib\\multiprocessing\\connection.py", line 1084 in wait\n'
    '  File "D:\\x\\Lib\\multiprocessing\\queues.py", line 113 in get\n'
    '  File "D:\\x\\localm\\inference\\backends\\llamacpp\\_runner.py", line 1030 in _poll\n'
    '  File "D:\\x\\localm\\inference\\backends\\llamacpp\\_runner.py", line 1103 '
    "in spawn_and_load\n\n"
    "Thread 0x00001111 (most recent call first):\n"
    '  File "D:\\x\\localm\\inference\\routes\\models.py", line 72 in _blocking_route\n'
    '  File "D:\\x\\Lib\\asyncio\\events.py", line 88 in _run\n')


def test_a_gil_stall_with_a_runner_waiting_names_the_loop_thread_not_the_runner():
    """The issue-2840 shape: a runner thread waiting on its worker
    (``_exhaustive_wait`` under ``_poll``) is listed before the loop thread.
    The record names the loop thread's frame, never the waiting runner."""
    w = sw.StallWatch(threshold=10.0, heartbeat_gap=lambda: None, loop_thread=0x1111)
    got = []
    w._trace_sink = got.append
    w._gil_stall(_RUNNER_WAITING_DUMP, 12.5, {})
    rec = w.records(with_threads=True)[0]
    assert rec["where"].startswith(
        "localm/inference/routes/models.py:72 in _blocking_route on the event loop thread"), \
        rec["where"]
    assert "_poll" not in rec["where"] and "other running" not in rec["where"], rec["where"]
    assert [e["ident"] for e in rec["threads"]] == [0x1111, 0x202c]
    assert rec["threads"][1]["frames"][3] == \
        "localm/inference/backends/llamacpp/_runner.py:1030 in _poll"
    headers = [ln for ln in got[0].splitlines() if ln.startswith("Thread 0x")]
    assert headers[0].startswith("Thread 0x00001111 ") and "[event loop thread]" in headers[0]


def test_compact_stack_text_shortens_frame_lines_linearly():
    from localm.pathscrub import compact_stack_text
    text = ('Thread 0x00000001 "x" daemon=False (most recent call first):\n'
            '  File "C:\\Users\\someone\\app\\localm\\a.py", line 3 in f\n'
            '  File "/home/someone/lib/python3.12/queue.py", line 171, in get\n'
            "    source line kept\n")
    assert compact_stack_text(text).splitlines() == [
        'Thread 0x00000001 "x" daemon=False (most recent call first):',
        "  localm/a.py:3 in f", "  queue.py:171 in get", "    source line kept"]
    hostile = 'File "' + '", line 1 in a' * 8000
    t0 = time.perf_counter()
    compact_stack_text(hostile)
    assert time.perf_counter() - t0 < 0.5
