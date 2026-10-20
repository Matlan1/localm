# SPDX-License-Identifier: AGPL-3.0-or-later
"""JsonlEventWriter: one intact JSON object per line, from any thread."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

from localm.plugins.coder.eventstream import JsonlEventWriter


def test_each_event_is_one_line_written_unchanged():
    buf = io.StringIO()
    w = JsonlEventWriter(buf)
    w({"type": "tool_call", "tool": "read_file", "args": {"path": "a.py"}})
    w({"type": "token", "text": "multi\nline é"})
    lines = buf.getvalue().splitlines()
    assert [json.loads(ln) for ln in lines] == [
        {"type": "tool_call", "tool": "read_file", "args": {"path": "a.py"}},
        {"type": "token", "text": "multi\nline é"},
    ]


def test_output_is_ascii_even_for_non_ascii_and_lone_surrogates():
    buf = io.StringIO()
    JsonlEventWriter(buf)({"type": "token", "text": "é\ud800"})
    assert buf.getvalue().isascii()


def test_non_json_values_are_written_as_their_string():
    buf = io.StringIO()
    JsonlEventWriter(buf)({"type": "info", "path": Path("x")})
    assert json.loads(buf.getvalue())["path"] == str(Path("x"))


def test_each_write_is_flushed():
    class Counting(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1

    buf = Counting()
    w = JsonlEventWriter(buf)
    w({"type": "a"})
    w({"type": "b"})
    assert buf.flushes == 2


class _YieldingStream:
    """A stream whose write is not atomic: it hands the CPU to other threads
    between small pieces, as a pipe or console write can."""

    def __init__(self):
        self.parts = []

    def write(self, text):
        for i in range(0, len(text), 4):
            self.parts.append(text[i:i + 4])
            time.sleep(0)

    def flush(self):
        pass


def test_concurrent_writers_never_interleave_a_line():
    stream = _YieldingStream()
    w = JsonlEventWriter(stream)

    def spam(tag):
        for i in range(50):
            w({"type": "token", "text": f"{tag}-{i}" * 10})

    threads = [threading.Thread(target=spam, args=(t,)) for t in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = "".join(stream.parts).splitlines()
    assert len(lines) == 200
    assert all(json.loads(ln)["type"] == "token" for ln in lines)
