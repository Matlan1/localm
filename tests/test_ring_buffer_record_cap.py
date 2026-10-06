# SPDX-License-Identifier: AGPL-3.0-or-later
"""The recent-activity ring bounds each stored record, not only the count."""

import logging

import pytest

from localm import debuglog


@pytest.fixture
def fresh_ring(monkeypatch):
    saved_handlers = list(debuglog.logger.handlers)
    saved_level = debuglog.logger.level
    saved_ring = debuglog._ring_handler
    debuglog.logger.handlers = []
    debuglog.logger.setLevel(logging.NOTSET)
    debuglog._ring_handler = None
    monkeypatch.setattr(debuglog, "load_ring_buffer", lambda: None)
    debuglog.install_ring_buffer()
    try:
        yield
    finally:
        debuglog.logger.handlers = saved_handlers
        debuglog.logger.setLevel(saved_level)
        debuglog._ring_handler = saved_ring


def _suffix(n: int) -> str:
    return f"...(truncated, {n} chars)"


def test_huge_info_record_is_truncated_with_marker(fresh_ring):
    big = "Q" * 5_000_000
    logging.getLogger("localm.web").info("query=%s", big)
    entries = debuglog.recent_activity()
    assert len(entries) == 1
    entry = entries[0]
    total = len(entry.split("query=", 1)[0]) + len("query=") + len(big)
    assert len(entry) <= debuglog._RING_MAX_RECORD_CHARS + len(_suffix(total))
    assert entry.endswith(_suffix(total))
    assert "query=QQQ" in entry


def test_short_record_is_stored_unchanged(fresh_ring):
    logging.getLogger("localm.web").info("short line")
    entry = debuglog.recent_activity()[0]
    assert entry.endswith("localm.web: short line")
    assert "truncated" not in entry


def test_record_at_the_ceiling_is_not_marked(fresh_ring):
    probe = logging.LogRecord("localm.web", logging.INFO, __file__, 1, "", (), None)
    overhead = len(debuglog._ring_handler.format(probe))
    msg = "x" * (debuglog._RING_MAX_RECORD_CHARS - overhead)
    logging.getLogger("localm.web").info(msg)
    entry = debuglog.recent_activity()[0]
    assert len(entry) == debuglog._RING_MAX_RECORD_CHARS
    assert "truncated" not in entry


def test_native_line_is_truncated_too(fresh_ring):
    debuglog.record_native_line("N" * 100_000)
    entry = debuglog.recent_activity()[-1]
    assert len(entry) <= debuglog._RING_MAX_RECORD_CHARS + len(_suffix(100_000 + 60))
    assert "(truncated," in entry
