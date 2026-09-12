# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG indexing progress lines name the indexed documents (file names, the
absolute path of a pruned file, the indexed root folder). They stay on the
job's ephemeral event stream, which the GUI shows live, and never enter the
always-on activity ring that a bug report carries. Only the two
embedding-degrade warnings reach the ring, and the non-finite one without the
document's name."""

import logging

import pytest

from localm import _log_digest as ld
from localm import debuglog
from localm.plugins.builtin.rag import plug


@pytest.fixture
def fresh_ring():
    """A clean activity ring on the localm logger for one test; the logger's
    prior handlers, level and ring are restored afterwards."""
    logger = debuglog.logger
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_ring = debuglog._ring_handler
    for h in list(logger.handlers):
        if isinstance(h, debuglog._RingBufferHandler):
            logger.removeHandler(h)
    debuglog._ring_handler = None
    assert debuglog.install_ring_buffer() is True
    yield logger
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    debuglog._ring_handler = saved_ring


class _Job:
    def __init__(self):
        self.pushed = []
        self.progressed = []

    def push(self, event):
        self.pushed.append(event)

    def progress(self, **kw):
        self.progressed.append(kw)


LINES = [
    "indexed secret-report.pdf (3 chunks)",
    "pruned: D:/Legal/settlement.docx (file is gone)",
    "embeddings unavailable (boom) - indexing lexical-only",
    "embeddings had non-finite (NaN/inf) values for secret.pdf - indexing it lexical-only",
]
NAMES = ("secret-report.pdf", "settlement.docx", "secret.pdf", "D:/Legal")


def _drive(job):
    cb = plug._job_progress(job)
    for line in LINES:
        cb(line)
    return cb


def test_document_identity_never_reaches_the_activity_ring(fresh_ring):
    job = _Job()
    _drive(job)

    ring = "\n".join(debuglog.recent_activity())
    for name in NAMES:
        assert name not in ring, f"{name!r} reached the always-on activity ring"
    # The degrade warnings do reach it, at WARNING, with the wording
    # tests/test_rag_api_mode.py pins.
    assert "embeddings unavailable" in ring
    assert "indexing lexical-only" in ring
    assert "embeddings had non-finite" in ring
    assert ring.count("WARNING") == 2
    # The job stream still carries every line verbatim.
    assert [e["text"] for e in job.pushed] == LINES
    assert all(e["type"] == "line" for e in job.pushed)


def test_no_document_identity_at_warning_level(caplog):
    job = _Job()
    with caplog.at_level(logging.DEBUG, logger="localm"):
        _drive(job)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    for msg in warnings:
        for name in NAMES:
            assert name not in msg
    assert all(msg.startswith("rag index degrade: ") for msg in warnings)
    # The identity-bearing lines are DEBUG, and withheld while the content
    # gate (debug log on, no privacy surface) is closed - as it is here.
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug) == 2
    assert all(m == "rag index progress (document identity withheld)" for m in debug)


def test_identity_is_logged_at_debug_only_when_content_is_allowed(caplog, monkeypatch):
    monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)
    job = _Job()
    with caplog.at_level(logging.DEBUG, logger="localm"):
        _drive(job)

    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug == ["rag index: indexed secret-report.pdf (3 chunks)",
                     "rag index: pruned: D:/Legal/settlement.docx (file is gone)"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("secret.pdf" in m for m in warnings), \
        "the non-finite degrade never names the document, even with content allowed"


def test_structured_progress_still_reaches_the_job(fresh_ring):
    job = _Job()
    cb = plug._job_progress(job)
    cb("re-embedding 3/10 chunks of secret-report.pdf", phase="reembed", done=3,
       total=10, unit="chunks")
    assert job.progressed == [{"phase": "reembed", "done": 3, "total": 10, "unit": "chunks"}]
    assert "secret-report.pdf" not in "\n".join(debuglog.recent_activity())


class TestRunLogDigest:
    """The opt-in --debug run log holds the DEBUG lines; the digest a bug
    report renders strips the content-bearing ones and keeps the degrade
    warnings."""

    def test_content_bearing_rag_line_is_withheld_from_the_digest(self):
        # The degrade fires before the document's own "indexed" line, as
        # store.py emits them. A record AFTER a content record is withheld by
        # the digest's resync rule (_drop_content_records) whatever its level;
        # the degrade warning also lives in the activity ring, which the
        # report renders separately.
        text = (
            "2026-09-12 10:00:00,000 WARNING localm: rag index degrade: embeddings "
            "unavailable (boom) - indexing lexical-only\n"
            "2026-09-12 10:00:01,000 DEBUG   localm: rag index: indexed "
            "secret-report.pdf (3 chunks)\n"
        )
        digest = ld.build_digest(text)
        assert "secret-report.pdf" not in digest
        assert "withheld" in digest
        assert "rag index degrade: embeddings unavailable" in digest

    def test_identity_withheld_line_does_not_taint_the_digest(self):
        text = (
            "2026-09-12 10:00:00,000 DEBUG   localm: rag index progress "
            "(document identity withheld)\n"
            "2026-09-12 10:00:01,000 WARNING localm: rag index degrade: embeddings "
            "unavailable (boom) - indexing lexical-only\n"
        )
        digest = ld.build_digest(text)
        assert "record(s) withheld" not in digest
        assert "rag index degrade: embeddings unavailable" in digest

    def test_marker_matches_the_content_line_and_not_the_degrade_line(self):
        assert ld.is_content_record(
            {"level": "DEBUG", "logger": "localm", "lines": [
                "2026-09-12 10:00:00,000 DEBUG   localm: rag index: pruned: "
                "D:/Legal/settlement.docx (file is gone)"]})
        assert not ld.is_content_record(
            {"level": "WARNING", "logger": "localm", "lines": [
                "2026-09-12 10:00:01,000 WARNING localm: rag index degrade: "
                "embeddings had non-finite (NaN/inf) values for a document - "
                "indexing it lexical-only"]})
