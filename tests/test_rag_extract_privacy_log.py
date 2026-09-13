# SPDX-License-Identifier: AGPL-3.0-or-later
"""rag/extract.py's archive-safety warnings (a member over the per-member
decompressed-size cap, the whole-archive inflated-bytes budget, a PDF
extraction stopped early) name the archive member and/or the document.
Neither must ever reach the always-on activity ring or a WARNING-level log
record; only an identity-free WARNING does, and the identity travels at
DEBUG only when debug_content_enabled() allows content in the debug log."""

import io
import logging
import zipfile

import pytest

from localm import _log_digest as ld
from localm import debuglog
from localm.rag import extract
from localm.rag.extract import extract_bytes


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


MEMBER_NAME = "secret-contract.docx"
ARCHIVE_NAME = "secret-quarterly-report.zip"
PDF_NAME = "secret-board-minutes.pdf"


def _zip_with_oversized_member(member: str, size: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, b"\x00" * size)
    return buf.getvalue()


def _zip_oversized_in_total(n_members: int, size: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(n_members):
            zf.writestr(f"m{i}.bin", b"\x00" * size)
    return buf.getvalue()


def _extract_ignoring_refusal(data: bytes, name: str) -> str:
    """Extract, treating a refusal as an empty result - the refusal itself
    is never what these tests measure, only what got logged along the way."""
    try:
        return extract_bytes(data, name)
    except Exception:
        return ""


class TestMemberCapWarning:
    """_extract_zip: one archive member over MAX_ARCHIVE_MEMBER_BYTES."""

    def test_identity_never_reaches_the_activity_ring(self, fresh_ring, monkeypatch):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_MEMBER_BYTES", 1_000)
        _extract_ignoring_refusal(
            _zip_with_oversized_member(MEMBER_NAME, 5_000), "container.zip")

        ring = "\n".join(debuglog.recent_activity())
        assert MEMBER_NAME not in ring
        assert "container.zip" not in ring
        assert "an archive member exceeded the decompressed-size limit" in ring
        assert "WARNING" in ring

    def test_identity_absent_from_every_warning_record(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_MEMBER_BYTES", 1_000)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _extract_ignoring_refusal(
                _zip_with_oversized_member(MEMBER_NAME, 5_000), "container.zip")

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "the member-cap safety warning did not fire"
        for msg in warnings:
            assert MEMBER_NAME not in msg
            assert "container.zip" not in msg

        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert not any(MEMBER_NAME in m for m in debug), \
            "identity leaked at DEBUG while the content gate was closed"
        assert any(m == "rag extract identity withheld" for m in debug)

    def test_identity_logged_at_debug_only_when_content_is_allowed(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_MEMBER_BYTES", 1_000)
        monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _extract_ignoring_refusal(
                _zip_with_oversized_member(MEMBER_NAME, 5_000), "container.zip")

        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(MEMBER_NAME in m and m.startswith("rag extract identity: ")
                   for m in debug), \
            "identity was not logged at DEBUG when content is allowed"


class TestWholeArchiveBudgetWarning:
    """_extract_zip: the running whole-archive inflated-bytes budget."""

    def test_identity_never_reaches_the_activity_ring(self, fresh_ring, monkeypatch):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_INFLATED_BYTES", 2_000)
        _extract_ignoring_refusal(_zip_oversized_in_total(5, 1_000), ARCHIVE_NAME)

        ring = "\n".join(debuglog.recent_activity())
        assert ARCHIVE_NAME not in ring
        assert "whole-archive decompressed-size budget" in ring
        assert "WARNING" in ring

    def test_identity_absent_from_every_warning_record(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_INFLATED_BYTES", 2_000)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _extract_ignoring_refusal(_zip_oversized_in_total(5, 1_000), ARCHIVE_NAME)

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "the whole-archive budget safety warning did not fire"
        for msg in warnings:
            assert ARCHIVE_NAME not in msg

    def test_identity_logged_at_debug_only_when_content_is_allowed(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_ARCHIVE_INFLATED_BYTES", 2_000)
        monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _extract_ignoring_refusal(_zip_oversized_in_total(5, 1_000), ARCHIVE_NAME)

        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(ARCHIVE_NAME in m and m.startswith("rag extract identity: ")
                   for m in debug), \
            "identity was not logged at DEBUG when content is allowed"


def _multi_page_pdf(n_pages: int, text: str = "hello") -> bytes:
    """A minimal valid N-page PDF, all pages sharing one content stream and
    one font, built by hand (same technique as tests/test_rag_pdf_bounds.py's
    _multi_page_pdf) so the test needs no PDF-writer dependency - only pypdf,
    to read it back."""
    content = b"BT /F1 18 Tf 20 100 Td (" + text.encode("latin-1") + b") Tj ET"
    content_obj_num = n_pages + 3
    font_obj_num = n_pages + 4
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>").encode(),
    ]
    for _ in range(n_pages):
        objs.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
             f"/Contents {content_obj_num} 0 R "
             f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>").encode()
        )
    objs.append(
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content + b"\nendstream"
    )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    startxref = len(pdf)
    pdf += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        pdf += ("%010d 00000 n \n" % off).encode()
    pdf += (b"trailer\n<< /Root 1 0 R /Size " + str(len(objs) + 1).encode()
            + b" >>\nstartxref\n" + str(startxref).encode() + b"\n%%EOF\n")
    return pdf


class TestPdfStoppedEarlyWarning:
    """_extract_pdf: the page-count cap stops the walk early."""

    @pytest.fixture(autouse=True)
    def _require_pypdf(self):
        pytest.importorskip("pypdf")

    def test_identity_never_reaches_the_activity_ring(self, fresh_ring, monkeypatch):
        monkeypatch.setattr(extract, "MAX_PDF_PAGES", 1)
        extract._extract_pdf(_multi_page_pdf(5), PDF_NAME)

        ring = "\n".join(debuglog.recent_activity())
        assert PDF_NAME not in ring
        assert "PDF extraction stopped early" in ring
        assert "WARNING" in ring

    def test_identity_absent_from_every_warning_record(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_PDF_PAGES", 1)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            extract._extract_pdf(_multi_page_pdf(5), PDF_NAME)

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "the PDF-stopped-early safety warning did not fire"
        for msg in warnings:
            assert PDF_NAME not in msg

    def test_identity_logged_at_debug_only_when_content_is_allowed(self, monkeypatch, caplog):
        monkeypatch.setattr(extract, "MAX_PDF_PAGES", 1)
        monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            extract._extract_pdf(_multi_page_pdf(5), PDF_NAME)

        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(PDF_NAME in m and m.startswith("rag extract identity: ")
                   for m in debug), \
            "identity was not logged at DEBUG when content is allowed"


class TestLogDigestMarker:
    """The digest a bug report renders strips the identity-bearing DEBUG
    line and keeps the identity-free WARNING, whatever fires it."""

    def test_marker_matches_the_identity_line_and_not_the_warning(self):
        assert ld.is_content_record(
            {"level": "DEBUG", "logger": "localm", "lines": [
                "2026-09-13 10:00:00,000 DEBUG   localm: rag extract identity: "
                "archive member secret-contract.docx in container.zip exceeds "
                "the decompressed-size limit"]})
        assert not ld.is_content_record(
            {"level": "WARNING", "logger": "localm", "lines": [
                "2026-09-13 10:00:01,000 WARNING localm: rag: an archive member "
                "exceeded the decompressed-size limit; skipped"]})
