# SPDX-License-Identifier: AGPL-3.0-or-later
"""_extract_pdf() must be bounded in both page count and wall-clock time.

Before this fix it iterated every page of reader.pages with no cap and no
deadline; MAX_TEXT_CHARS was applied only after every page was already in
memory, so it never shortened the loop. Fuzzing measured 52s / 150MB peak on
a 50,000-page PDF; at the 30MB input cap that extrapolates to minutes on one
plugin-pool worker (see localm/rag/extract.py's MAX_PDF_PAGES /
MAX_PDF_EXTRACT_SECONDS).
"""

import pytest

from localm.rag import extract


def _multi_page_pdf(n_pages: int, text: str = "hello") -> bytes:
    """A minimal valid N-page PDF, all pages sharing one content stream and
    one font, built by hand (same technique as tests/test_rag.py's
    _tiny_pdf) so the test needs no PDF-writer dependency - only pypdf, to
    read it back."""
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


@pytest.fixture(autouse=True)
def _require_pypdf():
    pytest.importorskip("pypdf")


def test_page_cap_stops_the_walk(monkeypatch):
    monkeypatch.setattr(extract, "MAX_PDF_PAGES", 5)
    from pypdf._page import PageObject
    calls = {"n": 0}
    real_extract_text = PageObject.extract_text

    def counting_extract_text(self, *a, **kw):
        calls["n"] += 1
        return real_extract_text(self, *a, **kw)

    monkeypatch.setattr(PageObject, "extract_text", counting_extract_text)

    text = extract._extract_pdf(_multi_page_pdf(20), "many-pages.pdf")

    assert calls["n"] == 5, "the page loop did not stop at the cap"
    assert "[page 5]" in text
    assert "[page 6]" not in text
    assert "page cap" in text
    assert "5" in text


def test_deadline_stops_the_walk(monkeypatch):
    monkeypatch.setattr(extract, "MAX_PDF_EXTRACT_SECONDS", 1.0)
    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 0.4
        return clock["t"]

    monkeypatch.setattr(extract, "_monotonic", fake_monotonic)

    text = extract._extract_pdf(_multi_page_pdf(20), "slow.pdf")

    # deadline = 0.4 (first call, taken before the loop) + 1.0 = 1.4.
    # Loop checks the clock once per page BEFORE extracting; it advances by
    # 0.4 each call, so it should stop after a small, bounded number of pages.
    page_markers = [line for line in text.splitlines() if line.startswith("[page ")]
    assert 0 < len(page_markers) < 20, (
        f"expected the deadline to stop the walk before all 20 pages, got {text!r}")
    assert "time limit" in text
    assert "1" in text


def test_an_ordinary_pdf_is_untouched_by_the_bounds():
    text = extract._extract_pdf(_multi_page_pdf(3), "small.pdf")
    assert "[page 1]" in text
    assert "[page 2]" in text
    assert "[page 3]" in text
    assert "truncated" not in text


def test_text_budget_stops_the_walk(monkeypatch):
    monkeypatch.setattr(extract, "MAX_TEXT_CHARS", 200)

    text = extract._extract_pdf(_multi_page_pdf(50), "many-pages.pdf")

    page_markers = [line for line in text.splitlines() if line.startswith("[page ")]
    assert 0 < len(page_markers) < 50, (
        f"expected the text budget to stop the walk before all 50 pages, got {text!r}")
    assert "truncated" in text
