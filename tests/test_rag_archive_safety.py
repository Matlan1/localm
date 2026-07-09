# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG archive-extraction safety (Antigravity-audit HIGH-8 / MED-17 / MED-21).

HIGH-8  _extract_zip/_extract_tar bound each single member (80 MB) but had NO cap
        on the number of members or the running accumulated output - the
        whole-archive MAX_TEXT_CHARS truncation was applied only AFTER every
        member was decoded and joined in memory. A 30 MB zip of thousands of
        highly-compressible members could build ~30 GB in RAM first (a
        decompression-amplification DoS).

MED-17  Compressed-tar suffixes (.tgz/.tbz/.txz) were whitelisted as extractable
        but had no dispatch branch, so a valid tarball raised "Unsupported file
        type". And a single gzip/bzip2/xz-compressed non-tar file was sniffed to
        ".tar" and routed to the tar extractor, which failed.

MED-21  A per-member extraction error was appended into the indexed text as
        "[file: X - error: ...]", so failures became retrievable knowledge-base
        content instead of being surfaced/logged (AGENTS.md rule 5).
"""

import bz2
import gzip
import io
import tarfile
import zipfile

from localm.rag import extract
from localm.rag.extract import extract_bytes


def _zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _targz(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            b = data.encode() if isinstance(data, str) else data
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


def test_archive_extraction_is_bounded(monkeypatch):
    """HIGH-8: the extractor stops once the accumulated-text budget is reached,
    instead of decoding every member into memory first."""
    # Shrink the budget so the test stays small and fast.
    monkeypatch.setattr(extract, "MAX_TEXT_CHARS", 5_000)

    calls = {"n": 0}
    real_sniff = extract.sniff_format

    def counting_sniff(data, filename):
        calls["n"] += 1
        return real_sniff(data, filename)

    monkeypatch.setattr(extract, "sniff_format", counting_sniff)

    # 40 text members of 2000 chars each = 80,000 chars total, budget 5,000.
    members = {f"f{i:03d}.txt": ("x" * 2000) for i in range(40)}
    out = extract_bytes(_zip(members), "bundle.zip")

    # Bounded output, and it must NOT have sniffed all 40 members (early stop).
    assert len(out) <= 5_000 + 4_000, f"output not bounded: {len(out)}"
    assert calls["n"] < 40, (
        f"extractor decoded/sniffed all {calls['n']} members before truncating - "
        "the memory-amplification window is open")
    assert "truncated" in out.lower(), "a truncation note should mark the cut-off"


def test_compressed_tarball_extracts(monkeypatch):
    """MED-17: a real .tar.gz (and .tgz) extracts its members instead of raising
    'Unsupported file type'."""
    data = _targz({"a.txt": "alpha content", "b.txt": "beta content"})
    out = extract_bytes(data, "bundle.tgz")
    assert "alpha content" in out and "beta content" in out


def test_single_gzip_file_extracts_inner_text(monkeypatch):
    """MED-17: a single gzip-compressed non-tar file decompresses to its inner
    content instead of being mis-routed to the tar extractor and failing."""
    inner = b"this is a plain text file that was gzipped, not a tarball"
    gz = gzip.compress(inner)
    out = extract_bytes(gz, "notes.txt.gz")
    assert "plain text file that was gzipped" in out


def test_single_bzip2_file_extracts_inner_text(monkeypatch):
    inner = b"bzip2 compressed single text file, definitely not a tar"
    out = extract_bytes(bz2.compress(inner), "notes.txt.bz2")
    assert "not a tar" in out


def test_member_error_is_not_indexed_as_content(monkeypatch):
    """MED-21: a member that fails extraction must NOT have its error string
    folded into the returned (indexed) text."""
    # A file that sniffs as PDF (starts with %PDF-) but is not a valid PDF, so
    # _extract_pdf raises inside the member loop.
    members = {
        "good.txt": "genuinely readable content",
        "broken.pdf": b"%PDF-1.5\nnot really a pdf at all \x00\x01\x02",
    }
    out = extract_bytes(_zip(members), "bundle.zip")
    assert "genuinely readable content" in out
    assert "error:" not in out.lower(), \
        f"per-member error leaked into indexed text: {out!r}"
    assert "broken.pdf" not in out or "error" not in out.lower()
