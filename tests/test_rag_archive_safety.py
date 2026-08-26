# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG archive-extraction safety. Three properties:

  * _extract_zip/_extract_tar cap each single member (80 MB), the number of
    members, AND the running accumulated output. Without the last two, a 30 MB
    zip of thousands of highly-compressible members builds ~30 GB in RAM before
    the whole-archive MAX_TEXT_CHARS truncation is applied.

  * Compressed-tar suffixes (.tgz/.tbz/.txz) have a dispatch branch, and a
    single gzip/bzip2/xz-compressed non-tar file is not routed to the tar
    extractor.

  * A per-member extraction error is surfaced/logged, never appended into the
    indexed text as "[file: X - error: ...]" where it becomes retrievable
    knowledge-base content.
"""

import bz2
import gzip
import io
import lzma
import tarfile
import zipfile

import pytest

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
    """The extractor stops once the accumulated-text budget is reached,
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
    """A real .tar.gz (and .tgz) extracts its members instead of raising
    'Unsupported file type'."""
    data = _targz({"a.txt": "alpha content", "b.txt": "beta content"})
    out = extract_bytes(data, "bundle.tgz")
    assert "alpha content" in out and "beta content" in out


def test_single_gzip_file_extracts_inner_text(monkeypatch):
    """A single gzip-compressed non-tar file decompresses to its inner
    content instead of being mis-routed to the tar extractor and failing."""
    inner = b"this is a plain text file that was gzipped, not a tarball"
    gz = gzip.compress(inner)
    out = extract_bytes(gz, "notes.txt.gz")
    assert "plain text file that was gzipped" in out


def test_single_bzip2_file_extracts_inner_text(monkeypatch):
    inner = b"bzip2 compressed single text file, definitely not a tar"
    out = extract_bytes(bz2.compress(inner), "notes.txt.bz2")
    assert "not a tar" in out


def test_nested_single_stream_compression_is_bounded():
    """A file that is a single compressed stream wrapped many times over
    (gzip(gzip(gzip(...)))) must be REFUSED with a clean ExtractError once the
    nesting passes the depth bound, NOT recurse unboundedly.

    The single-stream fallback in _extract_tar_or_stream re-enters extract_bytes
    on the decompressed inner bytes; with no depth bound a ~30 KB file nested
    past Python's recursion limit raises RecursionError - which is not an
    ExtractError and so escapes every `except ExtractError` guard in the add /
    upload / extract routes (a decompression-amplification DoS). The depth bound
    (MAX_EXTRACT_DEPTH) is small, so testing just past it stays fast (nesting
    deep enough to hit the recursion limit is O(n^2) to even build, because each
    level re-probes the whole nest as a possible tarball).

    Asserts the REFUSAL (pytest.raises): without the depth bound this shallow
    nesting extracts silently at every level, and a RecursionError does not
    match ExtractError either."""
    from localm.rag.extract import ExtractError, MAX_EXTRACT_DEPTH
    depth = MAX_EXTRACT_DEPTH + 5         # past the bound; small nest keeps it fast
    for name, comp in (("bomb.gz", gzip.compress),
                       ("bomb.bz2", bz2.compress),
                       ("bomb.xz", lzma.compress)):
        data = b"the innermost text file content\n" * 4
        for _ in range(depth):
            data = comp(data)
        assert len(data) < 200_000, f"{name}: nesting stayed bomb-shaped"
        with pytest.raises(ExtractError):
            extract_bytes(data, name)


def test_single_level_compression_still_extracts_after_depth_bound():
    """The depth bound must not break the legitimate one-level cases: a single
    gzip/bzip2/xz-wrapped text file, and a .tar.gz, still extract."""
    inner = b"a normal gzipped note, one level of compression only"
    assert "one level of compression only" in extract_bytes(
        gzip.compress(inner), "note.txt.gz")
    assert "one level of compression only" in extract_bytes(
        bz2.compress(inner), "note.txt.bz2")
    out = extract_bytes(_targz({"a.txt": "alpha inside tar"}), "bundle.tar.gz")
    assert "alpha inside tar" in out


def test_member_error_is_not_indexed_as_content(monkeypatch):
    """A member that fails extraction must NOT have its error string
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


# --------------------------------------------------------------------------- #
#  A whole-archive budget on bytes INFLATED, not just text produced.           #
# --------------------------------------------------------------------------- #
#
# The per-member cap (80 MB) and the member-count cap (5,000) MULTIPLY rather
# than compose, and the text budget only advances when a member YIELDS TEXT. A
# member that sniffs as binary is skipped and charges nothing, so the loop runs
# to the full member cap while still decompressing each one.
#
# These assert on BYTES ACTUALLY INFLATED, not on wall-clock time.

def _binary_bomb(count: int, size: int) -> dict:
    """The bomb shape: highly compressible NUL
    blocks named .bin, so sniff_format classifies them as binary and they are
    SKIPPED - contributing nothing to the text budget while each still costs a
    full decompression."""
    return {f"m{i:04d}.bin": b"\x00" * size for i in range(count)}


def _inflation_probe(monkeypatch) -> list:
    """Record the size of every member the extractor actually decompressed.

    sniff_format is called once per successfully-read member, with that member's
    DECOMPRESSED bytes, so summing its inputs measures inflation directly. The
    real function is still called, so routing behaviour is unchanged."""
    seen: list = []
    real_sniff = extract.sniff_format

    def counting_sniff(data, filename):
        seen.append(len(data))
        return real_sniff(data, filename)

    monkeypatch.setattr(extract, "sniff_format", counting_sniff)
    return seen


def _extract_ignoring_refusal(data: bytes, name: str) -> str:
    """Extract, treating a refusal as an empty result. An all-binary archive
    correctly ends in "no extractable text" - that refusal was never the defect,
    the work done to reach it was, and that is what these tests measure."""
    try:
        return extract_bytes(data, name)
    except Exception:
        return ""


@pytest.mark.parametrize("kind, build", [("zip", _zip), ("tar.gz", _targz)])
def test_binary_members_cannot_inflate_past_the_whole_archive_budget(
        monkeypatch, kind, build):
    cap = 5_000_000
    member = 1_000_000
    monkeypatch.setattr(extract, "MAX_ARCHIVE_INFLATED_BYTES", cap)
    seen = _inflation_probe(monkeypatch)

    # 40 MB of inflation available if nothing bounds it; the cap is 5 MB.
    _extract_ignoring_refusal(build(_binary_bomb(40, member)), f"bomb.{kind}")

    total = sum(seen)
    # One member of slack: the cap is checked AFTER the read that crosses it.
    assert total <= cap + member, (
        f"{kind}: inflated {total} bytes against a {cap}-byte whole-archive "
        "budget - the decompression-amplification window is open")
    assert len(seen) < 40, (
        f"{kind}: decompressed all {len(seen)} members; the budget never stopped it")


@pytest.mark.parametrize("kind, build", [("zip", _zip), ("tar.gz", _targz)])
def test_the_budget_marks_the_result_truncated(monkeypatch, kind, build):
    """Rule 5: hitting the budget must TELL the user the archive was cut short,
    not silently hand back partial text. Text members here, so there is a result
    to carry the note."""
    monkeypatch.setattr(extract, "MAX_ARCHIVE_INFLATED_BYTES", 2_000_000)
    members = {f"f{i:03d}.txt": ("x" * 500_000) for i in range(20)}
    out = _extract_ignoring_refusal(build(members), f"big.{kind}")
    assert "truncated" in out.lower(), f"{kind}: no truncation note on a cut-off archive"


@pytest.mark.parametrize("kind, build", [("zip", _zip), ("tar.gz", _targz)])
def test_an_ordinary_archive_is_untouched_by_the_budget(monkeypatch, kind, build):
    """No overcorrection. A normal small archive extracts every member in full
    and carries NO truncation note - the budget must be invisible in normal use."""
    seen = _inflation_probe(monkeypatch)
    members = {f"doc{i}.txt": f"hello from document {i}" for i in range(5)}
    out = _extract_ignoring_refusal(build(members), f"docs.{kind}")

    for i in range(5):
        assert f"hello from document {i}" in out, f"{kind}: lost member {i}"
    assert "truncated" not in out.lower(), f"{kind}: falsely marked truncated"
    assert len(seen) == 5, f"{kind}: expected all 5 members read, got {len(seen)}"
