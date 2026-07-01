# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain-text extraction from document files. Stdlib wherever possible."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

# Hard cap on extracted text per document - protects the chunker and the
# index from a runaway file. ~8 MB of text is far beyond any useful context.
MAX_TEXT_CHARS = 8_000_000

# Hard cap on the DECOMPRESSED bytes we will pull out of a zip container
# (.docx). The upload route caps the COMPRESSED payload (~30 MB), but a zip's
# deflate ratio is ~1000x, so a 1 MB upload can decompress to gigabytes and
# exhaust RAM (a "zip bomb" DoS) before the text cap above ever applies. 80 MB
# of decompressed XML is far more than any real document needs (the extracted
# text is itself capped at MAX_TEXT_CHARS) while making the amplification
# attack impossible. We enforce it with a BOUNDED stream read, not by trusting
# the zip header's self-reported size (which an attacker controls).
MAX_ARCHIVE_MEMBER_BYTES = 80_000_000

# Suffixes handled by extract_text. Anything not listed is refused (binary
# formats would poison the index with mojibake).
_PLAIN_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".ps1", ".bat", ".sql", ".r", ".lua", ".xml", ".css",
}

EXTRACTABLE_SUFFIXES = _PLAIN_SUFFIXES | {".pdf", ".docx", ".html", ".htm", ".ipynb"}


class ExtractError(Exception):
    """A document could not be converted to text. The message says why and,
    where applicable, what to install."""


def _decode_text(data: bytes) -> str:
    """
    Decode a plain-text file without trusting it to be UTF-8.

    Windows editors routinely save "text files" as UTF-16; blindly decoding
    those as UTF-8 yields NUL-interleaved mojibake that an LLM cannot read -
    the attachment then looks present but carries no information.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")          # BOM, either endianness
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    # BOM-less UTF-16 must be sniffed BEFORE trying UTF-8: NUL is a valid
    # UTF-8 codepoint, so ASCII-range UTF-16 (every other byte NUL) decodes
    # as UTF-8 "successfully" into NUL-riddled garbage. Real text files
    # contain no NULs at all.
    sample = data[:512]
    if sample and sample.count(0) > len(sample) // 4:
        endian = "utf-16-be" if sample[0:1] == b"\x00" else "utf-16-le"
        return data.decode(endian, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Legacy single-byte encoding - cp1252 maps every byte, nothing is lost
        return data.decode("cp1252", errors="replace")


def extract_text(path: Path) -> str:
    """Return the plain text of *path*. Raises ExtractError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ExtractError(f"Not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ExtractError(f"Cannot read {path.name}: {e}")
    return extract_bytes(data, path.name)


def extract_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from in-memory file content (chat attachments) -
    nothing is written to disk, so privacy mode stays trace-free."""
    suffix = Path(filename).suffix.lower()

    if suffix in _PLAIN_SUFFIXES:
        text = _decode_text(data)
    elif suffix in (".html", ".htm"):
        from localm.netpolicy import html_to_text
        text = html_to_text(_decode_text(data))
    elif suffix == ".docx":
        text = _extract_docx(data, filename)
    elif suffix == ".ipynb":
        text = _extract_ipynb(data, filename)
    elif suffix == ".pdf":
        text = _extract_pdf(data, filename)
    else:
        raise ExtractError(
            f"Unsupported file type '{suffix}' ({filename}). Supported: "
            + ", ".join(sorted(EXTRACTABLE_SUFFIXES)))

    text = text.strip()
    if not text:
        raise ExtractError(f"No extractable text in {filename}")
    return text[:MAX_TEXT_CHARS]


def _read_zip_member(zf: zipfile.ZipFile, member: str, filename: str) -> str:
    """Read one zip member as UTF-8 text with a HARD cap on the DECOMPRESSED
    size (zip-bomb guard). The compressed upload is capped upstream, but a
    zip's deflate ratio is ~1000x, so a 1 MB upload can decompress to
    gigabytes; a bounded STREAM read - not trusting the header's self-reported
    ZipInfo.file_size, which an attacker controls - is what actually prevents
    the amplification from exhausting RAM (CWE-409)."""
    limit = MAX_ARCHIVE_MEMBER_BYTES
    with zf.open(member) as fh:            # ZipExtFile decompresses lazily on read
        raw = fh.read(limit + 1)          # bounded: at most limit+1 decompressed bytes
    if len(raw) > limit:
        raise ExtractError(
            f"{filename}: embedded '{member}' exceeds the "
            f"{limit // 1_000_000} MB decompressed-size limit "
            "(possible zip bomb); refusing to extract.")
    return raw.decode("utf-8", errors="replace")


def _extract_docx(data: bytes, filename: str) -> str:
    """.docx is a zip; the body lives in word/document.xml. Paragraph tags
    (<w:p>) become newlines, text runs (<w:t>) are concatenated - no
    python-docx needed."""
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = _read_zip_member(zf, "word/document.xml", filename)
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        raise ExtractError(f"Cannot parse {filename} as .docx: {e}")
    # Tabs and explicit breaks inside runs
    xml = re.sub(r"<w:(?:tab|br|cr)\b[^>]*/?>", "\t", xml)
    # Split on the paragraph END tag - a LINEAR str.split, not a backtracking
    # regex. The old `<w:p\b.*?</w:p>` findall was quadratic on malformed XML
    # with tens of thousands of unmatched <w:p openers (ReDoS: ~135s CPU on a
    # 50k-opener input). str.split is O(n); the inner <w:t> match is bounded to
    # a single paragraph chunk and cannot backtrack across paragraph boundaries.
    paragraphs = []
    for para in xml.split("</w:p>"):
        runs = re.findall(r"<w:t\b[^>]*>(.*?)</w:t>", para, flags=re.DOTALL)
        if runs:
            paragraphs.append(_unescape_xml("".join(runs)))
    return "\n\n".join(paragraphs)


def _unescape_xml(s: str) -> str:
    import html
    return html.unescape(s)


def _extract_ipynb(data: bytes, filename: str) -> str:
    try:
        nb = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ExtractError(f"Cannot parse {filename} as a notebook: {e}")
    # A notebook is a JSON object with a "cells" list, but an uploaded file may
    # be malformed (cells as a string, a cell as an int, source as a non-list).
    # Validate the shape and coerce defensively so a wrong type raises a clean
    # ExtractError -> 422, never an unhandled TypeError/AttributeError -> 500.
    if not isinstance(nb, dict):
        raise ExtractError(f"{filename} is not a valid notebook (expected a JSON object)")
    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ExtractError(f"{filename} is not a valid notebook ('cells' is not a list)")
    parts = []
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue                      # skip a malformed cell, do not crash
        source = cell.get("source", "")
        # "source" is normally a list of line strings, but hand-written or
        # malformed notebooks may store a bare string (or other JSON); coerce.
        if isinstance(source, list):
            src = "".join(str(s) for s in source)
        else:
            src = str(source)
        if src.strip():
            parts.append(f"[cell {i} - {cell.get('cell_type', '?')}]\n{src}")
    return "\n\n".join(parts)


def _extract_pdf(data: bytes, filename: str) -> str:
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ExtractError(
            "PDF support needs the pypdf package. Install it with: "
            "pip install \"localm[rag]\"  (or: pip install pypdf)")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(f"[page {i + 1}]\n{txt}")
        return "\n\n".join(pages)
    except Exception as e:
        raise ExtractError(f"Cannot extract text from {filename}: {e}")
