"""Plain-text extraction from document files. Stdlib wherever possible."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

# Hard cap on extracted text per document — protects the chunker and the
# index from a runaway file. ~8 MB of text is far beyond any useful context.
MAX_TEXT_CHARS = 8_000_000

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
    """Extract plain text from in-memory file content (chat attachments) —
    nothing is written to disk, so privacy mode stays trace-free."""
    suffix = Path(filename).suffix.lower()

    if suffix in _PLAIN_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
    elif suffix in (".html", ".htm"):
        from localm.netpolicy import html_to_text
        text = html_to_text(data.decode("utf-8", errors="replace"))
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


def _extract_docx(data: bytes, filename: str) -> str:
    """.docx is a zip; the body lives in word/document.xml. Paragraph tags
    (<w:p>) become newlines, text runs (<w:t>) are concatenated — no
    python-docx needed."""
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        raise ExtractError(f"Cannot parse {filename} as .docx: {e}")
    # Tabs and explicit breaks inside runs
    xml = re.sub(r"<w:(?:tab|br|cr)\b[^>]*/?>", "\t", xml)
    paragraphs = []
    for para in re.findall(r"<w:p\b.*?</w:p>", xml, flags=re.DOTALL):
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
    parts = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if src.strip():
            parts.append(f"[cell {i} — {cell.get('cell_type', '?')}]\n{src}")
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
