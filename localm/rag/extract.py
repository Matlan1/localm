# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain-text extraction from document files. Stdlib wherever possible."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from typing import Callable, Optional
import io

# Hard cap on extracted text per document.
MAX_TEXT_CHARS = 8_000_000

# Hard cap on the DECOMPRESSED bytes read out of a zip container (.docx),
# enforced with a bounded stream read rather than the zip header's self-reported
# size.
MAX_ARCHIVE_MEMBER_BYTES = 80_000_000

# Hard cap on how many members an archive extractor will process, and the note
# emitted when either that cap or the whole-archive text budget is reached.
MAX_ARCHIVE_MEMBERS = 5_000
_ARCHIVE_TRUNCATED_NOTE = "[archive truncated: content budget reached]"

# Hard cap on the total bytes an archive extractor may INFLATE, across every
# member, whether or not that member yielded any text.
MAX_ARCHIVE_INFLATED_BYTES = 500_000_000

# Hard cap on how deeply extraction may descend into nested containers. The
# single-stream fallback (a bare .gz/.bz2/.xz that is not a tarball) re-enters
# extract_bytes on its decompressed inner bytes, so the descent is bounded here.
MAX_EXTRACT_DEPTH = 8

# Tar-family containers (plain tar plus single-stream gzip/bzip2/xz, which may be
# a compressed tarball or a single compressed file). All route to the tar-or-
# stream handler, which falls back to single-stream decompression when the
# payload is not actually a tar.
_TAR_LIKE_SUFFIXES = {".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz", ".txz"}

# Suffixes handled by extract_text. Anything not listed is refused.
_PLAIN_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".ps1", ".bat", ".sql", ".r", ".lua", ".xml", ".css",
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz", ".txz"}

EXTRACTABLE_SUFFIXES = _PLAIN_SUFFIXES | {".pdf", ".docx", ".html", ".htm", ".ipynb"} | _IMAGE_SUFFIXES | _ARCHIVE_SUFFIXES

# SECRET / key material: never indexed, whether via a folder walk or an
# explicitly-named API pick (confine_index_path with a policy). Covers PEM
# keys/certs, PKCS bundles, keystores, and formats that embed a private key
# inline (OpenVPN configs; direnv is handled by NAME in SECRET_INDEX_NAMES).
#
# Separate from UNINDEXABLE_SUFFIXES below: naming a secret is a refusal the
# caller is told about, while naming an unindexable file is an individual
# per-file failure that leaves the rest of the batch indexing.
SECRET_SUFFIXES = {
    ".pem", ".key", ".crt", ".cer", ".der", ".p12", ".pfx",
    ".keystore", ".jks", ".asc", ".gpg", ".kdbx",
    ".ppk", ".p8", ".pk8", ".pkcs12", ".p7b", ".p7c", ".ovpn",
}

# NON-SECRET files with no extractable text. Not a refusal: such a file is
# reported as an individual failure and the rest of the batch still indexes.
UNINDEXABLE_SUFFIXES = {
    # Executables / Binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".out", ".app", ".msi",
    # Unsupported Images (non-standard / non-multimodal target)
    ".bmp", ".ico", ".tiff",
    # Audio / Video
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    # Archives we don't want to expand as files directly (unsupported formats)
    ".7z", ".rar",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2",
    # Other binary data
    ".pyc", ".pyd", ".db", ".sqlite",
    # Model weights / large ML binaries, refused before the file is read.
    ".gguf", ".safetensors", ".pt", ".pth", ".onnx", ".ckpt", ".h5",
    ".pb", ".tflite", ".npz", ".npy", ".pkl",
}

# The union: everything a recursive folder walk filters out.
BLACKLISTED_SUFFIXES = SECRET_SUFFIXES | UNINDEXABLE_SUFFIXES

# Extensionless / dotfile secrets a recursive folder walk must skip, which the
# suffix blacklist cannot catch. confine_index_path (with a policy) applies it to
# explicit single-file picks too, so an API caller cannot name a key file
# directly; the local CLI (policy=None) is unconfined and still honours a pick.
SECRET_INDEX_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    ".netrc", ".pgpass", ".htpasswd", ".git-credentials",
    ".npmrc", ".pypirc", ".dockercfg", "credentials",
    ".envrc",   # direnv: a shell script that routinely exports secrets, like .env
}


# Config-doc TEMPLATES that are indexed rather than treated as secret. An
# allowlist: any .env* name not listed here is treated as secret.
SAFE_ENV_TEMPLATE_NAMES = {
    ".env.example",     # canonical; the one name upstream gitignore un-ignores
    ".env.template",
    ".env.sample",
    ".env.dist",
}


def is_secret_index_name(name: str) -> bool:
    """True for a filename that a recursive index walk should skip as likely
    secret material (extensionless keys, .env files, known credential files).

    A config-doc template in SAFE_ENV_TEMPLATE_NAMES is NOT secret - it is
    committed placeholder documentation. Checked FIRST so the .env* rule below
    cannot swallow it; everything the allowlist does not name stays secret.
    """
    low = name.lower()
    if low in SAFE_ENV_TEMPLATE_NAMES:
        return False
    if low in SECRET_INDEX_NAMES:
        return True
    if low == ".env" or low.startswith(".env."):
        return True
    return False

# Directories to skip when processing ZIP files (mirrors store.py _SKIP_DIRS)
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
              ".pytest_cache", ".mypy_cache", "dist", "build", ".idea",
              ".vscode"}

# Cache for the LLM format tie-break, keyed by unknown extension, so a corpus of
# same-extension files classifies at most once per process.
_EXT_CLASSIFICATION_CACHE: dict[str, str] = {}

# Canonical, lowercase format labels for the file types localm indexes. A known
# extension decides; the structural sniff below runs only for an unknown
# extension whose bytes decoded as text. The label feeds retrieval-filtering and
# display, not parsing.
_SUFFIX_FORMAT = {
    ".txt": "text", ".log": "text", ".rst": "text",
    ".md": "markdown", ".markdown": "markdown",
    ".csv": "csv", ".tsv": "csv",
    ".json": "json", ".jsonl": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini",
    ".xml": "xml", ".html": "html", ".htm": "html", ".css": "css",
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".sh": "shell", ".ps1": "powershell", ".bat": "batch",
    ".sql": "sql", ".r": "r", ".lua": "lua",
    ".pdf": "pdf", ".docx": "docx", ".ipynb": "notebook",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".webp": "image", ".gif": "image",
    ".zip": "archive", ".tar": "archive", ".gz": "archive", ".bz2": "archive",
    ".xz": "archive", ".tgz": "archive", ".tbz": "archive", ".txz": "archive",
}


# The structural sniff reads a bounded prefix only; JSON larger than
# _JSON_PARSE_MAX is confirmed by matching the closing bracket instead of parsed.
_SNIFF_PREFIX = 65_536
_JSON_PARSE_MAX = 1_000_000
# Tag names that mark HTML rather than generic XML. Single-letter/ambiguous names
# are excluded, so a real XML element does not read as HTML.
_HTML_TAG_RE = re.compile(
    r"</?(?:div|span|body|table|tr|td|th|ul|ol|li|h[1-6]|section|article|"
    r"head|title|nav|header|footer|button|form|input|img|p)\b")


def sniff_text_format(text: str) -> Optional[str]:
    """Best-effort STRUCTURAL format label for already-decoded text, using only
    cheap deterministic shape over a bounded prefix - no model call, no network,
    no stall. Returns a lowercase label when the structure is unambiguous, or
    ``None`` when unsure so the caller can fall back (to the extension, a gated
    LLM tie-break, or "text").
    """
    s = text.lstrip()
    if not s:
        return None
    head = s[:1]

    # JSON: parse when the document is small; for a large one, match the closing
    # bracket instead.
    if head in "{[":
        if len(s) <= _JSON_PARSE_MAX:
            try:
                json.loads(s)
                return "json"
            except Exception:
                pass
        else:
            close = "}" if head == "{" else "]"
            if s.rstrip()[-1:] == close:
                return "json"

    prefix = s[:_SNIFF_PREFIX]

    # HTML / XML: a leading angle bracket. An explicit XML declaration is
    # definitive; otherwise recognisable HTML tag names (incl. bare fragments)
    # beat the generic-tag fall-through to XML.
    if head == "<":
        low = prefix.lower()
        if low.startswith("<?xml"):
            return "xml"
        if "<!doctype html" in low or "<html" in low or _HTML_TAG_RE.search(low):
            return "html"
        if re.match(r"<[a-z][\w:.-]*[\s/>]", low):
            return "xml"

    # The remaining line-shape heuristics look at the first handful of non-blank
    # lines of the prefix only.
    lines = [ln for ln in prefix.splitlines() if ln.strip()][:20]

    # INI / TOML: a "[section]" header plus at least one "key = value" line.
    if head == "[" and any(re.match(r"\[[^\]]+\]\s*$", ln.strip()) for ln in lines[:5]):
        if any("=" in ln for ln in lines):
            return "ini"

    # CSV: a consistent, non-zero comma count across the first rows, plus either
    # >=3 columns or >=3 rows, and no code/prose punctuation in a row.
    if len(lines) >= 2 and "," in lines[0]:
        rows = lines[:10]
        counts = [ln.count(",") for ln in rows]
        looks_code = any(ch in ln for ln in rows for ch in "(){};")
        if (counts[0] >= 1 and len(set(counts)) == 1
                and (counts[0] >= 2 or len(rows) >= 3) and not looks_code):
            return "csv"

    # Markdown: an ATX heading at the very top, corroborated by another markdown
    # marker.
    if lines and re.match(r"#{1,6}\s+\S", lines[0]):
        corroborated = (
            any(re.match(r"#{1,6}\s+\S", ln) for ln in lines[1:])
            or any(ln.lstrip().startswith(("- ", "* ", "```", "> ")) for ln in lines)
            or "](" in prefix
        )
        if corroborated:
            return "markdown"

    # YAML: "key: value" block-mapping lines must be the dominant shape and carry
    # no assignment/call punctuation.
    kv = [ln for ln in lines
          if re.match(r"[A-Za-z0-9_.-]+\s*:(\s+\S|\s*$)", ln)
          and not any(ch in ln for ch in "={(")]
    if len(kv) >= 2 and len(kv) >= 0.6 * len(lines):
        return "yaml"

    return None


def _normalise_label(guess: str) -> str:
    """Reduce a raw LLM classification to a single clean lowercase token."""
    g = (guess or "").strip().lower().replace("`", "")
    parts = g.split()
    g = parts[0] if parts else ""
    return re.sub(r"[^a-z0-9_+#.-]", "", g)[:32]


def classify_format(text: str, filename: str = "", *,
                    classify_fn: Optional[Callable[[str], Optional[str]]] = None) -> str:
    """Return a short, lowercase format label for an already-extracted document.

    Free and deterministic first: a KNOWN extension is authoritative, then a
    structural content sniff (:func:`sniff_text_format`). Only when BOTH are
    inconclusive is *classify_fn* (an LLM tie-break) consulted - and only when the
    user left ``rag_classify_unknown_files`` on. *classify_fn* must itself be a
    no-op when no chat model is loaded (see the rag plugin's ``_make_self_classify``),
    so an embedding-only index never stalls on a chat call. Always returns a label
    (falling back to "text") so every chunk can carry one for filtering / display.
    """
    if not text.strip():
        return "text"
    suffix = Path(filename).suffix.lower()
    known = _SUFFIX_FORMAT.get(suffix)
    if known:
        return known
    sniffed = sniff_text_format(text)
    if sniffed:
        return sniffed
    # Unknown extension and inconclusive structure: the only place a model is
    # consulted, and only when the toggle is on. The outcome (a real guess, or the
    # "text" fallback) is cached per extension, so a same-extension corpus
    # attempts the tie-break at most once per process.
    if classify_fn is not None:
        cached = _EXT_CLASSIFICATION_CACHE.get(suffix)
        if cached is not None:
            return cached
        try:
            from localm.config import load_config
            enabled = bool(load_config().get("rag_classify_unknown_files", True))
        except Exception as e:
            # Config unreadable: fall back to the default (feature on) and log it.
            from localm.debuglog import logger as _dbg
            _dbg.debug("rag classify_format: could not load config, assuming "
                       "rag_classify_unknown_files=on: %s", e)
            enabled = True
        if enabled:
            label = _normalise_label(classify_fn(text[:1000]) or "") or "text"
            _EXT_CLASSIFICATION_CACHE[suffix] = label
            return label
    return "text"


def sniff_format(data: bytes, filename: str) -> Optional[str]:
    """Sniff the format of a file based on its magic bytes/content.
    Returns the mapped extension (e.g. '.pdf', '.docx', '.html', '.ipynb', '.txt', '.zip')
    or None if it appears to be binary and unsupported.
    """
    if not data:
        return ".txt"  # empty file behaves as text

    # 1. Signature checks
    if data.startswith(b"%PDF-"):
        return ".pdf"

    if data.startswith(b"PK\x03\x04"):
        # Could be .docx or a general .zip
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if "word/document.xml" in names:
                    return ".docx"
                return ".zip"
        except Exception:
            return None  # corrupt zip

    # Images
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpeg"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return ".webp"

    # Tar archives (uncompressed, or compressed with gzip/bzip2/xz)
    if len(data) >= 262 and data[257:262] == b"ustar":
        return ".tar"
    if data.startswith(b"\x1f\x8b"):
        return ".tar"
    if data.startswith(b"BZh"):
        return ".tar"
    if data.startswith(b"\xfd7zXZ\x00"):
        return ".tar"

    # HTML sniff
    sample_len = min(len(data), 1024)
    sample = data[:sample_len]
    sample_lower = sample.lower()
    if b"<html" in sample_lower or b"<!doctype html" in sample_lower:
        return ".html"

    # JSON / ipynb sniff
    stripped = sample.strip()
    if stripped.startswith((b"{", b"[")):
        try:
            parsed = json.loads(data.decode("utf-8", errors="ignore"))
            if isinstance(parsed, dict) and "cells" in parsed:
                return ".ipynb"
            return ".json"
        except Exception:
            pass

    # 2. Text check (avoid binary files)
    if b"\x00" in sample:
        return None

    # Check non-printable control characters ratio (excluding tab/cr/lf)
    control_count = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    if control_count > sample_len * 0.02:
        return None

    # Try decoding
    try:
        data.decode("utf-8")
        return ".txt"
    except UnicodeDecodeError:
        try:
            data.decode("cp1252")
            return ".txt"
        except UnicodeDecodeError:
            return None


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
    # BOM-less UTF-16 is sniffed BEFORE trying UTF-8: a NUL-heavy sample is
    # decoded as UTF-16 of the matching endianness.
    sample = data[:512]
    if sample and sample.count(0) > len(sample) // 4:
        endian = "utf-16-be" if sample[0:1] == b"\x00" else "utf-16-le"
        return data.decode(endian, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Legacy single-byte encoding: cp1252 maps every byte.
        return data.decode("cp1252", errors="replace")


def extract_text(path: Path,
                 describe_image_fn: Optional[Callable[[bytes, str], Optional[str]]] = None) -> str:
    """Return the plain text of *path*. Raises ExtractError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ExtractError(f"Not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ExtractError(f"Cannot read {path.name}: {e}")
    return extract_bytes(data, path.name, describe_image_fn=describe_image_fn)


def extract_bytes(data: bytes, filename: str,
                  describe_image_fn: Optional[Callable[[bytes, str], Optional[str]]] = None,
                  *, _depth: int = 0) -> str:
    """Extract plain text from in-memory file content (chat attachments) -
    nothing is written to disk, so privacy mode stays trace-free.

    ``_depth`` is an INTERNAL recursion counter: the archive/compression handlers
    re-enter this function on contained or decompressed bytes with ``_depth + 1``,
    and a value past ``MAX_EXTRACT_DEPTH`` is refused (a nested-container bomb).
    Callers never pass it."""
    if _depth > MAX_EXTRACT_DEPTH:
        raise ExtractError(
            f"{filename}: nested containers/compression too deep "
            f"(>{MAX_EXTRACT_DEPTH}); refusing to extract (possible "
            "decompression bomb).")
    suffix = Path(filename).suffix.lower()

    # Determine type using sniffer if suffix is not known/supported
    inferred_ext = suffix
    if suffix not in EXTRACTABLE_SUFFIXES:
        inferred = sniff_format(data, filename)
        if not inferred:
            raise ExtractError(
                f"Unsupported or binary file format for '{suffix}' ({filename}). Supported: "
                + ", ".join(sorted(EXTRACTABLE_SUFFIXES)))
        inferred_ext = inferred

    if inferred_ext == ".zip":
        text = _extract_zip(data, filename, describe_image_fn, _depth=_depth)
    elif inferred_ext in _TAR_LIKE_SUFFIXES:
        text = _extract_tar_or_stream(data, filename, describe_image_fn, _depth=_depth)
    elif inferred_ext in _IMAGE_SUFFIXES:
        if not describe_image_fn:
            raise ExtractError(
                f"No extractable text in {filename}. "
                "To index images, load a vision-capable model/projector."
            )
        mime_type = "image/png"
        if inferred_ext == ".jpg" or inferred_ext == ".jpeg":
            mime_type = "image/jpeg"
        elif inferred_ext == ".gif":
            mime_type = "image/gif"
        elif inferred_ext == ".webp":
            mime_type = "image/webp"

        try:
            desc = describe_image_fn(data, mime_type)
        except Exception as e:
            raise ExtractError(f"Image description failed: {e}")
        if not desc or not desc.strip():
            raise ExtractError(f"Active model returned empty description for {filename}")
        text = desc
    elif inferred_ext in _PLAIN_SUFFIXES:
        # The format label is derived separately by classify_format() at index
        # time and carried into chunk metadata.
        text = _decode_text(data)
    elif inferred_ext in (".html", ".htm"):
        from localm.netpolicy import html_to_text
        text = html_to_text(_decode_text(data))
    elif inferred_ext == ".docx":
        text = _extract_docx(data, filename)
    elif inferred_ext == ".ipynb":
        text = _extract_ipynb(data, filename)
    elif inferred_ext == ".pdf":
        text = _extract_pdf(data, filename)
    else:
        raise ExtractError(
            f"Unsupported file type '{suffix}' ({filename}). Supported: "
            + ", ".join(sorted(EXTRACTABLE_SUFFIXES)))

    text = text.strip()
    if not text:
        raise ExtractError(f"No extractable text in {filename}")
    return text[:MAX_TEXT_CHARS]


def _archive_log():
    from localm.debuglog import logger as _dbg
    return _dbg


def _archive_budget() -> int:
    """How many chars an archive extractor may accumulate before stopping. Leaves
    room for the truncation note so it survives the outer MAX_TEXT_CHARS cap."""
    return max(0, MAX_TEXT_CHARS - len(_ARCHIVE_TRUNCATED_NOTE) - 4)


def _join_archive(texts: list, truncated: bool) -> str:
    out = "\n\n".join(texts)
    if truncated:
        cap = _archive_budget()
        if len(out) > cap:
            out = out[:cap]
        out = (out + "\n\n" + _ARCHIVE_TRUNCATED_NOTE) if out else _ARCHIVE_TRUNCATED_NOTE
    return out


def _extract_zip(data: bytes, filename: str,
                 describe_image_fn: Optional[Callable[[bytes, str], Optional[str]]] = None,
                 *, _depth: int = 0) -> str:
    """Extract and merge text from a ZIP archive, BOUNDED in total output and
    member count so a many-member archive cannot amplify into a RAM DoS.
    Per-member read failures are logged, not folded into the indexed text."""
    import io
    texts: list = []
    total = 0
    processed = 0
    inflated = 0
    truncated = False
    limit = MAX_ARCHIVE_MEMBER_BYTES
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in sorted(zf.namelist()):
                if member.endswith("/") or any(part in _SKIP_DIRS or part.startswith(".") for part in Path(member).parts):
                    continue
                if total >= _archive_budget() or processed >= MAX_ARCHIVE_MEMBERS:
                    truncated = True
                    break
                processed += 1
                try:
                    with zf.open(member) as fh:
                        # Clamped to what is left of the whole-archive inflation
                        # budget, so the final member cannot overshoot it.
                        member_data = fh.read(min(limit, MAX_ARCHIVE_INFLATED_BYTES - inflated) + 1)
                    # Charge and break BEFORE the per-member limit check below,
                    # which skips an oversized member that was already inflated.
                    inflated += len(member_data)
                    if inflated >= MAX_ARCHIVE_INFLATED_BYTES:
                        truncated = True
                        _archive_log().warning(
                            "rag: %s exceeded the whole-archive decompressed-size "
                            "budget (%d MB); stopped early and truncated the text",
                            filename, MAX_ARCHIVE_INFLATED_BYTES // 1_000_000)
                        break
                    if len(member_data) > limit:
                        _archive_log().warning("rag: archive member %s in %s exceeds the "
                                               "decompressed-size limit; skipped", member, filename)
                        continue
                    inferred = sniff_format(member_data, member)
                    # Skip nested archives (zip / tar-family) to avoid loops.
                    if inferred and inferred != ".zip" and inferred not in _TAR_LIKE_SUFFIXES:
                        txt = extract_bytes(member_data, member, describe_image_fn,
                                            _depth=_depth + 1)
                        if txt.strip():
                            block = f"[file: {member}]\n{txt}"
                            texts.append(block)
                            total += len(block)
                except Exception as e:
                    _archive_log().debug("rag: could not read archive member %s in %s: %s",
                                         member, filename, e)
    except Exception as e:
        raise ExtractError(f"Cannot parse {filename} as zip: {e}")
    return _join_archive(texts, truncated)


def _extract_tar_or_stream(data: bytes, filename: str,
                           describe_image_fn: Optional[Callable[[bytes, str], Optional[str]]] = None,
                           *, _depth: int = 0) -> str:
    """Extract a tar-family payload. Handles plain and compressed TARBALLS
    (.tar/.tgz/.tbz/.txz/.tar.gz) via tarfile; when the payload is a SINGLE
    gzip/bzip2/xz-compressed file rather than a tar, decompresses that one
    stream and extracts its inner content.

    The single-stream branch RECURSES into extract_bytes on the decompressed
    bytes, so it passes ``_depth + 1`` - a nested .gz.gz.gz... is bounded by
    MAX_EXTRACT_DEPTH instead of recursing until RecursionError."""
    import tarfile
    import io
    import zlib
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
    except (tarfile.ReadError, EOFError, zlib.error):
        # ReadError, EOFError and zlib.error all mean "unreadable as a tar" and
        # take the same fallback: _decompress_single_stream re-reads the bytes as
        # a single stream and raises a per-file ExtractError if that fails too.
        # Anything else propagates.
        inner = _decompress_single_stream(data, filename)
        return extract_bytes(inner, _strip_compression_suffix(filename),
                             describe_image_fn, _depth=_depth + 1)
    try:
        return _extract_tar_members(tf, filename, describe_image_fn, _depth=_depth)
    finally:
        tf.close()


def _extract_tar_members(tf, filename: str, describe_image_fn, *, _depth: int = 0) -> str:
    texts: list = []
    total = 0
    inflated = 0
    truncated = False
    limit = MAX_ARCHIVE_MEMBER_BYTES
    # Collect at most MAX_ARCHIVE_MEMBERS headers by iterating the tar LAZILY, so
    # header parsing stops at the cap; only that bounded subset is sorted, for
    # deterministic order.
    members: list = []
    try:
        for member in tf:
            if len(members) >= MAX_ARCHIVE_MEMBERS:
                truncated = True
                break
            members.append(member)
    except Exception as e:
        raise ExtractError(f"Cannot parse {filename} as tar archive: {e}")
    members.sort(key=lambda m: m.name)
    try:
        for member in members:
            if not member.isfile():
                continue
            if any(part in _SKIP_DIRS or part.startswith(".") for part in Path(member.name).parts):
                continue
            if total >= _archive_budget():
                truncated = True
                break
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                # Same whole-archive inflation budget as _extract_zip, charged and
                # broken on BEFORE the per-member check below, which skips a
                # member that was already inflated.
                member_data = f.read(min(limit, MAX_ARCHIVE_INFLATED_BYTES - inflated) + 1)
                inflated += len(member_data)
                if inflated >= MAX_ARCHIVE_INFLATED_BYTES:
                    truncated = True
                    _archive_log().warning(
                        "rag: %s exceeded the whole-archive decompressed-size "
                        "budget (%d MB); stopped early and truncated the text",
                        filename, MAX_ARCHIVE_INFLATED_BYTES // 1_000_000)
                    break
                if len(member_data) > limit:
                    _archive_log().warning("rag: archive member %s in %s exceeds the "
                                           "decompressed-size limit; skipped", member.name, filename)
                    continue
                inferred = sniff_format(member_data, member.name)
                if inferred and inferred != ".zip" and inferred not in _TAR_LIKE_SUFFIXES:
                    txt = extract_bytes(member_data, member.name, describe_image_fn,
                                        _depth=_depth + 1)
                    if txt.strip():
                        block = f"[file: {member.name}]\n{txt}"
                        texts.append(block)
                        total += len(block)
            except Exception as e:
                _archive_log().debug("rag: could not read archive member %s in %s: %s",
                                     member.name, filename, e)
    except Exception as e:
        raise ExtractError(f"Cannot parse {filename} as tar archive: {e}")
    return _join_archive(texts, truncated)


def _decompress_single_stream(data: bytes, filename: str) -> bytes:
    """Decompress a single gzip/bzip2/xz stream with a BOUNDED read (bomb guard)."""
    import io
    import gzip
    import bz2
    import lzma
    limit = MAX_ARCHIVE_MEMBER_BYTES
    if data.startswith(b"\x1f\x8b"):
        fh = gzip.GzipFile(fileobj=io.BytesIO(data))
    elif data.startswith(b"BZh"):
        fh = bz2.BZ2File(io.BytesIO(data))
    elif data.startswith(b"\xfd7zXZ\x00"):
        fh = lzma.LZMAFile(io.BytesIO(data))
    else:
        raise ExtractError(f"{filename}: not a tar and not a recognised compressed stream")
    try:
        with fh:
            raw = fh.read(limit + 1)
    except Exception as e:
        raise ExtractError(f"{filename}: could not decompress: {e}")
    if len(raw) > limit:
        raise ExtractError(f"{filename}: decompressed content exceeds "
                           f"{limit // 1_000_000} MB limit (possible bomb); refusing to extract.")
    return raw


def _strip_compression_suffix(filename: str) -> str:
    low = filename.lower()
    for ext in (".tgz", ".tbz", ".txz"):
        if low.endswith(ext):
            return filename[:-len(ext)] + ".tar"
    for ext in (".gz", ".bz2", ".xz"):
        if low.endswith(ext):
            return filename[:-len(ext)]
    return filename


def _read_zip_member(zf: zipfile.ZipFile, member: str, filename: str) -> str:
    """Read one zip member as UTF-8 text with a HARD cap on the DECOMPRESSED
    size (zip-bomb guard). The compressed upload is capped upstream, but a
    zip's deflate ratio is ~1000x, so a 1 MB upload can decompress to
    gigabytes; a bounded STREAM read - not trusting the header's self-reported
    ZipInfo.file_size, which an attacker controls - is what prevents the
    amplification from exhausting RAM."""
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
    # Tabs and explicit breaks inside runs. The attribute span excludes both < and
    # >, so a match stops at the next tag instead of scanning past it.
    xml = re.sub(r"<w:(?:tab|br|cr)\b[^<>]*/?>", "\t", xml)
    # Split on the paragraph END tag with str.split; the inner <w:t> match is then
    # bounded to a single paragraph chunk.
    paragraphs = []
    for para in xml.split("</w:p>"):
        # Run extraction. The content group excludes <, and the attribute span
        # excludes both < and >, so each match stops at the next tag.
        runs = re.findall(r"<w:t\b[^<>]*>([^<]*)</w:t>", para)
        if runs:
            paragraphs.append(_unescape_xml("".join(runs)))
    return "\n\n".join(paragraphs)


def _unescape_xml(s: str) -> str:
    import html
    return html.unescape(s)


def _extract_ipynb(data: bytes, filename: str) -> str:
    try:
        nb = json.loads(data.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, ValueError, RecursionError) as e:
        # RecursionError from deeply-nested JSON is folded into ExtractError too:
        # it is a RuntimeError, not a JSONDecodeError.
        raise ExtractError(f"Cannot parse {filename} as a notebook: {type(e).__name__}")
    # A notebook is a JSON object with a "cells" list. Validate the shape and
    # coerce, so a malformed file raises ExtractError rather than TypeError.
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
        # "source" is normally a list of line strings; a bare string or other JSON
        # is coerced.
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
