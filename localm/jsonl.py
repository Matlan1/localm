"""Reading and writing newline-delimited JSON (JSONL) without losing records.

JSONL is delimited by LINE FEED and nothing else. Python's ``str.splitlines()``
splits on considerably more: U+000B, U+000C, U+001C, U+001D, U+001E, U+0085
(NEL), U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR).

``json.dumps(..., ensure_ascii=False)`` escapes the C0 control characters, but
U+0085, U+2028 and U+2029 are NOT JSON control characters, so they are written
RAW inside the string. A ``splitlines()`` reader then tears that one record into
two fragments, both of which fail to parse.

This is not hypothetical. Measured 2026-07-28 against a real user collection
(``rag/Eng``): 36 raw U+0085 characters turned 1192 valid records into 1228
"lines", 62 of which failed to parse. The consequences cascaded, and every one
of them looked like a different bug:

  1. the reader dropped 26 records (1192 -> 1166) and flagged the file corrupt;
  2. the vector sidecar (1192 entries) no longer matched the chunk count, so it
     was judged stale and semantic search silently degraded to lexical BM25;
  3. the good vector index was quarantined as ``vectors.json.rejected``, and the
     copies accumulated;
  4. the next ordinary save rewrote the file from the 1166 survivors, PERMANENTLY
     deleting 26 of the user's chunks - silent data loss on every load/save cycle.

So: split on LINE FEED only, and escape the separators on write so the files are
also safe for any other reader (U+2028/U+2029 additionally break JavaScript
``JSON.parse`` consumers). Both halves matter - the read fix recovers files
written before this module existed, the write fix stops producing new ones.
"""
import json
from typing import Any, Iterator

__all__ = ["split_jsonl", "iter_jsonl", "dumps_line", "dumps_lines", "UNSAFE_SEPARATORS"]

#: Characters ``str.splitlines()`` treats as line breaks, mapped to their JSON
#: escapes.
UNSAFE_SEPARATORS = {
    "\x0b": "\\u000b",
    "\x0c": "\\u000c",
    "\x1c": "\\u001c",
    "\x1d": "\\u001d",
    "\x1e": "\\u001e",
    "\x85": "\\u0085",
    " ": "\\u2028",
    " ": "\\u2029",
}


def split_jsonl(text: str) -> list:
    """Split JSONL *text* into records on LINE FEED only.

    A trailing carriage return (a CRLF-written file) is stripped, so a record is
    handed to ``json.loads`` identically on every platform. Blank records are
    preserved rather than filtered, so callers keep their own line numbering for
    error reporting; they typically skip falsy entries.
    """
    return [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]


def iter_jsonl(text: str) -> Iterator[tuple]:
    """Yield ``(lineno, raw)`` for each non-blank record, 1-based.

    The line number counts real newline-delimited records, so it matches both
    what a person sees in an editor and what ``split_jsonl`` produced.
    """
    for i, raw in enumerate(split_jsonl(text), 1):
        if raw.strip():
            yield i, raw


def dumps_line(obj: Any) -> str:
    """``json.dumps(obj, ensure_ascii=False)`` with every line-break-alike escaped.

    ``ensure_ascii=False`` is kept deliberately: these files hold user text, and
    escaping every non-ASCII character would roughly double the size of a
    non-English collection and make the file unreadable. Only the characters that
    can actually break a line-oriented reader are escaped.

    The replacements are safe to apply to the whole serialized record: none of
    these characters is JSON syntax, so any occurrence is necessarily inside a
    string literal, where the escape is exactly equivalent.
    """
    s = json.dumps(obj, ensure_ascii=False)
    for raw, esc in UNSAFE_SEPARATORS.items():
        if raw in s:
            s = s.replace(raw, esc)
    return s


def dumps_lines(objs) -> str:
    """Serialize *objs* into a JSONL body (no trailing newline)."""
    return "\n".join(dumps_line(o) for o in objs)
