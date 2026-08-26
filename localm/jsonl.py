"""Reading and writing newline-delimited JSON (JSONL) without losing records.

JSONL is delimited by LINE FEED and nothing else. Python's ``str.splitlines()``
splits on considerably more: U+000B, U+000C, U+001C, U+001D, U+001E, U+0085
(NEL), U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR).

``json.dumps(..., ensure_ascii=False)`` escapes the C0 control characters, but
U+0085, U+2028 and U+2029 are NOT JSON control characters, so they are written
RAW inside the string. A ``splitlines()`` reader then tears that one record into
two fragments, both of which fail to parse.

So: this module splits on LINE FEED only, and escapes the separators on write,
which also keeps the files safe for JavaScript ``JSON.parse`` consumers
(U+2028/U+2029 break those too).
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
    preserved, not filtered, so callers keep their own line numbering for
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

    ``ensure_ascii=False`` keeps non-ASCII user text verbatim; only the
    characters that can break a line-oriented reader are escaped.

    The replacements apply to the whole serialized record: none of these
    characters is JSON syntax, so any occurrence is inside a string literal,
    where the escape is exactly equivalent.
    """
    s = json.dumps(obj, ensure_ascii=False)
    for raw, esc in UNSAFE_SEPARATORS.items():
        if raw in s:
            s = s.replace(raw, esc)
    return s


def dumps_lines(objs) -> str:
    """Serialize *objs* into a JSONL body (no trailing newline)."""
    return "\n".join(dumps_line(o) for o in objs)
