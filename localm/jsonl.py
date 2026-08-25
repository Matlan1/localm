"""Reading and writing newline-delimited JSON (JSONL) without losing records."""
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
    """Split JSONL *text* into records on LINE FEED only."""
    return [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]


def iter_jsonl(text: str) -> Iterator[tuple]:
    """Yield ``(lineno, raw)`` for each non-blank record, 1-based."""
    for i, raw in enumerate(split_jsonl(text), 1):
        if raw.strip():
            yield i, raw


def dumps_line(obj: Any) -> str:
    """``json.dumps(obj, ensure_ascii=False)`` with every line-break-alike escaped."""
    s = json.dumps(obj, ensure_ascii=False)
    for raw, esc in UNSAFE_SEPARATORS.items():
        if raw in s:
            s = s.replace(raw, esc)
    return s


def dumps_lines(objs) -> str:
    """Serialize *objs* into a JSONL body (no trailing newline)."""
    return "\n".join(dumps_line(o) for o in objs)
