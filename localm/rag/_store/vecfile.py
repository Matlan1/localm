# SPDX-License-Identifier: AGPL-3.0-or-later
"""vectors.json serialised and parsed a few rows at a time.

The file is ``{"dim": <int or null>, "vectors": [<row>, ...]}``. One
``json.dumps`` or ``json.loads`` over the whole file is a single C call that
holds the GIL for its full length, seconds on a large collection, which stops
the event loop and every other thread. These helpers produce and accept the
same bytes while returning to the interpreter between small batches of rows.
"""

from __future__ import annotations

import codecs
import json
import re
from pathlib import Path
from typing import Iterator, Optional

#: Rows serialised per ``json.dumps`` call.
_ROWS_PER_PIECE = 64

#: Bytes read per read when parsing; a retry for an unfinished row doubles it.
_READ_BYTES = 1 << 20

_HEADER_RE = re.compile(
    r'[ \t\n\r]*\{[ \t\n\r]*"dim"[ \t\n\r]*:[ \t\n\r]*(null|-?(?:0|[1-9][0-9]*))'
    r'[ \t\n\r]*,[ \t\n\r]*"vectors"[ \t\n\r]*:[ \t\n\r]*\[')
_WS_RE = re.compile(r"[ \t\n\r]*")
_DECODER = json.JSONDecoder()

#: Returned by ``_parse_rows`` for a file that is valid so far but not in the
#: written shape.
_OTHER_SHAPE = object()


def counted_vectors_json_pieces(dim: Optional[int],
                                rows: list) -> Iterator[tuple[str, int]]:
    """Yield (piece, rows written so far) for vectors.json of *dim* and
    *rows*; the pieces concatenate to exactly
    ``json.dumps({"dim": dim, "vectors": rows})``."""
    yield '{"dim": ' + json.dumps(dim) + ', "vectors": [', 0
    for start in range(0, len(rows), _ROWS_PER_PIECE):
        end = min(len(rows), start + _ROWS_PER_PIECE)
        body = json.dumps(rows[start:end])[1:-1]
        yield (body if start == 0 else ", " + body), end
    yield "]}", len(rows)


def vectors_json_pieces(dim: Optional[int], rows: list) -> Iterator[str]:
    """The pieces of ``counted_vectors_json_pieces`` alone."""
    for piece, _done in counted_vectors_json_pieces(dim, rows):
        yield piece


class _Reader:
    """A UTF-8 text buffer over a binary file, extended one read at a time."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self._dec = codecs.getincrementaldecoder("utf-8")()
        self.buf = ""
        self.eof = False

    def more(self, nbytes: int = _READ_BYTES) -> None:
        """Append the next *nbytes*, decoded, to ``buf``; sets ``eof`` at the
        end of the file."""
        raw = self._fh.read(nbytes)
        if raw:
            self.buf += self._dec.decode(raw)
        else:
            self.buf += self._dec.decode(b"", final=True)
            self.eof = True

    def drop(self, i: int) -> int:
        """Discard ``buf[:i]``; returns 0, the index the old *i* now has."""
        if i:
            self.buf = self.buf[i:]
        return 0

    def skip_ws(self, i: int) -> int:
        """Index of the first non-whitespace character at or after *i*,
        reading more as needed; ``len(buf)`` only at the end of the file."""
        while True:
            i = _WS_RE.match(self.buf, i).end()
            if i < len(self.buf) or self.eof:
                return i
            i = self.drop(i)
            self.more()

    def value(self, i: int):
        """(value, end) of the JSON value starting at *i*, reading more while
        the value may be unfinished. Raises json.JSONDecodeError at the end of
        the file."""
        nbytes = _READ_BYTES
        while True:
            try:
                value, end = _DECODER.raw_decode(self.buf, i)
            except json.JSONDecodeError:
                if self.eof:
                    raise
            else:
                if end < len(self.buf) or self.eof:
                    return value, end
            self.more(nbytes)
            nbytes *= 2


def _parse_rows(r: _Reader):
    """The parsed file when it is in the written shape, else ``_OTHER_SHAPE``.
    Raises json.JSONDecodeError for a row list that is not valid JSON."""
    while not r.eof and len(r.buf) < 4096:
        r.more()
    m = _HEADER_RE.match(r.buf)
    if m is None:
        return _OTHER_SHAPE
    dim = None if m.group(1) == "null" else int(m.group(1))
    rows: list = []
    i = r.skip_ws(m.end())
    if i < len(r.buf) and r.buf[i] == "]":
        i += 1
    else:
        while True:
            row, i = r.value(r.drop(i))
            rows.append(row)
            i = r.skip_ws(i)
            c = r.buf[i] if i < len(r.buf) else ""
            if c == "]":
                i += 1
                break
            if c != ",":
                raise json.JSONDecodeError("Expecting ',' delimiter", r.buf, i)
            i = r.skip_ws(i + 1)
    i = r.skip_ws(i)
    if i < len(r.buf) and r.buf[i] == "}" and r.skip_ws(i + 1) >= len(r.buf):
        return {"dim": dim, "vectors": rows}
    return _OTHER_SHAPE


def load_vectors_json(path: Path):
    """The parsed content of the vectors.json at *path*: the same value
    ``json.loads(path.read_text(encoding="utf-8"))`` returns.

    A file in the shape ``vectors_json_pieces`` writes is parsed one row per
    decoder call. Any other shape is parsed with one ``json.loads`` over the
    whole text.

    Raises OSError when the file cannot be read, and ValueError
    (``json.JSONDecodeError`` or ``UnicodeDecodeError``) when it is not valid
    UTF-8 JSON."""
    with open(path, "rb") as fh:
        parsed = _parse_rows(_Reader(fh))
    if parsed is not _OTHER_SHAPE:
        return parsed
    return json.loads(Path(path).read_text(encoding="utf-8"))
