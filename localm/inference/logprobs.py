# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token log probabilities on the OpenAI routes.

A backend that supports them hands the server one record per generated token,
``(token_bytes, logprob, ((alt_bytes, alt_logprob), ...))``, in order, ahead of
the text that holds the token. The server's text then goes through marker
scrubbing, the reasoning split, tool-call parsing and stop sequences, so the
visible reply is only part of what the tokens spelled. :class:`LogprobAligner`
maps a range of the text the server received back to the tokens it came from,
so a reply reports the tokens of its visible text and no others.

Pure Python: no engine, no web framework."""

from __future__ import annotations

import bisect
import codecs
from typing import Any, Optional

from localm.textnorm import ScrubMap

# The most alternatives one token may report, on both routes.
MAX_TOP_LOGPROBS = 20


class LogprobAlignmentError(RuntimeError):
    """The received text does not match the text the token records spell."""


class SegMap:
    """A piecewise map from a concatenated stream back to a source stream.

    :meth:`add` appends *length* characters copied from source offset
    *source*; :meth:`map` turns a range of the concatenation into the source
    ranges it was copied from, in order."""

    def __init__(self) -> None:
        self._dst: list[int] = []
        self._src: list[int] = []
        self._len: list[int] = []
        self.total = 0

    def add(self, source: int, length: int) -> None:
        if length <= 0:
            return
        if self._dst and self._src[-1] + self._len[-1] == source:
            self._len[-1] += length
        else:
            self._dst.append(self.total)
            self._src.append(source)
            self._len.append(length)
        self.total += length

    def map(self, start: int, end: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        i = max(0, bisect.bisect_right(self._dst, start) - 1)
        while start < end and i < len(self._dst):
            d0, s0, n = self._dst[i], self._src[i], self._len[i]
            lo, hi = max(start, d0), min(end, d0 + n)
            if lo < hi:
                if out and out[-1][1] == s0 + lo - d0:
                    out[-1] = (out[-1][0], s0 + hi - d0)
                else:
                    out.append((s0 + lo - d0, s0 + hi - d0))
            i += 1
        return out


class LogprobAligner:
    """Maps ranges of a reply's received text back to the token records that
    produced them.

    :meth:`add` takes token records as the backend delivers them,
    :meth:`received` each piece of text the server got from the engine, and
    :meth:`finish` marks the end of the records. :meth:`take` then returns, for
    ranges of the received text, the records of the tokens that spelled them,
    each token once and in order; a token that spelled only text outside every
    range passed so far is never returned. ``offset`` on a returned entry is
    where that token's text starts in the received text.

    A marker that scrubbing rewrote into visible text (a quote) is reported by
    the tokens that wrote the marker; the line breaks and think tags a
    reasoning marker is rewritten into are reported by none.

    The received text must be the token bytes decoded as UTF-8 and passed
    through :func:`localm.textnorm.scrub_text`; :meth:`take` raises
    :class:`LogprobAlignmentError` when it is not. Not thread-safe."""

    def __init__(self) -> None:
        self._records: list[tuple[bytes, float, tuple]] = []
        self._raw: list[str] = []
        self._raw_len = 0
        self._raw_text = ""
        self._first: list[int] = []
        self._last: list[int] = []
        self._start: list[int] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending: Optional[int] = None
        self._received: list[str] = []
        self._received_text = ""
        self._scrub = ScrubMap()
        self._next = 0

    def add(self, records: list) -> None:
        for record in records:
            index = len(self._records)
            self._records.append(record)
            self._decode(bytes(record[0]), index)

    def finish(self) -> None:
        """Decode the bytes still held for an incomplete character."""
        self._decode(b"", len(self._records) - 1, final=True)

    def _decode(self, data: bytes, index: int, final: bool = False) -> None:
        if not final:
            self._start.append(self._raw_len)
        chars = self._decoder.decode(data, final=final)
        if chars:
            first = self._pending if self._pending is not None else index
            self._first.append(first)
            self._last.append(index)
            self._first.extend([index] * (len(chars) - 1))
            self._last.extend([index] * (len(chars) - 1))
            self._raw.append(chars)
            self._raw_len += len(chars)
            self._pending = None
        if self._decoder.getstate()[0]:
            if self._pending is None:
                self._pending = index
        else:
            self._pending = None

    def received(self, piece: str) -> None:
        if piece:
            self._received.append(piece)

    def _texts(self) -> tuple[str, str]:
        if len(self._raw_text) != self._raw_len:
            self._raw_text = "".join(self._raw)
            self._raw = [self._raw_text]
        if self._received:
            self._received_text += "".join(self._received)
            self._received = []
        return self._raw_text, self._received_text

    def take(self, ranges: list[tuple[int, int]]) -> list[dict]:
        """The entries of the tokens not yet returned that spelled any of
        *ranges* (offsets into the received text), in order. Each entry holds
        ``bytes``, ``logprob``, ``top`` and ``offset``."""
        if not ranges:
            return []
        raw, got = self._texts()
        spans = self._scrub.spans(raw)
        if (self._pending is not None and ranges[-1][1] > (spans[-1][3] if spans else 0)):
            self.finish()
            raw, got = self._texts()
            spans = self._scrub.spans(raw)
        out_starts = [span[2] for span in spans]
        src_starts = [span[0] for span in spans]
        out: list[dict] = []
        for a, b in ranges:
            if a >= b:
                continue
            if b > len(got) or not spans or spans[-1][3] < b:
                raise LogprobAlignmentError(
                    f"range {a}..{b} is past the text received or spelled by the "
                    "generated tokens")
            i = max(0, bisect.bisect_right(out_starts, a) - 1)
            while i < len(spans):
                s0, s1, o0, o1, replacement = spans[i]
                if o0 >= b:
                    break
                lo, hi = max(a, o0), min(b, o1)
                if lo < hi:
                    expect = (raw[s0 + lo - o0:s0 + hi - o0] if replacement is None
                              else replacement[lo - o0:hi - o0])
                    if expect != got[lo:hi]:
                        raise LogprobAlignmentError(
                            f"received text at {lo}..{hi} does not match the "
                            "text the generated tokens spell")
                    if replacement is None:
                        self._emit(s0 + lo - o0, s0 + hi - o0, lo, spans, src_starts, out)
                    elif not _structural(replacement):
                        self._emit(s0, s1, lo, spans, src_starts, out)
                i += 1
        return out

    def _emit(self, r0: int, r1: int, floor: int, spans: list, src_starts: list[int],
              out: list[dict]) -> None:
        first, last = self._first[r0], self._last[r1 - 1]
        for index in range(max(first, self._next), last + 1):
            data, logprob, top = self._records[index]
            out.append({"bytes": bytes(data), "logprob": logprob, "top": top,
                        "offset": max(floor, self._out_offset(index, spans, src_starts))})
        self._next = max(self._next, last + 1)

    def _out_offset(self, index: int, spans: list, src_starts: list[int]) -> int:
        at = self._start[index]
        i = max(0, bisect.bisect_right(src_starts, at) - 1)
        s0, _s1, o0, _o1, replacement = spans[i]
        return o0 if replacement is not None else o0 + at - s0


def _structural(replacement: str) -> bool:
    """True for a scrub replacement that only marks structure (a think tag and
    the line breaks around it, or nothing): text no token wrote."""
    return not replacement.replace("<think>", "").replace("</think>", "").strip()


def _token_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def chat_logprobs_content(entries: list[dict], top_n: int) -> list[dict]:
    """``choices[].logprobs.content`` items for *entries* (from
    :meth:`LogprobAligner.take`), each with *top_n* alternatives."""
    return [{
        "token": _token_text(e["bytes"]),
        "logprob": e["logprob"],
        "bytes": list(e["bytes"]),
        "top_logprobs": [{"token": _token_text(alt), "logprob": lp, "bytes": list(alt)}
                         for alt, lp in e["top"][:top_n]],
    } for e in entries]


def completion_logprobs(entries: list[dict], top_n: int,
                        base_offset: int = 0) -> dict[str, Any]:
    """The legacy ``/v1/completions`` ``logprobs`` object for *entries*:
    ``tokens``, ``token_logprobs``, ``top_logprobs`` (each a token -> logprob
    map of the *top_n* most likely tokens plus the sampled one) and
    ``text_offset`` (each token's start in the returned text, after
    *base_offset* characters)."""
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    for e in entries:
        token = _token_text(e["bytes"])
        tokens.append(token)
        token_logprobs.append(e["logprob"])
        alts: dict[str, float] = {}
        for alt, lp in e["top"][:top_n]:
            alts.setdefault(_token_text(alt), lp)
        alts.setdefault(token, e["logprob"])
        top_logprobs.append(alts)
        text_offset.append(base_offset + e["offset"])
    return {"tokens": tokens, "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs, "text_offset": text_offset}
