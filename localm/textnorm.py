# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared scrubbing of model-internal control markers in chat output.

Some finetunes emit their training-format control markers as plain text:
harmony-style channel tags (``<|channel|>analysis <|message|>``), the Gemma 4
turn/tool dialect (``<|turn>model``, ``<|"|>`` quote tokens), a turn-open marker
together with its role word (``<start_of_turn>model``, ``<|im_start|>assistant``),
or reserved vocabulary placeholders (``<unused7>``). These are model internals,
not content.

Thinking-channel markers are not dropped but normalised to canonical
``<think> ... </think>`` so every frontend handles reasoning one way; the rest
are removed. This lives in one place and is applied once at the engine boundary
(:meth:`localm.inference.engine.Engine.chat_stream`) so every backend - GGUF, HF,
or any future one - inherits it instead of each re-implementing (or forgetting)
it. The functions are idempotent: a second pass over already-scrubbed text is a
no-op, so a backend that also scrubs internally is safe.
"""

from __future__ import annotations

import functools
import re
from re import _constants as _sre_constants
from re import _parser as _sre_parser
from typing import Iterator, Optional

# Reasoning-channel openers/closers -> canonical think tags. Up to four
# whitespace characters inside the tag are tolerated.
# Harmony: <|channel|>analysis<|message|>REASONING ... <|channel|>final<|message|>ANSWER
# Gemma 4: <|channel>thought / REASONING / <channel|>ANSWER
_THINK_OPEN_RE = re.compile(
    r"<\|?\s{0,4}channel\s{0,4}\|?>"
    r"(thought|thinking|analysis|reasoning|commentary|reflection)"
    r"\n?(<\|?\s{0,4}message\s{0,4}\|?>)?"
)
_THINK_CLOSE_RE = re.compile(
    r"<\s{0,4}channel\s{0,4}\|>"                                      # gemma4 close
    r"|<\|?\s{0,4}channel\s{0,4}\|?>final\n?(<\|?\s{0,4}message\s{0,4}\|?>)?"  # harmony final-channel switch
)

# Native reasoning tags emitted without the harmony/Gemma channel wrapper.
# "think" alone is excluded so canonical <think>/</think> tags pass through
# untouched and the transform stays idempotent.
_THINK_BARE_OPEN_RE = re.compile(
    r"<\s{0,4}(?:reasoning|thinking|thought|reflection)\s{0,4}>", re.IGNORECASE)
_THINK_BARE_CLOSE_RE = re.compile(
    r"<\s{0,4}/\s{0,4}(?:reasoning|thinking|thought|reflection)\s{0,4}>", re.IGNORECASE)

_MARKER_RE = re.compile(
    r"<\|?\s{0,4}channel\s{0,4}\|?>"                                  # leftover channel tag
    r"|<\s{0,4}channel\s{0,4}\|>"                                     # leftover gemma4 close
    r"|<\|?\s{0,4}message\s{0,4}\|?>"                                 # stray harmony separator
    r"|<\|start\|>(assistant|user|system)?"
    r"|<\|return\|>"
    r"|<\|turn>(user|model|assistant|system)?\n?"            # Gemma 4 turn open
    r"|<turn\|>"                                              # Gemma 4 turn close
    # <|tool_call> / <|tool_response> are not scrubbed: the coder agent parses
    # them out of this same stream.
    r"|<\|tool>|<tool\|>"                                     # Gemma 4 tool declarations
    r"|<\|think\|>|<think\|>"                                 # Gemma 4 thinking enable token
    r"|<unused\d{1,8}>?"                                          # Gemma reserved tokens
    r"|\[TOOL_CALLS\]"                                        # Mistral tool-call token
    # A turn-OPEN marker carries the role word, so the role suffix is matched
    # with it - removing the marker alone leaves a bare "model" / "assistant" at
    # the head of the reply. The matching turn-CLOSE markers are not listed
    # here: the backend handles those as stop strings, ending the turn rather
    # than editing the text.
    r"|<start_of_turn>(?:user|model|assistant|system|tool)?\n?"   # Gemma 1-3 turn open
    r"|<\|im_start\|>(?:user|model|assistant|system|tool)?\n?"    # ChatML turn open
    r"|<\|start_header_id\|>"
    r"(?:user|model|assistant|system|tool|ipython)?"
    r"<\|end_header_id\|>\n?"                                 # Llama 3 role header
)

# Longest text a partial marker could span across two stream pieces. Stays at or
# above the longest string _SCRUB_RE can match, or scrub_stream commits a cut
# inside a marker and leaks its tail as text.
# See test_marker_hold_covers_the_longest_possible_match.
_MARKER_HOLD = 56


# Every substitution scrub_text makes, first listed wins at a position. Each
# pattern starts with a character in _MARKER_START, which scrub_stream relies on
# to release text early.
# See test_every_scrub_pattern_starts_with_a_marker_start_character.
_SCRUB_SUBS = (
    (re.compile(re.escape('<|"|>')), '"'),            # Gemma 4 quote token
    (_THINK_OPEN_RE, "<think>\n"),
    (_THINK_CLOSE_RE, "\n</think>\n"),
    (_THINK_BARE_OPEN_RE, "<think>"),                  # native <reasoning> etc.
    (_THINK_BARE_CLOSE_RE, "</think>"),
    (_MARKER_RE, ""),
)
_MARKER_START = "<["

# _SCRUB_SUBS as one alternation: scrub_text rewrites every marker in a single
# left-to-right pass, and a rewrite's output is never matched again.
# See test_adjacent_markers_stream_like_one_shot.
_SCRUB_RE = re.compile("|".join(
    f"(?P<s{i}>{'(?i:' + rx.pattern + ')' if rx.flags & re.IGNORECASE else rx.pattern})"
    for i, (rx, _replacement) in enumerate(_SCRUB_SUBS)))
_SCRUB_REPLACEMENTS = {f"s{i}": replacement
                       for i, (_rx, replacement) in enumerate(_SCRUB_SUBS)}


def scrub_text(text: str) -> str:
    """Apply marker normalisation/removal to a complete text chunk."""
    return _SCRUB_RE.sub(lambda m: _SCRUB_REPLACEMENTS[m.lastgroup], text)


def _tagged(items, ignorecase: bool) -> list:
    return [(op, av, ignorecase) for op, av in items]


# _SCRUB_RE's parse tree, read by _prefix_fits.
_SCRUB_TREE = _tagged(_sre_parser.parse(_SCRUB_RE.pattern, _SCRUB_RE.flags), False)


def _char_in_class(ch: str, members, ignorecase: bool) -> bool:
    c = _sre_constants
    for op, value in members:
        if op == c.LITERAL:
            if ch == chr(value) or (ignorecase and ch.lower() == chr(value).lower()):
                return True
        elif op == c.CATEGORY and value == c.CATEGORY_SPACE:
            if ch.isspace():
                return True
        elif op == c.CATEGORY and value == c.CATEGORY_DIGIT:
            if ch.isdecimal():
                return True
        else:
            raise ValueError(f"unsupported character class member {op} {value}")
    return False


def _prefix_fits(items: list, s: str, k: int) -> bool:
    """True when ``s[k:]`` is a prefix of some string the parsed pattern
    *items* matches, i.e. more text could still complete a match there.

    Supports the constructs _SCRUB_RE uses (literals, ``\\s`` / ``\\d``
    classes, alternation, groups, bounded repeats); raises ValueError on any
    other."""
    c = _sre_constants
    if k == len(s):
        return True
    if not items:
        return False
    (op, av, ignorecase), rest = items[0], items[1:]
    if op == c.LITERAL:
        ch = s[k]
        if ch == chr(av) or (ignorecase and ch.lower() == chr(av).lower()):
            return _prefix_fits(rest, s, k + 1)
        return False
    if op == c.IN:
        return _char_in_class(s[k], av, ignorecase) and _prefix_fits(rest, s, k + 1)
    if op == c.BRANCH:
        return any(_prefix_fits(_tagged(alt, ignorecase) + rest, s, k) for alt in av[1])
    if op == c.SUBPATTERN:
        _group, add_flags, _del_flags, sub = av
        inner = ignorecase or bool(add_flags & re.IGNORECASE)
        return _prefix_fits(_tagged(sub, inner) + rest, s, k)
    if op == c.MAX_REPEAT or op == c.MIN_REPEAT:
        lo, hi, sub = av
        if lo == 0 and _prefix_fits(rest, s, k):
            return True
        if hi == 0:
            return False
        again = (op, (max(lo - 1, 0), hi - 1, sub), ignorecase)
        return _prefix_fits(_tagged(sub, ignorecase) + [again] + rest, s, k)
    raise ValueError(f"unsupported regex construct {op}")


@functools.lru_cache(maxsize=4096)
def _could_become_marker(text: str) -> bool:
    """True when *text* is a prefix of some string scrub_text would rewrite."""
    return _prefix_fits(_SCRUB_TREE, text, 0)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_think(text: str, exit_marker: Optional[str] = None) -> tuple[str, str]:
    """Split *text* (already scrubbed to canonical ``<think>...</think>``) into
    ``(content, reasoning)``: the visible answer with the think block(s) removed,
    and the concatenated reasoning with the tags removed. An unclosed ``<think>``
    runs to the end, except that with *exit_marker* set, an unclosed block that
    contains the marker ends where the marker begins and the rest is content
    (see ``gbnf.think_exit_marker``). Multiple blocks are concatenated.

    Linear single pass: scans with ``str.find`` and slices each segment exactly
    once, so it stays O(n) even on pathologically interleaved tags.
    ThinkSplitter, which re-slices its whole buffer per tag, is used for the
    streaming path, where each piece is small."""
    content: list[str] = []
    reasoning: list[str] = []
    i, n, in_think = 0, len(text), False
    while i < n:
        if in_think:
            j = text.find(_THINK_CLOSE, i)
            if j == -1:
                m = text.find(exit_marker, i) if exit_marker else -1
                if m != -1:
                    reasoning.append(text[i:m])
                    content.append(text[m:])
                    break
                reasoning.append(text[i:])          # unclosed think runs to the end
                break
            reasoning.append(text[i:j])
            i = j + len(_THINK_CLOSE)
            in_think = False
        else:
            j = text.find(_THINK_OPEN, i)
            if j == -1:
                content.append(text[i:])
                break
            content.append(text[i:j])
            i = j + len(_THINK_OPEN)
            in_think = True
    return "".join(content), "".join(reasoning)


def strip_think(text: str) -> str:
    """Visible content of *text* with every reasoning channel removed.

    Scrubs dialect markers to canonical ``<think>`` tags first (idempotent on
    already-scrubbed text), then drops the think channel, including an UNCLOSED
    trailing block (a truncated thinking reply must never leak scratchpad).

    This is the helper every INTERNAL consumer of model output runs before
    storing or parsing a reply (memory consolidation, episodic summaries, job
    results, compaction summaries, coder reflection). The /v1 routes already
    split reasoning for clients; this covers everything that never passes
    through them."""
    return split_think(scrub_text(text or ""))[0]


def _held_tag_suffix(s: str, tag: str) -> int:
    """Length of the longest proper prefix of *tag* that is a suffix of *s* -
    how much of the tail must be held back because it might begin *tag*."""
    k = min(len(s), len(tag) - 1)
    while k > 0:
        if s.endswith(tag[:k]):
            return k
        k -= 1
    return 0


class ThinkSplitter:
    """Stateful splitter for a token stream of already-scrubbed text.

    Feed each piece; get ``(content, reasoning)`` for that piece with the
    ``<think>`` / ``</think>`` tags removed and the reasoning routed out of the
    visible content. Tags split across pieces are handled by holding back a short
    tail until the next piece arrives; call :meth:`flush` at end of stream to
    release any held tail (an unterminated think block flushes as reasoning).

    With *exit_marker* set, the marker inside a think block starts a held
    stretch: if the block then closes, the stretch was reasoning; if the stream
    ends with the block still open, it flushes as content. The result always
    equals ``split_think(text, exit_marker)`` over the whole stream.
    """

    def __init__(self, exit_marker: Optional[str] = None) -> None:
        self._buf = ""
        self._in_think = False
        self._exit_marker = exit_marker or None
        self._marked = False     # holding from exit_marker until </think> or the end
        self._scan = 0           # where the next </think> search in a held stretch starts

    def feed(self, piece: str) -> tuple[str, str]:
        self._buf += piece
        out_c: list[str] = []
        out_r: list[str] = []
        while True:
            if self._in_think and self._marked:
                i = self._buf.find(_THINK_CLOSE, self._scan)
                if i == -1:
                    self._scan = max(0, len(self._buf) - len(_THINK_CLOSE) + 1)
                    break
                out_r.append(self._buf[:i])
                self._buf = self._buf[i + len(_THINK_CLOSE):]
                self._in_think = self._marked = False
                self._scan = 0
            elif self._in_think:
                i = self._buf.find(_THINK_CLOSE)
                m = self._buf.find(self._exit_marker) if self._exit_marker else -1
                if m != -1 and (i == -1 or m < i):
                    out_r.append(self._buf[:m])
                    self._buf = self._buf[m:]
                    self._marked = True
                    self._scan = 0
                    continue
                if i == -1:
                    hold = _held_tag_suffix(self._buf, _THINK_CLOSE)
                    if self._exit_marker:
                        hold = max(hold, _held_tag_suffix(self._buf, self._exit_marker))
                    cut = len(self._buf) - hold
                    out_r.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out_r.append(self._buf[:i])
                self._buf = self._buf[i + len(_THINK_CLOSE):]
                self._in_think = False
            else:
                i = self._buf.find(_THINK_OPEN)
                if i == -1:
                    hold = _held_tag_suffix(self._buf, _THINK_OPEN)
                    cut = len(self._buf) - hold
                    out_c.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out_c.append(self._buf[:i])
                self._buf = self._buf[i + len(_THINK_OPEN):]
                self._in_think = True
        return "".join(out_c), "".join(out_r)

    def flush(self) -> tuple[str, str]:
        """Release the held tail at end of stream. A still-open think block
        flushes its remainder as reasoning, unless it is held from the exit
        marker; otherwise as content."""
        buf, self._buf = self._buf, ""
        if self._in_think and not self._marked:
            return "", buf
        return buf, ""


# How many ``[TOOL_CALLS]`` tokens one reply may emit before the stream is cut.
_MARKER_FLOOD_LIMIT = 16
_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]")


def _cut_marker_flood(chunk: str, seen: int) -> tuple[str, int, bool]:
    """``(chunk, seen, flooded)``: *chunk* cut before the marker that exceeds
    ``_MARKER_FLOOD_LIMIT`` (counting the *seen* markers of earlier chunks)."""
    for m in _TOOL_CALLS_RE.finditer(chunk):
        seen += 1
        if seen > _MARKER_FLOOD_LIMIT:
            return chunk[:m.start()], seen, True
    return chunk, seen, False


def _close_source(pieces: Iterator[str]) -> None:
    """Close *pieces* when it is a generator, which stops its generation."""
    close = getattr(pieces, "close", None)
    if close is not None:
        close()


def _note_marker_flood() -> None:
    from localm.debuglog import logger
    logger.warning(
        "model emitted [TOOL_CALLS] more than %d times in one reply; "
        "the reply was cut", _MARKER_FLOOD_LIMIT)


def _marker_end(buf: str, at: int) -> int:
    """End of the scrub_text match starting at *at* in *buf*, or *at* when
    none starts there."""
    m = _SCRUB_RE.match(buf, at)
    return at if m is None else m.end()


def _commit_point(buf: str) -> int:
    """How much of *buf* scrub_stream can scrub and release now.

    Holds from the first marker-start character whose text so far could still
    grow into a marker (only possible within the last ``_MARKER_HOLD``
    characters), then backs up to the start of any complete marker that would
    straddle the cut. Everything else is released."""
    n = len(buf)
    cut = n
    for i in range(max(0, n - _MARKER_HOLD), n):
        if buf[i] in _MARKER_START and _could_become_marker(buf[i:]):
            cut = i
            break
    moved = True
    while moved:
        moved = False
        for q in range(max(0, cut - _MARKER_HOLD), cut):
            if buf[q] in _MARKER_START and _marker_end(buf, q) > cut:
                cut = q
                moved = True
                break
    return cut


def scrub_stream(pieces: Iterator[str]) -> Iterator[str]:
    """Normalise/remove internal model markers in a text stream.

    Each piece is scrubbed and yielded as soon as it arrives, except text from
    a ``<`` or ``[`` that could still grow into a marker: a marker (or its
    optional role suffix, e.g. ``<|turn>model``) starting there can straddle
    two pieces, and scrubbing it early would strip the marker head and leak its
    tail as text. That tail stays buffered until the text from it can no
    longer grow into a longer marker match, or the stream ends. The cut never
    lands inside a marker.

    A stream that emits ``[TOOL_CALLS]`` more than ``_MARKER_FLOOD_LIMIT`` times
    is cut at that marker and the source iterator is closed, ending the
    generation.
    """
    buf = ""
    seen = 0
    for piece in pieces:
        buf += piece
        cut = _commit_point(buf)
        if cut <= 0:
            continue
        chunk = buf[:cut]
        buf = buf[cut:]
        chunk, seen, flooded = _cut_marker_flood(chunk, seen)
        out = scrub_text(chunk)
        if out:
            yield out
        if flooded:
            _note_marker_flood()
            _close_source(pieces)
            return
    chunk, seen, flooded = _cut_marker_flood(buf, seen)
    buf = scrub_text(chunk)
    if buf:
        yield buf
    if flooded:
        _note_marker_flood()
