# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared scrubbing of model-internal control markers in chat output.

Some finetunes emit their training-format control markers as plain text:
harmony-style channel tags (``<|channel|>analysis <|message|>``), the Gemma 4
turn/tool dialect (``<|turn>model``, ``<|"|>`` quote tokens), a turn-open marker
together with its role word (``<start_of_turn>model``, ``<|im_start|>assistant``),
or reserved vocabulary placeholders (``<unused7>``). These are model internals,
not content.

Thinking-channel markers are not dropped but normalised to canonical
``<think> ... </think>`` so every frontend handles reasoning one way, a harmony
function call (``to=functions.NAME``) becomes a canonical ``<tool_call>`` block,
and the rest are removed. This lives in one place and is applied once at the engine boundary
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

# Harmony tool-call header, with the recipient in the role or the channel
# section: <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>
# or <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>, the
# "functions." prefix optional. The call's JSON arguments follow until <|call|>
# (or the next harmony boundary).
_HARMONY_CALL_RE = re.compile(
    r"(?:<\|start\|>assistant\s{0,4}to=(?:functions\.)?[A-Za-z0-9_.-]{1,64}\s{0,4}"
    r"<\|channel\|>(?:commentary|analysis)"
    r"|<\|channel\|>(?:commentary|analysis)\s{0,4}to=(?:functions\.)?[A-Za-z0-9_.-]{1,64})"
    r"\s{0,4}(?:<\|constrain\|>\s{0,4})?(?:json)?\s{0,4}<\|message\|>"
)
_HARMONY_CALL_NAME_RE = re.compile(r"to=(functions\.)?([A-Za-z0-9_.-]{1,64})")
# Recipients of harmony's built-in tools; a header addressed to one is not a
# function call.
_HARMONY_BUILTIN_RE = re.compile(r"python|(?:browser|container)\.[A-Za-z0-9_.-]*")
# Markers that end an open harmony call message.
_HARMONY_BOUNDARY_RE = re.compile(r"<\|(?:call|end|return|start)\|>|<\|?\s{0,4}channel")
_CALL_CLOSE = "}\n</tool_call>"

# Native reasoning tags emitted without the harmony/Gemma channel wrapper.
# "think" alone is excluded so canonical <think>/</think> tags pass through
# untouched and the transform stays idempotent.
_THINK_BARE_OPEN_RE = re.compile(
    r"<\s{0,4}(?:reasoning|thinking|thought|reflection)\s{0,4}>", re.IGNORECASE)
_THINK_BARE_CLOSE_RE = re.compile(
    r"<\s{0,4}/\s{0,4}(?:reasoning|thinking|thought|reflection)\s{0,4}>", re.IGNORECASE)

# Markers that scrub to nothing wherever they sit. A turn-open marker followed by
# one of these is left for the next pass, so the role word the marker owns is
# adjacent to it once the inert marker is gone.
_INERT_AHEAD = (
    r"<\|\s{0,4}channel\s{0,4}\|?>(?!thought|thinking|analysis|reasoning|commentary"
    r"|reflection|final)"
    r"|<\s{0,4}channel\s{0,4}>(?!thought|thinking|analysis|reasoning|commentary"
    r"|reflection|final)"
    r"|<\|?\s{0,4}message\s{0,4}\|?>"
    r"|<\|return\|>"
    r"|<\|end\|>"
    r"|<turn\|>"
    r"|<\|tool>|<tool\|>"
    r"|<\|think\|>|<think\|>"
    r"|<unused\d{1,8}>?"
    r"|\[TOOL_CALLS\]"
    r"|<\|start_header_id\|>(?:user|model|assistant|system|tool|ipython)?<\|end_header_id\|>"
)

_MARKER_RE = re.compile(
    r"<\|?\s{0,4}channel\s{0,4}\|?>"                                  # leftover channel tag
    r"|<\s{0,4}channel\s{0,4}\|>"                                     # leftover gemma4 close
    r"|<\|?\s{0,4}message\s{0,4}\|?>"                                 # stray harmony separator
    r"|<\|start\|>(?:(?:assistant|user|system)|(?!" + _INERT_AHEAD + r"))"
    r"|<\|return\|>"
    r"|<\|call\|>"                                             # harmony tool-call end
    r"|<\|end\|>"                                              # harmony message end
    r"|<\|turn>(?:(?:user|model|assistant|system)\n?|(?!" + _INERT_AHEAD + r")\n?)"  # Gemma 4 turn open
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
    r"|<start_of_turn>(?:(?:user|model|assistant|system|tool)\n?|(?!" + _INERT_AHEAD + r")\n?)"
    r"|<\|im_start\|>(?:(?:user|model|assistant|system|tool)\n?|(?!" + _INERT_AHEAD + r")\n?)"
    r"|<\|start_header_id\|>"
    r"(?:user|model|assistant|system|tool|ipython)?"
    r"<\|end_header_id\|>\n?"                                 # Llama 3 role header
)

# Longest text a partial marker could span across two stream pieces. Stays at or
# above the longest string _SCRUB_RE can match, or scrub_stream commits a cut
# inside a marker and leaks its tail as text.
# See test_marker_hold_covers_the_longest_possible_match.
_MARKER_HOLD = 168


# Every substitution scrub_text makes, first listed wins at a position. Each
# pattern starts with a character in _MARKER_START, which scrub_stream relies on
# to release text early. A ``None`` replacement is computed per match by
# _ScrubState.marker.
# See test_every_scrub_pattern_starts_with_a_marker_start_character.
_SCRUB_SUBS = (
    (re.compile(re.escape('<|"|>')), '"'),            # Gemma 4 quote token
    (_HARMONY_CALL_RE, None),                          # harmony function call
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
                       for i, (_rx, replacement) in enumerate(_SCRUB_SUBS)
                       if replacement is not None}
# _SCRUB_RE without the harmony call header, for a header addressed to a
# built-in tool.
_SCRUB_NO_CALL_RE = re.compile("|".join(
    f"(?P<s{i}>{'(?i:' + rx.pattern + ')' if rx.flags & re.IGNORECASE else rx.pattern})"
    for i, (rx, _replacement) in enumerate(_SCRUB_SUBS) if rx is not _HARMONY_CALL_RE))
_G_CALL, _G_THINK_OPEN, _G_THINK_CLOSE = (
    f"s{i}" for i, (rx, _r) in enumerate(_SCRUB_SUBS)
    if rx in (_HARMONY_CALL_RE, _THINK_OPEN_RE, _THINK_CLOSE_RE))
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ScrubState:
    """What scrubbing has read and written so far, as far as the next rewrite
    depends on it: whether the output has an open ``<think>`` block (the last
    think tag in it is an opening one), whether a harmony call is open, and how
    far its JSON arguments have been read.

    A harmony call header is rewritten to the opening of a ``<tool_call>``
    block whose JSON is ``{"name": NAME, "arguments": `` followed by the
    model's arguments; an open think block is closed first. The block is closed
    right after the arguments' JSON object or array ends, or at the next harmony
    boundary marker. A think-close marker after a call header closed the think
    block is rewritten to a line break. Every output character depends only on
    the text before it."""

    __slots__ = ("think", "call", "_tail", "_scan", "_depth", "_quote", "_escape",
                 "_call_ended_think")

    def __init__(self) -> None:
        self.think = False
        self.call = False
        self._tail = ""
        self._scan = False       # reading an open call's JSON arguments
        self._depth = 0
        self._quote = False
        self._escape = False
        self._call_ended_think = False   # a call header closed the last think block

    def copy(self) -> _ScrubState:
        other = _ScrubState()
        for slot in self.__slots__:
            setattr(other, slot, getattr(self, slot))
        return other

    def see(self, out: str) -> None:
        """Note *out*, the next text written."""
        if not out:
            return
        window = self._tail + out
        opened, closed = window.rfind(_THINK_OPEN), window.rfind(_THINK_CLOSE)
        if opened > closed:
            self.think = True
            self._call_ended_think = False
        elif closed > opened:
            self.think = False
        self._tail = window[-(len(_THINK_CLOSE) - 1):]

    def text(self, seg: str) -> int:
        """Note *seg*, the next text copied unchanged, up to the end of an open
        call's arguments. Returns how many characters that is (call
        :meth:`close` next) or -1 when all of *seg* was noted."""
        end = self._args_end(seg) if self._scan else -1
        self.see(seg if end < 0 else seg[:end])
        return end

    def _args_end(self, seg: str) -> int:
        for i, ch in enumerate(seg):
            if self._quote:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._quote = False
            elif self._depth == 0:
                if ch in "{[":
                    self._depth = 1
                elif not ch.isspace():
                    self._scan = False
                    return -1
            elif ch == '"':
                self._quote = True
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
                if self._depth == 0:
                    self._scan = False
                    return i + 1
        return -1

    def close(self) -> str:
        """The close of the open call, noted as written."""
        self.call = self._scan = False
        self.see(_CALL_CLOSE)
        return _CALL_CLOSE

    def marker(self, m: re.Match[str]) -> str:
        """The replacement for scrub match *m*, noted as written."""
        group = str(m.lastgroup)
        out = ""
        if self.call and (group in (_G_CALL, _G_THINK_OPEN, _G_THINK_CLOSE)
                          or _HARMONY_BOUNDARY_RE.match(m.group())):
            out = self.close()
        name_match = _HARMONY_CALL_NAME_RE.search(m.group()) if group == _G_CALL else None
        name = name_match.group(2) if name_match else ""
        builtin = (name_match is not None and not name_match.group(1)
                   and _HARMONY_BUILTIN_RE.fullmatch(name) is not None)
        if builtin:
            replacement = _SCRUB_NO_CALL_RE.sub(
                lambda inner: _SCRUB_REPLACEMENTS[str(inner.lastgroup)], m.group())
        elif group == _G_CALL:
            replacement = "\n" + _THINK_CLOSE if self.think else ""
            self._call_ended_think = self.think
            replacement += '<tool_call>\n{"name": "' + name + '", "arguments": '
            self.call = self._scan = True
            self._depth = 0
            self._quote = self._escape = False
        elif group == _G_THINK_CLOSE and not self.think and self._call_ended_think:
            replacement = "\n"
            self._call_ended_think = False
        else:
            replacement = _SCRUB_REPLACEMENTS[group]
        self.see(replacement)
        return out + replacement


def _scrub_part(text: str, state: _ScrubState, rx: re.Pattern[str] = _SCRUB_RE) -> str:
    """*text* scrubbed in one marker pass, continuing from *state* (which is
    advanced)."""
    out: list[str] = []

    def copied(seg: str) -> None:
        while seg:
            k = state.text(seg)
            if k < 0:
                out.append(seg)
                return
            out.append(seg[:k])
            out.append(state.close())
            seg = seg[k:]

    pos = 0
    for m in rx.finditer(text):
        copied(text[pos:m.start()])
        out.append(state.marker(m))
        pos = m.end()
    copied(text[pos:])
    return "".join(out)

# The same alternation with every turn-open marker matched bare, as one marker
# pass matched it before a turn-open marker waited out the inert markers after
# it. ScrubMap maps this one pass.
_ONE_PASS_RE = re.compile(_SCRUB_RE.pattern.replace("(?!" + _INERT_AHEAD + ")", ""))


_MAX_SCRUB_PASSES = 32


def _scrub_once(text: str) -> str:
    """One marker pass after the first. A harmony call header is scrubbed like
    one addressed to a built-in tool."""
    return _SCRUB_NO_CALL_RE.sub(lambda m: _SCRUB_REPLACEMENTS[str(m.lastgroup)], text)


def _scrub_all(text: str, state: _ScrubState) -> str:
    """*text* scrubbed to a fixed point: the first pass continues from *state*
    (which is advanced), the later passes are :func:`_scrub_once`."""
    out = _scrub_part(text, state)
    for _ in range(_MAX_SCRUB_PASSES - 1):
        if out is text or out == text:
            return out
        text = out
        out = _scrub_once(text)
    if out is text or out == text:
        return out
    from localm.debuglog import logger
    logger.warning(
        "scrub_text still changed the text after %d passes; markers nested that "
        "deep were left in the reply", _MAX_SCRUB_PASSES)
    return out


def scrub_text(text: str) -> str:
    """Apply marker normalisation/removal to a complete text chunk.

    Repeats the single marker pass until the text stops changing, so removing a
    marker that sat inside another one (``<|start_header_id|><|message|>
    <|end_header_id|>``) also removes the marker it exposed, and the result is
    a fixed point. Gives up after ``_MAX_SCRUB_PASSES`` passes and logs a
    warning; the text returned then still holds markers nested that deep."""
    return _scrub_all(text, _ScrubState())


class ScrubMap:
    """Where each part of :func:`scrub_text`'s output came from, for a text
    that only grows.

    Maps one marker pass in which a turn-open marker is matched bare. For a text
    that needs another pass (a marker nested inside another marker, or a
    turn-open marker followed by inert markers and then its role word) the
    spans rebuild the first pass's output, which differs from
    :func:`scrub_text`'s; LogprobAligner then refuses the reply with
    LogprobAlignmentError instead of aligning against the wrong text.

    :meth:`spans` takes the text so far (each call's text must extend the
    previous call's) and returns ``(src_start, src_end, out_start, out_end,
    replacement)`` tuples covering the text and ``scrub_text(text)`` end to end,
    in order: ``replacement`` is None for text copied unchanged and the
    replacement string for a rewritten marker. Text within ``_MARKER_HOLD``
    characters of the end is mapped as if the text ended there; a marker it
    turns into once more text arrives is mapped on a later call. The close of
    a harmony call is a replacement span with ``src_start == src_end``."""

    def __init__(self) -> None:
        self._final: list[tuple[int, int, int, int, Optional[str]]] = []
        self._src = 0
        self._out = 0
        self._state = _ScrubState()
        self._spans: list[tuple[int, int, int, int, Optional[str]]] = []
        self._seen = -1

    def spans(self, text: str) -> list[tuple[int, int, int, int, Optional[str]]]:
        n = len(text)
        if n == self._seen:
            return self._spans
        tail: list[tuple[tuple[int, int, int, int, Optional[str]], _ScrubState, _ScrubState]] = []
        state = self._state.copy()
        src, out = self._src, self._out

        def copied(s0: int, s1: int) -> int:
            o = out
            while s0 < s1:
                before = state.copy()
                k = state.text(text[s0:s1])
                stop = s1 if k < 0 else s0 + k
                tail.append(((s0, stop, o, o + stop - s0, None), before, state.copy()))
                o += stop - s0
                if k < 0:
                    break
                before = state.copy()
                close = state.close()
                tail.append(((stop, stop, o, o + len(close), close), before, state.copy()))
                o += len(close)
                s0 = stop
            return o

        for m in _ONE_PASS_RE.finditer(text, src):
            out = copied(src, m.start())
            before = state.copy()
            replacement = state.marker(m)
            tail.append(((m.start(), m.end(), out, out + len(replacement), replacement),
                         before, state.copy()))
            out += len(replacement)
            src = m.end()
        copied(src, n)
        settled = n - _MARKER_HOLD
        rest: list[tuple[int, int, int, int, Optional[str]]] = []
        for i, (span, before, after) in enumerate(tail):
            s0, s1, o0, o1, replacement = span
            if ((s1 <= settled) if replacement is None
                    else (s0 <= settled) if s0 == s1 else (s0 < settled)):
                _join_span(self._final, span)
                self._state = after
            elif replacement is None and s0 < settled:
                cut = settled - s0
                _join_span(self._final, (s0, settled, o0, o0 + cut, None))
                before.text(text[s0:settled])
                self._state = before
                rest = [(settled, s1, o0 + cut, o1, None)] + [t[0] for t in tail[i + 1:]]
                break
            else:
                rest = [t[0] for t in tail[i:]]
                break
        if self._final:
            self._src, self._out = self._final[-1][1], self._final[-1][3]
        spans = list(self._final)
        for span in rest:
            _join_span(spans, span)
        self._spans = spans
        self._seen = n
        return self._spans


def _join_span(spans: list, span: tuple[int, int, int, int, Optional[str]]) -> None:
    """Append *span* to *spans*, merged into the last one when both are copied
    text and *span* continues it."""
    if spans and span[4] is None and spans[-1][4] is None and spans[-1][1] == span[0]:
        s0, _s1, o0, _o1, _r = spans[-1]
        spans[-1] = (s0, span[1], o0, span[3], None)
    else:
        spans.append(span)


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
        elif op == c.RANGE:
            if value[0] <= ord(ch) <= value[1]:
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
    if op == c.ASSERT_NOT:
        return _prefix_fits(rest, s, k)
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

    After each :meth:`feed` or :meth:`flush`, ``spans`` holds the
    ``(start, end)`` offsets, in the whole stream fed so far, of the slices that
    call's content was cut from, in order.
    """

    def __init__(self, exit_marker: Optional[str] = None) -> None:
        self._buf = ""
        self._pos = 0            # stream offset of _buf[0]
        self._in_think = False
        self._exit_marker = exit_marker or None
        self._marked = False     # holding from exit_marker until </think> or the end
        self._scan = 0           # where the next </think> search in a held stretch starts
        self.spans: list[tuple[int, int]] = []

    def _skip(self, n: int) -> None:
        self._buf = self._buf[n:]
        self._pos += n

    def _take(self, n: int, out: list[str], content: bool) -> None:
        if n:
            out.append(self._buf[:n])
            if content:
                self.spans.append((self._pos, self._pos + n))
        self._skip(n)

    def feed(self, piece: str) -> tuple[str, str]:
        self._buf += piece
        self.spans = []
        out_c: list[str] = []
        out_r: list[str] = []
        while True:
            if self._in_think and self._marked:
                i = self._buf.find(_THINK_CLOSE, self._scan)
                if i == -1:
                    self._scan = max(0, len(self._buf) - len(_THINK_CLOSE) + 1)
                    break
                self._take(i, out_r, False)
                self._skip(len(_THINK_CLOSE))
                self._in_think = self._marked = False
                self._scan = 0
            elif self._in_think:
                i = self._buf.find(_THINK_CLOSE)
                m = self._buf.find(self._exit_marker) if self._exit_marker else -1
                if m != -1 and (i == -1 or m < i):
                    self._take(m, out_r, False)
                    self._marked = True
                    self._scan = 0
                    continue
                if i == -1:
                    hold = _held_tag_suffix(self._buf, _THINK_CLOSE)
                    if self._exit_marker:
                        hold = max(hold, _held_tag_suffix(self._buf, self._exit_marker))
                    self._take(len(self._buf) - hold, out_r, False)
                    break
                self._take(i, out_r, False)
                self._skip(len(_THINK_CLOSE))
                self._in_think = False
            else:
                i = self._buf.find(_THINK_OPEN)
                if i == -1:
                    hold = _held_tag_suffix(self._buf, _THINK_OPEN)
                    self._take(len(self._buf) - hold, out_c, True)
                    break
                self._take(i, out_c, True)
                self._skip(len(_THINK_OPEN))
                self._in_think = True
        return "".join(out_c), "".join(out_r)

    def flush(self) -> tuple[str, str]:
        """Release the held tail at end of stream. A still-open think block
        flushes its remainder as reasoning, unless it is held from the exit
        marker; otherwise as content."""
        self.spans = []
        out: list[str] = []
        if self._in_think and not self._marked:
            self._take(len(self._buf), out, False)
            return "", "".join(out)
        self._take(len(self._buf), out, True)
        return "".join(out), ""


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


def _scrub_released(chunk: str, state: _ScrubState) -> Optional[tuple[str, _ScrubState]]:
    """``(_scrub_all(chunk, state), the advanced state)`` with *state* left as
    it was, or None when text that follows *chunk* could still change it: the
    end of the text is held back (:func:`_commit_point`) before the first marker
    pass or after any pass, as a marker removed in one pass can leave the end of
    the text open to a marker in the next."""
    if _commit_point(chunk) < len(chunk):
        return None
    after = state.copy()
    text, out = chunk, _scrub_part(chunk, after)
    for _ in range(_MAX_SCRUB_PASSES - 1):
        if out is text or out == text:
            return out, after
        text = out
        if _commit_point(text) < len(text):
            return None
        out = _scrub_once(text)
    after = state.copy()
    return _scrub_all(chunk, after), after


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

    The scrubbing state (an open think block, an open harmony call) carries
    from piece to piece, so the output equals ``scrub_text`` of the whole
    stream.
    """
    buf = ""
    seen = 0
    state = _ScrubState()
    for piece in pieces:
        buf += piece
        cut = _commit_point(buf)
        if cut <= 0:
            continue
        chunk, counted, flooded = _cut_marker_flood(buf[:cut], seen)
        if flooded:
            out = _scrub_all(chunk, state)
        else:
            released = _scrub_released(chunk, state)
            if released is None:
                continue
            out, state = released
        buf = buf[cut:]
        seen = counted
        if out:
            yield out
        if flooded:
            _note_marker_flood()
            _close_source(pieces)
            return
    chunk, seen, flooded = _cut_marker_flood(buf, seen)
    buf = _scrub_all(chunk, state)
    if buf:
        yield buf
    if flooded:
        _note_marker_flood()
