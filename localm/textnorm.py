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

import re
from typing import Iterator

# Reasoning-channel openers/closers -> canonical think tags. Whitespace inside
# the tag is tolerated.
# Harmony: <|channel|>analysis<|message|>REASONING ... <|channel|>final<|message|>ANSWER
# Gemma 4: <|channel>thought / REASONING / <channel|>ANSWER
_THINK_OPEN_RE = re.compile(
    r"<\|?\s*channel\s*\|?>"
    r"(thought|thinking|analysis|reasoning|commentary|reflection)"
    r"\n?(<\|?\s*message\s*\|?>)?"
)
_THINK_CLOSE_RE = re.compile(
    r"<\s*channel\s*\|>"                                      # gemma4 close
    r"|<\|?\s*channel\s*\|?>final\n?(<\|?\s*message\s*\|?>)?"  # harmony final-channel switch
)

# Native reasoning tags emitted without the harmony/Gemma channel wrapper.
# "think" alone is excluded so canonical <think>/</think> tags pass through
# untouched and the transform stays idempotent.
_THINK_BARE_OPEN_RE = re.compile(
    r"<\s*(?:reasoning|thinking|thought|reflection)\s*>", re.IGNORECASE)
_THINK_BARE_CLOSE_RE = re.compile(
    r"<\s*/\s*(?:reasoning|thinking|thought|reflection)\s*>", re.IGNORECASE)

_MARKER_RE = re.compile(
    r"<\|?\s*channel\s*\|?>"                                  # leftover channel tag
    r"|<\s*channel\s*\|>"                                     # leftover gemma4 close
    r"|<\|?\s*message\s*\|?>"                                 # stray harmony separator
    r"|<\|start\|>(assistant|user|system)?"
    r"|<\|return\|>"
    r"|<\|turn>(user|model|assistant|system)?\n?"            # Gemma 4 turn open
    r"|<turn\|>"                                              # Gemma 4 turn close
    # <|tool_call> / <|tool_response> are not scrubbed: the coder agent parses
    # them out of this same stream.
    r"|<\|tool>|<tool\|>"                                     # Gemma 4 tool declarations
    r"|<\|think\|>|<think\|>"                                 # Gemma 4 thinking enable token
    r"|<unused\d+>?"                                          # Gemma reserved tokens
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
# above the longest string _MARKER_RE can match, or scrub_stream commits a cut
# inside a marker and leaks its tail as text.
# See test_marker_hold_covers_every_marker_at_every_stream_split.
_MARKER_HOLD = 48


def scrub_text(text: str) -> str:
    """Apply marker normalisation/removal to a complete text chunk."""
    text = text.replace('<|"|>', '"')          # Gemma 4 quote token
    text = _THINK_OPEN_RE.sub("<think>\n", text)
    text = _THINK_CLOSE_RE.sub("\n</think>\n", text)
    text = _THINK_BARE_OPEN_RE.sub("<think>", text)    # native <reasoning> etc.
    text = _THINK_BARE_CLOSE_RE.sub("</think>", text)
    return _MARKER_RE.sub("", text)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_think(text: str) -> tuple[str, str]:
    """Split *text* (already scrubbed to canonical ``<think>...</think>``) into
    ``(content, reasoning)``: the visible answer with the think block(s) removed,
    and the concatenated reasoning with the tags removed. An unclosed ``<think>``
    runs to the end. Multiple blocks are concatenated.

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
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, piece: str) -> tuple[str, str]:
        self._buf += piece
        out_c: list[str] = []
        out_r: list[str] = []
        while True:
            if self._in_think:
                i = self._buf.find(_THINK_CLOSE)
                if i == -1:
                    hold = _held_tag_suffix(self._buf, _THINK_CLOSE)
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
        flushes its remainder as reasoning; otherwise as content."""
        buf, self._buf = self._buf, ""
        if self._in_think:
            return "", buf
        return buf, ""


def scrub_stream(pieces: Iterator[str]) -> Iterator[str]:
    """Normalise/remove internal model markers in a text stream.

    The trailing ``_MARKER_HOLD`` characters stay buffered because a marker
    (or its optional role suffix, e.g. ``<|turn>model``) can straddle two
    pieces - scrubbing them too early would strip the marker head and leak its
    tail as text. Only the committed region is scrubbed and yielded; the cut
    never lands inside a potential marker (markers start with ``<``).
    """
    buf = ""
    for piece in pieces:
        buf += piece
        cut = len(buf) - _MARKER_HOLD
        if cut <= 0:
            continue
        # Back the cut up to the last '<' before the boundary so a marker
        # straddling it stays whole in the buffer.
        lt = buf.rfind("<", max(0, cut - _MARKER_HOLD), cut)
        if lt != -1:
            cut = lt
        if cut <= 0:
            continue
        out = scrub_text(buf[:cut])
        buf = buf[cut:]
        if out:
            yield out
    buf = scrub_text(buf)
    if buf:
        yield buf
