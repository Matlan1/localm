# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared scrubbing of model-internal control markers in chat output."""

from __future__ import annotations

import re
from typing import Iterator

# Reasoning-channel openers/closers -> canonical think tags. Whitespace inside
# the tag (around "channel"/"message" and the bars) is tolerated so spaced
# variants do not leak.
# Harmony: <|channel|>analysis<|message|>REASONING ... <|channel|>final<|message|>ANSWER
# Gemma 4: <|channel>thought\nREASONING\n<channel|>ANSWER
_THINK_OPEN_RE = re.compile(
    r"<\|?\s*channel\s*\|?>"
    r"(thought|thinking|analysis|reasoning|commentary|reflection)"
    r"\n?(<\|?\s*message\s*\|?>)?"
)
_THINK_CLOSE_RE = re.compile(
    r"<\s*channel\s*\|>"                                      # gemma4 close
    r"|<\|?\s*channel\s*\|?>final\n?(<\|?\s*message\s*\|?>)?"  # harmony final-channel switch
)

# Native reasoning tags some finetunes emit WITHOUT the harmony/Gemma channel
# wrapper (e.g. a bare <reasoning>...</reasoning>, <thinking>, <thought>,
# <reflection>). Without normalising these to canonical <think>...</think> they
# escape the reasoning/content split and leak into the visible answer (CHAT-2).
# "think" alone is excluded so the already-canonical <think>/</think> tags pass
# through untouched (and stay idempotent).
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
    # NOTE: <|tool_call> / <|tool_response> markers are deliberately NOT
    # scrubbed - the coder agent parses them out of this same stream.
    r"|<\|tool>|<tool\|>"                                     # Gemma 4 tool declarations
    r"|<\|think\|>|<think\|>"                                 # Gemma 4 thinking enable token
    r"|<unused\d+>?"                                          # Gemma reserved tokens
)

# Longest text a partial marker could span across two stream pieces.
_MARKER_HOLD = 32


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
    """Split *text* (already scrubbed to canonical ``<think>...</think>``) into ``(content, reasoning)``: the visible answer with the think block(s) removed, and the concatenated reasoning with the tags removed."""
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
    """Visible content of *text* with every reasoning channel removed."""
    return split_think(scrub_text(text or ""))[0]


def _held_tag_suffix(s: str, tag: str) -> int:
    """Length of the longest proper prefix of *tag* that is a suffix of *s* - how much of the tail must be held back because it might begin *tag*."""
    k = min(len(s), len(tag) - 1)
    while k > 0:
        if s.endswith(tag[:k]):
            return k
        k -= 1
    return 0


class ThinkSplitter:
    """Stateful splitter for a token stream of already-scrubbed text."""

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
        """Release the held tail at end of stream."""
        buf, self._buf = self._buf, ""
        if self._in_think:
            return "", buf
        return buf, ""


def scrub_stream(pieces: Iterator[str]) -> Iterator[str]:
    """Normalise/remove internal model markers in a text stream."""
    buf = ""
    for piece in pieces:
        buf += piece
        cut = len(buf) - _MARKER_HOLD
        if cut <= 0:
            continue
        # Back the cut up to the last '<' just before the boundary so a marker
        # straddling it stays whole in the buffer. A legit '<' in prose only
        # delays its emission one round - the window slides past it as more
        # text arrives.
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
