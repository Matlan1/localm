"""Shared scrubbing of model-internal control markers in chat output.

Some finetunes emit their training-format control markers as plain text:
harmony-style channel tags (``<|channel|>analysis <|message|>``), the Gemma 4
turn/tool dialect (``<|turn>model``, ``<|"|>`` quote tokens), or reserved
vocabulary placeholders (``<unused7>``). These are model internals, not content.

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
    return _MARKER_RE.sub("", text)


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
