# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shrink the coder's own conversation so a model's pre-tokenizer accepts it.

The coder builds most of what it sends: tool results, file contents, notices.
When the server refuses a request with a pre-tokenizer refusal
(``localm.inference.pretokenizer_guard``), the messages the coder built are cut
down until they fit; a message the user typed is never changed."""

from __future__ import annotations

from typing import Optional

import regex

from localm.inference.pretokenizer_guard import Refusal
from localm.textguard import compose, slice_guarded

# The character class each run refusal names, as a `regex` class.
_RUN_CLASSES = {
    "letters": r"[\p{L}\p{M}]",
    "letters and spaces": r"[\p{L}\p{M} ]",
    "digits": r"[\p{N}]",
    "line breaks": r"[\r\n]",
    "whitespace characters": r"[\s]",
}
_BLOCK_CLASS = r"[^ \t\n\r\f\v]"
# Characters kept below a refused limit, for the marker and any template text
# that shares the piece.
_MARGIN = 1024


class InputTooLongForModel(ValueError):
    """Raised when text the user typed is more than the model's pre-tokenizer
    takes in one message; the message says what to do instead."""


def _omitted(n: int) -> str:
    return f"\n[... {n} characters omitted to fit this model's input limit ...]\n"


def shrink(content, budget: int):
    """*content* cut to at most *budget* characters: its head and tail with a
    marker naming how much was left out between them. Untrusted ranges are
    kept over the text that survives. Unchanged when it already fits."""
    text = str(content)
    if len(text) <= budget:
        return content
    marker_len = len(_omitted(len(text)))
    keep = max(0, budget - marker_len)
    head = (keep * 2) // 3
    tail = keep - head
    return compose(slice_guarded(content, 0, head), _omitted(len(text) - keep),
                   slice_guarded(content, len(text) - tail, len(text)))


def elide_runs(content, char_class: str, limit: int):
    """``(content, n)``: *content* with every run of *char_class* longer than
    *limit* replaced by its first characters and a marker, and how many runs
    were replaced."""
    keep = max(1, limit // 2)
    pattern = regex.compile(char_class + "{" + str(limit + 1) + ",}")
    text = str(content)
    hits = list(pattern.finditer(text))
    if not hits:
        return content, 0
    parts: list = []
    pos = 0
    for m in hits:
        parts.append(slice_guarded(content, pos, m.start() + keep))
        parts.append(f" [... {m.end() - m.start() - keep} characters omitted] ")
        pos = m.end()
    parts.append(slice_guarded(content, pos, len(text)))
    return compose(*parts), len(hits)


def fit_messages(messages: list[dict], refusal: Refusal, *, protected: set,
                 reserve: int) -> tuple[list[int], list[int]]:
    """Cut the messages at indexes not in *protected* so the request
    *refusal* rejected would pass, in place.

    *reserve* is the room kept for text sent alongside every message (the
    system prompt). Returns ``(changed, offenders)``: the indexes changed, and
    the protected indexes that alone still break the limit."""
    changed: list[int] = []
    offenders: list[int] = []
    if refusal.kind == "chars":
        budget = max(_MARGIN, refusal.limit - reserve - _MARGIN)
        if refusal.part:
            for i, msg in enumerate(messages):
                if len(str(msg.get("content", ""))) <= budget:
                    continue
                if i in protected:
                    offenders.append(i)
                    continue
                msg["content"] = shrink(msg["content"], budget)
                changed.append(i)
            return changed, offenders
        # The input as a whole was over the limit: shrink the largest messages
        # the coder built until the total fits.
        total = sum(len(str(m.get("content", ""))) for m in messages)
        target = budget
        order = sorted((i for i in range(len(messages)) if i not in protected),
                       key=lambda i: -len(str(messages[i].get("content", ""))))
        for i in order:
            if total <= target:
                break
            size = len(str(messages[i].get("content", "")))
            new_size = max(_MARGIN, size - (total - target))
            if new_size >= size:
                continue
            messages[i]["content"] = shrink(messages[i]["content"], new_size)
            total -= size - len(str(messages[i]["content"]))
            changed.append(i)
        if total > target:
            offenders.extend(i for i in sorted(protected)
                             if len(str(messages[i].get("content", ""))) > 0)
        return changed, offenders
    if refusal.kind == "run":
        char_class = _RUN_CLASSES.get(refusal.run_kind or "")
        if char_class is None:
            return changed, offenders
        limit = max(1, refusal.limit)
    else:
        char_class, limit = _BLOCK_CLASS, max(1, refusal.limit // 2)
    probe = regex.compile(char_class + "{" + str(limit + 1) + ",}")
    for i, msg in enumerate(messages):
        if probe.search(str(msg.get("content", ""))) is None:
            continue
        if i in protected:
            offenders.append(i)
            continue
        msg["content"], n = elide_runs(msg["content"], char_class, limit)
        if n:
            changed.append(i)
    return changed, offenders


def user_message_refusal(refusal: Refusal) -> Optional[str]:
    """The message for the user when their own text is what the model's
    pre-tokenizer cannot take, naming the limit; None for an unknown kind."""
    if refusal.kind == "chars":
        return (f"Your message is too long for this model: its pre-tokenizer takes "
                f"at most {refusal.limit} characters in one message. Send the text "
                "in smaller parts, or attach it as a file the agent can read.")
    if refusal.kind == "run":
        return (f"Your message has an unbroken run of {refusal.run_kind} longer than "
                f"this model's pre-tokenizer takes ({refusal.limit}). Break it up "
                "with spaces or punctuation, or put the text in a file the agent "
                "can read.")
    if refusal.kind == "block":
        return ("Your message has a long block of text with no spaces or line "
                "breaks, which this model's pre-tokenizer cannot take. Break it up, "
                "or put the text in a file the agent can read.")
    return None
