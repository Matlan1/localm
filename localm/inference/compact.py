# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Conversation compaction for chat sessions.

When a chat history approaches the context ceiling, older turns are
summarised by the model itself and replaced with a compact summary
exchange, keeping the most recent turns verbatim. The kept tail always
starts at a user turn, so the latest user request survives verbatim and the
bridge exchange keeps user/assistant alternation. If summarisation fails
for any reason (model error, empty or all-reasoning output), the bridge
carries a bounded digest of excerpts from the removed turns instead. Either
way the function never raises and always returns a usable history: chat
keeps working instead of dying at the ceiling.

Used by the CLI interactive chat; the GUI implements the same protocol
client-side against /v1/chat/completions.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from localm.textguard import (
    compose, compose_join, slice_guarded, untrusted_spans_of,
)

# Fraction of the context ceiling at which compaction kicks in
COMPACT_RATIO = 0.70

# Minimum number of most recent messages kept verbatim; the cut moves back to
# the nearest user turn, so the tail can be longer.
KEEP_RECENT = 4

# Budget for the generated summary. The summariser is asked to answer without
# its reasoning channel.
SUMMARY_MAX_TOKENS = 1024

# Character budget of the fallback digest of removed turns (about
# SUMMARY_MAX_TOKENS tokens), and the smallest excerpt kept per turn.
DIGEST_MAX_CHARS = 4 * SUMMARY_MAX_TOKENS
DIGEST_MIN_EXCERPT = 160

_DIGEST_NOTE = (
    "[Earlier conversation was condensed to fit the context window. "
    "Summarisation was unavailable, so these are excerpts of the earlier "
    "turns:]"
)


def _text_of(message: dict) -> str:
    """Plain text of a message; multipart images contribute a placeholder."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("[image]")
    return " ".join(parts)


def estimate_tokens(
    messages: List[dict],
    count_tokens: Optional[Callable[[str], int]] = None,
) -> int:
    """Token estimate for a message list (images count a flat 750 each)."""
    total = 0
    for m in messages:
        text = _text_of(m)
        if count_tokens is not None:
            try:
                total += count_tokens(text)
            except Exception:
                total += max(1, len(text) // 4)
        else:
            total += max(1, len(text) // 4)
        if not isinstance(m.get("content"), str):
            total += 750 * sum(
                1 for p in m.get("content", [])
                if isinstance(p, dict) and p.get("type") == "image_url"
            )
    return total


def _split(messages: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """(leading system messages, older middle, recent tail).

    The tail holds at least the last KEEP_RECENT messages and always starts at
    a user message: the cut moves back to the nearest user message, or, when
    there is none before it, forward to the next one. With no user message to
    cut at, older is empty."""
    head = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    if len(rest) <= KEEP_RECENT:
        return head, [], rest
    cut = len(rest) - KEEP_RECENT
    back = cut
    while back > 0 and rest[back].get("role") != "user":
        back -= 1
    if back > 0:
        cut = back
    else:
        while cut < len(rest) and rest[cut].get("role") != "user":
            cut += 1
        if cut >= len(rest):
            return head, [], rest
    return head, rest[:cut], rest[cut:]


def _excerpt_of(message: dict, limit: int):
    """The first *limit* characters of a message's text. Reasoning blocks are
    removed from text that carries no untrusted ranges; text that does keeps
    its ranges."""
    from localm.textnorm import strip_think
    content = message.get("content", "")
    text = content if isinstance(content, str) else _text_of(message)
    if not untrusted_spans_of(text):
        text = strip_think(str(text)).strip()
    if len(text) <= limit:
        return slice_guarded(text, 0, len(text))
    return compose(slice_guarded(text, 0, limit), " ...")


def digest_messages(older: List[dict]):
    """A bounded digest of *older*: one excerpt per message in conversation
    order; when the budget runs out the oldest messages are left out and
    counted."""
    per = max(DIGEST_MIN_EXCERPT, DIGEST_MAX_CHARS // max(1, len(older)))
    lines = []
    used = 0
    for m in reversed(older):
        line = compose(f"{str(m.get('role', 'user')).upper()}: ",
                       _excerpt_of(m, per))
        if lines and used + len(line) > DIGEST_MAX_CHARS:
            break
        lines.append(line)
        used += len(line)
    lines.reverse()
    omitted = len(older) - len(lines)
    parts = [_DIGEST_NOTE]
    if omitted:
        parts.append(f"({omitted} earlier message(s) omitted)")
    return compose_join("\n\n", [*parts, *lines])


def compact_messages(
    messages: List[dict],
    generate: Callable[[List[dict], int], str],
) -> Tuple[List[dict], bool]:
    """
    Summarise everything but the system prompt and the recent tail (see
    ``_split``). Returns (new_messages, changed).

    *generate(messages, max_tokens)* runs the model and returns its text.
    Any failure inside it, or an empty or all-reasoning reply, makes the
    bridge carry a digest of the removed turns instead (logged at WARNING);
    this function never raises.
    """
    head, older, recent = _split(messages)
    if not older:
        return messages, False

    excerpt = "\n\n".join(
        f"{m.get('role', 'user').upper()}: {_text_of(m)[:600]}" for m in older
    )
    summary_prompt = (
        "Summarise the following conversation in under 200 words. Keep "
        "facts, names, decisions, and anything the user asked to remember. "
        "Reply with the summary only.\n\n" + excerpt
    )

    summary = ""
    failure = ""
    try:
        summary = (generate(
            [{"role": "user", "content": summary_prompt}],
            SUMMARY_MAX_TOKENS,
        ) or "").strip()
    except Exception as e:
        summary = ""
        failure = f"{type(e).__name__}: {e}"
    # Keep only the visible answer; an all-reasoning reply becomes empty.
    from localm.textnorm import strip_think
    summary = strip_think(summary).strip()
    if not summary:
        from localm.debuglog import logger
        logger.warning(
            "compaction: summarisation unavailable (%s); keeping a digest of "
            "%d removed message(s)", failure or "empty or reasoning-only reply",
            len(older))

    if summary:
        bridge = [
            {"role": "user",
             "content": f"[Conversation summary]\n{summary}"},
            {"role": "assistant",
             "content": "Understood. Continuing from this summary."},
        ]
    else:
        bridge = [
            {"role": "user", "content": digest_messages(older)},
            {"role": "assistant",
             "content": "Understood. Continuing from these excerpts."},
        ]

    return [*head, *bridge, *recent], True


def maybe_compact(
    messages: List[dict],
    *,
    limit_tokens: int,
    generate: Callable[[List[dict], int], str],
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Tuple[List[dict], bool]:
    """
    Compact *messages* when they exceed COMPACT_RATIO of *limit_tokens*.

    limit_tokens <= 0 disables auto-compaction (unlimited window).
    Returns (messages, compacted).
    """
    if limit_tokens <= 0:
        return messages, False
    if estimate_tokens(messages, count_tokens) < COMPACT_RATIO * limit_tokens:
        return messages, False
    return compact_messages(messages, generate)
