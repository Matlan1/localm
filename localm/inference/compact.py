# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Conversation compaction for chat sessions.

When a chat history approaches the context ceiling, older turns are
summarised by the model itself and replaced with a compact summary
exchange, keeping the most recent turns verbatim. The kept tail starts at a
message the user wrote where one is available (see ``_split``), and the
latest user-written message is carried verbatim into the bridge when it falls
outside the tail; the bridge keeps user/assistant alternation. If
summarisation fails for any reason (model error, empty or all-reasoning
output), the bridge carries a bounded digest of excerpts from the removed
turns instead. Either way the function never raises and always returns a
usable history: chat keeps working instead of dying at the ceiling.

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
    messages: list[dict],
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


def _is_users_own(message: dict) -> bool:
    """True for a user message the user wrote: role ``user``, no ``origin``
    marker, and not a ``<tool_result`` block."""
    if message.get("role") != "user" or message.get("origin"):
        return False
    content = message.get("content", "")
    return not (isinstance(content, str) and content.lstrip().startswith("<tool_result"))


def _split(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(leading system messages, older middle, recent tail).

    The tail holds at least the last KEEP_RECENT messages. Its first message
    is, in order of preference: the nearest user-written message at or before
    the default cut, else the next one after it; else the nearest user
    message of any kind at or before the default cut; else the nearest
    assistant message at or before it, else the next one after it. With none
    of these, older is empty."""
    head = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    if len(rest) <= KEEP_RECENT:
        return head, [], rest
    default = len(rest) - KEEP_RECENT
    backward = range(default, 0, -1)
    forward = range(default + 1, len(rest))
    for test, scan in (
        (_is_users_own, backward),
        (_is_users_own, forward),
        (lambda m: m.get("role") == "user", backward),
        (lambda m: m.get("role") == "assistant", backward),
        (lambda m: m.get("role") == "assistant", forward),
    ):
        for i in scan:
            if test(rest[i]):
                return head, rest[:i], rest[i:]
    return head, [], rest


def _request_part(older: list[dict], recent: list[dict]):
    """A "Current request (verbatim):" section holding the last user-written
    message when it is in *older* and *recent* has none, else ``""``."""
    if any(_is_users_own(m) for m in recent):
        return ""
    for m in reversed(older):
        if _is_users_own(m):
            content = m.get("content", "")
            text = content if isinstance(content, str) else _text_of(m)
            return compose("\n\nCurrent request (verbatim):\n",
                           slice_guarded(text, 0, len(text)))
    return ""


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


def digest_messages(older: list[dict]):
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


def compactable(messages: list[dict]) -> bool:
    """True when ``compact_messages`` would change *messages*."""
    return bool(_split(messages)[1])


def compact_messages(
    messages: list[dict],
    generate: Callable[[list[dict], int], str],
) -> tuple[list[dict], bool]:
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

    body = (compose("[Conversation summary]\n", summary) if summary
            else digest_messages(older))
    body = compose(body, _request_part(older, recent))
    bridge = [{"role": "user", "content": body}]
    if recent and recent[0].get("role") == "user":
        bridge.append({
            "role": "assistant",
            "content": ("Understood. Continuing from this summary." if summary
                        else "Understood. Continuing from these excerpts.")})

    return [*head, *bridge, *recent], True


def maybe_compact(
    messages: list[dict],
    *,
    limit_tokens: int,
    generate: Callable[[list[dict], int], str],
    count_tokens: Optional[Callable[[str], int]] = None,
    on_compact: Optional[Callable[[], None]] = None,
) -> tuple[list[dict], bool]:
    """
    Compact *messages* when they exceed COMPACT_RATIO of *limit_tokens*.

    limit_tokens <= 0 disables auto-compaction (unlimited window).
    *on_compact*, when given, is called just before a compaction that will
    change *messages*. Returns (messages, compacted).
    """
    if limit_tokens <= 0:
        return messages, False
    if estimate_tokens(messages, count_tokens) < COMPACT_RATIO * limit_tokens:
        return messages, False
    if on_compact is not None and compactable(messages):
        on_compact()
    return compact_messages(messages, generate)
