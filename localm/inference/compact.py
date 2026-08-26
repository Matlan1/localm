# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Conversation compaction for chat sessions.

When a chat history approaches the context ceiling, older turns are
summarised by the model itself and replaced with a compact summary
exchange, keeping the most recent turns verbatim. If summarisation fails
for any reason (model error, empty output), the fallback is a hard trim -
older messages are simply dropped with a visible note. Either way the
function never raises and always returns a usable history: chat keeps
working instead of dying at the ceiling.

Used by the CLI interactive chat; the GUI implements the same protocol
client-side against /v1/chat/completions.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

# Fraction of the context ceiling at which compaction kicks in
COMPACT_RATIO = 0.70

# Most recent messages kept verbatim (≈ the last two full turns)
KEEP_RECENT = 4

# Budget for the generated summary. Thinking-family models spend their first
# few hundred tokens on the reasoning channel (stripped before storage), so this
# leaves room for the visible summary AFTER the scratchpad.
SUMMARY_MAX_TOKENS = 1024

_TRIM_NOTE = (
    "[Earlier conversation was removed to fit the context window. "
    "Summarisation was unavailable; continue from the recent messages.]"
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
    """(leading system messages, older middle, recent tail)."""
    head = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    if len(rest) <= KEEP_RECENT:
        return head, [], rest
    return head, rest[:-KEEP_RECENT], rest[-KEEP_RECENT:]


def compact_messages(
    messages: List[dict],
    generate: Callable[[List[dict], int], str],
) -> Tuple[List[dict], bool]:
    """
    Summarise everything but the system prompt and the last KEEP_RECENT
    messages. Returns (new_messages, changed).

    *generate(messages, max_tokens)* runs the model and returns its text.
    Any failure inside it triggers the hard-trim fallback - this function
    never raises.
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
    try:
        summary = (generate(
            [{"role": "user", "content": summary_prompt}],
            SUMMARY_MAX_TOKENS,
        ) or "").strip()
    except Exception:
        summary = ""
    # A thinking model may spend the whole budget on its reasoning channel.
    # Keep only the visible answer; an all-reasoning reply becomes empty and
    # takes the hard-trim fallback below.
    from localm.textnorm import strip_think
    summary = strip_think(summary).strip()

    if summary:
        bridge = [
            {"role": "user",
             "content": f"[Conversation summary]\n{summary}"},
            {"role": "assistant",
             "content": "Understood. Continuing from this summary."},
        ]
    else:
        # Hard trim - drop the middle entirely, but say so
        bridge = [
            {"role": "user", "content": _TRIM_NOTE},
            {"role": "assistant", "content": "Understood."},
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
