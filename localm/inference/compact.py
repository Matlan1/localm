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

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Callable, Optional

from localm.textguard import (
    compose, compose_join, slice_guarded, untrusted_spans_of,
)

# Fraction of the context ceiling at which compaction kicks in
COMPACT_RATIO = 0.70

# Fraction of the context ceiling a compaction aims to bring the conversation
# down to when the window is known.
COMPACT_TARGET = 0.50

# Smallest reply buffer, in tokens, kept free in the window, and the share of
# the window used instead when that is larger.
REPLY_RESERVE_MIN = 2048
REPLY_RESERVE_SHARE = 0.10

# Summaries a CompactionCache keeps.
CACHE_ENTRIES = 64

# Minimum number of most recent messages kept verbatim; the cut moves back to
# the nearest user turn, so the tail can be longer.
KEEP_RECENT = 4

# Largest reply asked of the summariser, which answers without its reasoning
# channel; a small window gets a quarter of the window, at least
# SUMMARY_MIN_TOKENS.
SUMMARY_MAX_TOKENS = 512
SUMMARY_MIN_TOKENS = 64

# Most summariser requests one compaction makes.
MAX_SUMMARY_PASSES = 8

# Smallest excerpt room, in tokens, a rolling summary pass is made for.
MIN_CHUNK_TOKENS = 48

# Smallest share of a rolling pass, in tokens, one excerpt is cut down to.
MIN_ENTRY_TOKENS = 24

# Tokens a rolling pass keeps free for the earlier summary's labels and an
# omitted-messages note, on top of the earlier summary itself.
ROLLING_RESERVE_EXTRA = 32

# Characters per token assumed when no tokenizer is available.
CHARS_PER_TOKEN = 3

# Character budget of the fallback digest of removed turns, and the smallest
# excerpt kept per turn.
DIGEST_MAX_CHARS = 4096
DIGEST_MIN_EXCERPT = 160

SUMMARY_INSTRUCTION = (
    "Summarise the following conversation in under 200 words. Keep "
    "facts, names, decisions, and anything the user asked to remember. "
    "Reply with the summary only."
)
ROLLING_SUMMARY_INSTRUCTION = (
    "Summarise the following conversation in under 200 words. It begins "
    "with a summary of its earlier part: keep the facts, names, decisions, "
    "and anything the user asked to remember from that summary and from the "
    "later turns. Reply with the summary only."
)

_DIGEST_NOTE = (
    "[Earlier conversation was condensed to fit the context window. "
    "Summarisation was unavailable, so these are excerpts of the earlier "
    "turns:]"
)


def _text_of(message: dict) -> str:
    """Plain text of a message; multipart images and audio clips contribute a placeholder."""
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
            elif part.get("type") == "input_audio":
                parts.append("[audio]")
    return " ".join(parts)


def estimate_tokens(
    messages: list[dict],
    count_tokens: Optional[Callable[[str], int]] = None,
) -> int:
    """Token estimate for a message list (images and audio clips count a flat 750 each)."""
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
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_audio")
            )
    return total


def _is_users_own(message: dict) -> bool:
    """True for a user message the user wrote: role ``user``, no ``origin``
    marker, and not a ``<tool_result`` block."""
    if message.get("role") != "user" or message.get("origin"):
        return False
    content = message.get("content", "")
    return not (isinstance(content, str) and content.lstrip().startswith("<tool_result"))


def _split(messages: list[dict], keep_recent: int = KEEP_RECENT
           ) -> tuple[list[dict], list[dict], list[dict]]:
    """(leading system messages, older middle, recent tail).

    The tail holds at least the last *keep_recent* messages. Its first message
    is, in order of preference: the nearest user-written message at or before
    the default cut, else the next one after it; else the nearest user
    message of any kind at or before the default cut; else the nearest
    assistant message at or before it, else the next one after it. With none
    of these, older is empty."""
    head = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    keep_recent = max(KEEP_RECENT, keep_recent)
    if len(rest) <= keep_recent:
        return head, [], rest
    default = len(rest) - keep_recent
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


def window_tokens(value) -> int:
    """*value* when it is a positive int, else 0 (unknown window)."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def summary_reply_tokens(limit_tokens: int, cap: int = SUMMARY_MAX_TOKENS) -> int:
    """The max_tokens of one summariser request in a window of *limit_tokens*
    tokens: *cap* for an unknown window (0), else a quarter of the window
    capped at *cap* and never below SUMMARY_MIN_TOKENS."""
    limit_tokens = window_tokens(limit_tokens)
    if not limit_tokens:
        return cap
    return max(SUMMARY_MIN_TOKENS, min(cap, limit_tokens // 4))


def template_margin(limit_tokens: int) -> int:
    """Tokens of a window of *limit_tokens* kept free for the chat template."""
    return 64 + window_tokens(limit_tokens) // 64


def token_meter(count_tokens: Optional[Callable[[str], int]] = None
                ) -> Callable[[str], int]:
    """A function measuring text in tokens: the larger of an estimate (one
    token per CHARS_PER_TOKEN ASCII characters, rounded up, plus one per
    non-ASCII character) and *count_tokens* when it returns a non-negative
    int. A *count_tokens* that raises is logged at DEBUG and the estimate is
    used. See test_a_broken_counter_is_measured_with_the_estimate."""
    def measure(text) -> int:
        text = str(text)
        wide = sum(1 for c in text if ord(c) > 127)
        estimate = wide + -(-(len(text) - wide) // CHARS_PER_TOKEN)
        if count_tokens is not None:
            try:
                n = count_tokens(text)
            except Exception:
                from localm.debuglog import logger
                logger.debug("compaction: token count failed; using the estimate",
                             exc_info=True)
                return estimate
            if isinstance(n, int) and not isinstance(n, bool) and n > estimate:
                return n
        return estimate
    return measure


def earlier_summary_tokens(reply_tokens: int) -> int:
    """Tokens a rolling pass gives the summary so far: four thirds of
    *reply_tokens*, the measure of a reply of that many tokens at four
    characters per token."""
    return reply_tokens * 4 // 3


def fit_tokens(text, tokens: int, measure: Callable[[str], int]):
    """*text* when it measures at most *tokens*, else its start followed by
    " ...", cut so the whole measures at most *tokens* where any start can.
    Untrusted ranges of *text* are kept over the part that survives."""
    if measure(text) <= tokens:
        return text
    raw = str(text)
    keep = int(len(raw) * max(0, tokens) / max(1, measure(text)))
    for _ in range(40):
        cut = compose(slice_guarded(text, 0, keep), " ...")
        if keep <= 0 or measure(cut) <= tokens:
            return cut
        keep = int(keep * 0.85)
    return compose(slice_guarded(text, 0, 0), " ...")


def plan_summary_passes(
    entries: list,
    *,
    limit_tokens: int,
    reply_tokens: int,
    overhead_tokens: int,
    measure: Callable[[str], int],
    has_earlier: bool = False,
) -> list[list]:
    """Split *entries* (one excerpt per removed message, oldest first) into
    the excerpt groups of successive summariser requests. With *has_earlier*
    the first request also carries an earlier summary, so every group is
    sized for a rolling pass.

    One group holding every entry when the window is unknown (0), when they
    fit one request (excerpt + *overhead_tokens* + *reply_tokens* + the
    template margin within *limit_tokens*), or when the window leaves less
    than MIN_CHUNK_TOKENS for a rolling pass. Otherwise groups that each fit
    a rolling pass, whose prompt also carries an earlier summary of up to
    ``earlier_summary_tokens(reply_tokens)`` tokens; an entry larger than a
    group is cut to fit. With
    more than MAX_SUMMARY_PASSES groups, entries are cut to an equal share
    (at least MIN_ENTRY_TOKENS) and regrouped; groups past the bound are
    left out from the middle (the first group and the newest ones are kept)
    and the first kept group after the gap opens with an
    "(N message(s) omitted)" line."""
    limit_tokens = window_tokens(limit_tokens)
    if not entries or not limit_tokens:
        return [list(entries)]
    sep = measure("\n\n")
    room = limit_tokens - reply_tokens - template_margin(limit_tokens) - overhead_tokens
    sizes = [measure(e) for e in entries]
    chunk_room = room - earlier_summary_tokens(reply_tokens) - ROLLING_RESERVE_EXTRA
    if sum(sizes) + sep * (len(entries) - 1) <= (chunk_room if has_earlier else room):
        return [list(entries)]
    if chunk_room < MIN_CHUNK_TOKENS:
        return [list(entries)]

    def _group(cap: int) -> list[list]:
        groups: list[list] = []
        used = 0
        for entry, size in zip(entries, sizes, strict=True):
            if size > cap:
                entry = fit_tokens(entry, cap, measure)
                size = measure(entry)
            if groups and used + sep + size <= chunk_room:
                groups[-1].append(entry)
                used += sep + size
            else:
                groups.append([entry])
                used = size
        return groups

    groups = _group(chunk_room)
    if len(groups) > MAX_SUMMARY_PASSES:
        share = MAX_SUMMARY_PASSES * chunk_room // len(entries) - sep
        groups = _group(max(MIN_ENTRY_TOKENS, min(chunk_room, share)))
    if len(groups) > MAX_SUMMARY_PASSES:
        keep_new = MAX_SUMMARY_PASSES - 1
        dropped = groups[1:len(groups) - keep_new]
        omitted = sum(len(g) for g in dropped)
        groups = [groups[0], *groups[len(groups) - keep_new:]]
        groups[1] = [f"({omitted} message(s) omitted)", *groups[1]]
    return groups


def summarise_in_passes(
    groups: list[list],
    run_pass: Callable[[object, object], str],
    *,
    reply_tokens: int,
    measure: Callable[[str], int],
    earlier: str = "",
) -> tuple[str, str]:
    """Summarise *groups* (from ``plan_summary_passes``) oldest first,
    continuing from the summary *earlier* when one is given.

    *run_pass(earlier, excerpt)* makes one summariser request and returns its
    text: *earlier* is None on the first pass when no earlier summary was
    given, else the summary so far, cut to
    ``earlier_summary_tokens(reply_tokens)`` tokens; *excerpt* is the group's
    entries joined by blank lines. Returns ``(summary, "")`` with the last pass's visible text, or
    ``("", reason)`` when a pass raises or returns no visible text (the
    remaining passes are not made)."""
    from localm.textnorm import strip_think
    summary = str(earlier or "")
    for group in groups:
        earlier = (fit_tokens(summary, earlier_summary_tokens(reply_tokens), measure)
                   if summary else None)
        try:
            out = run_pass(earlier, compose_join("\n\n", group))
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"
        # Keep only the visible answer; an all-reasoning reply becomes empty.
        out = strip_think(str(out or "")).strip()
        if not out:
            return "", "empty or reasoning-only reply"
        summary = out
    return summary, ""


def reply_reserve(capacity) -> int:
    """Tokens of a window of *capacity* tokens kept free for the reply:
    REPLY_RESERVE_MIN, or REPLY_RESERVE_SHARE of the window when that is
    larger. 0 for an unknown window."""
    capacity = window_tokens(capacity)
    if not capacity:
        return 0
    return max(REPLY_RESERVE_MIN, int(capacity * REPLY_RESERVE_SHARE))


def _message_key(message: dict) -> str:
    """Canonical JSON of the parts of *message* a model reads."""
    keyed = {k: message.get(k) for k in ("role", "content", "name", "tool_calls",
                                          "tool_call_id") if k in message}
    return json.dumps(keyed, sort_keys=True, ensure_ascii=False, default=str)


def prefix_digests(messages: list[dict]) -> list[str]:
    """``digests[i]`` identifies ``messages[:i]``: a SHA-256 chain over each
    message's canonical JSON, so two lists share ``digests[i]`` exactly when
    their first *i* messages are equal. ``len(messages) + 1`` entries."""
    h = hashlib.sha256(b"localm-compaction-prefix")
    out = [h.hexdigest()]
    for m in messages:
        h.update(b"\x00")
        h.update(_message_key(m).encode("utf-8", "surrogatepass"))
        out.append(h.copy().hexdigest())
    return out


class CompactionCache:
    """Summaries of conversation prefixes, by ``(namespace, prefix digest)``.

    A later request that starts with a summarised prefix reuses its summary
    instead of summarising again. Held in memory only; the least recently used
    entry is dropped past *max_entries*. Thread-safe."""

    def __init__(self, max_entries: int = CACHE_ENTRIES) -> None:
        self._max = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: OrderedDict = OrderedDict()

    def get(self, namespace: str, digest: str) -> Optional[str]:
        """The summary stored for *digest* under *namespace*, or None."""
        key = (namespace, digest)
        with self._lock:
            summary = self._entries.get(key)
            if summary is not None:
                self._entries.move_to_end(key)
            return summary

    def put(self, namespace: str, digest: str, summary: str) -> None:
        """Store *summary* for *digest* under *namespace*."""
        key = (namespace, digest)
        with self._lock:
            self._entries[key] = summary
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _bridge(summary, older: list[dict], recent: list[dict]) -> list[dict]:
    """The bridge messages replacing *older*: the summary (or, with no
    summary, a digest of *older*) plus the latest user-written request when
    it is in *older* only, and an assistant acknowledgement when *recent*
    starts with a user message."""
    body = (compose("[Conversation summary]\n", summary) if summary
            else digest_messages(older))
    body = compose(body, _request_part(older, recent))
    bridge = [{"role": "user", "content": body}]
    if recent and recent[0].get("role") == "user":
        bridge.append({
            "role": "assistant",
            "content": ("Understood. Continuing from this summary." if summary
                        else "Understood. Continuing from these excerpts.")})
    return bridge


def _tail_count(rest: list[dict], keep_tokens: int,
                measure: Callable[[str], int]) -> int:
    """How many of the newest messages of *rest* fit *keep_tokens* (each
    measured with a 4-token allowance for its role and delimiters), at least
    KEEP_RECENT."""
    used = kept = 0
    for m in reversed(rest):
        size = measure(_text_of(m)) + 4
        if kept >= KEEP_RECENT and used + size > keep_tokens:
            break
        used += size
        kept += 1
    return max(KEEP_RECENT, kept)


def compact_messages(
    messages: list[dict],
    generate: Callable[[list[dict], int], str],
    *,
    limit_tokens: int = 0,
    count_tokens: Optional[Callable[[str], int]] = None,
    target_tokens: int = 0,
    cache: Optional[CompactionCache] = None,
    namespace: str = "",
    fits: Optional[Callable[[list[dict]], bool]] = None,
) -> tuple[list[dict], bool]:
    """
    Summarise everything but the system prompt and the recent tail (see
    ``_split``). Returns (new_messages, changed).

    *generate(messages, max_tokens)* runs the model and returns its text.
    *limit_tokens* is the model's context window (0 when unknown) and
    *count_tokens* its tokenizer (see ``token_meter``): each summariser
    request, prompt plus max_tokens, is sized to fit the window, and removed
    turns that do not fit one request are summarised in rolling passes (see
    ``plan_summary_passes``). One request carries every removed turn, unsized,
    when the window is unknown or leaves less than MIN_CHUNK_TOKENS for a
    rolling pass. Any failure inside *generate*, or an empty or
    all-reasoning reply, makes the bridge carry a digest of the removed turns
    instead (logged at WARNING); this function never raises.

    *target_tokens*, when positive, is the size the compacted conversation
    aims for: the tail keeps as many of the newest messages (at least
    KEEP_RECENT) as fit beside the system prompt and a summary of up to the
    reply budget. Without it the tail is the last KEEP_RECENT messages.

    With *cache*, a summary is stored under *namespace* for the prefix it
    covers. A conversation that starts with a stored prefix reuses that
    summary unchanged, with no model call, when the result passes *fits*
    (always, when *fits* is None); otherwise only the messages after the
    stored prefix are summarised, continuing from the stored summary. A digest
    fallback is never stored.
    """
    head, older, recent = _split(messages)
    if not older:
        return messages, False
    rest = messages[len(head):]

    limit_tokens = window_tokens(limit_tokens)
    measure = token_meter(count_tokens)
    reply_tokens = summary_reply_tokens(limit_tokens)

    digests = prefix_digests(messages) if cache is not None else []
    start, earlier = 0, ""
    if cache is not None:
        for i in range(len(rest) - 1, 0, -1):
            hit = cache.get(namespace, digests[len(head) + i])
            if hit is None:
                continue
            reused = [*head, *_bridge(hit, rest[:i], rest[i:]), *rest[i:]]
            if fits is None or fits(reused):
                return reused, True
            start, earlier = i, hit
            break

    if target_tokens > 0:
        head_tokens = sum(measure(_text_of(m)) + 4 for m in head)
        keep = target_tokens - head_tokens - reply_tokens - template_margin(limit_tokens)
        budget_split = _split(messages, _tail_count(rest, keep, measure))
        if budget_split[1]:
            head, older, recent = budget_split
    cut = len(older)
    if start and cut <= start:
        return [*head, *_bridge(earlier, rest[:start], rest[start:]),
                *rest[start:]], True
    fresh = rest[start:cut]

    entries = [f"{m.get('role', 'user').upper()}: {_text_of(m)[:600]}"
               for m in fresh]
    groups = plan_summary_passes(
        entries, limit_tokens=limit_tokens, reply_tokens=reply_tokens,
        overhead_tokens=measure(ROLLING_SUMMARY_INSTRUCTION + "\n\n"),
        measure=measure, has_earlier=bool(earlier))

    def _pass(earlier, excerpt) -> str:
        if earlier is None:
            prompt = SUMMARY_INSTRUCTION + "\n\n" + str(excerpt)
        else:
            prompt = (ROLLING_SUMMARY_INSTRUCTION
                      + "\n\nSummary of the earlier part:\n" + str(earlier)
                      + "\n\nLater turns:\n" + str(excerpt))
        return generate([{"role": "user", "content": prompt}], reply_tokens)

    summary, failure = summarise_in_passes(
        groups, _pass, reply_tokens=reply_tokens, measure=measure,
        earlier=earlier)
    if not summary:
        from localm.debuglog import logger
        logger.warning(
            "compaction: summarisation unavailable (%s); keeping a digest of "
            "%d removed message(s)", failure, len(fresh))
        if earlier:
            return [*head, *_bridge(compose(earlier, "\n\n", digest_messages(fresh)),
                                    older, recent), *recent], True
    elif cache is not None:
        cache.put(namespace, digests[len(head) + cut], summary)

    return [*head, *_bridge(summary, older, recent), *recent], True


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
    return compact_messages(messages, generate, limit_tokens=limit_tokens,
                            count_tokens=count_tokens)
