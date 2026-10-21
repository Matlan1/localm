# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model calls made by unattended work (memory consolidation, scheduled jobs).

Such a call shares the loaded model with the chat the user is typing into. On a
model that answers one request at a time, a chat request arriving mid-call waits
for the whole call, so these calls run without the reasoning channel, under a
token cap, and stop as soon as a chat request is waiting.
"""

from __future__ import annotations

BACKGROUND_MAX_TOKENS = 1024


class BackgroundYield(BaseException):
    """A background model call stopped because a chat request is waiting for the
    model.

    Derives from BaseException, like ``asyncio.CancelledError``, so a step's own
    ``except Exception`` cannot treat it as a failed model call and carry on with
    the rest of its pass. The caller of the pass catches it and defers the work.
    """


def foreground_waiting(engine) -> bool:
    """True when a request is queued behind the generation now running on *engine*."""
    waiting = getattr(engine, "streams_waiting", 0)
    return isinstance(waiting, int) and not isinstance(waiting, bool) and waiting > 0


def background_complete(engine, prompt: str, *, yield_to_foreground: bool = False,
                        max_tokens: int = BACKGROUND_MAX_TOKENS) -> str:
    """Run *prompt* as one user turn and return the raw reply, reasoning included.

    The reply is generated with thinking off and at most *max_tokens* tokens. With
    *yield_to_foreground*, raises :class:`BackgroundYield` before starting, or
    between two tokens, once a request is waiting for the model; the generation is
    cancelled and the model is free for that request. A reply cut at *max_tokens*
    is logged as a warning and returned as it stands.
    """
    if yield_to_foreground and foreground_waiting(engine):
        raise BackgroundYield()
    stream = engine.chat_stream(
        [{"role": "user", "content": prompt}], max_tokens=max_tokens, thinking=False)
    parts: list[str] = []
    try:
        for chunk in stream:
            parts.append(chunk)
            if yield_to_foreground and foreground_waiting(engine):
                raise BackgroundYield()
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if getattr(engine, "last_finish_reason", "stop") == "length":
        from localm.debuglog import logger
        logger.warning("background model call hit its %d-token limit; the reply is "
                       "cut short", max_tokens)
    return "".join(parts)
