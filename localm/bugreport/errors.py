# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reportable failure types: a user-facing failure, and a rate-limited upload.
"""

from __future__ import annotations

from typing import Optional

class LocalmError(Exception):
    """A user-facing, reportable failure.

    Raise this (not a bare Exception) when a command hits a known, describable
    problem. The CLI's single graceful handler turns it into a "sorry, X went
    wrong because Y" message and offers a bug report - the command that raises it
    knows nothing about reporting. Carries a human *summary*, an optional
    *reason*, and diagnostic *context* for the report."""

    def __init__(self, summary: str, reason: str = "",
                 context: Optional[dict] = None, *,
                 stage: Optional[str] = None, hint: Optional[str] = None):
        super().__init__(summary)
        self.summary = summary
        self.reason = reason
        self.context = context or {}
        # For an UPLOAD failure: WHERE it failed (offline_or_dns / unreachable /
        # tls / timeout / server_rejected / no_endpoint / rate_limited / unknown)
        # and a friendly, actionable *hint* to show the user. Both None for a
        # non-upload LocalmError.
        self.stage = stage
        self.hint = hint


class RateLimitedError(LocalmError):
    """The bug-report proxy returned HTTP 429 (rate limited). Carries *retry_after*
    (seconds) so a caller can wait and retry once instead of failing outright. A
    subclass of LocalmError, so existing ``except LocalmError`` handlers still treat
    it as a (non-fatal) send failure; callers that want the retry catch it first."""

    def __init__(self, retry_after: int = 30, reason: str = ""):
        try:
            ra = max(1, int(retry_after))
        except (TypeError, ValueError):
            ra = 30
        super().__init__("the bug-report server is rate limiting reports",
                         reason=reason or f"try again in about {ra} seconds",
                         context={"retry_after": ra},
                         stage="rate_limited",
                         hint=(f"The bug-report server is busy right now and asked to "
                               f"wait about {ra}s before trying again. Your report is "
                               f"saved either way."))
        self.retry_after = ra
