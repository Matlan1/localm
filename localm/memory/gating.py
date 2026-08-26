# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Privacy gate for the memory layer (the one place that imports the session mode).

A memory row is a durable, session-derived trace, so EVERY write to it (add,
update, delete, reinforcement bump, consolidation, the legacy migration import)
must be gated on the effective session mode, exactly like chat history and coder
episodes. ``writes_allowed(surface)`` is the single decision point callers use
before writing; reads/recall of memories from earlier non-privacy sessions stay
allowed ("no new traces", not amnesia). A blocked write must surface a skip, never
report success.

Kept in its own leaf module so ``store.py`` stays free of the audit import and is
trivially unit-testable, and so callers (the chat inlet/routes, the coder) share
one gate.
"""

from __future__ import annotations


def writes_allowed(surface: str) -> bool:
    """True when durable memory writes are permitted for *surface* ("chat" |
    "coder" | "server"). False in privacy mode (the default). Resolved fresh each
    call so a mid-run config/env change is honoured."""
    from localm.audit import SessionMode, effective_mode
    return effective_mode(surface) != SessionMode.PRIVACY
