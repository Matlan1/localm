# SPDX-License-Identifier: AGPL-3.0-or-later
"""Privacy gate for the memory layer (the one place that imports the session mode)."""

from __future__ import annotations


def writes_allowed(surface: str) -> bool:
    """True when durable memory writes are permitted for *surface* ('chat' | 'coder' | 'server')."""
    from localm.audit import SessionMode, effective_mode
    return effective_mode(surface) != SessionMode.PRIVACY
