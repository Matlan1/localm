# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single shared Rich ``Console`` for output that can be reached from more than one thread during startup."""

from __future__ import annotations

from rich.console import Console

console = Console()

def show_url(url: str) -> str:
    """*url* made safe to interpolate into a ``console.print`` markup string."""
    from rich.markup import escape
    return escape(url)
