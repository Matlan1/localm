# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single shared Rich ``Console`` for output that can be reached from more
than one thread during startup.

Rich's ``Console.print()`` is thread-safe only against OTHER prints on the SAME
instance (it holds an internal lock on ``self``); two independent ``Console()``
objects writing to the same stdout share no lock, so their multi-segment writes
(plain text plus ANSI style codes) can interleave character by character.

``localm gui``/``localm serve`` preload the model on a background thread (see
``plugins/gui/cli.py``'s ``_preload``) while the main thread is still printing
its startup banner, so every module in that preload call chain - ``engine.
load()``, the GGUF/HF backends, VRAM auto-sizing - imports THIS one instance
rather than creating its own.
"""

from __future__ import annotations

from rich.console import Console

console = Console()

def show_url(url: str) -> str:
    """*url* made safe to interpolate into a ``console.print`` markup string.

    Rich reads ``[...]`` as a style tag, and an IPv6 URL authority is bracketed
    by RFC 3986, so an unescaped address whose literal starts with a lowercase
    hex letter - every link-local (``fe80::``) and unique-local (``fd..``) one -
    renders as the empty string. ``[::1]`` and ``[2001:db8::5]`` survive,
    because Rich's tag pattern needs a letter, ``#``, ``/`` or ``@`` after the
    bracket.

    The escape belongs HERE, not inside ``bindhost.url_host``: that builds URLs
    also handed to requests and to sockets, where a Rich escape is
    corruption."""
    from rich.markup import escape
    return escape(url)
