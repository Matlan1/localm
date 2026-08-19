# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single shared Rich ``Console`` for output that can be reached from more
than one thread during startup.

``localm gui``/``localm serve`` preload the model on a background thread so
the GUI URL prints immediately instead of waiting for the (possibly slow)
load (see ``plugins/gui/cli.py``'s ``_preload``). That background thread
prints through the model-load call chain (``engine.load()``, the GGUF/HF
backends, VRAM auto-sizing) at the same time the main thread is still
printing its own startup banner lines. Rich's ``Console.print()`` is
thread-safe, but only against OTHER prints on the SAME instance (it holds an
internal lock on ``self``); two independent ``Console()`` objects writing to
the same stdout share no lock, so their multi-segment writes (plain text +
ANSI style codes) can interleave character-by-character. Every module in that
preload call chain imports THIS one instance instead of creating its own, so
their prints are actually serialized.
"""

from __future__ import annotations

from rich.console import Console

console = Console()

def show_url(url: str) -> str:
    """*url* made safe to interpolate into a ``console.print`` markup string.

    Rich reads ``[...]`` as a style tag, and an IPv6 URL authority is bracketed
    by RFC 3986 - so a printed address SILENTLY LOSES ITS HOST when the literal
    happens to start with a lowercase hex letter, which is every link-local
    (``fe80::``) and every unique-local (``fd..``) address. Measured: Rich
    renders ``[fd7a:115c:a1e0::e44:2839]`` as the empty string, while ``[::1]``
    and ``[2001:db8::5]`` survive because Rich's tag pattern needs a letter,
    ``#``, ``/`` or ``@`` after the bracket. So the bug reaches only SOME
    addresses, which is exactly the kind that gets shipped: the common test
    cases print correctly.

    Escaping here rather than inside ``bindhost.url_host`` is deliberate -
    ``url_host`` builds URLs that are also handed to requests and to sockets,
    where a Rich escape would be corruption. This is a display concern and it
    belongs at the display boundary."""
    from rich.markup import escape
    return escape(url)
