# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single shared Rich ``Console`` for output that can be reached from more
than one thread during startup, and the terminal-safe console class every
localm console is built from.

Rich's ``Console.print()`` is thread-safe only against OTHER prints on the SAME
instance (it holds an internal lock on ``self``); two independent ``Console()``
objects writing to the same stdout share no lock, so their multi-segment writes
(plain text plus ANSI style codes) can interleave character by character.

``localm gui``/``localm serve`` preload the model on a background thread (see
``plugins/gui/cli.py``'s ``_preload``) while the main thread is still printing
its startup banner, so every module in that preload call chain - ``engine.
load()``, the GGUF/HF backends, VRAM auto-sizing - imports THIS one instance
rather than creating its own.

``strip_controls`` removes every C0 control character except newline and tab,
DEL, and every C1 control character, so a string can carry no ANSI CSI/OSC
sequence (screen clear, cursor movement, title or clipboard write, hyperlink)
and no single-byte terminal command such as ``ESC c``.

``SafeConsole`` is a drop-in ``rich.console.Console`` that applies
``strip_controls`` to the text of every segment it renders, whatever the
renderable (a markup string, ``Text``, ``Syntax``, ``Table``, ``Panel``, a
``Progress`` description). Rich's own styling is applied after that step and
its cursor control travels as control segments, so both are unaffected. Markup
in a string is still parsed: text localm did not author that is interpolated
into markup also needs ``markup_safe`` (or ``rich.markup.escape``).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from rich.console import Console, ConsoleOptions, RenderableType
from rich.segment import Segment

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_controls(text: str) -> str:
    """*text* without C0 controls (except ``\\n`` and ``\\t``), DEL and C1 controls."""
    return _CONTROL_CHARS.sub("", text) if _CONTROL_CHARS.search(text) else text


def markup_safe(value: object) -> str:
    """``str(value)`` with its control characters removed and Rich markup
    escaped, ready to interpolate into a ``console.print`` markup string."""
    from rich.markup import escape
    return escape(strip_controls(str(value)))


class SafeConsole(Console):
    """A ``Console`` whose rendered text never carries a terminal control character."""

    def render(self, renderable: RenderableType,
               options: Optional[ConsoleOptions] = None) -> Iterable[Segment]:
        for segment in super().render(renderable, options):
            if segment.control or not _CONTROL_CHARS.search(segment.text):
                yield segment
            else:
                yield Segment(strip_controls(segment.text), segment.style)


console = SafeConsole()

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
