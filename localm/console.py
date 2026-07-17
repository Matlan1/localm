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
