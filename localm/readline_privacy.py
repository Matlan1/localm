# SPDX-License-Identifier: AGPL-3.0-or-later
"""Readline history suppression for privacy mode."""

from __future__ import annotations


def suppress_readline_history() -> None:
    """Prevent interactive REPL input from being persisted to ``~/.python_history``."""
    try:
        import readline as _rl
        _rl.set_history_length(0)
        _rl.clear_history()

        import atexit as _atexit
        # LIFO: registered last runs first.
        # Clears whatever the REPL accumulated just before site.py's write.
        _atexit.register(_rl.clear_history)
    except (ImportError, AttributeError):
        pass   # readline not available - nothing to suppress
