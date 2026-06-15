"""
Readline history suppression for privacy mode.

This is kernel-level (not coder-specific): the plain ``localm`` chat REPL and the
coder plugin both call it to keep interactive input out of ``~/.python_history``
when privacy mode is active. Keeping it here means the kernel REPL does not have
to import the coder plugin package just to suppress history.
"""

from __future__ import annotations


def suppress_readline_history() -> None:
    """
    Prevent interactive REPL input from being persisted to ``~/.python_history``.

    Python's ``site.py`` registers an ``atexit`` handler that calls
    ``readline.write_history_file('~/.python_history')`` on interpreter exit.
    We cannot easily remove that handler without touching private internals,
    but we can defuse it:

    * ``set_history_length(0)`` - tells ``write_history_file`` to write 0
      entries when it runs.
    * ``clear_history()`` - empties the in-memory ring immediately so that
      nothing accumulated before this call leaks either.
    * A second ``atexit`` registration of ``clear_history`` runs *after* ours
      registers - since atexit is LIFO, ours fires first, clearing the buffer
      just before site.py's write handler runs (which then writes 0 entries).

    Safe no-op if readline is unavailable (Windows without pyreadline, or
    when Python was compiled without readline support).
    """
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
