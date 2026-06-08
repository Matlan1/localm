"""
Privacy-mode helpers for localcoder.

When ``--mode privacy`` is active this module is responsible for suppressing
every data-persistence point that the process can control:

  1. Python readline history (``~/.python_history``) — suppressed at startup.
  2. Subprocess shell history (bash/sh ``HISTFILE``) — suppressed per call.
  3. External-provider warning — printed when prompts would leave the machine.

What privacy mode CANNOT suppress (document honestly):
  - The ``localcoder`` command itself in the user's terminal history
    (PSReadLine / bash history).  The user typed it; we can't unsign it.
  - OS-level process-creation logs (Windows Event Log, Linux auditd/syslog).
    Requires elevated privileges to disable and is outside our scope.
  - DNS queries and network-layer logs from ``fetch_url``.  Route through
    a local resolver / VPN if that matters to you.
  - Files the *agent* is explicitly asked to write (write_file, patch_file).
    Those are intentional outputs, not traces.
"""

from __future__ import annotations

import os
import sys

from .display import console


# ---------------------------------------------------------------------------
#  1. Python readline history suppression
# ---------------------------------------------------------------------------

def suppress_readline_history() -> None:
    """
    Prevent interactive REPL input from being persisted to ``~/.python_history``.

    Python's ``site.py`` registers an ``atexit`` handler that calls
    ``readline.write_history_file('~/.python_history')`` on interpreter exit.
    We cannot easily remove that handler without touching private internals,
    but we can defuse it:

    * ``set_history_length(0)`` — tells ``write_history_file`` to write 0
      entries when it runs.
    * ``clear_history()`` — empties the in-memory ring immediately so that
      nothing accumulated before this call leaks either.
    * A second ``atexit`` registration of ``clear_history`` runs *after* ours
      registers — since atexit is LIFO, ours fires first, clearing the buffer
      just before site.py's write handler runs (which then writes 0 entries).

    Safe no-op if readline is unavailable (Windows without pyreadline, or
    when Python was compiled without readline support).
    """
    try:
        import readline as _rl
        _rl.set_history_length(0)
        _rl.clear_history()

        import atexit as _atexit
        # LIFO: registered last → runs first.
        # Clears whatever the REPL accumulated just before site.py's write.
        _atexit.register(_rl.clear_history)
    except (ImportError, AttributeError):
        pass   # readline not available — nothing to suppress


# ---------------------------------------------------------------------------
#  2. Subprocess environment
# ---------------------------------------------------------------------------

def subprocess_privacy_env() -> dict[str, str]:
    """
    Return a copy of the current environment with shell history vars zeroed.

    Used by ``tool_run_shell`` in privacy mode so that any bash/sh/zsh child
    process cannot write command history to disk.  For non-interactive shells
    (our normal case) these are no-ops, but they are an explicit statement of
    intent and guard against edge cases where a script opens an interactive
    sub-shell.

    Variables overridden:
      HISTFILE       — path where bash/zsh writes history on exit.
      HISTSIZE       — in-memory history depth (0 = disabled in bash).
      HISTFILESIZE   — max lines written to HISTFILE (0 = truncate to empty).
      HISTIGNORE     — ``*`` ignores every command in bash history.
      HISTCONTROL    — ``ignorespace:ignoredups`` (belt-and-suspenders).
      LESSHIST(FILE) — less pager history.
      MYSQL_HISTFILE — mysql CLI history.
      SQLITE_HISTORY — sqlite3 CLI history.

    We deliberately do NOT set env vars for fish or PowerShell because:
      * fish: non-interactive fish sessions never save history regardless.
      * PowerShell: PSReadLine only runs in interactive sessions; our
        subprocesses use ``cmd.exe /C`` (Windows) or ``/bin/sh -c`` (Unix),
        neither of which loads PSReadLine.
    """
    env = dict(os.environ)

    null = "NUL" if sys.platform == "win32" else os.devnull

    env.update({
        "HISTFILE":      null,
        "HISTSIZE":      "0",
        "HISTFILESIZE":  "0",
        "HISTIGNORE":    "*",
        "HISTCONTROL":   "ignorespace:ignoredups",
        "LESSHISTFILE":  null,
        "MYSQL_HISTFILE": null,
        "SQLITE_HISTORY": null,
    })

    return env


# ---------------------------------------------------------------------------
#  3. External-provider privacy warning
# ---------------------------------------------------------------------------

_PROVIDER_NAMES = {
    "openai":    "OpenAI",
    "anthropic": "Anthropic",
}

def warn_external_provider(provider: str) -> None:
    """
    Print a prominent warning when privacy mode is active but prompts will be
    sent to an external API.

    Privacy mode suppresses *local* persistence (no log files, no readline
    history), but it cannot prevent the API provider from receiving, logging,
    or training on your prompts.  The user needs to know this.
    """
    name = _PROVIDER_NAMES.get(provider, provider)
    console.print(
        f"\n[bold yellow]⚠  Privacy mode + {name} API[/bold yellow]\n"
        f"[yellow]   Your prompts will be sent to {name}'s servers.\n"
        f"   {name} may log or use them per their privacy policy.\n"
        f"   Use a local model (--model) for full privacy.[/yellow]\n"
    )
