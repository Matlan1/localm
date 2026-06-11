"""
Privacy-mode helpers for localcoder.

When ``--mode privacy`` is active this module is responsible for suppressing
every data-persistence point that the process can control:

  1. Python readline history (``~/.python_history``) — suppressed at startup.
  2. Subprocess shell history (bash/sh ``HISTFILE``) — suppressed per call.
  3. Shell history files cleaned at exit — PSReadLine, bash, zsh history files
     are scrubbed of lines referencing the binary name.
  4. External-provider warning — printed when prompts would leave the machine.

Shell history coverage by shell:
  cmd.exe      No persistent history at all.  Nothing to do.
  PowerShell   PSReadLine writes incrementally to a plain-text file.  We
               remove any localcoder-referencing lines on exit.
  bash/zsh     $HISTFILE (defaults to ~/.bash_history / ~/.zsh_history).
               Cleaned on exit; also suppressed in child shells via env vars.
  fish         Non-interactive fish sessions never save history.  No action.

Online providers (--online / --anthropic) are *always* explicit opt-in flags.
You cannot accidentally send data to an external API.  The privacy-mode warning
fires only when both flags are active simultaneously (e.g. --mode privacy set
in config.toml but --online passed on the command line), to surface the
contradiction clearly.

What privacy mode CANNOT suppress:
  - OS-level process-creation logs (Windows Event Log, Linux auditd/syslog).
    Requires elevated privileges to suppress and is outside our scope.
  - DNS queries and network-layer logs from ``fetch_url``.
  - Files the *agent* is explicitly asked to write (write_file, patch_file).
    Those are intentional outputs, not traces.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from .display import console


# ---------------------------------------------------------------------------
#  1. Python readline history suppression (startup)
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
      LESSHISTFILE   — less pager history.
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
        "HISTFILE":       null,
        "HISTSIZE":       "0",
        "HISTFILESIZE":   "0",
        "HISTIGNORE":     "*",
        "HISTCONTROL":    "ignorespace:ignoredups",
        "LESSHISTFILE":   null,
        "MYSQL_HISTFILE": null,
        "SQLITE_HISTORY": null,
    })

    return env


# ---------------------------------------------------------------------------
#  3. Shell history file scrubbing (exit)
# ---------------------------------------------------------------------------

def _psreadline_history_paths() -> list[Path]:
    """
    Return candidate PSReadLine history file paths (Windows only).

    PSReadLine saves to the same location for both Windows PowerShell 5 and
    PowerShell 7 on Windows:
      %APPDATA%\\Microsoft\\Windows\\PowerShell\\PSReadLine\\ConsoleHost_history.txt

    cmd.exe has no persistent history at all — nothing to clean.
    """
    if sys.platform != "win32":
        return []
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return []
    return [
        Path(appdata) / "Microsoft" / "Windows" / "PowerShell"
        / "PSReadLine" / "ConsoleHost_history.txt"
    ]


def _unix_history_paths() -> list[Path]:
    """
    Return candidate shell history file paths (Unix/Mac only).

    Priority:
      1. $HISTFILE (set by the parent shell — covers bash, zsh, and others).
      2. Well-known defaults as fallbacks for shells that don't export HISTFILE.
    """
    if sys.platform == "win32":
        return []

    seen: set[Path] = set()
    paths: list[Path] = []

    def _add(p: Path) -> None:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            paths.append(p)

    # Parent shell's explicit HISTFILE (bash, zsh both export this by default)
    env_histfile = os.environ.get("HISTFILE", "")
    if env_histfile and env_histfile not in (os.devnull, "/dev/null"):
        _add(Path(env_histfile))

    # Common defaults in case HISTFILE isn't exported
    home = Path.home()
    for name in (".bash_history", ".zsh_history", ".history", ".sh_history"):
        _add(home / name)

    return paths


def _scrub_history_file(path: Path, pattern: re.Pattern) -> bool:
    """
    Remove lines matching *pattern* from *path*.

    Handles both plain bash/PSReadLine format and zsh extended_history format
    (lines like ``: 1234567890:0;command``).

    Returns True if any lines were removed.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return False

    lines = raw.splitlines()
    clean = []
    changed = False
    for line in lines:
        # zsh extended_history: ": timestamp:elapsed;command"
        # strip the prefix to check the actual command part
        cmd_part = re.sub(r"^:\s*\d+:\d+;", "", line)
        if pattern.search(cmd_part):
            changed = True
        else:
            clean.append(line)

    if not changed:
        return False

    try:
        # Preserve original line ending style
        nl = "\r\n" if "\r\n" in raw else "\n"
        path.write_text(nl.join(clean) + (nl if clean else ""), encoding="utf-8")
    except (OSError, PermissionError):
        pass   # best-effort; don't crash on permission errors

    return changed


def clear_shell_history_traces(binary_name: str = "localcoder") -> int:
    """
    Scrub lines referencing *binary_name* from shell history files on exit.

    Cleans:
      Windows  — PSReadLine ConsoleHost_history.txt
      Unix     — $HISTFILE, ~/.bash_history, ~/.zsh_history, ~/.history

    cmd.exe has no persistent history — nothing to clean there.

    Returns the number of files that were modified.
    """
    # Match the binary name as a word at the start of the command or after
    # common prefixes like 'sudo ', 'env VAR=x ', etc.
    pattern = re.compile(
        r"(^|[|&;`]\s*|sudo\s+)" + re.escape(binary_name) + r"\b",
        re.IGNORECASE,
    )

    candidates: list[Path] = _psreadline_history_paths() + _unix_history_paths()
    modified = 0
    for path in candidates:
        if path.exists() and path.is_file():
            if _scrub_history_file(path, pattern):
                modified += 1
    return modified


# ---------------------------------------------------------------------------
#  4. External-provider privacy warning
# ---------------------------------------------------------------------------

_PROVIDER_NAMES = {
    "openai":    "OpenAI",
    "anthropic": "Anthropic",
}


def warn_external_provider(provider: str) -> None:
    """
    Print a prominent warning when privacy mode is active but prompts will be
    sent to an external API.

    Note: using an external provider is always an explicit opt-in (--online /
    --anthropic flags).  This warning fires only when privacy mode is also
    active, to surface the contradiction clearly (e.g. mode = "privacy" in
    config.toml but --online on the CLI).

    Privacy mode suppresses *local* persistence (no log files, no readline
    history, shell history cleaned on exit), but it cannot prevent the API
    provider from receiving, logging, or training on your prompts.
    """
    name = _PROVIDER_NAMES.get(provider, provider)
    console.print(
        f"\n[bold yellow]⚠  Privacy mode + {name} API[/bold yellow]\n"
        f"[yellow]   Your prompts will be sent to {name}'s servers.\n"
        f"   {name} may log or use them per their privacy policy.\n"
        f"   Use a local model (--model) for full privacy.[/yellow]\n"
    )
