# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Privacy-mode helpers for localcoder.

When ``--mode privacy`` is active this module is responsible for suppressing
every data-persistence point that the process can control:

  1. Python readline history (``~/.python_history``) - suppressed at startup.
  2. Subprocess shell history (bash/sh ``HISTFILE``) - suppressed per call.
  3. Shell history files cleaned at exit - PSReadLine, bash, zsh history files
     are scrubbed of lines referencing the binary name.
  4. External-provider warning - printed when prompts would leave the machine.

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
# The implementation lives in the kernel (localm.readline_privacy) so the plain
# ``localm`` chat REPL can suppress history without importing this coder plugin.
# Re-exported here for the coder's own use and for back-compat (callers and
# tests import suppress_readline_history from localm.plugins.coder.privacy).
from localm.readline_privacy import suppress_readline_history  # noqa: F401,E402


# ---------------------------------------------------------------------------
#  2. Subprocess environment
# ---------------------------------------------------------------------------

# Persistence modes, least to most recording. Used to compare a live session's
# mode against a directory's own declared one.
_MODE_RANK = {"privacy": 0, "log": 1, "full": 2}


def refuse_move_into_stricter_project(session_mode: str, dest_cwd) -> str | None:
    """Reason to refuse moving a session into *dest_cwd*, or None to allow it.

    A session's persistence mode is resolved ONCE, from the directory it started
    in, and cannot be changed afterwards - the audit log is already open, which
    is what the REPL's /mode reports. Changing the DIRECTORY is therefore the one
    way a session's mode and its location can come to disagree, and the
    disagreement is not harmless: the Markdown transcript is written to the
    session's CURRENT cwd at close, so a ``full`` session moved into a project
    marked private would leave a complete record in <dest>/.localcoder/sessions/
    and the episodic store would take the work too.

    The mode cannot be lowered to match, so such a move is REFUSED and the reason
    named: nothing is written into a project that declared itself private, and
    nothing pretends the setting was honoured.

    KEYED ON THE PROJECT'S OWN DECLARATION, NOT ON effective_mode(). The global
    default coder mode is ``privacy``, so effective_mode() answers "privacy" for
    every ordinary directory that has said nothing at all, and keying on it would
    refuse moving a recording session anywhere. Only
    ``.localcoder/config.toml`` is a project SAYING something, so only that is
    treated as a claim to respect; a project that has said nothing is not
    asserting privacy, and the session's own mode still governs.

    An UNREADABLE project config refuses too, for the reason
    audit.effective_mode gives at the same fork: the file is the one place a user
    can mark a project private, and this cannot tell whether this one did.

    Shared by the web cwd route and the REPL's /cd, so the two surfaces cannot
    drift.
    """
    from pathlib import Path as _Path

    from localm.plugins.coder.project_config import (
        ProjectConfigUnreadable,
        load_project_config,
    )
    try:
        declared = load_project_config(_Path(dest_cwd)).get("mode")
    except ProjectConfigUnreadable:
        return (
            "That project's .localcoder/config.toml could not be read, so there "
            "is no way to tell whether it is marked private. This session "
            f"records at '{session_mode}', and its persistence cannot be lowered "
            "once it has started, so it is not moved there. Fix the file, or "
            "start a new session in that directory."
        )
    except Exception:                                          # noqa: BLE001
        # Any other failure to read the project's own config is not a claim of
        # privacy - fall through and let the session's own mode govern, matching
        # effective_mode's bare `except Exception: pass` at the same fork.
        return None

    if not isinstance(declared, str):
        return None
    declared = declared.strip().lower()
    if declared not in _MODE_RANK:
        return None
    if _MODE_RANK[declared] >= _MODE_RANK.get(session_mode, 0):
        return None
    return (
        f"This session records at '{session_mode}', but that project sets "
        f"'{declared}' in its .localcoder/config.toml. A session's persistence "
        "cannot be lowered once it has started, so moving it there would write a "
        "record the project asked not to keep. Start a new session in that "
        "directory instead."
    )


def subprocess_privacy_env() -> dict[str, str]:
    """Return a copy of the current environment with shell history vars zeroed.

    Used by ``tool_run_shell`` in privacy mode so that any bash/sh/zsh child
    process cannot write command history to disk. For non-interactive shells
    (the normal case) these are no-ops, and they also guard the edge case where a
    script opens an interactive sub-shell.

    Variables overridden:
      HISTFILE       - path where bash/zsh writes history on exit.
      HISTSIZE       - in-memory history depth (0 = disabled in bash).
      HISTFILESIZE   - max lines written to HISTFILE (0 = truncate to empty).
      HISTIGNORE     - ``*`` ignores every command in bash history.
      HISTCONTROL    - ``ignorespace:ignoredups``.
      LESSHISTFILE   - less pager history.
      MYSQL_HISTFILE - mysql CLI history.
      SQLITE_HISTORY - sqlite3 CLI history.

    No env vars are set for fish or PowerShell:
      * fish: non-interactive fish sessions never save history regardless.
      * PowerShell: PSReadLine only runs in interactive sessions, and these
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

    cmd.exe has no persistent history at all - nothing to clean.
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
      1. $HISTFILE (set by the parent shell - covers bash, zsh, and others).
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
        data = path.read_bytes()
    except (OSError, PermissionError) as exc:
        # Surface, don't silence: an unreadable history file means we cannot
        # know whether it holds secret-bearing command lines, so the privacy
        # guarantee did not hold for it. Mirrors the write-failure warning below.
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] could not read history "
            f"file [yellow]{path}[/yellow] ({exc}); it was not scrubbed."
        )
        return False

    # Read bytes (not read_text) so the original newline style is observable:
    # read_text() applies universal-newline translation, turning CRLF into LF
    # before it can be detected, which would defeat the "preserve CRLF" logic
    # below. Detect from the raw bytes and write byte-exact.
    raw = data.decode("utf-8", errors="replace")
    crlf = b"\r\n" in data
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
        # Preserve the original line ending style byte-exactly (write_bytes does
        # no OS newline translation, unlike write_text).
        nl = "\r\n" if crlf else "\n"
        out = (nl.join(clean) + (nl if clean else "")).encode("utf-8")
        path.write_bytes(out)
    except (OSError, PermissionError) as exc:
        # Surface, don't silence: the secret-bearing lines are STILL on disk.
        # Reporting changed=True here would tell the user a scrub happened that
        # did not, breaking the privacy guarantee, so return False instead and
        # warn so the unscrubbed file is discoverable.
        console.print(
            f"\n[bold yellow]Warning:[/bold yellow] could not scrub history "
            f"file [yellow]{path}[/yellow] ({exc}); command lines remain on disk."
        )
        return False

    return changed


def clear_shell_history_traces(binary_name: str = "localcoder") -> int:
    """Scrub lines referencing the coder invocation from shell history on exit.

    Matches BOTH spellings of the coder command:
      * the standalone ``binary_name`` console-script (default ``localcoder``);
      * the documented ``localm coder`` subcommand form (the real, primary
        invocation), allowing any inter-word whitespace.

    A bare ``localm`` line for some *other* subcommand (``localm gui``,
    ``localm serve``, the chat REPL, ...) is left untouched: only the coder
    pollutes its own history, and wiping every ``localm`` line would delete
    unrelated history.

    Cleans:
      Windows  - PSReadLine ConsoleHost_history.txt
      Unix     - $HISTFILE, ~/.bash_history, ~/.zsh_history, ~/.history

    cmd.exe has no persistent history - nothing to clean there.

    Returns the number of files that were modified.
    """
    # Match either the standalone binary name or the "localm coder" subcommand,
    # as a word at the start of the command or after common prefixes like
    # 'sudo ', a pipe, '&&', etc.  ``\b`` after each alternative keeps the match
    # word-bounded so substrings ("localcoderlib", "localm coderfoo") are kept.
    invocation = (
        r"(?:" + re.escape(binary_name) + r"\b"
        + r"|localm\s+coder\b"
        + r")"
    )
    pattern = re.compile(
        r"(^|[|&;`]\s*|sudo\s+)" + invocation,
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
    """Print a prominent warning when privacy mode is active but prompts will be
    sent to an external API.

    Using an external provider is always an explicit opt-in (--online /
    --anthropic flags). This warning fires only when privacy mode is also active,
    to surface the contradiction (e.g. mode = "privacy" in config.toml but
    --online on the CLI).

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
