# SPDX-License-Identifier: AGPL-3.0-or-later
"""Privacy-mode helpers for localcoder."""

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
    """Reason to refuse moving a session into *dest_cwd*, or None to allow it."""
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
    """Return a copy of the current environment with shell history vars zeroed."""
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
    """Return candidate PSReadLine history file paths (Windows only)."""
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
    """Return candidate shell history file paths (Unix/Mac only)."""
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
    """Remove lines matching *pattern* from *path*."""
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
    # read_text() applies universal-newline translation, turning \r\n into \n
    # before we can detect it, which made the "preserve CRLF" logic below a
    # no-op everywhere except Windows (where text-mode WRITE re-added \r\n by
    # accident). Detect from the raw bytes and write byte-exact instead.
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
    """Scrub lines referencing the coder invocation from shell history on exit."""
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
    """Print a prominent warning when privacy mode is active but prompts will be sent to an external API."""
    name = _PROVIDER_NAMES.get(provider, provider)
    console.print(
        f"\n[bold yellow]⚠  Privacy mode + {name} API[/bold yellow]\n"
        f"[yellow]   Your prompts will be sent to {name}'s servers.\n"
        f"   {name} may log or use them per their privacy policy.\n"
        f"   Use a local model (--model) for full privacy.[/yellow]\n"
    )
