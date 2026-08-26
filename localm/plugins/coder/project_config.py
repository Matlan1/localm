# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Per-project configuration loader for localcoder.

Reads ``.localcoder/config.toml`` from the working directory (or any parent).
Values are merged under CLI options - CLI flags always win.

Supported keys
--------------
model           = "gemma4-4b"
max_turns       = 20
auto_approve    = false
always_confirm  = ["run_shell", "run_shell_background"]  # prompt even under auto_approve
memory_file     = ".localcoder/memory.md" # overrides default search order
max_tokens      = 2048
temperature     = 0.7
mode            = "privacy"               # privacy | log | full
seed            = 1234                    # RNG seed for reproducible sampling
verify          = "pytest -x && ruff check"  # exit-code check for interactive
                                          # sessions; overrides auto-detection

The file is optional; absent keys fall through to CLI defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_CONFIG_NAME = ".localcoder/config.toml"


def find_project_config(cwd: Path) -> Path | None:
    """Walk up from cwd looking for .localcoder/config.toml."""
    current = cwd.resolve()
    for _ in range(10):   # cap the walk
        candidate = current / _CONFIG_NAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


class ProjectConfigUnreadable(Exception):
    """``.localcoder/config.toml`` EXISTS but could not be read or parsed.

    Carries the offending path as ``.path`` so a local surface can point at it."""

    def __init__(self, message: str, path: Path | None = None):
        super().__init__(message)
        self.path = path


def load_project_config(cwd: Path) -> dict[str, Any]:
    """
    Load .localcoder/config.toml nearest to cwd.

    Returns an empty dict when NO file is found. Raises ProjectConfigUnreadable
    when a file EXISTS but cannot be read or parsed.

    That distinction is load-bearing. Returning ``{}`` for an unparseable file is
    byte-identical to "there is no project config here", and two of the keys it
    would silently drop are SAFETY settings rather than preferences:
    ``always_confirm`` (the user's "prompt me before a shell command even under
    --yes") empties at cli/_main.py, so shell tool calls run unprompted, and
    ``mode = "privacy"`` is dropped there and in audit.effective_mode, so a
    session the user marked private falls through to the global coder_mode and a
    transcript is written to disk.

    The message names the full path because every consumer is a LOCAL surface
    (the coder CLI and the debug log, which pathscrub redacts for bug reports);
    effective_mode never propagates it, it fails safe to PRIVACY instead.

    Unknown keys are still ignored silently: that is forward compatibility, not
    a failure.
    """
    path = find_project_config(cwd)
    if path is None:
        return {}                          # genuinely absent: no project config

    try:
        import tomllib                     # Python 3.11+
    except ImportError:                    # pragma: no cover - requires-python >=3.12
        try:
            import tomli as tomllib        # backport for 3.10 and below
        except ImportError as e:
            raise ProjectConfigUnreadable(
                f"{path}: no TOML parser is available to read it", path) from e

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        raise ProjectConfigUnreadable(f"{path}: {e}", path) from e

    _KNOWN = {"model", "max_turns", "auto_approve", "always_confirm",
              "memory_file", "max_tokens", "temperature", "mode",
              "seed", "verify"}
    return {k: v for k, v in raw.items() if k in _KNOWN}
