"""
Per-project configuration loader for localcoder.

Reads ``.localcoder/config.toml`` from the working directory (or any parent).
Values are merged under CLI options — CLI flags always win.

Supported keys
--------------
model        = "gemma4-4b"
max_turns    = 20
auto_approve = false
memory_file  = ".localcoder/memory.md"   # overrides default search order
max_tokens   = 2048
temperature  = 0.7

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


def load_project_config(cwd: Path) -> dict[str, Any]:
    """
    Load .localcoder/config.toml nearest to cwd.

    Returns an empty dict if no file is found or if tomllib is unavailable.
    Keys are validated against the known set; unknown keys are silently ignored.
    """
    path = find_project_config(cwd)
    if path is None:
        return {}

    try:
        try:
            import tomllib                     # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib        # backport for 3.10 and below
            except ImportError:
                return {}                      # toml support unavailable

        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception:
        return {}

    _KNOWN = {"model", "max_turns", "auto_approve", "memory_file",
              "max_tokens", "temperature"}
    return {k: v for k, v in raw.items() if k in _KNOWN}
