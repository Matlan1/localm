# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-project configuration loader for localcoder."""

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
    """``.localcoder/config.toml`` EXISTS but could not be read or parsed."""

    def __init__(self, message: str, path: Path | None = None):
        super().__init__(message)
        self.path = path


def load_project_config(cwd: Path) -> dict[str, Any]:
    """Load .localcoder/config.toml nearest to cwd."""
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
