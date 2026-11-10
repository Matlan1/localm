# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Small constants and paths shared with the plugin engine
(``localm/plugins/engine.py``), which is the one plugin manifest parser,
loader and install/enable/disable mechanism. ``plugins_dir()`` and
``_RESERVED_NAMES`` are read by ``engine.py`` and ``media_config.py``.
"""

from __future__ import annotations

from pathlib import Path


def plugins_dir() -> Path:
    """Root directory scanned for external plugins."""
    from localm.config import home_dir
    return home_dir() / "plugins"


_RESERVED_NAMES = {
    "pull", "run", "serve", "list", "remove", "info", "config",
    "doctor", "coder", "plugin", "imagine", "mcp", "alias", "completion",
    "gui", "benchmark",
}
