# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared 'must run inside the project venv' guard for the console entry points.

``python -m localm`` already refuses to run outside a virtualenv, but the
``localm`` / ``localcoder`` console scripts (pyproject.toml [project.scripts])
called their ``main()`` directly with no such check. A stray GLOBAL install (a
separate ``pip install localm``) then runs with ``sys.prefix == sys.base_prefix``
and fails later with a cryptic ModuleNotFoundError for a runtime-only dependency
instead of a clear "run me from the venv" message (NEW-J / NEW-J-CODER)."""

from __future__ import annotations

import sys


def require_venv() -> None:
    """Exit with a friendly message when not running inside a virtualenv."""
    if sys.prefix == sys.base_prefix:
        sys.exit(
            "Error: localm must be run from its virtual environment. Use "
            "'.venv\\Scripts\\localm' (Windows) or '.venv/bin/localm' "
            "(Linux/macOS), or activate the venv first.")
