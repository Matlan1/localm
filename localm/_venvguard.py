# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared 'can this interpreter actually run localm' guard for the entry points."""

from __future__ import annotations

import sys

# localm's core runtime dependencies - always installed with the package (these
# are core requires, not optional extras). If they resolve, this really is a
# localm environment that can run, whatever the interpreter's directory is named.
_CORE_DEPS = ("click", "fastapi")


def _runtime_deps_present() -> bool:
    """True when localm's core runtime deps are importable here."""
    import importlib.util
    for mod in _CORE_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def require_venv() -> None:
    """Exit with a friendly message ONLY when localm is run from an interpreter that lacks its runtime dependencies."""
    if sys.prefix != sys.base_prefix:
        return                       # inside a virtualenv
    if _runtime_deps_present():
        return                       # installed outside a venv, deps present
    sys.exit(
        "Error: localm must be run from its virtual environment. Use "
        "'.venv\\Scripts\\localm' (Windows) or '.venv/bin/localm' "
        "(Linux/macOS), or activate the venv first, or install localm and its "
        "dependencies into this environment.")
