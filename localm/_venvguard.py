# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared 'can this interpreter actually run localm' guard for the entry points.

``python -m localm`` and the ``localm`` / ``localcoder`` console scripts
(pyproject.toml [project.scripts]) call this before ``main()``. It turns an
interpreter that does NOT have localm's runtime dependencies - which otherwise
dies later with a cryptic ``ModuleNotFoundError`` for a runtime-only import -
into a clear, actionable message.

The check gates on whether the dependencies are actually importable, NOT on
whether ``sys.prefix`` looks like a virtualenv. A ``sys.prefix ==
sys.base_prefix`` check conflates "not in a .venv" with "deps missing" and
falsely blocks every legitimate install that is not a ``.venv`` directory: a
``pipx`` install, a container image, a system-wide or cold install, and CI's
``pip install -e .``. Those all HAVE the deps and must run; only a genuinely
dep-less interpreter sees the message.
"""

from __future__ import annotations

import sys

# The import names of the core dependencies (pyproject.toml [project]
# dependencies) that ``import localm.cli`` loads. If they all resolve, every
# entry point can start, whatever the interpreter's directory is named; a
# feature whose own dependency is missing reports it when used.
_CORE_DEPS = ("click", "rich", "requests", "certifi", "pydantic", "fastapi",
              "starlette")


def _missing_runtime_deps() -> list:
    """The names in ``_CORE_DEPS`` that cannot be located here. Uses find_spec so
    the deps are LOCATED, not imported (fast, no import side effects). A finder
    that raises counts as missing (fail toward the friendly message)."""
    import importlib.util
    missing = []
    for mod in _CORE_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


def _runtime_deps_present() -> bool:
    """True when every name in ``_CORE_DEPS`` can be located here."""
    return not _missing_runtime_deps()


def require_venv() -> None:
    """Exit with a friendly message naming what is missing ONLY when localm is
    run from an interpreter that lacks its startup dependencies, inside a
    virtualenv (another project's venv) or not. ANY environment where they are
    installed (localm's venv, pipx / container / system-wide / cold install / CI
    editable install) runs normally."""
    missing = _missing_runtime_deps()
    if not missing:
        return
    sys.exit(
        "Error: localm must be run from its virtual environment. This Python "
        f"({sys.executable}) is missing: {', '.join(missing)}. Use "
        "'.venv\\Scripts\\localm' (Windows) or '.venv/bin/localm' "
        "(Linux/macOS), or activate the venv first, or install localm and its "
        "dependencies into this environment.")
