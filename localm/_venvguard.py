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

# The import names of localm's core runtime dependencies (pyproject.toml
# [project] dependencies, not optional extras), less pywin32, whose absence
# localm reports instead of failing on. If they all resolve, this really is a
# localm environment that can run, whatever the interpreter's directory is named.
_CORE_DEPS = ("click", "rich", "requests", "certifi", "uv", "regex",
              "huggingface_hub", "pydantic", "fastapi", "starlette", "uvicorn",
              "cryptography", "qrcode", "zeroconf", "PIL", "pypdf")


def _runtime_deps_present() -> bool:
    """True when localm's core runtime deps are importable here. Uses find_spec so
    the deps are LOCATED, not imported (fast, no import side effects). A finder
    that raises is treated as 'not present' (fail toward the friendly message)."""
    import importlib.util
    for mod in _CORE_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def require_venv() -> None:
    """Exit with a friendly message ONLY when localm is run from an interpreter
    that lacks its runtime dependencies, inside a virtualenv (another project's
    venv) or not. ANY environment where the deps are installed (localm's venv,
    pipx / container / system-wide / cold install / CI editable install) runs
    normally."""
    if _runtime_deps_present():
        return
    sys.exit(
        "Error: localm must be run from its virtual environment. Use "
        "'.venv\\Scripts\\localm' (Windows) or '.venv/bin/localm' "
        "(Linux/macOS), or activate the venv first, or install localm and its "
        "dependencies into this environment.")
