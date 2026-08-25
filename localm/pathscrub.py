# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redact machine-identifying absolute paths out of text that crosses a trust boundary (an HTTP response body, a shareable bug report)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# A home-rooted path whose account segment must go even when it is NOT exactly
# Path.home() - a different account, or any path under the well-known user
# roots that an exact-prefix replacement would miss.
#
# Kept as a PATTERN STRING, not a pre-compiled object: the flags depend on
# sys.platform, and compiling at import time would freeze them, so a test that
# patches sys.platform to exercise the Windows branch would silently get the
# case-SENSITIVE regex. The original implementation this replaces read
# sys.platform per call, and this stays byte-identical to it. re caches compiled
# patterns, so there is no per-call compile cost.
_USER_ROOT_PATTERN = r"([A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)[^\\/\r\n]+"


def _sub_prefix(text: str, prefix: str, replacement: str) -> str:
    """Replace *prefix* in BOTH separator forms (and case-insensitively on Windows)."""
    if not prefix:
        return text
    for variant in {prefix, prefix.replace("\\", "/")}:
        if sys.platform == "win32":
            text = re.sub(re.escape(variant), replacement, text,
                          flags=re.IGNORECASE)
        else:
            text = text.replace(variant, replacement)
    return text


def scrub_user_paths(text: str) -> str:
    """Drop the home directory (which contains the username) from any paths, so text a stranger may read does not leak the account name; the file/line structure that matters for debugging is kept."""
    if not text:
        return text
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    if home:
        text = _sub_prefix(text, home, "~")
    flags = re.IGNORECASE if sys.platform == "win32" else 0
    return re.sub(_USER_ROOT_PATTERN, r"\1<redacted>", text, flags=flags)


def _machine_prefixes() -> List[Tuple[str, str]]:
    """(prefix, replacement) for every directory that names THIS machine's layout, longest first."""
    found: List[Tuple[str, str]] = []

    def add(value, label: str) -> None:
        """Register BOTH the raw value and its resolved form."""
        try:
            if not value:
                return
            raw = str(value)
            found.append((raw, label))
        except Exception:
            return
        try:
            resolved = str(Path(value).resolve())
            if resolved != raw:
                found.append((resolved, label))
        except Exception:
            pass

    try:
        from localm.config import home_dir
        add(home_dir(), "<data>")
    except Exception:
        pass
    try:
        import localm
        add(Path(localm.__file__).resolve().parent.parent, "<install>")
    except Exception:
        pass
    # The venv / interpreter root: third-party frames live under it, and on a
    # per-user install it carries the account name too.
    add(getattr(sys, "prefix", ""), "<env>")
    add(getattr(sys, "base_prefix", ""), "<env>")

    uniq = {(p, label) for p, label in found if p and p not in ("/", "\\")}
    return sorted(uniq, key=lambda item: len(item[0]), reverse=True)


def path_scrubber() -> Callable[[str], str]:
    """A ``scrub_paths`` bound to the prefixes resolved ONCE."""
    prefixes = _machine_prefixes()

    def scrub(text: str) -> str:
        if not text:
            return text
        for prefix, label in prefixes:
            text = _sub_prefix(text, prefix, label)
        return scrub_user_paths(text)

    return scrub


def scrub_paths(text: str) -> str:
    """``scrub_user_paths`` plus the localm data dir, install root and env prefix."""
    if not text:
        return text
    return path_scrubber()(text)
