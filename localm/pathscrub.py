# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redact machine-identifying absolute paths out of text that crosses a trust
boundary (an HTTP response body, a shareable bug report).

Two levels:

``scrub_user_paths``
    Drops the user's HOME directory and, as an always-on backstop, the account
    name in any ``C:\\Users\\<name>`` / ``/home/<name>`` / ``/Users/<name>``
    path. ``bugreport._scrub_home`` is a thin alias for it, so there is exactly
    ONE implementation of the username rule.

``scrub_paths``
    Everything above PLUS the localm data dir (``LOCALM_HOME``), the install
    root, and the Python environment prefix. Use it for anything handed to a
    caller who is not the local operator; on a portable install the data dir
    can sit nowhere near ``Path.home()``, so the username rule alone would not
    catch it.

Both KEEP THE STRUCTURE that makes the text useful - the file name, the line
number, the reason - and replace only the leading directories: these paths are
redacted, never muted.

Every lookup is guarded INDIVIDUALLY and the username backstop runs
unconditionally, so a failure to resolve one prefix can never cause the raw
text to be emitted as though it had been scrubbed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# A home-rooted path whose account segment must go even when it is NOT exactly
# Path.home() - a different account, or any path under the well-known user
# roots that an exact-prefix replacement would miss.
#
# The Windows separator must stay matched as 1-OR-2 backslashes: text that has
# passed through repr() or json.dumps() carries every backslash doubled, and a
# single-backslash-only pattern does not match that form at all. Forward slashes
# are untouched by both encodings, so they stay single.
#
# Kept as a PATTERN STRING, not a pre-compiled object: the flags depend on
# sys.platform, so compiling at import time would freeze them. re caches
# compiled patterns, so there is no per-call compile cost.
_USER_ROOT_PATTERN = r"([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)|/home/|/Users/)[^\\/\r\n]+"


def _sub_prefix(text: str, prefix: str, replacement: str) -> str:
    """Replace *prefix* in every separator form it can appear in, and
    case-insensitively on Windows.

    Three forms are covered: the value as given, the forward-slash form, and
    the doubled-backslash form that text carries after passing through repr()
    or json.dumps(). On a prefix with no backslash the doubled variant is
    identical to *prefix* and the set below dedupes it away.
    """
    if not prefix:
        return text
    for variant in {prefix, prefix.replace("\\", "/"), prefix.replace("\\", "\\\\")}:
        if sys.platform == "win32":
            text = re.sub(re.escape(variant), replacement, text,
                          flags=re.IGNORECASE)
        else:
            text = text.replace(variant, replacement)
    return text


def scrub_user_paths(text: str) -> str:
    """Drop the home directory (which contains the username) from any paths, so
    text a stranger may read does not leak the account name; the file/line
    structure that matters for debugging is kept.

    Only the home lookup is guarded: when ``Path.home()`` raises or is empty,
    the username is still stripped by the fallback regex over the common home
    roots. That backstop ALWAYS runs, even when the home lookup succeeded.
    """
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
    """(prefix, replacement) for every directory that names THIS machine's
    layout, longest first.

    The order is load-bearing: the data dir often sits INSIDE the install root
    (a portable install) or inside the venv prefix, and replacing the parent
    first would rewrite the head of the child's path and leave the remainder
    dangling.
    """
    found: List[Tuple[str, str]] = []

    def add(value, label: str) -> None:
        """Register BOTH the raw value and its resolved form.

        The two differ in the shipped configuration: localm provisions its
        interpreter with uv, whose directory is a version-less alias that
        ``resolve()`` follows to the versioned real path, and frame text
        carries the alias. The same applies to a data dir reached through a
        junction or symlink, or to macOS's /tmp -> /private/tmp.

        Guarded per prefix: one unresolvable location does not cost the others,
        and a resolve() failure still leaves the raw form registered.
        """
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
    """A ``scrub_paths`` bound to the prefixes resolved ONCE.

    ``scrub_paths`` re-resolves the prefix list on every call, and each
    ``Path.resolve()`` is a filesystem call on Windows. Callers with many
    strings should bind once with this instead.
    """
    prefixes = _machine_prefixes()

    def scrub(text: str) -> str:
        if not text:
            return text
        for prefix, label in prefixes:
            text = _sub_prefix(text, prefix, label)
        return scrub_user_paths(text)

    return scrub


def scrub_paths(text: str) -> str:
    """``scrub_user_paths`` plus the localm data dir, install root and env
    prefix. Use for anything handed to a caller who is not the local operator.

    Returns *text* unchanged when it is empty or ``None``.
    """
    if not text:
        return text
    return path_scrubber()(text)
