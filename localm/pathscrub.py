# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redact machine-identifying absolute paths out of text that crosses a trust
boundary (an HTTP response body, a shareable bug report).

Two levels, for two different readers:

``scrub_user_paths``
    Drops the user's HOME directory and, as an always-on backstop, the account
    name in any ``C:\\Users\\<name>`` / ``/home/<name>`` / ``/Users/<name>``
    path. This is the bug-report policy; ``bugreport._scrub_home`` is a thin
    alias for it, so there is exactly ONE implementation of the username rule.

``scrub_paths``
    Everything above PLUS the localm data dir (``LOCALM_HOME``), the install
    root, and the Python environment prefix. A bug report is read by the
    maintainer and wants the install layout; an API response handed to a
    lower-privileged or anonymous caller does not, and on a portable install
    the data dir is often nowhere near ``Path.home()`` so the username rule
    alone would not catch it.

Both KEEP THE STRUCTURE that makes the text useful - the file name, the line
number, the reason - and replace only the leading directories. A caller told
"load failed" with the cause removed has been handed a mystery; a caller told
"not an embedding model" with the path replaced by ``<data>`` has everything
they can act on and nothing they should not see.

Every lookup is guarded INDIVIDUALLY and the username backstop runs
unconditionally, so a failure to resolve one prefix can never cause the raw
text to be emitted as though it had been scrubbed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# Matches a home-rooted path's account segment under the well-known user roots.
# Kept as a pattern STRING, not a compiled object: the flags depend on
# sys.platform, which is read per call.
_USER_ROOT_PATTERN = r"([A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)[^\\/\r\n]+"


def _sub_prefix(text: str, prefix: str, replacement: str) -> str:
    """Replace *prefix* in BOTH separator forms (and case-insensitively on
    Windows). A value entered with forward slashes does not match the native
    backslash form, and on Windows a lowercased drive/segment is the same
    directory - a plain case-sensitive ``str.replace`` leaks on those variants.
    """
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
    """Drop the home directory (which contains the username) from any paths, so
    text a stranger may read does not leak the account name; the file/line
    structure that matters for debugging is kept.

    Guard ONLY the home lookup: if ``Path.home()`` raises or is empty we must
    NOT ship the raw text as if scrubbed (privacy fail), so fail safe and strip
    the username with the fallback regex over the common home roots. That
    backstop ALWAYS runs, even when the home lookup succeeded.
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

    Longest-first matters: the data dir often sits INSIDE the install root (a
    portable install) or inside the venv prefix. Replacing the parent first
    would rewrite the head of the child's path and leave the remainder dangling,
    so the more specific prefix has to win.
    """
    found: List[Tuple[str, str]] = []

    def add(value, label: str) -> None:
        """Register BOTH the raw value and its resolved form.

        Registering only the resolved form silently no-ops whenever the two
        differ, and they differ in the SHIPPED configuration: localm provisions
        its interpreter with uv, whose directory is a version-less alias
        (``cpython-3.12-windows-x86_64-none``) that ``resolve()`` follows to the
        versioned real path. Frame text carries the alias, so a resolved-only
        prefix matches nothing and every stdlib frame comes back with a full
        absolute path. The same applies to a data dir reached through a junction
        or symlink, or to macOS's /tmp -> /private/tmp, where the leak is the
        data dir and hence the account name.

        Guarded per prefix: one unresolvable location must not cost the others,
        and a resolve() failure must still leave the raw form registered.
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
    # The venv / interpreter root, which carries the account name on a per-user
    # install.
    add(getattr(sys, "prefix", ""), "<env>")
    add(getattr(sys, "base_prefix", ""), "<env>")

    uniq = {(p, label) for p, label in found if p and p not in ("/", "\\")}
    return sorted(uniq, key=lambda item: len(item[0]), reverse=True)


def path_scrubber() -> Callable[[str], str]:
    """A ``scrub_paths`` bound to the prefixes resolved ONCE.

    ``scrub_paths`` re-resolves the prefix list on every call, which is fine for
    a one-off string but wasteful for a hot loop - ``/debug/stacks`` scrubs
    hundreds of frame strings per request, and each ``Path.resolve()`` is a
    filesystem call on Windows. Callers with many strings should bind once.
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

    Returns *text* unchanged when it is empty or ``None``, so a caller can pass
    an optional value straight through without a "is there an error at all"
    check turning into a false positive on an empty string.
    """
    if not text:
        return text
    return path_scrubber()(text)
