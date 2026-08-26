# SPDX-License-Identifier: AGPL-3.0-or-later
"""Path-aware glob scope matching: compile a gitignore/pathspec-style glob into a
full-relative-path regex, cached per scope. Used by the Agent scope enforcement."""

from __future__ import annotations

import re

# Compiled scope patterns, keyed by the scope string.
_SCOPE_RE_CACHE: dict[str, "re.Pattern[str]"] = {}

def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """
    Compile a path-aware glob into a regex anchored to a full relative path.

    Semantics (gitignore / pathspec style), unlike plain ``fnmatch``:
      - ``*``  matches any run of characters WITHIN one path segment - it does
        NOT cross ``/``. So ``src/*.py`` matches ``src/a.py`` but not
        ``src/a/b.py``.
      - ``**`` matches across segments. ``**/`` matches any number of leading
        directories (including none); a trailing ``**`` matches the rest.
      - ``?``  matches a single non-``/`` character.
      - all other characters are matched literally.
    """
    i, n = 0, len(pattern)
    out = ["(?s:"]
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 2] == "**":
                j = i
                while j < n and pattern[j] == "*":
                    j += 1
                if pattern[j:j + 1] == "/":
                    # '**/' -> zero or more leading directory segments
                    out.append("(?:[^/]+/)*")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "/":
            out.append("/")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append(r")\Z")
    return re.compile("".join(out))

def _scope_pattern(scope: str) -> "re.Pattern[str]":
    rx = _SCOPE_RE_CACHE.get(scope)
    if rx is None:
        rx = _glob_to_regex(scope)
        _SCOPE_RE_CACHE[scope] = rx
    return rx
