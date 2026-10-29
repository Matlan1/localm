# SPDX-License-Identifier: AGPL-3.0-or-later
"""Redact machine-identifying absolute paths out of text that crosses a trust
boundary (an HTTP response body, a shareable bug report).

Three levels:

``scrub_user_paths``
    Drops the user's HOME directory and, as an always-on backstop, the account
    name in any ``C:\\Users\\<name>`` / ``/home/<name>`` / ``/Users/<name>``
    path. ``bugreport._scrub_home`` is a thin alias for it, so there is exactly
    ONE implementation of the username rule.

``scrub_paths``
    Everything above PLUS the localm data dir (``LOCALM_HOME``), the install
    root and the Python environment prefix. Use it for an error text shown to
    every caller, the local operator included; on a portable install the data
    dir can sit nowhere near ``Path.home()``, so the username rule alone would
    not catch it.

``scoped_path_scrubber``
    Everything above PLUS the folder of every absolute path the model registry
    stores (a model registered in place), and then the leading directories of
    ANY remaining absolute host path, written as ``<path>``
    (``scrub_host_paths``: a drive-letter or UNC path, or a POSIX path whose
    first component is a name at this machine's filesystem root). Use it for
    anything handed to a caller who is not the local operator
    (``http_server.path_formatter``).

All KEEP THE STRUCTURE that makes the text useful - the file name, the line
number, the reason - and replace only the leading directories: these paths are
redacted, never muted.

Every lookup is guarded INDIVIDUALLY and the username backstop runs
unconditionally, so a failure to resolve one prefix can never cause the raw
text to be emitted as though it had been scrubbed.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable

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


# The placeholder scrub_paths writes in place of the localm data dir.
DATA_DIR_LABEL = "<data>"

# The placeholder scoped_path_scrubber writes in place of the leading
# directories of any other absolute host path.
PATH_LABEL = "<path>"

# One segment of a Windows path: no separator, line break, tab, or character
# Windows forbids in a name other than ``<`` and ``>`` (so a segment that is
# already a placeholder, such as ``<redacted>``, stays part of the path).
# Spaces are allowed.
_WIN_SEG = r"[^\\/\r\n\t\"'|*?:]++"

# The leading directories of a drive-letter path, the drive included:
# ``D:\a\b\``, its doubled form (repr() / JSON) and ``D:/a/b/``. Group 1 is the
# separator; one separator style per match.
_DRIVE_DIRS_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:(\\{1,2}|/)(?:" + _WIN_SEG + r"\1)*")

# The leading directories of a UNC path, machine and share included:
# ``\\host\share\dir\`` and its doubled form. It starts the text or follows
# whitespace, a quote or one of ``( [ { = , ; | ` < *``, or a colon that ends a
# word of two or more characters. Group 1 is the separator. The ``\\?\`` device
# form is not matched here.
_UNC_DIRS_RE = re.compile(
    r"(?:(?<![^\s\"'(\[{=,;|`<*])|(?<=[A-Za-z0-9_]{2}:))"
    r"(\\{1,2})\1(?!\?)[^\\/\s\"'|*?:<>]++(?:\1" + _WIN_SEG + r")*\1")

# The leading directories of a POSIX path, ``/first/more/``, with group 1 the
# first segment. Not matched after a word character, ``.``, ``~``, ``/``,
# ``\``, ``<``, ``>``, ``@`` or ``-`` (a URL, a relative path, ``~/x``,
# ``<data>/x``). Whitespace ends a segment.
_POSIX_SEG = r"[^/\s\"'<>|\\:*?]++"
_POSIX_DIRS_RE = re.compile(
    r"(?<![\w.~/\\<>@-])/(" + _POSIX_SEG + r")(?:/" + _POSIX_SEG + r")*/")


def _root_names() -> set[str] | None:
    """The names in this machine's filesystem root (``/``; on Windows the root
    of the current drive), lower-cased on Windows. None when the root cannot
    be listed."""
    try:
        names = os.listdir(os.sep)
    except OSError:
        return None
    if sys.platform == "win32":
        return {n.lower() for n in names}
    return set(names)


def scrub_host_paths(text: str, root_names: set[str] | None = None) -> str:
    """*text* with the leading directories of every absolute host path written
    as ``<path>`` plus the path's own separator, its last component kept: a
    drive-letter path, a UNC path, and a POSIX path whose first component is in
    *root_names* (the names at this machine's filesystem root, listed when not
    given; compared case-insensitively on Windows). When the root cannot be
    listed, every POSIX-shaped path counts as a host path. Idempotent."""
    if not text:
        return text
    text = _DRIVE_DIRS_RE.sub(lambda m: PATH_LABEL + m.group(1), text)
    text = _UNC_DIRS_RE.sub(lambda m: PATH_LABEL + m.group(1), text)
    if "/" not in text:
        return text
    if root_names is None:
        root_names = _root_names()
    fold = sys.platform == "win32"

    def posix(m: re.Match) -> str:
        first = m.group(1).lower() if fold else m.group(1)
        if root_names is not None and first not in root_names:
            return m.group(0)
        return PATH_LABEL + "/"

    return _POSIX_DIRS_RE.sub(posix, text)


def _norm(path: str) -> str:
    """*path* with forward slashes and no trailing separator, lower-cased on
    Windows."""
    path = path.replace("\\", "/").rstrip("/")
    return path.lower() if sys.platform == "win32" else path


def _is_under(path: str, folder: str) -> bool:
    """True when *path* is *folder* or inside it, compared as ``_norm`` forms."""
    a, b = _norm(path), _norm(folder)
    return bool(b) and (a == b or a.startswith(b + "/"))


_ABS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|\\\\|/")


def _registry_folders() -> list[str]:
    """The folder of every absolute path held in a top-level string field of a
    registry entry (the model file, a projector, a source path), read without
    creating any directory. A filesystem root or drive root is left out. Empty
    when the registry cannot be read, logged at DEBUG."""
    try:
        from localm.config import peek_registry
        registry = peek_registry()
    except Exception as e:
        try:
            from localm.debuglog import logger
            logger.debug("path scrub: registry folders not read: %s", e)
        except Exception:
            pass
        return []
    folders: set = set()
    for entry in registry.values():
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if not isinstance(value, str) or not _ABS_PATH_RE.match(value):
                continue
            folder = os.path.dirname(value.rstrip("\\/")).rstrip("\\/")
            if not folder or re.fullmatch(r"[A-Za-z]:|\\\\[^\\/]*", folder):
                continue
            folders.add(folder)
    return sorted(folders)


def _machine_prefixes(registry_folders: bool = False) -> list[tuple[str, str]]:
    """(prefix, replacement) for every directory that names THIS machine's
    layout, longest first. With *registry_folders*, also the folder of every
    absolute registry path outside those directories and home, as ``<path>``.

    The order is load-bearing: the data dir often sits INSIDE the install root
    (a portable install) or inside the venv prefix, and replacing the parent
    first would rewrite the head of the child's path and leave the remainder
    dangling.
    """
    found: list[tuple[str, str]] = []

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
        add(home_dir(), DATA_DIR_LABEL)
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

    # Registry folders inside a prefix above, or inside home, keep that label.
    if registry_folders:
        try:
            home = str(Path.home())
        except Exception:
            home = ""
        bases = [p for p, _label in found if p] + ([home] if home else [])
        for folder in _registry_folders():
            if not any(_is_under(folder, base) for base in bases):
                found.append((folder, PATH_LABEL))

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


def scoped_path_scrubber() -> Callable[[str], str]:
    """A scrubber for text handed to a caller who is not the local operator,
    bound to the prefixes resolved ONCE: ``path_scrubber``'s prefixes plus the
    registry model folders (``<path>``), ``scrub_user_paths``, then
    ``scrub_host_paths`` for every other absolute host path. Idempotent, and
    safe to apply to text ``scrub_paths`` already scrubbed."""
    prefixes = _machine_prefixes(registry_folders=True)
    roots: list = []

    def scrub(text: str) -> str:
        if not text:
            return text
        for prefix, label in prefixes:
            text = _sub_prefix(text, prefix, label)
        text = scrub_user_paths(text)
        if "/" in text and not roots:
            roots.append(_root_names())
        return scrub_host_paths(text, roots[0] if roots else None)

    return scrub


def map_strings(value, fn: Callable[[str], str]):
    """*value* with *fn* applied to every string inside it, through nested
    dicts, lists and tuples (tuples come back as lists); dict keys and other
    values are returned unchanged."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [map_strings(v, fn) for v in value]
    return value


def scrub_paths(text: str) -> str:
    """``scrub_user_paths`` plus the localm data dir, install root and env
    prefix. Other folders are kept: for a caller who is not the local operator,
    pass the result through ``http_server.path_formatter`` as well.

    Returns *text* unchanged when it is empty or ``None``.
    """
    if not text:
        return text
    return path_scrubber()(text)
