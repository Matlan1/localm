# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin ids: the rule a plugin name must satisfy before it becomes a path
component under the installed-plugins or store root."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _is_valid_plugin_name(name: Any) -> bool:
    """True iff *name* is a legal plugin id: ONE path component, shaped like an
    identifier once hyphens are folded to underscores.

    The SAME rule ``parse_spec`` applies to a manifest's ``[plugin] name``, and
    it is enforced at the two places a name becomes a path (``_installed_dir``
    / ``_store_dir``).
    """
    if not name or not isinstance(name, str):
        return False
    # isidentifier() rejects every separator, dot, space and leading digit, so
    # '.', '..', '../x', 'a/b' and a drive-qualified path are all refused; the
    # Path(name).name comparison re-checks that the id is a single path
    # component. A SHAPE check, not a uniqueness one: 'MyTool' names the same
    # directory as 'mytool' on a case-insensitive filesystem.
    return name == Path(name).name and name.replace("-", "_").isidentifier()


def _check_plugin_name(name: str) -> str:
    """Return *name* if it is a legal plugin id, else raise ValueError."""
    if not _is_valid_plugin_name(name):
        raise ValueError(f"invalid plugin name: {name!r}")
    return name
