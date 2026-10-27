# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the sha256 digests another localm install recorded in its ``registry.json``.

A localm home is a folder holding ``registry.json`` beside ``models/``. Importing
it as a folder of loose GGUFs would hash every file again, although the install
that pulled or registered them already stored each file's digest.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_REGISTRY_MAX_BYTES = 64 * 1024 * 1024
_HEX_DIGEST_RE = re.compile(r"[0-9a-fA-F]{64}")


def _digest_key(path: Path) -> Optional[str]:
    """The lookup key for the local file *path*: absolute, symlinks resolved,
    case-folded where the filesystem is case-insensitive. None when it cannot be
    resolved."""
    try:
        return os.path.normcase(str(path.resolve()))
    except (OSError, ValueError, RuntimeError):
        return None


def _stored_key(stored: str) -> Optional[str]:
    """The lookup key for a path recorded in a foreign registry. The path is
    only normalised as text and never resolved, so a recorded network path cannot
    make this machine contact a server; None for a network, relative or
    NUL-carrying path."""
    if stored.startswith(("\\\\", "//")) or "\x00" in stored or not os.path.isabs(stored):
        return None
    return os.path.normcase(os.path.normpath(stored))


def _is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:
        return False


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def find_localm_home(p: Path) -> Optional[Path]:
    """The localm home *p* denotes: *p* itself when it holds ``registry.json`` and
    ``models/``, or the parent of *p* when *p* is the ``models`` folder of such a
    home; else None."""
    for candidate in (p, p.parent if p.name == "models" else None):
        if candidate is not None and _is_file(candidate / "registry.json") \
                and _is_dir(candidate / "models"):
            return candidate
    return None


@dataclass(frozen=True)
class RecordedDigests:
    """The digests another localm install recorded, and when it last wrote them.

    *digests* maps the normalised stored ``path`` of each usable entry to its
    lower-cased sha256; *written_at* is the modification time of the registry
    they came from."""
    digests: dict = field(default_factory=dict)
    written_at: float = 0.0

    def __len__(self) -> int:
        return len(self.digests)

    def lookup(self, path: Path) -> Optional[str]:
        """The sha256 recorded for the file *path*, or None. None also when the
        file is empty, cannot be statted, or was modified after the registry was
        last written (the record may describe other bytes)."""
        key = _digest_key(path)
        stored = self.digests.get(key) if key is not None else None
        if stored is None:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        if st.st_size <= 0 or st.st_mtime > self.written_at:
            return None
        return stored


def recorded_digests(home: Path, entry_path: Callable[..., Optional[str]]
                     ) -> "tuple[RecordedDigests, Optional[str]]":
    """``(recorded, problem)`` for the registry of the localm *home*.

    The registry is a hand-editable file, so an entry is used only when
    *entry_path* accepts its path and its ``sha256`` is 64 hex digits. A path
    recorded twice with different digests is left out. An entry that cannot be
    used is skipped; *problem* is a sentence naming why the whole registry could
    not be read (*recorded* is then empty), else None.
    """
    registry = home / "registry.json"
    try:
        st = registry.stat()
        if st.st_size > _REGISTRY_MAX_BYTES:
            return RecordedDigests(), "its registry.json is too large to read"
        with open(registry, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8-sig"))
    except (OSError, ValueError, RecursionError) as exc:
        return RecordedDigests(), f"its registry.json could not be read ({type(exc).__name__})"
    if not isinstance(data, dict):
        return RecordedDigests(), "its registry.json is not a model table"
    digests: dict[str, str] = {}
    conflicting: set[str] = set()
    for entry in data.values():
        stored = entry_path(entry)
        sha = entry.get("sha256") if isinstance(entry, dict) else None
        if stored is None or not isinstance(sha, str) or not _HEX_DIGEST_RE.fullmatch(sha):
            continue
        key = _stored_key(stored)
        if key is None:
            continue
        sha = sha.lower()
        if digests.setdefault(key, sha) != sha:
            conflicting.add(key)
    for key in conflicting:
        del digests[key]
    return RecordedDigests(digests, st.st_mtime), None
