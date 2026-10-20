# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which project tool hooks the user has trusted.

A project's ``.localcoder/config.toml`` can declare commands that run around
every tool call (hooks.py). A cloned repository can carry such a file, so its
hooks run only once the user has trusted them. Trust is recorded per config
file as a digest of the hooks it declares: editing, adding or removing a hook
changes the digest, and the changed hooks need trusting again.

The store is ``<data dir>/coder-hook-trust.json``, a JSON object mapping a
config file's normalised absolute path to ``{"sha256": <digest>,
"trusted_at": <unix time>}``. It is written only when the user asks for trust
to be remembered.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from localm.debuglog import logger

_FILENAME = "coder-hook-trust.json"

# Held, outside the store's cross-process lock, across every read-modify-write
# of the store. LOCK ORDER: _LOCK (outer) -> the store's .lock file.
_LOCK = threading.Lock()


def _store() -> Path:
    from localm.config import home_dir
    return home_dir() / _FILENAME


def config_key(config_path: Path) -> str:
    """The store key for *config_path*: its absolute path, case-folded where the
    filesystem is case-insensitive."""
    return os.path.normcase(str(Path(config_path).resolve()))


def hooks_digest(hooks_table: Any) -> str:
    """SHA-256 of the ``[hooks]`` table exactly as parsed, keys sorted, so any
    change to a command, matcher, timeout or the list of hooks changes it."""
    canonical = json.dumps(hooks_table, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrustStoreUnreadable(Exception):
    """The trust store exists but could not be read or is not a JSON object."""


def _load_strict() -> dict:
    """The store's contents. Returns {} when there is no store; raises
    TrustStoreUnreadable when one exists but is unreadable or malformed."""
    try:
        raw = json.loads(_store().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise TrustStoreUnreadable(f"{_store()}: {e}") from e
    if not isinstance(raw, dict):
        raise TrustStoreUnreadable(f"{_store()}: not a JSON object")
    return raw


def is_trusted(config_path: Path, digest: str) -> bool:
    """Whether the hooks with *digest* in *config_path* were trusted.

    An unreadable store trusts nothing: the failure is logged at WARNING and
    False is returned. Never raises."""
    try:
        entries = _load_strict()
    except TrustStoreUnreadable as e:
        logger.warning("coder hook trust: %s; no project hooks are trusted until "
                       "it is fixed or removed", e)
        return False
    entry = entries.get(config_key(config_path))
    return isinstance(entry, dict) and entry.get("sha256") == digest


@contextmanager
def _locked():
    from localm.config import _cross_process_lock
    store = _store()
    store.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _cross_process_lock(store):
        yield


def _write(entries: dict) -> None:
    """Replace the store with *entries* through a temporary file in the same
    folder. Raises OSError when it could not be written; the old file is then
    unchanged."""
    from localm.config import _replace_atomic
    store = _store()
    fd, tmp = tempfile.mkstemp(prefix=store.name + ".", suffix=".tmp",
                               dir=str(store.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(entries, indent=2, sort_keys=True))
        _replace_atomic(Path(tmp), store)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remember_trust(config_path: Path, digest: str) -> None:
    """Record that the hooks with *digest* in *config_path* are trusted,
    replacing any earlier record for that file.

    Raises OSError when the store cannot be written, and TrustStoreUnreadable
    when an existing store cannot be read (it is left untouched rather than
    overwritten)."""
    with _locked():
        entries = _load_strict()
        entries[config_key(config_path)] = {
            "sha256": digest, "trusted_at": int(time.time())}
        _write(entries)


def forget_trust(config_path: Path) -> bool:
    """Remove the record for *config_path*. Returns True when one was removed.

    Raises OSError when the store cannot be written, and TrustStoreUnreadable
    when an existing store cannot be read."""
    with _locked():
        entries = _load_strict()
        if entries.pop(config_key(config_path), None) is None:
            return False
        _write(entries)
        return True
