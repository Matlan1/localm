# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owner-only credential store for optional third-party model-source tokens
(Hugging Face, CivitAI). See dev-notes/ADR-0015-civitai-model-source.md,
Decision 4.

Values never live in config.json: PATCH /v1/config and `localm config` both
intercept these keys before they reach validate_update/update_config and route
them here instead, to a small file of their own, written with the same
owner-restricted primitive auth.key/auth.json/sessions.json already use.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

# Serializes the read-modify-write in set_credentials against other THREADS in
# this process; a cross-process lock (see set_credentials) covers other
# localm processes (the CLI alongside the running server) the same way
# auth.py's _KEYSTORE_LOCK does for the keystore.
_CRED_LOCK = threading.Lock()

# key -> the env var huggingface_hub / CivitAI tooling already recognizes, so a
# value a user has set for other tools is picked up here for free.
_ENV_FALLBACK = {
    "hf_token": "HF_TOKEN",
    "civitai_api_key": "CIVITAI_API_KEY",
}

CREDENTIAL_KEYS = frozenset(_ENV_FALLBACK)

_MAX_CREDENTIAL_LEN = 4096


def credentials_path() -> Path:
    from localm.config import home_dir
    return home_dir() / "model_source_credentials.json"


def _read_all_checked() -> tuple:
    """``(records, read_ok)``. ``read_ok`` is False ONLY when the file is
    PRESENT and could not be read even after ``config._read_json_checked``'s
    own retry - mirrors ``auth._load_keystore_checked`` for this store. A
    read-modify-write (``set_credentials``) must refuse on ``read_ok`` False
    rather than persist ``{}`` over an existing credential."""
    from localm.config import _read_json_checked
    data, read_ok = _read_json_checked(credentials_path(), {})
    return (data if isinstance(data, dict) else {}), read_ok


def _read_all() -> dict:
    return _read_all_checked()[0]


def _write_all(records: dict) -> bool:
    from localm.config import atomic_write_private
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # retrying=True: set_credentials serializes every writer of this file
    # cross-process before calling here. See
    # test_credentials_two_process_set_no_lost_writes.
    return atomic_write_private(path, json.dumps(records, indent=2), retrying=True)


def get_credential(key: str) -> Optional[str]:
    """The stored value for *key*, else its env var fallback, else None."""
    stored = _read_all().get(key)
    if isinstance(stored, str) and stored:
        return stored
    env_name = _ENV_FALLBACK.get(key)
    if env_name:
        env_val = os.environ.get(env_name)
        if env_val:
            return env_val
    return None


def get_hf_token() -> Optional[str]:
    return get_credential("hf_token")


def get_civitai_api_key() -> Optional[str]:
    return get_credential("civitai_api_key")


def set_credentials(updates: dict) -> None:
    """Apply *updates* (a subset of CREDENTIAL_KEYS -> str | None) as one
    read-modify-write. A blank or None value clears that key. Raises
    ValueError on an unknown key or a non-string value before anything is
    written, and ``config.ConfigUnreadable`` if the store file exists but
    could not be read - refusing rather than silently persisting ``{}`` (and
    so dropping every other stored credential) over it, mirroring
    ``auth.create_key``'s identical refusal for the scoped-key store."""
    checked: dict = {}
    for key, value in updates.items():
        if key not in CREDENTIAL_KEYS:
            raise ValueError(f"unknown credential key: {key!r}")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key}: expected a string, got {value!r}")
        s = (value or "").strip()
        if len(s) > _MAX_CREDENTIAL_LEN:
            raise ValueError(
                f"{key}: too long ({len(s)} characters, max {_MAX_CREDENTIAL_LEN})")
        checked[key] = s
    from localm.config import ConfigUnreadable, _cross_process_lock, ensure_dirs
    ensure_dirs()
    path = credentials_path()
    with _CRED_LOCK, _cross_process_lock(path):
        records, read_ok = _read_all_checked()
        if not read_ok:
            raise ConfigUnreadable(
                f"{path.name} exists but could not be read, so saving would "
                f"replace every stored credential with just this change; "
                f"refused. Fix or remove {path.name} (deleting it clears "
                f"stored credentials).")
        for key, s in checked.items():
            if s:
                records[key] = s
            else:
                records.pop(key, None)
        _write_all(records)
