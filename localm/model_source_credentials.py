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
from pathlib import Path
from typing import Optional

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


def _read_all() -> dict:
    from localm.config import _read_json
    records = _read_json(credentials_path(), {})
    return records if isinstance(records, dict) else {}


def _write_all(records: dict) -> bool:
    from localm.config import atomic_write_private
    return atomic_write_private(credentials_path(), json.dumps(records, indent=2))


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
    written."""
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
    records = _read_all()
    for key, s in checked.items():
        if s:
            records[key] = s
        else:
            records.pop(key, None)
    _write_all(records)
