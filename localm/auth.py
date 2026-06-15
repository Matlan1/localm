"""
Single shared API-key authentication for localm's HTTP surface.

localm has no accounts or login - "auth" is one shared secret (the API key) that
a client presents as ``Authorization: Bearer <key>`` to reach protected
endpoints. This module is the single source of truth for that key.

Key resolution precedence (first hit wins):

    1. LOCALM_API_KEY environment variable (non-empty)  - transient override,
       e.g. injected by the launcher for one run
    2. <data dir>/auth.key file                         - persistent; written by
       the launcher's Auth card or ``localm key`` and read by every entry point
    3. none                                             - open mode (no key)

When no key resolves the server runs open (local/dev) UNLESS auth is *required*
(``LOCALM_REQUIRE_AUTH`` env var, or config ``"require_auth": true``), in which
case protected endpoints refuse every request until a key is configured. This is
the fail-closed switch; without it, loopback installs stay conveniently keyless.

The empty string is treated as "no key" everywhere (so ``LOCALM_API_KEY=""`` and
an empty file both mean open mode, not a usable empty key).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

ENV_VAR = "LOCALM_API_KEY"
REQUIRE_ENV_VAR = "LOCALM_REQUIRE_AUTH"
_TRUTHY = ("1", "true", "yes", "on")


def key_file() -> Path:
    """Path to the persisted API key, inside the resolved localm data dir.

    Uses the lazy ``home_dir()`` so it honours LOCALM_HOME / portable installs
    at call time (and lets tests redirect it)."""
    from localm.config import home_dir
    return home_dir() / "auth.key"


def generate_key(nbytes: int = 32) -> str:
    """Return a fresh, URL-safe random key. Not persisted - the caller decides
    whether to store it (see ``set_api_key`` / ``regenerate_key``)."""
    return secrets.token_urlsafe(nbytes)


def _read_key_file() -> Optional[str]:
    try:
        text = key_file().read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return text or None


def get_api_key() -> Optional[str]:
    """The active key, or None for open mode. Env var (non-empty) wins over the
    persisted file. Empty / whitespace-only values count as no key."""
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return env.strip()
    return _read_key_file()


def set_api_key(key: Optional[str]) -> None:
    """Persist *key* to ``auth.key`` (atomic write, owner-only perms). An empty
    or None *key* clears it, returning the server to open mode."""
    if not key or not key.strip():
        clear_api_key()
        return
    from localm.config import ensure_dirs
    ensure_dirs()
    path = key_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(key.strip() + "\n", encoding="utf-8")
    os.replace(tmp, path)          # atomic on Windows + POSIX (same dir)
    _restrict_perms(path)


def clear_api_key() -> None:
    """Remove the persisted key (open mode). A leftover env var still applies."""
    try:
        key_file().unlink(missing_ok=True)
    except OSError:
        pass


def regenerate_key(nbytes: int = 32) -> str:
    """Generate a new random key, persist it, and return it (for display)."""
    key = generate_key(nbytes)
    set_api_key(key)
    return key


def require_auth_enabled() -> bool:
    """True when the server must refuse requests if no key is configured.

    Enabled via the ``LOCALM_REQUIRE_AUTH`` env var or config
    ``"require_auth": true``. Default false keeps loopback installs keyless."""
    if os.environ.get(REQUIRE_ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("require_auth", False))
    except Exception:
        return False


def _restrict_perms(path: Path) -> None:
    """Best-effort: restrict the key file to the current user. No-op on failure
    or unsupported platforms - the data dir is already user-scoped."""
    try:
        if os.name == "posix":
            os.chmod(path, 0o600)
        else:
            import subprocess
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if user:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r",
                     "/grant:r", f"{user}:F"],
                    capture_output=True, check=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Scoped keystore (auth.json) - named keys with explicit scopes              #
# --------------------------------------------------------------------------- #
# The owner key above (env LOCALM_API_KEY / auth.key) is implicitly ADMIN.
# auth.json holds ADDITIONAL named keys, each limited to a set of scopes, so a
# client (a chat session, a read-only dashboard, a third-party tool) can be
# issued a key that does only what it needs. Only a hash of each key is stored;
# the plaintext is shown once at creation and is never recoverable.


def keystore_file() -> Path:
    """Path to the scoped-key store (inside the localm data dir)."""
    from localm.config import home_dir
    return home_dir() / "auth.json"


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load_keystore() -> list:
    try:
        data = json.loads(keystore_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_keystore(records: list) -> None:
    from localm.config import ensure_dirs
    ensure_dirs()
    path = keystore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    _restrict_perms(path)


def list_keys() -> list:
    """Public metadata for every named key (never the hash)."""
    return [
        {"id": r.get("id"), "name": r.get("name", ""),
         "scopes": r.get("scopes", []), "created": r.get("created")}
        for r in _load_keystore()
    ]


def create_key(name: str, scope_list) -> dict:
    """Mint a named key with *scope_list*, persist its hash, and return a record
    INCLUDING the plaintext key once - the caller must surface it now, it cannot
    be recovered. Raises ValueError on an unknown scope."""
    from localm import scopes as S
    clean = S.normalize(scope_list)
    bad = [s for s in clean if not S.is_valid_scope(s)]
    if bad:
        raise ValueError(f"Unknown scope(s): {', '.join(bad)}")
    key = generate_key()
    record = {
        "id": secrets.token_hex(6),
        "name": (name or "").strip() or "key",
        "hash": _hash_key(key),
        "scopes": clean,
        "created": time.time(),
    }
    records = _load_keystore()
    records.append(record)
    _save_keystore(records)
    return {"id": record["id"], "name": record["name"],
            "scopes": record["scopes"], "key": key}


def revoke_key(key_id: str) -> bool:
    """Delete a named key by id. Returns True if it existed."""
    records = _load_keystore()
    remaining = [r for r in records if r.get("id") != key_id]
    if len(remaining) == len(records):
        return False
    _save_keystore(remaining)
    return True


def any_key_configured() -> bool:
    """True when auth is in effect: an owner key (env/auth.key) OR any named key.
    When this is False the server runs open (unless require_auth_enabled())."""
    return get_api_key() is not None or bool(_load_keystore())


def verify(presented: Optional[str]) -> Optional[set]:
    """Resolve a presented bearer token to the set of scopes it grants, or None
    if it matches nothing. The owner key grants ADMIN (every scope)."""
    if not presented or not presented.strip():
        return None
    import hmac
    from localm import scopes as S
    presented = presented.strip()
    owner = get_api_key()
    if owner and hmac.compare_digest(presented, owner):
        return {S.ADMIN}
    presented_hash = _hash_key(presented)
    for r in _load_keystore():
        h = r.get("hash", "")
        if h and hmac.compare_digest(h, presented_hash):
            return set(r.get("scopes", []))
    return None
