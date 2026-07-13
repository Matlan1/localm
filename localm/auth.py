# SPDX-License-Identifier: AGPL-3.0-or-later
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
import threading
import time
from pathlib import Path
from typing import Optional

from localm.debuglog import logger

ENV_VAR = "LOCALM_API_KEY"
REQUIRE_ENV_VAR = "LOCALM_REQUIRE_AUTH"
_TRUTHY = ("1", "true", "yes", "on")

# Minimum length for an owner key to count as "strong enough" to authenticate a
# NETWORK bind. Enforced at set time (set_api_key) AND at the network-bind gate
# (cli._exposed_bind_warning), so a trivially-guessable key supplied via the
# LOCALM_API_KEY env var or a hand-edited auth.key cannot be served to the LAN.
MIN_KEY_LEN = 8


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


# Transient read retry for the owner key file, mirroring config._read_json: a
# concurrent atomic replace (set_api_key), an antivirus / indexer, or a backup
# scanner can hold auth.key open for a microsecond on Windows, making read_text
# raise a TRANSIENT PermissionError (WinError 5). Ride it out with a short bounded
# retry rather than momentarily reading the owner key as absent, which would flap
# the owner's own auth. A PERSISTENT failure still returns None here; the
# fail-closed guard in any_key_configured() (via _owner_key_present) then keeps
# auth IN EFFECT so the server locks instead of dropping to open mode.
_KEY_READ_RETRIES = 8
_KEY_READ_BACKOFF = 0.01       # seconds; escalates linearly to the cap
_KEY_READ_BACKOFF_CAP = 0.05


def _read_key_file() -> Optional[str]:
    """The persisted owner key, or None when the file is absent or persistently
    unreadable. A transient Windows sharing violation is ridden out with a bounded
    retry (see the constants above); a persistent unreadable file returns None but
    is separately treated as auth-in-effect by any_key_configured() (fail closed).
    A read failure is SURFACED (rule 5), never silently equated with "no key"."""
    path = key_file()
    for attempt in range(_KEY_READ_RETRIES):
        try:
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        except FileNotFoundError:
            return None                        # genuinely absent -> open by design
        except PermissionError as e:
            # Transient class on Windows (a concurrent replace / AV / indexer).
            # Retry briefly; a persistent failure falls through to the warning.
            if attempt < _KEY_READ_RETRIES - 1:
                time.sleep(min(_KEY_READ_BACKOFF * (attempt + 1),
                               _KEY_READ_BACKOFF_CAP))
                continue
            logger.warning("owner key file %s exists but is unreadable (%s); "
                           "treating auth as IN EFFECT (fail closed) until it can "
                           "be read - fix its permissions", path, e)
            return None
        except (OSError, ValueError) as e:
            # Not the transient sharing-violation class (a real IO/permission
            # error, a directory, or non-UTF-8 content): do not spin the retry
            # budget. Still surfaced, and still fails closed via _owner_key_present.
            logger.warning("owner key file %s exists but is unreadable (%s); "
                           "treating auth as IN EFFECT (fail closed) until it can "
                           "be read - fix its permissions", path, e)
            return None
    return None


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
    key = key.strip()
    if len(key) < MIN_KEY_LEN:
        raise ValueError(
            f"API key must be at least {MIN_KEY_LEN} characters long.")
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
    except OSError as e:
        # Surface, do not silence: this is a security step (removing the persisted
        # key). If the file cannot be deleted the key STILL GRANTS access, so a
        # silent pass would imply a clear that did not happen (rule 5). Warn loudly.
        logger.warning("could not remove the API key file %s (%s); the key may "
                       "still be active until it is deleted by hand", key_file(), e)
    try:
        keystore_file().unlink(missing_ok=True)
    except OSError as e:
        # Same: a leftover keystore means scoped keys remain valid. Do not let a
        # failed delete look like a successful clear.
        logger.warning("could not remove the keystore %s (%s); scoped keys may "
                       "still be active until it is deleted by hand",
                       keystore_file(), e)


def regenerate_key(nbytes: int = 32) -> str:
    """Generate a new random key, persist it, and return it (for display)."""
    key = generate_key(nbytes)
    set_api_key(key)
    return key


def require_auth_enabled() -> bool:
    """True when the server must refuse requests if no key is configured.

    Enabled via the ``LOCALM_REQUIRE_AUTH`` env var or config
    ``"require_auth": true``. Default false keeps loopback installs keyless.

    On a config-read failure this resolves to True (LM-DA-021), matching the
    newer fail-closed precedent this codebase established for the identical
    "does a security kill-switch fail toward more or less access when config
    is unreadable" question: netpolicy.network_mode() (HON-2, dbac9e1c) and
    this module's own _owner_key_present()/any_key_configured() (f9a2ad48)
    both resolve toward MORE restriction on a read failure, never less - "the
    exact fail-open a safety toggle must never do". This function used to
    return False here (reviewed and accepted by the 2026-07-02 security audit
    at the time) but was left unrevisited when the stricter precedent landed
    nine days later."""
    if os.environ.get(REQUIRE_ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("require_auth", False))
    except Exception:
        # Fail CLOSED, not open (see docstring, LM-DA-021): an admin who
        # explicitly set require_auth: true must never silently drop to open
        # mode because of a read glitch. load_config() itself only raises here
        # for something severe (e.g. ensure_dirs() hitting an inaccessible home
        # dir) - ordinary corrupt/locked config.json is already absorbed by its
        # own .bak/retry fallback in config._read_json and never reaches this
        # except - so an install that never touched require_auth hitting this
        # rare path gets a temporary lockout instead of a silently-open server,
        # the safe direction for a security kill-switch. Admins who need a
        # config-independent fail-closed switch regardless of this reasoning
        # can still set LOCALM_REQUIRE_AUTH (checked above, before this try).
        logger.warning(
            "config unreadable; cannot confirm require_auth - treating as "
            "required (fail closed) until it can be read")
        return True


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
        # Best-effort (see docstring): the data dir is already user-scoped, so this
        # per-file tightening is defense-in-depth. A failure (icacls missing, an FS
        # without perms) leaves the home-dir scoping in effect, which is the real
        # protection; a perms nicety must never raise.
        pass


# --------------------------------------------------------------------------- #
#  Scoped keystore (auth.json) - named keys with explicit scopes              #
# --------------------------------------------------------------------------- #
# The owner key above (env LOCALM_API_KEY / auth.key) is implicitly ADMIN.
# auth.json holds ADDITIONAL named keys, each limited to a set of scopes, so a
# client (a chat session, a read-only dashboard, a third-party tool) can be
# issued a key that does only what it needs. Only a hash of each key is stored;
# the plaintext is shown once at creation and is never recoverable.


# Serializes the read-modify-write of the keystore. create_key/revoke_key both
# load the full record list, mutate it, and write it back; without this lock two
# concurrent calls can read the same list and the last writer clobbers the
# other's change (a lost write / dropped key). Module-level so it is shared by
# every caller in the process.
_KEYSTORE_LOCK = threading.Lock()


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


# Filesystem-access level a credential may reach on the SERVER HOST. A single
# graded dial (nested: host > shared > none), NOT a scope, so it can be set below
# host even on an owner's own device (the device just carries a key with a lower
# level). "none" = no host FS at all (device upload only); "host" = the whole
# server filesystem.
#
# "shared" is RESERVED scaffolding for a future "confined to owner-designated
# shared roots" tier. It is a recognised level (so a stored value normalises and
# fails safe) but is NOT enforced anywhere yet: require_fs_host() grants only
# "host", so a "shared" key currently reaches no more host FS than "none". It is
# deliberately kept OUT of the user-facing `localm key create --fs-access` choices
# until that confinement is built, so we never advertise a tier that does nothing
# (AGENTS rule 5: do not hide problems). Implement the confinement, then re-expose
# it in the CLI choice.
FS_ACCESS_LEVELS = ("none", "shared", "host")


def norm_fs_access(level: Optional[str]) -> str:
    """Coerce *level* to a known FS-access level; unknown/blank -> the safe
    default 'none' (least privilege)."""
    lv = (level or "none").strip().lower()
    return lv if lv in FS_ACCESS_LEVELS else "none"


def fs_access_for(token: str, default: str = "none") -> str:
    """The stored host-filesystem-access level for the key behind *token*, or
    *default* if the key is unknown or has no level recorded (a legacy key minted
    before this attribute existed defaults to the safe 'none')."""
    if not token or not token.strip():
        return default
    h = _hash_key(token.strip())
    for r in _load_keystore():
        if r.get("hash") == h:
            return norm_fs_access(r.get("fs_access", default))
    return default


def list_keys() -> list:
    """Public metadata for every named key (never the hash)."""
    return [
        {"id": r.get("id"), "name": r.get("name", ""),
         "scopes": r.get("scopes", []), "created": r.get("created"),
         "expires": r.get("expires"), "last_used": r.get("last_used"),
         "fs_access": norm_fs_access(r.get("fs_access", "none"))}
        for r in _load_keystore()
    ]


def create_key(name: str, scope_list, *, allow_privileged: bool = False,
               expires: Optional[float] = None, fs_access: str = "none") -> dict:
    """Mint a named key with *scope_list*, persist its hash, and return a record
    INCLUDING the plaintext key once - the caller must surface it now, it cannot
    be recovered. Raises ValueError on an unknown scope.

    *fs_access* is the host-filesystem reach this key grants ("none" | "shared" |
    "host"); it defaults to the safe "none" so a new scoped key cannot browse the
    server disk unless the owner deliberately grants it. The owner/ADMIN key
    always resolves to "host" regardless of this field (see effective_fs_access).

    PRIVILEGED_SCOPES (admin / keys:admin / plugins:admin / config:write /
    coder:full) are refused with PermissionError unless *allow_privileged* is
    True. Callers must only set that for an owner/ADMIN principal, so a merely
    keys:admin-scoped key cannot mint itself owner-equivalent access (privilege
    self-escalation).

    *expires* is an optional epoch-seconds deadline after which verify() rejects
    the key; None (default) never expires."""
    from localm import scopes as S
    clean = S.normalize(scope_list)
    bad = [s for s in clean if not S.is_valid_scope(s)]
    if bad:
        raise ValueError(f"Unknown scope(s): {', '.join(bad)}")
    if not allow_privileged:
        privileged = [s for s in clean if s in S.PRIVILEGED_SCOPES]
        if privileged:
            raise PermissionError(
                "Refusing to grant privileged scope(s): "
                f"{', '.join(privileged)}. Only the owner key may mint keys "
                "with these capabilities."
            )
    key = generate_key()
    record = {
        "id": secrets.token_hex(6),
        "name": (name or "").strip() or "key",
        "hash": _hash_key(key),
        "scopes": clean,
        "created": time.time(),
        "expires": float(expires) if expires is not None else None,
        "fs_access": norm_fs_access(fs_access),
    }
    with _KEYSTORE_LOCK:
        records = _load_keystore()
        records.append(record)
        _save_keystore(records)
    return {"id": record["id"], "name": record["name"],
            "scopes": record["scopes"], "expires": record["expires"],
            "fs_access": record["fs_access"], "key": key}


def revoke_key(key_id: str) -> bool:
    """Delete a named key by id. Returns True if it existed.

    Also drops any browser SESSIONS minted from that key, so revoking a key cuts
    off a paired device immediately instead of leaving its cookie session valid
    until expiry. (The cookie auth path also re-validates a scoped session's key on
    every request via key_hash_live, so this is belt-and-suspenders cleanup that
    keeps the session store tidy rather than the sole enforcement.)"""
    with _KEYSTORE_LOCK:
        records = _load_keystore()
        target = next((r for r in records if r.get("id") == key_id), None)
        remaining = [r for r in records if r.get("id") != key_id]
        if len(remaining) == len(records):
            return False
        _save_keystore(remaining)
    if target and target.get("hash"):
        try:
            from localm import sessions
            sessions.revoke_by_key_hash(target["hash"])
        except Exception as e:
            logger.debug("session cleanup after key revoke failed (non-fatal): %s", e)
    return True


def key_hash_live(key_hash: Optional[str]) -> bool:
    """True when a keystore key with this sha256 *key_hash* still exists AND has not
    expired. Used to tie a scoped-key browser session to its key's lifecycle: the
    cookie path checks this every request so revoking or expiring the underlying key
    also invalidates the session, mirroring verify()'s per-request check for a
    bearer. (Owner/ADMIN sessions are deliberately NOT gated on this - the owner key
    is not in the keystore, and an owner session is decoupled from the key VALUE so a
    key roll does not log the owner out.)"""
    if not key_hash:
        return False
    now = time.time()
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            return not (exp is not None and now > float(exp))
    return False


def _keystore_configured() -> bool:
    """True when the scoped keystore should count as 'auth in effect'.

    A present-but-UNPARSEABLE or unreadable auth.json counts as configured, so a
    transient corruption fails CLOSED (every request still needs a key, and the
    damaged store verifies none, so access is locked) instead of silently
    dropping to open mode and exposing a scoped-keys-only install. A genuinely
    absent or empty (``[]``) keystore is NOT configured (a fresh or cleared
    install runs open by design, matching _load_keystore()'s empty result).
    """
    path = keystore_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False                       # absent -> no scoped keys
    except OSError:
        return True                        # exists but unreadable -> fail closed
    try:
        data = json.loads(raw)
    except ValueError:
        return True                        # exists but corrupt -> fail closed
    return bool(data) if isinstance(data, list) else True


def _owner_key_present() -> bool:
    """True when an owner key is in effect: the LOCALM_API_KEY env var is set, OR
    the auth.key file EXISTS - readable or NOT.

    A present-but-unreadable file counts as present so auth stays IN EFFECT (fail
    CLOSED) instead of silently dropping to open/keyless mode when a read glitch
    (a transient AV/indexer lock, or a persistent permissions/profile change)
    makes auth.key unreadable to the running process. This mirrors
    _keystore_configured's own missing-vs-unreadable branching; only a genuinely
    ABSENT file (and no env key) is "no owner key" -> open by design.

    Distinct from get_api_key(), which still returns the key VALUE (or None when it
    cannot be read): when the file is present but unreadable this returns True
    (auth in effect) while verify() matches nothing, so every request is rejected
    (401 / locked) rather than served open - the safe direction."""
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return True
    try:
        return key_file().exists()
    except OSError:
        # Cannot even stat the path (e.g. a parent-directory permission problem):
        # fail CLOSED rather than assume the owner key is absent.
        return True


def any_key_configured() -> bool:
    """True when auth is in effect: an owner key (env, or an auth.key file that is
    present even if unreadable - see _owner_key_present) OR a configured scoped
    keystore. When this is False the server runs open (unless
    require_auth_enabled()). A corrupt/unreadable keystore, and now a
    present-but-unreadable owner key, both count as configured so they fail CLOSED
    rather than silently open: a keyed install must not lose its auth to a damaged
    or unreadable credential file (checkup 2026-07-11 HIGH)."""
    return _owner_key_present() or _keystore_configured()


# Throttle for last-used stamping: verify() runs on EVERY request, so it must not
# rewrite the keystore each time. Stamp a key's last_used at most once per this many
# seconds (per process), tracked in memory so the hot path stays lock-free until a
# write is actually due.
_LAST_USED_THROTTLE_S = 300
_last_used_writes: dict = {}
_LAST_USED_LOCK = threading.Lock()


def _touch_last_used(key_hash: str) -> None:
    """Best-effort: stamp a just-verified key's last_used (throttled). Never raises -
    a usage timestamp is non-essential and must not break verification."""
    if not key_hash:
        return
    now = time.monotonic()
    with _LAST_USED_LOCK:
        # Use dict-ABSENCE (not a 0.0 sentinel) for "never stamped this process":
        # time.monotonic()'s epoch is platform-defined and is seconds-since-boot
        # on Linux, so on a freshly-booted machine `now` can be < the throttle
        # window and `now - 0.0` would wrongly throttle the very FIRST stamp,
        # leaving last_used None until 5 min of uptime had passed.
        prev = _last_used_writes.get(key_hash)
        if prev is not None and now - prev < _LAST_USED_THROTTLE_S:
            return
        _last_used_writes[key_hash] = now
    try:
        with _KEYSTORE_LOCK:
            records = _load_keystore()
            for r in records:
                # Plain == is fine here (not constant-time): the key is ALREADY
                # verified; this only locates its row to stamp, not authenticating.
                if r.get("hash") == key_hash:
                    r["last_used"] = time.time()
                    _save_keystore(records)
                    break
    except Exception as e:
        # Best-effort: a usage stamp must never break auth. Surface the reason at
        # debug level (RULE 5: do not mute silently) rather than a bare pass.
        logger.debug("last_used stamp failed (non-fatal): %s", e)


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
            exp = r.get("expires")
            if exp is not None and time.time() > float(exp):
                return None       # matched a real key, but it has expired
            _touch_last_used(presented_hash)
            return set(r.get("scopes", []))
    return None
