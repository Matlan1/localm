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
import hmac
import json
import os
import re
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

# Characters an owner key may contain, enforced at set time (set_api_key). This is
# EXACTLY the alphabet generate_key() emits (secrets.token_urlsafe -> base64url), so
# a generated key always passes; note ~49% of generated keys contain an underscore,
# so "-" alone would reject half of them. It is also a strict subset of RFC 7235
# token68, so a conforming key is always safe to put in an Authorization header.
# Explicit ASCII classes, not \w or str.isalnum(): both match non-ASCII letters and
# digits ("ä", "٣"), which is the very thing this rejects. See set_api_key.
_KEY_CHARSET = re.compile(r"\A[A-Za-z0-9_-]+\Z")


def ct_equal(presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time compare of two secrets. Safe for ANY input, never raises.

    Use this for every secret comparison rather than calling hmac.compare_digest()
    on str. compare_digest() raises TypeError if EITHER operand is a non-ASCII str,
    so an ASCII expected value (a token_urlsafe or a hexdigest) does NOT protect the
    compare: a bearer/CSRF token reaches us as a latin-1 decoded HTTP header, so any
    caller can supply a non-ASCII operand and turn a wrong-credential 401/403 into an
    unhandled 500. Comparing bytes keeps the compare constant-time AND total.

    surrogatepass because os.environ carries lone surrogates on Windows, which a
    plain utf-8 encode raises on - that would swap one crash for another.
    """
    # A falsy operand means "no credential presented" / "no secret configured";
    # neither is secret-dependent, so short-circuiting leaks nothing.
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "surrogatepass"),
                               expected.encode("utf-8", "surrogatepass"))


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


# The three genuinely distinct states auth.key can be in. Keep them distinct:
# collapsing "unreadable" into "absent" silently opens a keyed server, and
# collapsing "holds nothing" into "holds a key" locks its owner out (REG-579).
_KEY_ABSENT = "absent"
_KEY_UNREADABLE = "unreadable"
_KEY_OK = "ok"


def _read_owner_key_file():
    """Read auth.key once and report WHICH state it is in: ``(status, text, err)``.

      (_KEY_ABSENT, None, None)     no file -> no owner key, open by design
      (_KEY_UNREADABLE, None, err)  it exists but cannot be read or decoded: we
                                    cannot TELL whether a key exists -> callers
                                    fail CLOSED
      (_KEY_OK, text, None)         we read it; *text* is what it holds (maybe "")

    THE single place that decides what auth.key contains. The value path
    (get_api_key) and the in-effect path (any_key_configured) each used to read
    and judge this file on their own, and REG-579 was precisely those two
    disagreeing: one read "empty means no key", the other read "the file exists,
    so a key exists", and the server locked its owner out of it. One reader, one
    answer, so they cannot drift apart again.

    A transient Windows sharing violation (a concurrent set_api_key replace, an
    antivirus, the indexer holding it for a microsecond) is ridden out with a
    bounded retry rather than flapping the owner's auth.

    ``utf-8-sig``, not ``utf-8``: a BOM is what a Windows editor or PowerShell's
    ``Out-File -Encoding utf8`` writes at the front of a hand-made file, and
    ``str.strip()`` does NOT remove U+FEFF. Read as plain utf-8, a BOM-only
    "empty" file would look like a key nobody can present (the REG-579 lockout
    again), and a BOM + a real key would stop the owner's correct key matching."""
    path = key_file()
    for attempt in range(_KEY_READ_RETRIES):
        try:
            return _KEY_OK, path.read_text(encoding="utf-8-sig"), None
        except FileNotFoundError:
            return _KEY_ABSENT, None, None     # genuinely absent -> open by design
        except PermissionError as e:
            # Transient class on Windows. Retry briefly; a persistent failure
            # falls through to unreadable (and the caller's warning).
            if attempt < _KEY_READ_RETRIES - 1:
                time.sleep(min(_KEY_READ_BACKOFF * (attempt + 1),
                               _KEY_READ_BACKOFF_CAP))
                continue
            return _KEY_UNREADABLE, None, e
        except (OSError, ValueError) as e:
            # Not the transient sharing-violation class (a real IO error, a
            # directory in its place, undecodable bytes): do not spin the budget.
            return _KEY_UNREADABLE, None, e
    return _KEY_UNREADABLE, None, None


def _key_text_or_none(text: str) -> Optional[str]:
    """The key *text* holds, or None when it holds none.

    NUL bytes are stripped alongside whitespace: a file truncated by a crash or a
    sync is padded with NULs, and a run of NULs is not a key anyone could ever
    present - so it means "no key", exactly like an empty file. Treating it as a
    key would put auth in effect with nothing to match: the REG-579 lockout."""
    return text.strip().strip("\x00").strip() or None


def _read_key_file() -> Optional[str]:
    """The persisted owner key, or None when the file is absent or persistently
    unreadable. A persistent unreadable file returns None but is separately
    treated as auth-in-effect by any_key_configured() (fail closed). A read
    failure is SURFACED (rule 5), never silently equated with "no key"."""
    status, text, err = _read_owner_key_file()
    if status == _KEY_UNREADABLE:
        logger.warning("owner key file %s exists but is unreadable (%s); "
                       "treating auth as IN EFFECT (fail closed) until it can "
                       "be read - fix its permissions", key_file(), err)
        return None
    if status == _KEY_ABSENT:
        return None
    return _key_text_or_none(text)


def get_api_key() -> Optional[str]:
    """The active key, or None for open mode. Env var (non-empty) wins over the
    persisted file. Empty / whitespace-only values count as no key."""
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return env.strip()
    return _read_key_file()


def set_api_key(key: Optional[str]) -> None:
    """Persist *key* to ``auth.key`` (atomic write, owner-only perms). An empty
    or None *key* clears it, returning the server to open mode.

    Raises ValueError if *key* is shorter than MIN_KEY_LEN or uses characters
    outside _KEY_CHARSET. Both are CONFIG-time guards on what a user may CHOOSE
    here; they are not a promise about what verify() will see. The LOCALM_API_KEY
    env var and a hand-edited auth.key both bypass this function entirely, so
    verify() stays liberal and never raises on whatever it is handed."""
    if not key or not key.strip():
        clear_api_key()
        return
    key = key.strip()
    if len(key) < MIN_KEY_LEN:
        raise ValueError(
            f"API key must be at least {MIN_KEY_LEN} characters long.")
    if not _KEY_CHARSET.match(key):
        # A key is carried in an HTTP Authorization header, which cannot hold these
        # characters reliably. A NON-ASCII key is the sharp case and is verified:
        # clients send UTF-8 but RFC 7230 obs-text decodes latin-1, so the server
        # sees mojibake and refuses the owner's OWN key from most clients. Spaces
        # and control characters break or inject into the header outright. Refuse at
        # set time rather than persist a key that mysteriously fails to log in.
        # This guard does NOT make verify() safe: LOCALM_API_KEY and a hand-edited
        # auth.key never come through here, and a hostile caller picks its own
        # presented token anyway. verify() is total on its own (see ct_equal);
        # this only stops you CHOOSING a key that will not work.
        raise ValueError(
            "API key must use only letters, numbers, '-' and '_' (the characters "
            "'localm key generate' produces). Spaces, punctuation, and non-English "
            "letters cannot be sent reliably in an HTTP Authorization header, so "
            "most clients would fail to authenticate with such a key.")
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
    or unsupported platforms - the data dir is already user-scoped.

    The implementation moved to ``config.restrict_file_perms`` so sessions.json
    and jobs.json (which hold the key DIGEST) get the identical treatment as
    auth.key (which holds the PLAINTEXT); they previously did not on Windows.
    Kept as a name here because it is referenced throughout this module."""
    from localm.config import restrict_file_perms
    restrict_file_perms(path)


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
    # KNOWN ACCEPTED WEAKNESS on the user-chosen-owner-key path, recorded rather
    # than justified. CodeQL py/weak-sensitive-data-hashing (alert 88) flags this
    # unsalted SHA-256, and on one of the two paths that reach it the rule is
    # RIGHT. Do not read this comment as a defence of the status quo.
    #
    # Two kinds of secret arrive here, and they are not equivalent:
    #   * NAMED KEYSTORE KEYS are always secrets.token_urlsafe(32) (see
    #     create_key) - 256 bits of CSPRNG output. No dictionary and no rainbow
    #     table touches 2^256, so a KDF's work factor buys nothing here.
    #   * THE OWNER KEY CAN BE USER-CHOSEN, and may be a human-memorable
    #     password. `localm key set KEY` persists a key the user provides
    #     (docs/cli.md), and set_api_key's own docstring above records that its
    #     MIN_KEY_LEN and charset checks are CONFIG-time guards only: the
    #     LOCALM_API_KEY env var and a hand-edited auth.key BYPASS THAT FUNCTION
    #     ENTIRELY. So a short, low-entropy owner key reaches this line
    #     unvalidated, and its digest is then persisted in sessions.json and
    #     jobs.json. Against that input a single unsalted SHA-256 is exactly what
    #     the rule warns about.
    #
    # Why it is not simply swapped for a KDF here: this runs on the PER-REQUEST
    # verify path (_principal_from_token), so a naive scrypt/argon2 at this line
    # is a latency cost on every authenticated request and a cheap DoS lever.
    # The fix therefore has to move the expensive check off the hot path - verify
    # once at session establishment (sessions.py already stores a key_hash), or
    # salt+KDF the owner key at SET time and keep generated tokens on the cheap
    # path behind a per-key marker. That is a design decision with a measurable
    # latency budget, not a one-line edit, and it is open rather than settled.
    #
    # SEPARATE, and fixed: the files holding this digest (sessions.json,
    # jobs.json) were not permission-restricted on Windows while auth.key holding
    # the PLAINTEXT was, so the digest was readable where the secret was not.
    # See config.restrict_file_perms and its callers. That was the ACL half of
    # alert 88; it is NOT the whole of it, and the alert should not be treated as
    # closed by it.
    #
    # surrogatepass for the same reason as ct_equal: a plain utf-8 encode raises
    # UnicodeEncodeError on a lone surrogate, which would just move the crash here
    # from the compare. Byte-identical to encode("utf-8") for every key that
    # encodes at all, so no stored hash changes.
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


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


# One-shot latch for the empty-auth.key notice (see _owner_key_present). This runs
# on EVERY request via any_key_configured() -> require_auth, so an unthrottled
# warning would put one line per request in the log for a persistent state.
_empty_owner_key_warned = False


def _owner_key_present() -> bool:
    """True when an owner key is in effect: the LOCALM_API_KEY env var is set, OR
    the auth.key file exists AND is not readably empty.

    A present-but-UNREADABLE file counts as present so auth stays IN EFFECT (fail
    CLOSED) instead of silently dropping to open/keyless mode when a read glitch
    (a transient AV/indexer lock, or a persistent permissions/profile change)
    makes auth.key unreadable to the running process. Only a genuinely ABSENT
    file, or one we can READ and see holds no key, is "no owner key" -> open by
    design.

    That empty-means-no-key split is what _keystore_configured already does for
    the identical question (absent or empty ``[]`` -> not configured; unreadable
    or corrupt -> fail closed), and what get_api_key()/set_api_key() already do
    for an empty value. Fail-closed is for "we cannot TELL whether a key exists";
    a readable empty file is not that case - we can tell, and the answer is no.
    Treating it as a key put auth in effect with nothing for verify() to match,
    401ing every request and locking the owner out of their own server with no
    way back (POST /v1/keys needs auth, and keys.py's loopback auto-seed only
    fires when the server was_open), for a file that unambiguously means no key
    (REG-579). localm itself never writes one - set_api_key('') unlinks instead -
    so an empty file is always an anomaly, and it is surfaced once rather than
    silently changing the server's security posture.

    Distinct from get_api_key(), which returns the key VALUE (or None when it
    cannot be read): when the file is present but unreadable this returns True
    (auth in effect) while verify() matches nothing, so every request is rejected
    (401 / locked) rather than served open - the safe direction."""
    global _empty_owner_key_warned
    env = os.environ.get(ENV_VAR)
    if env and env.strip():
        return True
    status, text, _err = _read_owner_key_file()
    if status == _KEY_ABSENT:
        return False                       # genuinely absent -> open by design
    if status == _KEY_UNREADABLE:
        # Present but we cannot read or decode it (a permission/profile change, a
        # parent-directory problem, a directory in its place, undecodable bytes):
        # we cannot tell whether a key exists, so fail CLOSED. Surfaced by
        # _read_key_file's own warning on the value path.
        return True
    if _key_text_or_none(text) is not None:
        # Re-arm the notice: a server that later drops KEYED -> OPEN (the file is
        # truncated while running) must say so again, or the second downgrade
        # would be the silent one.
        _empty_owner_key_warned = False
        return True
    if not _empty_owner_key_warned:
        _empty_owner_key_warned = True
        logger.warning(
            "owner key file %s exists but holds no key; treating it as NO key, "
            "so the server runs in OPEN (keyless) mode - set a key (the "
            "launcher, `localm key set`, or LOCALM_API_KEY) or delete the file",
            key_file())
    return False


def any_key_configured() -> bool:
    """True when auth is in effect: an owner key (env, or an auth.key file holding
    one - see _owner_key_present) OR a configured scoped keystore. When this is
    False the server runs open (unless require_auth_enabled()). A corrupt/
    unreadable keystore, and a present-but-unreadable owner key, both count as
    configured so they fail CLOSED rather than silently open: a keyed install must
    not lose its auth to a damaged or unreadable credential file (checkup
    2026-07-11 HIGH). A credential file we CAN read and which holds no key (an
    empty auth.key, an empty ``[]`` keystore) is not that case - it means exactly
    what it says, no key (REG-579)."""
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
    from localm import scopes as S
    presented = presented.strip()
    owner = get_api_key()
    if ct_equal(presented, owner):
        return {S.ADMIN}
    presented_hash = _hash_key(presented)
    for r in _load_keystore():
        # ct_equal, not compare_digest: both sides are normally hexdigests, but a
        # hand-edited or corrupted keystore row could hold a non-ASCII "hash" and
        # must fail to match, not 500 every request that reaches this loop.
        h = r.get("hash", "")
        if ct_equal(h, presented_hash):
            exp = r.get("expires")
            if exp is not None and time.time() > float(exp):
                return None       # matched a real key, but it has expired
            _touch_last_used(presented_hash)
            return set(r.get("scopes", []))
    return None
