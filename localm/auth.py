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
from collections import OrderedDict
from pathlib import Path
from typing import Callable
from typing import Optional

from localm.debuglog import logger

ENV_VAR = "LOCALM_API_KEY"
REQUIRE_ENV_VAR = "LOCALM_REQUIRE_AUTH"
_TRUTHY = ("1", "true", "yes", "on")

# Minimum length for an owner key to pass the length floor that gates a
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


def resolve_bearer_token(instance_token: Optional[str] = None) -> Optional[str]:
    """The bearer credential value (if any) a self-call or management client
    should present: the owner key (env, else the persisted ``auth.key``) if
    one is configured, else *instance_token* in OPEN (keyless) mode, else
    None.

    This is the precedence every self-authenticated caller of localm's own
    HTTP surface needs and none may skip: ``_origin_guard``'s open-mode gate
    (``http_server.py``) requires proof of a local process - a key when one
    exists, or the per-instance attach token (the 0600 registry file) when it
    does not - for any unsafe method or metadata GET not listed in
    ``_CROSS_ORIGIN_OK``; and ``_enforce_request``'s key check (every gated
    route once a key IS configured) accepts a real key ONLY - it has no
    notion of instance tokens at all, so presenting one there always 401s.
    *instance_token* is used ONLY when no key is configured, matching the
    server's own condition for requiring one at all; a protected-mode server
    always keeps using the real key.

    THE single place this precedence is decided. ``resolve_bearer_headers``
    below is a thin wrapper for callers that want a ready-to-use headers
    dict; callers that build their own request (or, like ``HttpEngine``,
    store the token and build the header per-call) should call this
    directly rather than re-deriving ``get_api_key() or instance_token``
    inline - see ``localm/inference/http_engine.py``'s ``HttpEngine``/
    ``remote_model_status`` callers (``cli/chat.py``, ``coder/cli/_main.py``)
    for the pattern.
    """
    key = get_api_key()
    if key:
        return key
    return instance_token or None


def resolve_bearer_headers(instance_token: Optional[str] = None) -> dict:
    """The ``Authorization`` header (if any) a self-call or management client
    should send - see ``resolve_bearer_token`` for the precedence. Returns a
    plain ``dict`` (empty when neither credential is available) rather than
    mutating a caller-supplied one, so every call site stays a simple
    ``headers = resolve_bearer_headers(...)``.

    Hoisted out of five independent copies of this precedence: four already
    implemented it in full (``selfclient.read_activity``/``self_request``,
    ``cli/models.py``'s ``unload_cmd``/``stop_cmd``); the fifth
    (``media/comfy_client.py``'s ``_localm_unload``) implemented only the
    first line (env var, via ``os.environ`` directly rather than
    ``get_api_key()``) and had no instance_token fallback at all - fixed
    alongside this extraction rather than ported as-is.
    """
    token = resolve_bearer_token(instance_token)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


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
    _atomic_write_private(path, key.strip() + "\n")
    # The key changed, so any memoised derivation for the OLD one is stale.
    _forget_cached_digests()
    # Derive at SET time, not on the per-request verify path: this is the
    # moment we are allowed to spend real time on a memory-hard KDF, and it
    # also means the salt is on disk before anything can stamp an identity
    # with it. NOT just one ~100ms derivation, though: _owner_digest first
    # re-verifies every KEPT historical record (_OWNER_KDF_KEEP) before
    # minting a new one, so a caller that sets the key repeatedly (an admin
    # route, a rotation loop, tests/test_auth.py's own 200x charset test)
    # pays that on every call, in whatever thread called it - MEASURED
    # 2026-08-12 at up to 9 full derivations (~1.7-2s on a loaded box) before
    # _OWNER_KDF_KEEP was cut from 8 to 3 for exactly this reason (see
    # dev-notes/FIX-2026-08-12-test-set-api-key-hang-preexisting.md). Still
    # best-effort regardless: a failure costs a derivation on first use, not
    # access.
    try:
        _owner_digest(key)
    except Exception as e:
        logger.warning("could not pre-derive the owner key identity (%s); it "
                       "will be derived on first use instead", e)


def clear_api_key() -> list[dict[str, str]]:
    """Remove the persisted key (open mode). A leftover env var still applies.

    RETURNS the list of things that could NOT be removed. An EMPTY list means the
    clear genuinely completed; a non-empty one means credentials SURVIVE and the
    caller must not report success. Each entry is::

        {"what":  "the API key file",        # path-free label, safe anywhere
         "path":  "<absolute path>",         # LOCAL surfaces only
         "error": "PermissionError: ..."}    # LOCAL surfaces only

    **A caller on a NETWORK surface must expose only ``what``.** ``path`` is an
    absolute filesystem path (which carries the account name, so putting it in an
    HTTP response is a rule-2 disclosure) and ``error`` is raw OS exception text
    (``py/stack-trace-exposure``). The local CLI prints all three, because there
    the user owns the machine and needs the path to fix it by hand.

    The return value exists at all because the warnings below go to
    ``debuglog.logger``, which a user never sees without ``--debug``. That made
    every caller's "cleared" message unconditional and therefore FALSE whenever a
    delete failed - the precise thing AGENTS.md rule 5 forbids, since this is a
    security step and a surviving key STILL GRANTS ACCESS. Warning into a log
    nobody reads is not surfacing; the caller has to be able to ask."""
    failures: list[dict[str, str]] = []
    try:
        key_file().unlink(missing_ok=True)
    except OSError as e:
        # Surface, do not silence: this is a security step (removing the persisted
        # key). If the file cannot be deleted the key STILL GRANTS access, so a
        # silent pass would imply a clear that did not happen (rule 5). Warn loudly.
        logger.warning("could not remove the API key file %s (%s); the key may "
                       "still be active until it is deleted by hand", key_file(), e)
        failures.append({"what": "the API key file", "path": str(key_file()),
                         "error": f"{type(e).__name__}: {e}"})
    try:
        keystore_file().unlink(missing_ok=True)
    except OSError as e:
        # Same: a leftover keystore means scoped keys remain valid. Do not let a
        # failed delete look like a successful clear.
        logger.warning("could not remove the keystore %s (%s); scoped keys may "
                       "still be active until it is deleted by hand",
                       keystore_file(), e)
        failures.append({"what": "the keystore (scoped device keys)",
                         "path": str(keystore_file()),
                         "error": f"{type(e).__name__}: {e}"})
    # The derivation records describe credentials that no longer exist. They hold
    # no plaintext and authenticate nothing, but a clear that leaves credential
    # artefacts behind is not a clear - and a stale salt would silently re-link a
    # future identical key to the cleared install's identities.
    try:
        owner_kdf_file().unlink(missing_ok=True)
    except OSError as e:
        logger.warning("could not remove the owner key derivation records %s "
                       "(%s); delete it by hand", owner_kdf_file(), e)
        failures.append({"what": "the owner key derivation records",
                         "path": str(owner_kdf_file()),
                         "error": f"{type(e).__name__}: {e}"})
    _forget_cached_digests()
    return failures


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


def _restrict_perms(path: Path) -> bool:
    """Best-effort: restrict the key file to the current user. No-op on failure
    or unsupported platforms - the data dir is already user-scoped. Returns True
    when the tightening is believed to have happened.

    The implementation moved to ``config.restrict_file_perms`` so sessions.json
    and jobs.json (which hold the key DIGEST) get the identical treatment as
    auth.key (which holds the PLAINTEXT); they previously did not on Windows.

    RETAINED although ``_atomic_write_private`` no longer calls it (that moved
    to ``config.atomic_write_private``, which calls ``restrict_file_perms``
    itself): this name is the unit-level equivalence point that pins the fix.
    ``tests/test_auth_kdf.py`` calls it directly against
    ``sessions._restrict_perms`` and asserts both return the same value and
    produce the same ACL. Same reasoning as the note on that sibling.

    The bool is PASSED THROUGH rather than discarded so a caller doing the atomic
    temp+replace dance can restrict the temp file and retry on the destination
    only when the first attempt failed - the contract the shared writer relies
    on. Dropping it here made that pattern silently degrade to always retrying."""
    from localm.config import restrict_file_perms
    return restrict_file_perms(path)


def _atomic_write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, owner-restricted from the moment the
    bytes first exist on disk. See ``config.atomic_write_private`` for the full
    reasoning, the two precedents it composes, and the two details that are
    invisible when they are wrong (do NOT add ``os.O_BINARY``; keep the write
    loop).

    The implementation moved there once the same dance was needed by
    sessions.json, the instance registry entry and the GPU coordination entry
    as well as this module's three credential files - one implementation so a
    sixth caller cannot get a weaker sixth variant, the same reason
    ``_restrict_perms`` above delegates rather than reimplementing.

    Kept as a name here because it is referenced throughout this module and by
    ``tests/test_auth_secret_write_perms.py``. The bool the shared writer
    returns is deliberately dropped: no caller in this module logs a
    subsystem-named warning of its own, and reporting the failed tightening is
    already ``restrict_file_perms``'s job."""
    from localm.config import atomic_write_private
    atomic_write_private(path, text)


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


# --------------------------------------------------------------------------- #
#  Key digests: a KDF for the user-choosable owner key, fast for generated ones #
# --------------------------------------------------------------------------- #
# Two kinds of secret are digested here and they are NOT equivalent:
#
#   * NAMED KEYSTORE KEYS are always secrets.token_urlsafe(32) (create_key is the
#     only writer) - 256 bits of CSPRNG output. No dictionary and no rainbow table
#     touches 2^256, so a KDF's work factor buys nothing. These stay on the cheap
#     path, marked EXPLICITLY on the record (never inferred from the key's shape).
#   * THE OWNER KEY CAN BE USER-CHOSEN and may be human-memorable. `localm key set
#     KEY` persists a key the user provides, and LOCALM_API_KEY / a hand-edited
#     auth.key bypass set_api_key entirely, so they are not even length- or
#     charset-checked. Its digest is PERSISTED (sessions.json key_hash, jobs.json
#     owner), where one fast unsalted hash is an offline brute-force oracle that
#     recovers the PLAINTEXT - which does authenticate. That is CodeQL alert 88
#     (py/weak-sensitive-data-hashing) and it is a true positive.
#
# The digest is also a stable PRINCIPAL IDENTIFIER: it is recomputed later and
# compared with == (jobs.runner._shell_still_authorized, principal_id -> job
# ownership, sessions.key_hash). So it must be DETERMINISTIC for a given key - a
# per-call random salt would break every one of those. The salt is therefore
# persisted PER KEY, and the derivation is memoised per process (see
# _digest_cache) so the KDF never runs on the per-request verify path more than
# once per key.
_SCRYPT_N = 2 ** 14            # RFC 7914 interactive-login cost
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# 128*N*r = 16 MiB is the actual working set; ask for headroom explicitly rather
# than relying on OpenSSL's default cap, which has varied across versions and
# fails the derivation outright when it is too low.
_SCRYPT_MAXMEM = 64 * 1024 * 1024

# Marker recorded ON a stored record saying which construction produced its
# digest, so the cheap path is a DECLARED property of that record and not a guess
# from the key's shape. A record with no marker predates this and is a generated
# keystore token (create_key was the only writer), so it reads as _ALG_FAST and is
# upgraded in place on its next successful verify.
_ALG_FAST = "sha256"
_ALG_KDF = "scrypt"

# How many owner-key verifier records to keep. The same install legitimately sees
# more than one owner key over time (a LOCALM_API_KEY override for one run, then
# the file again; a rotation), and the SAME key must always derive the SAME digest
# or job ownership and session identity would flip underneath the user. Keeping a
# few records makes that stable; the cap stops the file growing without bound.
#
# It is ALSO the direct bound on how many full scrypt derivations a single
# set_api_key call can burn: _owner_kdf_record_for re-verifies EVERY kept
# record (a real derivation each, there is no cheap way to rule one out - see
# _memo_key's docstring on why a fast index was rejected here) before minting
# a new one. MEASURED 2026-08-12: at the original value of 8, a saturated
# records list made every set_api_key call run up to 9 full derivations
# (~1.7-2s on a loaded box, not the ~100ms its call site used to budget for) -
# see dev-notes/FIX-2026-08-12-test-set-api-key-hang-preexisting.md. 3 is the
# smallest value with headroom above the 2-key scenario described above (an
# env-var override alternating with the persisted file).
_OWNER_KDF_KEEP = 3


def owner_kdf_file() -> Path:
    """Path to the owner-key KDF verifier records (inside the localm data dir)."""
    from localm.config import home_dir
    return home_dir() / "auth.kdf.json"


def _fast_digest(key: str) -> str:
    """Unsalted sha256 of a GENERATED 256-bit token. Never call this on a secret
    a human may have chosen - that is what the KDF path exists for.

    The only value this produces that is ever PERSISTED is a keystore record's
    hash, and create_key is the only writer of those: every one is
    secrets.token_urlsafe(32). At 256 bits of CSPRNG entropy no dictionary and no
    rainbow table applies, so hash speed is irrelevant there.

    surrogatepass for the same reason as ct_equal: a plain utf-8 encode raises
    UnicodeEncodeError on a lone surrogate, which would move the crash here from
    the compare. Byte-identical to encode("utf-8") for every key that encodes at
    all, so no stored digest changes."""
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def _legacy_owner_identity(key: str) -> str:
    """The digest the owner key USED to be identified by, before the KDF landed.

    Reproducing a historical value is inherently tied to the construction that
    produced it, so this is the one place a possibly-user-chosen key still meets
    a fast hash. It is confined to the MIGRATION path and is never persisted: it
    is used to FIND rows still carrying the old identity so they can be rewritten
    to the derived one (and, in the jobs plugin, to recognise a job stamped
    before the upgrade so it is not orphaned). Nothing is stored under it and
    nothing authenticates from it.

    Separate function from _fast_digest despite the identical body, so the two
    uses cannot be confused: one is "a generated token's permanent id", this one
    is "a legacy value we are migrating AWAY from". When no supported install
    predates the KDF any more, this function and its callers can be deleted
    outright - _fast_digest cannot."""
    return _fast_digest(key)


def _memo_key(key: str) -> str:
    """Lookup handle for the in-memory memo: the presented secret's fast digest.

    A keyed HMAC was tried here and REVERTED. It read better in isolation (a
    scraped memo would mean nothing in another process) but it was wrong twice
    over: it deviated from the remedy the maintainer specified ("cache the
    presented secret's fast digest -> verified marker"), and CodeQL models
    hmac.new(..., sha256) as the SAME weak-password-hash sink, so it turned two
    traced flows into eight while changing no observable behaviour. It bought a
    property nobody needed at the cost of the one the design asked for.

    This value is never persisted and never leaves the process; the digest that
    IS written to disk for the owner key is the scrypt derivation."""
    return _fast_digest(key)


def _scrypt_derive(key: str, salt: bytes, n: int, r: int, p: int,
                   dklen: int) -> str:
    """Salted scrypt of *key* -> hex. A module-level function (not an inline call)
    so the memoisation above it can be proven to work: a test patches this and
    counts the calls."""
    return hashlib.scrypt(key.encode("utf-8", "surrogatepass"), salt=salt,
                          n=n, r=r, p=p, dklen=dklen,
                          maxmem=_SCRYPT_MAXMEM).hex()


# Memoises the EXPENSIVE derivation only: fast-path digests are never inserted, so
# a caller spraying random bearer tokens cannot pollute or grow this (the entries
# it would create are exactly the ones we refuse to make). Bounded LRU because an
# unbounded cache on a per-request path is a memory leak. Keyed on the presented
# secret's fast DIGEST plus the data home - never the plaintext, and never a bare
# digest, because one process legitimately serves more than one LOCALM_HOME (the
# test suite does) and each home has its own salt.
_DIGEST_CACHE_MAX = 64
_digest_cache: "OrderedDict[str, str]" = OrderedDict()
_DIGEST_CACHE_LOCK = threading.Lock()

# Serialises MINTING an owner-key record. Without it two concurrent requests
# presenting the owner key with a cold memo both find no record, both mint a
# FRESH SALT, and both write - so the two requests stamp two different identities
# and only one of them matches what ends up on disk. The identity would then be
# unstable exactly when the server is busiest. Distinct from _DIGEST_CACHE_LOCK
# (which only guards the dict) because it must be held across the whole
# read-derive-write, and it is taken BEFORE any sessions lock, never after.
#
# RLock, not Lock: _hash_key takes it to make its check-then-derive atomic and
# then calls _owner_digest, which takes it again so that set_api_key's DIRECT
# call is covered by the same mutex. A plain Lock would deadlock that nesting.
_OWNER_KDF_LOCK = threading.RLock()


def _cache_scope() -> str:
    """Scope component for a cache entry: which data home the salt came from."""
    try:
        return str(owner_kdf_file())
    except Exception:
        return ""


def _cache_get(ck: str) -> Optional[str]:
    with _DIGEST_CACHE_LOCK:
        hit = _digest_cache.get(ck)
        if hit is not None:
            _digest_cache.move_to_end(ck)
        return hit


def _cache_put(ck: str, digest: str) -> None:
    with _DIGEST_CACHE_LOCK:
        _digest_cache[ck] = digest
        _digest_cache.move_to_end(ck)
        while len(_digest_cache) > _DIGEST_CACHE_MAX:
            _digest_cache.popitem(last=False)


def _forget_cached_digests() -> None:
    """Drop the memoised derivations. Called whenever the owner key changes, so a
    rotation takes effect at once instead of serving a stale identity."""
    with _DIGEST_CACHE_LOCK:
        _digest_cache.clear()


def _load_owner_kdf() -> list:
    """The owner-key verifier records, or [] when there are none.

    A present-but-unreadable/corrupt file returns [] so a fresh record is minted
    rather than locking the owner out of their own identity - the file is a
    derivation aid, not a credential: it holds no plaintext, and nothing
    AUTHENTICATES from it (the owner key is verified by plaintext ct_equal against
    auth.key). Failing closed here would buy no security and would break a working
    install, so the damage is SURFACED and repaired instead of being fatal."""
    path = owner_kdf_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        logger.warning("owner key KDF file %s is unreadable or corrupt (%s); "
                       "minting a fresh derivation record. Sessions and jobs "
                       "owned by the owner key are re-linked automatically on "
                       "the next successful verify", path, e)
        return []
    if not isinstance(data, dict):
        return []
    recs = data.get("records")
    return [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []


def _save_owner_kdf(records: list) -> None:
    from localm.config import ensure_dirs
    ensure_dirs()
    path = owner_kdf_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restrict the TEMP file (it already holds the whole payload) before the
    # rename, so a crash between the two lines cannot leave an unrestricted
    # copy behind. This function had that half right while set_api_key and
    # _save_keystore did not; _atomic_write_private is the one implementation
    # all three now share, and it additionally creates the temp file already
    # restricted rather than tightening it a moment later.
    _atomic_write_private(path, json.dumps({"v": 1, "records": records},
                                           indent=2))


def _owner_kdf_record_for(key: str, records: list) -> Optional[dict]:
    """The existing verifier record matching *key*, or None.

    Verifies with EACH record's OWN stored parameters, which is what lets the cost
    parameters be raised later without invalidating a key derived under the old
    ones. Only ever called for a key already known to be the owner key.

    NOT "once per process": that only holds on the verify() path (_hash_key
    memoises the result - see _digest_cache). set_api_key calls this again on
    EVERY set, uncached, so a caller that sets the key repeatedly (an admin
    route, a rotation loop, a test) pays a full scrypt derivation per kept
    record, per call. _OWNER_KDF_KEEP is the actual bound on that cost."""
    for r in records:
        if r.get("alg") != _ALG_KDF:
            continue
        try:
            salt = bytes.fromhex(str(r.get("salt", "")))
            digest = _scrypt_derive(key, salt, int(r["n"]), int(r["r"]),
                                    int(r["p"]), int(r["dklen"]))
        except (ValueError, KeyError, TypeError) as e:
            # A hand-edited or truncated row must not break the whole lookup.
            logger.debug("skipping an unusable owner KDF record (%s)", e)
            continue
        if ct_equal(digest, str(r.get("digest", ""))):
            return r
    return None


def _owner_digest(key: str) -> str:
    """The KDF-derived principal identity for the OWNER key *key*.

    Reuses the persisted record for this key when there is one, so the value is
    stable forever; otherwise mints one (fresh salt, current cost parameters) and
    migrates any identity previously recorded under the legacy unsalted digest.

    Holds _OWNER_KDF_LOCK across the whole read-derive-write so two callers
    cannot mint two different salts for the same key (see the lock's comment).
    Reentrant, so _hash_key may hold it already."""
    with _OWNER_KDF_LOCK:
        return _owner_digest_locked(key)


def _owner_digest_locked(key: str) -> str:
    records = _load_owner_kdf()
    existing = _owner_kdf_record_for(key, records)
    if existing is not None:
        return str(existing["digest"])
    salt = secrets.token_bytes(16)
    rec = {
        "alg": _ALG_KDF, "salt": salt.hex(), "n": _SCRYPT_N, "r": _SCRYPT_R,
        "p": _SCRYPT_P, "dklen": _SCRYPT_DKLEN, "created": time.time(),
        "digest": _scrypt_derive(key, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
                                 _SCRYPT_DKLEN),
    }
    records.append(rec)
    try:
        _save_owner_kdf(records[-_OWNER_KDF_KEEP:])
    except OSError as e:
        # Not fatal: the digest is still correct for THIS process, so auth keeps
        # working. But it is not persisted, so the next process mints a different
        # salt and any identity stored under this one stops matching. Say so -
        # a silent failure here would look like a stable identity that is not
        # (AGENTS.md rule 5).
        logger.warning("could not persist the owner key derivation record %s "
                       "(%s); sessions and jobs stamped in this process may not "
                       "be recognised after a restart", owner_kdf_file(), e)
    _migrate_legacy_owner_identity(_legacy_owner_identity(key), rec["digest"])
    return str(rec["digest"])


def _migrate_legacy_owner_identity(legacy: str, current: str) -> None:
    """Re-link anything recorded under the owner key's OLD unsalted digest.

    Runs once, when a key's KDF record is first minted (so an upgraded install
    migrates on the owner's next successful verify, and an install that never
    re-authenticates is never touched). Best-effort by design: a failure here
    costs a re-login, never access, so it is surfaced rather than raised.

    jobs.json is deliberately NOT rewritten here - auth.py has no business
    importing a plugin's store, and that side already has a documented
    back-compat path that accepts the legacy digest and stamps the job as
    owner-owned the first time it matches (see jobs.runner)."""
    if not legacy or not current or legacy == current:
        return
    try:
        from localm import sessions
        n = sessions.relink_key_hash(legacy, current)
        if n:
            logger.debug("re-linked %d session(s) to the owner key's derived "
                         "identity", n)
    except Exception as e:
        logger.warning("could not re-link existing sessions to the owner key's "
                       "new derived identity (%s); those sessions stay valid, "
                       "but a job created from one may not be recognised as the "
                       "owner's until the next sign-in", e)


def _is_owner_key(key: str) -> bool:
    """True when *key* is the owner key currently in effect. An exact identity
    check against the live value, NOT a guess from the key's length or alphabet -
    a user-chosen owner key is indistinguishable from a token by shape."""
    return ct_equal(key, get_api_key())


def _hash_key(key: str) -> str:
    """The stable identity digest for *key*.

    The owner key (user-choosable, possibly human-memorable) gets a salted scrypt
    derivation; a generated keystore token gets the cheap unsalted digest, which
    is sound at 256 bits of CSPRNG entropy. The expensive path is memoised per
    process, so it costs one derivation per key rather than one per request.

    The memo is keyed by a per-process MAC (_memo_key), NOT by a fast hash of the
    secret: the fast digest is now computed only on the branch that actually
    returns one, so a possibly-user-chosen owner key never reaches an unsalted
    hash here."""
    ck = _memo_key(key) + "@" + _cache_scope()
    hit = _cache_get(ck)
    if hit is not None:
        return hit
    if not _is_owner_key(key):
        # A generated keystore token, or a token that matches nothing at all
        # (every wrong guess lands here). Cheap, and never cached, so an attacker
        # spraying tokens neither pays nor grows anything.
        return _fast_digest(key)
    with _OWNER_KDF_LOCK:
        # Re-check under the lock: a concurrent request may have minted the
        # record (and warmed the memo) while we waited, and deriving again here
        # would mint a SECOND salt for the same key.
        hit = _cache_get(ck)
        if hit is not None:
            return hit
        derived = _owner_digest(key)
        _cache_put(ck, derived)
    return derived


def _record_digest_for(record: dict, key: str, fast: Callable[[], str]) -> Optional[str]:
    """The digest *key* would produce under *record*'s DECLARED construction, or
    None when the record cannot be evaluated.

    *fast* is a CALLABLE, not a string, so the cheap digest is computed only on
    the branch that actually returns one. See _find_keystore_record for why.

    A record with no "alg" predates the marker, and create_key was the only writer
    then, so it is a generated token on the cheap path. An UNKNOWN alg returns
    None (refuse to match) rather than falling back to the cheap digest: guessing
    would let a future strong record be matched by a weak comparison."""
    alg = record.get("alg") or _ALG_FAST
    if alg == _ALG_FAST:
        return fast()
    if alg == _ALG_KDF:
        try:
            return _scrypt_derive(key, bytes.fromhex(str(record.get("salt", ""))),
                                  int(record["n"]), int(record["r"]),
                                  int(record["p"]), int(record["dklen"]))
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("keystore record %s has unusable KDF parameters (%s); "
                         "it cannot match", record.get("id"), e)
            return None
    logger.warning("keystore record %s declares an unknown hash algorithm %r; "
                   "refusing to match it. A key written by a NEWER localm cannot "
                   "be verified by this one - upgrade rather than editing the "
                   "keystore by hand", record.get("id"), alg)
    return None


def _find_keystore_record(key: str, records: list) -> Optional[dict]:
    """The keystore record *key* authenticates against, or None.

    The cheap digest is computed LAZILY, on first use, rather than up front. The
    presented key here may be a human-chosen owner key, and hashing it unsalted
    just because the loop MIGHT later meet a fast-alg row is the same mistake
    _owner_identity already avoids on its own branch ("computed only on the
    branch that actually returns one"); this path had not been given the same
    treatment. An owner key's record declares the KDF alg, so its fast digest was
    never the value that authenticated it - it was computed and discarded. Now it
    is not computed at all unless a record actually declares the fast
    construction (CodeQL 247).

    Memoised, so a keystore holding many generated-token rows still hashes once
    per lookup rather than once per row."""
    _fast_memo: list = []

    def fast() -> str:
        if not _fast_memo:
            _fast_memo.append(_fast_digest(key))
        return _fast_memo[0]

    for r in records:
        cand = _record_digest_for(r, key, fast)
        if cand is None:
            continue
        # ct_equal, not compare_digest: both sides are normally hexdigests, but a
        # hand-edited or corrupted keystore row could hold a non-ASCII "hash" and
        # must fail to match, not 500 every request that reaches this loop.
        if ct_equal(str(r.get("hash", "")), cand):
            return r
    return None


def _mark_record_alg(key_id: Optional[str], alg: str) -> None:
    """Stamp the construction marker onto a legacy record, in place.

    This is the transparent format upgrade: a store written before the marker
    existed keeps verifying, and the first successful verify records what it
    actually is, so the ambiguity is resolved once instead of being re-guessed
    forever. Best-effort - a read-only store must not break authentication."""
    if not key_id:
        return
    try:
        with _KEYSTORE_LOCK:
            records = _load_keystore()
            for r in records:
                if r.get("id") == key_id and not r.get("alg"):
                    r["alg"] = alg
                    _save_keystore(records)
                    return
    except Exception as e:
        logger.debug("could not stamp the hash-alg marker on key %s (%s); it "
                     "still verifies, the upgrade is retried next time",
                     key_id, e)


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
    _atomic_write_private(path, json.dumps(records, indent=2))


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
    rec = _find_keystore_record(token.strip(), _load_keystore())
    if rec is None:
        return default
    return norm_fs_access(rec.get("fs_access", default))


def norm_rag_roots(roots) -> list:
    """Coerce *roots* to a clean, order-preserving, de-duplicated list of
    folder-path strings for a key's per-key RAG-indexing allowlist; anything
    that is not a non-empty string is dropped. None/blank -> [] (no per-key
    restriction - the key falls back to the global ``rag_allowed_roots`` policy,
    see rag/store.py's ``confine_index_path``).

    *roots* must be a list/tuple; a bare string is explicitly REJECTED (as [])
    rather than iterated - a plain Python string is itself iterable character by
    character, so without this guard a caller accidentally passing a single path
    string (instead of ``[path]``) would silently explode it into one
    single-character "root" per character."""
    if not roots or not isinstance(roots, (list, tuple)):
        return []
    out: list = []
    seen: set = set()
    for r in roots:
        if not isinstance(r, str):
            continue
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def rag_roots_for(token: str, default: Optional[list] = None) -> list:
    """The stored per-key RAG-indexing folder allowlist for the key behind
    *token*, or *default* (``[]`` if not given) if the key is unknown or has no
    list recorded. A legacy key minted before this attribute existed, and any
    key that never had one set, resolves to the safe default of NO per-key
    restriction - it falls back to the global policy, same "absent means
    unrestricted-by-this-field" shape as ``fs_access_for``'s legacy default."""
    if default is None:
        default = []
    if not token or not token.strip():
        return default
    rec = _find_keystore_record(token.strip(), _load_keystore())
    if rec is None:
        return default
    return norm_rag_roots(rec.get("rag_roots", default))


def list_keys() -> list:
    """Public metadata for every named key (never the hash)."""
    return [
        {"id": r.get("id"), "name": r.get("name", ""),
         "scopes": r.get("scopes", []), "created": r.get("created"),
         "expires": r.get("expires"), "last_used": r.get("last_used"),
         "fs_access": norm_fs_access(r.get("fs_access", "none")),
         "rag_roots": norm_rag_roots(r.get("rag_roots", []))}
        for r in _load_keystore()
    ]


def create_key(name: str, scope_list, *, allow_privileged: bool = False,
               expires: Optional[float] = None, fs_access: str = "none",
               rag_roots: Optional[list] = None) -> dict:
    """Mint a named key with *scope_list*, persist its hash, and return a record
    INCLUDING the plaintext key once - the caller must surface it now, it cannot
    be recovered. Raises ValueError on an unknown scope.

    *fs_access* is the host-filesystem reach this key grants ("none" | "shared" |
    "host"); it defaults to the safe "none" so a new scoped key cannot browse the
    server disk unless the owner deliberately grants it. The owner/ADMIN key
    always resolves to "host" regardless of this field (see effective_fs_access).

    *rag_roots* is an optional per-key RAG-indexing folder allowlist, following
    the exact same shape as *fs_access*: empty/None (the default) grants no
    per-key restriction, so the key falls back to the global
    ``rag_allowed_roots`` policy that already applies to every caller
    (rag/store.py's ``confine_index_path``) - unchanged behavior for every key
    minted before this field existed. A non-empty list instead CONFINES the key
    to exactly those folders: the home directory, the working directory and the
    global allowed-roots list are deliberately not implied on top of it, so a
    scoped key handed to an integration reaches only what it was explicitly
    given - "each key gets its own explicit folder allowlist". The owner/ADMIN
    key is never confined by this field regardless of what is stored (see
    effective_rag_roots), exactly like fs_access.

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
        # _fast_digest, not _hash_key: this key was just generated by
        # generate_key(), so it is 256 bits of CSPRNG output by construction and
        # the cheap digest is sound. The "alg" marker RECORDS that decision on the
        # row, so the verify path reads it as a declared property instead of
        # inferring it from the key's length or alphabet - a user-chosen key is
        # indistinguishable from a token by shape, which is what made the original
        # "it is always a generated token" premise wrong (CodeQL 88 / C13).
        "hash": _fast_digest(key),
        "alg": _ALG_FAST,
        "scopes": clean,
        "created": time.time(),
        "expires": float(expires) if expires is not None else None,
        "fs_access": norm_fs_access(fs_access),
        "rag_roots": norm_rag_roots(rag_roots),
    }
    with _KEYSTORE_LOCK:
        records = _load_keystore()
        records.append(record)
        _save_keystore(records)
    return {"id": record["id"], "name": record["name"],
            "scopes": record["scopes"], "expires": record["expires"],
            "fs_access": record["fs_access"], "rag_roots": record["rag_roots"],
            "key": key}


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
            if sessions.revoke_by_key_hash(target["hash"]) is None:
                # WARNING, not debug: the docstring above calls this
                # belt-and-suspenders because the cookie path re-validates a
                # scoped session's key every request - but an ADMIN-scoped
                # session is exempt from that re-check (so an owner-key roll
                # cannot sign the owner out). So for an ADMIN-scoped DEVICE key
                # this cleanup IS the enforcement, and a failed write leaves its
                # cookie working. The key itself is genuinely revoked either way,
                # which is what this function reports, so the honest altitude is
                # a warning naming the residue rather than failing the revoke.
                logger.warning(
                    "key %s was revoked, but its browser sessions could not be "
                    "dropped; an admin-scoped session for it may still be "
                    "usable until the session store is writable again", key_id)
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


def scopes_for_key_hash(key_hash: Optional[str]) -> Optional[set]:
    """The scopes a LIVE keystore key with this sha256 *key_hash* grants, or None.

    The by-hash sibling of ``verify()``: same liveness rules (a missing or expired
    record grants nothing), but keyed on the stored digest rather than a presented
    secret, so a background path that only ever recorded a principal id can still
    ask what that principal is allowed to do. ``verify()`` cannot serve that - it
    needs the plaintext key, which is exactly what is never persisted.

    Returns None (not an empty set) when nothing matches, so "no such key" stays
    distinguishable from "a key that grants nothing". A caller making a privilege
    decision must treat None as DENY.

    Does NOT resolve the owner key: it is not a keystore entry, so a caller that
    cares about the owner must compare against ``get_api_key()`` itself. Nor is it
    a substitute for ``key_hash_live``: a caller wanting only liveness should keep
    asking that, and this adds the scope question on top."""
    if not key_hash:
        return None
    now = time.time()
    # _load_keystore() fails OPEN (returns [] on OSError/ValueError), so a
    # transient unreadable auth.json makes this return None - which the contract
    # above requires callers to read as DENY. That is the safe direction, and it
    # is why the return is Optional rather than a bare set.
    for r in _load_keystore():
        if r.get("hash") == key_hash:
            exp = r.get("expires")
            if exp is not None and now > float(exp):
                return None
            return set(r.get("scopes", []))
    return None


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
    rec = _find_keystore_record(presented, _load_keystore())
    if rec is None:
        return None
    exp = rec.get("expires")
    if exp is not None and time.time() > float(exp):
        return None               # matched a real key, but it has expired
    if not rec.get("alg"):
        # Legacy row: it verified on the cheap path, which is what it has always
        # been. Record that now so the construction is declared rather than
        # assumed on every later verify.
        _mark_record_alg(rec.get("id"), _ALG_FAST)
    _touch_last_used(str(rec.get("hash", "")))
    return set(rec.get("scopes", []))
