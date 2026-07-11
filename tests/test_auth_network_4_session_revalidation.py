# SPDX-License-Identifier: AGPL-3.0-or-later
"""AUTH-NETWORK-4 (security): resolving a cookie session to its owning key's
hash was implemented twice with different revalidation behavior.

The canonical helper, ``_principal_from_token()``, re-checks that a non-ADMIN
session's underlying scoped key is still LIVE (``auth.key_hash_live``) before
trusting the session - so revoking or expiring the key also cuts off any
browser session it minted, mirroring the bearer-token path's per-request
``verify()``. ``principal_id()``'s cookie branch bypassed that helper and
called ``sessions.lookup()`` directly, skipping the liveness check entirely.

``principal_id()`` gates ``job_owner_ok`` (KEY-SCOPE-2): a revoked/expired
key's still-resident session cookie could therefore still resolve a key hash
that job-ownership checks trust, even though the very same cookie would
already be rejected everywhere else auth is enforced (``_enforce_request`` /
``caller_scopes`` / ``effective_fs_access``, which all go through
``_principal_from_token``).

Regression: ``principal_id()``'s cookie branch must call
``_principal_from_token()`` (the revalidating path), not ``sessions.lookup()``
directly.
"""

from starlette.requests import Request

from localm import auth, scopes, sessions
from localm.inference.http_server import principal_id


def _cookie_request(sid: str) -> Request:
    raw = [(b"cookie", f"localm_session={sid}".encode())]
    return Request({"type": "http", "headers": raw, "method": "GET",
                    "path": "/", "query_string": b""})


def test_live_scoped_key_session_still_resolves(monkeypatch):
    """Sanity: a session for a key that IS still live resolves normally."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")  # any_key_configured() gate
    key = auth.create_key("A", [scopes.MODELS_READ])["key"]
    key_hash = auth._hash_key(key)
    sid = sessions.create(scopes={scopes.MODELS_READ}, key_hash=key_hash, fs_access="none")
    assert principal_id(_cookie_request(sid)) == key_hash


def test_revoked_key_session_no_longer_resolves_a_principal(monkeypatch):
    """The security-relevant case: mint a session for a scoped key, then have
    the key stop being live (revoked/expired) WITHOUT the session itself being
    explicitly revoked (e.g. natural TTL expiry never runs revoke_by_key_hash
    - see auth.revoke_key's docstring: key_hash_live re-validation is the
    PRIMARY enforcement, session cleanup on revoke is only "belt and
    suspenders"). The stale session's cookie must NOT resolve a principal."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    created = auth.create_key("A", [scopes.MODELS_READ])
    key_hash = auth._hash_key(created["key"])
    sid = sessions.create(scopes={scopes.MODELS_READ}, key_hash=key_hash, fs_access="none")

    # The session is still resident in the session store...
    assert sessions.lookup(sid) is not None
    stale_records = sessions._load()  # snapshot the REAL record, schema untouched

    # ...but the underlying key is no longer live. revoke_key() also cleans up
    # sessions minted from the key as "belt and suspenders" (see its own
    # docstring); restore the exact snapshot afterward so the test isolates
    # the ACTUAL bug under test - key_hash_live revalidation - not that
    # separate cleanup path.
    assert auth.revoke_key(created["id"]) is True
    assert auth.key_hash_live(key_hash) is False
    sessions._save(stale_records)
    assert sessions.lookup(sid) is not None  # confirm the stale session round-trips

    assert principal_id(_cookie_request(sid)) is None


def test_admin_owner_session_exempt_from_key_liveness(monkeypatch):
    """An ADMIN/owner session is deliberately NOT gated on key_hash_live (an
    owner key roll must not log the owner out) - only SCOPED-key sessions are."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    sid = sessions.create(scopes={scopes.ADMIN}, key_hash=None, fs_access="host")
    assert principal_id(_cookie_request(sid)) is None  # key_hash is None: no key identity to report,
    # but this must not raise/crash: an ADMIN session must not be rejected outright either.
    from localm.inference.http_server import _principal_from_token
    prin = _principal_from_token(sid, "cookie")
    assert prin is not None
    held, key_hash, fs = prin
    assert scopes.ADMIN in held


def test_principal_id_cookie_branch_matches_principal_from_token(monkeypatch):
    """The two "resolve a cookie to its owning key hash" implementations must
    now be ONE implementation, not two that can silently diverge again."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    key = auth.create_key("A", [scopes.MODELS_READ])["key"]
    key_hash = auth._hash_key(key)
    sid = sessions.create(scopes={scopes.MODELS_READ}, key_hash=key_hash, fs_access="none")

    from localm.inference.http_server import _principal_from_token
    prin = _principal_from_token(sid, "cookie")
    assert prin is not None
    assert principal_id(_cookie_request(sid)) == prin[1]
