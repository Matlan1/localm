# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolving a cookie session to its owning key's hash goes through ONE helper.

``_principal_from_token()`` re-checks that a non-ADMIN session's underlying
scoped key is still LIVE (``auth.key_hash_live``) before trusting the session,
so revoking or expiring the key also cuts off any browser session it minted,
mirroring the bearer-token path's per-request ``verify()``.

``principal_id()`` gates ``job_owner_ok``, so its cookie branch must call
``_principal_from_token()`` rather than ``sessions.lookup()`` directly:
otherwise a revoked/expired key's still-resident session cookie resolves a key
hash that job-ownership checks trust, while the same cookie is rejected
everywhere else auth is enforced (``_enforce_request`` / ``caller_scopes`` /
``effective_fs_access``, which all go through ``_principal_from_token``).
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
    explicitly revoked (natural TTL expiry never runs revoke_by_key_hash; see
    auth.revoke_key's docstring, where key_hash_live re-validation is the
    PRIMARY enforcement). The stale session's cookie must NOT resolve a
    principal."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    created = auth.create_key("A", [scopes.MODELS_READ])
    key_hash = auth._hash_key(created["key"])
    sid = sessions.create(scopes={scopes.MODELS_READ}, key_hash=key_hash, fs_access="none")

    # The session is still resident in the session store...
    assert sessions.lookup(sid) is not None
    stale_records = sessions._load()  # snapshot the REAL record, schema untouched

    # ...but the underlying key is no longer live. revoke_key() also cleans up
    # sessions minted from the key, so the exact snapshot is restored afterward
    # to isolate key_hash_live revalidation from that cleanup path.
    assert auth.revoke_key(created["id"]) is True
    assert auth.key_hash_live(key_hash) is False
    sessions._save(stale_records)
    assert sessions.lookup(sid) is not None  # confirm the stale session round-trips

    assert principal_id(_cookie_request(sid)) is None


def test_admin_owner_session_exempt_from_key_liveness(monkeypatch):
    """An ADMIN/owner session is NOT gated on key_hash_live (an
    owner key roll must not log the owner out) - only SCOPED-key sessions are."""
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    sid = sessions.create(scopes={scopes.ADMIN}, key_hash=None, fs_access="host")
    assert principal_id(_cookie_request(sid)) is None  # key_hash is None: no key identity to report,
    # but this must not raise/crash: an ADMIN session must not be rejected outright either.
    from localm.inference.http_server import _principal_from_token
    prin = _principal_from_token(sid, "cookie")
    assert prin is not None
    held, key_hash, fs, rag_roots = prin
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
