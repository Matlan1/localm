# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the shared API-key auth (localm/auth.py) and its enforcement in
the HTTP server's _require_auth dependency."""

import logging

import pytest
from fastapi import HTTPException


def _req(token=None, method="GET"):
    """Minimal Starlette Request carrying an optional Bearer token, for unit-
    calling the request-aware auth dependencies (S2: header-or-cookie auth)."""
    from starlette.requests import Request
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "method": method, "headers": headers,
                    "path": "/", "query_string": b"", "client": ("test", 0)})


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth with a throwaway data dir and a clean auth environment."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    # config.py freezes these paths at import, so LOCALM_HOME alone won't
    # redirect load_config/save_config (used by require_auth_enabled). Point
    # them at the throwaway dir so a test never touches the real config.
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    return a


def test_generate_is_random(auth):
    a, b = auth.generate_key(), auth.generate_key()
    assert a and b and a != b


def test_set_get_clear_roundtrip(auth):
    # Keys must be >= 8 chars (auth.MIN_KEY_LEN); use a realistic one.
    assert auth.get_api_key() is None
    auth.set_api_key("s3cret-key")
    assert auth.get_api_key() == "s3cret-key"
    auth.clear_api_key()
    assert auth.get_api_key() is None


def test_env_overrides_file(auth, monkeypatch):
    auth.set_api_key("file-key-1")
    monkeypatch.setenv("LOCALM_API_KEY", "envkey")
    assert auth.get_api_key() == "envkey"


def test_empty_values_mean_no_key(auth, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "   ")
    assert auth.get_api_key() is None       # whitespace env = open
    auth.set_api_key("   ")                  # whitespace write = clear
    assert auth.get_api_key() is None


def test_regenerate_persists_new_key(auth):
    k1 = auth.regenerate_key()
    k2 = auth.regenerate_key()
    assert k1 != k2
    assert auth.get_api_key() == k2


def test_require_flag_env_and_config(auth, monkeypatch):
    assert auth.require_auth_enabled() is False
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    assert auth.require_auth_enabled() is True
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH")
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["require_auth"] = True
    save_config(cfg)
    assert auth.require_auth_enabled() is True


def test_config_read_failure_resolves_required(auth, monkeypatch, caplog):
    # LM-DA-021: an unreadable config must NOT silently downgrade an explicit
    # require_auth: true to "not required" (fail-open). It fails SAFE to
    # "required" and surfaces a warning, matching the newer fail-closed
    # precedent this codebase established for the identical question in
    # netpolicy.network_mode() (HON-2) and this module's own
    # any_key_configured() - erring toward MORE restriction, never less.
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth.require_auth_enabled() is True
    assert any("require_auth" in r.message and "required" in r.message.lower()
               for r in caplog.records), \
        "config-read failure resolved silently (no warning)"


def test_config_read_failure_env_var_short_circuits(auth, monkeypatch):
    # LOCALM_REQUIRE_AUTH is checked BEFORE config is ever read, so a truthy
    # env var must still return True without touching (or being tripped up
    # by) an unreadable config file - proves the ordering, not just the
    # fail-closed default above.
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    assert auth.require_auth_enabled() is True


def test_require_auth_dependency(auth, monkeypatch):
    from localm.inference.http_server import _require_auth

    # open mode: no key, not required -> allowed
    assert _require_auth(_req()) is None

    # required but no key configured -> fail closed (401; was 503). A 401 makes
    # the GUI prompt for a key rather than show a server-down overlay.
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    with pytest.raises(HTTPException) as exc:
        _require_auth(_req())
    assert exc.value.status_code == 401
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH")

    # key configured (file): missing/invalid credentials -> 401
    auth.set_api_key("owner-key-1")
    with pytest.raises(HTTPException) as exc:
        _require_auth(_req())
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        _require_auth(_req("wrong"))
    assert exc.value.status_code == 401

    # correct credentials -> allowed
    assert _require_auth(_req("owner-key-1")) is None


# --------------------------------------------------------------------------- #
#  Scoped keystore + scope enforcement (Phase 1)                              #
# --------------------------------------------------------------------------- #

def test_keystore_create_list_revoke(auth):
    from localm import scopes as S
    assert auth.list_keys() == []
    made = auth.create_key("reader", [S.CHAT, S.MODELS_READ])
    assert made["key"]
    assert set(made["scopes"]) == {S.CHAT, S.MODELS_READ}
    listed = auth.list_keys()
    assert len(listed) == 1
    assert "hash" not in listed[0]                 # never expose the hash
    assert auth.verify(made["key"]) == {S.CHAT, S.MODELS_READ}
    assert auth.revoke_key(made["id"]) is True
    assert auth.verify(made["key"]) is None
    assert auth.list_keys() == []


def test_create_key_rejects_unknown_scope(auth):
    with pytest.raises(ValueError):
        auth.create_key("x", ["bad scope"])


def test_any_key_configured(auth):
    assert auth.any_key_configured() is False
    auth.create_key("k", ["chat"])
    assert auth.any_key_configured() is True


def test_corrupt_keystore_fails_closed(auth):
    # SEC: a scoped-keys-only install (no owner key) must NOT silently drop to
    # open mode if auth.json gets corrupted/truncated. Before the fix this read
    # as "no keys" -> open. Now a present-but-unparseable keystore counts as
    # configured (fail CLOSED: every request needs a key, none verify -> locked).
    auth.create_key("k", ["chat"])
    auth.keystore_file().write_text("{ this is not valid json", encoding="utf-8")
    assert auth.any_key_configured() is True          # fail closed, not open
    assert auth.verify("anything") is None            # corrupt store grants nothing


def test_unreadable_keystore_fails_closed(auth):
    # A keystore path that exists but cannot be read as a file (here a directory,
    # which makes read_text raise OSError) also counts as configured (fail
    # closed), not open - distinct from the absent case.
    auth.keystore_file().mkdir(parents=True, exist_ok=True)
    assert auth.any_key_configured() is True


def test_empty_keystore_is_not_configured(auth):
    # A genuinely empty ([]) keystore is "no scoped keys" -> open is correct
    # (must stay distinct from the corrupt case above).
    auth.keystore_file().write_text("[]", encoding="utf-8")
    assert auth.any_key_configured() is False
    # A valid-JSON-but-malformed (non-list) keystore is treated as broken -> closed.
    auth.keystore_file().write_text('{"oops": "object not list"}', encoding="utf-8")
    assert auth.any_key_configured() is True


def test_unreadable_owner_key_fails_closed(auth):
    # SEC (checkup 2026-07-11 HIGH): an OWNER key file that EXISTS but cannot be
    # read (here a directory, which makes read_text raise OSError) must count as
    # auth-in-effect (fail CLOSED) - the server locks (every request needs a key,
    # none verify) instead of silently dropping to open/keyless mode. Mirrors
    # test_unreadable_keystore_fails_closed, for the owner key path.
    auth.key_file().mkdir(parents=True, exist_ok=True)
    assert auth.any_key_configured() is True          # fail closed, not open


def test_absent_owner_key_is_open(auth):
    # The genuinely ABSENT owner key (no file, no env, no keystore) stays open by
    # design - must remain DISTINCT from the present-but-unreadable case above.
    assert not auth.key_file().exists()
    assert auth.any_key_configured() is False


def test_transient_unreadable_owner_key_is_retried(auth, monkeypatch):
    # A TRANSIENT read failure on auth.key (a concurrent atomic replace / an
    # antivirus / an indexer holding it for a microsecond on Windows) must be
    # ridden out with a bounded retry, not read as "no key" for that request
    # (which would flap the owner's own auth), matching config._read_json.
    from pathlib import Path
    auth.set_api_key("s3cret-key")
    real_read = Path.read_text
    key_path = auth.key_file()
    seen = {"n": 0}

    def flaky(self, *a, **k):
        if self == key_path and seen["n"] == 0:
            seen["n"] += 1
            raise PermissionError("simulated transient sharing violation")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert auth.get_api_key() == "s3cret-key"          # rode out the transient
    assert seen["n"] == 1                               # proved it hit that path


def test_owner_key_is_admin(auth, monkeypatch):
    from localm import scopes as S
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    assert auth.verify("ownersecret") == {S.ADMIN}
    assert auth.verify("nope") is None


def test_ct_equal_is_total_and_correct(auth):
    """ct_equal is the house idiom for every secret compare, so its contract is
    'never raises, whatever the caller sends'. The vulnerable shape is a non-ASCII
    operand against an ASCII secret (a token_urlsafe or a hexdigest): compare_digest
    raises if EITHER side is non-ASCII, so the ASCII side does not protect it."""
    token = "aGVsbG8-d29ybGQ_1234"          # the shape secrets.token_urlsafe emits
    # Correct matches (the negative case: a blanket False would pass everything else).
    assert auth.ct_equal(token, token) is True
    assert auth.ct_equal("pässwort", "pässwort") is True
    assert auth.ct_equal("key\ud800bad", "key\ud800bad") is True
    # Wrong values reject.
    assert auth.ct_equal("wrong", token) is False
    assert auth.ct_equal("pässwort", "pässworT") is False
    # The bug shape: non-ASCII on either side must reject, not raise.
    assert auth.ct_equal("pässwort", token) is False
    assert auth.ct_equal(token, "pässwort") is False
    assert auth.ct_equal("key\ud800bad", token) is False
    # Absent operands are "no credential" / "no secret", never a match.
    for a, b in [(None, token), (token, None), ("", token), (token, ""), (None, None)]:
        assert auth.ct_equal(a, b) is False


def test_non_ascii_presented_token_rejects_and_never_raises(auth, monkeypatch):
    """hmac.compare_digest() raises TypeError on a non-ASCII str, and it raises if
    EITHER operand is non-ASCII - so an ASCII owner key does NOT protect the compare.
    A bearer token reaches verify() as a latin-1 decoded header, so any caller could
    turn a clean 401 into an unhandled 500 by sending a non-ASCII token."""
    from localm import scopes as S
    monkeypatch.setenv("LOCALM_API_KEY", "asciiownerkey1234")
    # Negative cases: rejection and acceptance must both still work on ASCII, or
    # this test would pass on a verify() that simply rejected everything.
    assert auth.verify("wrong-ascii-key") is None
    assert auth.verify("asciiownerkey1234") == {S.ADMIN}
    # The bug: a non-ASCII presented token must reject cleanly, not raise.
    assert auth.verify("pässwort") is None
    assert auth.verify("ünïcode-guess") is None


def test_non_ascii_owner_key_verifies_and_rejects_wrong(auth, monkeypatch):
    """A non-ASCII owner key (from the env var, or an auth.key written before
    set_api_key refused them) must still authenticate its owner and cleanly reject
    a wrong key. Before the fix BOTH answered 500, locking the owner out."""
    from localm import scopes as S
    monkeypatch.setenv("LOCALM_API_KEY", "pässwort-owner-key")
    assert auth.verify("pässwort-owner-key") == {S.ADMIN}
    # Negative cases: a wrong key must not ride in on the non-ASCII path.
    assert auth.verify("wrong-ascii-key") is None
    assert auth.verify("pässwort-owner-keX") is None


def test_lone_surrogate_token_rejects_and_never_raises(auth, monkeypatch):
    """os.environ can carry lone surrogates on Windows (surrogatepass decoding), and
    a plain .encode("utf-8") raises UnicodeEncodeError on one - which would swap the
    non-ASCII crash for a surrogate crash. The compare must be total."""
    monkeypatch.setenv("LOCALM_API_KEY", "asciiownerkey1234")
    assert auth.verify("key\ud800bad") is None


def test_set_api_key_refuses_characters_a_header_cannot_carry(auth):
    """A key rides in an HTTP Authorization header, so it is restricted to the
    generator's own alphabet at set time. Non-ASCII is the sharp case (clients send
    UTF-8, RFC 7230 obs-text decodes latin-1 -> mojibake -> the owner's own key is
    refused); spaces and control characters break the header outright."""
    for bad in ("pässwort-key", "key with spaces", "key\r\ninjected",
                "key!punctuation", "key\ud800surrogate"):
        with pytest.raises(ValueError, match="letters, numbers"):
            auth.set_api_key(bad)
        assert auth.get_api_key() is None              # nothing was persisted
    # Negative cases: every character the generator emits must still be accepted,
    # or `localm key generate` would reject its own output.
    auth.set_api_key("passwort-key")
    assert auth.get_api_key() == "passwort-key"
    auth.set_api_key("Under_scores-and-Digits123")
    assert auth.get_api_key() == "Under_scores-and-Digits123"


def test_set_api_key_accepts_every_generated_key(auth):
    """~49% of generate_key() outputs contain an underscore and ~48% a dash, so a
    charset that allowed only "-" would reject about half of them. regenerate_key()
    feeds generate_key() straight into set_api_key(), so a mismatch here would make
    `localm key generate` fail at random."""
    for _ in range(200):
        key = auth.generate_key()
        auth.set_api_key(key)                          # must never raise
        assert auth.get_api_key() == key


def test_non_ascii_bearer_token_gets_401_not_500(auth, monkeypatch):
    """End-to-end shape of the bug: an UNAUTHENTICATED caller sending a non-ASCII
    bearer token to a protected route must get a clean 401, never an unhandled 500.
    raise_server_exceptions=False so a server-side raise surfaces as a real 500
    response instead of propagating out of the client call (the house default of
    True would re-raise the TypeError and never yield a status to assert on)."""
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    auth.set_api_key("asciiownerkey1234")
    app = create_app(None)
    with TestClient(app, raise_server_exceptions=False) as client:
        # httpx refuses to encode a non-ASCII str header, so send raw bytes -
        # exactly what an attacker puts on the wire. Starlette decodes latin-1.
        r = client.get("/v1/models",
                       headers={"Authorization": b"Bearer p\xc3\xa4sswort"})
        assert r.status_code == 401, f"expected a clean 401, got {r.status_code}"
        # Negative cases: a wrong ASCII key must still 401, and the real owner
        # key must still 200, or a blanket-reject would pass this test.
        assert client.get("/v1/models", headers={
            "Authorization": "Bearer wrong-ascii-key"}).status_code == 401
        assert client.get("/v1/models", headers={
            "Authorization": "Bearer asciiownerkey1234"}).status_code == 200


def test_require_scope_enforcement(auth):
    from fastapi import HTTPException
    from localm import scopes as S
    from localm.inference.http_server import require_scope, _require_auth

    # open mode: no keys configured -> allowed
    assert require_scope(S.PLUGINS_ADMIN)(_req()) is None

    # a named key WITHOUT plugins:admin
    made = auth.create_key("reader", [S.CHAT])
    cred = _req(made["key"])

    # now a key exists -> enforced; this key lacks plugins:admin -> 403
    with pytest.raises(HTTPException) as exc:
        require_scope(S.PLUGINS_ADMIN)(cred)
    assert exc.value.status_code == 403

    # it does satisfy a chat-scoped requirement, and any-valid-key auth
    assert require_scope(S.CHAT)(cred) is None
    assert _require_auth(cred) is None

    # a wrong/unknown key -> 401
    with pytest.raises(HTTPException) as exc:
        require_scope(S.CHAT)(_req("wrong"))
    assert exc.value.status_code == 401


def test_owner_key_grants_every_scope(auth, monkeypatch):
    from localm import scopes as S
    from localm.inference.http_server import require_scope
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    cred = _req("ownersecret")
    assert require_scope(S.PLUGINS_ADMIN)(cred) is None
    assert require_scope(S.KEYS_ADMIN)(cred) is None


def test_require_owner_dependency_rejects_non_owner(auth, monkeypatch):
    """require_owner() (design-audit LM-DA-020, reaffirming LM-DA-SEC-06):
    job_owner_ok's per-route ownership check is now Depends()-injectable, the
    same pattern require_scope already uses, so a new per-owner route cannot
    omit it by construction. Exercises a route wired via
    Depends(require_owner(...)) through a real TestClient request - mirroring
    the existing job_owner_ok route-level coverage in
    test_jobs_owner_binding.py / test_media_gallery_ownership.py - rather than
    unit-calling the dependency directly, since require_owner's gate composes
    with a nested path-param-reading resolve() dependency that only a real
    request can drive end to end."""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import principal_id, require_owner

    owner_key = auth.create_key("alice", [S.CHAT])["key"]
    other_key = auth.create_key("bob", [S.CHAT])["key"]
    owner_id = principal_id(_req(owner_key))
    things = {"t1": owner_id}

    def _resolve(thing_id: str):
        return (thing_id if thing_id in things else None,
                things.get(thing_id), f"No such thing: {thing_id}")

    app = FastAPI()

    @app.get("/things/{thing_id}")
    def get_thing(thing: str = Depends(require_owner(_resolve))):
        return {"thing": thing}

    def _h(key):
        return {"Authorization": f"Bearer {key}"}

    with TestClient(app) as c:
        # the owner reaches its own thing
        assert c.get("/things/t1", headers=_h(owner_key)).status_code == 200
        # a different valid key gets the SAME 404 a missing id would (never 403,
        # so a foreign key cannot even confirm the thing exists - KEY-SCOPE-2)
        assert c.get("/things/t1", headers=_h(other_key)).status_code == 404
        assert c.get("/things/nope", headers=_h(owner_key)).status_code == 404
        # the owner/admin key reaches every principal's things
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        assert c.get("/things/t1", headers=_h("ownersecret")).status_code == 200


# --------------------------------------------------------------------------- #
#  Privilege self-escalation: a keys:admin key must not mint privileged keys  #
# --------------------------------------------------------------------------- #

def test_create_key_blocks_privileged_scopes_by_default(auth):
    from localm import scopes as S
    for priv in (S.ADMIN, S.KEYS_ADMIN, S.PLUGINS_ADMIN, S.CONFIG_WRITE):
        with pytest.raises(PermissionError):
            auth.create_key("evil", [S.CHAT, priv])
    assert auth.list_keys() == []          # nothing was persisted


def test_create_key_allows_privileged_for_owner(auth):
    from localm import scopes as S
    made = auth.create_key("admin-key", [S.ADMIN], allow_privileged=True)
    assert S.ADMIN in made["scopes"]
    assert auth.verify(made["key"]) == {S.ADMIN}


def test_keys_endpoint_blocks_privilege_self_escalation(auth, monkeypatch):
    """POST /v1/keys: a non-owner key holding only keys:admin can mint ordinary
    keys but is refused (403) when it tries to grant itself privileged scopes;
    the owner key can."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import create_app

    # The owner mints a keys:admin-only key (the attacker principal).
    attacker = auth.create_key("ka", [S.KEYS_ADMIN], allow_privileged=True)
    app = create_app(None)
    with TestClient(app) as client:
        hdr = {"Authorization": f"Bearer {attacker['key']}"}
        # keys:admin -> mint an ADMIN key: refused.
        esc = client.post("/v1/keys", json={"name": "pwn", "scopes": [S.ADMIN]},
                          headers=hdr)
        assert esc.status_code == 403
        # keys:admin -> mint an ordinary (non-privileged) key: allowed.
        ok = client.post("/v1/keys", json={"name": "reader", "scopes": [S.CHAT]},
                         headers=hdr)
        assert ok.status_code == 200
        # the owner key CAN mint an ADMIN key.
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        owner = client.post("/v1/keys", json={"name": "adm", "scopes": [S.ADMIN]},
                            headers={"Authorization": "Bearer ownersecret"})
        assert owner.status_code == 200
        assert S.ADMIN in owner.json()["scopes"]


def test_keys_endpoint_expires_in_is_server_clock(auth, monkeypatch):
    """POST /v1/keys with expires_in (relative seconds) sets the deadline from the
    SERVER clock (not the client's), verify() honours it, and a bad expires_in 400s."""
    import time

    from fastapi.testclient import TestClient

    from localm import scopes as S
    from localm.inference.http_server import create_app

    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    app = create_app(None)
    with TestClient(app) as client:
        own = {"Authorization": "Bearer ownersecret"}
        before = time.time()
        r = client.post(
            "/v1/keys",
            json={"name": "phone", "scopes": [S.CHAT], "expires_in": 3600},
            headers=own)
        assert r.status_code == 200
        exp = r.json()["expires"]
        assert before + 3600 - 5 <= exp <= time.time() + 3600 + 5   # server-anchored
        assert auth.verify(r.json()["key"]) == {S.CHAT}             # not yet expired
        bad = client.post(
            "/v1/keys",
            json={"name": "x", "scopes": [S.CHAT], "expires_in": "soon"},
            headers=own)
        assert bad.status_code == 400


def test_keys_endpoint_returns_owner_flag_and_presets(auth, monkeypatch):
    """GET /v1/keys carries is_owner (so the GUI hides owner-only scopes from a mere
    keys:admin device) and key_presets (so presets show even without config:read)."""
    from fastapi.testclient import TestClient

    from localm import scopes as S
    from localm.inference.http_server import create_app

    ka = auth.create_key("manager", [S.KEYS_ADMIN], allow_privileged=True)
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    app = create_app(None)
    with TestClient(app) as client:
        owner = client.get(
            "/v1/keys", headers={"Authorization": "Bearer ownersecret"}).json()
        assert owner["is_owner"] is True
        assert any(p["name"] == "Companion" for p in owner["presets"])
        # A non-owner keys:admin key lists keys but is NOT owner; presets still ride.
        km = client.get(
            "/v1/keys", headers={"Authorization": f"Bearer {ka['key']}"}).json()
        assert km["is_owner"] is False
        assert isinstance(km["presets"], list)


def test_model_read_routes_require_models_read_scope(auth, monkeypatch):
    """SECURITY.md promises every /v1 route is auth-gated when a key is set.
    /v1/models and /v1/models/{id} must require models:read; /health stays open."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import create_app

    fake = {"m1": {"path": "m1.gguf", "source": "local", "sha256": "abc"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: fake)

    # A key exists -> auth enforced. This key lacks models:read.
    weak = auth.create_key("weak", [S.CHAT])
    app = create_app(None)
    with TestClient(app) as client:
        # No credentials -> 401 on both model-read routes.
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models/m1").status_code == 401
        # A key without models:read -> 403.
        weak_hdr = {"Authorization": f"Bearer {weak['key']}"}
        assert client.get("/v1/models", headers=weak_hdr).status_code == 403
        assert client.get("/v1/models/m1", headers=weak_hdr).status_code == 403
        # A key WITH models:read -> 200.
        reader = auth.create_key("reader", [S.MODELS_READ])
        rdr_hdr = {"Authorization": f"Bearer {reader['key']}"}
        assert client.get("/v1/models", headers=rdr_hdr).status_code == 200
        assert client.get("/v1/models/m1", headers=rdr_hdr).status_code == 200
        # The owner key (implicit ADMIN) -> 200.
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        own_hdr = {"Authorization": "Bearer ownersecret"}
        assert client.get("/v1/models", headers=own_hdr).status_code == 200
        # /health is an unauthenticated liveness probe even with a key set.
        assert client.get("/health").status_code in (200, 503)


def test_model_detail_does_not_leak_absolute_path(auth, monkeypatch):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app
    fake = {"m1": {"path": r"C:\Users\Secret\models\m1.gguf",
                   "source": "local", "sha256": "abc"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: fake)
    app = create_app(None)
    with TestClient(app) as client:
        r = client.get("/v1/models/m1")
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == "m1.gguf"          # basename only
        assert "Secret" not in body["path"]


def test_first_key_from_loopback_gui_seeds_owner_session(auth):
    """S3 lockout guard: minting the FIRST key from the loopback GUI in open mode
    must NOT lock the owner out. It seeds a persistent owner key and sets the
    session cookies, so the browser stays authenticated (as owner) and a
    follow-up request via the cookie - now that auth is on - is authorized."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import SESSION_COOKIE, create_app

    assert not auth.any_key_configured()          # starts open/keyless
    app = create_app(None)                         # bind_host unset -> loopback default
    with TestClient(app) as client:
        shell = app.state.shell_token
        # open-mode management (minting the first key) needs the loopback shell token
        r = client.post("/v1/keys", json={"name": "phone", "scopes": [S.CHAT]},
                        headers={"Authorization": f"Bearer {shell}"})
        assert r.status_code == 200
        assert r.json()["scopes"] == [S.CHAT]      # the phone key stays scoped
        # a persistent OWNER key was seeded so the local owner keeps full access
        owner = auth.get_api_key()
        assert owner is not None and auth.verify(owner) == {S.ADMIN}
        # the response established the browser session (opaque session cookie; CSRF
        # is derived from the session, not a cookie)
        set_cookies = " ".join(r.headers.get_list("set-cookie"))
        assert SESSION_COOKIE in set_cookies
        # auth is now on; the follow-up rides the session cookie from the jar
        # (NOT the now-dead shell token) and is authorized -> no lockout.
        assert auth.any_key_configured()
        assert client.get("/v1/keys").status_code == 200


def test_second_key_does_not_reseed_owner_or_session(auth, monkeypatch):
    """The S3 guard fires ONLY on the open->protected transition: with a key
    already configured, creating another key seeds no owner key and sets no
    session cookie (was_open is False)."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import create_app

    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # already protected
    app = create_app(None)
    with TestClient(app) as client:
        r = client.post("/v1/keys", json={"name": "phone", "scopes": [S.CHAT]},
                        headers={"Authorization": "Bearer ownersecret"})
        assert r.status_code == 200
        assert not r.headers.get_list("set-cookie")   # no session seeded
        # the env owner key is untouched; no auth.key was written
        assert auth._read_key_file() is None
