# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the shared API-key auth (localm/auth.py) and its enforcement in
the HTTP server's _require_auth dependency."""

import logging

import pytest
from fastapi import HTTPException


def _req(token=None, method="GET"):
    """Minimal Starlette Request carrying an optional Bearer token, for unit-
    calling the request-aware auth dependencies (header-or-cookie auth)."""
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
    # config.py freezes these paths at import, so LOCALM_HOME alone does not
    # redirect load_config/save_config. Point them at the throwaway dir.
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
    # An unreadable config fails safe to "required" and surfaces a warning,
    # rather than downgrading an explicit require_auth: true to "not required".
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth.require_auth_enabled() is True
    assert any("require_auth" in r.message and "required" in r.message.lower()
               for r in caplog.records), \
        "config-read failure resolved silently (no warning)"


def test_config_read_failure_env_var_short_circuits(auth, monkeypatch):
    # LOCALM_REQUIRE_AUTH is checked before config is read, so a truthy env var
    # returns True even with an unreadable config file.
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    assert auth.require_auth_enabled() is True


def test_require_auth_dependency(auth, monkeypatch):
    from localm.inference.http_server import _require_auth

    # open mode: no key, not required -> allowed
    assert _require_auth(_req()) is None

    # required but no key configured -> fail closed (401)
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
#  Scoped keystore + scope enforcement                                        #
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
    # A present-but-unparseable keystore counts as configured (fail closed:
    # every request needs a key, and none verify).
    auth.create_key("k", ["chat"])
    auth.keystore_file().write_text("{ this is not valid json", encoding="utf-8")
    assert auth.any_key_configured() is True          # fail closed, not open
    assert auth.verify("anything") is None            # corrupt store grants nothing


def test_unreadable_keystore_fails_closed(auth):
    # A keystore path that exists but cannot be read as a file (here a directory,
    # so read_text raises OSError) also counts as configured (fail closed),
    # distinct from the absent case.
    auth.keystore_file().mkdir(parents=True, exist_ok=True)
    assert auth.any_key_configured() is True


def test_empty_keystore_is_not_configured(auth):
    # A genuinely empty ([]) keystore is "no scoped keys" -> open.
    auth.keystore_file().write_text("[]", encoding="utf-8")
    assert auth.any_key_configured() is False
    # A valid-JSON-but-malformed (non-list) keystore is treated as broken -> closed.
    auth.keystore_file().write_text('{"oops": "object not list"}', encoding="utf-8")
    assert auth.any_key_configured() is True


def test_unreadable_owner_key_fails_closed(auth):
    # An owner key file that exists but cannot be read (here a directory, so
    # read_text raises OSError) counts as auth-in-effect (fail closed): the
    # server locks instead of dropping to open/keyless mode.
    auth.key_file().mkdir(parents=True, exist_ok=True)
    assert auth.any_key_configured() is True          # fail closed, not open


def test_absent_owner_key_is_open(auth):
    # A genuinely absent owner key (no file, no env, no keystore) stays open.
    assert not auth.key_file().exists()
    assert auth.any_key_configured() is False


def test_transient_unreadable_owner_key_is_retried(auth, monkeypatch):
    # A transient read failure on auth.key is ridden out with a bounded retry
    # rather than read as "no key" for that request.
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
    # Correct matches.
    assert auth.ct_equal(token, token) is True
    assert auth.ct_equal("pässwort", "pässwort") is True
    assert auth.ct_equal("key\ud800bad", "key\ud800bad") is True
    # Wrong values reject.
    assert auth.ct_equal("wrong", token) is False
    assert auth.ct_equal("pässwort", "pässworT") is False
    # Non-ASCII on either side rejects rather than raising.
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
    # Negative cases: rejection and acceptance both still work on ASCII.
    assert auth.verify("wrong-ascii-key") is None
    assert auth.verify("asciiownerkey1234") == {S.ADMIN}
    # A non-ASCII presented token rejects cleanly rather than raising.
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
    # Negative cases: every character the generator emits is accepted.
    auth.set_api_key("passwort-key")
    assert auth.get_api_key() == "passwort-key"
    auth.set_api_key("Under_scores-and-Digits123")
    assert auth.get_api_key() == "Under_scores-and-Digits123"


def test_set_api_key_accepts_every_generated_key(auth):
    """~49% of generate_key() outputs contain an underscore and ~48% a dash, so a
    charset that allowed only "-" would reject about half of them. regenerate_key()
    feeds generate_key() straight into set_api_key(), so a mismatch here would make
    `localm key generate` fail at random.

    A direct check on the charset itself covers THAT property without relying
    on chance. The loop below does real set_api_key/get_api_key round trips,
    which exercises the length/charset guards end to end; 30 samples puts the
    odds of missing either character below 1e-9 (0.51**30). The count stays
    small because each call pays a real memory-hard KDF derivation via the
    owner-KDF path (see _OWNER_KDF_KEEP in auth.py)."""
    assert auth._KEY_CHARSET.match("-")
    assert auth._KEY_CHARSET.match("_")
    for _ in range(30):
        key = auth.generate_key()
        auth.set_api_key(key)                          # must never raise
        assert auth.get_api_key() == key


def test_non_ascii_bearer_token_gets_401_not_500(auth, monkeypatch):
    """An UNAUTHENTICATED caller sending a non-ASCII bearer token to a protected
    route must get a clean 401, never an unhandled 500.
    raise_server_exceptions=False so a server-side raise surfaces as a real 500
    response instead of propagating out of the client call (the house default of
    True would re-raise the TypeError and never yield a status to assert on)."""
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app

    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    auth.set_api_key("asciiownerkey1234")
    app = create_app(None)
    with TestClient(app, raise_server_exceptions=False) as client:
        # httpx refuses to encode a non-ASCII str header, so send raw bytes.
        # Starlette decodes latin-1.
        r = client.get("/v1/models",
                       headers={"Authorization": b"Bearer p\xc3\xa4sswort"})
        assert r.status_code == 401, f"expected a clean 401, got {r.status_code}"
        # Negative cases: a wrong ASCII key 401s and the real owner key 200s.
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
    """require_owner() makes job_owner_ok's per-route ownership check
    Depends()-injectable, the same pattern require_scope uses. Exercises a route
    wired via Depends(require_owner(...)) through a real TestClient request
    rather than unit-calling the dependency, since require_owner's gate composes
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
        # a different valid key gets the SAME 404 a missing id would, never 403,
        # so a foreign key cannot confirm the thing exists
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


def test_keys_endpoint_wires_fs_access_owner_only(auth, monkeypatch):
    """POST /v1/keys forwards fs_access from the request body into create_key().
    Granting host reach follows the same owner-only gate as a privileged scope:
    a non-owner keys:admin caller is refused
    (403) and nothing is persisted, while the owner key succeeds and the minted
    key actually carries fs_access='host'."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import create_app

    manager = auth.create_key("mgr", [S.KEYS_ADMIN], allow_privileged=True)
    app = create_app(None)
    with TestClient(app) as client:
        hdr = {"Authorization": f"Bearer {manager['key']}"}
        # non-owner requesting fs_access=host: refused, nothing minted.
        before = len(auth.list_keys())
        refused = client.post(
            "/v1/keys",
            json={"name": "pwn", "scopes": [S.CHAT], "fs_access": "host"},
            headers=hdr)
        assert refused.status_code == 403
        assert len(auth.list_keys()) == before   # nothing persisted on refusal
        # non-owner omitting fs_access still works and gets the safe default.
        ok = client.post(
            "/v1/keys", json={"name": "reader", "scopes": [S.CHAT]}, headers=hdr)
        assert ok.status_code == 200
        assert ok.json()["fs_access"] == "none"
        # the owner CAN grant fs_access=host, and it is actually stored (not
        # silently dropped to "none").
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        granted = client.post(
            "/v1/keys",
            json={"name": "device", "scopes": [S.CHAT], "fs_access": "host"},
            headers={"Authorization": "Bearer ownersecret"})
        assert granted.status_code == 200
        assert granted.json()["fs_access"] == "host"
        stored = [k for k in auth.list_keys() if k["name"] == "device"][0]
        assert stored["fs_access"] == "host"


def test_keys_endpoint_wires_rag_roots_owner_only(auth, monkeypatch):
    """POST /v1/keys forwards rag_roots from the request body into create_key().
    A key-scoped rag_roots list REPLACES the whitelist
    rather than narrowing it (rag/store.py's confine_index_path), so it can point
    a new key at a folder outside the caller's own reach - granting one follows
    the same owner-only gate as fs_access=host: a non-owner keys:admin caller is
    refused (403) and nothing is persisted, while the owner key succeeds and the
    minted key actually carries the roots."""
    from fastapi.testclient import TestClient
    from localm import scopes as S
    from localm.inference.http_server import create_app

    manager = auth.create_key("mgr", [S.KEYS_ADMIN], allow_privileged=True)
    app = create_app(None)
    with TestClient(app) as client:
        hdr = {"Authorization": f"Bearer {manager['key']}"}
        # non-owner requesting a rag_roots confinement: refused, nothing minted.
        before = len(auth.list_keys())
        refused = client.post(
            "/v1/keys",
            json={"name": "pwn", "scopes": [S.CHAT], "rag_roots": ["C:/secret"]},
            headers=hdr)
        assert refused.status_code == 403
        assert len(auth.list_keys()) == before   # nothing persisted on refusal
        # non-owner omitting rag_roots still works and gets the safe default.
        ok = client.post(
            "/v1/keys", json={"name": "reader", "scopes": [S.CHAT]}, headers=hdr)
        assert ok.status_code == 200
        assert ok.json()["rag_roots"] == []
        # a malformed (non-list) rag_roots 400s rather than being silently coerced.
        bad = client.post(
            "/v1/keys",
            json={"name": "x", "scopes": [S.CHAT], "rag_roots": "C:/not-a-list"},
            headers=hdr)
        assert bad.status_code == 400
        # the owner CAN grant rag_roots, and it is actually stored (not silently
        # dropped to unrestricted).
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        granted = client.post(
            "/v1/keys",
            json={"name": "device", "scopes": [S.CHAT],
                  "rag_roots": ["C:/docs/a", "C:/docs/b"]},
            headers={"Authorization": "Bearer ownersecret"})
        assert granted.status_code == 200
        assert granted.json()["rag_roots"] == ["C:/docs/a", "C:/docs/b"]
        stored = [k for k in auth.list_keys() if k["name"] == "device"][0]
        assert stored["rag_roots"] == ["C:/docs/a", "C:/docs/b"]


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
    fake = {"m1": {"path": r"Z:\Users\Secret\models\m1.gguf",
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


# --------------------------------------------------------------------------- #
#  resolve_bearer_token / resolve_bearer_headers precedence: the owner key wins #
#  over an instance token wherever a self-call resolves its own credential.    #
# --------------------------------------------------------------------------- #

def test_resolve_bearer_token_prefers_owner_key_over_instance_token(auth):
    auth.set_api_key("owner-secret-key")
    assert auth.resolve_bearer_token("inst-token-abc") == "owner-secret-key"


def test_resolve_bearer_token_falls_back_to_instance_token_when_open(auth):
    assert auth.get_api_key() is None      # open mode
    assert auth.resolve_bearer_token("inst-token-abc") == "inst-token-abc"


def test_resolve_bearer_token_none_when_neither_available(auth):
    assert auth.get_api_key() is None
    assert auth.resolve_bearer_token(None) is None
    assert auth.resolve_bearer_token("") is None


def test_resolve_bearer_token_env_key_also_wins(auth, monkeypatch):
    auth.set_api_key("file-key")
    monkeypatch.setenv("LOCALM_API_KEY", "env-key")
    assert auth.resolve_bearer_token("inst-token-abc") == "env-key"


def test_resolve_bearer_headers_matches_resolve_bearer_token(auth):
    """resolve_bearer_headers is a thin wrapper - same precedence, dict shape."""
    auth.set_api_key("owner-secret-key")
    assert auth.resolve_bearer_headers("inst-token-abc") == {
        "Authorization": "Bearer owner-secret-key"}
    auth.clear_api_key()
    assert auth.resolve_bearer_headers("inst-token-abc") == {
        "Authorization": "Bearer inst-token-abc"}
    assert auth.resolve_bearer_headers(None) == {}


# --------------------------------------------------------------------------- #
#  Per-key rag_roots: a per-key field defaulting to "no restriction", which the #
#  owner/ADMIN key is exempt from regardless of what is stored on it.          #
# --------------------------------------------------------------------------- #

def test_norm_rag_roots_coerces_and_dedupes(auth):
    assert auth.norm_rag_roots(None) == []
    assert auth.norm_rag_roots([]) == []
    assert auth.norm_rag_roots("not-a-list") == []       # a bare str is not a list
    assert auth.norm_rag_roots([123, None, "", "   "]) == []   # junk entries dropped
    assert auth.norm_rag_roots(["D:/docs", "D:/docs", " D:/other "]) == \
        ["D:/docs", "D:/other"]              # de-duped, stripped, order preserved


def test_create_key_rag_roots_defaults_to_empty_and_round_trips(auth):
    from localm import scopes as S
    unrestricted = auth.create_key("n", [S.CHAT])
    scoped = auth.create_key("s", [S.CHAT], rag_roots=["D:/shared/docs", "D:/other"])
    assert unrestricted["rag_roots"] == []               # safe default: no restriction
    assert scoped["rag_roots"] == ["D:/shared/docs", "D:/other"]
    by_name = {k["name"]: k for k in auth.list_keys()}
    assert by_name["n"]["rag_roots"] == []
    assert by_name["s"]["rag_roots"] == ["D:/shared/docs", "D:/other"]


def test_rag_roots_for_unknown_or_missing_token_is_default(auth):
    assert auth.rag_roots_for("") == []
    assert auth.rag_roots_for(None) == []
    assert auth.rag_roots_for("not-a-real-key") == []
    assert auth.rag_roots_for("not-a-real-key", default=["fallback"]) == ["fallback"]


def test_rag_roots_for_reads_the_stored_list(auth):
    made = auth.create_key("s", ["chat"], rag_roots=["D:/shared"])
    assert auth.rag_roots_for(made["key"]) == ["D:/shared"]
    plain = auth.create_key("p", ["chat"])
    assert auth.rag_roots_for(plain["key"]) == []        # legacy/unset -> unrestricted


def test_rag_roots_for_revoked_key_falls_back_to_default(auth):
    made = auth.create_key("s", ["chat"], rag_roots=["D:/shared"])
    auth.revoke_key(made["id"])
    assert auth.rag_roots_for(made["key"]) == []          # gone -> the safe default
