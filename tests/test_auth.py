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
    monkeypatch.setattr("localm.config.load_config_checked", boom)
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
    monkeypatch.setattr("localm.config.load_config_checked", boom)
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


def test_keys_endpoint_surfaces_owner_key_downgrade_warning(auth, monkeypatch):
    """GET /v1/keys carries owner_key_warning when the owner-key file has been
    downgraded (present but empty). Not owner-gated: the downgrade puts the
    server in OPEN mode, where the caller authenticates with the shell token
    rather than an owner key - an is_owner gate would hide the warning in
    exactly the case it matters."""
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    monkeypatch.setattr(auth, "_empty_owner_key_warned", False, raising=False)
    monkeypatch.setattr(auth, "_empty_owner_key_downgrade_message", None, raising=False)
    auth.key_file().parent.mkdir(parents=True, exist_ok=True)
    auth.key_file().write_text("", encoding="utf-8")

    app = create_app(None)
    with TestClient(app) as client:
        # Open mode's management gate refuses a no-Origin caller without the
        # shell token the GUI shell injects (test_open_mode_management_gate.py).
        headers = {"Authorization": f"Bearer {app.state.shell_token}"}
        data = client.get("/v1/keys", headers=headers).json()
        assert data["owner_key_warning"] and "auth.key" in data["owner_key_warning"]
        assert auth.owner_key_downgrade_warning() == data["owner_key_warning"]


def test_keys_endpoint_reports_no_owner_key_warning_when_keyed(auth, monkeypatch):
    """NEGATIVE CASE: a real owner key must not manufacture a warning."""
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    monkeypatch.setattr(auth, "_empty_owner_key_warned", False, raising=False)
    monkeypatch.setattr(auth, "_empty_owner_key_downgrade_message", None, raising=False)
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")

    app = create_app(None)
    with TestClient(app) as client:
        data = client.get(
            "/v1/keys", headers={"Authorization": "Bearer ownersecret"}).json()
        assert not data["owner_key_warning"]


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


# --------------------------------------------------------------------------- #
#  Digests, keystore records and their logging                                #
# --------------------------------------------------------------------------- #

class _TimeProxy:
    """Stands in for ``auth.time``: fixed ``time()``/``monotonic()`` when set,
    recorded ``sleep()`` calls, everything else from the real module."""

    def __init__(self, now=None, mono=None):
        self.now = now
        self.mono = mono
        self.sleeps = []

    def time(self):
        import time
        return time.time() if self.now is None else self.now

    def monotonic(self):
        import time
        return time.monotonic() if self.mono is None else self.mono

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def __getattr__(self, name):
        import time
        return getattr(time, name)


def _logged(caplog, needle):
    return [r for r in caplog.records if needle.lower() in str(r.msg).lower()]


def _sha256_hex(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_keystore(auth, records):
    import json
    path = auth.keystore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def _set_last_used(auth, key_id, value):
    import json
    path = auth.keystore_file()
    records = json.loads(path.read_text(encoding="utf-8"))
    for r in records:
        if r["id"] == key_id:
            r["last_used"] = value
    path.write_text(json.dumps(records), encoding="utf-8")


def _last_used(auth, key_id):
    return next(k["last_used"] for k in auth.list_keys() if k["id"] == key_id)


def test_credential_file_names(auth):
    assert auth.key_file().name == "auth.key"
    assert auth.keystore_file().name == "auth.json"
    assert auth.owner_kdf_file().name == "auth.kdf.json"


def test_generate_key_length_follows_nbytes(auth):
    assert len(auth.generate_key()) == 43
    assert len(auth.generate_key(16)) == 22


def test_fast_digest_accepts_a_lone_surrogate(auth):
    import hashlib
    key = "ab\ud800cd"
    assert auth._fast_digest(key) == hashlib.sha256(
        key.encode("utf-8", "surrogatepass")).hexdigest()


def test_scrypt_derive_honours_dklen_surrogates_and_its_memory_budget(auth):
    import hashlib
    salt = b"s" * 16
    small = auth._scrypt_derive("k", salt, 16, 8, 1, 32)
    assert small == hashlib.scrypt(b"k", salt=salt, n=16, r=8, p=1, dklen=32).hex()
    assert len(small) == 64
    odd = "a\udc80"
    assert auth._scrypt_derive(odd, salt, 16, 8, 1, 32) == hashlib.scrypt(
        odd.encode("utf-8", "surrogatepass"), salt=salt, n=16, r=8, p=1,
        dklen=32).hex()
    big = auth._scrypt_derive("k", salt, 2 ** 15, 8, 1, 32)
    assert big == hashlib.scrypt(b"k", salt=salt, n=2 ** 15, r=8, p=1, dklen=32,
                                 maxmem=64 * 1024 * 1024).hex()


def test_atomic_write_private_passes_retrying_through(auth, monkeypatch, tmp_path):
    import localm.config as cfg
    seen = []
    monkeypatch.setattr(cfg, "atomic_write_private",
                        lambda path, text, **kw: seen.append(kw))
    auth._atomic_write_private(tmp_path / "f", "x")
    auth._atomic_write_private(tmp_path / "f", "x", retrying=True)
    assert seen == [{"retrying": False}, {"retrying": True}]


def test_keystore_is_saved_indented_through_the_retrying_writer(auth, monkeypatch):
    seen = []
    monkeypatch.setattr(auth, "_atomic_write_private",
                        lambda path, text, **kw: seen.append((path, text, kw)))
    auth._save_keystore([{"id": "a"}])
    assert seen == [(auth.keystore_file(), '[\n  {\n    "id": "a"\n  }\n]',
                     {"retrying": True})]


def test_credential_stores_are_created_under_a_nested_new_home(auth, monkeypatch, tmp_path):
    import json
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "a" / "b"))
    auth._save_keystore([])
    assert json.loads(auth.keystore_file().read_text(encoding="utf-8")) == []
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "c" / "d"))
    auth._save_owner_kdf([{"x": 1}])
    assert auth.owner_kdf_file().read_text(encoding="utf-8") == (
        '{\n  "v": 1,\n  "records": [\n    {\n      "x": 1\n    }\n  ]\n}')


def test_owner_digest_takes_its_lock_in_an_existing_or_nested_home(
        auth, monkeypatch, tmp_path, caplog):
    for home in (tmp_path, tmp_path / "x" / "y"):
        monkeypatch.setenv("LOCALM_HOME", str(home))
        auth._forget_cached_digests()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="localm"):
            digest = auth._owner_digest("owner-key-123")
        assert len(digest) == 64
        assert _logged(caplog, "could not lock") == []
        assert auth.owner_kdf_file().is_file()


def test_owner_digest_record_mints_once_then_reuses_it(auth):
    import json
    digest, minted = auth._owner_digest_record("owner-key-123")
    [rec] = json.loads(auth.owner_kdf_file().read_text(encoding="utf-8"))["records"]
    assert minted is True
    assert digest == rec["digest"] and len(digest) == 64
    assert len(bytes.fromhex(rec["salt"])) == 16
    assert isinstance(rec["created"], float)
    assert auth._owner_digest_record("owner-key-123") == (digest, False)


def test_an_unpersistable_owner_kdf_record_is_reported(auth, monkeypatch, caplog):
    def refuse(records):
        raise OSError("disk full")
    monkeypatch.setattr(auth, "_save_owner_kdf", refuse)
    with caplog.at_level(logging.WARNING, logger="localm"):
        digest, minted = auth._owner_digest_record("owner-key-123")
    assert minted is True and len(digest) == 64
    [r] = _logged(caplog, "owner key derivation record")
    assert r.msg == ("could not persist the owner key derivation record %s (%s); "
                     "sessions and jobs stamped in this process may not be "
                     "recognised after a restart")
    assert r.args[0] == auth.owner_kdf_file()
    assert str(r.args[1]) == "disk full"


def test_a_corrupt_owner_kdf_file_is_reported_and_ignored(auth, caplog):
    path = auth.owner_kdf_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth._load_owner_kdf() == []
    [r] = _logged(caplog, "is unreadable or corrupt")
    assert r.msg == ("owner key KDF file %s is unreadable or corrupt (%s); "
                     "minting a fresh derivation record. Sessions and jobs "
                     "owned by the owner key are re-linked automatically on "
                     "the next successful verify")
    assert r.args[0] == path
    assert isinstance(r.args[1], ValueError)


def test_legacy_identity_relink_needs_two_distinct_digests(auth, monkeypatch):
    from localm import sessions
    calls = []
    monkeypatch.setattr(sessions, "relink_key_hash",
                        lambda old, new: calls.append((old, new)) or 0)
    for legacy, current in (("", "new"), ("old", ""), ("", ""), ("same", "same")):
        auth._migrate_legacy_owner_identity(legacy, current)
    assert calls == []
    auth._migrate_legacy_owner_identity("old", "new")
    assert calls == [("old", "new")]


def test_legacy_identity_relink_is_logged(auth, monkeypatch, caplog):
    from localm import sessions
    monkeypatch.setattr(sessions, "relink_key_hash", lambda old, new: 3)
    with caplog.at_level(logging.DEBUG, logger="localm"):
        auth._migrate_legacy_owner_identity("old", "new")
    [r] = _logged(caplog, "re-linked")
    assert r.msg == "re-linked %d session(s) to the owner key's derived identity"
    assert r.args == (3,)


def test_a_failed_legacy_identity_relink_is_reported(auth, monkeypatch, caplog):
    from localm import sessions

    def boom(old, new):
        raise RuntimeError("store locked")
    monkeypatch.setattr(sessions, "relink_key_hash", boom)
    with caplog.at_level(logging.WARNING, logger="localm"):
        auth._migrate_legacy_owner_identity("old", "new")
    [r] = _logged(caplog, "could not re-link")
    assert r.msg == ("could not re-link existing sessions to the owner key's "
                     "new derived identity (%s); those sessions stay valid, "
                     "but a job created from one may not be recognised as the "
                     "owner's until the next sign-in")
    assert [str(a) for a in r.args] == ["store locked"]


def test_owner_kdf_record_without_a_salt_uses_an_empty_salt(auth):
    key = "owner-key-123"
    rec = {"alg": "scrypt", "n": 16, "r": 8, "p": 1, "dklen": 32,
           "digest": auth._scrypt_derive(key, b"", 16, 8, 1, 32)}
    assert auth._owner_kdf_record_for(key, [rec]) is rec


def test_record_digest_for_derives_a_declared_kdf_record(auth):
    salt = b"\x01" * 16
    rec = {"alg": "scrypt", "salt": salt.hex(), "n": 16, "r": 8, "p": 1, "dklen": 32}
    assert auth._record_digest_for(rec, "k", lambda: "fast") == \
        auth._scrypt_derive("k", salt, 16, 8, 1, 32)


def test_record_with_an_unknown_alg_is_refused_and_reported(auth, caplog):
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth._record_digest_for({"id": "k1", "alg": "argon9"}, "k",
                                       lambda: "fast") is None
    [r] = _logged(caplog, "unknown hash algorithm")
    assert r.msg == ("keystore record %s declares an unknown hash algorithm %r; "
                     "refusing to match it. A key written by a NEWER localm cannot "
                     "be verified by this one - upgrade rather than editing the "
                     "keystore by hand")
    assert r.args == ("k1", "argon9")


def test_keystore_lookup_skips_an_unusable_record_and_matches_a_kdf_one(auth):
    key = "tok"
    salt = b"\x02" * 16
    kdf = {"id": "b", "alg": "scrypt", "salt": salt.hex(), "n": 16, "r": 8,
           "p": 1, "dklen": 32, "hash": auth._scrypt_derive(key, salt, 16, 8, 1, 32)}
    unknown = {"id": "a", "alg": "argon9", "hash": "x"}
    assert auth._find_keystore_record(key, [unknown, kdf]) is kdf


def test_a_memoised_owner_digest_is_served_without_rechecking_the_owner(auth, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-123")
    auth._forget_cached_digests()
    first = auth._hash_key("owner-key-123")
    checks = []
    real = auth._is_owner_key
    monkeypatch.setattr(auth, "_is_owner_key", lambda k: checks.append(k) or real(k))
    assert auth._hash_key("owner-key-123") == first
    assert checks == []
    auth._forget_cached_digests()


def test_a_digest_memoised_while_waiting_for_the_lock_is_reused(auth, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-123")
    auth._forget_cached_digests()
    ck = auth._memo_key("owner-key-123") + "@" + auth._cache_scope()
    real = auth._is_owner_key

    def owner_then_concurrent_fill(k):
        result = real(k)
        auth._cache_put(ck, "filled-by-a-concurrent-request")
        return result
    monkeypatch.setattr(auth, "_is_owner_key", owner_then_concurrent_fill)
    derived = []
    monkeypatch.setattr(auth, "_owner_digest", lambda k: derived.append(k) or "derived")
    assert auth._hash_key("owner-key-123") == "filled-by-a-concurrent-request"
    assert derived == []
    auth._forget_cached_digests()


def test_digest_memo_evicts_the_least_recently_used_entry(auth):
    auth._forget_cached_digests()
    newest = auth._DIGEST_CACHE_MAX
    for i in range(newest + 1):
        auth._cache_put(f"k{i}", f"d{i}")
    assert auth._cache_get("k0") is None
    assert auth._cache_get(f"k{newest}") == f"d{newest}"
    assert len(auth._digest_cache) == auth._DIGEST_CACHE_MAX
    auth._forget_cached_digests()


def test_digest_memo_scope_is_the_kdf_file_of_the_current_home(auth):
    assert auth._cache_scope() == str(auth.owner_kdf_file())


def test_owner_key_read_retries_with_escalating_backoff(auth, monkeypatch):
    from pathlib import Path
    auth.set_api_key("s3cret-key")
    key_path = auth.key_file()
    real_read = Path.read_text
    fails = {"left": auth._KEY_READ_RETRIES - 1}

    def flaky(self, *a, **k):
        if self == key_path and fails["left"]:
            fails["left"] -= 1
            raise PermissionError("sharing violation")
        return real_read(self, *a, **k)
    clock = _TimeProxy()
    monkeypatch.setattr(auth, "time", clock)
    monkeypatch.setattr(Path, "read_text", flaky)
    status, text, err = auth._read_owner_key_file()
    assert (status, text.strip(), err) == (auth._KEY_OK, "s3cret-key", None)
    assert clock.sleeps == [
        min(auth._KEY_READ_BACKOFF * (i + 1), auth._KEY_READ_BACKOFF_CAP)
        for i in range(auth._KEY_READ_RETRIES - 1)]


def test_persistently_unreadable_owner_key_reports_its_error(auth, monkeypatch):
    from pathlib import Path
    auth.set_api_key("s3cret-key")
    key_path = auth.key_file()
    real_read = Path.read_text

    def denied(self, *a, **k):
        if self == key_path:
            raise PermissionError("denied")
        return real_read(self, *a, **k)
    clock = _TimeProxy()
    monkeypatch.setattr(auth, "time", clock)
    monkeypatch.setattr(Path, "read_text", denied)
    status, text, err = auth._read_owner_key_file()
    assert status == auth._KEY_UNREADABLE and text is None
    assert isinstance(err, PermissionError)
    assert len(clock.sleeps) == auth._KEY_READ_RETRIES - 1


def test_unreadable_owner_key_file_is_reported(auth, monkeypatch, caplog):
    path = auth.key_file()
    path.mkdir(parents=True)
    monkeypatch.setattr(auth, "time", _TimeProxy())
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth._read_key_file() is None
    [r] = _logged(caplog, "exists but is unreadable")
    assert r.msg == ("owner key file %s exists but is unreadable (%s); "
                     "treating auth as IN EFFECT (fail closed) until it can "
                     "be read - fix its permissions")
    assert r.args[0] == path
    assert isinstance(r.args[1], OSError)


def test_verify_stamps_last_used_on_the_verified_key_only(auth, monkeypatch):
    monkeypatch.setattr(auth, "_last_used_writes", {})
    a = auth.create_key("a", ["chat"])
    b = auth.create_key("b", ["chat"])
    assert auth.verify(b["key"]) == {"chat"}
    assert isinstance(_last_used(auth, b["id"]), float)
    assert _last_used(auth, a["id"]) is None


def test_last_used_is_rewritten_at_most_once_per_throttle_window(auth, monkeypatch):
    monkeypatch.setattr(auth, "_last_used_writes", {})
    clock = _TimeProxy(mono=10_000.0)
    monkeypatch.setattr(auth, "time", clock)
    made = auth.create_key("a", ["chat"])
    key_hash = _sha256_hex(made["key"])
    auth._touch_last_used(key_hash)
    assert isinstance(_last_used(auth, made["id"]), float)
    _set_last_used(auth, made["id"], 1.0)
    clock.mono = 10_000.0 + auth._LAST_USED_THROTTLE_S - 1
    auth._touch_last_used(key_hash)
    assert _last_used(auth, made["id"]) == 1.0
    clock.mono = 10_000.0 + auth._LAST_USED_THROTTLE_S
    auth._touch_last_used(key_hash)
    assert _last_used(auth, made["id"]) != 1.0


def test_a_failed_last_used_stamp_is_logged_not_raised(auth, monkeypatch, caplog):
    import localm.config as cfg
    monkeypatch.setattr(auth, "_last_used_writes", {})
    path = auth.keystore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{corrupt", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="localm"):
        auth._touch_last_used("a" * 64)
    [r] = _logged(caplog, "stamp failed")
    assert r.msg == "last_used stamp failed (non-fatal): %s"
    assert isinstance(r.args[0], cfg.ConfigUnreadable)


def test_create_key_names_every_refused_scope(auth):
    with pytest.raises(ValueError) as bad:
        auth.create_key("n", ["bad one", "bad two"])
    assert str(bad.value) == "Unknown scope(s): bad one, bad two"
    with pytest.raises(PermissionError) as priv:
        auth.create_key("n", ["admin", "config:write"])
    assert str(priv.value) == (
        "Refusing to grant privileged scope(s): admin, config:write. Only the "
        "owner key may mint keys with these capabilities.")


def test_create_key_record_shape(auth):
    made = auth.create_key("  my key  ", ["chat"])
    assert made["name"] == "my key"
    assert len(made["id"]) == 12
    int(made["id"], 16)
    [listed] = auth.list_keys()
    assert isinstance(listed["created"], float)
    for blank in (None, "", "   "):
        assert auth.create_key(blank, ["chat"])["name"] == "key"


def test_legacy_record_attributes_fall_back_to_the_callers_default(auth):
    _write_keystore(auth, [{"id": "legacy1", "hash": _sha256_hex("legacy-token")}])
    assert auth.fs_access_for("legacy-token", default="host") == "host"
    assert auth.rag_roots_for("legacy-token", default=["/srv/docs"]) == ["/srv/docs"]
    assert auth.verify("legacy-token") == set()


def test_fs_access_defaults_to_none(auth):
    assert auth.fs_access_for("") == "none"
    assert auth.fs_access_for(None) == "none"
    assert auth.fs_access_for("unknown-token") == "none"
    assert auth.norm_fs_access("bogus") == "none"


def test_norm_rag_roots_skips_non_strings_and_keeps_going(auth):
    assert auth.norm_rag_roots(["a", 5, "b", None, "a"]) == ["a", "b"]


def test_removal_with_no_target_does_not_report_open_mode(auth):
    assert auth.removal_leaves_no_key() is False


@pytest.mark.parametrize("value", ["TRUE", "Yes", " on "])
def test_require_auth_env_is_case_insensitive(auth, monkeypatch, value):
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", value)
    assert auth.require_auth_enabled() is True


def test_require_auth_config_load_failure_message(auth, monkeypatch, caplog):
    import localm.config as cfg

    def boom():
        raise OSError("no home")
    monkeypatch.setattr(cfg, "load_config_checked", boom)
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth.require_auth_enabled() is True
    [r] = _logged(caplog, "config unreadable")
    assert r.msg == ("config unreadable; cannot confirm require_auth - treating as "
                     "required (fail closed) until it can be read")


def test_verify_refuses_absent_or_blank_tokens(auth):
    assert auth.verify(None) is None
    assert auth.verify("") is None
    assert auth.verify("   ") is None


def test_verify_accepts_a_key_until_its_expiry_has_passed(auth, monkeypatch):
    monkeypatch.setattr(auth, "_last_used_writes", {})
    clock = _TimeProxy(now=1_000_000.0)
    monkeypatch.setattr(auth, "time", clock)
    live = auth.create_key("live", ["chat"], expires=1_000_010.0)
    stale = auth.create_key("stale", ["chat"], expires=999_999.0)
    assert auth.verify(stale["key"]) is None
    clock.now = 1_000_010.0
    assert auth.verify(live["key"]) == {"chat"}
    clock.now = 1_000_010.5
    assert auth.verify(live["key"]) is None


def test_an_unreadable_keystore_refuses_a_write_naming_the_file(auth):
    import localm.config as cfg
    path = auth.keystore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(cfg.ConfigUnreadable) as refused:
        auth.create_key("a", ["chat"])
    assert str(refused.value) == (
        "auth.json exists but could not be read as a JSON list, so saving would "
        "replace every key in it with just this change; refused. Fix or remove "
        "auth.json (deleting it resets to no scoped keys).")


def test_a_kdf_keystore_record_without_a_salt_uses_an_empty_salt(auth):
    rec = {"alg": "scrypt", "n": 16, "r": 8, "p": 1, "dklen": 32}
    assert auth._record_digest_for(rec, "k", lambda: "fast") == \
        auth._scrypt_derive("k", b"", 16, 8, 1, 32)


def test_removal_leaves_no_key_follows_env_owner_and_remaining_records(auth, monkeypatch):
    assert auth.removal_leaves_no_key(revoke_id="ghost") is False
    made = auth.create_key("only", ["chat"])
    assert auth.removal_leaves_no_key(revoke_id=made["id"]) is True
    other = auth.create_key("other", ["chat"])
    assert auth.removal_leaves_no_key(revoke_id=made["id"]) is False
    auth.revoke_key(other["id"])
    assert auth.removal_leaves_no_key(clear_all=True) is True
    auth.set_api_key("owner-key-123")
    assert auth.removal_leaves_no_key(revoke_id=made["id"]) is False
    auth.clear_api_key()
    monkeypatch.setenv("LOCALM_API_KEY", "env-key-1234")
    assert auth.removal_leaves_no_key(clear_all=True) is False


def test_verify_stamps_the_alg_marker_only_on_a_legacy_record(auth, monkeypatch):
    monkeypatch.setattr(auth, "_last_used_writes", {})
    marks = []
    monkeypatch.setattr(auth, "_mark_record_alg", lambda *a: marks.append(a))
    made = auth.create_key("a", ["chat"])
    assert auth.verify(made["key"]) == {"chat"}
    assert marks == []
    _write_keystore(auth, [{"id": "legacy1", "hash": _sha256_hex("legacy-token"),
                            "scopes": ["chat"]}])
    assert auth.verify("legacy-token") == {"chat"}
    assert marks == [("legacy1", auth._ALG_FAST)]
