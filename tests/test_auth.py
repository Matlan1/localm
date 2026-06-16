"""Tests for the shared API-key auth (localm/auth.py) and its enforcement in
the HTTP server's _require_auth dependency."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


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
    assert auth.get_api_key() is None
    auth.set_api_key("s3cret")
    assert auth.get_api_key() == "s3cret"
    auth.clear_api_key()
    assert auth.get_api_key() is None


def test_env_overrides_file(auth, monkeypatch):
    auth.set_api_key("filekey")
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


def test_require_auth_dependency(auth, monkeypatch):
    from localm.inference.http_server import _require_auth

    # open mode: no key, not required -> allowed
    assert _require_auth(None) is None

    # required but no key -> 503
    monkeypatch.setenv("LOCALM_REQUIRE_AUTH", "1")
    with pytest.raises(HTTPException) as exc:
        _require_auth(None)
    assert exc.value.status_code == 503
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH")

    # key configured (file): missing/invalid credentials -> 401
    auth.set_api_key("k")
    with pytest.raises(HTTPException) as exc:
        _require_auth(None)
    assert exc.value.status_code == 401
    bad = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong")
    with pytest.raises(HTTPException) as exc:
        _require_auth(bad)
    assert exc.value.status_code == 401

    # correct credentials -> allowed
    good = HTTPAuthorizationCredentials(scheme="Bearer", credentials="k")
    assert _require_auth(good) is None


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


def test_owner_key_is_admin(auth, monkeypatch):
    from localm import scopes as S
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    assert auth.verify("ownersecret") == {S.ADMIN}
    assert auth.verify("nope") is None


def test_require_scope_enforcement(auth):
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from localm import scopes as S
    from localm.inference.http_server import require_scope, _require_auth

    # open mode: no keys configured -> allowed
    assert require_scope(S.PLUGINS_ADMIN)(None) is None

    # a named key WITHOUT plugins:admin
    made = auth.create_key("reader", [S.CHAT])
    cred = HTTPAuthorizationCredentials(scheme="Bearer", credentials=made["key"])

    # now a key exists -> enforced; this key lacks plugins:admin -> 403
    with pytest.raises(HTTPException) as exc:
        require_scope(S.PLUGINS_ADMIN)(cred)
    assert exc.value.status_code == 403

    # it does satisfy a chat-scoped requirement, and any-valid-key auth
    assert require_scope(S.CHAT)(cred) is None
    assert _require_auth(cred) is None

    # a wrong/unknown key -> 401
    bad = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong")
    with pytest.raises(HTTPException) as exc:
        require_scope(S.CHAT)(bad)
    assert exc.value.status_code == 401


def test_owner_key_grants_every_scope(auth, monkeypatch):
    from fastapi.security import HTTPAuthorizationCredentials
    from localm import scopes as S
    from localm.inference.http_server import require_scope
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    cred = HTTPAuthorizationCredentials(scheme="Bearer", credentials="ownersecret")
    assert require_scope(S.PLUGINS_ADMIN)(cred) is None
    assert require_scope(S.KEYS_ADMIN)(cred) is None


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
