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
