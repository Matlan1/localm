# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keys and devices: key expiry, the privileged coder:full scope, and key presets.

These are the pure-logic checks; the route-level coder:full to
unrestricted-session wiring and the scoped-key pairing-QR endpoint are covered
where the GUI app harness lives.
"""
from __future__ import annotations

import time

import pytest

from localm import scopes as S
from localm.settings_schema import validate_update


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the data dir so the keystore (auth.json) is written under tmp."""
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))   # home_dir() is lazy -> keystore here
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)     # ensure_dirs() must not touch real home
    monkeypatch.setattr(_cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", h / "config.json")
    return h


# --- coder:full is a known, privileged, owner-only scope -------------------- #

def test_coder_full_is_privileged_and_known():
    assert S.CODER_FULL == "coder:full"
    assert S.is_valid_scope(S.CODER_FULL)
    assert S.CODER_FULL in S.PRIVILEGED_SCOPES
    assert S.CODER_FULL in S.all_known_scopes()


def test_coder_full_grants_base_coder():
    # coder:full is the unrestricted coder, so it must satisfy a plain `coder`
    # route gate; the reverse is not true, and admin still implies everything.
    assert S.grants({S.CODER_FULL}, S.CODER)
    assert not S.grants({S.CODER}, S.CODER_FULL)
    assert S.grants({S.ADMIN}, S.CODER_FULL)


def test_non_owner_cannot_mint_coder_full(home):
    from localm import auth
    with pytest.raises(PermissionError):
        auth.create_key("evil", ["coder:full"])              # allow_privileged=False
    made = auth.create_key("dev", ["coder:full"], allow_privileged=True)
    assert "coder:full" in made["scopes"]
    assert auth.verify(made["key"]) == {"coder:full"}


# --- expiry ----------------------------------------------------------------- #

def test_expired_key_is_rejected(home):
    from localm import auth
    made = auth.create_key("phone", ["chat"], expires=time.time() - 5)
    assert made["expires"] is not None
    assert auth.verify(made["key"]) is None                  # expired -> no scopes


def test_unexpired_key_verifies_and_lists_expiry(home):
    from localm import auth
    exp = time.time() + 3600
    made = auth.create_key("phone", ["chat"], expires=exp)
    assert auth.verify(made["key"]) == {"chat"}
    listed = {k["id"]: k for k in auth.list_keys()}[made["id"]]
    assert listed["expires"] == pytest.approx(exp, abs=1)


def test_no_expiry_never_expires(home):
    from localm import auth
    made = auth.create_key("forever", ["chat"])
    assert made["expires"] is None
    assert auth.verify(made["key"]) == {"chat"}


# --- last-used (device hygiene) --------------------------------------------- #

def test_verify_stamps_last_used_and_throttles(home):
    from localm import auth
    made = auth.create_key("phone", ["chat"])
    by_id = lambda: {k["id"]: k for k in auth.list_keys()}[made["id"]]   # noqa: E731
    assert by_id()["last_used"] is None              # never used yet
    auth.verify(made["key"])
    lu = by_id()["last_used"]
    assert lu is not None                            # stamped on first use
    auth.verify(made["key"])                         # within the throttle window
    assert by_id()["last_used"] == lu                # not rewritten every request


# --- presets ---------------------------------------------------------------- #

def test_key_presets_seeded_in_defaults():
    from localm.config import DEFAULT_CONFIG
    names = {p["name"] for p in DEFAULT_CONFIG["key_presets"]}
    assert {"Minimal", "Companion", "Full", "Admin"} <= names
    for p in DEFAULT_CONFIG["key_presets"]:
        for sc in p["scopes"]:
            assert S.is_valid_scope(sc), sc


def test_key_presets_validation_rejects_bad():
    with pytest.raises(ValueError):
        validate_update({"key_presets": [{"name": "X", "scopes": ["nope!!"]}]})
    with pytest.raises(ValueError):
        validate_update({"key_presets": [{"name": "", "scopes": ["chat"]}]})


def test_key_presets_validation_normalizes():
    out = validate_update(
        {"key_presets": [{"name": " P ", "scopes": ["coder", "chat", "coder"]}]})
    p = out["key_presets"][0]
    assert p["name"] == "P"
    assert p["scopes"] == ["chat", "coder"]                  # deduped + sorted
