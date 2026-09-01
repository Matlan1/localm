# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owner-only credential store for optional HF/CivitAI tokens (ADR-0015 decision
4). The keys are deliberately absent from DEFAULT_CONFIG, so PATCH /v1/config and
`localm config` must intercept them BEFORE validate_update/update_config runs;
the fires-control below asserts directly against the on-disk config.json that a
saved token never lands there, which is the property the whole design exists
for."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-msc7-abc123"


def _owner():
    return {"Authorization": f"Bearer {OWNER_KEY}"}


def _scoped(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Protected-mode app: an owner (ADMIN) key via env plus a scoped
    config:read/config:write key in the keystore, under an isolated data dir.
    Mirrors test_config_admin_gating.py's fixture exactly.

    Strips HF_TOKEN/CIVITAI_API_KEY from the environment: get_credential()'s
    env fallback is a real feature, and this DEV MACHINE has a real HF_TOKEN
    set, which otherwise leaks through as a false "value" in every assertion
    that a credential is unset or cleared."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)      # owner key -> {ADMIN}
    from localm import auth
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    app = create_app(None)
    with TestClient(app) as c:
        yield c, scoped, home


# --------------------------------------------------------------------------- #
#  the store module itself                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolates BOTH access shapes config.py uses: home_dir() re-derives from
    LOCALM_HOME live (what model_source_credentials.py calls), while
    load_config/update_config read the CONFIG_FILE/REGISTRY_FILE/MODELS_DIR
    globals frozen at import time - a CLI test exercising the ordinary
    (non-credential) config_cmd path needs both patched.

    Also strips HF_TOKEN/CIVITAI_API_KEY (see app_env's docstring above) so
    every test starts from a real "nothing set" baseline regardless of the
    host machine's own environment."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def test_get_credential_none_when_unset(isolated_home):
    from localm.model_source_credentials import get_hf_token
    assert get_hf_token() is None


def test_set_then_get_round_trips(isolated_home):
    from localm.model_source_credentials import get_hf_token, set_credentials
    set_credentials({"hf_token": "hf_abcDEF123"})
    assert get_hf_token() == "hf_abcDEF123"


def test_set_strips_whitespace(isolated_home):
    from localm.model_source_credentials import get_civitai_api_key, set_credentials
    set_credentials({"civitai_api_key": "  spaced-key  "})
    assert get_civitai_api_key() == "spaced-key"


def test_empty_string_clears(isolated_home):
    from localm.model_source_credentials import get_hf_token, set_credentials
    set_credentials({"hf_token": "hf_abcDEF123"})
    assert get_hf_token() == "hf_abcDEF123"
    set_credentials({"hf_token": ""})
    assert get_hf_token() is None


def test_none_clears(isolated_home):
    from localm.model_source_credentials import get_hf_token, set_credentials
    set_credentials({"hf_token": "hf_abcDEF123"})
    set_credentials({"hf_token": None})
    assert get_hf_token() is None


def test_env_var_fallback_used_when_unset(isolated_home, monkeypatch):
    from localm.model_source_credentials import get_hf_token
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert get_hf_token() == "from-env"


def test_stored_value_wins_over_env_var(isolated_home, monkeypatch):
    from localm.model_source_credentials import get_hf_token, set_credentials
    monkeypatch.setenv("HF_TOKEN", "from-env")
    set_credentials({"hf_token": "from-store"})
    assert get_hf_token() == "from-store"


def test_civitai_env_fallback_is_its_own_var_name(isolated_home, monkeypatch):
    from localm.model_source_credentials import get_civitai_api_key
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("CIVITAI_API_KEY", "civ-env-key")
    assert get_civitai_api_key() == "civ-env-key"


def test_unknown_key_rejected(isolated_home):
    from localm.model_source_credentials import set_credentials
    with pytest.raises(ValueError, match="unknown credential key"):
        set_credentials({"not_a_real_key": "x"})


def test_non_string_value_rejected(isolated_home):
    from localm.model_source_credentials import set_credentials
    with pytest.raises(ValueError, match="expected a string"):
        set_credentials({"hf_token": 12345})


def test_oversized_value_rejected(isolated_home):
    from localm.model_source_credentials import set_credentials
    with pytest.raises(ValueError, match="too long"):
        set_credentials({"hf_token": "x" * 5000})


def test_batched_apply_is_one_file_write(isolated_home):
    """Both keys land from a single call without clobbering each other - the
    read-modify-write must merge, not overwrite key-by-key."""
    from localm.model_source_credentials import (credentials_path,
                                                  get_civitai_api_key,
                                                  get_hf_token, set_credentials)
    set_credentials({"hf_token": "hf_1", "civitai_api_key": "civ_1"})
    assert get_hf_token() == "hf_1"
    assert get_civitai_api_key() == "civ_1"
    on_disk = json.loads(credentials_path().read_text(encoding="utf-8"))
    assert on_disk == {"hf_token": "hf_1", "civitai_api_key": "civ_1"}

    # Updating one key leaves the other untouched.
    set_credentials({"hf_token": "hf_2"})
    assert get_hf_token() == "hf_2"
    assert get_civitai_api_key() == "civ_1"


def test_credentials_file_is_owner_restricted(isolated_home):
    """restrict_file_perms is applied via atomic_write_private, the same
    primitive auth.key/sessions.json use - assert the file is not left at the
    platform's open default."""
    from localm.model_source_credentials import credentials_path, set_credentials
    set_credentials({"hf_token": "hf_1"})
    path = credentials_path()
    assert path.is_file()
    import os
    if os.name != "nt":
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"credentials file mode is {oct(mode)}, expected 0600"


def test_a_raised_validation_error_writes_nothing(isolated_home):
    """set_credentials validates every key BEFORE writing any of them: a bad
    civitai_api_key must not leave a partially-applied hf_token on disk."""
    from localm.model_source_credentials import (credentials_path, get_hf_token,
                                                  set_credentials)
    with pytest.raises(ValueError):
        set_credentials({"hf_token": "hf_should_not_land", "civitai_api_key": 999})
    assert get_hf_token() is None
    assert not credentials_path().is_file()


# --------------------------------------------------------------------------- #
#  the field never reaches config.json, over the real PATCH /v1/config route  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key,value", [
    ("hf_token", "hf_realtoken1234567890"),
    ("civitai_api_key", "civitai-real-key-abcdef"),
])
def test_owner_can_set_credential_and_it_never_reaches_config_json(app_env, key, value):
    c, _scoped_key, home = app_env
    r = c.patch("/v1/config", headers=_owner(), json={key: value})
    assert r.status_code == 200, r.text

    from localm.model_source_credentials import get_credential
    assert get_credential(key) == value

    # The fires-control: read the ACTUAL FILE on disk, not a helper's opinion of
    # it. A leak here would defeat the entire point of the owner-only store.
    config_file = home / "config.json"
    if config_file.is_file():
        raw = config_file.read_text(encoding="utf-8")
        assert value not in raw, f"{key}'s value leaked into config.json: {raw!r}"
        assert key not in json.loads(raw)

    # Never echoed back in the PATCH response either.
    assert value not in r.text
    assert key not in r.json()

    # Nor in a subsequent GET /v1/config.
    got = c.get("/v1/config", headers=_owner())
    assert value not in got.text
    assert key not in got.json()


@pytest.mark.parametrize("key", ["hf_token", "civitai_api_key"])
def test_scoped_key_cannot_set_credential(app_env, key):
    """Mirrors test_config_admin_gating.py's admin_only pattern: 403 for the
    non-owner config:write key, and the value stays unset."""
    c, scoped, _home = app_env
    from localm.model_source_credentials import get_credential
    denied = c.patch("/v1/config", headers=_scoped(scoped), json={key: "attacker-value"})
    assert denied.status_code == 403, denied.text
    assert "owner" in denied.text.lower()
    assert get_credential(key) is None

    # Not shipped to the scoped key's schema either.
    scoped_schema = {f["key"] for f in
                     c.get("/v1/config/schema", headers=_scoped(scoped)).json()["fields"]}
    assert key not in scoped_schema


def test_owner_can_clear_a_stored_credential(app_env):
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import get_hf_token
    assert c.patch("/v1/config", headers=_owner(),
                   json={"hf_token": "hf_first"}).status_code == 200
    assert get_hf_token() == "hf_first"
    assert c.patch("/v1/config", headers=_owner(),
                   json={"hf_token": ""}).status_code == 200
    assert get_hf_token() is None


def test_patch_with_only_a_credential_key_still_succeeds(app_env):
    """The rest of validate_update/update_config must tolerate an otherwise-empty
    body once the credential key is popped out of it."""
    c, _scoped_key, _home = app_env
    r = c.patch("/v1/config", headers=_owner(), json={"civitai_api_key": "solo-key"})
    assert r.status_code == 200, r.text


def test_invalid_credential_value_returns_400_not_500(app_env):
    c, _scoped_key, _home = app_env
    r = c.patch("/v1/config", headers=_owner(), json={"hf_token": "x" * 5000})
    assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
#  schema: masked, admin_only, carries a link, never a value                  #
# --------------------------------------------------------------------------- #

def test_schema_field_shape_for_owner(app_env):
    c, _scoped_key, _home = app_env
    fields = {f["key"]: f for f in
              c.get("/v1/config/schema", headers=_owner()).json()["fields"]}
    for key in ("hf_token", "civitai_api_key"):
        f = fields[key]
        assert f["widget"] == "secret"
        assert f["secret"] is True
        assert f["admin_only"] is True
        assert "default" not in f
        assert "value" not in f
        assert "shipped_default" not in f
        assert f["link"]["url"].startswith("https://")


def test_schema_field_carries_no_value_even_when_a_secret_is_stored(app_env):
    """The strongest form of "never round-tripped in plaintext": set a real
    token, then confirm the schema still emits nothing for it."""
    c, _scoped_key, _home = app_env
    assert c.patch("/v1/config", headers=_owner(),
                   json={"hf_token": "hf_should_never_appear"}).status_code == 200
    schema_text = c.get("/v1/config/schema", headers=_owner()).text
    assert "hf_should_never_appear" not in schema_text


# --------------------------------------------------------------------------- #
#  `localm config hf_token ...` routes through the same store, not config.json #
# --------------------------------------------------------------------------- #

def test_cli_config_sets_credential_not_config_json(isolated_home):
    from localm.cli.models import config_cmd
    from localm.model_source_credentials import get_hf_token
    config_cmd.callback(key="hf_token", value="hf_cli_token_1")
    assert get_hf_token() == "hf_cli_token_1"

    config_file = isolated_home / "config.json"
    if config_file.is_file():
        raw = config_file.read_text(encoding="utf-8")
        assert "hf_cli_token_1" not in raw


def test_cli_config_clears_credential_with_empty_string(isolated_home):
    from localm.cli.models import config_cmd
    from localm.model_source_credentials import get_hf_token
    config_cmd.callback(key="hf_token", value="hf_cli_token_1")
    assert get_hf_token() == "hf_cli_token_1"
    config_cmd.callback(key="hf_token", value="")
    assert get_hf_token() is None


def test_cli_config_still_rejects_unknown_key(isolated_home):
    import click
    from localm.cli.models import config_cmd
    with pytest.raises(click.ClickException):
        config_cmd.callback(key="not_a_real_setting", value="x")


def test_cli_config_ordinary_key_unaffected(isolated_home):
    """Regression sanity: the credential branch must not swallow ordinary keys -
    the pre-existing config_cmd path for a real DEFAULT_CONFIG key is untouched."""
    from localm.cli.models import config_cmd
    from localm.config import load_config
    config_cmd.callback(key="temperature", value="0.9")
    assert load_config()["temperature"] == 0.9
