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


def test_set_credentials_creates_the_home_dir_if_missing(tmp_path, monkeypatch):
    """The home directory does not exist yet here (unlike isolated_home's
    fixture, which pre-creates it) - set_credentials must create it itself,
    matching sessions.py's own atomic-write convention, rather than assuming
    some earlier startup step already ran."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    assert not home.exists()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    from localm.model_source_credentials import get_hf_token, set_credentials
    set_credentials({"hf_token": "hf_first_write"})
    assert get_hf_token() == "hf_first_write"


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


def _fail_next_read_once(monkeypatch, target_name: str) -> None:
    """Make the NEXT read-mode open() of a file named *target_name* raise a
    plain (non-Permission) OSError exactly once, then behave normally
    afterward. Mirrors test_auth_keystore_fail_closed.py's helper - see its
    docstring for why both builtins.open and io.open are patched, and why a
    plain OSError (not PermissionError) is used."""
    import builtins
    import io as io_mod
    real_open = builtins.open
    state = {"armed": True}

    def flaky(file, mode="r", *args, **kwargs):
        name = getattr(file, "name", None)
        if name is None:
            name = str(file).replace("\\", "/").rsplit("/", 1)[-1]
        if (state["armed"] and name == target_name
                and "r" in mode and "w" not in mode and "a" not in mode):
            state["armed"] = False
            raise OSError(f"simulated persistent read failure for {target_name}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky)
    monkeypatch.setattr(io_mod, "open", flaky)


def test_set_credentials_refuses_rather_than_wipes_on_unreadable_store(isolated_home, monkeypatch):
    """_read_all() used to fail OPEN (return {}) on any read error, and
    set_credentials() read-modify-wrote on that result - so a single
    reported-unreadable read while setting civitai_api_key would silently
    drop an already-stored hf_token. Pins the fix: refuse instead."""
    from localm.model_source_credentials import (credentials_path, get_civitai_api_key,
                                                  get_hf_token, set_credentials)
    set_credentials({"hf_token": "hf_should_survive"})
    assert get_hf_token() == "hf_should_survive"

    import localm.config as cfg
    _fail_next_read_once(monkeypatch, credentials_path().name)

    exc = None
    try:
        set_credentials({"civitai_api_key": "civ_should_not_wipe_hf"})
    except Exception as e:  # noqa: BLE001
        exc = e

    # DATA FIRST: hf_token must survive a reported-unreadable read, whether
    # set_credentials raised or - on the unfixed code - silently "succeeded"
    # by replacing the store with just the new key.
    assert get_hf_token() == "hf_should_survive", (
        "a reported-unreadable read destroyed the existing hf_token instead "
        "of refusing the write")

    # And the failure must be LOUD, never a silent partial overwrite.
    assert isinstance(exc, cfg.ConfigUnreadable), (
        f"set_credentials() must raise ConfigUnreadable on an unreadable "
        f"store read rather than silently succeed; got {exc!r}")

    # One-shot: a normal call right after succeeds, and both values persist.
    set_credentials({"civitai_api_key": "civ_1"})
    assert get_hf_token() == "hf_should_survive"
    assert get_civitai_api_key() == "civ_1"


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


def test_get_credential_source_and_is_set(isolated_home, monkeypatch):
    from localm.model_source_credentials import (get_credential_source,
                                                  is_credential_set,
                                                  set_credentials)
    assert get_credential_source("hf_token") is None
    assert is_credential_set("hf_token") is False

    set_credentials({"hf_token": "hf_stored_1"})
    assert get_credential_source("hf_token") == "stored"
    assert is_credential_set("hf_token") is True

    monkeypatch.setenv("HF_TOKEN", "hf_env_1")
    assert get_credential_source("hf_token") == "stored"
    assert is_credential_set("hf_token") is True

    set_credentials({"hf_token": ""})
    assert get_credential_source("hf_token") == "env"
    assert is_credential_set("hf_token") is True

    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert get_credential_source("hf_token") is None
    assert is_credential_set("hf_token") is False


def test_whitespace_only_value_agrees_between_get_and_source(isolated_home, monkeypatch):
    from localm.model_source_credentials import get_credential, get_credential_source
    monkeypatch.setenv("HF_TOKEN", " ")
    assert get_credential("hf_token") is None
    assert get_credential_source("hf_token") is None

    monkeypatch.setenv("HF_TOKEN", "  real-token  ")
    assert get_credential("hf_token") == "real-token"
    assert get_credential_source("hf_token") == "env"


# --------------------------------------------------------------------------- #
#  PATCH /v1/config changes a stored credential only once every other gate    #
#  in the same request has passed                                              #
# --------------------------------------------------------------------------- #

def _stored_credentials() -> dict:
    """The credentials file's own contents, read straight from disk."""
    from localm.model_source_credentials import credentials_path
    path = credentials_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def open_client(monkeypatch):
    """Open-mode app (no API key anywhere) with the loopback shell token that
    PATCH /v1/config requires when no key is configured."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(None)
    with TestClient(app, headers={
            "Authorization": f"Bearer {app.state.shell_token}"}) as c:
        yield c


@pytest.mark.parametrize("new_value", ["", "hf_replacement_value"])
def test_rejected_config_field_leaves_the_stored_credential_alone(app_env, new_value):
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import set_credentials
    set_credentials({"hf_token": "hf_keep_me"})

    r = c.patch("/v1/config", headers=_owner(),
                json={"hf_token": new_value, "import_max_depth": 50})

    assert _stored_credentials().get("hf_token") == "hf_keep_me", (
        "a PATCH that validate_update rejected still changed the stored hf_token")
    assert r.status_code == 400, r.text
    assert "import_max_depth" in r.json()["detail"]


def test_require_auth_lockout_refusal_leaves_the_stored_credential_alone(open_client):
    from localm.auth import any_key_configured
    from localm.model_source_credentials import set_credentials
    assert not any_key_configured(), "the lock-out guard only fires with no key"
    set_credentials({"hf_token": "hf_keep_me"})

    r = open_client.patch("/v1/config",
                          json={"hf_token": "", "require_auth": True})

    assert _stored_credentials().get("hf_token") == "hf_keep_me", (
        "a PATCH refused by the require_auth lock-out guard still cleared hf_token")
    assert r.status_code == 400, r.text
    assert "require_auth" in r.json()["detail"]
    from localm.config import load_config
    assert load_config().get("require_auth") is not True


def test_unconfirmed_embedding_switch_leaves_the_stored_credential_alone(app_env):
    """The needs_confirm dry run writes nothing; the confirmed re-send, which
    carries the same body, applies the credential change with the rest."""
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import set_credentials
    from localm.rag.store import Collection
    set_credentials({"hf_token": "hf_keep_me"})
    coll = Collection("docs").create()
    coll._chunks = [{"source": "doc0.txt", "pos": 0, "text": "alpha"}]
    coll._vectors = [[0.1] * 8]
    coll._meta["embedding_model"] = "old-model"
    coll._save()
    body = {"hf_token": "", "embedding_model": "new-model"}

    dry = c.patch("/v1/config", headers=_owner(), json=body)

    assert _stored_credentials().get("hf_token") == "hf_keep_me", (
        "the unconfirmed needs_confirm dry run cleared hf_token")
    assert dry.status_code == 200, dry.text
    assert dry.json().get("needs_confirm") is True

    confirmed = c.patch("/v1/config", headers=_owner(),
                        json={**body, "confirm": True})

    assert "hf_token" not in _stored_credentials()
    assert confirmed.status_code == 200, confirmed.text
    from localm.config import load_config
    assert load_config().get("embedding_model") == "new-model"


def test_config_save_timeout_leaves_the_stored_credential_alone(app_env, monkeypatch):
    """A 504 from the bounded config write means the credential store was never
    touched: the credential change is applied only after update_config returns."""
    import threading

    import localm.config as cfg
    import localm.inference.routes.config as config_routes
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import set_credentials
    set_credentials({"hf_token": "hf_keep_me"})
    entered = threading.Event()
    release = threading.Event()

    def stalled_update_config(mutator):
        entered.set()
        release.wait(10)
        return {}

    monkeypatch.setattr(cfg, "update_config", stalled_update_config)
    monkeypatch.setattr(config_routes, "_CONFIG_RMW_TIMEOUT_S", 0.2)
    try:
        r = c.patch("/v1/config", headers=_owner(),
                    json={"hf_token": "", "import_max_depth": 5})
    finally:
        release.set()

    assert entered.is_set(), "the stalled update_config was never reached"
    assert _stored_credentials().get("hf_token") == "hf_keep_me", (
        "hf_token was cleared although the config save timed out")
    assert r.status_code == 504, r.text


def test_accepted_patch_applies_a_blank_credential_with_the_config_change(app_env):
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import set_credentials
    set_credentials({"hf_token": "hf_to_remove", "civitai_api_key": "civ_keep"})

    r = c.patch("/v1/config", headers=_owner(),
                json={"hf_token": "", "import_max_depth": 5})

    assert _stored_credentials() == {"civitai_api_key": "civ_keep"}
    assert r.status_code == 200, r.text
    from localm.config import load_config
    assert load_config()["import_max_depth"] == 5


def test_malformed_credential_is_refused_before_any_config_write(app_env):
    c, _scoped_key, _home = app_env
    from localm.config import load_config
    before = load_config()["import_max_depth"]
    assert before != 5

    r = c.patch("/v1/config", headers=_owner(),
                json={"hf_token": "x" * 5000, "import_max_depth": 5})

    assert load_config()["import_max_depth"] == before
    assert _stored_credentials() == {}
    assert r.status_code == 400, r.text
    assert "hf_token" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  an unreadable credential store is reported as such, never as "not set"     #
# --------------------------------------------------------------------------- #

_CORRUPT_STORE = b'{"hf_token": '


def _answering(c):
    """A client on *c*'s already-started app that returns an unhandled server
    error as a 500 response instead of raising it into the test."""
    return TestClient(c.app, raise_server_exceptions=False)


def _write_corrupt_store():
    from localm.model_source_credentials import credentials_path
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CORRUPT_STORE)
    return path


def test_unreadable_store_reports_unreadable_not_unset(isolated_home, monkeypatch):
    from localm.model_source_credentials import get_credential_source
    _write_corrupt_store()
    assert get_credential_source("hf_token") == "unreadable"
    assert get_credential_source("civitai_api_key") == "unreadable"

    monkeypatch.setenv("HF_TOKEN", "hf_env_1")
    assert get_credential_source("hf_token") == "unreadable"


def test_schema_marks_credential_status_unknown_on_an_unreadable_store(isolated_home):
    from localm.config import load_config
    from localm.settings_schema import schema_json
    _write_corrupt_store()

    fields = {f["key"]: f for f in schema_json(values=load_config())}

    for key in ("hf_token", "civitai_api_key"):
        f = fields[key]
        assert f.get("status_unknown") is True, (
            f"{key}: an unreadable store was reported without the unknown flag: {f}")
        assert f["is_set"] is False
        assert f["env_set"] is False


def test_schema_status_unknown_is_false_on_a_readable_store(isolated_home):
    from localm.config import load_config
    from localm.model_source_credentials import set_credentials
    from localm.settings_schema import schema_json
    set_credentials({"hf_token": "hf_stored_1"})

    fields = {f["key"]: f for f in schema_json(values=load_config())}

    assert fields["hf_token"]["status_unknown"] is False
    assert fields["hf_token"]["is_set"] is True
    assert fields["civitai_api_key"]["status_unknown"] is False
    assert fields["civitai_api_key"]["is_set"] is False


def test_schema_route_marks_credential_status_unknown(app_env):
    c, _scoped_key, _home = app_env
    _write_corrupt_store()

    fields = {f["key"]: f for f in
              c.get("/v1/config/schema", headers=_owner()).json()["fields"]}

    assert fields["hf_token"]["status_unknown"] is True
    assert fields["hf_token"]["is_set"] is False


def test_patch_on_an_unreadable_store_is_a_409_naming_the_file(app_env):
    c, _scoped_key, _home = app_env
    path = _write_corrupt_store()

    r = _answering(c).patch("/v1/config", headers=_owner(),
                            json={"hf_token": "hf_new"})

    assert path.read_bytes() == _CORRUPT_STORE
    assert r.status_code == 409, r.text
    assert "model_source_credentials.json" in r.json()["detail"]


def test_patch_on_an_unreadable_store_applies_no_config_change(app_env):
    c, _scoped_key, _home = app_env
    from localm.config import load_config
    before = load_config()["import_max_depth"]
    assert before != 5
    path = _write_corrupt_store()

    r = _answering(c).patch("/v1/config", headers=_owner(),
                            json={"hf_token": "", "import_max_depth": 5})

    assert path.read_bytes() == _CORRUPT_STORE
    assert load_config()["import_max_depth"] == before, (
        "the config change was applied although the credential change was refused")
    assert r.status_code == 409, r.text
    assert "model_source_credentials.json" in r.json()["detail"]


def test_store_unreadable_only_at_write_time_is_still_a_409(app_env, monkeypatch):
    """The store turning unreadable between the readability check and the
    write is refused by set_credentials itself and still answered as a 409."""
    import localm.model_source_credentials as msc
    c, _scoped_key, _home = app_env
    path = _write_corrupt_store()
    checked = []
    monkeypatch.setattr(msc, "check_credentials_readable",
                        lambda: checked.append(True))

    r = _answering(c).patch("/v1/config", headers=_owner(),
                            json={"hf_token": "hf_new"})

    assert checked == [True], "the patched readability check was not reached"
    assert path.read_bytes() == _CORRUPT_STORE
    assert r.status_code == 409, r.text
    assert "model_source_credentials.json" in r.json()["detail"]


def test_patch_on_an_unreadable_config_file_is_a_409_naming_the_file(app_env):
    import localm.config as cfg
    c, _scoped_key, _home = app_env
    from localm.model_source_credentials import set_credentials
    set_credentials({"hf_token": "hf_keep_me"})
    corrupt = b'{"import_max_depth": '
    cfg.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.CONFIG_FILE.write_bytes(corrupt)
    cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".bak").unlink(missing_ok=True)

    r = _answering(c).patch("/v1/config", headers=_owner(),
                            json={"hf_token": "", "import_max_depth": 5})

    assert cfg.CONFIG_FILE.read_bytes() == corrupt
    assert _stored_credentials().get("hf_token") == "hf_keep_me"
    assert r.status_code == 409, r.text
    assert "config.json" in r.json()["detail"]


def test_cli_config_credential_on_an_unreadable_store_is_a_usage_error(isolated_home):
    import click
    from localm.cli.models import config_cmd
    path = _write_corrupt_store()

    exc = None
    try:
        config_cmd.callback(key="hf_token", value="hf_cli_token_1")
    except Exception as e:  # noqa: BLE001
        exc = e

    assert path.read_bytes() == _CORRUPT_STORE
    assert isinstance(exc, click.ClickException), (
        f"an unreadable store must be a ClickException, not a crash; got {exc!r}")
    assert "model_source_credentials.json" in exc.message


def test_cli_config_setting_on_an_unreadable_config_is_a_usage_error(isolated_home):
    import click

    import localm.config as cfg
    from localm.cli.models import config_cmd
    corrupt = b'{"temperature": '
    cfg.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.CONFIG_FILE.write_bytes(corrupt)
    cfg.CONFIG_FILE.with_name(cfg.CONFIG_FILE.name + ".bak").unlink(missing_ok=True)

    exc = None
    try:
        config_cmd.callback(key="temperature", value="0.9")
    except Exception as e:  # noqa: BLE001
        exc = e

    assert cfg.CONFIG_FILE.read_bytes() == corrupt
    assert isinstance(exc, click.ClickException), (
        f"an unreadable config.json must be a ClickException, not a crash; got {exc!r}")
    assert "config.json" in exc.message
