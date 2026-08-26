# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for GET /v1/config/schema.

The schema-driven settings page fetches this endpoint to learn each field's
widget/options/min/max/help plus its CURRENT value (injected as `default`), so
the form can render the right typed control pre-filled. The endpoint must:
  - return the typed field list (the schema, not a flat config dump),
  - carry the right widget + options for SELECT fields (e.g. `mode`),
  - inject the current/default value for non-secret fields,
  - never leak a secret field's value,
  - require config:read once an API key is configured.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

SECRET = "schema-endpoint-key-xyz"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Open-mode client with an isolated config home (so load_config is real)."""
    os.environ.pop("LOCALM_API_KEY", None)
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    # The one PATCH /v1/config test below is a management write that needs the
    # loopback shell token in open mode; seed it by default. GET schema reads are
    # unaffected, and the protected-mode test overrides the header with its key.
    app = create_app(None)
    with TestClient(
        app, headers={"Authorization": f"Bearer {app.state.shell_token}"}) as c:
        yield c


def _fields(client):
    r = client.get("/v1/config/schema")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "fields" in body and isinstance(body["fields"], list)
    return {f["key"]: f for f in body["fields"]}


def test_returns_typed_field_list_with_widgets(client):
    by_key = _fields(client)
    # A representative spread of the widget types the form must render.
    assert by_key["mode"]["widget"] == "select"
    assert by_key["n_ctx"]["widget"] == "number"
    assert by_key["require_auth"]["widget"] == "toggle"
    assert by_key["net_allow"]["widget"] == "list"
    assert by_key["binary_dir"]["widget"] == "folder"


def test_select_field_carries_options_and_current_value(client):
    by_key = _fields(client)
    mode = by_key["mode"]
    assert mode["options"] == ["privacy", "log", "full"]
    # Current value is injected as `default` (here, the config default).
    from localm.config import DEFAULT_CONFIG
    assert mode["default"] == DEFAULT_CONFIG["mode"]


def test_number_field_carries_min_max_and_current_value(client):
    by_key = _fields(client)
    from localm.config import DEFAULT_CONFIG
    n_ctx = by_key["n_ctx"]
    assert n_ctx["min"] == 512
    assert n_ctx["default"] == DEFAULT_CONFIG["n_ctx"]
    port = by_key["port"]
    assert port["min"] == 1 and port["max"] == 65535


def test_reflects_current_config_not_just_static_defaults(client):
    # Persist a non-default value, then the schema must report it as `default`.
    r = client.patch("/v1/config", json={"n_ctx": 8192})
    assert r.status_code == 200, r.text
    by_key = _fields(client)
    assert by_key["n_ctx"]["default"] == 8192


def test_hidden_fields_present_but_marked_hidden(client):
    # plugins_enabled / plugins are real config keys; the schema marks them
    # widget=hidden so the form skips them rather than the server omitting them.
    by_key = _fields(client)
    assert by_key["plugins_enabled"]["widget"] == "hidden"
    assert by_key["plugins"]["widget"] == "hidden"


def test_never_leaks_a_secret_value(client, monkeypatch):
    """A secret field must be advertised (so the form renders a masked input)
    but must NEVER carry a value, even if one is set in the live config."""
    from localm import settings_schema as ss

    # Inject a temporary secret field + a config value for it.
    secret_field = ss.SettingField(
        "fake_secret", ss.Widget.SECRET, "Fake secret",
        group="Security", secret=True)
    monkeypatch.setattr(ss, "CORE_FIELDS", ss.CORE_FIELDS + [secret_field])

    def _cfg_with_secret():
        from localm.config import DEFAULT_CONFIG
        d = dict(DEFAULT_CONFIG)
        d["fake_secret"] = "super-sensitive-token"
        return d

    monkeypatch.setattr("localm.config.load_config", _cfg_with_secret)
    by_key = _fields(client)
    assert by_key["fake_secret"]["widget"] == "secret"
    assert by_key["fake_secret"]["secret"] is True
    # No value is serialized for a secret field.
    assert "default" not in by_key["fake_secret"]


def test_requires_config_read_when_key_set(client):
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        denied = client.get("/v1/config/schema")
        assert denied.status_code == 401
        ok = client.get(
            "/v1/config/schema",
            headers={"Authorization": f"Bearer {SECRET}"})
        assert ok.status_code == 200
        assert "fields" in ok.json()
