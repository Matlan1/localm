# SPDX-License-Identifier: AGPL-3.0-or-later
"""chat_web_max_lookups: the Settings > Chat ceiling on web lookups per message.

The chat's web loop runs in the GUI (settings-perf.js) and reads this value from
GET /v1/config; 0 means no ceiling. These pin the server half: the key exists
with its default, validates as a whole number in range, round-trips through the
config route, and the default matches the GUI's own fallback."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG

_JS = (Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui"
       / "static" / "app" / "settings-perf.js")


def test_default_is_a_generous_whole_number():
    assert DEFAULT_CONFIG["chat_web_max_lookups"] == 20


def test_gui_fallback_matches_the_config_default():
    js = _JS.read_text(encoding="utf-8")
    m = re.search(r"export const WEB_DEFAULT_MAX_LOOKUPS = (\d+);", js)
    assert m, "settings-perf.js no longer declares WEB_DEFAULT_MAX_LOOKUPS"
    assert int(m.group(1)) == DEFAULT_CONFIG["chat_web_max_lookups"]


def test_the_field_is_in_the_chat_group_and_applies_live():
    field = next(f for f in ss.CORE_FIELDS if f.key == "chat_web_max_lookups")
    assert field.group == "Chat" and field.owner == "chat"
    assert field.applies == ss.Applies.LIVE
    assert (field.min, field.max) == (0, 100)


@pytest.mark.parametrize("value, expected", [(0, 0), ("7", 7), (100, 100)])
def test_valid_values_are_coerced_to_int(value, expected):
    got = ss.validate_update({"chat_web_max_lookups": value})
    assert got == {"chat_web_max_lookups": expected}
    assert isinstance(got["chat_web_max_lookups"], int)


@pytest.mark.parametrize("value", [-1, 101, "many", True])
def test_out_of_range_or_non_numbers_are_rejected(value):
    with pytest.raises(ValueError):
        ss.validate_update({"chat_web_max_lookups": value})


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    app = create_app(None)
    return TestClient(app, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def test_the_config_route_serves_and_saves_it(client):
    assert client.get("/v1/config").json()["chat_web_max_lookups"] == 20
    r = client.patch("/v1/config", json={"chat_web_max_lookups": 5})
    assert client.get("/v1/config").json()["chat_web_max_lookups"] == 5
    assert r.status_code == 200
    bad = client.patch("/v1/config", json={"chat_web_max_lookups": 500})
    assert client.get("/v1/config").json()["chat_web_max_lookups"] == 5
    assert bad.status_code == 400
