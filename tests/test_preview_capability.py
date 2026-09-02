# SPDX-License-Identifier: AGPL-3.0-or-later
"""The preview (artifacts canvas) permission gate surfaced on /api/capabilities.

``preview`` answers ONE question: is THIS caller offered the canvas button.
``gui_preview_enabled`` is the global kill switch; ``gui_preview_owner_only``
then narrows the answer to the owner. Open mode (no key configured) and an ADMIN
key are the owner, exactly as in ``effective_fs_access``.

Scope note, so no reader mistakes this for containment: the renderable block is
already in the caller's own DOM either way, so this decides what the GUI OFFERS,
not what a reply carries.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def prev_app(tmp_path, monkeypatch):
    """GUI stack on a throwaway home, no owner key by default (open mode)."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


def _preview(client, headers=None):
    r = client.get("/api/capabilities", headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()["preview"]


class TestPreviewCapability:
    def test_open_mode_is_offered_the_canvas(self, prev_app):
        with TestClient(prev_app) as c:
            assert _preview(c) is True

    def test_the_kill_switch_withdraws_it_from_the_owner_too(self, prev_app):
        with TestClient(prev_app) as c:
            _set(gui_preview_enabled=False)
            assert _preview(c) is False

    def test_owner_only_still_offers_it_in_open_mode(self, prev_app):
        with TestClient(prev_app) as c:
            _set(gui_preview_owner_only=True)
            assert _preview(c) is True

    def test_owner_only_withdraws_it_from_a_non_owner_key(self, prev_app, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        from localm import auth
        with TestClient(prev_app) as c:
            scoped = auth.create_key("s", [S.CONFIG_READ])["key"]
            _set(gui_preview_owner_only=True)
            assert _preview(c, _hdr(scoped)) is False
            assert _preview(c, _hdr("ownersecret")) is True

    def test_without_owner_only_a_scoped_key_keeps_it(self, prev_app, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        from localm import auth
        with TestClient(prev_app) as c:
            scoped = auth.create_key("s2", [S.CONFIG_READ])["key"]
            assert _preview(c, _hdr(scoped)) is True

    def test_the_kill_switch_beats_owner_only(self, prev_app, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        with TestClient(prev_app) as c:
            _set(gui_preview_enabled=False, gui_preview_owner_only=True)
            assert _preview(c, _hdr("ownersecret")) is False


class TestPreviewSettingsAreOwnerOnly:
    def test_both_preview_fields_are_admin_only(self):
        from localm.settings_schema import CORE_FIELDS
        fields = {f.key: f for f in CORE_FIELDS
                  if f.key in ("gui_preview_enabled", "gui_preview_owner_only")}
        assert set(fields) == {"gui_preview_enabled", "gui_preview_owner_only"}
        for key, f in fields.items():
            assert f.admin_only is True, f"{key} must not be settable by a scoped key"

    def test_defaults_ship_on_for_everyone(self):
        from localm.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["gui_preview_enabled"] is True
        assert DEFAULT_CONFIG["gui_preview_owner_only"] is False
