# SPDX-License-Identifier: AGPL-3.0-or-later
"""The browser plugin's own routes, and the switch in front of them.

Holding the browser capability is not enough on its own: ``browser_enabled``
must also be on. That is the ADR's "scope grants eligibility, a separate switch
grants use" shape, and it is enforced here as well as in the coder tools, so a
GUI caller cannot reach a browser the setting says is off.

These need no browser: the switch is checked before anything is launched.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    _cfg.ensure_dirs()
    from localm.plugins.builtin.browser import plug
    from localm.plugins.engine import attach_engine
    application = FastAPI()
    attach_engine(application)
    application.include_router(plug._router)
    return application


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


class TestTheSwitchGuardsTheRoutes:
    def test_opening_is_refused_while_the_setting_is_off(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/session", json={})
            assert r.status_code == 409, r.text
            assert "switched off" in r.json()["detail"]

    def test_navigating_is_refused_while_the_setting_is_off(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/navigate", json={"url": "https://example.com/"})
            assert r.status_code == 409, r.text

    def test_with_the_setting_on_navigate_reports_no_open_browser(self, app):
        """Past the switch, and refusing for the right reason: nothing is open.
        Without this the 409 above would equally pass on a route that always
        refuses."""
        with TestClient(app) as c:
            _set(browser_enabled=True)
            r = c.post("/api/browser/navigate", json={"url": "https://example.com/"})
            assert r.status_code == 404, r.text
            assert "No browser is open" in r.json()["detail"]

    def test_state_reports_the_switch_without_opening_anything(self, app):
        with TestClient(app) as c:
            body = c.get("/api/browser/state").json()
            assert body == {"open": False, "enabled": False}
            _set(browser_enabled=True)
            assert c.get("/api/browser/state").json()["enabled"] is True

    def test_stopping_when_nothing_is_open_is_not_an_error(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/stop")
            assert r.status_code == 200
            assert r.json() == {"closed": False}


class TestTheManifest:
    def test_it_ships_disabled_and_declares_its_extra(self):
        from localm.plugins.engine import parse_spec
        spec = parse_spec(Path("localm/plugins/builtin/browser"), builtin=True)
        assert spec.default_enabled is False, "a browser must not be on by default"
        assert spec.scope == "browser"
        assert spec.requires_extras == ["browser"]

    def test_it_joins_the_coder_nav_category(self):
        from localm.plugins.engine import parse_spec
        browser = parse_spec(Path("localm/plugins/builtin/browser"), builtin=True)
        coder = parse_spec(Path("localm/plugins/builtin/coder"), builtin=True)
        assert browser.surface.group == "coder"
        assert coder.surface.group == "coder", (
            "the agent must join the category too, or the category has one member "
            "and the browser renders as a flat tab beside it")

    def test_the_manifest_parses_without_warnings(self):
        from localm.plugins.engine import parse_spec
        warns: list = []
        parse_spec(Path("localm/plugins/builtin/browser"), builtin=True,
                   warnings=warns)
        assert warns == [], warns
