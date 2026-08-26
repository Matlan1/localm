# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI /api/* capability routes must be gated on their scope, not just "any
valid key" (_require_auth). Policy: chat is BASELINE (any valid key may chat);
every OTHER capability is optional and gated by its own scope. The owner key
(admin) and the owner-paired companion imply every scope and are unaffected; only
under-scoped non-owner keys lose access.

Also covers GET /api/capabilities - the baseline endpoint that tells the GUI which
tabs the CURRENT key may show (so a tab the key can't use is never rendered).
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def scoped_app(tmp_path, monkeypatch):
    """Full stack (engine + GUI) on a throwaway home, so app.state.plugin_manager
    exists for /api/capabilities and create_key writes to the throwaway auth.json.
    No owner key by default -> a created scoped key is the only auth in effect."""
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
    attach_engine(app)                       # sets app.state.plugin_manager
    switched = []

    async def switch_model(name):
        switched.append(name)

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model,
               active_model=lambda: switched[-1] if switched else "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


# Routes that must require an EXISTING-key scope. (method, path, body, scope) where
# a key WITHOUT scope -> 403 and a key WITH scope (or owner) -> not 403.
WRITE_ROUTES = [
    ("post", "/api/models/load", {"model": "nope"}, S.MODELS_WRITE),
    ("post", "/api/models/pull", {"spec": "nope"}, S.MODELS_WRITE),
    ("post", "/api/models/remove", {"model": "nope"}, S.MODELS_WRITE),
    ("post", "/api/models/alias", {"model": "nope", "alias": "x"}, S.MODELS_WRITE),
]
READ_ROUTES = [
    ("get", "/api/models", None, S.MODELS_READ),
    ("get", "/api/vram-estimate", None, S.MODELS_READ),
    ("get", "/api/discover/search?q=x", None, S.MODELS_READ),
    ("get", "/api/discover/files?repo=x/y", None, S.MODELS_READ),
]


class TestModelRoutesAreScoped:
    def test_underscoped_key_is_403_on_every_model_route(self, scoped_app):
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]   # no models:* at all
        with TestClient(scoped_app) as c:
            for method, path, body, _ in READ_ROUTES + WRITE_ROUTES:
                r = getattr(c, method)(path, headers=_hdr(narrow),
                                       **({"json": body} if body else {}))
                assert r.status_code == 403, f"{method} {path} should be 403 for an mcp-only key, got {r.status_code}"

    def test_models_read_allows_read_but_not_write(self, scoped_app):
        from localm import auth
        reader = auth.create_key("reader", [S.MODELS_READ])["key"]
        with TestClient(scoped_app) as c:
            for method, path, body, _ in READ_ROUTES:
                r = getattr(c, method)(path, headers=_hdr(reader))
                assert r.status_code != 403, f"models:read should reach {path}, got {r.status_code}"
            for method, path, body, _ in WRITE_ROUTES:
                r = getattr(c, method)(path, headers=_hdr(reader), json=body)
                assert r.status_code == 403, f"models:read must NOT write via {path}"

    def test_models_write_reaches_write_routes(self, scoped_app):
        from localm import auth
        writer = auth.create_key("writer", [S.MODELS_WRITE])["key"]
        with TestClient(scoped_app) as c:
            for method, path, body, _ in WRITE_ROUTES:
                r = getattr(c, method)(path, headers=_hdr(writer), json=body)
                assert r.status_code != 403, f"models:write should reach {path}, got {r.status_code}"

    def test_owner_key_reaches_everything(self, scoped_app, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")     # owner = admin
        with TestClient(scoped_app) as c:
            for method, path, body, _ in READ_ROUTES + WRITE_ROUTES:
                r = getattr(c, method)(path, headers=_hdr("ownersecret"),
                                       **({"json": body} if body else {}))
                assert r.status_code != 403, f"owner must reach {path}, got {r.status_code}"


class TestConfigTierRoutes:
    def test_config_read_reaches_companion_but_not_fs_or_logs(self, scoped_app):
        # The host file browser (/api/fs/dirs) is not config:read-gated: it
        # requires HOST filesystem access (owner, or a key with fs_access=host).
        # Log export is config:WRITE.
        from localm import auth
        reader = auth.create_key("cfgread", [S.CONFIG_READ])["key"]
        with TestClient(scoped_app) as c:
            assert c.get("/api/companion", headers=_hdr(reader)).status_code != 403
            assert c.get("/api/fs/dirs", headers=_hdr(reader)).status_code == 403, \
                "config:read alone must NOT reach the host file browser anymore"
            # log export is config:WRITE (parity with bug-report) -> denied
            r = c.post("/api/logs/export", headers=_hdr(reader), json={"dest": ""})
            assert r.status_code == 403

    def test_mcp_key_denied_fs_companion_logs(self, scoped_app):
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        with TestClient(scoped_app) as c:
            assert c.get("/api/fs/dirs", headers=_hdr(narrow)).status_code == 403
            assert c.get("/api/companion", headers=_hdr(narrow)).status_code == 403
            assert c.post("/api/logs/export", headers=_hdr(narrow),
                          json={"dest": ""}).status_code == 403

    def test_config_write_plus_fs_host_reaches_logs_export(self, scoped_app):
        from localm import auth
        # config:write is PRIVILEGED, so only an owner may mint it
        # (allow_privileged). The route also needs host filesystem access: dest
        # is an arbitrary host directory it mkdir(parents=True)s and copies logs
        # into, so it is gated on the same dial as the /api/fs/dirs picker.
        writer = auth.create_key("cfgwrite", [S.CONFIG_WRITE],
                                 allow_privileged=True, fs_access="host")["key"]
        with TestClient(scoped_app) as c:
            r = c.post("/api/logs/export", headers=_hdr(writer), json={"dest": ""})
            assert r.status_code != 403

    def test_config_write_without_fs_host_denied_logs_export(self, scoped_app, tmp_path):
        """config:write alone is NOT enough: the route writes into a caller-named
        host directory, and its does-it-exist 400 is a directory-existence oracle
        for the whole disk."""
        from localm import auth
        writer = auth.create_key("cfgwrite-nofs", [S.CONFIG_WRITE],
                                 allow_privileged=True)["key"]   # fs_access="none"
        # dest must EXIST or the filesystem assertion is vacuous: export_logs
        # 400s on a missing dest before it mkdirs anything.
        dest = tmp_path / "exfil"
        dest.mkdir()
        with TestClient(scoped_app) as c:
            r = c.post("/api/logs/export", headers=_hdr(writer),
                       json={"dest": str(dest)})
            assert r.status_code == 403
        assert list(dest.iterdir()) == [], "denied export still wrote into dest"


class TestBaselineRoutesStayOpen:
    def test_stats_and_capabilities_are_baseline(self, scoped_app):
        """A zero-extra-capability key (only mcp here) still gets the status bar
        and can learn its own capabilities - both are baseline (any valid key)."""
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        with TestClient(scoped_app) as c:
            assert c.get("/api/stats", headers=_hdr(narrow)).status_code == 200
            assert c.get("/api/capabilities", headers=_hdr(narrow)).status_code == 200

    def test_no_key_still_401s_baseline_when_keystore_configured(self, scoped_app):
        """With a keystore configured, even a baseline route needs SOME valid key
        (baseline = any valid key, not no key)."""
        from localm import auth
        auth.create_key("narrow", [S.MCP])           # configures the keystore
        with TestClient(scoped_app) as c:
            assert c.get("/api/capabilities").status_code == 401
            assert c.get("/api/stats").status_code == 401


class TestCapabilitiesEndpoint:
    def _caps(self, client, key=None):
        r = client.get("/api/capabilities", **({"headers": _hdr(key)} if key else {}))
        assert r.status_code == 200, r.status_code
        return r.json()

    def test_core_flags_track_scopes(self, scoped_app):
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        reader = auth.create_key("mread", [S.MODELS_READ])["key"]
        cfg = auth.create_key("cread", [S.CONFIG_READ])["key"]
        pls = auth.create_key("pread", [S.PLUGINS_READ])["key"]
        with TestClient(scoped_app) as c:
            n = self._caps(c, narrow)["core"]
            assert n == {"chat": True, "models": False,
                         "plugins": False, "settings": False}
            assert self._caps(c, reader)["core"]["models"] is True
            assert self._caps(c, reader)["core"]["settings"] is False
            assert self._caps(c, cfg)["core"]["settings"] is True
            assert self._caps(c, pls)["core"]["plugins"] is True

    def test_owner_sees_all_core_tabs(self, scoped_app, monkeypatch):
        auth_key = "ownersecret"
        monkeypatch.setenv("LOCALM_API_KEY", auth_key)
        with TestClient(scoped_app) as c:
            caps = self._caps(c, auth_key)
            assert caps["core"] == {"chat": True, "models": True,
                                    "plugins": True, "settings": True}
            assert caps["open"] is False

    def test_open_mode_grants_all(self, scoped_app):
        """No key configured -> open/dev mode -> every tab granted, open=True."""
        with TestClient(scoped_app) as c:
            caps = self._caps(c)            # no auth header, no keystore
            assert caps["open"] is True
            assert all(caps["core"].values())

    def test_plugins_are_scope_filtered(self, scoped_app):
        """The chat plugin (scope 'chat') is listed only for a key that grants
        chat; an mcp-only key does not see it in its capability plugin list."""
        from localm import auth
        narrow = auth.create_key("narrow", [S.MCP])["key"]
        chatter = auth.create_key("chatter", [S.CHAT])["key"]
        with TestClient(scoped_app) as c:
            def names(key):
                return {p["name"] for p in self._caps(c, key)["plugins"]}
            assert "chat" not in names(narrow)
            assert "chat" in names(chatter)
