# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-demand GUI surface mount: localm/inference/http_server.py
``mount_gui_surface`` + ``POST /v1/surfaces/gui``.

An ``api``-mode instance (``localm serve``) serves only /v1; a later ``localm
gui`` in the same dir asks it to mount the GUI surface live - one process, no
second model load. These pin: the mount is gated (this instance's attach token
OR an owner API key), idempotent, and actually adds the GUI routes; and that the
gate refuses an unauthenticated / wrong-credential caller. The route exposes the
coder agent, so the negative cases are the point.
"""

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app, mount_gui_surface


def _api_app(tmp_path, instance_token="inst-secret-token"):
    """A fresh api-mode app (no engine) wired as advertise() would: an instance
    id, token, and bind coordinates on app.state, surface mode 'api'."""
    app = create_app(None)
    app.state.instance_id = "iid-test"
    app.state.instance_token = instance_token
    app.state.instance_mode = "api"
    app.state.instance_port = 8642
    app.state.instance_scheme = "http"
    app.state.bind_host = "127.0.0.1"
    return app


# ------------------------------------------------------------------ #
#  mount_gui_surface (unit)                                          #
# ------------------------------------------------------------------ #

class TestMountGuiSurfaceUnit:
    def test_mounts_then_is_idempotent_without_duplicating_routes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        app = _api_app(tmp_path)
        # NEG: the GUI routes are absent on a bare api app.
        paths = {getattr(r, "path", None) for r in app.router.routes}
        assert "/api/models" not in paths
        assert getattr(app.state, "gui_mounted", False) is False

        assert mount_gui_surface(app) is True
        paths_after = [getattr(r, "path", None) for r in app.router.routes]
        assert "/api/models" in paths_after
        assert app.state.gui_mounted is True
        assert app.state.instance_mode == "full"
        n_models_routes = paths_after.count("/api/models")

        # Second call: already mounted -> no-op, no duplicate routes.
        assert mount_gui_surface(app) is False
        paths_again = [getattr(r, "path", None) for r in app.router.routes]
        assert paths_again.count("/api/models") == n_models_routes

    def test_full_instance_is_not_remounted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        app = _api_app(tmp_path)
        app.state.gui_mounted = True   # a 'full' instance already has the GUI
        assert mount_gui_surface(app) is False

    def test_attach_failure_rolls_back_the_mounted_flag(self, tmp_path, monkeypatch):
        """If attach_gui raises, the claimed gui_mounted flag is rolled back so a
        later real attempt can still mount (no permanently-wedged surface)."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        app = _api_app(tmp_path)
        import localm.plugins.gui.web as web

        def _boom(*a, **k):
            raise RuntimeError("attach failed")

        monkeypatch.setattr(web, "attach_gui", _boom)
        with pytest.raises(RuntimeError):
            mount_gui_surface(app)
        assert getattr(app.state, "gui_mounted", False) is False

    def test_missing_port_raises_rather_than_dialling_none(self, tmp_path, monkeypatch):
        """No bind port on app.state -> a loud 500, not a broken
        'http://127.0.0.1:None/v1' self-url. The flag must not be claimed."""
        from fastapi import HTTPException
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        app = _api_app(tmp_path)
        app.state.instance_port = None
        with pytest.raises(HTTPException) as ei:
            mount_gui_surface(app)
        assert ei.value.status_code == 500
        assert getattr(app.state, "gui_mounted", False) is False


# ------------------------------------------------------------------ #
#  POST /v1/surfaces/gui (integration, open mode)                   #
# ------------------------------------------------------------------ #

class TestSurfaceEndpointOpenMode:
    @pytest.fixture(autouse=True)
    def _open_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    def test_unauthenticated_post_is_refused(self, tmp_path):
        app = _api_app(tmp_path)
        shell = getattr(app.state, "shell_token", None)
        h = {"Authorization": f"Bearer {shell}"} if shell else {}
        with TestClient(app) as client:
            assert client.get("/api/models", headers=h).status_code == 404   # not mounted yet
            r = client.post("/v1/surfaces/gui")                    # no token
            assert r.status_code == 403

    def test_wrong_token_is_refused(self, tmp_path):
        app = _api_app(tmp_path, instance_token="the-real-token")
        shell = getattr(app.state, "shell_token", None)
        h = {"Authorization": f"Bearer {shell}"} if shell else {}
        with TestClient(app) as client:
            r = client.post("/v1/surfaces/gui",
                            headers={"Authorization": "Bearer not-the-token"})
            assert r.status_code == 403
            assert client.get("/api/models", headers=h).status_code == 404   # still not mounted

    def test_instance_token_mounts_then_routes_appear(self, tmp_path):
        app = _api_app(tmp_path, instance_token="the-real-token")
        shell = getattr(app.state, "shell_token", None)
        h = {"Authorization": f"Bearer {shell}"} if shell else {}
        with TestClient(app) as client:
            r = client.post("/v1/surfaces/gui",
                            headers={"Authorization": "Bearer the-real-token"})
            assert r.status_code == 200
            assert r.json()["status"] == "mounted"
            assert r.json()["mode"] == "full"
            # The GUI surface is now live on the same process.
            assert client.get("/api/models", headers=h).status_code == 200
            assert "text/html" in client.get("/").headers.get("content-type", "")
            assert client.get("/whoami").json()["mode"] == "full"

    def test_second_mount_reports_already_mounted(self, tmp_path):
        app = _api_app(tmp_path, instance_token="tok")
        with TestClient(app) as client:
            h = {"Authorization": "Bearer tok"}
            assert client.post("/v1/surfaces/gui", headers=h).json()["status"] == "mounted"
            assert client.post("/v1/surfaces/gui", headers=h).json()["status"] == "already_mounted"


# ------------------------------------------------------------------ #
#  POST /v1/surfaces/gui (integration, protected mode)              #
# ------------------------------------------------------------------ #

class TestSurfaceEndpointProtectedMode:
    @pytest.fixture(autouse=True)
    def _protected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")   # owner = ADMIN
        monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    def test_owner_key_mounts(self, tmp_path):
        app = _api_app(tmp_path, instance_token="inst-tok")
        with TestClient(app) as client:
            r = client.post("/v1/surfaces/gui",
                            headers={"Authorization": "Bearer ownersecret"})
            assert r.status_code == 200
            assert client.get("/api/models",
                              headers={"Authorization": "Bearer ownersecret"}).status_code == 200

    def test_instance_token_still_mounts_under_a_key(self, tmp_path):
        app = _api_app(tmp_path, instance_token="inst-tok")
        with TestClient(app) as client:
            r = client.post("/v1/surfaces/gui",
                            headers={"Authorization": "Bearer inst-tok"})
            assert r.status_code == 200

    def test_invalid_key_is_refused(self, tmp_path):
        app = _api_app(tmp_path, instance_token="inst-tok")
        with TestClient(app) as client:
            r = client.post("/v1/surfaces/gui",
                            headers={"Authorization": "Bearer wrongkey"})
            assert r.status_code == 403


# ------------------------------------------------------------------ #
#  gui CLI client: _mount_remote_gui (the attaching side)           #
# ------------------------------------------------------------------ #

class TestMountRemoteGuiClient:
    def test_posts_with_the_attach_token_and_maps_200_to_true(self, monkeypatch):
        import requests
        from localm.plugins.gui import cli
        seen = {}

        class _Resp:
            status_code = 200

        def _fake_post(url, headers=None, timeout=None, verify=None):
            seen["url"] = url
            seen["headers"] = headers
            return _Resp()

        monkeypatch.setattr(requests, "post", _fake_post)
        ok = cli._mount_remote_gui(
            {"scheme": "https", "port": 8651, "token": "the-token"})
        assert ok is True
        assert seen["url"] == "https://127.0.0.1:8651/v1/surfaces/gui"
        assert seen["headers"]["Authorization"] == "Bearer the-token"

    def test_non_200_maps_to_false(self, monkeypatch):
        import requests
        from localm.plugins.gui import cli

        class _Resp:
            status_code = 500

        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: _Resp())
        assert cli._mount_remote_gui(
            {"scheme": "http", "port": 8642, "token": "t"}) is False

    def test_network_error_maps_to_false(self, monkeypatch):
        import requests
        from localm.plugins.gui import cli

        def _boom(*a, **k):
            raise requests.RequestException("refused")

        monkeypatch.setattr(requests, "post", _boom)
        assert cli._mount_remote_gui(
            {"scheme": "http", "port": 8642, "token": "t"}) is False

    def test_missing_token_or_port_short_circuits(self):
        from localm.plugins.gui import cli
        assert cli._mount_remote_gui({"scheme": "http", "port": 8642}) is False
        assert cli._mount_remote_gui({"scheme": "http", "token": "t"}) is False
