# SPDX-License-Identifier: AGPL-3.0-or-later
"""A background job is bound to the key that created it. The shared JobManager
bus keeps /api/jobs/{id}/events and /cancel baseline (any valid key), so they are
gated by OWNER-binding instead: only the creating key, or an admin/owner, may
stream or cancel a job. A non-owner gets a 404, never a 403 that would confirm
the unguessable id exists."""

import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S
from localm.plugins.gui.jobs import Job
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def scoped_app(tmp_path, monkeypatch):
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
               switch_model=lambda n: None, active_model=lambda: "model-a")
    return app


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _inject(app, owner, status="running"):
    """Register a job in the manager directly (no subprocess/thread) with a
    pre-queued 'end' event so the SSE stream returns promptly."""
    j = Job(id=uuid.uuid4().hex[:12], kind="test", argv=[], owner=owner)
    j.status = status
    j.push({"type": "end", "status": "done"})
    app.state.jobs._jobs[j.id] = j
    return j


class TestJobOwnerBinding:
    def test_noncreator_gets_404_on_events_and_cancel(self, scoped_app):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        b = auth.create_key("B", [S.MODELS_READ])["key"]   # same scopes, different key
        job = _inject(scoped_app, owner=auth._hash_key(a))
        with TestClient(scoped_app) as c:
            assert c.get(f"/api/jobs/{job.id}/events",
                         headers=_hdr(b)).status_code == 404
            assert c.post(f"/api/jobs/{job.id}/cancel",
                          headers=_hdr(b)).status_code == 404

    def test_creator_reaches_its_own_job(self, scoped_app):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(scoped_app, owner=auth._hash_key(a))
        with TestClient(scoped_app) as c:
            assert c.get(f"/api/jobs/{job.id}/events",
                         headers=_hdr(a)).status_code == 200
            assert c.post(f"/api/jobs/{job.id}/cancel",
                          headers=_hdr(a)).status_code == 200

    def test_owner_admin_bypasses(self, scoped_app, monkeypatch):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(scoped_app, owner=auth._hash_key(a))
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")     # owner = admin
        with TestClient(scoped_app) as c:
            assert c.get(f"/api/jobs/{job.id}/events",
                         headers=_hdr("ownersecret")).status_code == 200
            assert c.post(f"/api/jobs/{job.id}/cancel",
                          headers=_hdr("ownersecret")).status_code == 200

    def test_unowned_job_is_reachable_by_any_key(self, scoped_app):
        """A job created in open mode (owner None) stays unrestricted."""
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(scoped_app, owner=None)
        with TestClient(scoped_app) as c:
            assert c.post(f"/api/jobs/{job.id}/cancel",
                          headers=_hdr(a)).status_code == 200

    def test_cookie_auth_same_key_matches_owner(self, scoped_app):
        """The GUI authenticates by an OPAQUE session cookie minted at login; the
        principal id must be identical for the same key whether it arrives via the
        Authorization header or via a session cookie (job ownership parity)."""
        from localm import auth, sessions
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        job = _inject(scoped_app, owner=auth._hash_key(a))
        # A session minted from key `a` records _hash_key(a) as its owning
        # principal; the cookie value is the OPAQUE session id, never the key.
        sid = sessions.create(scopes={S.MODELS_READ}, key_hash=auth._hash_key(a),
                              fs_access="none")
        with TestClient(scoped_app) as c:
            assert sid != a                    # the raw key is never the cookie value
            c.cookies.set("localm_session", sid)   # GUI cookie auth (safe GET, no CSRF)
            assert c.get(f"/api/jobs/{job.id}/events").status_code == 200

    def test_missing_job_is_404(self, scoped_app):
        from localm import auth
        a = auth.create_key("A", [S.MODELS_READ])["key"]
        with TestClient(scoped_app) as c:
            assert c.get("/api/jobs/does-not-exist/events",
                         headers=_hdr(a)).status_code == 404


class TestPrincipalId:
    def test_stable_and_distinct_and_none_without_token(self, scoped_app):
        """principal_id is the keystore hash: stable per key, distinct across keys,
        None when no token is presented (open-mode / anonymous job creation)."""
        from starlette.requests import Request
        from localm import auth
        from localm.inference.http_server import principal_id

        def _req(headers):
            raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
            return Request({"type": "http", "headers": raw, "method": "GET",
                            "path": "/", "query_string": b""})

        a = auth.create_key("A", [S.MODELS_READ])["key"]
        b = auth.create_key("B", [S.MODELS_READ])["key"]
        assert principal_id(_req({"authorization": f"Bearer {a}"})) == auth._hash_key(a)
        assert (principal_id(_req({"authorization": f"Bearer {a}"}))
                != principal_id(_req({"authorization": f"Bearer {b}"})))
        assert principal_id(_req({})) is None
