# SPDX-License-Identifier: AGPL-3.0-or-later
"""An API-only server sends a browser at / to the API docs, and a GUI mounted
onto it later serves / itself."""

from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.plugins.gui.web import attach_gui


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    return hs.create_app(None, api_landing=True)


def test_an_api_only_server_redirects_a_browser_to_the_docs(tmp_path, monkeypatch):
    with TestClient(_app(tmp_path, monkeypatch)) as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_a_gui_mounted_later_serves_the_root(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: "")
    with TestClient(app) as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 200, r.status_code
    assert "text/html" in r.headers["content-type"]
