# SPDX-License-Identifier: AGPL-3.0-or-later
"""FastAPI auto-docs (/docs, /redoc, /openapi.json) are served only on a loopback
bind. On a network bind they 404 so an unauthenticated remote client cannot
enumerate the API surface (paths + schemas). The decision keys off the CONFIGURED
bind host (app.state.bind_host), never the request peer - portmux relays every
connection through loopback, so the peer always looks like 127.0.0.1.
"""

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

_DOCS = ["/openapi.json", "/docs", "/redoc"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return create_app(None, api_landing=True)


def test_docs_available_on_default_unset_bind(app):
    # bind_host unset (tests / standalone mount) -> treated as loopback -> docs on
    c = TestClient(app)
    for path in _DOCS:
        assert c.get(path).status_code == 200, path


def test_docs_available_on_loopback_bind(app):
    for host in ("127.0.0.1", "localhost", "::1"):
        app.state.bind_host = host
        c = TestClient(app)
        for path in _DOCS:
            assert c.get(path).status_code == 200, f"{host} {path}"


def test_docs_hidden_on_network_bind(app):
    # A genuinely network-exposed bind must not serve the API map to anyone.
    for host in ("0.0.0.0", "192.168.1.50", "10.0.0.7"):
        app.state.bind_host = host
        c = TestClient(app)
        for path in _DOCS:
            r = c.get(path)
            assert r.status_code == 404, f"{host} {path} -> {r.status_code}"


def test_network_bind_still_serves_intended_open_route(app):
    # The gate is scoped to the docs paths only - /health (the one intended
    # unauthenticated route) is unaffected on a network bind.
    app.state.bind_host = "0.0.0.0"
    c = TestClient(app)
    # 503 (no engine) proves the route ran, i.e. it was NOT 404'd by the docs gate
    assert c.get("/health").status_code == 503
