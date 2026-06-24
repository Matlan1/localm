# SPDX-License-Identifier: AGPL-3.0-or-later
"""R47: the GUI needs a manual "file a bug report" trigger. The CLI has
`localm bug-report`; this is the server endpoint a GUI button posts to. It is
management-gated (same-origin / shell-token + config-write scope) because a report
carries local diagnostics.
"""

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def test_route_registered_and_gated():
    app = create_app(_engine())
    routes = {getattr(r, "path", None): r for r in app.routes}
    assert "/api/bug-report" in routes
    route = routes["/api/bug-report"]
    assert "POST" in route.methods
    assert route.dependencies, "bug-report endpoint must carry an auth dependency"


def test_no_token_refused_in_open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    with TestClient(create_app(_engine())) as c:
        # NEGATIVE: a no-Origin/no-token client must not be able to drive it.
        r = c.post("/api/bug-report", json={"message": "x"})
    assert r.status_code == 403


def test_files_a_report_with_shell_token(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"message": "video generation froze the whole UI"},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    p = Path(data["path"])
    assert p.is_file() and p.suffix == ".md"
    body = p.read_text(encoding="utf-8")
    assert "video generation froze" in body


def test_gui_button_description_payload(monkeypatch):
    """The GUI "Report a bug" button posts ``description`` (+ optional
    ``include_log``); the response carries the filename + maintainer the GUI
    shows. ``description`` and ``message`` are interchangeable aliases."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "The mic does nothing.", "include_log": False},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    assert data["filename"].startswith("bug-") and data["filename"].endswith(".md")
    assert data["maintainer"]
    p = Path(data["path"])
    assert p.is_file()
    assert "The mic does nothing." in p.read_text(encoding="utf-8")


def test_blank_description_rejected(monkeypatch):
    """A blank report (no description/message) must be refused, not filed empty."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "   "},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 400
