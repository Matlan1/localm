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


def test_response_carries_report_markdown_for_download(monkeypatch):
    """The response includes the saved report's markdown so the GUI can offer a
    browser download for manual sending (a phone/LAN tester cannot open a
    server-side path)."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "downloadable please"},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "downloadable please" in data["report_markdown"]


def test_upload_failure_returns_stage_and_message(monkeypatch):
    """A failed upload is surfaced with the diagnosed stage + friendly message (not
    a false success), and the report file + downloadable markdown are still there."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    from localm import bugreport

    def _boom(*a, **k):
        raise bugreport.LocalmError(
            "could not reach the bug-report server", reason="getaddrinfo failed",
            stage="offline_or_dns", hint="You may be offline.")

    monkeypatch.setattr(bugreport, "upload_report", _boom)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "send this", "upload": True},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True and data["uploaded"] is False
    assert data["upload_stage"] == "offline_or_dns"
    assert data["upload_message"] == "You may be offline."
    assert "send this" in data["report_markdown"]     # still downloadable for retry
    assert Path(data["path"]).is_file()               # file kept for manual send


def test_browser_client_context_lands_in_report(monkeypatch):
    """The GUI attaches a ``client`` block (user agent, page, viewport, recent JS
    console errors). It is rendered into the report; unknown fields are dropped by
    the server-side sanitizer and never reach the file."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={
                "description": "the page went blank after generating an image",
                "client": {
                    "userAgent": "Mozilla/5.0 TestBrowser",
                    "page": "#studio",
                    "viewport": "1280x720",
                    "console": ["TypeError: render is not a function",
                                "x" * 5000],
                    "evilField": "must be dropped",
                },
            },
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    body = Path(r.json()["path"]).read_text(encoding="utf-8")
    assert "## Browser / client" in body
    assert "TestBrowser" in body and "1280x720" in body
    assert "TypeError: render is not a function" in body
    assert "must be dropped" not in body   # sanitizer drops unknown fields
