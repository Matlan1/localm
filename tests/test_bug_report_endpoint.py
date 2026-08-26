# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server endpoint a GUI "file a bug report" button posts to (the CLI
equivalent is `localm bug-report`). Management-gated (same-origin /
shell-token + config-write scope), because a report carries local diagnostics.
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
        # A no-Origin/no-token client must not be able to drive it.
        r = c.post("/api/bug-report", json={"message": "x"})
    assert r.status_code == 403


def test_save_does_not_run_on_the_event_loop(monkeypatch):
    """Filing a report does a synchronous log read + scrub + file write
    (bugreport.save_user_report), which would stall every concurrent request.
    asyncio.get_running_loop() succeeds only on the event-loop thread and
    raises RuntimeError in a threadpool worker (same technique as
    test_comfy_models_offloaded_638.py), so this is structural rather than
    timed."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import asyncio

    from localm import bugreport

    seen: dict = {}
    real_save = bugreport.save_user_report

    def _probing_save(*a, **kw):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True      # on the event-loop thread
        except RuntimeError:
            seen["on_loop"] = False     # off-loop (threadpool worker)
        return real_save(*a, **kw)

    monkeypatch.setattr(bugreport, "save_user_report", _probing_save)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "must not stall the loop"},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200, r.text
    assert seen.get("on_loop") is False, (
        "file_bug_report_ep called save_user_report ON the event loop: a slow "
        "log digest or disk write would stall every other concurrent request")


def test_upload_does_not_run_on_the_event_loop(monkeypatch):
    """upload_report is a blocking HTTPS POST to the proxy on a 15s socket
    timeout, so an unreachable proxy would freeze every other client for up to
    15s. Same oracle as the sibling test above: asyncio.get_running_loop()
    succeeds only on the event-loop thread, so this is structural rather than
    timed."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import asyncio

    from localm import bugreport

    seen: dict = {}

    def _probing_upload(title, body, **kw):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True      # on the event-loop thread
        except RuntimeError:
            seen["on_loop"] = False     # off-loop (threadpool worker)
        return {"url": "https://example.invalid/issues/1"}

    monkeypatch.setattr(bugreport, "upload_report", _probing_upload)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "must not stall the loop", "upload": True},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200, r.text
    assert r.json().get("uploaded") is True
    assert seen.get("on_loop") is False, (
        "file_bug_report_ep called upload_report ON the event loop: an "
        "unreachable proxy would stall every other concurrent request for the "
        "length of its 15s timeout - and this fires exactly when the user is "
        "already having a problem, which is why they are filing")


def test_a_scrub_failure_does_not_write_or_report_success(monkeypatch):
    """save_user_report SCRUBS before it saves (home paths, secrets). With the
    whole call offloaded through run_in_threadpool_bounded, an exception raised
    inside the closure must reach the caller as a failure, with no report left
    on disk. build_report() calls _scrub_secrets(summary) unconditionally as
    its first line, before save_report() is ever reached, so a raise there
    exercises the whole chain."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    from localm import bugreport
    from localm.config import home_dir

    def _boom(text):
        raise RuntimeError("scrub exploded")

    monkeypatch.setattr(bugreport, "_scrub_secrets", _boom)
    app = create_app(_engine())
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "trigger the scrub failure"},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code != 200, (
        f"a scrub failure must not report success, got {r.status_code}: {r.text}")
    reports_dir = home_dir() / "bug-reports"
    written = list(reports_dir.glob("*.md")) if reports_dir.is_dir() else []
    assert written == [], (
        f"a scrub failure must not leave a report on disk: {written}")


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


def test_what_expected_and_happened_fields_land_in_their_own_sections(monkeypatch):
    """The GUI's two optional fields must render as their OWN
    sections, distinct from ``description`` and from each other - not all
    three collapsed into one duplicated sentence."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={
                "description": "clicked generate on the image tab",
                "what_i_expected": "a picture of a cat",
                "what_happened": "a blank grey square appeared instead",
            },
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    body = Path(r.json()["path"]).read_text(encoding="utf-8")
    assert "## What I expected" in body
    assert "a picture of a cat" in body
    assert "a blank grey square appeared instead" in body
    # The title comes from what-happened, and the upload title below matches it.
    assert body.startswith("# localm bug report: a blank grey square")


def test_blank_description_with_what_happened_still_accepted(monkeypatch):
    """A blank ``description`` is fine as long as ``what_happened`` carries
    the report - the two are independent, optional inputs."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "   ", "what_happened": "it crashed on startup"},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    body = Path(r.json()["path"]).read_text(encoding="utf-8")
    assert "it crashed on startup" in body


def test_gui_upload_does_not_publish_the_edit_disclaimer(monkeypatch):
    """End to end through the REAL route: unlike the other upload tests here,
    bugreport.upload_report is NOT mocked - only the network transport is, so
    this exercises the actual strip logic. The GUI's report_markdown (for the
    download button)
    must KEEP the disclaimer; the bytes that would actually reach GitHub
    must NOT carry it or the maintainer's email."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    from localm import bugreport

    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example/report", None))

    captured = {}

    class _FakeResp:
        status = 201
        def read(self):
            return b'{"url": "https://github.com/Matlan1/localm/issues/42"}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_verified_urlopen(req, timeout=None):
        import json
        captured["body"] = json.loads(req.data.decode("utf-8"))["body"]
        return _FakeResp()

    monkeypatch.setattr("localm.http_ssl.verified_urlopen", _fake_verified_urlopen)

    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={"description": "leaks the disclaimer?", "upload": True},
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["uploaded"] is True

    # Download copy: disclaimer present.
    assert bugreport.MAINTAINER_EMAIL in data["report_markdown"]
    assert "You can edit anything above before sending" in data["report_markdown"]

    # What reached the mocked GitHub transport: disclaimer gone.
    assert "body" in captured, "upload never reached the transport layer"
    assert bugreport.MAINTAINER_EMAIL not in captured["body"]
    assert "You can edit anything above before sending" not in captured["body"]
    assert "leaks the disclaimer?" in captured["body"]   # real content intact


def test_upload_title_prefers_what_happened_over_description(monkeypatch):
    """The uploaded (public GitHub issue) title must match the report body's
    own H1 - both derived from what_happened when present, not the
    (different, older) description-only derivation."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    from localm import bugreport

    captured = {}

    def _fake_upload(title, body, **kw):
        captured["title"] = title
        return {"url": "https://github.com/Matlan1/localm/issues/999"}

    monkeypatch.setattr(bugreport, "upload_report", _fake_upload)
    app = create_app(_engine())
    with TestClient(app) as c:
        r = c.post(
            "/api/bug-report",
            json={
                "description": "clicked generate on the image tab",
                "what_happened": "a blank grey square appeared instead",
                "upload": True,
            },
            headers={"Authorization": f"Bearer {app.state.shell_token}"},
        )
    assert r.status_code == 200
    assert captured["title"].startswith("a blank grey square")


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
