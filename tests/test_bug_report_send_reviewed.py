# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GUI bug report is sent only as the saved text the user was shown.

``POST /api/bug-report`` saves and returns the text with its digest and never
uploads; ``POST /api/bug-report/send`` uploads that saved file only while its
text still has that digest.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import bugreport
from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    uploads = []

    def _fake_upload(title, body, **kw):
        uploads.append((title, body))
        return {"url": "https://github.com/x/localm/issues/5"}

    monkeypatch.setattr(bugreport, "upload_report", _fake_upload)
    app = create_app(_engine())
    hdrs = {"Authorization": f"Bearer {app.state.shell_token}"}
    with TestClient(app) as c:
        yield c, hdrs, uploads, tmp_path


def _save(c, hdrs, **body):
    body.setdefault("description", "it broke")
    r = c.post("/api/bug-report", json=body, headers=hdrs)
    assert r.status_code == 200, r.text
    return r.json()


def test_save_route_refuses_upload_and_saves_nothing(client):
    c, hdrs, uploads, home = client
    r = c.post("/api/bug-report", headers=hdrs,
               json={"description": "send it now", "upload": True})
    assert r.status_code == 400
    assert uploads == []
    assert list(home.glob("bug-reports/*.md")) == []


def test_save_route_returns_the_saved_text_and_its_digest(client):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs, what_happened="the grey square appeared")
    text = Path(data["path"]).read_text(encoding="utf-8")
    assert data["report_markdown"] == text
    assert data["report_sha256"] == bugreport.report_digest(text)
    assert len(data["report_sha256"]) == 64
    assert uploads == []
    assert "uploaded" not in data


def test_send_uploads_exactly_the_saved_text_with_its_own_title(client):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs, description="clicked generate",
                 what_happened="a blank grey square appeared")
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": data["filename"], "sha256": data["report_sha256"]})
    assert r.status_code == 200, r.text
    assert r.json()["uploaded"] is True
    assert r.json()["issue_url"].endswith("/issues/5")
    assert len(uploads) == 1
    title, body = uploads[0]
    assert body == data["report_markdown"]
    assert title == "a blank grey square appeared"


def test_send_refuses_a_report_changed_after_it_was_shown(client):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs)
    path = Path(data["path"])
    path.write_text(data["report_markdown"] + "\nadded after review\n", encoding="utf-8")
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": data["filename"], "sha256": data["report_sha256"]})
    assert r.status_code == 409
    assert uploads == []


def test_send_refuses_a_digest_that_was_not_shown(client):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs)
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": data["filename"], "sha256": "0" * 64})
    assert r.status_code == 409
    assert uploads == []


@pytest.mark.parametrize("body", [
    {"filename": "bug-x.md"},
    {"filename": "bug-x.md", "sha256": ""},
    {"filename": "bug-x.md", "sha256": 7},
], ids=["absent", "empty", "not-a-string"])
def test_send_without_a_digest_is_refused(client, body):
    c, hdrs, uploads, _ = client
    _save(c, hdrs)
    r = c.post("/api/bug-report/send", headers=hdrs, json=body)
    assert r.status_code == 400
    assert uploads == []


@pytest.mark.parametrize("filename", [
    "../config.json", "..\\config.json", "bug-x.md/../../config.json",
    "bug-../x.md", "bug-x.txt", "x/bug-x.md", "C:\\x\\bug-1.md", "/tmp/bug-1.md",
    "bug-.md", "", None, 12, "bug-" + "a" * 200 + ".md", "bug-x.md.md", "bug-xé.md",
])
def test_send_refuses_anything_but_a_saved_report_name(client, filename):
    c, hdrs, uploads, home = client
    (home / "config.json").write_text('{"x": 1}', encoding="utf-8")
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": filename,
                     "sha256": bugreport.report_digest('{"x": 1}')})
    assert r.status_code == 400
    assert uploads == []


def test_send_of_a_missing_report_is_404(client):
    c, hdrs, uploads, _ = client
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": "bug-20990101-000000.md", "sha256": "0" * 64})
    assert r.status_code == 404
    assert uploads == []


def test_a_failed_send_can_be_retried_with_the_same_reviewed_file(client, monkeypatch):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs)
    calls = []

    def _flaky(title, body, **kw):
        calls.append(body)
        if len(calls) == 1:
            raise bugreport.LocalmError("could not reach the bug-report server",
                                        stage="offline_or_dns", hint="You may be offline.")
        return {"url": "https://github.com/x/localm/issues/6"}

    monkeypatch.setattr(bugreport, "upload_report", _flaky)
    ref = {"filename": data["filename"], "sha256": data["report_sha256"]}
    first = c.post("/api/bug-report/send", headers=hdrs, json=ref)
    assert first.status_code == 200
    assert first.json()["uploaded"] is False
    assert first.json()["upload_stage"] == "offline_or_dns"
    second = c.post("/api/bug-report/send", headers=hdrs, json=ref)
    assert second.json()["uploaded"] is True
    assert calls == [data["report_markdown"], data["report_markdown"]]


def test_a_rate_limited_send_is_reported_not_claimed(client, monkeypatch):
    c, hdrs, _, _ = client
    data = _save(c, hdrs)

    def _limited(title, body, **kw):
        raise bugreport.RateLimitedError(12, reason="slow down")

    monkeypatch.setattr(bugreport, "upload_report", _limited)
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": data["filename"], "sha256": data["report_sha256"]})
    assert r.status_code == 200
    body = r.json()
    assert body["uploaded"] is False
    assert body["rate_limited"] is True
    assert body["retry_after"] == 12


def test_a_saved_report_that_is_not_utf8_is_a_read_error(client):
    c, hdrs, uploads, _ = client
    data = _save(c, hdrs)
    Path(data["path"]).write_bytes("# localm bug report: caf\u00e9\n".encode("cp1252"))
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": data["filename"], "sha256": data["report_sha256"]})
    assert r.status_code == 404
    assert uploads == []


@pytest.mark.parametrize("text,title", [
    ("# localm bug report: the grey square\n\nbody", "the grey square"),
    ("# My own heading\nbody", "My own heading"),
    ("no heading at all\n", "user-reported issue"),
    ("# localm bug report:   \n", "user-reported issue"),
    ("# localm bug report: " + "x" * 300, "x" * 120),
])
def test_report_title_from_markdown(text, title):
    assert bugreport.report_title_from_markdown(text) == title


def test_an_upload_that_outruns_its_budget_is_reported_not_claimed(client, monkeypatch):
    import threading

    import localm.inference.routes.admin as admin

    c, hdrs, _, _ = client
    data = _save(c, hdrs)
    release = threading.Event()

    def _stuck(title, body, **kw):
        release.wait(5)
        return {"url": "x"}

    monkeypatch.setattr(bugreport, "upload_report", _stuck)
    monkeypatch.setattr(admin, "_BUG_REPORT_UPLOAD_TIMEOUT_S", 0.2)
    try:
        r = c.post("/api/bug-report/send", headers=hdrs,
                   json={"filename": data["filename"], "sha256": data["report_sha256"]})
    finally:
        release.set()
    assert r.status_code == 200
    assert r.json()["uploaded"] is False
    assert r.json()["upload_stage"] == "upload"
    assert Path(data["path"]).is_file()


def test_a_report_overwritten_by_a_later_save_is_not_sent_as_the_first(client, monkeypatch):
    import time as _time

    c, hdrs, uploads, _ = client
    real = _time.strftime
    monkeypatch.setattr(_time, "strftime", lambda fmt, *a: (
        "20261014-000000" if fmt == "%Y%m%d-%H%M%S" else real(fmt, *a)))
    first = _save(c, hdrs, what_happened="the first report")
    second = _save(c, hdrs, what_happened="a different second report")
    assert first["filename"] == second["filename"]
    r = c.post("/api/bug-report/send", headers=hdrs,
               json={"filename": first["filename"], "sha256": first["report_sha256"]})
    assert r.status_code == 409
    assert uploads == []
