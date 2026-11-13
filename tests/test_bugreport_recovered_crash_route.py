# SPDX-License-Identifier: AGPL-3.0-or-later
"""A server that started after a crash of the run before it says so, and names
the report it saved, on GET /api/bug-report/recovered."""

from fastapi.testclient import TestClient

from localm.bugreport import crash_recovery
from localm.inference.http_server import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    app = create_app(None)
    return TestClient(app), {"Authorization": f"Bearer {app.state.shell_token}"}


def test_no_crash_before_this_start(tmp_path, monkeypatch):
    monkeypatch.setattr(crash_recovery, "_RECOVERED_REPORT", None)
    client, hdrs = _client(tmp_path, monkeypatch)
    with client as c:
        assert c.get("/api/bug-report/recovered", headers=hdrs).json() == {"crashed": False}


def test_a_crash_report_filed_at_start_is_named(tmp_path, monkeypatch):
    report = tmp_path / "bug-20261113-080000.md"
    report.write_text("# localm bug report: localm server crashed\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(crash_recovery, "_RECOVERED_REPORT", None)
    monkeypatch.setattr(crash_recovery, "_report_one_crash_marker",
                        lambda *a, **k: report)
    monkeypatch.setattr(crash_recovery, "_all_crash_markers", lambda d: ["m"])
    monkeypatch.setattr(crash_recovery, "_all_crash_stopping_records", lambda d: [])
    assert crash_recovery.check_and_report_prior_crash() == report

    client, hdrs = _client(tmp_path, monkeypatch)
    with client as c:
        body = c.get("/api/bug-report/recovered", headers=hdrs).json()
    assert body["crashed"] is True
    assert body["filename"] == "bug-20261113-080000.md"
    assert body["title"]
    assert str(tmp_path) not in str(body)
