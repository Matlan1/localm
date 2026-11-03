# SPDX-License-Identifier: AGPL-3.0-or-later
"""The read-only issues tracker: configured-endpoint gating, the record of the
reports this install filed, and the per-number lookups through the proxy."""

import json

import pytest

from localm import issue_tracker
from tests._net_config import set_config


def _opener(payload, status=200):
    def op(method, url, data, headers, timeout):
        op.url = url
        op.token = headers.get("X-Localm-Token")
        return status, json.dumps(payload).encode("utf-8")
    return op


def test_available(monkeypatch):
    set_config(monkeypatch, lambda: {})
    assert issue_tracker.available() is False
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    assert issue_tracker.available() is True


def test_get_issue(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    op = _opener({"ok": True, "issue": {"number": 12, "state": "closed", "title": "t"}})
    out = issue_tracker.get_issue(12, opener=op)
    assert out["number"] == 12 and out["state"] == "closed"
    assert "/issues?number=12" in op.url


def test_get_issue_bad_number_raises(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        issue_tracker.get_issue("xyz")


def _count_proxy_calls(monkeypatch):
    calls = []

    def op(method, url, data, headers, timeout):
        calls.append(url)
        return 200, json.dumps({"issues": [], "issue": None}).encode("utf-8")

    monkeypatch.setattr("localm._proxy._default_opener", op)
    return calls


@pytest.mark.parametrize("how", ["config", "env"])
def test_net_mode_off_makes_no_request(monkeypatch, how):
    cfg = {"bugreport_upload_url": "https://w"}
    if how == "config":
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        cfg["net_mode"] = "off"
    else:
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
    set_config(monkeypatch, lambda: dict(cfg))
    calls = _count_proxy_calls(monkeypatch)
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as e2:
        issue_tracker.get_issue(3)
    assert calls == []
    assert "net_mode = off" in e2.value.summary


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_net_mode_other_than_off_still_requests(monkeypatch, mode):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w", "net_mode": mode})
    calls = _count_proxy_calls(monkeypatch)
    issue_tracker.get_issue(3)
    assert len(calls) == 1


@pytest.fixture
def fresh_record(monkeypatch):
    monkeypatch.setattr(issue_tracker, "_SESSION_ONLY", [])
    monkeypatch.delenv("LOCALM_MODE", raising=False)


def _mode(monkeypatch, mode):
    monkeypatch.setenv("LOCALM_MODE", mode)


def _record_file():
    return issue_tracker._record_path()


def test_filed_report_is_recorded_on_disk(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    assert issue_tracker.record_filed(41, "it broke") is True
    assert _record_file().is_file()
    assert issue_tracker.recorded() == [{"number": 41, "title": "it broke"}]
    assert issue_tracker.persists() is True


def test_privacy_mode_writes_nothing_and_keeps_the_number_in_memory(monkeypatch, fresh_record):
    _mode(monkeypatch, "privacy")
    assert issue_tracker.record_filed(42, "private one") is True
    assert not _record_file().exists()
    assert issue_tracker.recorded() == [{"number": 42, "title": "private one"}]
    assert issue_tracker.persists() is False


def test_unresolvable_mode_is_privacy(monkeypatch, fresh_record):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr("localm.audit.effective_mode", boom)
    assert issue_tracker.record_filed(43, "t") is True
    assert not _record_file().exists()


def test_invalid_numbers_are_not_recorded(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    for bad in (None, "", 0, -3, True, "12", 1.5):
        assert issue_tracker.record_filed(bad, "t") is False
    assert issue_tracker.recorded() == []


def test_record_dedupes_and_caps(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    issue_tracker.record_filed(5, "a")
    issue_tracker.record_filed(5, "a again")
    assert [e["number"] for e in issue_tracker.recorded()] == [5]
    top = 100 + issue_tracker.MAX_RECORDED + 4
    for n in range(100, top + 1):
        issue_tracker.record_filed(n, "x")
    got = issue_tracker.recorded()
    assert len(got) == issue_tracker.MAX_RECORDED
    assert got[-1]["number"] == top


def test_malformed_record_is_read_as_empty_and_logged(monkeypatch, fresh_record, caplog):
    _mode(monkeypatch, "log")
    _record_file().parent.mkdir(parents=True, exist_ok=True)
    _record_file().write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert issue_tracker.recorded() == []
    assert "not valid JSON" in caplog.text


def test_unwritable_record_is_reported_false_not_raised(monkeypatch, fresh_record, caplog):
    _mode(monkeypatch, "log")
    _record_file().parent.mkdir(parents=True, exist_ok=True)
    _record_file().mkdir()
    with caplog.at_level("WARNING"):
        assert issue_tracker.record_filed(7, "t") is False
    assert "could not record filed issue #7" in caplog.text


def test_upload_report_records_the_issue_it_filed(monkeypatch, fresh_record):
    from localm.bugreport import upload_report
    _mode(monkeypatch, "log")

    def op(u, d, h, t):
        return 201, json.dumps({"ok": True, "url": "https://x/9", "number": 9})
    upload_report("a title", "body", url="https://w", token="t", opener=op)
    assert issue_tracker.recorded() == [{"number": 9, "title": "a title"}]


def test_upload_report_without_a_number_records_nothing(monkeypatch, fresh_record):
    from localm.bugreport import upload_report
    _mode(monkeypatch, "log")

    def op(u, d, h, t):
        return 200, json.dumps({"ok": True})
    upload_report("a title", "body", url="https://w", token="t", opener=op)
    assert issue_tracker.recorded() == []


def test_failed_upload_records_nothing(monkeypatch, fresh_record):
    from localm.bugreport import LocalmError, upload_report
    _mode(monkeypatch, "log")

    def op(u, d, h, t):
        return 500, "boom"
    with pytest.raises(LocalmError):
        upload_report("a title", "body", url="https://w", token="t", opener=op)
    assert issue_tracker.recorded() == []


def _stand_in(monkeypatch, issues):
    """Local stand-in for the proxy: GET /issues?number=N."""
    calls = []

    def op(method, url, data, headers, timeout):
        calls.append(url)
        n = int(url.split("number=")[1])
        if n == 666:
            return 500, b"upstream exploded"
        return 200, json.dumps({"ok": True, "issue": issues.get(n)}).encode("utf-8")

    monkeypatch.setattr("localm._proxy._default_opener", op)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    return calls


def test_filed_status_looks_up_only_recorded_numbers_one_by_one(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    calls = _stand_in(monkeypatch, {
        1: {"number": 1, "title": "one", "state": "open", "html_url": "u1"},
        2: {"number": 2, "title": "two", "state": "closed",
            "closed_at": "2026-10-01T00:00:00Z", "html_url": "u2"},
        3: {"number": 3, "title": "NOT OURS", "state": "open"}})
    issue_tracker.record_filed(1, "one")
    issue_tracker.record_filed(2, "two")
    rows = issue_tracker.filed_status()
    assert [r["number"] for r in rows] == [2, 1]
    assert rows[0]["state"] == "closed" and rows[0]["closed_at"].startswith("2026")
    assert rows[1]["state"] == "open"
    assert sorted(c.rsplit("/", 1)[1] for c in calls) == ["issues?number=1", "issues?number=2"]
    assert not any("state=" in c or "per_page" in c for c in calls)


def test_filed_status_with_nothing_recorded_makes_no_request(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    calls = _stand_in(monkeypatch, {})
    assert issue_tracker.filed_status() == []
    assert calls == []


def test_lookup_failure_is_shown_per_row_never_as_closed(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    _stand_in(monkeypatch, {1: {"number": 1, "title": "ok", "state": "open"}})
    issue_tracker.record_filed(1, "ok")
    issue_tracker.record_filed(666, "server error")
    issue_tracker.record_filed(404, "gone")
    rows = {r["number"]: r for r in issue_tracker.filed_status()}
    assert rows[1]["state"] == "open" and "error" not in rows[1]
    assert "state" not in rows[666] and rows[666]["error"]
    assert rows[666]["title"] == "server error"
    assert rows[404]["error"] == "not found" and "state" not in rows[404]


def test_net_mode_off_gives_error_rows_and_sends_nothing(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    calls = _count_proxy_calls(monkeypatch)
    issue_tracker.record_filed(8, "t")
    rows = issue_tracker.filed_status()
    assert calls == []
    assert "net_mode = off" in rows[0]["error"]


def _client(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(issue_tracker, "endpoint", lambda: ("https://w", "tok"))
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)
    return app, TestClient(app)


def test_issues_route_lists_only_recorded_reports(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    calls = _stand_in(monkeypatch, {
        5: {"number": 5, "title": "mine", "state": "open", "html_url": "u"},
        6: {"number": 6, "title": "NOT OURS", "state": "open"}})
    issue_tracker.record_filed(5, "mine")
    app, c = _client(monkeypatch)
    with c:
        r = c.get("/api/issues",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    data = r.json()
    assert [i["number"] for i in data["issues"]] == [5]
    assert data["persisted"] is True and data["issues_url"].startswith("https://github.com/")
    assert len(calls) == 1 and calls[0].endswith("/issues?number=5")


def test_issues_route_under_net_mode_off_reports_and_sends_nothing(monkeypatch, fresh_record):
    _mode(monkeypatch, "log")
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    calls = _count_proxy_calls(monkeypatch)
    issue_tracker.record_filed(5, "mine")
    app, c = _client(monkeypatch)
    with c:
        r = c.get("/api/issues",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert calls == []
    data = r.json()
    assert data["available"] is True
    assert "net_mode = off" in data["issues"][0]["error"]
    assert "state" not in data["issues"][0]
