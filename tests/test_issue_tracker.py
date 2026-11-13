# SPDX-License-Identifier: AGPL-3.0-or-later
"""The read-only issues tracker: configured-endpoint gating, the record of the
reports this install filed, and the per-number lookups through the proxy."""

import json
import socket
import threading
import time

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
    assert e2.value.stage == "network_off"
    from localm import config
    assert e2.value.reason == config.NET_OFF_REMEDY


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


def test_a_malformed_record_is_not_overwritten_by_a_new_filing(monkeypatch, fresh_record, caplog):
    _mode(monkeypatch, "log")
    _record_file().parent.mkdir(parents=True, exist_ok=True)
    _record_file().write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert issue_tracker.record_filed(9, "t") is False
    assert _record_file().read_text(encoding="utf-8") == "{not json"
    assert "could not record filed issue #9" in caplog.text


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


def test_deeply_nested_record_is_read_as_empty_and_logged(monkeypatch, fresh_record, caplog):
    _mode(monkeypatch, "log")
    _record_file().parent.mkdir(parents=True, exist_ok=True)
    _record_file().write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert issue_tracker.recorded() == []
    assert "not valid JSON" in caplog.text


@pytest.mark.parametrize("content", ['{"issues": 5}', '{"issues": "abc"}', '{"issues": {"number": 1}}',
                                     '{"issues": null}', '{}', '[1, 2]', '5', 'null'])
def test_a_record_without_an_issue_list_is_read_as_empty_and_logged(
        monkeypatch, fresh_record, caplog, content):
    _mode(monkeypatch, "log")
    _record_file().parent.mkdir(parents=True, exist_ok=True)
    _record_file().write_text(content, encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert issue_tracker.recorded() == []
    assert "has no list of issues" in caplog.text
    assert issue_tracker.record_filed(50, "after") is False
    assert _record_file().read_text(encoding="utf-8") == content


def test_a_failed_write_leaves_the_record_whole(monkeypatch, fresh_record, caplog):
    import os
    _mode(monkeypatch, "log")
    issue_tracker.record_filed(1, "one")
    issue_tracker.record_filed(2, "two")
    before = _record_file().read_bytes()
    real_write = os.write

    def half_then_die(fd, data):
        real_write(fd, bytes(data)[:len(data) // 2])
        raise OSError(28, "No space left on device")
    with monkeypatch.context() as m, caplog.at_level("WARNING"):
        m.setattr("os.write", half_then_die)
        assert issue_tracker.record_filed(3, "three") is False
    assert _record_file().read_bytes() == before
    assert "could not record filed issue #3" in caplog.text


def test_the_replace_is_the_only_moment_the_record_changes(monkeypatch, fresh_record):
    import os
    _mode(monkeypatch, "log")
    issue_tracker.record_filed(1, "one")
    before = _record_file().read_bytes()
    real_replace = os.replace
    seen = []

    def checking_replace(src, dst):
        seen.append(_record_file().read_bytes())
        real_replace(src, dst)
    monkeypatch.setattr("os.replace", checking_replace)
    assert issue_tracker.record_filed(2, "two") is True
    assert seen == [before]
    assert [e["number"] for e in issue_tracker.recorded()] == [1, 2]


def _record_numbers(monkeypatch, count):
    _mode(monkeypatch, "log")
    for n in range(1, count + 1):
        assert issue_tracker.record_filed(n, f"report {n}") is True


def test_the_budget_gives_the_first_lookup_its_full_timeout():
    assert issue_tracker.FILED_STATUS_BUDGET_S >= issue_tracker.LOOKUP_TIMEOUT_S


def test_rows_past_the_deadline_are_not_requested_and_say_so(monkeypatch, fresh_record):
    _record_numbers(monkeypatch, issue_tracker.MAX_LISTED)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 0.5)
    calls = []

    def slow(method, url, data, headers, timeout):
        calls.append(url)
        time.sleep(0.2)
        n = int(url.split("number=")[1])
        return 200, json.dumps({"issue": {"number": n, "state": "open"}}).encode("utf-8")
    monkeypatch.setattr("localm._proxy._default_opener", slow)
    rows = issue_tracker.filed_status()
    assert [r["number"] for r in rows] == list(range(issue_tracker.MAX_LISTED, 0, -1))
    assert rows[0]["state"] == "open" and "error" not in rows[0]
    assert rows[-1] == {"number": 1, "title": "report 1", "error": issue_tracker.NOT_CHECKED}
    time.sleep(1.0)
    assert 1 <= len(calls) <= 4


def test_a_lookup_that_never_returns_cannot_hold_the_answer(monkeypatch, fresh_record):
    _record_numbers(monkeypatch, 3)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 0.3)
    release = threading.Event()

    def stuck(method, url, data, headers, timeout):
        release.wait(20)
        return 200, b'{"issue": {"number": 3, "state": "open"}}'
    monkeypatch.setattr("localm._proxy._default_opener", stuck)
    started = time.monotonic()
    try:
        rows = issue_tracker.filed_status()
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert elapsed < 5
    assert [r.get("error") for r in rows] == [issue_tracker.NOT_CHECKED] * 3
    assert all("state" not in r for r in rows)


def test_a_late_answer_does_not_change_the_returned_rows(monkeypatch, fresh_record):
    _record_numbers(monkeypatch, 1)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 0.2)
    answered = threading.Event()

    def late(method, url, data, headers, timeout):
        time.sleep(0.5)
        answered.set()
        return 200, b'{"issue": {"number": 1, "state": "open"}}'
    monkeypatch.setattr("localm._proxy._default_opener", late)
    rows = issue_tracker.filed_status()
    assert answered.wait(5)
    time.sleep(0.1)
    assert rows == [{"number": 1, "title": "report 1", "error": issue_tracker.NOT_CHECKED}]


def test_each_lookup_gets_only_the_time_that_is_left(monkeypatch, fresh_record):
    _record_numbers(monkeypatch, 2)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 4.0)
    timeouts = []

    def op(method, url, data, headers, timeout):
        timeouts.append(timeout)
        return 200, b'{"issue": null}'
    monkeypatch.setattr("localm._proxy._default_opener", op)
    issue_tracker.filed_status()
    assert len(timeouts) == 2
    assert all(0 < t <= 4.0 for t in timeouts)
    assert timeouts[1] <= timeouts[0]
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 100.0)
    timeouts.clear()
    issue_tracker.filed_status()
    assert timeouts == [issue_tracker.LOOKUP_TIMEOUT_S] * 2


def test_a_failed_row_keeps_its_error_beside_rows_past_the_deadline(monkeypatch, fresh_record):
    _record_numbers(monkeypatch, 3)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 0.4)

    def op(method, url, data, headers, timeout):
        if url.endswith("number=3"):
            return 502, b"deleted upstream"
        if url.endswith("number=2"):
            return 200, b'{"issue": {"number": 2, "state": "closed"}}'
        time.sleep(1)
        return 200, b'{"issue": {"number": 1, "state": "open"}}'
    monkeypatch.setattr("localm._proxy._default_opener", op)
    rows = {r["number"]: r for r in issue_tracker.filed_status()}
    assert rows[3]["error"] == "the localm proxy returned an error"
    assert rows[2]["state"] == "closed"
    assert rows[1]["error"] == issue_tracker.NOT_CHECKED


def test_an_unexpected_lookup_failure_stays_on_its_row(monkeypatch, fresh_record, caplog):
    _record_numbers(monkeypatch, 2)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})

    def op(method, url, data, headers, timeout):
        if url.endswith("number=2"):
            raise ValueError("bad status line")
        return 200, b'{"issue": {"number": 1, "state": "open"}}'
    monkeypatch.setattr("localm._proxy._default_opener", op)
    with caplog.at_level("WARNING"):
        rows = {r["number"]: r for r in issue_tracker.filed_status()}
    assert rows[2]["error"] == "the lookup failed" and "state" not in rows[2]
    assert rows[1]["state"] == "open"
    assert "issue #2" in caplog.text


@pytest.fixture
def silent_proxy():
    """A local port that takes connections and never answers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(64)
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
    finally:
        server.close()


def test_a_silent_proxy_through_the_real_client_is_bounded(monkeypatch, fresh_record, silent_proxy):
    _record_numbers(monkeypatch, issue_tracker.MAX_LISTED)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": silent_proxy})
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 1.5)
    monkeypatch.setattr(issue_tracker, "LOOKUP_TIMEOUT_S", 0.5)
    started = time.monotonic()
    rows = issue_tracker.filed_status()
    elapsed = time.monotonic() - started
    assert elapsed < 5
    assert len(rows) == issue_tracker.MAX_LISTED
    assert all("error" in r and "state" not in r for r in rows)
    assert rows[0]["error"] == "could not reach the localm proxy"
    assert rows[-1]["error"] == issue_tracker.NOT_CHECKED


def test_the_issues_route_answers_inside_the_budget(monkeypatch, fresh_record, silent_proxy):
    _record_numbers(monkeypatch, issue_tracker.MAX_LISTED)
    monkeypatch.setattr(issue_tracker, "FILED_STATUS_BUDGET_S", 0.8)
    monkeypatch.setattr(issue_tracker, "LOOKUP_TIMEOUT_S", 0.4)
    app, c = _client(monkeypatch)
    monkeypatch.setattr(issue_tracker, "endpoint", lambda: (silent_proxy, None))
    with c:
        started = time.monotonic()
        r = c.get("/api/issues",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
        elapsed = time.monotonic() - started
    assert r.status_code == 200
    assert elapsed < 5
    data = r.json()
    assert [i["number"] for i in data["issues"]] == list(range(issue_tracker.MAX_LISTED, 0, -1))
    assert all("error" in i and "state" not in i for i in data["issues"])
