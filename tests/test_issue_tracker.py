# SPDX-License-Identifier: AGPL-3.0-or-later
"""The read-only issues tracker: configured-endpoint gating + the list/get calls
through the proxy."""

import json

import pytest

from localm import issue_tracker


def _opener(payload, status=200):
    def op(method, url, data, headers, timeout):
        op.url = url
        op.token = headers.get("X-Localm-Token")
        return status, json.dumps(payload).encode("utf-8")
    return op


def test_available(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert issue_tracker.available() is False
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    assert issue_tracker.available() is True


def test_list_issues(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "bugreport_upload_token": "tok"})
    op = _opener({"ok": True, "issues": [
        {"number": 5, "title": "real bug", "state": "open", "labels": ["bug"]}]})
    out = issue_tracker.list_issues("open", opener=op)
    assert out[0]["number"] == 5
    assert "/issues?state=open" in op.url
    assert op.token == "tok"


def test_list_issues_defaults_state_all(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    op = _opener({"issues": []})
    issue_tracker.list_issues("garbage", opener=op)
    assert "state=all" in op.url


def test_list_issues_unconfigured_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        issue_tracker.list_issues()


def test_get_issue(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    op = _opener({"ok": True, "issue": {"number": 12, "state": "closed", "title": "t"}})
    out = issue_tracker.get_issue(12, opener=op)
    assert out["number"] == 12 and out["state"] == "closed"
    assert "/issues?number=12" in op.url


def test_get_issue_bad_number_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
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
    monkeypatch.setattr("localm.config.load_config", lambda: dict(cfg))
    calls = _count_proxy_calls(monkeypatch)
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as e1:
        issue_tracker.list_issues()
    with pytest.raises(LocalmError) as e2:
        issue_tracker.get_issue(3)
    assert calls == []
    assert "net_mode = off" in e1.value.summary
    assert "net_mode = off" in e2.value.summary


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_net_mode_other_than_off_still_requests(monkeypatch, mode):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "net_mode": mode})
    calls = _count_proxy_calls(monkeypatch)
    issue_tracker.list_issues()
    issue_tracker.get_issue(3)
    assert len(calls) == 2


def test_issues_route_under_net_mode_off_reports_and_sends_nothing(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    monkeypatch.setattr(issue_tracker, "endpoint", lambda: ("https://w", "tok"))
    calls = _count_proxy_calls(monkeypatch)
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)
    with TestClient(app) as c:
        r = c.get("/api/issues",
                  headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert calls == []
    data = r.json()
    assert data["available"] is True and data["issues"] == []
    assert "net_mode = off" in data["error"]
