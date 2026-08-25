# SPDX-License-Identifier: AGPL-3.0-or-later
"""The read-only issues tracker: configured-endpoint gating + the list/get calls through the proxy."""

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
