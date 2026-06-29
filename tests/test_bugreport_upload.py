# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "Send to maintainer" upload channel: the app POSTs a user-reviewed report
to the configured proxy (a Cloudflare Worker holding the GitHub token), which
files it as a GitHub issue. No token ships in the app.

Pins: config gating, the POST shape + optional shared secret, and the honesty
rule - a failed upload raises (never a false success), and the saved file stays.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import bugreport
from localm.inference.http_server import create_app


# ------------------------------- config gating --------------------------- #

def test_upload_not_configured_by_default(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert bugreport.upload_config() == (None, None)
    assert bugreport.upload_available() is False


def test_upload_config_reads_url_and_token(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://proxy.example/",
        "bugreport_upload_token": "  s3cret  ",
    })
    url, token = bugreport.upload_config()
    assert url == "https://proxy.example/"
    assert token == "s3cret"
    assert bugreport.upload_available() is True


# ------------------------------- upload_report ---------------------------- #

def test_upload_report_posts_and_returns_issue_url():
    seen = {}

    def opener(url, data, headers, timeout):
        import json
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = json.loads(data.decode("utf-8"))
        return 201, '{"ok": true, "url": "https://github.com/x/localm/issues/9"}'

    res = bugreport.upload_report(
        "image gen froze", "## report\nbody",
        url="https://proxy.example/file", token="tok", opener=opener)
    assert res["url"].endswith("/issues/9")
    assert seen["url"] == "https://proxy.example/file"
    assert seen["payload"]["title"] == "image gen froze"
    assert seen["payload"]["body"] == "## report\nbody"
    # The shared secret rides a header, never the GitHub token (which is not ours).
    assert seen["headers"]["X-Localm-Token"] == "tok"
    assert not any("github" in k.lower() for k in seen["headers"])


def test_upload_report_no_endpoint_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with pytest.raises(bugreport.LocalmError):
        bugreport.upload_report("t", "b")


def test_upload_report_non_2xx_raises():
    def opener(url, data, headers, timeout):
        return 502, '{"error": "GitHub rejected the issue"}'

    with pytest.raises(bugreport.LocalmError) as ei:
        bugreport.upload_report("t", "b", url="https://proxy.example", opener=opener)
    assert "502" in (ei.value.reason or "")


def test_upload_report_omits_token_header_when_none():
    seen = {}

    def opener(url, data, headers, timeout):
        seen["headers"] = headers
        return 201, "{}"

    bugreport.upload_report("t", "b", url="https://proxy.example", token=None, opener=opener)
    assert "X-Localm-Token" not in seen["headers"]


# ------------------------------- CLI menu --------------------------------- #

def test_cli_menu_upload_branch_calls_upload(tmp_path, monkeypatch):
    """When an upload endpoint is configured, the CLI report menu offers "[1] Send
    now", and picking it uploads the (edited) report rather than opening a browser."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config",
                        lambda: ("https://proxy.example", "tok"))
    sent = {}

    def fake_upload(title, body, *, url=None, token=None, **kw):
        sent.update(title=title, url=url, token=token, body=body)
        return {"url": "https://github.com/x/localm/issues/3"}

    monkeypatch.setattr(bugreport, "upload_report", fake_upload)
    opened = []
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "1")
    assert sent.get("url") == "https://proxy.example"
    assert sent.get("token") == "tok"
    assert opened == []   # upload does not open a browser


def test_cli_menu_no_upload_option_when_unconfigured(tmp_path, monkeypatch, capsys):
    """Without an endpoint, the menu has no "Send now" option and "1" maps to nothing
    (email is [2], not [1]) - so picking "1" opens no browser."""
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setattr(bugreport, "upload_config", lambda: (None, None))
    opened = []
    bugreport.report_failure(
        summary="bug", interactive=True,
        open_browser=lambda u: opened.append(u), prompt=lambda _t: "1")
    out = capsys.readouterr().out
    assert "Send to the maintainer now" not in out
    assert opened == []


# ------------------------------- endpoint --------------------------------- #

def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def _post(app, body):
    with TestClient(app) as c:
        return c.post("/api/bug-report", json=body,
                      headers={"Authorization": f"Bearer {app.state.shell_token}"})


def test_endpoint_uploads_on_request(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(bugreport, "upload_report",
                        lambda title, body: {"url": "https://github.com/x/localm/issues/12"})
    r = _post(create_app(_engine()),
              {"description": "the thing broke", "upload": True})
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    assert data["uploaded"] is True
    assert data["issue_url"].endswith("/issues/12")
    assert Path(data["path"]).is_file()   # still saved to disk


def test_endpoint_upload_failure_is_surfaced_not_hidden(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    def boom(title, body):
        raise bugreport.LocalmError("could not reach the bug-report server",
                                    reason="timed out")

    monkeypatch.setattr(bugreport, "upload_report", boom)
    r = _post(create_app(_engine()),
              {"description": "broke again", "upload": True})
    # The save still succeeds (200), but the upload failure is reported honestly.
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] is True
    assert data["uploaded"] is False
    assert "could not reach" in data["upload_error"]
    assert "issue_url" not in data
    assert Path(data["path"]).is_file()


def test_endpoint_does_not_upload_without_flag(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    called = []
    monkeypatch.setattr(bugreport, "upload_report",
                        lambda *a, **k: called.append(1) or {"url": "x"})
    r = _post(create_app(_engine()), {"description": "just save it"})
    assert r.status_code == 200
    assert "uploaded" not in r.json()
    assert called == []   # never uploads unless explicitly asked
