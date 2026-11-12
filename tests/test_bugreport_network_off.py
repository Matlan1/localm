# SPDX-License-Identifier: AGPL-3.0-or-later
"""With Network access off nothing leaves the machine: a bug report is not
uploaded (no connection is opened), and the GUI and the CLI say why and how to
turn the network on."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm import _proxy, bugreport
from localm.bugreport import LocalmError
from tests._net_config import set_config

URL = "https://upload.invalid/report"


def _counting_opener(calls):
    def op(url, data, headers, timeout):
        calls.append(url)
        return 200, '{"url": "https://github.com/x/localm/issues/1"}'
    return op


def _off(monkeypatch, how):
    cfg = {"bugreport_upload_url": URL}
    if how == "config":
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        cfg["net_mode"] = "off"
    else:
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
    set_config(monkeypatch, lambda: dict(cfg))


@pytest.mark.parametrize("how", ["config", "env"])
def test_upload_with_network_off_opens_no_connection(monkeypatch, how):
    _off(monkeypatch, how)
    calls = []
    with pytest.raises(LocalmError) as ei:
        bugreport.upload_report("t", "b", url=URL, opener=_counting_opener(calls))
    assert calls == []
    assert ei.value.stage == "network_off"
    assert "Settings > Server & network > Network access" in ei.value.hint
    assert "localm config net_mode ask" in ei.value.hint


def test_upload_with_an_unreadable_config_opens_no_connection(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config_checked", lambda: ({}, False))
    calls = []
    with pytest.raises(LocalmError) as ei:
        bugreport.upload_report("t", "b", url=URL, token="",
                                opener=_counting_opener(calls))
    assert calls == []
    assert ei.value.stage == "network_off"


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_upload_with_network_on_sends_once(monkeypatch, mode):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {"bugreport_upload_url": URL, "net_mode": mode})
    calls = []
    out = bugreport.upload_report("t", "b", url=URL, opener=_counting_opener(calls))
    assert calls == [URL]
    assert out["url"].endswith("/issues/1")


def test_every_proxy_caller_must_state_the_off_switch():
    with pytest.raises(TypeError, match="net_exempt"):
        _proxy.request("https://proxy.invalid", "/x",
                       opener=lambda *a: (200, b"{}"))


def test_an_exempt_proxy_call_is_sent_while_off(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    calls = []

    def op(method, url, data, headers, timeout):
        calls.append(url)
        return 200, b"{}"
    assert _proxy.request("https://proxy.invalid", "/x", net_exempt=True, opener=op) == {}
    assert calls == ["https://proxy.invalid/x"]
    with pytest.raises(LocalmError) as ei:
        _proxy.request("https://proxy.invalid", "/x", net_exempt=False, opener=op)
    assert ei.value.stage == "network_off"
    assert calls == ["https://proxy.invalid/x"]


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.loaded = True
    return e


def test_gui_send_with_network_off_says_why_and_sends_nothing(tmp_path, monkeypatch):
    from localm.inference.http_server import create_app
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    opened = []

    def _no_socket(req, **kw):
        opened.append(getattr(req, "full_url", req))
        raise AssertionError("a connection was opened with network access off")
    monkeypatch.setattr("localm.http_ssl.verified_urlopen", _no_socket)
    app = create_app(_engine())
    hdrs = {"Authorization": f"Bearer {app.state.shell_token}"}
    with TestClient(app) as c:
        saved = c.post("/api/bug-report", json={"description": "it broke"}, headers=hdrs)
        assert saved.status_code == 200, saved.text
        data = saved.json()
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        r = c.post("/api/bug-report/send", headers=hdrs,
                   json={"filename": data["filename"], "sha256": data["report_sha256"]})
    assert opened == []
    body = r.json()
    assert r.status_code == 200
    assert body["uploaded"] is False
    assert body["upload_stage"] == "network_off"
    assert "Settings > Server & network > Network access" in body["upload_message"]


def test_cli_send_with_network_off_says_why_and_sends_nothing(monkeypatch):
    from localm.bugreport import consent
    _off(monkeypatch, "env")
    opened = []
    monkeypatch.setattr("localm.http_ssl.verified_urlopen",
                        lambda req, **kw: opened.append(req))
    printed = []
    monkeypatch.setattr(consent.console, "print",
                        lambda *a, **kw: printed.append(" ".join(str(x) for x in a)))
    consent._auto_send("it broke", None, "report text", True, URL, None)
    assert opened == []
    out = "\n".join(printed)
    assert "Network access is off" in out
    assert "localm config net_mode ask" in out
