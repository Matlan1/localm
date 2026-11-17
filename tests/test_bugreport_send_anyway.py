# SPDX-License-Identifier: AGPL-3.0-or-later
"""Send anyway: with Network access off, the owner can send one reviewed bug
report on an explicit choice. The default send still refuses, the choice opens
exactly one connection and writes no setting, and a caller that is not the owner
cannot use it."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from localm import bugreport
from localm.bugreport import LocalmError
from localm.inference.http_server import create_app

OWNER_KEY = "owner-admin-key-send-anyway-7f3a"
URL = "http://127.0.0.1:9/report"


class _Resp:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return b'{"url": "https://example.invalid/issues/1"}'


@pytest.fixture
def opened(monkeypatch):
    calls = []

    def _urlopen(req, **kw):
        calls.append(getattr(req, "full_url", req))
        return _Resp()
    monkeypatch.setattr("localm.http_ssl.verified_urlopen", _urlopen)
    return calls


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A protected-mode data dir whose config.json on disk has Network access off
    and the upload pointed at an address nothing listens on."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    (home / "config.json").write_text(json.dumps(
        {"net_mode": "off", "bugreport_upload_url": URL, "update_url": ""}),
        encoding="utf-8")
    return home


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def client(home, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    from localm import auth
    scoped = auth.create_key("dev", ["config:read", "config:write"],
                             allow_privileged=True)["key"]
    app = create_app(None)
    with TestClient(app) as c:
        yield c, scoped


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _save(c, key):
    r = c.post("/api/bug-report", json={"description": "it broke"}, headers=_hdr(key))
    assert r.status_code == 200, r.text
    d = r.json()
    return {"filename": d["filename"], "sha256": d["report_sha256"]}


def test_the_default_send_still_refuses_and_offers_send_anyway_to_the_owner(
        client, home, opened):
    c, _ = client
    r = c.post("/api/bug-report/send", json=_save(c, OWNER_KEY), headers=_hdr(OWNER_KEY))
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["uploaded"] is False
    assert body["upload_stage"] == "network_off"
    assert body["can_send_anyway"] is True
    assert opened == []


def test_send_anyway_sends_exactly_one_request_and_writes_no_setting(
        client, home, opened):
    c, _ = client
    before = _digest(home / "config.json")
    rep = _save(c, OWNER_KEY)
    r = c.post("/api/bug-report/send", json={**rep, "send_anyway": True},
               headers=_hdr(OWNER_KEY))
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["uploaded"] is True, body
    assert opened == [URL]
    assert _digest(home / "config.json") == before
    from localm import config
    assert config.network_mode() == "off"


@pytest.mark.parametrize("value", ["true", 1, "yes", {"x": 1}])
def test_only_a_literal_true_is_the_choice(client, home, opened, value):
    c, _ = client
    r = c.post("/api/bug-report/send", json={**_save(c, OWNER_KEY), "send_anyway": value},
               headers=_hdr(OWNER_KEY))
    assert r.json()["upload_stage"] == "network_off"
    assert opened == []


def test_a_scoped_key_cannot_send_anyway(client, home, opened):
    c, scoped = client
    before = _digest(home / "config.json")
    rep = _save(c, scoped)
    r = c.post("/api/bug-report/send", json={**rep, "send_anyway": True},
               headers=_hdr(scoped))
    assert r.status_code == 403, r.text
    assert opened == []
    assert _digest(home / "config.json") == before
    plain = c.post("/api/bug-report/send", json=rep, headers=_hdr(scoped)).json()
    assert plain["upload_stage"] == "network_off"
    assert plain["can_send_anyway"] is False
    assert opened == []


def test_a_request_with_no_key_cannot_send_anyway(client, opened):
    c, _ = client
    rep = _save(c, OWNER_KEY)
    r = c.post("/api/bug-report/send", json={**rep, "send_anyway": True})
    assert r.status_code in (401, 403), r.text
    assert opened == []


def test_upload_report_send_anyway_bypasses_only_the_off_switch(home, opened):
    with pytest.raises(LocalmError) as ei:
        bugreport.upload_report("t", "b", url=URL)
    assert ei.value.stage == "network_off"
    assert opened == []
    out = bugreport.upload_report("t", "b", url=URL, send_anyway=True)
    assert out["url"].endswith("/issues/1")
    assert opened == [URL]


def test_cli_send_anyway_sends_one_report(home, opened, monkeypatch):
    from click.testing import CliRunner

    from localm.cli import main
    before = _digest(home / "config.json")
    res = CliRunner().invoke(main, ["bug-report", "-w", "it broke", "--send-anyway"])
    assert res.exit_code == 0, res.output
    assert "Sent to the maintainer" in res.output
    assert opened == [URL]
    assert _digest(home / "config.json") == before


def test_cli_send_with_network_off_names_send_anyway(home, opened):
    from click.testing import CliRunner

    from localm.cli import main
    res = CliRunner().invoke(main, ["bug-report", "-w", "it broke", "--send"])
    assert res.exit_code == 1, res.output
    assert "--send-anyway" in res.output
    assert opened == []


@pytest.mark.parametrize("answer, sends", [("y", True), ("", False), ("n", False)])
def test_the_menu_asks_in_place_and_defaults_to_no(home, opened, monkeypatch,
                                                   answer, sends):
    from localm.bugreport import consent
    monkeypatch.setattr(consent.console, "print", lambda *a, **kw: None)
    asked = []
    consent._upload_with_retries("it broke", None, "report text", URL, None,
                                 lambda q: asked.append(q) or answer, "report.md")
    assert asked == ["  Send this one report anyway? [y/N]"]
    assert opened == ([URL] if sends else [])


def test_a_rate_limited_send_anyway_from_the_menu_is_retried_once(home, monkeypatch):
    from localm.bugreport import RateLimitedError, consent
    monkeypatch.setattr(consent.console, "print", lambda *a, **kw: None)
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def _upload(title, body, **kw):
        calls.append(kw.get("send_anyway"))
        if not kw.get("send_anyway"):
            raise LocalmError("off", stage="network_off")
        if len(calls) == 2:
            raise RateLimitedError(1)
        return {"url": "https://example.invalid/issues/1"}
    monkeypatch.setattr(bugreport, "upload_report", _upload)
    consent._upload_with_retries("it broke", None, "report text", URL, None,
                                 lambda q: "y", "report.md")
    assert calls == [False, True, True]
