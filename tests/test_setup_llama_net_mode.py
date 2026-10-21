# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-llama`` honours ``net_mode``: with ``net_mode off`` it makes no
network request, whichever way it is reached, and a local build still installs."""

from __future__ import annotations

import urllib.request

import pytest

import localm.setup_llama as sl
from localm import netpolicy


@pytest.fixture
def opener(monkeypatch):
    calls = []
    monkeypatch.setattr(sl, "_verified_urlopen",
                        lambda req, *, timeout=None, handlers=(): calls.append(req) or object())
    return calls


def _request():
    return urllib.request.Request("https://example.invalid/llama.zip")


def test_no_request_is_made_with_net_mode_off(monkeypatch, opener):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(netpolicy.NetworkPolicyError) as ei:
        sl.verified_urlopen(_request(), timeout=5)
    assert ei.value.off is True
    assert "net_mode" in str(ei.value)
    assert opener == []


@pytest.mark.parametrize("mode", ["ask", "on"])
def test_requests_pass_through_when_net_mode_allows(monkeypatch, opener, mode):
    monkeypatch.setenv("LOCALM_NET_MODE", mode)
    req = _request()
    sl.verified_urlopen(req, timeout=5)
    assert opener == [req]


def test_the_download_exemption_for_net_mode_off_is_honoured(monkeypatch, opener):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    monkeypatch.setattr(netpolicy, "downloads_allowed_when_off", lambda: True)
    sl.verified_urlopen(_request(), timeout=5)
    assert len(opener) == 1


def test_a_refused_download_surfaces_as_an_artifact_error(monkeypatch, opener, tmp_path):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(sl.ArtifactError) as ei:
        sl._download("https://example.invalid/llama.zip", tmp_path / "a.zip")
    assert "net_mode" in str(ei.value)
    assert opener == []


def _fake_wheel_install(monkeypatch):
    import sys
    from pathlib import Path
    captured = {}

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["env"] = kw.get("env")
        return _R()

    monkeypatch.setattr(sl.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "localm_llama_runtime", None)
    assert sl._install_runtime_wheel(Path("some/pkg/dir")) is True
    return captured["env"]


def test_the_runtime_wheel_install_is_offline_with_net_mode_off(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    env = _fake_wheel_install(monkeypatch)
    assert env["UV_OFFLINE"] == "1" and env["PIP_NO_INDEX"] == "1"


def test_the_runtime_wheel_install_may_use_the_index_when_net_mode_allows(monkeypatch):
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")
    monkeypatch.delenv("UV_OFFLINE", raising=False)
    monkeypatch.delenv("PIP_NO_INDEX", raising=False)
    env = _fake_wheel_install(monkeypatch)
    assert "UV_OFFLINE" not in env and "PIP_NO_INDEX" not in env
