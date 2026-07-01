# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for SSRF-PULL (model-pull SSRF + net_mode kill switch).

The full DNS-rebinding TOCTOU fix (SSRF-REBIND, a pinned HTTPAdapter) is a
separate, TLS-sensitive effort; these cover the safe core: the net_mode=off gate
and per-hop check_url validation of the pull URL / its redirects.
"""

import pytest


def test_pull_model_refused_when_net_mode_off(monkeypatch):
    import localm.netpolicy as netpolicy
    from localm.model_manager import pull as pullmod
    monkeypatch.setattr(netpolicy, "network_mode", lambda: "off")
    # A remote URL spec must be refused with no download when the kill switch is on.
    assert pullmod.pull_model("https://example.com/model.gguf") is False


def test_ssrf_resolver_refuses_loopback(monkeypatch):
    from localm.model_manager.pull import _ssrf_resolve_final_url
    from localm.netpolicy import NetworkPolicyError
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")     # isolate the IP-class check
    with pytest.raises(NetworkPolicyError):
        _ssrf_resolve_final_url("http://127.0.0.1:8000/model.gguf")


def test_ssrf_resolver_refuses_cloud_metadata(monkeypatch):
    from localm.model_manager.pull import _ssrf_resolve_final_url
    from localm.netpolicy import NetworkPolicyError
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    with pytest.raises(NetworkPolicyError):
        _ssrf_resolve_final_url("http://169.254.169.254/latest/meta-data/")


def test_pull_url_refuses_loopback_returns_false(monkeypatch, tmp_path):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    from localm.model_manager.pull import _pull_url
    # A loopback download URL is refused by the SSRF resolver -> False, no write.
    assert _pull_url("http://127.0.0.1:9/model.gguf", "m") is False
