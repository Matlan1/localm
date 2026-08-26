# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for model-pull SSRF and the net_mode kill switch.

These cover the net_mode=off gate and the per-hop check_url validation of the
pull URL and its redirects.
"""

import pytest


def test_pull_model_refused_when_net_mode_off(monkeypatch):
    import localm.netpolicy as netpolicy
    from localm.model_manager import pull as pullmod
    monkeypatch.setattr(netpolicy, "network_mode", lambda: "off")
    # A remote URL spec must be refused with no download when the kill switch is on.
    assert pullmod.pull_model("https://example.com/model.gguf") is False


@pytest.mark.parametrize("url", [
    pytest.param("http://127.0.0.1:8000/model.gguf", id="loopback"),
    pytest.param("http://169.254.169.254/latest/meta-data/", id="cloud-metadata"),
])
def test_ssrf_resolver_refuses_private_targets(monkeypatch, url):
    from localm.model_manager.pull import _ssrf_resolve_final_url
    from localm.netpolicy import NetworkPolicyError
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")     # isolate the IP-class check
    with pytest.raises(NetworkPolicyError):
        _ssrf_resolve_final_url(url)


def test_pull_url_refuses_loopback_returns_false(monkeypatch, tmp_path):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    from localm.model_manager.pull import _pull_url
    # A loopback download URL is refused by the SSRF resolver -> False, no write.
    assert _pull_url("http://127.0.0.1:9/model.gguf", "m") is False
