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


def test_pull_model_proceeds_when_net_mode_off_but_downloads_allowed(monkeypatch):
    """net_allow_model_downloads exempts an explicit pull from the off floor.
    Asserted on dispatch: _pull_url is mocked to return a sentinel that pull_model
    cannot produce on its own, so a True (the sentinel, coerced) proves execution
    reached the dispatch, not just that no exception happened to be raised."""
    import localm.netpolicy as netpolicy
    from localm.model_manager import pull as pullmod
    monkeypatch.setattr(netpolicy, "network_mode", lambda: "off")
    monkeypatch.setattr(netpolicy, "downloads_allowed_when_off", lambda: True)
    called = []
    monkeypatch.setattr(pullmod, "_pull_url",
                        lambda *a, **kw: called.append(1) or True)
    assert pullmod.pull_model("https://example.com/model.gguf") is True
    assert called == [1]


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


def test_ssrf_resolver_refuses_off_by_default(monkeypatch):
    """pull_model's own top-level gate is not the only net_mode=off check on
    this path - _ssrf_resolve_final_url's per-hop check_url calls have their
    own, independent floor. A public host is used here specifically to prove
    it is the OFF gate refusing, not the SSRF/private-address one."""
    from localm.model_manager.pull import _ssrf_resolve_final_url
    from localm.netpolicy import NetworkPolicyError
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    with pytest.raises(NetworkPolicyError, match="net_mode=off"):
        _ssrf_resolve_final_url("https://huggingface.co/org/repo/resolve/main/model.gguf")


def test_ssrf_resolver_proceeds_when_downloads_allowed_despite_off(monkeypatch):
    """net_allow_model_downloads must reach the per-hop check_url calls
    directly, not just pull_model's own top-level gate - the two are
    independent checks on the same request.

    check_url reads net_allow_model_downloads from its own already-loaded
    config snapshot (never via downloads_allowed_when_off - a second
    load_config() call would break check_url's documented one-read
    discipline), so this patches config.load_config, not the helper -
    _ssrf_resolve_final_url's own allow_off is computed via the helper
    (proving that path too), but check_url decides independently."""
    import localm.config as _cfg
    import localm.netpolicy as netpolicy
    from localm.model_manager.pull import _ssrf_resolve_final_url
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    monkeypatch.setattr(
        _cfg, "load_config",
        lambda: {"net_mode": "off", "net_allow_model_downloads": True})
    calls = []

    class _Resp:
        status_code = 200
        headers: dict = {}

    def _fake_pinned_request(method, url, **kw):
        calls.append((method, url))
        return _Resp()

    monkeypatch.setattr(netpolicy, "pinned_request", _fake_pinned_request)
    url = "https://huggingface.co/org/repo/resolve/main/model.gguf"
    assert _ssrf_resolve_final_url(url) == url
    assert calls == [("HEAD", url)], "the override must let the real HEAD probe run"


def test_pull_url_refuses_loopback_returns_false(monkeypatch, tmp_path):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    from localm.model_manager.pull import _pull_url
    # A loopback download URL is refused by the SSRF resolver -> False, no write.
    assert _pull_url("http://127.0.0.1:9/model.gguf", "m") is False
