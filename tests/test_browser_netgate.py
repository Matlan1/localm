# SPDX-License-Identifier: AGPL-3.0-or-later
"""The automated browser's per-request network gate.

Every request the browser makes (the navigation AND every subresource the page
pulls on its own) is decided by ``localm.browser.netgate.decide``. netpolicy
makes the network decision; the gate adds scheme triage and an optional
browser-specific narrowing.

The load-bearing property, and most of this file: the browser-specific rules can
only ever REFUSE MORE. No combination of them reaches a destination the global
policy already denied.
"""

from pathlib import Path

import pytest

from localm.browser import netgate


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    """A throwaway config the tests can write net_* keys into."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    _cfg.ensure_dirs()
    return home


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


class TestSchemeTriage:
    @pytest.mark.parametrize("url", [
        "about:blank", "data:text/html,<b>x</b>", "blob:http://x/y",
    ])
    def test_inert_schemes_pass_without_consulting_the_policy(self, url, cfg_home):
        _set(net_mode="off")           # even the hardest floor
        assert netgate.decide(url) is None

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd", "file://C:/Windows/win.ini", "ftp://example.com/x",
        "chrome://settings", "view-source:http://example.com",
    ])
    def test_every_other_scheme_is_refused(self, url, cfg_home):
        _set(net_mode="allow")
        reason = netgate.decide(url)
        assert reason and "not allowed" in reason, reason

    def test_file_scheme_is_refused_even_with_a_browser_allow_rule(self, cfg_home):
        _set(net_mode="allow")
        assert netgate.decide("file:///etc/passwd",
                              extra_allow=["*"]) is not None


class TestGlobalPolicyGoverns:
    def test_allowed_when_the_policy_allows(self, cfg_home):
        _set(net_mode="allow", net_allow_private=True)
        assert netgate.decide("http://127.0.0.1:9/page") is None

    def test_net_mode_off_refuses(self, cfg_home):
        _set(net_mode="off")
        reason = netgate.decide("https://example.com/")
        assert reason and "net_mode=off" in reason

    def test_deny_list_refuses(self, cfg_home):
        _set(net_mode="allow", net_deny=["example.com"])
        reason = netgate.decide("https://example.com/")
        assert reason and "deny list" in reason

    def test_allow_list_refuses_a_host_not_on_it(self, cfg_home):
        _set(net_mode="allow", net_allow=["example.com"])
        assert netgate.decide("https://other.example.org/") is not None
        assert netgate.decide("https://example.com/x") is None

    def test_private_address_refused_by_the_ssrf_guard(self, cfg_home):
        _set(net_mode="allow", net_allow_private=False)
        reason = netgate.decide("http://127.0.0.1:8080/admin")
        assert reason and "non-public" in reason

    def test_metadata_address_refused(self, cfg_home):
        _set(net_mode="allow", net_allow_private=False)
        assert netgate.decide("http://169.254.169.254/latest/meta-data/") is not None


class TestBrowserRulesOnlyNarrow:
    """The invariant the whole opt-in exists under: browser rules refuse more,
    never less."""

    def test_extra_deny_refuses_a_host_the_policy_allowed(self, cfg_home):
        _set(net_mode="allow")
        assert netgate.decide("https://example.com/") is None
        reason = netgate.decide("https://example.com/", extra_deny=["example.com"])
        assert reason and "browser deny list" in reason

    def test_extra_allow_refuses_everything_not_listed(self, cfg_home):
        _set(net_mode="allow")
        assert netgate.decide("https://a.example/", extra_allow=["b.example"]) is not None
        assert netgate.decide("https://b.example/", extra_allow=["b.example"]) is None

    def test_extra_allow_cannot_reopen_net_mode_off(self, cfg_home):
        _set(net_mode="off")
        reason = netgate.decide("https://example.com/", extra_allow=["example.com"])
        assert reason and "net_mode=off" in reason

    def test_extra_allow_cannot_reopen_a_denied_host(self, cfg_home):
        _set(net_mode="allow", net_deny=["example.com"])
        reason = netgate.decide("https://example.com/",
                                extra_allow=["example.com"])
        assert reason and "deny list" in reason

    def test_extra_allow_cannot_reach_a_private_address(self, cfg_home):
        _set(net_mode="allow", net_allow_private=False)
        reason = netgate.decide("http://127.0.0.1:8080/",
                                extra_allow=["127.0.0.1"])
        assert reason and "non-public" in reason

    def test_extra_allow_cannot_bypass_the_global_allow_list(self, cfg_home):
        _set(net_mode="allow", net_allow=["good.example"])
        reason = netgate.decide("https://evil.example/",
                                extra_allow=["evil.example"])
        assert reason and "allow list" in reason


class TestFailSafe:
    def test_a_policy_error_refuses_rather_than_passes(self, cfg_home, monkeypatch):
        def boom(url, **kw):
            raise RuntimeError("config exploded")
        monkeypatch.setattr(netgate.netpolicy, "check_url", boom)
        reason = netgate.decide("https://example.com/")
        assert reason and "could not be evaluated" in reason

    def test_a_parser_differential_url_is_refused(self, cfg_home):
        _set(net_mode="allow")
        assert netgate.decide(r"http://127.0.0.1\@example.com/") is not None


class TestAsyncWrapper:
    def test_decide_async_matches_decide(self, cfg_home):
        import asyncio
        _set(net_mode="allow", net_deny=["example.com"])

        async def run():
            return (await netgate.decide_async("https://example.com/"),
                    await netgate.decide_async("https://other.example/"))
        denied, allowed = asyncio.run(run())
        assert denied is not None and "deny list" in denied
        assert allowed is None


class TestTimeoutsNest:
    """The marshalling timeout must OUTLAST every browser timeout it wraps.

    Inverted, the caller abandons the call while the browser is still working,
    so the page's own timeout never gets to produce a real error and the worker
    keeps running past the report. Asserted as the RELATION, not as literals, so
    retuning one end cannot silently break it.
    """

    def _default_ms(self, fn, name):
        import inspect
        return inspect.signature(fn).parameters[name].default

    def test_every_page_timeout_fits_inside_the_call_timeout(self):
        from localm.browser.session import BrowserSession, DEFAULT_CALL_TIMEOUT
        inner = [
            self._default_ms(BrowserSession.navigate, "timeout_ms"),
            self._default_ms(BrowserSession.click, "timeout_ms"),
            self._default_ms(BrowserSession.fill, "timeout_ms"),
        ]
        for ms in inner:
            assert ms / 1000.0 < DEFAULT_CALL_TIMEOUT, (
                "a browser timeout of %sms is not inside the %ss call timeout"
                % (ms, DEFAULT_CALL_TIMEOUT))
