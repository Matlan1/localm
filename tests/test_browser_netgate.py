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

    The methods are found by walking BrowserSession's source for every public
    method that calls self._call, so a new one is checked without being listed.
    """

    #: Public methods that call self._call without a timeout_ms of their own,
    #: mapped to the reason they need no inner bound.
    EXEMPT: dict = {}

    def _marshalling_methods(self):
        import ast
        import inspect
        import textwrap
        from localm.browser.session import BrowserSession
        tree = ast.parse(textwrap.dedent(inspect.getsource(BrowserSession)))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        found = {}
        for node in cls.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "_call"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "self"):
                    found[node.name] = node
                    break
        return found

    def test_the_walk_finds_the_driving_methods(self):
        found = set(self._marshalling_methods())
        expected = {"navigate", "click", "fill", "click_coords", "scroll",
                    "type_text", "press_key"}
        assert expected <= found, "the walk missed %s" % sorted(expected - found)

    def test_every_marshalled_method_bounds_its_work_inside_the_call_timeout(self):
        import ast
        import inspect
        from localm.browser.session import BrowserSession, DEFAULT_CALL_TIMEOUT
        problems = []
        for name, node in sorted(self._marshalling_methods().items()):
            if name in self.EXEMPT:
                continue
            params = inspect.signature(getattr(BrowserSession, name)).parameters
            if "timeout_ms" not in params:
                problems.append("%s calls self._call with no timeout_ms and no "
                                "exemption" % name)
                continue
            ms = params["timeout_ms"].default
            if not (isinstance(ms, (int, float)) and not isinstance(ms, bool)
                    and 0 < ms / 1000.0 < DEFAULT_CALL_TIMEOUT):
                problems.append("%s: a timeout_ms of %r is not inside the %ss "
                                "call timeout" % (name, ms, DEFAULT_CALL_TIMEOUT))
            if not any(isinstance(n, ast.Name) and n.id == "timeout_ms"
                       for n in ast.walk(node)):
                problems.append("%s declares timeout_ms and never uses it" % name)
        assert problems == [], problems

    def test_every_exemption_names_a_method_that_exists(self):
        found = set(self._marshalling_methods())
        stale = sorted(set(self.EXEMPT) - found)
        assert stale == [], "exemptions for methods that no longer call _call: %s" % stale


class TestWebSocketsUseTheSameHostPolicy:
    """A WebSocket reaches a host exactly like a request does, so it is decided
    by the same policy rather than refused for its scheme. netpolicy only
    understands http(s), so ws(s) is mapped onto it."""

    def test_an_allowed_host_passes(self, cfg_home):
        _set(net_mode="allow")
        assert netgate.decide("ws://example.com/socket") is None
        assert netgate.decide("wss://example.com/socket") is None

    def test_a_denied_host_is_refused(self, cfg_home):
        _set(net_mode="allow", net_deny=["example.com"])
        for url in ("ws://example.com/s", "wss://example.com/s"):
            reason = netgate.decide(url)
            assert reason and "deny list" in reason, url

    def test_net_mode_off_refuses_a_websocket(self, cfg_home):
        _set(net_mode="off")
        reason = netgate.decide("ws://example.com/s")
        assert reason and "net_mode=off" in reason

    def test_a_private_address_websocket_is_refused(self, cfg_home):
        _set(net_mode="allow", net_allow_private=False)
        reason = netgate.decide("ws://127.0.0.1:9/s")
        assert reason and "non-public" in reason

    def test_the_allow_list_applies_to_websockets(self, cfg_home):
        _set(net_mode="allow", net_allow=["good.example"])
        assert netgate.decide("ws://good.example/s") is None
        assert netgate.decide("ws://evil.example/s") is not None

    def test_the_mapping_preserves_host_and_port(self, cfg_home):
        assert netgate._policy_url("ws://h:81/p?q=1", "ws") == "http://h:81/p?q=1"
        assert netgate._policy_url("wss://h/p", "wss") == "https://h/p"
        assert netgate._policy_url("http://h/p", "http") == "http://h/p"
