# SPDX-License-Identifier: AGPL-3.0-or-later
"""Search providers behind the SearchProvider contract
(localm.web_retrieval.providers) and the netpolicy.web_search delegate."""

from __future__ import annotations

import pytest
import requests

from localm import netpolicy
from localm.web_retrieval import (
    DuckDuckGoHTMLProvider,
    SearchProviderError,
    SearchResult,
    SearXNGProvider,
    provider_from_config,
    search,
)
from tests._web_retrieval_fixtures import (
    BRAVE_SEARCH,
    DDG_ENDPOINT,
    LITE_ENDPOINT,
    FakeResponse,
    Transport,
    allow_public,
    ddg_html,
    ddg_no_results,
    searx_json,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


class _StubProvider:
    name = "stub"

    def __init__(self, results=None, exc=None):
        self.results = results or []
        self.exc = exc
        self.calls: list[tuple[str, int]] = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.exc is not None:
            raise self.exc
        return list(self.results)


class TestDuckDuckGoHTMLProvider:
    def test_parses_results_with_ranks_and_decodes_uddg(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(text=ddg_html([
            ("Example Docs", "https://example.com/docs", "Official documentation."),
            ("Direct", "https://direct.example.org/page", "A direct link."),
        ])))
        out = DuckDuckGoHTMLProvider().search("example docs", 5)
        assert [r.url for r in out] == ["https://example.com/docs",
                                        "https://direct.example.org/page"]
        assert [r.rank for r in out] == [1, 2]
        assert all(r.provider == "duckduckgo-html" for r in out)
        assert out[0].title == "Example Docs"
        assert out[0].snippet == "Official documentation."
        method, url, kw = t.calls[0]
        assert (method, url) == ("POST", DDG_ENDPOINT)
        assert kw["allow_redirects"] is False
        assert kw["data"] == {"q": "example docs", "b": "", "kl": "wt-wt"}
        assert kw["headers"]["Host"] == "html.duckduckgo.com"
        assert kw["headers"]["Referer"] == "https://html.duckduckgo.com/"
        assert kw["headers"]["Accept-Language"]
        assert kw["headers"]["Accept"].startswith("text/html")

    def test_max_results_and_caps(self, monkeypatch):
        allow_public(monkeypatch)
        rows = [(f"T{i} " + "x" * 400, f"https://r{i}.example/", "s " * 400)
                for i in range(6)]
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT, FakeResponse(text=ddg_html(rows)))
        out = DuckDuckGoHTMLProvider().search("q", 4)
        assert len(out) == 4
        assert all(len(r.title) <= 300 and len(r.snippet) <= 500 for r in out)
        assert "  " not in out[0].snippet

    def test_redirect_refused(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT,
            FakeResponse(status=302, redirect="http://127.0.0.1/"))
        with pytest.raises(netpolicy.NetworkPolicyError, match="redirect"):
            DuckDuckGoHTMLProvider().search("q", 5)

    def test_http_error_propagates(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT, FakeResponse(status=503, text=""))
        with pytest.raises(requests.HTTPError, match="HTTP 503"):
            DuckDuckGoHTMLProvider().search("q", 5)

    def test_no_results_page_yields_empty_list(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT, FakeResponse(text=ddg_no_results()))
        assert DuckDuckGoHTMLProvider().search("q", 5) == []

    def test_page_without_results_or_no_results_message_is_unreadable(
            self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT,
            FakeResponse(text="<html><body>captcha?</body></html>"))
        with pytest.raises(SearchProviderError) as info:
            DuckDuckGoHTMLProvider().search("q", 5)
        assert str(info.value) == ("DuckDuckGo answered with a page localm "
                                   "could not read results from.")

    def test_policy_off_refuses_before_any_request(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_mode": "off"})
        t = Transport().install(monkeypatch)
        with pytest.raises(netpolicy.NetworkPolicyError):
            DuckDuckGoHTMLProvider().search("q", 5)
        assert t.calls == []


class TestSearXNGProvider:
    def test_query_url_results_and_ranks(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*", FakeResponse(
            json_body=searx_json([("T1", "https://a.example/", "c1"),
                                  ("T2", "https://b.example/", "c2")])))
        out = SearXNGProvider("https://searx.example/").search("query", 5)
        assert [(r.title, r.url, r.snippet, r.rank) for r in out] == [
            ("T1", "https://a.example/", "c1", 1),
            ("T2", "https://b.example/", "c2", 2)]
        assert all(r.provider == "searxng" for r in out)
        method, url, kw = t.calls[0]
        assert method == "GET"
        assert url.startswith("https://searx.example/search?")
        assert "format=json" in url and "q=query" in url
        assert kw["allow_redirects"] is False

    def test_max_results_honoured(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*", FakeResponse(
                json_body=searx_json([("T", f"https://r{i}.example/", "c")
                                      for i in range(5)])))
        assert len(SearXNGProvider("https://searx.example").search("q", 2)) == 2

    def test_redirect_refused(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*",
            FakeResponse(status=302, redirect="http://127.0.0.1/"))
        with pytest.raises(netpolicy.NetworkPolicyError, match="redirect"):
            SearXNGProvider("https://searx.example").search("q", 5)

    @pytest.mark.parametrize("payload", [["not", "a", "dict"],
                                         {"results": "nope"}])
    def test_malformed_payload_is_a_provider_error(self, monkeypatch, payload):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*",
            FakeResponse(json_body=payload))
        with pytest.raises(SearchProviderError):
            SearXNGProvider("https://searx.example").search("q", 5)

    def test_non_dict_items_skipped(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*", FakeResponse(json_body={
                "results": ["junk", {"title": "T", "url": "https://a.example/",
                                     "content": "c"}]}))
        out = SearXNGProvider("https://searx.example").search("q", 5)
        assert [r.url for r in out] == ["https://a.example/"]


class TestProviderFromConfig:
    def test_default_is_the_search_chain(self):
        from localm.web_retrieval import (BraveSearchProvider,
                                          DefaultSearchProvider,
                                          DuckDuckGoLiteProvider)
        p = provider_from_config({})
        assert isinstance(p, DefaultSearchProvider)
        assert [type(r) for r in p.routes] == [
            DuckDuckGoHTMLProvider, DuckDuckGoLiteProvider, BraveSearchProvider]

    def test_net_search_url_selects_searxng_and_strips_trailing_slash(self):
        p = provider_from_config({"net_search_url": "http://127.0.0.1:8080/"})
        assert isinstance(p, SearXNGProvider)
        assert p.base_url == "http://127.0.0.1:8080"

    def test_live_config_read_when_none_given(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_search_url": "https://s.example"})
        assert isinstance(provider_from_config(), SearXNGProvider)

    def test_unreadable_config_selects_duckduckgo_and_warns(self, monkeypatch,
                                                            caplog):
        def boom():
            raise OSError("config unreadable")
        monkeypatch.setattr("localm.config.load_config", boom)
        with caplog.at_level("WARNING", logger=netpolicy.logger.name):
            provider = provider_from_config()
        from localm.web_retrieval import DefaultSearchProvider
        assert isinstance(provider, DefaultSearchProvider)
        assert any("could not load config" in r.getMessage()
                   for r in caplog.records)


class TestSearchFunction:
    def test_empty_query_rejected_before_provider_call(self):
        stub = _StubProvider()
        with pytest.raises(ValueError):
            search("   ", provider=stub)
        assert stub.calls == []

    def test_max_results_clamped_to_one_through_ten(self):
        stub = _StubProvider(results=[SearchResult("t", "https://a.example/",
                                                   "s", 1, "stub")])
        search("q", max_results=50, provider=stub)
        search("q", max_results=0, provider=stub)
        assert [c[1] for c in stub.calls] == [10, 1]

    def test_query_stripped(self):
        stub = _StubProvider(results=[SearchResult("t", "https://a.example/",
                                                   "s", 1, "stub")])
        search("  hello  ", provider=stub)
        assert stub.calls[0][0] == "hello"

    def test_no_results_is_a_provider_error_and_a_runtime_error(self):
        with pytest.raises(SearchProviderError, match="no results"):
            search("q", provider=_StubProvider())
        with pytest.raises(RuntimeError):
            search("q", provider=_StubProvider())

    def test_provider_failure_propagates_with_no_fallback(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        with pytest.raises(ConnectionError):
            search("q", provider=_StubProvider(exc=ConnectionError("down")))
        assert t.calls == []


class TestNetpolicyWebSearchDelegate:
    def test_legacy_dict_shape_and_provider_selection(self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*", FakeResponse(
            json_body=searx_json([("T1", "https://a.example/", "c1")])))
        out = netpolicy.web_search("query")
        assert out == [{"title": "T1", "url": "https://a.example/",
                        "snippet": "c1"}]
        assert set(out[0]) == {"title", "url", "snippet"}
        assert t.urls("POST") == []

    def test_configured_searxng_failure_is_not_swapped_for_duckduckgo(
            self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*",
                lambda url, **kw: (_ for _ in ()).throw(ConnectionError("down")))
        with pytest.raises(ConnectionError):
            netpolicy.web_search("query")
        assert t.urls("POST") == []


def _reset_exc():
    import urllib3.exceptions
    return requests.ConnectionError(urllib3.exceptions.ProtocolError(
        "Connection aborted.", ConnectionResetError(
            10054, "An existing connection was forcibly closed by the remote "
            "host", None, 10054, None)))


def _ddg_ok():
    return FakeResponse(text=ddg_html([
        ("Matlan1/localm", "https://github.com/Matlan1/localm", "localm repo")]))


class TestDuckDuckGoResilience:
    def test_reset_then_success_returns_results(self, monkeypatch):
        from tests._web_retrieval_fixtures import no_sleep
        allow_public(monkeypatch)
        slept = no_sleep(monkeypatch)
        t = Transport().install(monkeypatch)
        answers = [_reset_exc(), _ddg_ok()]

        def flaky(url, **kw):
            nxt = answers.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        t.route("POST", DDG_ENDPOINT, flaky)
        out = DuckDuckGoHTMLProvider().search("Matlan1 LocalM GitHub repository", 5)
        assert [r.url for r in out] == ["https://github.com/Matlan1/localm"]
        assert len(t.urls("POST")) == 2
        assert slept == [1.0]

    def test_reset_everywhere_names_every_service_and_cause(self, monkeypatch):
        from localm.web_retrieval import retrieve
        from tests._web_retrieval_fixtures import no_sleep
        allow_public(monkeypatch)
        slept = no_sleep(monkeypatch)
        t = Transport().install(monkeypatch)
        reset = lambda url, **kw: (_ for _ in ()).throw(_reset_exc())  # noqa: E731
        t.route("POST", DDG_ENDPOINT, reset)
        t.route("POST", LITE_ENDPOINT, reset)
        t.route("GET", BRAVE_SEARCH + "*", reset)
        b = retrieve("Matlan1 LocalM GitHub repository")
        assert b.search_status == "failed"
        assert b.search_error == (
            "Web search failed on every search service localm tried ("
            "DuckDuckGo: html.duckduckgo.com closed the connection before "
            "answering; DuckDuckGo lite: lite.duckduckgo.com closed the "
            "connection before answering; Brave Search: search.brave.com "
            "closed the connection before answering).")
        for banned in ("10054", "localm config", "Try again", "try again"):
            assert banned not in b.search_error
        assert len(t.urls("POST")) == 6 and len(t.urls("GET")) == 3
        assert slept == [1.0, 2.0] * 3

    @pytest.mark.parametrize("status", [202, 403, 418, 429])
    def test_bot_check_status_is_reported_and_not_retried(self, monkeypatch,
                                                          status):
        from localm.web_retrieval.providers import BOT_CHECK_MESSAGE
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=status, text="x"))
        with pytest.raises(SearchProviderError) as info:
            DuckDuckGoHTMLProvider().search("q", 5)
        assert str(info.value) == BOT_CHECK_MESSAGE
        assert len(t.urls("POST")) == 1

    def test_challenge_form_page_is_a_bot_check(self, monkeypatch):
        from localm.web_retrieval.providers import (BOT_CHECK_MESSAGE,
                                                    BotCheckError)
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(text=(
            '<html><body><form id="challenge-form" action="/anomaly">'
            "</form></body></html>")))
        with pytest.raises(BotCheckError) as info:
            DuckDuckGoHTMLProvider().search("q", 5)
        assert str(info.value) == BOT_CHECK_MESSAGE
        assert len(t.urls("POST")) == 1

    def test_tls_failure_is_not_retried(self, monkeypatch):
        from tests._web_retrieval_fixtures import no_sleep
        allow_public(monkeypatch)
        slept = no_sleep(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, lambda url, **kw: (_ for _ in ()).throw(
            requests.exceptions.SSLError("certificate verify failed")))
        with pytest.raises(requests.exceptions.SSLError):
            DuckDuckGoHTMLProvider().search("q", 5)
        assert len(t.urls("POST")) == 1 and slept == []

    def test_requests_are_spaced(self, monkeypatch):
        import time

        from localm.web_retrieval import providers
        from tests._web_retrieval_fixtures import no_sleep
        allow_public(monkeypatch)
        slept = no_sleep(monkeypatch)
        monkeypatch.setattr(providers, "_MIN_INTERVAL", 5.0)
        monkeypatch.setitem(providers._last_request, "duckduckgo",
                            time.monotonic())
        Transport().install(monkeypatch).route("POST", DDG_ENDPOINT, _ddg_ok())
        DuckDuckGoHTMLProvider().search("q", 5)
        assert len(slept) == 1 and 4.0 < slept[0] <= 5.0

    def test_user_agent_names_this_project(self, monkeypatch):
        import localm
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, _ddg_ok())
        DuckDuckGoHTMLProvider().search("q", 5)
        ua = t.calls[0][2]["headers"]["User-Agent"]
        assert f"localm/{localm.__version__}" in ua
        assert "+https://github.com/Matlan1/localm" in ua
