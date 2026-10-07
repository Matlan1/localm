# SPDX-License-Identifier: AGPL-3.0-or-later
"""The built-in search chain recovers on its own before reporting anything.

DuckDuckGo's HTML page, then DuckDuckGo's lite page, then Brave Search; a
bot-checked service is asked once more after a pause; a configured SearXNG
instance is repaired (URL normalised, HTML results when JSON is refused) and
never swapped for a public service. Every request goes through the real
policy check and the ``_session_for`` transport seam (``Transport``).
"""

from __future__ import annotations

import urllib.parse

import pytest

from localm import netpolicy
from localm.web_retrieval import (
    BraveSearchProvider,
    DefaultSearchProvider,
    DuckDuckGoLiteProvider,
    SearXNGProvider,
    retrieve,
)
from localm.web_retrieval.providers import BotCheckError
from tests._web_retrieval_fixtures import (
    BRAVE_SEARCH,
    DDG_ENDPOINT,
    LITE_ENDPOINT,
    FakeResponse,
    Transport,
    allow_public,
    ddg_html,
    no_sleep,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


def lite_html(rows) -> str:
    """lite.duckduckgo.com markup for ``(title, url, snippet)`` rows (the
    shape measured on 2026-10-07)."""
    out = ['<html><body><table border="0">']
    for i, (title, url, snippet) in enumerate(rows, 1):
        out.append(
            f'<tr><td valign="top">{i}.&nbsp;</td><td>'
            f'<a rel="nofollow" href="{url}" class=\'result-link\'>{title}</a>'
            f"</td></tr><tr><td>&nbsp;</td>"
            f"<td class='result-snippet'>{snippet}</td></tr>"
            f"<tr><td>&nbsp;</td><td><span class='link-text'>x</span></td></tr>")
    out.append("</table></body></html>")
    return "".join(out)


def brave_html(rows) -> str:
    """search.brave.com markup for ``(title, url, snippet)`` rows (the shape
    measured on 2026-10-07, svelte class suffixes included)."""
    out = ['<html><body><section id="mixed-main">']
    for i, (title, url, snippet) in enumerate(rows):
        out.append(
            f'<div class="snippet svelte-jmfu5f" data-pos="{i}" data-type="web">'
            '<div class="result-body"><div class="result-wrapper">'
            '<div class="result-content">'
            f'<a href="{url}" target="_self" class="svelte-14r20fy l1">'
            '<div class="site-name-wrapper"><cite class="snippet-url">x</cite>'
            '</div>'
            f'<div class="title search-snippet-title line-clamp-1" title="{title}">'
            f"{title}</div></a>"
            '<div class="generic-snippet svelte-1cwdgg3"><div class="content">'
            f"{snippet}</div></div>"
            '<div class="item-attributes"><div><strong>Author:</strong>'
            "<span>someone</span></div></div>"
            "</div></div></div></div>")
    out.append("</section></body></html>")
    return "".join(out)


def _ok(text: str) -> FakeResponse:
    return FakeResponse(headers={"Content-Type": "text/html"}, text=text)


def _raise(exc_factory):
    return lambda url, **kw: (_ for _ in ()).throw(exc_factory())


def _reset():
    import requests
    import urllib3.exceptions
    return requests.ConnectionError(urllib3.exceptions.ProtocolError(
        "Connection aborted.", ConnectionResetError(10054, "reset")))


ROW = ("Matlan1/localm", "https://github.com/Matlan1/localm", "localm repo")
BANNED = ("localm config", "Try again", "try again", "10054")


class TestParsers:
    def test_lite_parser_reads_results_and_drops_ad_links(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", LITE_ENDPOINT, _ok(lite_html([
            ("Ad", "https://duckduckgo.com/y.js?ad_domain=x", "sponsored"),
            ROW,
            ("PyPI", "https://pypi.org/project/localm/", "Package page"),
        ])))
        out = DuckDuckGoLiteProvider().search("q", 5)
        assert [(r.title, r.url, r.snippet, r.rank) for r in out] == [
            ("Matlan1/localm", "https://github.com/Matlan1/localm",
             "localm repo", 1),
            ("PyPI", "https://pypi.org/project/localm/", "Package page", 2)]
        assert all(r.provider == "duckduckgo-lite" for r in out)
        method, url, kw = t.calls[0]
        assert kw["data"]["q"] == "q"
        assert kw["headers"]["Referer"] == "https://lite.duckduckgo.com/"

    def test_brave_parser_reads_title_url_snippet(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", BRAVE_SEARCH + "*", _ok(brave_html([
            ROW, ("Docs &amp; more", "https://docs.example/a", "Docs text")])))
        out = BraveSearchProvider().search("localm repo", 5)
        assert [(r.title, r.url, r.snippet) for r in out] == [
            ("Matlan1/localm", "https://github.com/Matlan1/localm",
             "localm repo"),
            ("Docs & more", "https://docs.example/a", "Docs text")]
        query = urllib.parse.parse_qs(urllib.parse.urlparse(t.urls("GET")[0]).query)
        assert query["q"] == ["localm repo"]

    def test_brave_captcha_page_is_a_bot_check(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "GET", BRAVE_SEARCH + "*",
            _ok("<html><body>Please solve the captcha</body></html>"))
        with pytest.raises(BotCheckError):
            BraveSearchProvider().search("q", 5)


class TestFallbackChain:
    def test_ddg_reset_falls_back_to_lite(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, _raise(_reset))
        t.route("POST", LITE_ENDPOINT, _ok(lite_html([ROW])))
        b = retrieve("localm repo", fetch_top=0)
        assert b.search_status == "ok"
        assert b.provider == "duckduckgo-lite"
        assert [s.url for s in b.sources] == [ROW[1]]
        assert t.urls("POST") == [DDG_ENDPOINT] * 3 + [LITE_ENDPOINT]

    @pytest.mark.parametrize("status", [202, 403, 418, 429])
    def test_ddg_bot_check_falls_back_without_retrying_it(self, monkeypatch,
                                                         status):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=status, text="x"))
        t.route("POST", LITE_ENDPOINT, _ok(lite_html([ROW])))
        out = DefaultSearchProvider().search("q", 5)
        assert [r.provider for r in out] == ["duckduckgo-lite"]
        assert t.urls("POST") == [DDG_ENDPOINT, LITE_ENDPOINT]

    def test_unparseable_ddg_page_falls_back(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, _ok("<html><body>changed</body></html>"))
        t.route("POST", LITE_ENDPOINT, _ok("<html><body>also</body></html>"))
        t.route("GET", BRAVE_SEARCH + "*", _ok(brave_html([ROW])))
        out = DefaultSearchProvider().search("q", 5)
        assert [(r.provider, r.url) for r in out] == [("brave", ROW[1])]

    def test_both_ddg_pages_bot_checked_brave_answers(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=202, text=""))
        t.route("POST", LITE_ENDPOINT, FakeResponse(status=429, text=""))
        t.route("GET", BRAVE_SEARCH + "*", _ok(brave_html([ROW])))
        b = retrieve("q", fetch_top=0)
        assert b.search_status == "ok" and b.provider == "brave"

    def test_bot_checked_service_is_asked_again_after_a_pause(self, monkeypatch):
        allow_public(monkeypatch)
        slept = no_sleep(monkeypatch)
        t = Transport().install(monkeypatch)
        answers = [FakeResponse(status=202, text=""),
                   _ok(ddg_html([ROW]))]
        t.route("POST", DDG_ENDPOINT, lambda url, **kw: answers.pop(0))
        t.route("POST", LITE_ENDPOINT, _raise(_reset))
        t.route("GET", BRAVE_SEARCH + "*", FakeResponse(status=429, text=""))
        out = DefaultSearchProvider().search("q", 5)
        assert [r.provider for r in out] == ["duckduckgo-html"]
        assert 3.0 in slept
        assert t.urls("POST").count(DDG_ENDPOINT) == 2

    def test_every_service_failing_names_each_cause(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=202, text=""))
        t.route("POST", LITE_ENDPOINT, _raise(_reset))
        t.route("GET", BRAVE_SEARCH + "*", FakeResponse(status=503, text=""))
        b = retrieve("q")
        assert b.search_status == "failed"
        assert b.search_error == (
            "Web search failed on every search service localm tried ("
            "DuckDuckGo: it answered with a bot check instead of results; "
            "DuckDuckGo lite: lite.duckduckgo.com closed the connection before "
            "answering; Brave Search: search.brave.com had a server error, "
            "HTTP 503). These services limit automated searches from one "
            "network; a self-hosted SearXNG search backend can be set under "
            "Settings > Network.")
        for banned in BANNED:
            assert banned not in b.search_error

    def test_all_services_answering_empty_is_an_empty_search(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, _ok(ddg_html([])))
        t.route("POST", LITE_ENDPOINT, _ok(lite_html([])))
        t.route("GET", BRAVE_SEARCH + "*", _ok(brave_html([])))
        b = retrieve("q")
        assert b.search_status == "empty"

    def test_out_of_time_services_are_named(self, monkeypatch):
        from localm.web_retrieval import providers
        allow_public(monkeypatch)
        monkeypatch.setattr(providers, "_SEARCH_BUDGET", 0.0)
        t = Transport().install(monkeypatch)
        b = retrieve("q")
        assert t.calls == []
        assert b.search_error == (
            "Web search failed on every search service localm tried ("
            "DuckDuckGo: not tried: out of time; DuckDuckGo lite: not tried: "
            "out of time; Brave Search: not tried: out of time).")


class TestPolicy:
    def test_network_off_refuses_before_any_request(self, monkeypatch):
        allow_public(monkeypatch, net_mode="off")
        t = Transport().install(monkeypatch)
        with pytest.raises(netpolicy.NetworkPolicyError) as info:
            retrieve("q")
        assert info.value.off is True
        assert t.calls == []

    def test_denied_duckduckgo_goes_straight_to_brave(self, monkeypatch):
        allow_public(monkeypatch, net_deny=["duckduckgo.com"])
        t = Transport().install(monkeypatch)
        t.route("GET", BRAVE_SEARCH + "*", _ok(brave_html([ROW])))
        b = retrieve("q", fetch_top=0)
        assert b.provider == "brave"
        assert t.urls("POST") == []

    def test_policy_blocked_service_named_when_others_fail(self, monkeypatch):
        allow_public(monkeypatch, net_deny=["search.brave.com"])
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=503, text=""))
        t.route("POST", LITE_ENDPOINT, FakeResponse(status=503, text=""))
        b = retrieve("q")
        assert "Brave Search: blocked by the network policy" in b.search_error
        assert t.urls("GET") == []


class TestConfiguredSearXNG:
    def test_failure_never_reaches_a_public_service(self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*", _raise(_reset))
        b = retrieve("q")
        assert b.search_status == "failed"
        assert b.search_error == ("The search backend set in Settings > "
                                  "Network failed: searx.example closed the "
                                  "connection before answering.")
        assert all("searx.example" in u for u in t.urls())

    def test_json_refused_reads_the_html_results(self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        t = Transport().install(monkeypatch)

        def answer(url, **kw):
            if "format=json" in url:
                return FakeResponse(status=403, text="Forbidden")
            return _ok(
                '<main><article class="result result-default category-general">'
                '<a href="https://github.com/Matlan1/localm" class="url_header">'
                '<div class="url_wrapper">github.com</div></a>'
                '<h3><a href="https://github.com/Matlan1/localm">Matlan1/localm'
                "</a></h3><p class=\"content\">localm repo</p></article>"
                '<article class="result result-default">'
                '<h3><a href="https://pypi.org/project/localm/">PyPI</a></h3>'
                '<p class="content empty_element">This site did not provide any '
                "description.</p></article></main>")
        t.route("GET", "https://searx.example/search?*", answer)
        out = SearXNGProvider("https://searx.example").search("q", 5)
        assert [(r.title, r.url, r.snippet) for r in out] == [
            ("Matlan1/localm", "https://github.com/Matlan1/localm",
             "localm repo"),
            ("PyPI", "https://pypi.org/project/localm/", "")]
        assert len(t.urls("GET")) == 2

    @pytest.mark.parametrize("configured", [
        "https://searx.example/search?q=old",
        "https://searx.example/search/",
        "https://searx.example/#top",
        "https://searx.example//",
    ])
    def test_configured_url_is_normalised(self, monkeypatch, configured):
        allow_public(monkeypatch, net_search_url=configured)
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*", FakeResponse(
            json_body={"results": [{"title": "T", "url": ROW[1],
                                    "content": "c"}]}))
        b = retrieve("hello", fetch_top=0)
        assert b.search_status == "ok"
        assert t.urls("GET")[0].startswith("https://searx.example/search?q=hello")


class TestPrivacyEcho:
    def test_coder_web_search_names_every_service_asked(self, monkeypatch,
                                                        capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_web_search
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(status=202, text=""))
        t.route("POST", LITE_ENDPOINT, _ok(lite_html([ROW])))
        t.route("GET", ROW[1], _ok("<html><body><main><p>"
                                   + "localm is a local tool. " * 20
                                   + "</p></main></body></html>"))
        t.route("GET", "https://raw.githubusercontent.com/*",
                FakeResponse(status=404))
        t.route("GET", "https://api.github.com/*", FakeResponse(status=404))
        tool_web_search(tmp_path, "localm", max_results=1, _privacy=True)
        err = capsys.readouterr().err
        assert f"[localm privacy] web_search endpoint: {DDG_ENDPOINT}" in err
        assert f"[localm privacy] web_search endpoint: {LITE_ENDPOINT}" in err
