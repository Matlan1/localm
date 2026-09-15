# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end retrieval through the real netpolicy path (policy check, pinned
transport seam, redirects, byte cap, charset decoding) with the socket replaced
by a URL-routed double (localm.web_retrieval.retrieve)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from localm import netpolicy
from localm.web_retrieval import (
    GROUNDING_FAILED,
    GROUNDING_PAGE_BACKED,
    GROUNDING_SNIPPET_ONLY,
    SearchResult,
    retrieve,
)
from tests._web_retrieval_fixtures import (
    ANSWER,
    DDG_ENDPOINT,
    QUERY,
    FakeResponse,
    Transport,
    allow_public,
    ddg_html,
    html_page,
    html_response,
    nav_heavy_page,
    searx_json,
)

_LONG = " ".join(f"Sentence number {i} about the visit." for i in range(15))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


def _page(text: str, title: str = "Page") -> str:
    return html_page(f"<main><h1>{title}</h1><p>{text}</p></main>", title=title)


def _search_route(t: Transport, rows) -> None:
    t.route("POST", DDG_ENDPOINT, FakeResponse(text=ddg_html(rows)))


class _StubProvider:
    name = "stub"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        return list(self.results)


class TestEvidenceStatesAndDuplicates:
    """Acceptance: a failed top result and a duplicate URL produce per-source
    status, no duplicate entries, and the retrieval does not fail."""

    @pytest.fixture
    def bundle(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [
            ("Broken", "https://a.example/down", "Snippet for the broken page"),
            ("Good B", "https://b.example/page", "Snippet B"),
            ("Dup of B", "http://www.b.example/page/?utm_source=x", "Snippet B2"),
            ("Good C", "https://c.example/page", "Snippet C"),
            ("Skipped D", "https://d.example/", "Snippet D"),
            ("Skipped E", "https://e.example/", ""),
        ])
        t.route("GET", "https://a.example/down", FakeResponse(status=503))
        t.route("GET", "https://b.example/page",
                html_response(_page(f"{_LONG} {ANSWER}", "Page B")))
        t.route("GET", "https://c.example/page",
                html_response(_page(_LONG, "Page C")))
        b = retrieve(QUERY)
        b._transport = t
        return b

    def test_sources_ids_ranks_and_no_duplicates(self, bundle):
        assert [s.id for s in bundle.sources] == ["S1", "S2", "S3", "S4", "S5"]
        assert [s.provider_rank for s in bundle.sources] == [1, 2, 4, 5, 6]
        assert len({s.canonical_url for s in bundle.sources}) == 5
        assert all("www.b.example" not in s.url for s in bundle.sources)

    def test_per_source_status_and_grounding(self, bundle):
        s1, s2, s3, s4, s5 = bundle.sources
        assert (s1.retrieval_status, s1.grounding) == ("failed", GROUNDING_FAILED)
        assert "HTTP 503" in s1.error
        assert (s2.retrieval_status, s2.grounding) == ("fetched", GROUNDING_PAGE_BACKED)
        assert (s3.retrieval_status, s3.grounding) == ("fetched", GROUNDING_PAGE_BACKED)
        assert (s4.retrieval_status, s4.grounding) == ("skipped", GROUNDING_SNIPPET_ONLY)
        assert (s5.retrieval_status, s5.grounding) == ("skipped", GROUNDING_FAILED)
        assert s2.final_url == "https://b.example/page"
        assert s2.region == "main" and s2.text_chars > 0
        assert s4.final_url is None

    def test_only_the_top_three_were_fetched(self, bundle):
        assert sorted(bundle._transport.urls("GET")) == [
            "https://a.example/down", "https://b.example/page",
            "https://c.example/page"]

    def test_bundle_is_page_backed_with_the_answer(self, bundle):
        assert bundle.search_status == "ok"
        assert bundle.grounding == GROUNDING_PAGE_BACKED
        assert bundle.page_backed is True
        assert ANSWER in bundle.evidence_text()
        assert any(c.source_id == "S2" and ANSWER in c.text for c in bundle.chunks)

    def test_failed_source_keeps_its_snippet_as_evidence(self, bundle):
        s1 = bundle.chunks_for("S1")
        assert len(s1) == 1 and s1[0].kind == "snippet"
        assert s1[0].text == "Snippet for the broken page"

    def test_budget_caps_hold(self, bundle):
        assert bundle.total_chars <= bundle.budget_chars
        for s in bundle.sources:
            assert sum(len(c.text) for c in bundle.chunks_for(s.id)) <= \
                bundle.per_source_cap_chars

    def test_to_dict_is_json_serialisable_and_complete(self, bundle):
        d = json.loads(json.dumps(bundle.to_dict()))
        assert d["grounding"] == GROUNDING_PAGE_BACKED
        assert [s["id"] for s in d["sources"]] == ["S1", "S2", "S3", "S4", "S5"]
        assert d["sources"][0]["error"].startswith("RuntimeError: HTTP 503")
        assert d["chunks"] and {"source_id", "text", "score", "offset", "kind"} \
            <= set(d["chunks"][0])

    def test_prompt_text_names_sources_and_states(self, bundle):
        text = bundle.to_prompt_text()
        assert "[S1] Broken - https://a.example/down (failed, RuntimeError: HTTP 503)" in text
        assert "[S2] Good B - https://b.example/page (page-backed)" in text
        assert "[S1 snippet] Snippet for the broken page" in text
        assert "[S2] " in text and ANSWER in text


class TestWebFunc002EndToEnd:
    """Acceptance: the 7,000-character-chrome reproduction has the answer in
    the evidence bundle."""

    @pytest.mark.parametrize("mode", ["semantic", "menus", "plain"])
    def test_answer_present_in_bundle(self, monkeypatch, mode):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("City guide", "https://city.example/guide", "guide")])
        t.route("GET", "https://city.example/guide",
                html_response(nav_heavy_page(mode)))
        b = retrieve(QUERY)
        assert b.grounding == GROUNDING_PAGE_BACKED
        assert ANSWER in b.evidence_text()
        assert b.total_chars <= b.budget_chars

    def test_legacy_prefix_would_have_missed_it(self):
        legacy = netpolicy.html_to_text(nav_heavy_page("plain"))
        assert ANSWER not in legacy[:6000]
        assert ANSWER in legacy


class TestCharset:
    """Acceptance: text declared as iso-8859-1 round-trips unchanged."""

    def test_iso_8859_1_page_text_reaches_the_bundle(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Linz", "https://linz.example/", "Linz page")])
        markup = html_page(f"<main><p>{_LONG} Grüße aus Linz, {ANSWER}</p></main>",
                           charset="iso-8859-1")
        t.route("GET", "https://linz.example/",
                html_response(markup, charset="iso-8859-1"))
        b = retrieve(QUERY)
        assert "Grüße aus Linz" in b.evidence_text()
        assert "Gr��e" not in b.evidence_text()

    def test_declared_codec_that_is_not_a_text_encoding_falls_back_to_utf8(
            self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Linz", "https://linz.example/", "Linz page")])
        markup = html_page(f"<main><p>{_LONG} Grüße aus Linz. {ANSWER}</p></main>")
        t.route("GET", "https://linz.example/",
                html_response(markup, charset="utf-8",
                              content_type="text/html; charset=base64"))
        b = retrieve(QUERY)
        assert b.sources[0].grounding == GROUNDING_PAGE_BACKED
        assert "Grüße aus Linz" in b.evidence_text()

    def test_meta_charset_used_when_header_has_none(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Linz", "https://linz.example/", "Linz page")])
        markup = html_page(f"<main><p>{_LONG} Grüße aus Linz. {ANSWER}</p></main>",
                           charset="windows-1252")
        t.route("GET", "https://linz.example/",
                html_response(markup, charset="windows-1252",
                              content_type="text/html"))
        b = retrieve(QUERY)
        assert "Grüße aus Linz" in b.evidence_text()


class TestSearchFailures:
    def test_configured_searxng_failure_is_explicit_and_never_duckduckgo(
            self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        t = Transport().install(monkeypatch)
        t.route("GET", "https://searx.example/search?*",
                lambda url, **kw: (_ for _ in ()).throw(ConnectionError("down")))
        b = retrieve(QUERY)
        assert b.provider == "searxng"
        assert b.search_status == "failed"
        assert b.search_error == "ConnectionError: down"
        assert b.sources == [] and b.chunks == []
        assert b.grounding == GROUNDING_FAILED
        assert t.urls("POST") == []
        assert "Search failed: ConnectionError: down" in b.to_prompt_text()

    def test_searxng_http_error_is_in_the_bundle(self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*", FakeResponse(status=500))
        b = retrieve(QUERY)
        assert b.search_status == "failed" and "HTTP 500" in b.search_error

    def test_searxng_empty_results_is_empty_not_failed(self, monkeypatch):
        allow_public(monkeypatch, net_search_url="https://searx.example")
        Transport().install(monkeypatch).route(
            "GET", "https://searx.example/search?*",
            FakeResponse(json_body=searx_json([])))
        b = retrieve(QUERY)
        assert b.search_status == "empty" and b.sources == []
        assert b.grounding == GROUNDING_FAILED

    def test_duckduckgo_unparseable_page_is_empty(self, monkeypatch):
        allow_public(monkeypatch)
        Transport().install(monkeypatch).route(
            "POST", DDG_ENDPOINT,
            FakeResponse(text="<html><body>captcha</body></html>"))
        assert retrieve(QUERY).search_status == "empty"

    def test_policy_off_raises(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"net_mode": "off"})
        with pytest.raises(netpolicy.NetworkPolicyError):
            retrieve(QUERY)

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            retrieve("   ", provider=_StubProvider([]))


class TestPageFailures:
    def test_denied_domain_is_a_per_source_policy_failure(self, monkeypatch):
        allow_public(monkeypatch, net_deny=["denied.example"])
        t = Transport().install(monkeypatch)
        _search_route(t, [("Denied", "https://denied.example/x", "sd"),
                          ("Ok", "https://ok.example/", "so")])
        t.route("GET", "https://ok.example/", html_response(_page(_LONG)))
        b = retrieve(QUERY)
        s1, s2 = b.sources
        assert s1.grounding == GROUNDING_FAILED
        assert s1.error.startswith("refused by policy:") and "deny list" in s1.error
        assert s2.grounding == GROUNDING_PAGE_BACKED
        assert "https://denied.example/x" not in t.urls("GET")

    def test_redirect_into_private_space_is_a_per_source_failure(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Jump", "https://jump.example/", "sj"),
                          ("Ok", "https://ok.example/", "so")])
        t.route("GET", "https://jump.example/",
                FakeResponse(status=302, redirect="http://127.0.0.1/admin"))
        t.route("GET", "https://ok.example/", html_response(_page(_LONG)))
        b = retrieve(QUERY)
        assert b.sources[0].grounding == GROUNDING_FAILED
        assert "refused by policy" in b.sources[0].error
        assert b.sources[1].grounding == GROUNDING_PAGE_BACKED

    def test_all_reads_fail_is_snippet_only(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("A", "https://a.example/", "snippet a"),
                          ("B", "https://b.example/", "snippet b")])
        t.route("GET", "https://a.example/", FakeResponse(status=500))
        t.route("GET", "https://b.example/",
                lambda url, **kw: (_ for _ in ()).throw(OSError("reset")))
        b = retrieve(QUERY)
        assert b.grounding == GROUNDING_SNIPPET_ONLY
        assert b.page_backed is False
        assert all(c.kind == "snippet" for c in b.chunks)
        assert all(s.grounding == GROUNDING_FAILED for s in b.sources)
        assert b.sources[1].error == "OSError: reset"

    def test_page_without_text_is_snippet_only(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Empty", "https://empty.example/", "the snippet")])
        t.route("GET", "https://empty.example/",
                html_response("<html><body><script>x()</script></body></html>"))
        b = retrieve(QUERY)
        s = b.sources[0]
        assert (s.retrieval_status, s.grounding) == ("fetched", GROUNDING_SNIPPET_ONLY)
        assert s.error == "page had no extractable text"
        assert [c.kind for c in b.chunks] == ["snippet"]

    def test_two_urls_resolving_to_one_final_page_keep_one(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("X", "https://a.example/x", "sx"),
                          ("Y", "https://a.example/y", "sy")])
        final = html_response(_page(f"{_LONG} {ANSWER}"))
        t.route("GET", "https://a.example/x",
                FakeResponse(status=301, redirect="https://a.example/final"))
        t.route("GET", "https://a.example/y",
                FakeResponse(status=302, redirect="https://a.example/final/"))
        t.route("GET", "https://a.example/final", final)
        t.route("GET", "https://a.example/final/", final)
        b = retrieve(QUERY)
        s1, s2 = b.sources
        assert s1.grounding == GROUNDING_PAGE_BACKED
        assert s1.final_url == "https://a.example/final"
        assert (s2.retrieval_status, s2.grounding) == ("duplicate", GROUNDING_SNIPPET_ONLY)
        assert s2.error == "same page as S1"
        assert [c.kind for c in b.chunks_for("S2")] == ["snippet"]
        assert sum(1 for c in b.chunks if c.kind == "page" and ANSWER in c.text) == 1

    def test_straggler_past_deadline_is_failed_not_blocking(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Slow", "https://slow.example/", "ss"),
                          ("Fast", "https://fast.example/", "sf")])

        def slow(url, **kw):
            time.sleep(1.5)
            return html_response(_page(_LONG))
        t.route("GET", "https://slow.example/", slow)
        t.route("GET", "https://fast.example/", html_response(_page(_LONG)))
        started = time.monotonic()
        b = retrieve(QUERY, deadline_seconds=0.4)
        assert time.monotonic() - started < 1.4
        assert b.sources[0].grounding == GROUNDING_FAILED
        assert b.sources[0].error == "timed out after 0.4s"
        assert b.sources[1].grounding == GROUNDING_PAGE_BACKED


class TestConcurrencyAndOptions:
    def test_top_pages_are_read_concurrently(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        urls = [f"https://p{i}.example/" for i in range(3)]
        _search_route(t, [(f"P{i}", u, f"s{i}") for i, u in enumerate(urls)])
        barrier = threading.Barrier(3, timeout=5)

        def gated(url, **kw):
            barrier.wait()
            return html_response(_page(_LONG))
        for u in urls:
            t.route("GET", u, gated)
        b = retrieve(QUERY)
        assert [s.grounding for s in b.sources] == [GROUNDING_PAGE_BACKED] * 3

    def test_fetch_top_zero_reads_nothing(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("A", "https://a.example/", "sa")])
        b = retrieve(QUERY, fetch_top=0)
        assert t.urls("GET") == []
        assert b.sources[0].retrieval_status == "skipped"
        assert b.grounding == GROUNDING_SNIPPET_ONLY

    def test_non_html_body_used_verbatim(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("Txt", "https://t.example/notes.txt", "st")])
        t.route("GET", "https://t.example/notes.txt", FakeResponse(
            headers={"Content-Type": "text/plain; charset=utf-8"},
            body=f"  {_LONG}\n{ANSWER}\n"))
        b = retrieve(QUERY)
        assert b.sources[0].region == "text"
        assert b.sources[0].grounding == GROUNDING_PAGE_BACKED
        assert ANSWER in b.evidence_text()

    def test_html_sniffed_when_content_type_missing(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("H", "https://h.example/", "sh")])
        t.route("GET", "https://h.example/",
                FakeResponse(headers={}, body=_page(_LONG)))
        b = retrieve(QUERY)
        assert b.sources[0].region == "main"

    def test_page_title_fills_an_empty_provider_title(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search_route(t, [("", "https://a.example/", "sa")])
        t.route("GET", "https://a.example/", html_response(_page(_LONG, "Real Title")))
        assert retrieve(QUERY).sources[0].title == "Real Title"

    def test_injected_provider_and_fetcher_bypass_nothing_but_the_network(self):
        provider = _StubProvider([
            SearchResult("T", "https://a.example/", "s", 1, "stub")])
        seen = {}

        def fetch(url, *, timeout):
            seen["url"], seen["timeout"] = url, timeout
            return "https://a.example/final", "text/html", _page(_LONG)
        b = retrieve(QUERY, provider=provider, fetch=fetch, fetch_timeout=7,
                     search_candidates=50)
        assert provider.calls == [(QUERY, 10)]
        assert seen == {"url": "https://a.example/", "timeout": 7}
        assert b.provider == "stub"
        assert b.sources[0].final_url == "https://a.example/final"
        assert b.grounding == GROUNDING_PAGE_BACKED
