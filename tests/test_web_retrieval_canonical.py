# SPDX-License-Identifier: AGPL-3.0-or-later
"""URL canonicalization and duplicate removal (localm.web_retrieval.canonical)."""

from __future__ import annotations

import pytest

from localm.web_retrieval import (
    SearchResult,
    canonicalize_url,
    dedup_key,
    dedup_results,
)
from localm.web_retrieval.canonical import is_fetchable


def _r(url: str, rank: int) -> SearchResult:
    return SearchResult(title=f"t{rank}", url=url, snippet=f"s{rank}",
                        rank=rank, provider="test")


class TestCanonicalizeUrl:
    def test_case_port_query_order_and_fragment(self):
        assert canonicalize_url(
            "HTTP://Example.COM:80/a/b/?utm_source=x&b=2&a=1#frag"
        ) == "http://example.com/a/b?a=1&b=2"

    def test_https_default_port_dropped_non_default_kept(self):
        assert canonicalize_url("https://a.example:443/x") == "https://a.example/x"
        assert canonicalize_url("https://a.example:8443/x") == \
            "https://a.example:8443/x"

    def test_empty_path_becomes_root_and_root_keeps_its_slash(self):
        assert canonicalize_url("https://a.example") == "https://a.example/"
        assert canonicalize_url("https://a.example/") == "https://a.example/"

    def test_trailing_dot_host_and_surrounding_whitespace(self):
        assert canonicalize_url("  https://A.Example./p/  ") == "https://a.example/p"

    @pytest.mark.parametrize("param", ["utm_campaign", "fbclid", "gclid", "msclkid"])
    def test_tracking_parameters_removed(self, param):
        assert canonicalize_url(f"https://a.example/p?{param}=1&q=2") == \
            "https://a.example/p?q=2"

    def test_ordinary_parameters_survive(self):
        assert canonicalize_url("https://a.example/p?ref=abc&id=7") == \
            "https://a.example/p?id=7&ref=abc"

    def test_blank_valued_parameter_kept(self):
        assert canonicalize_url("https://a.example/p?flag=&x=1") == \
            "https://a.example/p?flag=&x=1"

    def test_userinfo_kept(self):
        assert canonicalize_url("https://user:pw@a.example/x") == \
            "https://user:pw@a.example/x"

    @pytest.mark.parametrize("url", ["mailto:x@y.example", "javascript:alert(1)",
                                     "ftp://a.example/f", "not a url"])
    def test_non_http_returned_stripped_but_otherwise_unchanged(self, url):
        assert canonicalize_url(f"  {url} ") == url

    def test_unparseable_url_does_not_raise(self):
        assert isinstance(canonicalize_url("https://[bad/x"), str)


class TestDedupKey:
    def test_scheme_and_www_insensitive(self):
        a = dedup_key("http://www.a.example/x/")
        b = dedup_key("https://a.example/x")
        assert a == b

    def test_different_paths_differ(self):
        assert dedup_key("https://a.example/x") != dedup_key("https://a.example/y")

    def test_different_query_values_differ(self):
        assert dedup_key("https://a.example/x?id=1") != \
            dedup_key("https://a.example/x?id=2")

    def test_www_only_stripped_as_a_prefix_label(self):
        assert dedup_key("https://wwwx.example/") != dedup_key("https://x.example/")


class TestIsFetchable:
    @pytest.mark.parametrize("url,expected", [
        ("https://a.example/", True),
        ("http://a.example", True),
        ("mailto:x@y.example", False),
        ("javascript:alert(1)", False),
        ("", False),
        ("https:///nohost", False),
        ("/relative/path", False),
    ])
    def test_cases(self, url, expected):
        assert is_fetchable(url) is expected


class TestDedupResults:
    def test_first_occurrence_wins_and_keeps_its_rank(self):
        out = dedup_results([
            _r("https://a.example/page", 1),
            _r("https://b.example/", 2),
            _r("http://www.a.example/page/?utm_source=mail", 3),
            _r("https://c.example/", 4),
        ])
        assert [r.url for r in out] == ["https://a.example/page",
                                        "https://b.example/",
                                        "https://c.example/"]
        assert [r.rank for r in out] == [1, 2, 4]

    def test_unfetchable_results_dropped(self):
        out = dedup_results([_r("mailto:x@y.example", 1), _r("", 2),
                             _r("https://ok.example/", 3)])
        assert [r.rank for r in out] == [3]

    def test_empty_input(self):
        assert dedup_results([]) == []

    def test_original_result_objects_returned_unchanged(self):
        first = _r("https://a.example/x#frag", 1)
        assert dedup_results([first]) == [first]
        assert first.url.endswith("#frag")
