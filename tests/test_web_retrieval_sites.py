# SPDX-License-Identifier: AGPL-3.0-or-later
"""GitHub and Stack Exchange URLs are read from the site's content endpoint.

A search hit or a ``fetch_url`` on ``github.com/<owner>/<repo>`` yields the
README (raw file, then the REST readme endpoint, then the HTML page); a blob
URL yields the raw file; a Stack Exchange question yields the question and
its top answers from the Stack Exchange API. Every request goes through the
real policy check and the ``_session_for`` transport seam (``Transport``).
"""

from __future__ import annotations

import base64
import json

import pytest

from localm import netpolicy
from localm.web_retrieval import retrieve
from localm.web_retrieval.sites import site_reader
from tests._web_retrieval_fixtures import (
    DDG_ENDPOINT,
    FakeResponse,
    Transport,
    allow_public,
    ddg_html,
    html_page,
    html_response,
)

README = ("# localm\n\nlocalm is an offline local-LLM inference and plugin "
          "engine. It downloads and runs local models itself.\n")
RAW_README = "https://raw.githubusercontent.com/Matlan1/localm/HEAD/README.md"
API_README = "https://api.github.com/repos/Matlan1/localm/readme"
REPO = "https://github.com/Matlan1/localm"
FILE_LIST_CHROME = html_page(
    "<main><div>Notifications Fork Star Branches Go to file</div>"
    + "<div>localm folder localm folder</div>" * 40 + "</main>")

SO_URL = ("https://stackoverflow.com/questions/39907742/"
          "github-api-is-responding-with-a-403")
SO_API_Q = ("https://api.stackexchange.com/2.3/questions/39907742?"
            "site=stackoverflow&filter=withbody")
SO_API_A = ("https://api.stackexchange.com/2.3/questions/39907742/answers?"
            "site=stackoverflow&filter=withbody&sort=votes&order=desc&pagesize=3")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


def _text(body: str, ctype: str = "text/plain; charset=utf-8") -> FakeResponse:
    return FakeResponse(headers={"Content-Type": ctype}, body=body)


def _api_readme(text: str, name: str = "README.rst") -> FakeResponse:
    payload = {"name": name, "encoding": "base64",
               "content": base64.b64encode(text.encode()).decode()}
    return _text(json.dumps(payload), "application/json; charset=utf-8")


def _search(t: Transport, url: str, snippet: str = "repo snippet") -> None:
    t.route("POST", DDG_ENDPOINT,
            FakeResponse(text=ddg_html([("Hit", url, snippet)])))


class TestGitHubRepository:
    def test_search_hit_reads_the_raw_readme(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", RAW_README, _text(README))
        b = retrieve("localm repository", search_candidates=1)
        src = b.sources[0]
        assert src.retrieval_status == "fetched"
        assert src.grounding == "page-backed"
        assert src.final_url == RAW_README
        assert any("offline local-LLM inference" in c.text for c in b.chunks)
        assert REPO not in t.urls("GET") and API_README not in t.urls("GET")

    def test_raw_missing_falls_back_to_the_rest_readme(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", RAW_README, FakeResponse(status=404, body="404: Not Found"))
        t.route("GET", API_README, _api_readme("Python README in rst form. " * 5))
        b = retrieve("localm repository", search_candidates=1)
        assert b.sources[0].final_url == API_README
        assert any("README in rst form" in c.text for c in b.chunks)
        assert t.urls("GET") == [RAW_README, API_README]

    def test_both_endpoints_failing_falls_back_to_the_page(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", RAW_README, FakeResponse(status=404))
        t.route("GET", API_README, FakeResponse(status=403))
        t.route("GET", REPO, html_response(FILE_LIST_CHROME))
        b = retrieve("localm repository", search_candidates=1)
        assert t.urls("GET") == [RAW_README, API_README, REPO]
        assert b.sources[0].final_url == REPO

    def test_page_failure_is_what_gets_reported(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", RAW_README, FakeResponse(status=404))
        t.route("GET", API_README, FakeResponse(status=404))
        t.route("GET", REPO, FakeResponse(status=503))
        b = retrieve("localm repository", search_candidates=1)
        assert b.sources[0].error == "github.com had a server error, HTTP 503"

    def test_denied_github_is_refused_before_any_request(self, monkeypatch):
        allow_public(monkeypatch, net_deny=["github.com"])
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        b = retrieve("localm repository", search_candidates=1)
        assert b.sources[0].retrieval_status == "failed"
        assert "refused by policy" in b.sources[0].error
        assert t.urls("GET") == []

    def test_allow_list_without_raw_hosts_reads_the_page(self, monkeypatch):
        allow_public(monkeypatch, net_allow=["github.com", "duckduckgo.com"])
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", API_README, FakeResponse(status=404))
        t.route("GET", REPO, html_response(FILE_LIST_CHROME))
        b = retrieve("localm repository", search_candidates=1)
        assert t.urls("GET") == [API_README, REPO]
        assert b.sources[0].final_url == REPO

    def test_direct_fetch_text_reads_the_readme(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", RAW_README, _text(README))
        final_url, text = netpolicy.fetch_text("https://www.github.com/Matlan1/localm.git/")
        assert final_url == RAW_README
        assert text == README.strip()

    def test_tree_url_reads_that_directory_readme(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        raw = ("https://raw.githubusercontent.com/python/cpython/main/Doc/"
               "README.md")
        api = "https://api.github.com/repos/python/cpython/readme/Doc?ref=main"
        t.route("GET", raw, FakeResponse(status=404))
        t.route("GET", api, _api_readme("Python documentation README. " * 4))
        final_url, text = netpolicy.fetch_text(
            "https://github.com/python/cpython/tree/main/Doc")
        assert t.urls("GET") == [raw, api]
        assert final_url == api
        assert text.startswith("Python documentation README.")

    def test_repo_and_tree_hits_with_the_same_readme_count_once(self,
                                                                monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("POST", DDG_ENDPOINT, FakeResponse(text=ddg_html([
            ("Repo", REPO, "repo snippet"),
            ("Tree", REPO + "/tree/master", "tree snippet")])))
        t.route("GET", RAW_README, _text(README))
        t.route("GET", "https://raw.githubusercontent.com/Matlan1/localm/"
                "master/README.md", _text(README + "\n"))
        b = retrieve("localm repository", search_candidates=2)
        assert b.sources[0].grounding == "page-backed"
        assert b.sources[1].retrieval_status == "duplicate"
        assert b.sources[1].error == "same page as S1"
        assert not any(c.source_id == "S2" and c.kind == "page"
                       for c in b.chunks)

    def test_blob_url_reads_the_raw_file(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        raw = ("https://raw.githubusercontent.com/Matlan1/localm/master/"
               "docs/privacy.md")
        t.route("GET", raw, _text("# Privacy\n\nNothing is written.\n"))
        final_url, text = netpolicy.fetch_text(
            "https://github.com/Matlan1/localm/blob/master/docs/privacy.md")
        assert final_url == raw
        assert text.startswith("# Privacy")


class TestUrlRecognition:
    @pytest.mark.parametrize("url", [
        "https://github.com/topics/python",
        "https://github.com/orgs/python",
        "https://github.com/Matlan1",
        "https://github.com/Matlan1/localm/issues/12",
        "https://github.com/Matlan1/localm/tree",
        "https://gist.github.com/Matlan1/abc",
        "https://github.com:8443/Matlan1/localm",
        "ftp://github.com/Matlan1/localm",
        "https://stackoverflow.com/users/1/x",
        "https://stackoverflow.com/questions/tagged/python",
        "https://api.stackexchange.com/questions/1",
        "https://example.com/questions/12",
    ])
    def test_not_recognised(self, url):
        assert site_reader(url) is None

    @pytest.mark.parametrize("url", [
        "https://github.com/Matlan1/localm",
        "http://www.github.com/Matlan1/localm/",
        "https://github.com/Matlan1/localm/blob/master/README.md",
        "https://github.com/Matlan1/localm/tree/master",
        "https://github.com/Matlan1/localm/tree/master/docs",
        "https://stackoverflow.com/questions/39907742/slug",
        "https://stackoverflow.com/q/39907742",
        "https://superuser.com/questions/1/x",
        "https://unix.stackexchange.com/questions/1/x",
        "https://ru.stackoverflow.com/questions/1/x",
        "https://meta.stackexchange.com/questions/1",
    ])
    def test_recognised(self, url):
        assert site_reader(url) is not None


def _se(items) -> FakeResponse:
    return _text(json.dumps({"items": items, "quota_max": 300,
                             "quota_remaining": 299}),
                 "application/json; charset=utf-8")


class TestStackExchange:
    def test_question_reads_from_the_api(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, SO_URL, "so snippet")
        t.route("GET", SO_API_Q, _se([{
            "title": "Github API is responding with a 403 when using Request&#39;s",
            "body": "<p>I get a <code>403</code> from the API.</p>"}]))
        t.route("GET", SO_API_A, _se([
            {"is_accepted": True, "score": 74,
             "body": "<p>GitHub requires a <b>User-Agent</b> header.</p>"},
            {"is_accepted": False, "score": 7, "body": "<p>Use a token.</p>"}]))
        b = retrieve("github api 403 requests", search_candidates=1)
        src = b.sources[0]
        assert src.grounding == "page-backed"
        assert src.final_url == SO_URL
        evidence = " ".join(c.text for c in b.chunks)
        assert "Request's" in evidence
        assert "Accepted answer (score 74):" in evidence
        assert "requires a User-Agent header" in evidence
        assert SO_URL not in t.urls("GET")

    def test_api_failure_falls_back_to_the_blocked_page(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, SO_URL, "so snippet")
        t.route("GET", SO_API_Q, FakeResponse(status=400, body="{}"))
        t.route("GET", SO_URL, FakeResponse(status=403, body="blocked"))
        b = retrieve("github api 403 requests", search_candidates=1)
        assert t.urls("GET") == [SO_API_Q, SO_URL]
        assert b.sources[0].error == (
            "stackoverflow.com refused access, HTTP 403; the site may block "
            "automated readers")
        assert b.sources[0].grounding == "failed"
        assert any(c.kind == "snippet" and "so snippet" in c.text
                   for c in b.chunks)

    def test_empty_question_list_falls_back(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", SO_API_Q, _se([]))
        t.route("GET", SO_URL, html_response(html_page(
            "<main><p>" + "Question page text. " * 20 + "</p></main>")))
        final_url, text = netpolicy.fetch_text(SO_URL)
        assert final_url == SO_URL
        assert "Question page text." in text
        assert t.urls("GET") == [SO_API_Q, SO_URL]


class TestCoderPrivacyEcho:
    def test_fetch_url_names_the_address_it_read_from(self, monkeypatch,
                                                      capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_fetch_url
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", RAW_README, _text(README))
        result = tool_fetch_url(tmp_path, REPO, _privacy=True)
        err = capsys.readouterr().err
        assert result.ok and "offline local-LLM inference" in result.output
        assert f"[localm privacy] fetch_url: {REPO}" in err
        assert f"[localm privacy] fetch_url read: {RAW_README}" in err

    def test_fetch_url_same_address_echoes_once(self, monkeypatch, capsys,
                                                tmp_path):
        from localm.plugins.coder.tools.web import tool_fetch_url
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", "https://plain.example/", _text("plain text page"))
        tool_fetch_url(tmp_path, "https://plain.example/", _privacy=True)
        err = capsys.readouterr().err
        assert err.count("[localm privacy]") == 1

    def test_fetch_url_echoes_stack_exchange_endpoints(self, monkeypatch,
                                                       capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_fetch_url
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", SO_API_Q, _se([{"title": "Q", "body": "<p>Body.</p>"}]))
        t.route("GET", SO_API_A, _se([{"is_accepted": True, "score": 3,
                                       "body": "<p>Answer.</p>"}]))
        tool_fetch_url(tmp_path, SO_URL, _privacy=True)
        err = capsys.readouterr().err
        assert f"[localm privacy] fetch_url: {SO_URL}" in err
        assert f"[localm privacy] fetch_url endpoint: {SO_API_Q}" in err
        assert f"[localm privacy] fetch_url endpoint: {SO_API_A}" in err

    def test_fetch_url_echoes_failed_github_endpoints(self, monkeypatch,
                                                      capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_fetch_url
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", RAW_README, FakeResponse(status=404))
        t.route("GET", API_README, FakeResponse(status=403))
        t.route("GET", REPO, html_response(FILE_LIST_CHROME))
        tool_fetch_url(tmp_path, REPO, _privacy=True)
        err = capsys.readouterr().err
        assert f"[localm privacy] fetch_url endpoint: {RAW_README}" in err
        assert f"[localm privacy] fetch_url endpoint: {API_README}" in err
        assert "fetch_url read:" not in err

    def test_web_search_echoes_endpoints(self, monkeypatch, capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_web_search
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        _search(t, REPO)
        t.route("GET", RAW_README, _text(README))
        tool_web_search(tmp_path, "localm repository", max_results=1,
                        _privacy=True)
        err = capsys.readouterr().err
        assert f"[localm privacy] web_search endpoint: {RAW_README}" in err
        assert f"[localm privacy] web_search read: {REPO}" in err

    def test_no_echo_without_privacy(self, monkeypatch, capsys, tmp_path):
        from localm.plugins.coder.tools.web import tool_fetch_url
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        t.route("GET", RAW_README, _text(README))
        tool_fetch_url(tmp_path, REPO)
        assert "[localm privacy]" not in capsys.readouterr().err


class TestBinaryBlob:
    def test_binary_raw_file_falls_back_to_the_page(self, monkeypatch):
        allow_public(monkeypatch)
        t = Transport().install(monkeypatch)
        raw = ("https://raw.githubusercontent.com/Matlan1/localm/master/"
               ".github/images/logo.png")
        page = "https://github.com/Matlan1/localm/blob/master/.github/images/logo.png"
        t.route("GET", raw, FakeResponse(headers={"Content-Type": "image/png"},
                                         body=b"\x89PNG\r\n\x1a\n" + b"\x00" * 64))
        t.route("GET", page, html_response(html_page(
            "<main><p>" + "logo.png 12 KB. " * 20 + "</p></main>")))
        final_url, text = netpolicy.fetch_text(page)
        assert t.urls("GET") == [raw, page]
        assert final_url == page
        assert "PNG" not in text
