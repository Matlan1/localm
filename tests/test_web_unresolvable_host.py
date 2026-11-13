# SPDX-License-Identifier: AGPL-3.0-or-later
"""A host name that does not resolve is a failure to connect, not a refusal
by the network policy: the web routes answer 502 saying the name did not
resolve, and a retrieval records a failed search instead of refusing. A search
that failed on every service is not prefixed with a second "Search failed"."""

import pytest
from fastapi.testclient import TestClient

from localm import netpolicy
from localm.web_retrieval.contracts import SearchProviderError
from localm.web_retrieval.errors import describe_failure
from tests.test_gui import web_app  # noqa: F401


def _unresolvable(*a, **kw):
    raise netpolicy.UnresolvableHost(
        "Could not resolve 'html.duckduckgo.com' to an address; refusing the "
        "request rather than connecting through an unvalidated re-resolution.")


def test_unresolvable_is_not_a_policy_refusal():
    from localm.web_retrieval.providers import is_policy_refusal
    assert is_policy_refusal(netpolicy.NetworkPolicyError("off")) is True
    assert is_policy_refusal(netpolicy.UnresolvableHost("x")) is False


def test_describe_failure_says_the_name_did_not_resolve():
    text = describe_failure(netpolicy.UnresolvableHost("x"), "https://nope.example/a")
    assert text.startswith("could not find nope.example: its name did not resolve")
    assert "refusing" not in text


@pytest.mark.parametrize("path, body, patch, prefix", [
    ("/api/web/search", {"query": "linz"}, "localm.netpolicy.web_search", "Search failed: "),
    ("/api/web/fetch", {"url": "https://nope.example/"}, "localm.netpolicy.fetch_text",
     "Fetch failed: "),
])
def test_web_routes_answer_502_for_a_name_that_does_not_resolve(
        web_app, monkeypatch, path, body, patch, prefix):  # noqa: F811
    monkeypatch.setattr(patch, _unresolvable)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with TestClient(web_app) as client:
        r = client.post(path, json=body)
    assert r.status_code == 502, r.text
    assert r.json()["detail"].startswith(prefix)
    assert "did not resolve" in r.json()["detail"]
    assert "refusing the request" not in r.json()["detail"]


def test_a_policy_refusal_is_still_403(web_app, monkeypatch):  # noqa: F811
    def off(*a, **kw):
        raise netpolicy.NetworkPolicyError("Network access is off.", off=True)
    monkeypatch.setattr("localm.netpolicy.web_search", off)
    with TestClient(web_app) as client:
        r = client.post("/api/web/search", json={"query": "linz"})
    assert r.status_code == 403


def test_a_search_that_failed_everywhere_is_not_prefixed_twice(web_app, monkeypatch):  # noqa: F811
    def failed(*a, **kw):
        raise SearchProviderError("Web search failed on every search service localm "
                                  "tried (DuckDuckGo: timed out).")
    monkeypatch.setattr("localm.netpolicy.web_search", failed)
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with TestClient(web_app) as client:
        r = client.post("/api/web/search", json={"query": "linz"})
    assert r.status_code == 502
    assert r.json()["detail"].startswith("Web search failed on every search service")


def test_retrieve_records_an_unresolvable_search_as_failed():
    import importlib
    R = importlib.import_module("localm.web_retrieval.retrieve")

    class Provider:
        name = "duckduckgo"
        endpoint = "https://html.duckduckgo.com/html/"

        def search(self, query, n):
            _unresolvable()

    bundle = R.retrieve("linz", provider=Provider())
    assert bundle.search_status == R.SEARCH_FAILED
    assert "did not resolve" in bundle.search_error

