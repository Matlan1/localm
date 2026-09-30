# SPDX-License-Identifier: AGPL-3.0-or-later
"""The web, image, music and video request routes never log the user's query,
URL, prompt, tags or lyrics: not into the always-on activity ring that a bug
report carries, and not at any level, whether or not the debug-content gate is
open. Their INFO lines carry the request's length and numeric parameters only,
and a non-finite duration is refused before anything is logged."""

from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import debuglog


@pytest.fixture
def fresh_ring():
    """A clean activity ring on the localm logger for one test; the logger's
    prior handlers, level and ring are restored afterwards."""
    logger = debuglog.logger
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_ring = debuglog._ring_handler
    for h in list(logger.handlers):
        if isinstance(h, debuglog._RingBufferHandler):
            logger.removeHandler(h)
    debuglog._ring_handler = None
    assert debuglog.install_ring_buffer() is True
    yield logger
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    debuglog._ring_handler = saved_ring


@pytest.fixture(params=["privacy", "full"])
def session_mode(request, monkeypatch):
    """The debug log is on; the session mode is privacy (content gate closed)
    or full (content gate open)."""
    import localm.audit as audit
    monkeypatch.setenv("LOCALM_MODE", request.param)
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    monkeypatch.setattr(audit, "_active_coder_privacy_count", 0)
    assert debuglog.debug_content_enabled() is (request.param != "privacy")
    return request.param


class _Jobs:
    def __init__(self):
        self.started = []

    def start_fn(self, kind, fn, *, result_path=None, owner=None, label=None):
        self.started.append(kind)
        return MagicMock(id=f"job-{kind}")


def _app_with(tmp_path, *plugins):
    """A bare app with *plugins* installed through the real PluginManager, a
    job registry that records starts without running them, and a known own
    address."""
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    manager = PluginManager(app, external_root=tmp_path / "plugins")
    for name in plugins:
        manager.install(name)
    app.state.jobs = _Jobs()
    app.state.self_url = "http://127.0.0.1:9/v1"
    return app


def _records_carrying(caplog, needle):
    """The captured records whose format string or arguments contain
    *needle*, rendered without applying the format (which can itself raise)."""
    return [f"{r.levelname} {r.msg!r} % {r.args!r}" for r in caplog.records
            if needle in str(r.msg) or needle in repr(r.args)]


# --------------------------------------------------------------------------- #
#  web: retrieve / search / fetch                                              #
# --------------------------------------------------------------------------- #

_QUERY = "SENTINEL-web-query-7Q4M symptoms of something private"
_URL = "https://intranet.example/hr/case?id=SENTINEL-web-url-7Q4M"


class _Provider:
    name = "stub"

    def __init__(self):
        self.queries = []

    def search(self, query, n):
        from localm.web_retrieval.contracts import SearchResult
        self.queries.append(query)
        return [SearchResult(title="t", url="https://example.test/a",
                             snippet="snippet text", rank=1, provider="stub")]


@pytest.fixture
def web_stubs(monkeypatch):
    """netpolicy's search, fetch and page-read calls and the retrieval
    provider, stubbed to answer without a network; each records its calls."""
    from localm import netpolicy
    calls = {"search": [], "fetch": [], "page": []}

    def _web_search(query, max_results=5):
        calls["search"].append(query)
        return [{"title": "t", "snippet": "s", "url": "https://example.test/a"}]

    def _fetch_text(url):
        calls["fetch"].append(url)
        return "https://example.test/final", "fetched page text"

    def _safe_fetch(url, *, timeout=None, **kwargs):
        calls["page"].append(url)
        return "https://example.test/a", "text/plain", "page text for the query"

    provider = _Provider()
    retrieve_mod = importlib.import_module("localm.web_retrieval.retrieve")
    monkeypatch.setattr(netpolicy, "web_search", _web_search)
    monkeypatch.setattr(netpolicy, "fetch_text", _fetch_text)
    monkeypatch.setattr(netpolicy, "safe_fetch", _safe_fetch)
    monkeypatch.setattr(retrieve_mod, "provider_from_config", lambda *a, **k: provider)
    calls["provider"] = provider.queries
    return calls


def _drive_web(client):
    return [
        client.post("/api/web/retrieve", json={"query": _QUERY}),
        client.post("/api/web/search", json={"query": _QUERY, "max_results": 3}),
        client.post("/api/web/fetch", json={"url": _URL, "max_chars": 5000}),
    ]


def test_web_query_and_url_never_reach_the_activity_ring(
        tmp_path, fresh_ring, session_mode, web_stubs, caplog):
    client = TestClient(_app_with(tmp_path, "web"), raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG, logger="localm"):
        responses = _drive_web(client)

    ring = "\n".join(debuglog.recent_activity())
    for needle in ("SENTINEL-web-query-7Q4M", "SENTINEL-web-url-7Q4M"):
        assert needle not in ring, (
            f"{needle!r} reached the always-on activity ring:\n{ring}")
    for needle in ("SENTINEL-web-query-7Q4M", "SENTINEL-web-url-7Q4M"):
        leaked = _records_carrying(caplog, needle)
        assert not leaked, f"{needle!r} was logged in {session_mode} mode: {leaked}"

    # Each route still logs that it ran, with non-content parameters only.
    assert f"web retrieve: {len(_QUERY)}-char query" in ring
    assert f"web search: {len(_QUERY)}-char query (max_results=3)" in ring
    assert f"web fetch: {len(_URL)}-char url (max_chars=5000)" in ring
    assert "web retrieve: status=ok, sources=1" in ring
    assert "web search: returned 1 result(s)" in ring
    assert "web fetch: retrieved 17 chars (truncated=False)" in ring
    assert web_stubs["provider"] == [_QUERY]
    assert web_stubs["search"] == [_QUERY]
    assert web_stubs["fetch"] == [_URL]
    assert [r.status_code for r in responses] == [200, 200, 200], \
        [r.text[:200] for r in responses]


# --------------------------------------------------------------------------- #
#  image / music / video generation requests                                   #
# --------------------------------------------------------------------------- #

_MEDIA_CASES = [
    ("image", "/api/imagine",
     {"prompt": "SENTINEL-image-prompt-7Q4M a private scene",
      "negative_prompt": "SENTINEL-image-negative-7Q4M"},
     "imagine: {n}-char prompt", "prompt"),
    ("music", "/api/music",
     {"tags": "SENTINEL-music-tags-7Q4M private mood",
      "lyrics": "SENTINEL-music-lyrics-7Q4M", "duration_seconds": 7.5},
     "music: {n}-char tags (duration=7.5s)", "tags"),
    ("video", "/api/video",
     {"prompt": "SENTINEL-video-prompt-7Q4M a private scene",
      "negative_prompt": "SENTINEL-video-negative-7Q4M", "seconds": 5.5},
     "video: {n}-char prompt (seconds=5.5)", "prompt"),
]


@pytest.mark.parametrize("plugin,route,body,info_line,text_field", _MEDIA_CASES,
                         ids=[c[0] for c in _MEDIA_CASES])
def test_media_prompt_never_reaches_the_activity_ring(
        tmp_path, fresh_ring, session_mode, caplog,
        plugin, route, body, info_line, text_field):
    app = _app_with(tmp_path, plugin)
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG, logger="localm"):
        response = client.post(route, json=body)

    sentinels = [v for v in body.values() if isinstance(v, str)]
    ring = "\n".join(debuglog.recent_activity())
    for needle in sentinels:
        assert needle not in ring, (
            f"{plugin}: {needle!r} reached the always-on activity ring:\n{ring}")
    for needle in sentinels:
        leaked = _records_carrying(caplog, needle)
        assert not leaked, (
            f"{plugin}: {needle!r} was logged in {session_mode} mode: {leaked}")

    assert info_line.format(n=len(body[text_field])) in ring, ring
    assert app.state.jobs.started == [route.rsplit("/", 1)[-1]]
    assert response.status_code == 200, response.text[:300]


@pytest.mark.parametrize("plugin,route,body", [
    ("music", "/api/music",
     '{"tags": "SENTINEL-nan-tags-7Q4M", "duration_seconds": NaN}'),
    ("music", "/api/music",
     '{"tags": "SENTINEL-nan-tags-7Q4M", "duration_seconds": Infinity}'),
    ("video", "/api/video",
     '{"prompt": "SENTINEL-nan-prompt-7Q4M", "seconds": NaN}'),
    ("video", "/api/video",
     '{"prompt": "SENTINEL-nan-prompt-7Q4M", "seconds": -Infinity}'),
], ids=["music-nan", "music-inf", "video-nan", "video-neg-inf"])
def test_non_finite_duration_is_refused_without_logging_the_text(
        tmp_path, fresh_ring, capsys, caplog, plugin, route, body):
    app = _app_with(tmp_path, plugin)
    client = TestClient(app, raise_server_exceptions=False)
    capsys.readouterr()
    with caplog.at_level(logging.DEBUG, logger="localm"):
        response = client.post(route, content=body.encode(),
                               headers={"content-type": "application/json"})
    err = capsys.readouterr().err

    needle = "SENTINEL-nan-"
    assert needle not in err, f"the request text reached stderr:\n{err}"
    leaked = _records_carrying(caplog, needle)
    assert not leaked, f"the request text was logged: {leaked}"
    assert needle not in "\n".join(debuglog.recent_activity())
    assert "Logging error" not in err, err
    assert app.state.jobs.started == []
    assert response.status_code in (400, 422), response.text[:300]
