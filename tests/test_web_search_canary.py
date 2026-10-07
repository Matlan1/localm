# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_web_search_canary.py searches each built-in service on its
own, so a layout break on one is reported while another still answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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

ROW = ("Matlan1/localm", "https://github.com/Matlan1/localm", "localm repo")
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_web_search_canary.py"


@pytest.fixture
def canary(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    spec = importlib.util.spec_from_file_location("check_web_search_canary",
                                                  _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ok(text: str) -> FakeResponse:
    return FakeResponse(headers={"Content-Type": "text/html"}, text=text)


def _lite(rows) -> str:
    cells = "".join(
        f"<tr><td><a class='result-link' href='{u}'>{t}</a></td></tr>"
        f"<tr><td class='result-snippet'>{s}</td></tr>" for t, u, s in rows)
    return f"<html><body><table>{cells}</table></body></html>"


def _brave(rows) -> str:
    items = "".join(
        f'<div data-type="web"><a href="{u}"><div class="search-snippet-title" '
        f'title="{t}">{t}</div></a><div class="generic-snippet">{s}</div></div>'
        for t, u, s in rows)
    return f"<html><body>{items}</body></html>"


def test_a_broken_service_is_reported_while_others_answer(canary, monkeypatch,
                                                          capsys):
    allow_public(monkeypatch)
    no_sleep(monkeypatch)
    t = Transport().install(monkeypatch)
    t.route("POST", DDG_ENDPOINT, _ok("<html><body>new layout</body></html>"))
    t.route("POST", LITE_ENDPOINT, _ok(_lite([ROW])))
    t.route("GET", BRAVE_SEARCH + "*", FakeResponse(status=429, text=""))
    services = canary.run_canary("q")
    assert [(r["provider"], r["ok"], r.get("bot_check")) for r in services] == [
        ("duckduckgo-html", False, False),
        ("duckduckgo-lite", True, None),
        ("brave", False, True)]
    assert "could not read results" in services[0]["error"]
    assert canary.main([]) == 0
    out = capsys.readouterr().out
    assert "WEB SEARCH CANARY: DEGRADED" in out
    assert "duckduckgo-html: DEGRADED" in out
    assert "brave: answered with a bot check" in out


def test_every_service_answering_is_ok(canary, monkeypatch, capsys):
    allow_public(monkeypatch)
    no_sleep(monkeypatch)
    t = Transport().install(monkeypatch)
    t.route("POST", DDG_ENDPOINT, _ok(ddg_html([ROW])))
    t.route("POST", LITE_ENDPOINT, _ok(_lite([ROW])))
    t.route("GET", BRAVE_SEARCH + "*", _ok(_brave([ROW])))
    assert canary.main([]) == 0
    out = capsys.readouterr().out
    assert "WEB SEARCH CANARY: OK" in out
    for name in ("duckduckgo-html", "duckduckgo-lite", "brave"):
        assert f"{name}: OK, 1 result(s)" in out


def test_configured_searxng_is_the_only_service_checked(canary, monkeypatch):
    allow_public(monkeypatch, net_search_url="https://searx.example")
    t = Transport().install(monkeypatch)
    t.route("GET", "https://searx.example/search?*", FakeResponse(
        json_body={"results": [{"title": "T", "url": ROW[1], "content": "c"}]}))
    assert canary.run_canary("q") == [
        {"provider": "searxng", "ok": True, "count": 1}]
    assert all("searx.example" in u for u in t.urls())
