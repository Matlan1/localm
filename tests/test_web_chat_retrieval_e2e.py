# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end: a fake model drives a real web-tool round trip through the real
server-side HTTP surfaces (ADR-0022 Phase 5, item 26).

The interactive chat surface's tool-call orchestration (parse the model's
reply, call the retrieval controller, re-inject the evidence, ask again) is
implemented client-side in JavaScript - see
``localm/plugins/gui/static/app/settings-perf.js`` ``runCompletion``/
``requestWebTool``, and ``localm/inference/chat_pipeline.py``'s own docstring,
which says client-side context injection (RAG, memory, web) is independent of
the server pipeline. No Python module reproduces that loop for the interactive
surface, so this test plays the browser's part in Python: it drives the two
real HTTP endpoints the browser actually calls (``POST /v1/chat/completions``
and ``POST /api/web/retrieve``), on ONE app, through ``TestClient``, with only
the model and the network edge (search provider + page fetch) replaced.

Everything between those two boundaries is real production code: ``create_app``
/ ``ChatPipeline`` (SSE-free non-streaming path here) for the chat completion,
and ``localm.plugins.builtin.web.plug.web_retrieve_endpoint`` ->
``localm.web_retrieval.retrieve`` -> canonicalization, concurrent page fetch,
extraction, query-aware evidence selection, ``EvidenceBundle`` construction and
``neutralise()`` for the retrieval. The tool-call TEXT is parsed with
``localm.plugins.builtin.jobs.webtool.parse_web_calls``: that is real
production code for the scheduled-jobs surface, proven byte-for-byte equivalent
in substance to the JS parser by the existing cross-surface drift tests in
tests/test_jobs_web_search.py (``TestEvidenceGroundingRules``,
``TestGrammarMirroredInGuiSurface``), and it is the closest real parser
available to reuse rather than re-implementing parsing for this test.

Complements test_jobs_web_search.py's TestRunChatWithWeb (the same round trip
through the scheduled-jobs' own in-process Python loop, no HTTP at all) by
covering the surface that loop does NOT: the actual chat HTTP endpoints a
real client calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app
from localm.plugins.builtin.jobs import webtool
from localm.plugins.engine import PluginManager
from tests._web_retrieval_fixtures import html_page, stub_retrieval

_TOOL_CALL = ('<tool_call>{"name": "web_search", '
             '"args": {"query": "weather in Paris"}}</tool_call>')
_FINAL_ANSWER = "It is sunny in Paris today. Source: [S1]."
_PARIS_URL = "https://example.com/paris"
_PARIS_ROW = ("Paris weather", _PARIS_URL, "Sunny, 24C (cached)")
_PARIS_PAGE = html_page(
    "<main><p>Paris weather today: sunny with a high of 24C and a light "
    "breeze from the west.</p></main>", title="Paris weather")


def _scripted_engine(replies):
    """A fake chat engine with the attribute surface the real
    ``/v1/chat/completions`` path (ChatPipeline) needs - same recipe as
    tests/test_chat_terminal_outcome.py's ``_engine()`` - except
    ``chat_stream`` yields the NEXT reply in *replies* on each call, so a
    multi-round tool-call conversation can be scripted the way
    tests/test_jobs_web_search.py's ``ScriptedEngine`` scripts the jobs
    surface."""
    engine = MagicMock()
    remaining = list(replies)

    def _chat_stream(messages, **kwargs):
        reply = remaining.pop(0) if remaining else ""
        for ch in reply:
            yield ch

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 8
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: True)
    return engine


@pytest.fixture
def web_chat_app(tmp_path):
    """``create_app(engine)`` (chat completions) plus the real web plugin
    mounted on the SAME app (retrieval). LOCALM_HOME/API-key isolation is
    already handled by conftest.py's autouse ``_isolate_localm_home``/
    ``_isolate_owner_key_env`` - duplicating it here with a second, differently
    -rooted tmp_path is exactly the HOME_DIR/LOCALM_HOME-env mismatch that
    fixture's own docstring warns about (confirmed by hand: it breaks the
    owner-key lookup and turns every plugin route into a 401/403)."""
    def _build(engine):
        app: FastAPI = create_app(engine)
        PluginManager(app, external_root=tmp_path / "noplugins").install("web")
        return app
    return _build


def test_fake_model_drives_a_real_web_tool_round_trip_to_a_cited_final_answer(
        web_chat_app, monkeypatch):
    """search request -> page acquisition -> evidence bundle -> grounded
    synthesis -> valid source IDs -> final answer, all through the real
    /v1/chat/completions and /api/web/retrieve HTTP surfaces on one app."""
    queries = stub_retrieval(monkeypatch, [_PARIS_ROW], {_PARIS_URL: _PARIS_PAGE})
    engine = _scripted_engine([_TOOL_CALL, _FINAL_ANSWER])
    app = web_chat_app(engine)
    # Every plugin POST goes through the real open-mode management gate
    # (http_server.py's _origin_guard): with no API key configured it demands
    # the per-process shell token create_app() mints into app.state.shell_token
    # for the loopback GUI shell - the credential this test's "browser" would
    # actually hold. /v1/chat/completions is separately exempt (_CROSS_ORIGIN_OK),
    # which is why round 1 below needs no header but round 2 does.
    shell_auth = {"Authorization": f"Bearer {app.state.shell_token}"}

    with TestClient(app) as client:
        # Round 1: the model is asked, and (per the script) emits a tool call.
        # The system prompt is the real jobs-surface constant - proven a
        # byte-for-byte content mirror of the browser's own WEB_TOOL_PROMPT by
        # the existing drift tests - standing in for what settings-perf.js
        # actually sends on the interactive surface.
        messages = [
            {"role": "system", "content": webtool.WEB_TOOL_SYSTEM},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]
        r1 = client.post("/v1/chat/completions", json={
            "model": "test-model", "stream": False, "messages": messages})
        assert r1.status_code == 200
        reply1 = r1.json()["choices"][0]["message"]["content"]
        assert reply1 == _TOOL_CALL

        # Recognizing the web_search call: real production parsing (the
        # server-side port of the browser's parseWebCall).
        calls = webtool.parse_web_calls(reply1)
        assert calls == [{"name": "web_search",
                          "args": {"query": "weather in Paris"}}]

        # Dispatch to the REAL retrieval controller through the REAL HTTP
        # endpoint - only the search provider and the page fetch are doubles;
        # canonicalization, concurrent read, extraction, chunk selection,
        # EvidenceBundle construction and neutralise() all run for real.
        r2 = client.post("/api/web/retrieve",
                         json={"query": calls[0]["args"]["query"]},
                         headers=shell_auth)
        assert r2.status_code == 200
        bundle = r2.json()
        assert queries == ["weather in Paris"]
        assert bundle["search_status"] == "ok"
        assert bundle["grounding"] == "page-backed"
        assert bundle["sources"][0]["id"] == "S1"
        assert bundle["sources"][0]["grounding"] == "page-backed"
        assert any("light breeze from the west" in c["text"]
                  for c in bundle["chunks"]), \
            "the real page text, not just the snippet, must reach the bundle"

        # Re-injection: exactly the text the browser injects
        # (toolEventPrompt/runWebCall build the same header + prompt_text
        # shape from this same JSON).
        evidence_message = (
            f'[Results of web_search "{calls[0]["args"]["query"]}"] '
            f'({bundle["grounding_summary"]})\n{bundle["prompt_text"]}')
        messages += [
            {"role": "assistant", "content": reply1},
            {"role": "user", "content": evidence_message},
        ]

        # Round 2: the model sees the evidence and (per the script)
        # synthesizes a final answer citing a valid source id.
        r3 = client.post("/v1/chat/completions", json={
            "model": "test-model", "stream": False, "messages": messages})
        assert r3.status_code == 200
        final = r3.json()["choices"][0]["message"]["content"]

    assert final == _FINAL_ANSWER
    assert "S1" in final
    assert calls[0]["args"]["query"] not in "".join(
        s["id"] for s in bundle["sources"]), "sanity: S-ids are not the query"
    # Both chat turns happened, each seeing the growing message history.
    assert engine.chat_stream.call_count == 2
    first_call_messages, second_call_messages = (
        c.args[0] for c in engine.chat_stream.call_args_list)
    assert len(first_call_messages) == 2
    assert len(second_call_messages) == 4
    assert second_call_messages[-1]["content"] == evidence_message
