# SPDX-License-Identifier: AGPL-3.0-or-later
"""A refused request must record WHY, not just the status.

A debug log line of the form

    DEBUG   localm: POST /v1/chat/completions -> 400 (9 ms, loop_lag=0.00s)

is status and timing and nothing else. The HTTPException detail - the one field
that says which check refused the request - reaches only the client, so the
cause is unrecoverable from the log.

FIXTURE PREMISE: these tests assert that ``debug_enabled()`` is genuinely ON,
because that is the condition the bug lives in - the failure is not "no log
without --debug", it is "--debug is on and STILL says nothing". A fixture that
left debug off could not express the failing case and would pass no matter what
the handler did.

Both an EARLY refusal (before the engine is resolved) and a LATE one (past
get_engine and past the chat pipeline) are covered: a test that only exercised
an early refusal would miss the late shape, whose log line is preceded by the
memory plugin's inlet record.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.debuglog as debuglog
from localm.inference import http_server as hs
from localm.inference.http_server import create_app


def _mock_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.gpu_placement = None
    type(engine).loaded = property(lambda self: True)
    return engine


def _reset_globals():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None


@pytest.fixture
def _debug_on(monkeypatch):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    assert debuglog.debug_enabled(), "test premise: --debug must be ON"


@pytest.fixture
def client():
    """A server WITH a model, so a request can be served and a late check can
    be the thing that refuses it."""
    _reset_globals()
    return TestClient(create_app(_mock_engine()))


@pytest.fixture
def client_no_model(monkeypatch):
    """A server with nothing loaded and nothing resolvable - where an unnamed
    request is still refused up front, on the route's first line."""
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    _reset_globals()
    return TestClient(create_app(None))


def _refusal_lines(caplog):
    return [r.getMessage() for r in caplog.records if "refused" in r.getMessage()]


def _empty_model(client):
    return client.post("/v1/chat/completions",
                       json={"model": "",
                             "messages": [{"role": "user", "content": "hi"}]})


def _bad_grammar(client):
    return client.post("/v1/chat/completions",
                       json={"model": "test-model", "grammar": "(" * 5000,
                             "messages": [{"role": "user", "content": "hi"}]})


def test_early_refusal_records_the_reason(_debug_on, client_no_model, caplog):
    """Refused on the route's first line, before the engine is resolved."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = _empty_model(client_no_model)

    assert r.status_code == 400
    detail = r.json()["detail"]
    lines = _refusal_lines(caplog)
    assert lines, "a refused request logged no reason at all"
    # Assert the DETAIL itself, not merely that some line was emitted.
    assert any(detail in line for line in lines), (
        f"the 400's detail {detail!r} never reached the log; got {lines!r}")
    assert any("/v1/chat/completions" in line and "400" in line for line in lines)


def test_late_refusal_records_the_reason(_debug_on, client, caplog):
    """The shape the 0.1.4 report actually had: refused near the END of the
    route, past get_engine and past the pipeline inlet."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = _bad_grammar(client)

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Invalid grammar" in detail
    lines = _refusal_lines(caplog)
    assert any(detail[:60] in line for line in lines), (
        f"a late refusal logged no reason; got {lines!r}")


def test_the_response_itself_is_unchanged(_debug_on, client_no_model, caplog):
    """The handler is a LOGGING seam. It delegates to fastapi's own handler, so
    the body and status a client sees must be exactly what they were before."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = _empty_model(client_no_model)

    assert r.status_code == 400
    assert r.json() == {"detail": "Model parameter is required and cannot be empty"}


def test_no_reason_line_when_debug_is_off(client, caplog, monkeypatch):
    """The gate is real, not incidental. Operational text only, and only when the
    user asked for a debug log."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert not debuglog.debug_enabled(), "test premise: --debug must be OFF"
    caplog.set_level(logging.DEBUG, logger="localm")

    r = _bad_grammar(client)

    assert r.status_code == 400
    assert not _refusal_lines(caplog)
