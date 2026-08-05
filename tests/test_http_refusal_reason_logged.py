# SPDX-License-Identifier: AGPL-3.0-or-later
"""A refused request must record WHY, not just the status.

The 0.1.4 release candidate produced this, at DEBUG, as the complete record of
a user-visible chat failure:

    DEBUG   localm: POST /v1/chat/completions -> 400 (9 ms, loop_lag=0.00s)

Status and timing and nothing else. The HTTPException detail - the one field
that says which check refused the request - reached only the client, so the
cause was unrecoverable from the log and two separate diagnoses of that single
line reached opposite wrong answers. AGENTS.md rule 5: a user-facing failure
whose cause cannot be learned from a debug log is a hidden problem.

FIXTURE PREMISE (diff-review-discipline.md item 19): these tests assert that
``debug_enabled()`` is genuinely ON, because that is the condition the bug lived
in - the complaint was never "no log without --debug", it was "--debug was on
and STILL said nothing". A fixture that left debug off could not express the
failing case and would pass no matter what the handler did.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.debuglog as debuglog
from localm.inference.http_server import create_app


def _mock_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.gpu_placement = None
    type(engine).loaded = property(lambda self: True)
    return engine


@pytest.fixture
def _debug_on(monkeypatch):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    assert debuglog.debug_enabled(), "test premise: --debug must be ON"


@pytest.fixture
def client():
    return TestClient(create_app(_mock_engine()))


def _refusal_lines(caplog):
    return [r.getMessage() for r in caplog.records if "refused" in r.getMessage()]


def test_chat_400_records_the_reason_not_only_the_status(_debug_on, client, caplog):
    """The exact shape from the 0.1.4 log: a 400 on the chat path."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = client.post("/v1/chat/completions",
                    json={"model": "", "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 400
    detail = r.json()["detail"]
    # The reason the CLIENT was given must also be in the log. Asserting the
    # detail itself (not merely that some line was emitted) is what makes this
    # fail if the handler logs a placeholder instead of the real cause.
    lines = _refusal_lines(caplog)
    assert lines, "a refused request logged no reason at all"
    assert any(detail in line for line in lines), (
        f"the 400's detail {detail!r} never reached the log; got {lines!r}")
    assert any("/v1/chat/completions" in line and "400" in line for line in lines)


def test_the_response_itself_is_unchanged(_debug_on, client, caplog):
    """The handler is a LOGGING seam. It delegates to fastapi's own handler, so
    the body and status a client sees must be exactly what they were before."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = client.post("/v1/chat/completions",
                    json={"model": "", "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 400
    assert r.json() == {"detail": "Model parameter is required and cannot be empty"}


def test_reason_is_logged_for_refusals_raised_deeper_than_the_first_check(
        _debug_on, client, caplog):
    """Generality: the empty-model 400 is raised on the route's FIRST line,
    before the engine is resolved and before the chat pipeline runs. A handler
    that only ever saw that one would look correct while missing every refusal
    that matters.

    This drives the grammar check instead - raised near the END of the route,
    past get_engine and past the pipeline inlet. That is the region the real
    0.1.4 refusal came from: its log line was preceded by the memory plugin's
    inlet record, which proves the request had already got that far."""
    caplog.set_level(logging.DEBUG, logger="localm")

    r = client.post("/v1/chat/completions",
                    json={"model": "test-model", "grammar": "(" * 5000,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Invalid grammar" in detail
    lines = _refusal_lines(caplog)
    assert any(detail[:60] in line for line in lines), (
        f"a deeper refusal ({r.status_code}) logged no reason; got {lines!r}")


def test_no_reason_line_when_debug_is_off(client, caplog, monkeypatch):
    """The gate is real, not incidental. Operational text only, and only when the
    user asked for a debug log."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert not debuglog.debug_enabled(), "test premise: --debug must be OFF"
    caplog.set_level(logging.DEBUG, logger="localm")

    r = client.post("/v1/chat/completions",
                    json={"model": "", "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 400
    assert not _refusal_lines(caplog)
