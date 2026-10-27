# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app-wide 422 for a body labelled JSON that is not UTF-8 applies only to
FastAPI's own body parse. A route that raises its own 400 chained from a
``UnicodeDecodeError`` keeps its status and text, and the 422 branch still
writes the debug refusal line."""

import logging
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from localm.inference.errors import (FASTAPI_BODY_PARSE_DETAIL,
                                     UNDECODABLE_BODY_MESSAGE,
                                     is_undecodable_body)

BAD_BYTES = b"\xff\xfe\x00"
JSON_HEADERS = {"Content-Type": "application/json"}


def _decode_error() -> UnicodeDecodeError:
    try:
        BAD_BYTES.decode("utf-8")
    except UnicodeDecodeError as e:
        return e
    raise AssertionError("test premise: the bytes must not decode")


def _chained(detail: str) -> HTTPException:
    try:
        raise _decode_error()
    except UnicodeDecodeError as cause:
        exc = HTTPException(400, detail)
        exc.__cause__ = cause
        return exc


class TestPredicate:
    def test_fastapis_own_body_parse_error_matches(self):
        assert is_undecodable_body(_chained(FASTAPI_BODY_PARSE_DETAIL))

    def test_a_routes_own_400_with_the_same_cause_does_not_match(self):
        assert not is_undecodable_body(
            _chained("The request body is not valid JSON."))

    def test_fastapis_text_without_the_decode_cause_does_not_match(self):
        assert not is_undecodable_body(HTTPException(400, FASTAPI_BODY_PARSE_DETAIL))


def _speech_app() -> FastAPI:
    from localm.inference.app_assembly.errors import register_exception_handlers
    from localm.plugins.builtin.tts import speech_route
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(speech_route.router)
    return app


def test_tts_route_keeps_its_own_400_for_a_non_utf8_body():
    r = TestClient(_speech_app(), raise_server_exceptions=False).post(
        "/v1/audio/speech", content=BAD_BYTES, headers=JSON_HEADERS)
    assert r.status_code == 400, r.text[:300]
    assert r.json()["detail"] == "The request body is not valid JSON."


def _api_client() -> TestClient:
    from localm.inference.http_server import create_app
    os.environ.pop("LOCALM_API_KEY", None)
    engine = MagicMock()
    engine.display_name = "m"
    engine.active_requests = 0
    type(engine).loaded = property(lambda self: True)
    return TestClient(create_app(engine), raise_server_exceptions=False)


def test_fastapis_own_parse_of_a_non_utf8_json_body_is_still_422():
    r = _api_client().post("/v1/chat/completions", content=BAD_BYTES,
                           headers=JSON_HEADERS)
    assert r.status_code == 422, r.text[:300]
    assert r.json()["detail"] == UNDECODABLE_BODY_MESSAGE


def test_the_422_branch_writes_the_debug_refusal_line(monkeypatch, caplog):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    with caplog.at_level(logging.DEBUG, logger="localm"):
        r = _api_client().post("/v1/chat/completions", content=BAD_BYTES,
                               headers=JSON_HEADERS)
    assert r.status_code == 422
    lines = [rec.getMessage() for rec in caplog.records
             if " refused 422: " in rec.getMessage()]
    assert lines, [rec.getMessage() for rec in caplog.records]
    assert "/v1/chat/completions" in lines[0]
