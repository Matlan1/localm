# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-CONTEXT-RESIDUAL: a diagnostic capture of the exact messages handed to
the backend, so a future recurrence of "the model says you wrote nothing" (a
stale/duplicated/missing history turn) is diagnosable from the debug log
instead of unreproducible.

http_server.py's `_log_assembled_prompt` writes this at the three points a
messages list actually reaches `engine.chat_stream` (`_stream_sse`,
`_stream_sse_completion`, `_generate_full`), content-gated on
`debug_content_enabled()` - the same gate every other content-logging site
uses (llama.py raw model output, memory/store.py, jobs/webtool.py). This is
chat history, so it must never reach the debug log in privacy mode, and it
must never reach a bug report even when the debug log has it (localm/
_log_digest.py's `_CONTENT_MARKER_RES` must recognise the new marker too).
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from localm import _log_digest as ld
from localm.inference.http_server import _debug_prompt_dump, create_app

SECRET_TURN = "my bank PIN hint is my mother's maiden name Vandermeulen"


def _yielding_engine(tokens=("hi", " there")):
    engine = MagicMock()
    _state = {"loaded": True}

    def _chat_stream(messages, **kwargs):
        for t in tokens:
            yield t

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 3
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "stop"
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


def _messages():
    return [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": SECRET_TURN},
    ]


def _captured_blob(caplog) -> str:
    """Fails loudly on an empty capture, so a not-in-log assertion cannot pass
    vacuously against a harness that captured nothing at all."""
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert blob.strip(), "NOTHING was captured from the 'localm' logger"
    return blob


def _debug_log(monkeypatch, caplog):
    monkeypatch.setattr("localm.debuglog.debug_enabled", lambda: True)
    caplog.set_level(logging.DEBUG, logger="localm")
    return caplog


class TestPrivacyModeDoesNotLeakTheHistory:
    def _not_allowed(self, monkeypatch):
        monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: False)

    def test_chat_completions_streaming(self, monkeypatch, caplog):
        self._not_allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": _messages(),
                "stream": True,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN not in blob, f"leaked chat history: {blob}"
        assert "assembled chat prompt" not in blob

    def test_chat_completions_nonstreaming(self, monkeypatch, caplog):
        self._not_allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": _messages(),
                "stream": False,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN not in blob, f"leaked chat history: {blob}"
        assert "assembled chat prompt" not in blob

    def test_raw_completions_streaming(self, monkeypatch, caplog):
        self._not_allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/completions", json={
                "model": "test-model",
                "prompt": SECRET_TURN,
                "stream": True,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN not in blob, f"leaked chat history: {blob}"
        assert "assembled chat prompt" not in blob


class TestNonPrivacyModeCapturesTheHistory:
    """NEGATIVE CASE: the gate must be a gate, not a blanket removal of the
    diagnostic - a non-privacy debug session must still get the capture."""

    def _allowed(self, monkeypatch):
        monkeypatch.setattr("localm.debuglog.debug_content_enabled", lambda: True)

    def test_chat_completions_streaming(self, monkeypatch, caplog):
        self._allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": _messages(),
                "stream": True,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN in blob, f"capture missing from: {blob}"
        assert "[1] user:" in blob

    def test_chat_completions_nonstreaming(self, monkeypatch, caplog):
        self._allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": _messages(),
                "stream": False,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN in blob, f"capture missing from: {blob}"

    def test_raw_completions_streaming(self, monkeypatch, caplog):
        self._allowed(monkeypatch)
        blob_src = _debug_log(monkeypatch, caplog)
        engine = _yielding_engine()
        with TestClient(create_app(engine)) as client:
            client.post("/v1/completions", json={
                "model": "test-model",
                "prompt": SECRET_TURN,
                "stream": True,
            })
        blob = _captured_blob(blob_src)
        assert SECRET_TURN in blob, f"capture missing from: {blob}"

def test_debug_prompt_dump_names_non_text_parts_without_embedding_them():
    # A multimodal content block can carry megabytes of base64 - the dump
    # must name the part, never inline its data. Unit-level: the HTTP layer's
    # own multimodal-support gate is a separate concern from this formatter.
    huge_data_url = "data:image/png;base64," + "A" * 5000
    messages = [{"role": "user", "content": [
        {"type": "text", "text": SECRET_TURN},
        {"type": "image_url", "image_url": {"url": huge_data_url}},
    ]}]
    dump = _debug_prompt_dump(messages)
    assert SECRET_TURN in dump
    assert huge_data_url not in dump
    assert "<image_url>" in dump


class TestLogDigestRecognizesTheNewMarker:
    """The other half of the privacy contract: even when the debug log DOES
    hold this capture (non-privacy debug session), a bug report must never
    surface it - localm/_log_digest.py's content scrubber has to know this
    site's marker too, or the digest builder silently stops protecting it."""

    def test_is_content_record_direct(self):
        assert ld.is_content_record({
            "level": "DEBUG", "logger": "localm",
            "lines": ["2026-08-26 10:00:00,000 DEBUG   localm: "
                      "assembled chat prompt (2 message(s)):",
                      f"[1] user: {SECRET_TURN}"],
        })

    def test_bare_operational_line_is_not_a_content_record(self):
        assert not ld.is_content_record({
            "level": "DEBUG", "logger": "localm",
            "lines": ["2026-08-26 10:00:00,000 DEBUG   localm: "
                      "GET /api/stats -> 200"],
        })

    def test_digest_drops_the_captured_history(self):
        text = (
            "2026-08-26 10:00:00,000 DEBUG   localm: assembled chat prompt "
            "(2 message(s)):\n"
            f"[1] user: {SECRET_TURN}\n"
            "2026-08-26 10:00:01,000 INFO    localm: request served\n"
        )
        digest = ld.build_digest(text)
        assert SECRET_TURN not in digest
        assert "Vandermeulen" not in digest
        assert "debug record(s) withheld" in digest
