# SPDX-License-Identifier: AGPL-3.0-or-later
"""A worker that dies mid-reply under standalone `localm run`: the console gets
one line (not a traceback), and the audit log and transcript record the failed
turn the way the server path does."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

import localm.audit as audit_mod
from localm.cli import chat as chat_mod
from localm.inference.backends.gguf import GgufBackend

_DEATH = ("The model process exited unexpectedly (worker exit -1 (the process was "
          "terminated)). The model has been unloaded and will reload on the next "
          "request.\nnative detail line two")


class _DyingRunner:
    def chat_stream(self, **_kwargs):
        yield "partial"
        raise RuntimeError(_DEATH)

    def shutdown(self, *a, **k):
        pass


def _backend() -> GgufBackend:
    b = GgufBackend("does-not-exist.gguf", n_ctx=512)
    b._loaded = True
    b._runner = _DyingRunner()
    return b


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def localm_log():
    lg = logging.getLogger("localm")
    h = _Collect()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    yield h
    lg.removeHandler(h)
    lg.setLevel(old)


def test_the_error_record_carries_one_line_and_no_traceback(localm_log):
    b = _backend()
    with pytest.raises(RuntimeError):
        list(b.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8))
    errors = [r for r in localm_log.records if r.levelno >= logging.WARNING]
    assert len(errors) == 1
    assert errors[0].exc_info is None
    msg = errors[0].getMessage()
    assert "worker failure during generation" in msg
    assert "\n" not in msg
    assert "worker exit -1" in msg


def test_a_death_the_runner_already_logged_adds_no_second_error_line(localm_log):
    from localm.inference.backends.base import WorkerDiedError

    class _LoggedDeath(_DyingRunner):
        def chat_stream(self, **_kwargs):
            yield "partial"
            raise WorkerDiedError(_DEATH)

    b = _backend()
    b._runner = _LoggedDeath()
    with pytest.raises(WorkerDiedError):
        list(b.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8))
    assert [r for r in localm_log.records if r.levelno >= logging.WARNING] == []
    assert any("dropping model instance" in r.getMessage() for r in localm_log.records)
    assert b._runner is None


def test_the_traceback_is_still_in_the_debug_log(localm_log):
    b = _backend()
    with pytest.raises(RuntimeError):
        list(b.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8))
    traced = [r for r in localm_log.records
              if r.levelno == logging.DEBUG and r.exc_info]
    assert len(traced) == 1
    assert traced[0].exc_info[0] is RuntimeError


def _drive_one_failed_turn(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", tmp_path / "sessions")
    audit = audit_mod.make_audit_log(mode, label="chat")
    transcript = audit_mod.make_transcript(mode, label="chat")
    engine = MagicMock()
    engine.display_name = "m"
    engine.count_tokens.return_value = 1
    engine.context_capacity.return_value = 4096

    def _stream(messages, **kw):
        yield "partial"
        raise RuntimeError(_DEATH)

    engine.chat_stream.side_effect = _stream
    inputs = iter(["what is 2+2"])

    def _fake_input(*a, **kw):
        try:
            return next(inputs)
        except StopIteration as e:
            raise EOFError() from e

    monkeypatch.setattr(chat_mod.console, "input", _fake_input)
    chat_mod._interactive(engine, None, {}, audit=audit, transcript=transcript)
    audit.close()
    return audit, transcript


def test_a_failed_turn_is_recorded_in_the_audit_log(tmp_path, monkeypatch):
    audit, _t = _drive_one_failed_turn(tmp_path, monkeypatch, audit_mod.SessionMode.LOG)
    rows = [json.loads(line) for line in audit.path.read_text(encoding="utf-8").splitlines()]
    kinds = [(r["type"], r["data"]) for r in rows]
    assert ("user", {"content": "what is 2+2"}) in kinds
    llm = [d for t, d in kinds if t == "llm"]
    assert len(llm) == 1 and "[inference error:" in llm[0]["content"]
    assert llm[0]["content"].startswith("partial")
    assert ("notice", {"kind": "finish_reason", "message": "error"}) in kinds


def test_a_failed_turn_is_recorded_in_the_transcript(tmp_path, monkeypatch):
    _a, transcript = _drive_one_failed_turn(tmp_path, monkeypatch, audit_mod.SessionMode.FULL)
    text = transcript.path.read_text(encoding="utf-8")
    assert "what is 2+2" in text
    assert "[inference error:" in text
    assert "[finish_reason: error]" in text


def test_privacy_mode_records_nothing_for_a_failed_turn(tmp_path, monkeypatch):
    audit, transcript = _drive_one_failed_turn(
        tmp_path, monkeypatch, audit_mod.SessionMode.PRIVACY)
    assert audit.path is None and transcript is None
    assert not (tmp_path / "sessions").exists()
