# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run` logs the worker exit reason when the model worker dies mid-reply,
so a bug report's log carries it (the backend lowers its own line for this
failure, and the standalone CLI is the one path that does not re-log it)."""

import logging
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.base import WorkerDiedError

REASON = "the model worker exited during the reply (worker exit code 3221225477)"


def _worker_records(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "worker exit code 3221225477" in r.getMessage()]


def _engine_raising(exc):
    engine = MagicMock(name="EngineInstance")
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=False)
    engine.chat_stream = MagicMock(side_effect=exc)
    return engine


def test_single_prompt_path_logs_a_warning_with_the_exit_reason(
        cli_runner, tmp_path, monkeypatch, caplog):
    model_f = tmp_path / "some-model.gguf"
    model_f.write_bytes(b"GGUF")
    engine_cls = MagicMock(name="Engine",
                           return_value=_engine_raising(WorkerDiedError(REASON + "\ndetail")))
    monkeypatch.setattr("localm.inference.engine.Engine", engine_cls)
    monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)

    from localm.cli.chat import run
    with caplog.at_level(logging.DEBUG, logger="localm"):
        result = cli_runner.invoke(run, [str(model_f), "--no-server", "-p", "hi"])

    assert result.exit_code == 1, result.output
    assert REASON in result.output
    assert len(_worker_records(caplog)) == 1, [r.getMessage() for r in caplog.records]
    assert "detail" not in _worker_records(caplog)[0].getMessage()


def test_interactive_path_logs_a_warning_with_the_exit_reason(monkeypatch, caplog):
    from localm.cli import chat as chat_mod
    engine = _engine_raising(WorkerDiedError(REASON))
    engine.display_name = "m"
    engine.context_capacity.return_value = 4096
    engine.count_tokens.return_value = 3
    answers = iter(["hello"])

    def _input(*a, **kw):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError()
    monkeypatch.setattr(chat_mod.console, "input", _input)

    with caplog.at_level(logging.DEBUG, logger="localm"):
        chat_mod._interactive(engine, None, {})

    assert len(_worker_records(caplog)) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("exc", [RuntimeError("attached server error"),
                                 ValueError("bad")])
def test_other_failures_are_not_logged_as_worker_deaths(exc, caplog):
    from localm.cli.chat import _log_worker_death
    with caplog.at_level(logging.DEBUG, logger="localm"):
        _log_worker_death(exc)
    assert caplog.records == []
