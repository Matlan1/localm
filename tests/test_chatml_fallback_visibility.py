# SPDX-License-Identifier: AGPL-3.0-or-later
"""llama.py's chatml-fallback warning (_warn_chatml_fallback) must reach a normal
user, not only debuglog.logger, which is invisible without --debug. See
llama.py's _apply_model_template (returns the fallback reason instead of only
logging it) and gguf.py's chat_stream (reads it from the worker's "done" envelope
and console.print's it once per loaded model, mirroring the existing
_grammar_unsupported pattern).

These tests assert visibility WITHOUT --debug, which is the exact condition the
gap lives in. A test that turned debug on and checked the log record would pass
regardless and prove nothing."""

from __future__ import annotations

import pytest

import localm.debuglog as debuglog
from localm.inference.backends.gguf import GgufBackend


class _FakeRunner:
    """Stand-in for ModelRunner: yields a chunk then reports a "done" envelope
    exactly like the real multiprocessing runner, without spawning a child."""

    def __init__(self, done: dict):
        self.last_done = None
        self._done = done

    def chat_stream(self, **_kwargs):
        yield "hi"
        self.last_done = self._done


def _backend() -> GgufBackend:
    b = GgufBackend("does-not-exist.gguf", n_ctx=512)
    b._loaded = True
    return b


_FALLBACK_DONE = {
    "finish_reason": "stop",
    "grammar_unsupported": False,
    "chatml_fallback_reason": "model has no embedded chat template",
}

_NO_FALLBACK_DONE = {
    "finish_reason": "stop",
    "grammar_unsupported": False,
    "chatml_fallback_reason": None,
}


@pytest.fixture(autouse=True)
def _debug_is_off(monkeypatch):
    """The test's own premise: --debug / LOCALM_DEBUG must genuinely be off,
    or this test would not be exercising the bug's condition at all."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert not debuglog.debug_enabled(), "test premise: --debug must be OFF"


def test_chatml_fallback_reaches_the_console_without_debug(capsys):
    backend = _backend()
    backend._runner = _FakeRunner(dict(_FALLBACK_DONE))

    tokens = list(backend.chat_stream(
        [{"role": "user", "content": "describe this"}], max_tokens=8))

    assert tokens == ["hi"]
    out = capsys.readouterr().out
    assert "chat template is not recognized" in out, out
    assert "degraded" in out, out


def test_no_fallback_stays_silent(capsys):
    """A model with a working template must never print the degrade notice -
    the fix must not turn into "always warn"."""
    backend = _backend()
    backend._runner = _FakeRunner(dict(_NO_FALLBACK_DONE))

    list(backend.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8))

    out = capsys.readouterr().out
    assert "chat template is not recognized" not in out, out


def test_chatml_fallback_warns_once_per_loaded_model(capsys):
    """A loaded model's template capability never changes between requests -
    the notice must not spam the console on every request, mirroring
    _grammar_unsupported's own per-instance latch."""
    backend = _backend()

    backend._runner = _FakeRunner(dict(_FALLBACK_DONE))
    list(backend.chat_stream([{"role": "user", "content": "first"}], max_tokens=8))
    first_out = capsys.readouterr().out
    assert "chat template is not recognized" in first_out, first_out

    backend._runner = _FakeRunner(dict(_FALLBACK_DONE))
    list(backend.chat_stream([{"role": "user", "content": "second"}], max_tokens=8))
    second_out = capsys.readouterr().out

    assert "chat template is not recognized" not in second_out, second_out


def test_chatml_fallback_also_logged_for_bug_reports(capsys, caplog):
    """The debug-log detail must still exist (bug-report ring buffer / --debug
    file) alongside the console notice - this fix adds a channel, it does not
    remove the existing one."""
    import logging
    caplog.set_level(logging.WARNING, logger="localm")

    backend = _backend()
    backend._runner = _FakeRunner(dict(_FALLBACK_DONE))
    list(backend.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8))

    assert any("chat template" in r.message for r in caplog.records), caplog.records
