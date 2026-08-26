# SPDX-License-Identifier: AGPL-3.0-or-later
"""GgufBackend.count_messages_tokens must surface a permanently-failing worker
RPC where a user can see it. A `_dbg.warning(...)` (debuglog.logger) alone is
invisible without --debug, so a console.print runs alongside the debug-log line,
guarded by the same once-per-process latch.

These tests assert visibility WITHOUT --debug."""

from __future__ import annotations

import pytest

import localm.debuglog as debuglog
import localm.inference.backends.gguf as gguf_mod
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._runner import RunnerBusy


class _FakeRunnerRaises:
    """count_messages_tokens fails, but plain count_tokens still succeeds -
    the realistic shape per gguf.py's own comment: "the super() return...
    calls self.count_tokens(text)... a real, untemplated tokenizer count
    when the worker can still answer plain count_tokens"."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def count_messages_tokens(self, messages):
        raise self._exc

    def count_tokens(self, text):
        return max(1, len(text) // 4)


class _FakeRunnerBusy:
    def count_messages_tokens(self, messages):
        raise RunnerBusy("worker busy with a live stream")

    def count_tokens(self, text):
        raise RunnerBusy("worker busy with a live stream")


def _backend() -> GgufBackend:
    b = GgufBackend("does-not-exist.gguf", n_ctx=512)
    b._loaded = True
    return b


@pytest.fixture(autouse=True)
def _debug_is_off_and_latch_reset(monkeypatch):
    """The test's own premise: --debug must genuinely be off. Also reset the
    MODULE-level once-per-process latch (not per-instance, unlike
    _grammar_unsupported/_chatml_fallback - see the comment above
    _count_messages_tokens_rpc_warned in gguf.py) so test order/repetition
    cannot mask the warning behind an earlier test's trip of the same latch."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    assert not debuglog.debug_enabled(), "test premise: --debug must be OFF"
    monkeypatch.setattr(gguf_mod, "_count_messages_tokens_rpc_warned", False)


def test_rpc_failure_reaches_the_console_without_debug(capsys):
    backend = _backend()
    backend._runner = _FakeRunnerRaises(RuntimeError("boom"))

    count = backend.count_messages_tokens([{"role": "user", "content": "hi"}])

    assert isinstance(count, int)   # falls back to super().count_messages_tokens
    out = capsys.readouterr().out
    assert "falling back to an estimate" in out, out


def test_rpc_failure_warns_once_per_process(capsys):
    backend = _backend()

    backend._runner = _FakeRunnerRaises(RuntimeError("boom"))
    backend.count_messages_tokens([{"role": "user", "content": "first"}])
    first_out = capsys.readouterr().out
    assert "falling back to an estimate" in first_out, first_out

    backend._runner = _FakeRunnerRaises(RuntimeError("boom again"))
    backend.count_messages_tokens([{"role": "user", "content": "second"}])
    second_out = capsys.readouterr().out
    assert "falling back to an estimate" not in second_out, second_out


def test_a_transient_busy_worker_stays_silent(capsys):
    """RunnerBusy (a live stream in progress) is expected and transient, and must
    never trip the permanent-failure console notice."""
    backend = _backend()
    backend._runner = _FakeRunnerBusy()

    backend.count_messages_tokens([{"role": "user", "content": "hi"}])

    out = capsys.readouterr().out
    assert "falling back to an estimate" not in out, out


def test_no_failure_stays_silent(capsys):
    backend = _backend()

    class _FakeRunnerOk:
        def count_messages_tokens(self, messages):
            return 42

    backend._runner = _FakeRunnerOk()

    count = backend.count_messages_tokens([{"role": "user", "content": "hi"}])

    assert count == 42
    out = capsys.readouterr().out
    assert "falling back to an estimate" not in out, out
