# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reply during which the grammar constraint failed is reported as failed
(GrammarUnsupportedError after its text), not returned as if it matched the
grammar; the model then refuses grammar requests. Both the GGUF and the
HuggingFace backend."""

from __future__ import annotations

import pytest

from localm.inference.backends.base import (
    GRAMMAR_FAILED_THIS_REPLY_MESSAGE, GrammarUnsupportedError,
)
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.hf import HFBackend

_MSG = [{"role": "user", "content": "answer in json"}]


class _FakeRunner:
    def __init__(self, done: dict):
        self.last_done = None
        self._done = done

    def chat_stream(self, **_kwargs):
        yield "{"
        yield "not json"
        self.last_done = self._done


def _gguf():
    b = GgufBackend("does-not-exist.gguf", n_ctx=512)
    b._loaded = True
    return b


def _hf():
    b = HFBackend("does-not-need-to-exist")
    b._loaded = True
    return b


@pytest.mark.parametrize("make", [_gguf, _hf], ids=["gguf", "hf"])
def test_the_failing_reply_is_reported_and_later_grammar_is_refused(make):
    backend = make()
    backend._runner = _FakeRunner({"finish_reason": "stop", "grammar_unsupported": True})
    seen = []
    with pytest.raises(GrammarUnsupportedError) as exc:
        for tok in backend.chat_stream(_MSG, max_tokens=8, grammar='root ::= "{"'):
            seen.append(tok)
    assert seen == ["{", "not json"], "the text that was generated still streams"
    assert str(exc.value) == GRAMMAR_FAILED_THIS_REPLY_MESSAGE
    assert backend._grammar_unsupported is True
    with pytest.raises(GrammarUnsupportedError):
        backend.validate_grammar('root ::= "{"')


def test_a_stall_stop_is_a_warning_without_a_traceback():
    import logging

    from localm.inference.backends.llamacpp._runner import _first_token_stall

    class _Stalling:
        def chat_stream(self, **_kwargs):
            yield from ()
            raise RuntimeError(_first_token_stall(900))

        def shutdown(self, *a, **k):
            pass

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    lg = logging.getLogger("localm")
    h, old = _Collect(logging.DEBUG), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        b = _gguf()
        b._runner = _Stalling()
        with pytest.raises(RuntimeError, match="Generation stalled"):
            list(b.chat_stream(_MSG, max_tokens=8))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    loud = [r for r in records if r.levelno >= logging.WARNING]
    assert [r.levelno for r in loud] == [logging.WARNING]
    assert "stall limit" in loud[0].getMessage()
    assert not any(r.exc_info for r in records), "a stall we chose has no traceback"


@pytest.mark.parametrize("value,on", [
    ("1", True), (" TRUE ", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("off", False), ("no", False),
])
def test_debug_switches_read_0_and_false_as_off(monkeypatch, value, on):
    from localm.inference.backends.llamacpp._abi import env_on
    monkeypatch.setenv("LOCALM_MTMD_CPU", value)
    assert env_on("LOCALM_MTMD_CPU") is on


def test_the_kv_cache_advice_says_f16_does_not_help_a_memory_shortage():
    from localm.kv_cache import context_refused_note
    note = context_refused_note("q8_0")
    assert "'KV cache type' in Settings > Model" in note
    assert "does not help when memory is short" in note
    assert context_refused_note("f16") == ""


@pytest.mark.parametrize("make", [_gguf, _hf], ids=["gguf", "hf"])
def test_a_reply_that_asked_for_no_grammar_is_not_reported(make):
    backend = make()
    backend._runner = _FakeRunner({"finish_reason": "stop", "grammar_unsupported": True})
    assert list(backend.chat_stream(_MSG, max_tokens=8)) == ["{", "not json"]
    assert backend._grammar_unsupported is True


@pytest.mark.parametrize("make", [_gguf, _hf], ids=["gguf", "hf"])
def test_a_reply_whose_grammar_held_is_not_reported(make):
    backend = make()
    backend._runner = _FakeRunner({"finish_reason": "stop", "grammar_unsupported": False})
    assert list(backend.chat_stream(_MSG, max_tokens=8, grammar='root ::= "{"')) == [
        "{", "not json"]
    assert not getattr(backend, "_grammar_unsupported", False)
