"""NEW-LAZY-GRAMMAR-SILENT-UNCONSTRAINED: a dropped lazy grammar must reach the caller."""

from __future__ import annotations

import inspect
import sys
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    GRAMMAR_LAZY_UNSUPPORTED_MESSAGE,
    GRAMMAR_UNSUPPORTED_MESSAGE,
    BaseBackend,
    GrammarUnsupportedError,
    InvalidGrammarError,
)
from localm.inference.backends.hf import HFBackend
from localm.inference.engine import Engine
from localm.inference.http_server import create_app

_TEXT_MSG = [{"role": "user", "content": "hello"}]
_GRAMMAR = 'root ::= "yes" | "no"'
_TRIGGERS = [r"(<tool_call>[\s\S]*)"]


# --------------------------------------------------------------------------- #
#  Harness                                                                     #
# --------------------------------------------------------------------------- #

class _RecordingBackend(BaseBackend):
    """Records whether generation actually ran."""

    supports_grammar = True

    def __init__(self) -> None:
        self.chat_stream_calls: List[dict] = []

    def load(self) -> None: ...
    def unload(self) -> None: ...

    @property
    def loaded(self) -> bool:
        return True

    def chat_stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        self.chat_stream_calls.append(dict(kwargs))
        yield "unconstrained text"


class _HFLikeBackend(HFBackend):
    """The REAL ``HFBackend.validate_grammar``, with only the model half stubbed."""

    supports_grammar = True

    def __init__(self) -> None:
        self.model_path = "test-model"
        self._runner = None
        self._loaded = True
        self._supports_images = False
        self._can_embed = False
        self._is_multimodal = False
        self.chat_stream_calls: List[dict] = []

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None: ...
    def unload(self) -> None: ...


class _HFLikeRecording(_HFLikeBackend):
    """As above, but generation is a recorder so the route tests can prove it never ran. ``HFBackend.chat_stream``'s own guard is tested separately, unstubbed."""

    def chat_stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        self.chat_stream_calls.append(dict(kwargs))
        yield "unconstrained text"


class _EngineWithBackend(Engine):
    """A real Engine (real ``validate_grammar`` delegation) over a hand-built backend."""

    def __init__(self, backend: BaseBackend) -> None:
        from localm.inference.engine import _LOAD_LOCK
        self.model_path = "test-model.gguf"
        self.display_name = "test-model"
        self._load_lock = _LOAD_LOCK
        self._backend = backend
        self.active_requests = 0
        self.unloading = False


def _post(engine, payload: dict):
    with TestClient(create_app(engine)) as client:
        return client.post("/v1/chat/completions", json=payload)


def _lazy_payload(**over) -> dict:
    body = {
        "model": "test-model",
        "messages": _TEXT_MSG,
        "grammar": _GRAMMAR,
        "grammar_lazy": True,
        "grammar_triggers": _TRIGGERS,
        "stream": False,
    }
    body.update(over)
    return body


# --------------------------------------------------------------------------- #
#  THE ACCEPTANCE CRITERION, at the layer where the collapse happened          #
# --------------------------------------------------------------------------- #

def test_lazy_refusal_reaches_the_caller_and_nothing_is_generated():
    """The whole defect in one test: ask for lazy, get told, generate nothing."""
    backend = _HFLikeRecording()
    r = _post(_EngineWithBackend(backend), _lazy_payload())

    # DATA FIRST: no unconstrained text was produced. This is the property; the
    # status code below is only the proxy for it.
    assert backend.chat_stream_calls == [], (
        "generation RAN for a lazy grammar this backend cannot apply - the caller "
        "received unconstrained text presented as a normal completion")
    assert r.status_code == 400
    body = r.text
    # WHICH refusal fired. Without this the test passes on an install where
    # supports_grammar is False and the blanket refusal fired instead, which would
    # be green with the lazy fix reverted entirely.
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in body
    assert GRAMMAR_UNSUPPORTED_MESSAGE not in body


def test_lazy_refusal_is_identical_on_the_streaming_path():
    """Streaming must refuse with the same status and reason, not open a 200 SSE stream and then degrade inside it - once bytes are committed the caller can no longer be told anything by the response."""
    backend = _HFLikeRecording()
    r = _post(_EngineWithBackend(backend), _lazy_payload(stream=True))

    assert backend.chat_stream_calls == []
    assert r.status_code == 400
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in r.text


def test_a_backend_that_can_do_lazy_is_still_served():
    """The control, and the no-regression guard for the GGUF path."""
    backend = _RecordingBackend()
    r = _post(_EngineWithBackend(backend), _lazy_payload())

    assert r.status_code == 200, r.text
    assert len(backend.chat_stream_calls) == 1
    assert backend.chat_stream_calls[0].get("grammar_lazy") is True
    assert backend.chat_stream_calls[0].get("grammar") == _GRAMMAR


def test_non_lazy_grammar_on_the_same_backend_is_untouched():
    """The lazy refusal must not leak into the ordinary grammar path: this backend can constrain generation, it just cannot do it lazily."""
    backend = _HFLikeRecording()
    r = _post(_EngineWithBackend(backend), _lazy_payload(grammar_lazy=False,
                                                         grammar_triggers=None))
    assert r.status_code == 200, r.text
    assert len(backend.chat_stream_calls) == 1


# --------------------------------------------------------------------------- #
#  The backend-level backstop: a caller that never went through the route      #
# --------------------------------------------------------------------------- #

def test_hf_validate_grammar_names_the_lazy_mode_not_the_missing_extra():
    b = _HFLikeBackend()
    b.validate_grammar(_GRAMMAR)                 # non-lazy: allowed
    b.validate_grammar(None, lazy=True)          # no grammar: nothing to refuse
    with pytest.raises(GrammarUnsupportedError) as ei:
        b.validate_grammar(_GRAMMAR, lazy=True)
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)
    assert GRAMMAR_UNSUPPORTED_MESSAGE not in str(ei.value)


def test_hf_chat_stream_refuses_lazy_even_when_validate_grammar_was_skipped():
    """Mirrors the image-rejection guarantee directly above it in chat_stream: ANY caller is refused, including one that never loaded the model or never called validate_grammar. ``_runner`` is None here, so if the guard did not fire first the failure would be a RuntimeError about the model not being loaded..."""
    b = _HFLikeBackend()
    with pytest.raises(GrammarUnsupportedError) as ei:
        list(b.chat_stream(_TEXT_MSG, grammar=_GRAMMAR, grammar_lazy=True,
                           grammar_triggers=_TRIGGERS))
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)


def test_every_validate_grammar_override_accepts_the_lazy_keyword():
    """The routes now call ``validate_grammar(g, lazy=...)`` by keyword."""
    from localm.inference.backends.gguf import GgufBackend

    for cls in (BaseBackend, GgufBackend, HFBackend, Engine):
        params = inspect.signature(cls.validate_grammar).parameters
        assert "lazy" in params, f"{cls.__name__}.validate_grammar dropped `lazy`"
        assert params["lazy"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{cls.__name__}.validate_grammar must take `lazy` keyword-only")


# --------------------------------------------------------------------------- #
#  The worker: never return "unconstrained" to mean "I gave up"                #
# --------------------------------------------------------------------------- #

def test_grammar_processor_raises_when_xgrammar_is_missing():
    """xgrammar is genuinely absent in this venv, so this exercises the real ImportError arm rather than a simulated one."""
    from localm.inference.backends._hf_worker import _grammar_processor

    assert "xgrammar" not in sys.modules or sys.modules.get("xgrammar") is None
    with pytest.raises(GrammarUnsupportedError):
        _grammar_processor(_GRAMMAR, object(), object())


def test_grammar_processor_returns_none_only_when_no_grammar_was_asked_for():
    from localm.inference.backends._hf_worker import _grammar_processor

    assert _grammar_processor(None, object(), object()) is None
    assert _grammar_processor("", object(), object()) is None


def test_grammar_processor_raises_invalid_grammar_when_compilation_fails(monkeypatch):
    """A grammar xgrammar cannot compile is the same event as the native GBNF parser returning NULL on the GGUF side, which already raises InvalidGrammarError and becomes a 400 naming the grammar."""
    import types

    from localm.inference.backends import _hf_worker

    xgr = types.ModuleType("xgrammar")

    class _Info:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size=None):
            return object()

    class _Compiler:
        def __init__(self, info): ...
        def compile_grammar(self, grammar):
            raise ValueError("bad rule at line 1")

    xgr.TokenizerInfo = _Info
    xgr.GrammarCompiler = _Compiler
    contrib = types.ModuleType("xgrammar.contrib")
    hf_mod = types.ModuleType("xgrammar.contrib.hf")
    hf_mod.LogitsProcessor = lambda compiled: object()
    monkeypatch.setitem(sys.modules, "xgrammar", xgr)
    monkeypatch.setitem(sys.modules, "xgrammar.contrib", contrib)
    monkeypatch.setitem(sys.modules, "xgrammar.contrib.hf", hf_mod)
    # `transformers` MUST be stubbed too, and it is not belt-and-braces: the
    # code under test imports xgrammar AND `from transformers import
    # LogitsProcessorList` in the SAME try, so on an install without
    # transformers that import raises first and the except-ImportError arm
    # returns GrammarUnsupportedError - never reaching the compile failure this
    # test exists to pin. transformers is in NEITHER core nor the dev/rag
    # extras, so CI (`.[dev,rag]`) never has it while a dev venv usually does:
    # measured 2026-08-13, this test passed locally and failed on both CI
    # platforms for exactly that reason, and it had been red on master
    # unnoticed because nothing was running the suite. Stubbing it makes the
    # test independent of whether transformers is installed, which is what its
    # docstring already claims ("the code path under test is this module's own
    # except-arm").
    tfm = types.ModuleType("transformers")
    tfm.LogitsProcessorList = list
    monkeypatch.setitem(sys.modules, "transformers", tfm)

    with pytest.raises(InvalidGrammarError) as ei:
        _hf_worker._grammar_processor(_GRAMMAR, object(), object())
    assert "bad rule at line 1" in str(ei.value)


def test_hf_worker_chat_stream_refuses_lazy_rather_than_dropping_it():
    """Third line of defence, for a caller driving HFWorker directly."""
    from localm.inference.backends._hf_worker import HFWorker

    w = object.__new__(HFWorker)
    w._is_multimodal = False
    with pytest.raises(GrammarUnsupportedError) as ei:
        list(HFWorker.chat_stream(w, _TEXT_MSG, grammar=_GRAMMAR,
                                  grammar_lazy=True, grammar_triggers=_TRIGGERS))
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE in str(ei.value)


# --------------------------------------------------------------------------- #
#  The two messages must stay distinguishable                                  #
# --------------------------------------------------------------------------- #

def test_the_two_refusal_messages_cannot_be_confused():
    """``coder/agent/context.py`` routes its recovery by SUBSTRING-matching these two messages, and the lazy one is tested first."""
    assert GRAMMAR_LAZY_UNSUPPORTED_MESSAGE not in GRAMMAR_UNSUPPORTED_MESSAGE
    assert GRAMMAR_UNSUPPORTED_MESSAGE not in GRAMMAR_LAZY_UNSUPPORTED_MESSAGE
    assert "lazy" in GRAMMAR_LAZY_UNSUPPORTED_MESSAGE.lower()
