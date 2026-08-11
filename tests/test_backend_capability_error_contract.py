# SPDX-License-Identifier: AGPL-3.0-or-later
"""A backend declares what it can do, and a refusal reaches the caller with its reason.

Two halves of one contract, because fixing either alone leaves a rule 5 violation:

* A backend that cannot apply a grammar must SAY SO. ``Engine.validate_grammar``
  used to probe ``getattr(backend, "validate_grammar", None)`` and no-op when it
  was absent, which asked "does this backend have a validator" and acted on the
  answer as though it had asked "can this backend apply a grammar". Only the GGUF
  backend defined it, so a grammar sent to an HF model was never checked and,
  without the optional ``[grammar]`` extra, never applied - the caller got a 200
  full of unconstrained text with nothing saying the constraint had been dropped.

* The non-streaming path caught only ``RuntimeError``. Every user-actionable
  backend error is a ``ValueError`` subclass, so all of them escaped to the
  generic handler and came back as ``{"detail": "Internal server error"}`` with
  the reason discarded, while the STREAMING twin of the same function delivered
  that reason to the client.

Fixing the first alone turns a silent wrong answer into an opaque 500. Fixing the
second alone leaves grammar silently dropped. Hence one test module.
"""

import importlib.util
import json
from typing import Iterator, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.inference.backends.base import (
    GRAMMAR_UNSUPPORTED_MESSAGE,
    BaseBackend,
    EmbedBatchTooLargeError,
    GrammarUnsupportedError,
    ImageDecodeUnavailable,
    InvalidGrammarError,
    UnsupportedInputError,
    VisionInputError,
)
from localm.inference.engine import Engine
from localm.inference.http_server import backend_error_status, create_app


_TEXT_MSG = [{"role": "user", "content": "hello"}]
_GRAMMAR = 'root ::= "yes" | "no"'


# --------------------------------------------------------------------------- #
#  Fixtures that can express the failing case                                  #
#                                                                              #
#  A MagicMock engine is FULLY CAPABLE by construction: engine.validate_grammar #
#  is a Mock that returns a Mock and can never refuse anything. A capability    #
#  test built on one cannot fail on the capability defect, however real its     #
#  assertions look. So the capability half below uses a REAL BaseBackend        #
#  subclass inside a REAL Engine, and only the error-MAPPING half (where the    #
#  capability is not what is under test) uses a mock.                           #
# --------------------------------------------------------------------------- #

class _MinimalBackend(BaseBackend):
    """A backend that declares nothing beyond the abstract minimum.

    This is what a new backend author writes on day one, and the whole point of
    the deny-by-default contract is that it is SAFE - so this class must never
    grow a ``supports_grammar`` declaration.
    """

    def __init__(self) -> None:
        self.chat_stream_calls: List[dict] = []

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat_stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        # Records that generation actually ran. A capability refusal must happen
        # BEFORE this, so an empty list is the proof that nothing was generated.
        self.chat_stream_calls.append(dict(kwargs))
        yield "unconstrained text"

    @property
    def loaded(self) -> bool:
        return True


class _GrammarCapableBackend(_MinimalBackend):
    """Declares grammar support the supported way: one class attribute."""

    supports_grammar = True


class _EngineWithBackend(Engine):
    """A real Engine (real ``validate_grammar``, real ``supports_grammar``) over a
    hand-built backend.

    Subclassed rather than constructed, because ``Engine.__init__`` calls
    ``create_backend`` and would resolve a real model file. Every METHOD under
    test is inherited unchanged, which is the point: this exercises the real
    delegation path, not a re-implementation of it.
    """

    def __init__(self, backend: BaseBackend, display_name: str = "test-model") -> None:
        from localm.inference.engine import _LOAD_LOCK
        self.model_path = "test-model.gguf"
        self.display_name = display_name
        self._load_lock = _LOAD_LOCK
        self._backend = backend
        self.active_requests = 0
        self.unloading = False


def _raising_stream(exc: BaseException):
    """A chat_stream that raises when ITERATED, which is how a real backend fails.

    ``GgufBackend.chat_stream`` and ``HFBackend.chat_stream`` are generator
    functions, so calling them returns a generator and the guard fires on the
    first ``next()``. A side_effect that raised at CALL time would exercise a
    different code path in ``_generate_full`` (before its try block) than the one
    production actually takes.
    """
    def _stream(messages, **kwargs):
        raise exc
        yield ""     # unreachable; makes this a generator function
    return _stream


def _mock_engine(*, stream_exc: Optional[BaseException] = None):
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.supports_grammar = True          # NOT what these tests are about
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    type(engine).loaded = property(lambda self: True)
    if stream_exc is not None:
        engine.chat_stream.side_effect = _raising_stream(stream_exc)
    else:
        engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    return engine


def _post(engine, payload: dict, path: str = "/v1/chat/completions",
          raise_server_exceptions: bool = True):
    with TestClient(create_app(engine),
                    raise_server_exceptions=raise_server_exceptions) as client:
        return client.post(path, json=payload)


def _post_observing_500(engine, payload: dict, path: str = "/v1/chat/completions"):
    """POST and read the response even when the app's generic handler fired.

    ``raise_server_exceptions=False`` is REQUIRED to see an unhandled-error
    response at all: Starlette's ServerErrorMiddleware builds the 500 from the
    registered ``@app.exception_handler(Exception)`` and then RE-RAISES, so a real
    server logs the traceback after the client already has its JSON. With the
    default TestClient setting that re-raise reaches the test instead, and the
    body - the thing under test - is never observed.

    So this helper is not a way to make a red test green. It is the only way to
    assert on the opaque 500 the client actually receives in production.
    """
    return _post(engine, payload, path, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
#  1. The capability contract on the backend                                   #
# --------------------------------------------------------------------------- #

class TestBackendGrammarCapability:
    def test_a_backend_that_declares_nothing_denies(self):
        # The default is the whole safety property: an author who never heard of
        # supports_grammar still gets the honest answer.
        assert _MinimalBackend().supports_grammar is False

    def test_no_grammar_requested_is_never_refused(self):
        b = _MinimalBackend()
        b.validate_grammar(None)
        b.validate_grammar("")
        # No exception. An incapable backend is only a problem when a grammar is
        # actually asked for; refusing every request would break plain chat.

    def test_a_grammar_against_an_incapable_backend_is_refused_with_a_reason(self):
        b = _MinimalBackend()
        with pytest.raises(GrammarUnsupportedError) as ei:
            b.validate_grammar(_GRAMMAR)
        # The shared constant, not a message invented at the raise site, so the
        # HTTP body and the CLI say the same thing.
        assert str(ei.value) == GRAMMAR_UNSUPPORTED_MESSAGE

    def test_the_refusal_message_is_actionable(self):
        # A reason the caller cannot act on is barely better than no reason:
        # name what is wrong AND both routes out of it.
        assert "grammar" in GRAMMAR_UNSUPPORTED_MESSAGE.lower()
        assert "GGUF" in GRAMMAR_UNSUPPORTED_MESSAGE
        assert "localm[grammar]" in GRAMMAR_UNSUPPORTED_MESSAGE

    def test_declaring_support_permits_a_grammar(self):
        _GrammarCapableBackend().validate_grammar(_GRAMMAR)   # must not raise

    def test_gguf_declares_grammar_support(self):
        from localm.inference.backends.gguf import GgufBackend
        assert GgufBackend("does-not-need-to-exist.gguf").supports_grammar is True

    def test_hf_grammar_support_tracks_the_optional_extra(self, monkeypatch):
        # The HF backend's grammar support IS xgrammar. Reporting True without it
        # would keep the silent degrade alive behind an honest-looking flag.
        from localm.inference.backends.hf import HFBackend
        backend = HFBackend("does-not-need-to-exist")

        real_find_spec = importlib.util.find_spec

        def _fake(name, *a, **kw):
            if name == "xgrammar":
                return None
            return real_find_spec(name, *a, **kw)

        monkeypatch.setattr(importlib.util, "find_spec", _fake)
        assert backend.supports_grammar is False

        def _present(name, *a, **kw):
            if name == "xgrammar":
                return object()
            return real_find_spec(name, *a, **kw)

        monkeypatch.setattr(importlib.util, "find_spec", _present)
        assert backend.supports_grammar is True

    def test_a_broken_xgrammar_install_reports_no_support(self, monkeypatch):
        from localm.inference.backends.hf import HFBackend
        backend = HFBackend("does-not-need-to-exist")

        def _boom(name, *a, **kw):
            raise ValueError("__spec__ is None")

        monkeypatch.setattr(importlib.util, "find_spec", _boom)
        assert backend.supports_grammar is False


class TestGrammarUnsupportedIsNotConfusedWithOtherFailures:
    """The class's PLACE in the hierarchy is load-bearing, not decoration."""

    def test_it_is_a_value_error(self):
        assert issubclass(GrammarUnsupportedError, ValueError)

    def test_it_is_not_an_invalid_grammar_error(self):
        # Otherwise the existing `except InvalidGrammarError` arm would catch it
        # and answer "Invalid grammar: ...", sending the caller to fix a grammar
        # that has nothing wrong with it.
        assert not issubclass(GrammarUnsupportedError, InvalidGrammarError)

    def test_it_is_not_an_unsupported_input_error(self):
        # cli/chat.py's `except UnsupportedInputError` arm DISCARDS the message
        # and prints vision guidance in its place (see the ordering comment
        # there). Inheriting from it would make a grammar refusal surface as
        # advice about picking a vision model.
        assert not issubclass(GrammarUnsupportedError, UnsupportedInputError)


# --------------------------------------------------------------------------- #
#  2. Engine delegation - the end-to-end property                              #
#                                                                              #
#  MEASURED, because the obvious assumption about these is wrong: restoring     #
#  Engine's old `fn = getattr(self._backend, "validate_grammar", None)` probe   #
#  leaves ALL 36 tests in this file GREEN. That is not a gap in them. The probe #
#  simply stopped being load-bearing the moment validate_grammar moved onto     #
#  BaseBackend: it is now always found, so getattr and a direct call do the     #
#  same thing. The getattr form is stale style, not a latent defect.            #
#                                                                              #
#  What CAN bring the defect back is the base class losing the capability, and  #
#  that is what these guard. Deleting supports_grammar and validate_grammar     #
#  from BaseBackend turns this section red with "DID NOT RAISE                  #
#  GrammarUnsupportedError" and turns the route tests below red with            #
#  "assert 200 == 400" - a success response carrying unconstrained text, which  #
#  is the original bug exactly.                                                 #
# --------------------------------------------------------------------------- #

class TestEngineDelegatesTheCapability:
    def test_engine_refuses_a_grammar_the_backend_cannot_apply(self):
        engine = _EngineWithBackend(_MinimalBackend())
        with pytest.raises(GrammarUnsupportedError):
            engine.validate_grammar(_GRAMMAR)

    def test_engine_allows_a_grammar_the_backend_supports(self):
        _EngineWithBackend(_GrammarCapableBackend()).validate_grammar(_GRAMMAR)

    def test_engine_reports_the_backend_capability(self):
        assert _EngineWithBackend(_MinimalBackend()).supports_grammar is False
        assert _EngineWithBackend(_GrammarCapableBackend()).supports_grammar is True


# --------------------------------------------------------------------------- #
#  3. The status table                                                         #
# --------------------------------------------------------------------------- #

class TestBackendErrorStatusTable:
    @pytest.mark.parametrize("exc,status", [
        (ImageDecodeUnavailable("no decoder"), 501),
        (VisionInputError("bad image"), 400),
        (UnsupportedInputError("no images"), 400),
        (GrammarUnsupportedError("no grammar"), 400),
        (InvalidGrammarError("bad grammar"), 400),
        (EmbedBatchTooLargeError("too many"), 413),
    ])
    def test_each_family_maps_to_its_status(self, exc, status):
        assert backend_error_status(exc) == status

    def test_arm_order_puts_the_subclasses_first(self):
        # ImageDecodeUnavailable and VisionInputError are BOTH
        # UnsupportedInputError subclasses. A base-class-first table would answer
        # 400 for both and report a missing image decoder as the caller's bad
        # input. This asserts the ORDER, which is the part a refactor loses.
        assert backend_error_status(ImageDecodeUnavailable("x")) == 501
        assert backend_error_status(ImageDecodeUnavailable("x")) != \
            backend_error_status(UnsupportedInputError("x"))

    def test_an_unrelated_value_error_is_not_claimed(self):
        # The guard against over-widening. Every mapped class IS a ValueError, so
        # a broad `except ValueError` would also catch a real bug in unrelated
        # code and report it to the user as their own bad input.
        assert backend_error_status(ValueError("a genuine bug")) is None

    def test_a_genuine_defect_is_not_claimed(self):
        assert backend_error_status(AttributeError("engine has no method")) is None
        assert backend_error_status(RuntimeError("out of VRAM")) is None

    def test_the_except_tuple_matches_the_table(self):
        # Derived, not duplicated: a class in the catch but not the table would
        # raise HTTPException(None, ...) the first time it ever fired.
        assert set(hs._BACKEND_ERROR_TYPES) == {t for t, _ in hs._BACKEND_ERROR_STATUS}
        for t in hs._BACKEND_ERROR_TYPES:
            assert backend_error_status(t("x")) is not None


# --------------------------------------------------------------------------- #
#  4. The non-streaming HTTP path reports the reason                           #
# --------------------------------------------------------------------------- #

class TestNonStreamingReportsTheReason:
    @pytest.mark.parametrize("exc,status,needle", [
        (ImageDecodeUnavailable("Pillow is not installed"), 501, "Pillow"),
        (VisionInputError("mtmd_bitmap_init failed (bad image buffer)"), 400,
         "mtmd_bitmap_init"),
        (UnsupportedInputError("This model cannot accept image input"), 400,
         "cannot accept image input"),
        (InvalidGrammarError("grammar failed to parse at 'root'"), 400,
         "failed to parse"),
    ])
    def test_each_family_returns_its_status_and_its_reason(self, exc, status, needle):
        r = _post(_mock_engine(stream_exc=exc),
                  {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        # The BODY first and the status second, deliberately. The reason reaching
        # the caller is the property; the status is how it is reported. A body
        # assertion that fails says "the caller was told nothing", which is hard
        # to argue with; a status assertion that fails says "400 != 500", which
        # reads like a number to adjust.
        assert needle in r.json()["detail"]
        assert r.json()["detail"] != "Internal server error"
        assert r.status_code == status

    def test_completions_route_reports_the_reason_too(self):
        # /v1/completions had NO exception handling around its non-streaming
        # generation at all, so every one of these reached the generic backstop.
        r = _post(_mock_engine(stream_exc=VisionInputError("bad image buffer")),
                  {"model": "test-model", "prompt": "hi", "stream": False},
                  path="/v1/completions")
        assert "bad image buffer" in r.json()["detail"]
        assert r.status_code == 400

    def test_a_working_request_is_untouched(self):
        r = _post(_mock_engine(),
                  {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"


class TestAGenuineBugStillReturnsAnOpaque500:
    """The other half of D3: do not widen the catch until a real defect is
    reported to the user as their fault."""

    def test_an_attribute_error_is_an_opaque_500(self):
        r = _post_observing_500(
            _mock_engine(stream_exc=AttributeError("'Mock' has no 'foo'")),
            {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        assert r.json() == {"detail": "Internal server error"}
        assert r.status_code == 500
        # The internal detail must NOT leak into a user-facing body.
        assert "Mock" not in r.text

    def test_an_unrelated_value_error_is_an_opaque_500(self):
        # The sharpest case: it IS a ValueError, like every mapped class, but it
        # is a bug rather than a refusal. A catch widened to `except ValueError`
        # would turn this into a 400 blaming the caller.
        r = _post_observing_500(
            _mock_engine(stream_exc=ValueError("int() got a str")),
            {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        assert r.json() == {"detail": "Internal server error"}
        assert r.status_code == 500
        assert "int() got a str" not in r.text


# --------------------------------------------------------------------------- #
#  5. The two paths agree                                                      #
# --------------------------------------------------------------------------- #

class TestBothPathsAgree:
    def _sse_text(self, body: str) -> str:
        """Concatenate every delta the stream carried, so an assertion can look
        for a reason without caring how it was chunked."""
        out = []
        for line in body.splitlines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            payload = json.loads(line[len("data: "):])
            for choice in payload.get("choices", []):
                out.append((choice.get("delta") or {}).get("content") or "")
        return "".join(out)

    def _finish_reasons(self, body: str) -> list:
        out = []
        for line in body.splitlines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            for choice in json.loads(line[len("data: "):]).get("choices", []):
                if choice.get("finish_reason"):
                    out.append(choice["finish_reason"])
        return out

    def test_a_grammar_refusal_is_byte_identical_on_both_paths(self):
        # The capability check runs BEFORE the route reads req.stream, so this
        # one CAN agree on the status - and it does, exactly.
        #
        # A real Engine over a real incapable backend: with a mock engine,
        # validate_grammar would be a Mock that never refuses, and this test
        # could not fail no matter what the route did.
        backend = _MinimalBackend()
        engine = _EngineWithBackend(backend)
        base = {"model": "test-model", "messages": _TEXT_MSG, "grammar": _GRAMMAR}

        streamed = _post(engine, {**base, "stream": True})
        plain = _post(engine, {**base, "stream": False})

        assert plain.status_code == streamed.status_code == 400
        assert plain.json()["detail"] == streamed.json()["detail"]
        assert "localm[grammar]" in plain.json()["detail"]
        # Nothing was generated on either path. This is the item-18 property
        # itself: the old behaviour was a 200 carrying unconstrained text.
        assert backend.chat_stream_calls == []

    def test_a_capable_backend_still_gets_its_grammar(self):
        # The refusal must not become a blanket ban - fires the other way.
        backend = _GrammarCapableBackend()
        engine = _EngineWithBackend(backend)
        r = _post(engine, {"model": "test-model", "messages": _TEXT_MSG,
                           "grammar": _GRAMMAR, "stream": False})
        assert r.status_code == 200
        assert len(backend.chat_stream_calls) == 1
        assert backend.chat_stream_calls[0]["grammar"] == _GRAMMAR

    def test_the_completions_route_refuses_a_grammar_identically(self):
        backend = _MinimalBackend()
        engine = _EngineWithBackend(backend)
        r = _post(engine, {"model": "test-model", "prompt": "hi",
                           "grammar": _GRAMMAR, "stream": False},
                  path="/v1/completions")
        assert r.status_code == 400
        assert "localm[grammar]" in r.json()["detail"]
        assert backend.chat_stream_calls == []

    # EVERY mid-generation family, paired. A family covered only on the
    # non-streaming side cannot fail on a regression that re-opens the gap for
    # that family specifically, and the gap between the two paths IS the defect
    # this module exists for - so one representative is not enough.
    #
    # EmbedBatchTooLargeError is absent by construction, not by oversight: it is
    # raised by HFBackend.embed, and /v1/embeddings has no streaming form to pair
    # against. Its non-streaming status is asserted in the table tests above.
    @pytest.mark.parametrize("exc_factory,status,reason", [
        (ImageDecodeUnavailable, 501, "Pillow is not installed"),
        (VisionInputError, 400, "mtmd_bitmap_init failed (bad image buffer)"),
        (UnsupportedInputError, 400, "This model cannot accept image input"),
        (InvalidGrammarError, 400, "grammar failed to parse at 'root'"),
    ])
    def test_a_mid_generation_reason_reaches_the_client_on_both_paths(
            self, exc_factory, status, reason):
        # These surface only once generation has started, and the streaming path
        # has already committed its 200 by then (the role chunk goes out before
        # the generator runs, so the headers are gone). The STATUS therefore
        # cannot agree, and saying it does would be a claim the transport cannot
        # support. What CAN agree, and what was missing, is that the reason
        # reaches the caller at all and that the failure is machine-detectable.
        payload = {"model": "test-model", "messages": _TEXT_MSG}

        streamed = _post(_mock_engine(stream_exc=exc_factory(reason)),
                         {**payload, "stream": True})
        plain = _post(_mock_engine(stream_exc=exc_factory(reason)),
                      {**payload, "stream": False})

        # Same input, same failure: the reason reaches the caller on BOTH.
        assert reason in self._sse_text(streamed.text)
        assert reason in plain.json()["detail"]
        assert plain.json()["detail"] != "Internal server error"
        # Machine-detectable on both: the stream marks its terminal frame, the
        # non-streaming path uses the status line.
        assert "error" in self._finish_reasons(streamed.text)
        assert plain.status_code == status

    @pytest.mark.parametrize("exc_factory,reason", [
        (ImageDecodeUnavailable, "Pillow is not installed"),
        (VisionInputError, "mtmd_bitmap_init failed (bad image buffer)"),
        (UnsupportedInputError, "This model cannot accept image input"),
        (InvalidGrammarError, "grammar failed to parse at 'root'"),
    ])
    def test_the_completions_route_agrees_with_its_own_stream_too(
            self, exc_factory, reason):
        # /v1/completions has the same two-path split and is the route that had
        # NO handler at all, so it gets the same paired treatment rather than
        # inheriting confidence from the chat route's coverage.
        payload = {"model": "test-model", "prompt": "hi"}

        streamed = _post(_mock_engine(stream_exc=exc_factory(reason)),
                         {**payload, "stream": True}, path="/v1/completions")
        plain = _post(_mock_engine(stream_exc=exc_factory(reason)),
                      {**payload, "stream": False}, path="/v1/completions")

        assert reason in streamed.text
        assert reason in plain.json()["detail"]
        assert plain.json()["detail"] != "Internal server error"
        assert plain.status_code == backend_error_status(exc_factory(reason))
