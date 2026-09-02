# SPDX-License-Identifier: AGPL-3.0-or-later
"""A backend declares what it can do, and a refusal reaches the caller with its reason.

Two halves of one contract:

* A backend that cannot apply a grammar SAYS SO. ``BaseBackend`` owns
  ``supports_grammar`` and ``validate_grammar``, so ``Engine.validate_grammar``
  asks a declared capability rather than probing for the presence of a method.
  A grammar sent to a backend that cannot apply it is refused, never dropped
  into a 200 full of unconstrained text.

* The non-streaming path maps every user-actionable backend error (all
  ``ValueError`` subclasses) to a status with its reason, matching what the
  STREAMING twin of the same function already delivers. A genuine defect still
  reaches the generic handler as an opaque 500.
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
    ContextCapacityExceededError,
    EmbedBatchTooLargeError,
    GrammarUnsupportedError,
    ImageDecodeUnavailable,
    InvalidGrammarError,
    PretokenizerUnsafeInputError,
    TriggerValidatorUnavailableError,
    UnsupportedInputError,
    VisionInputError,
)
from localm.inference.engine import Engine
from localm.inference.http_server import backend_error_status, create_app


_TEXT_MSG = [{"role": "user", "content": "hello"}]
_GRAMMAR = 'root ::= "yes" | "no"'


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
#                                                                              #
#  The capability tests use a real BaseBackend subclass inside a real Engine;  #
#  the error-mapping tests use a mock engine.                                  #
# --------------------------------------------------------------------------- #

class _MinimalBackend(BaseBackend):
    """A backend that declares nothing beyond the abstract minimum, standing in
    for the deny-by-default contract.

    It must never grow a ``supports_grammar`` declaration.
    """

    def __init__(self) -> None:
        self.chat_stream_calls: List[dict] = []

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat_stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        # Records that generation ran; an empty list means nothing was generated.
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

    Subclassed rather than constructed: ``Engine.__init__`` calls
    ``create_backend``, which resolves a real model file. Every METHOD under
    test is inherited unchanged, so this exercises the real delegation path.
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
    first ``next()``.
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
    """
    return _post(engine, payload, path, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
#  1. The capability contract on the backend                                   #
# --------------------------------------------------------------------------- #

class TestBackendGrammarCapability:
    def test_a_backend_that_declares_nothing_denies(self):
        # The default answer, for a backend that never sets supports_grammar.
        assert _MinimalBackend().supports_grammar is False

    def test_no_grammar_requested_is_never_refused(self):
        b = _MinimalBackend()
        b.validate_grammar(None)
        b.validate_grammar("")
        # An incapable backend refuses only when a grammar is actually asked for.

    def test_a_grammar_against_an_incapable_backend_is_refused_with_a_reason(self):
        b = _MinimalBackend()
        with pytest.raises(GrammarUnsupportedError) as ei:
            b.validate_grammar(_GRAMMAR)
        # The shared constant, not a message invented at the raise site.
        assert str(ei.value) == GRAMMAR_UNSUPPORTED_MESSAGE

    def test_the_refusal_message_is_actionable(self):
        # The reason names what is wrong and both routes out of it.
        assert "grammar" in GRAMMAR_UNSUPPORTED_MESSAGE.lower()
        assert "GGUF" in GRAMMAR_UNSUPPORTED_MESSAGE
        assert "localm[grammar]" in GRAMMAR_UNSUPPORTED_MESSAGE

    def test_declaring_support_permits_a_grammar(self):
        _GrammarCapableBackend().validate_grammar(_GRAMMAR)   # must not raise

    def test_gguf_declares_grammar_support(self):
        from localm.inference.backends.gguf import GgufBackend
        assert GgufBackend("does-not-need-to-exist.gguf").supports_grammar is True

    def test_hf_grammar_support_tracks_the_optional_extra(self, monkeypatch):
        # The HF backend reports grammar support only when xgrammar is present.
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
    """The class's PLACE in the hierarchy decides which except arm claims it."""

    def test_it_is_a_value_error(self):
        assert issubclass(GrammarUnsupportedError, ValueError)

    def test_it_is_not_an_invalid_grammar_error(self):
        # Not an InvalidGrammarError, so the except InvalidGrammarError arm does
        # not claim it.
        assert not issubclass(GrammarUnsupportedError, InvalidGrammarError)

    def test_it_is_not_an_unsupported_input_error(self):
        # Not an UnsupportedInputError, whose cli/chat.py arm discards the
        # message and prints vision guidance instead.
        assert not issubclass(GrammarUnsupportedError, UnsupportedInputError)


# --------------------------------------------------------------------------- #
#  2. Engine delegation - the end-to-end property                              #
#                                                                              #
#  These go red when BaseBackend loses supports_grammar / validate_grammar.    #
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
        (TriggerValidatorUnavailableError("probe pool busy"), 503),
        (EmbedBatchTooLargeError("too many"), 413),
        (ContextCapacityExceededError("too long"), 413),
        (PretokenizerUnsafeInputError("run too long"), 400),
    ])
    def test_each_family_maps_to_its_status(self, exc, status):
        assert backend_error_status(exc) == status

    def test_arm_order_puts_the_subclasses_first(self):
        # ImageDecodeUnavailable and VisionInputError are both
        # UnsupportedInputError subclasses, so the table order decides the status.
        assert backend_error_status(ImageDecodeUnavailable("x")) == 501
        assert backend_error_status(ImageDecodeUnavailable("x")) != \
            backend_error_status(UnsupportedInputError("x"))

        # TriggerValidatorUnavailableError is an InvalidGrammarError, so it must
        # be listed before its parent.
        assert backend_error_status(TriggerValidatorUnavailableError("x")) == 503
        assert backend_error_status(TriggerValidatorUnavailableError("x")) != \
            backend_error_status(InvalidGrammarError("x"))

    def test_the_unavailable_error_stays_catchable_as_an_invalid_grammar_error(self):
        # Existing except InvalidGrammarError arms must keep catching it.
        assert issubclass(TriggerValidatorUnavailableError, InvalidGrammarError)

    def test_an_unrelated_value_error_is_not_claimed(self):
        # Every mapped class is a ValueError, so the catch must not widen to
        # ValueError itself.
        assert backend_error_status(ValueError("a genuine bug")) is None

    def test_a_genuine_defect_is_not_claimed(self):
        assert backend_error_status(AttributeError("engine has no method")) is None
        assert backend_error_status(RuntimeError("out of VRAM")) is None

    def test_the_except_tuple_matches_the_table(self):
        # Derived from the catch tuple, not duplicated.
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
        # The body is asserted before the status.
        assert needle in r.json()["detail"]
        assert r.json()["detail"] != "Internal server error"
        assert r.status_code == status

    def test_completions_route_reports_the_reason_too(self):
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
    """A genuine defect is still an opaque 500, never reported to the user as
    their fault."""

    def test_an_attribute_error_is_an_opaque_500(self):
        r = _post_observing_500(
            _mock_engine(stream_exc=AttributeError("'Mock' has no 'foo'")),
            {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        assert r.json() == {"detail": "Internal server error"}
        assert r.status_code == 500
        # The internal detail does not reach a user-facing body.
        assert "Mock" not in r.text

    def test_an_unrelated_value_error_is_an_opaque_500(self):
        # A ValueError that is not one of the mapped families.
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
        """Concatenate every delta the stream carried, so an assertion can find
        a reason regardless of how it was chunked."""
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
        # The capability check runs before the route reads req.stream, so both
        # paths agree on the status. A real Engine over a real incapable backend.
        backend = _MinimalBackend()
        engine = _EngineWithBackend(backend)
        base = {"model": "test-model", "messages": _TEXT_MSG, "grammar": _GRAMMAR}

        streamed = _post(engine, {**base, "stream": True})
        plain = _post(engine, {**base, "stream": False})

        assert plain.status_code == streamed.status_code == 400
        assert plain.json()["detail"] == streamed.json()["detail"]
        assert "localm[grammar]" in plain.json()["detail"]
        # Nothing was generated on either path.
        assert backend.chat_stream_calls == []

    def test_a_capable_backend_still_gets_its_grammar(self):
        # The refusal is not a blanket ban.
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

    # Every mid-generation family, paired across both paths.
    #
    # EmbedBatchTooLargeError is absent: it is raised by HFBackend.embed, and
    # /v1/embeddings has no streaming form to pair against. Its non-streaming
    # status is asserted in the table tests above.
    @pytest.mark.parametrize("exc_factory,status,reason", [
        (ImageDecodeUnavailable, 501, "Pillow is not installed"),
        (VisionInputError, 400, "mtmd_bitmap_init failed (bad image buffer)"),
        (UnsupportedInputError, 400, "This model cannot accept image input"),
        (InvalidGrammarError, 400, "grammar failed to parse at 'root'"),
    ])
    def test_a_mid_generation_reason_reaches_the_client_on_both_paths(
            self, exc_factory, status, reason):
        # These surface only once generation has started, after the streaming
        # path has committed its 200, so the STATUS cannot agree. What is
        # asserted instead is that its reason reaches the caller on both paths
        # and that the failure is machine-detectable on both.
        payload = {"model": "test-model", "messages": _TEXT_MSG}

        streamed = _post(_mock_engine(stream_exc=exc_factory(reason)),
                         {**payload, "stream": True})
        plain = _post(_mock_engine(stream_exc=exc_factory(reason)),
                      {**payload, "stream": False})

        # Same input, same failure: its reason reaches the caller on both.
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
        # /v1/completions has the same two-path split and gets the same paired
        # treatment.
        payload = {"model": "test-model", "prompt": "hi"}

        streamed = _post(_mock_engine(stream_exc=exc_factory(reason)),
                         {**payload, "stream": True}, path="/v1/completions")
        plain = _post(_mock_engine(stream_exc=exc_factory(reason)),
                      {**payload, "stream": False}, path="/v1/completions")

        assert reason in streamed.text
        assert reason in plain.json()["detail"]
        assert plain.json()["detail"] != "Internal server error"
        assert plain.status_code == backend_error_status(exc_factory(reason))


# --------------------------------------------------------------------------- #
#  6. The RuntimeError family - a GENERATION failure, not a backend refusal    #
#                                                                              #
#  A RuntimeError (not enough free VRAM, a conversation past n_ctx_max, a      #
#  native decode error) is answered INLINE with finish_reason=error rather     #
#  than a status, on all four paths.                                           #
# --------------------------------------------------------------------------- #

_RUNTIME_REASON = "not enough free VRAM for this prompt (needs 6.2 GiB, 1.1 GiB free)"


def _sse_completion_text(body: str) -> str:
    """Concatenate the `text` deltas of a /v1/completions stream. Its chunks
    carry `text`, not chat's `delta.content`, so the chat helper above cannot
    read them."""
    out = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        for choice in json.loads(line[len("data: "):]).get("choices", []):
            out.append(choice.get("text") or "")
    return "".join(out)


def _completion_finish_reasons(body: str) -> list:
    out = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        for choice in json.loads(line[len("data: "):]).get("choices", []):
            if choice.get("finish_reason"):
                out.append(choice["finish_reason"])
    return out


class TestARuntimeErrorReachesTheClientOnAllFourPaths:
    """All FOUR legs are asserted separately: "these four agree" is the
    property."""

    def test_non_streaming_completions_reports_the_reason_and_marks_it(self):
        r = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                  {"model": "test-model", "prompt": "hi", "stream": False},
                  path="/v1/completions")
        body = r.json()
        # The reason is asserted before the status.
        assert _RUNTIME_REASON in body["choices"][0]["text"], body
        assert body != {"detail": "Internal server error"}
        # The failure is machine-detectable, not text-only.
        assert body["choices"][0]["finish_reason"] == "error", body
        assert r.status_code == 200

    def test_streaming_completions_reports_the_reason_and_marks_it(self):
        r = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                  {"model": "test-model", "prompt": "hi", "stream": True},
                  path="/v1/completions")
        assert _RUNTIME_REASON in _sse_completion_text(r.text)
        assert "error" in _completion_finish_reasons(r.text)

    def test_non_streaming_chat_reports_the_reason_and_marks_it(self):
        r = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                  {"model": "test-model", "messages": _TEXT_MSG, "stream": False})
        body = r.json()
        assert _RUNTIME_REASON in body["choices"][0]["message"]["content"], body
        assert body["choices"][0]["finish_reason"] == "error", body
        assert r.status_code == 200

    def test_streaming_chat_reports_the_reason_and_marks_it(self):
        r = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                  {"model": "test-model", "messages": _TEXT_MSG, "stream": True})
        assert _RUNTIME_REASON in TestBothPathsAgree()._sse_text(r.text)
        assert "error" in TestBothPathsAgree()._finish_reasons(r.text)

    def test_the_two_completions_legs_agree_with_each_other(self):
        """Same route, same failure, one streamed and one not: a caller that
        flips `stream` and nothing else gets the same answer."""
        streamed = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                         {"model": "test-model", "prompt": "hi", "stream": True},
                         path="/v1/completions")
        plain = _post(_mock_engine(stream_exc=RuntimeError(_RUNTIME_REASON)),
                      {"model": "test-model", "prompt": "hi", "stream": False},
                      path="/v1/completions")

        assert _RUNTIME_REASON in _sse_completion_text(streamed.text)
        assert _RUNTIME_REASON in plain.json()["choices"][0]["text"]
        assert "error" in _completion_finish_reasons(streamed.text)
        assert plain.json()["choices"][0]["finish_reason"] == "error"


class TestTheReasonIsRedactedNotMuted:
    """A generation failure's reason crosses a trust boundary into a response
    body, and it is NOT always a tidy sentence: the GGUF loader raises
    ``Failed to load model: <absolute path>`` with a native stderr tail, and an
    auto-reload inside chat_stream surfaces exactly that here.

    Each half gets its own assertion: the machine's directory layout must NOT
    reach the client, and the REASON must still reach it. Redacted, never muted.
    """

    _HOMEISH = r"C:\Users\someaccount\models\thing.gguf"

    def _post_failing(self, path, payload):
        return _post(_mock_engine(stream_exc=RuntimeError(
            f"Failed to load model: {self._HOMEISH} (no backends loaded)")),
            payload, path=path)

    def test_the_account_name_does_not_reach_a_completions_client(self):
        r = self._post_failing("/v1/completions",
                               {"model": "test-model", "prompt": "hi",
                                "stream": False})
        body = r.json()["choices"][0]["text"]
        assert "someaccount" not in body, body
        # ...and the failure reason survives.
        assert "no backends loaded" in body, body
        assert "thing.gguf" in body, body

    def test_the_account_name_does_not_reach_a_chat_client(self):
        r = self._post_failing("/v1/chat/completions",
                               {"model": "test-model", "messages": _TEXT_MSG,
                                "stream": False})
        body = r.json()["choices"][0]["message"]["content"]
        assert "someaccount" not in body, body
        assert "no backends loaded" in body, body

    def test_the_streaming_legs_scrub_too(self):
        # The streaming legs render the failure reason inline, so they scrub it
        # too.
        for path, payload in (
            ("/v1/completions", {"model": "test-model", "prompt": "hi"}),
            ("/v1/chat/completions", {"model": "test-model", "messages": _TEXT_MSG}),
        ):
            r = self._post_failing(path, {**payload, "stream": True})
            assert "someaccount" not in r.text, (path, r.text[:400])
            assert "no backends loaded" in r.text, (path, r.text[:400])


class TestTheRuntimeCatchIsNotWidened:
    """The arm names RuntimeError, not Exception: a genuine defect is still an
    opaque 500, never dressed up as an 'inference error' the user is invited to
    read."""

    def test_an_attribute_error_on_completions_is_still_an_opaque_500(self):
        r = _post_observing_500(
            _mock_engine(stream_exc=AttributeError("'Mock' has no 'foo'")),
            {"model": "test-model", "prompt": "hi", "stream": False},
            path="/v1/completions")
        assert r.json() == {"detail": "Internal server error"}
        assert r.status_code == 500
        assert "Mock" not in r.text

    def test_an_unrelated_value_error_on_completions_is_still_an_opaque_500(self):
        # A ValueError that is not one of the mapped families: not claimed by the
        # backend-error arm above it, nor by the RuntimeError arm below it.
        r = _post_observing_500(
            _mock_engine(stream_exc=ValueError("int() got a str")),
            {"model": "test-model", "prompt": "hi", "stream": False},
            path="/v1/completions")
        assert r.json() == {"detail": "Internal server error"}
        assert r.status_code == 500
        assert "int() got a str" not in r.text

    def test_a_working_completion_is_untouched(self):
        # finish_reason stays stop on the happy path; the arm does not mark
        # every response as an error.
        r = _post(_mock_engine(),
                  {"model": "test-model", "prompt": "hi", "stream": False},
                  path="/v1/completions")
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == "ok"
        assert r.json()["choices"][0]["finish_reason"] == "stop"
