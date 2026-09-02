# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input that would crash llama.cpp's pre-tokenizer never reaches native code.

A GGUF's ``tokenizer.ggml.pre`` selects one of llama.cpp's hardcoded
pre-tokenizer regex lists. Several of them throw ``std::regex_error`` on a long
run of characters from one character class, which crosses the ctypes boundary as
an uncaught native fault and kills the worker process. The guard scans in Python
first and refuses.

The class/limit table is the load-bearing part, so ``TestCalibration`` pins it:
each limit sits below the length at which that pre-tokenizer was measured to
abort, and each character class is at least as wide as the run its own regex can
carry. Widening a class or lowering a limit is safe; narrowing a class or raising
a limit lets a crashing input through, and that is what these tests catch.

Every "did it reach native code" assertion is made from OUTSIDE the call, with
``assert_not_called`` on a plain mock: raising from a ``side_effect`` would be an
input to the code under test, which catches broadly on some paths and would
swallow it.
"""

import random
import string
from unittest.mock import MagicMock, patch

import pytest

import localm.inference.http_server as hs
from localm.inference import pretokenizer_guard as guard
from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.inference.backends.llamacpp.llama import _Tokenizer

_LLAMA_API = "localm.inference.backends.llamacpp.llama.api"

# Captured at import, before any test can touch it, so a test that REPLACES or
# DELETES it is caught by identity rather than by presence.
from localm.inference.backends.gguf import GgufBackend as _GgufBackend  # noqa: E402
_ORIGINAL_GGUF_LOADED = vars(_GgufBackend).get("loaded")

# Every pre-type string the guard knows, and one it must never touch.
_AFFECTED = sorted(guard.UNSAFE_PRE_TYPES)
_UNAFFECTED = ["llama-bpe", "gpt-2", "qwen2", "deepseek-llm", "jais-2", ""]


def _letters(n, seed=0):
    rnd = random.Random(seed)
    return "".join(rnd.choice(string.ascii_lowercase) for _ in range(n))


def _digits(n, seed=0):
    rnd = random.Random(seed)
    return "".join(rnd.choice(string.digits) for _ in range(n))


def _over_limit_text(pre_type):
    """Text that this pre-type's policy must refuse, built from its own class."""
    policy = guard.UNSAFE_PRE_TYPES[pre_type]
    n = policy.max_run + 1
    if policy.char_class == guard._CLASS_DIGIT:
        return _digits(n)
    if policy.char_class == guard._CLASS_LETTER_SPACE:
        return " ".join(_letters(4, i) for i in range(n))[:n]
    return _letters(n)


def _tokenizer(pre_type):
    """A _Tokenizer with __init__ bypassed, so nothing native is constructed."""
    tok = _Tokenizer.__new__(_Tokenizer)
    tok._vocab = MagicMock()
    tok._ctx = None
    tok._pre_type = pre_type
    return tok


class TestCalibration:
    """The table's numbers, pinned against the lengths measured to abort."""

    # Shortest input at which each pre-tokenizer was measured to throw, on the
    # character class its own regex runs over.
    MEASURED_FIRST_CRASH = {
        "exaone-moe": 149,
        "gpt-4o": 198, "llama4": 198, "kanana2": 198, "talkie": 198,
        "minimax-m2": 198,
        "tekken": 198,
        "granite-embed-multi-97m": 198,
        "superbpe": 297,
        "tiny_aya": 1348, "cohere2moe": 1348, "youtu": 1348,
    }
    # Shortest total length measured to abort even with runs held under the
    # limit, for the pre-tokenizers whose cost also grows with total length.
    MEASURED_TOTAL_CRASH = {"tiny_aya": 224000, "cohere2moe": 224000,
                            "youtu": 224000, "superbpe": 300000}

    def test_every_affected_pre_type_has_a_measured_crash_length(self):
        assert set(self.MEASURED_FIRST_CRASH) == set(guard.UNSAFE_PRE_TYPES)

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_run_limit_sits_below_the_measured_crash_length(self, pre_type):
        limit = guard.UNSAFE_PRE_TYPES[pre_type].max_run
        assert limit < self.MEASURED_FIRST_CRASH[pre_type]

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_run_limit_keeps_a_margin_of_at_least_half(self, pre_type):
        # A limit chosen just under the measured value would have no headroom
        # against a different input shape reaching the same blowup sooner.
        limit = guard.UNSAFE_PRE_TYPES[pre_type].max_run
        assert limit <= self.MEASURED_FIRST_CRASH[pre_type] / 2

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_every_policy_bounds_total_length_too(self, pre_type):
        # Bounding runs alone was measured insufficient for superbpe and the
        # tiny_aya/youtu pair, so every policy carries a total bound.
        assert guard.UNSAFE_PRE_TYPES[pre_type].max_chars is not None

    @pytest.mark.parametrize("pre_type", sorted(MEASURED_TOTAL_CRASH))
    def test_total_limit_sits_below_the_measured_total_crash(self, pre_type):
        cap = guard.UNSAFE_PRE_TYPES[pre_type].max_chars
        assert cap < self.MEASURED_TOTAL_CRASH[pre_type] / 2

    def test_exaone_counts_spaces_as_part_of_the_run(self):
        # Its regex alternates letters with single spaces, so a space does not
        # end a run for it as it does for the others.
        assert guard.UNSAFE_PRE_TYPES["exaone-moe"].char_class == \
            guard._CLASS_LETTER_SPACE
        assert guard.check_text("gpt-4o", " ".join("ab" for _ in range(200))) is None
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text("exaone-moe", " ".join("ab" for _ in range(200)))

    def test_superbpe_runs_over_digits_not_letters(self):
        assert guard.UNSAFE_PRE_TYPES["superbpe"].char_class == guard._CLASS_DIGIT
        assert guard.check_text("superbpe", _letters(300)) is None
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text("superbpe", _digits(300))


class TestTheRunIsAClassNotARepeatedCharacter:
    """A run of DIFFERENT letters aborts at the same length as one repeated
    letter, so a same-character check would not prevent the crash."""

    @pytest.mark.parametrize("pre_type", [p for p in _AFFECTED if p != "superbpe"])
    def test_mixed_letters_are_refused(self, pre_type):
        limit = guard.UNSAFE_PRE_TYPES[pre_type].max_run
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text(pre_type, _letters(limit + 1))

    @pytest.mark.parametrize("pre_type", [p for p in _AFFECTED if p != "superbpe"])
    def test_one_repeated_letter_is_refused_too(self, pre_type):
        limit = guard.UNSAFE_PRE_TYPES[pre_type].max_run
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text(pre_type, "a" * (limit + 1))

    def test_mixed_digits_are_refused(self):
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text("superbpe", _digits(200))


class TestOrdinaryTextIsUntouched:
    """Real prose carries punctuation, which ends a run, so it passes."""

    PROSE = ("The model loads, then it answers; that is fine. "
             "We check it, and it works well! ") * 60

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_realistic_prose_passes(self, pre_type):
        assert guard.check_text(pre_type, self.PROSE) is None

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_a_short_message_passes(self, pre_type):
        assert guard.check_text(pre_type, "Hi, can you help me with this?") is None

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_empty_text_passes(self, pre_type):
        assert guard.check_text(pre_type, "") is None

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_a_run_exactly_at_the_limit_passes(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        at_limit = (_digits(policy.max_run)
                    if policy.char_class == guard._CLASS_DIGIT
                    else _letters(policy.max_run))
        assert guard.check_text(pre_type, f".{at_limit}.") is None

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_long_text_with_short_runs_passes(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        unit = "word word word. "
        body = unit * ((policy.max_chars - 1) // len(unit))
        assert len(body) <= policy.max_chars
        assert guard.check_text(pre_type, body) is None


class TestTotalLengthBound:
    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_text_over_the_total_limit_is_refused(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        body = "ok. " * (policy.max_chars // 4 + 8)
        assert len(body) > policy.max_chars
        with pytest.raises(PretokenizerUnsafeInputError, match="characters at once"):
            guard.check_text(pre_type, body)

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_text_exactly_at_the_total_limit_passes(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        body = ("ok. " * (policy.max_chars // 4 + 4))[:policy.max_chars]
        assert len(body) == policy.max_chars
        assert guard.check_text(pre_type, body) is None


class TestUnaffectedModelsAreNeverScanned:
    """The guard must be invisible to every pre-tokenizer not in the table, so
    normal models keep their tokenization behaviour and cost unchanged."""

    @pytest.mark.parametrize("pre_type", _UNAFFECTED + [None])
    def test_nothing_is_refused(self, pre_type):
        assert guard.check_text(pre_type, "a" * 500000) is None
        assert guard.check_text(pre_type, _letters(500000)) is None
        assert guard.check_text(pre_type, _digits(500000)) is None

    @pytest.mark.parametrize("pre_type", _UNAFFECTED + [None])
    def test_no_policy_is_returned(self, pre_type):
        assert guard.policy_for(pre_type) is None

    def test_an_unknown_pre_type_is_not_guessed_at(self):
        # Matching is exact; a near-miss name must not be treated as affected.
        assert guard.policy_for("llama4-something") is None
        assert guard.policy_for("LLAMA4") is None
        assert guard.check_text("llama4-something", _letters(500)) is None


class TestEncodeRefusesBeforeNativeCode:
    """The whole point: llama_tokenize is never reached for refused input."""

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_llama_tokenize_is_not_called(self, pre_type):
        mock_api = MagicMock()
        with patch(_LLAMA_API, mock_api):
            with pytest.raises(PretokenizerUnsafeInputError):
                _tokenizer(pre_type).encode(_over_limit_text(pre_type))
        mock_api.llama_tokenize.assert_not_called()

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_ordinary_text_still_reaches_llama_tokenize(self, pre_type):
        mock_api = MagicMock()
        mock_api.llama_tokenize.return_value = 3
        with patch(_LLAMA_API, mock_api):
            _tokenizer(pre_type).encode("Hello, world. How are you?")
        mock_api.llama_tokenize.assert_called_once()

    @pytest.mark.parametrize("pre_type", _UNAFFECTED + [None])
    def test_an_unaffected_model_tokenizes_anything(self, pre_type):
        mock_api = MagicMock()
        mock_api.llama_tokenize.return_value = 3
        with patch(_LLAMA_API, mock_api):
            _tokenizer(pre_type).encode(_letters(400000))
        mock_api.llama_tokenize.assert_called_once()


class TestPreTypeIsReadOnceAtLoad:
    def test_the_declared_value_is_read_from_gguf_metadata(self):
        mock_api = MagicMock()
        mock_api.has_model_meta_api.return_value = True
        mock_api.llama_model_meta_val_str.return_value = "llama4"
        assert guard.read_pre_type(object(), mock_api) == "llama4"
        key = mock_api.llama_model_meta_val_str.call_args[0][1]
        assert key == "tokenizer.ggml.pre"

    def test_a_runtime_without_the_metadata_api_reads_none(self):
        mock_api = MagicMock()
        mock_api.has_model_meta_api.return_value = False
        assert guard.read_pre_type(object(), mock_api) is None
        mock_api.llama_model_meta_val_str.assert_not_called()

    def test_a_raising_metadata_call_is_treated_as_unaffected(self):
        # A model whose pre-tokenizer cannot be read must tokenize exactly as it
        # did before this guard existed, not fail to load.
        mock_api = MagicMock()
        mock_api.has_model_meta_api.side_effect = OSError("no runtime")
        assert guard.read_pre_type(object(), mock_api) is None

    def test_encode_does_not_re_read_metadata(self):
        mock_api = MagicMock()
        mock_api.llama_tokenize.return_value = 2
        with patch(_LLAMA_API, mock_api):
            tok = _tokenizer("llama4")
            for _ in range(5):
                tok.encode("Hello there, friend.")
        mock_api.llama_model_meta_val_str.assert_not_called()


class TestTheRefusalIsActionable:
    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_the_message_names_the_pre_tokenizer_and_the_limit(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.check_text(pre_type, _over_limit_text(pre_type))
        msg = str(exc.value)
        assert policy.label in msg
        assert str(policy.max_run) in msg

    def test_it_is_a_value_error_so_the_worker_reports_it_per_request(self):
        assert issubclass(PretokenizerUnsafeInputError, ValueError)

    def test_it_maps_to_a_client_error_status(self):
        assert hs.backend_error_status(PretokenizerUnsafeInputError("x")) == 400


class TestWorkerSurvivesTheRefusal:
    """A refusal must cross the worker IPC boundary as a TYPED error. An
    untagged one is re-raised as RuntimeError, which the parent reads as "the
    worker faulted" and responds to by unloading the model."""

    TAG = "PretokenizerUnsafeInputError"

    def _runner_source(self):
        import inspect

        import localm.inference.backends.llamacpp._runner as runner
        return inspect.getsource(runner)

    def test_the_worker_tags_the_error(self):
        assert f'"error", str(e), "{self.TAG}"' in self._runner_source()

    def test_both_parent_decoders_honour_the_tag(self):
        # One decoder honouring a tag the other does not would make the same
        # envelope mean different things depending on the command in flight.
        src = self._runner_source()
        assert src.count(f'tag == "{self.TAG}"') == 2
        assert src.count('tag == "ContextCapacityExceededError"') == 2

    def test_the_error_is_importable_where_the_runner_expects_it(self):
        from localm.inference.backends.llamacpp._runner import (
            PretokenizerUnsafeInputError as imported,
        )
        assert imported is PretokenizerUnsafeInputError


class TestEmbedderPathIsGuardedToo:
    """granite-embed-multi-97m is an EMBEDDING pre-tokenizer, and the embedder
    reaches llama_tokenize by its own path rather than through _Tokenizer."""

    def test_the_embedding_pre_type_is_in_the_table(self):
        assert "granite-embed-multi-97m" in guard.UNSAFE_PRE_TYPES

    def test_embedder_tokenize_refuses_before_native_code(self):
        from localm.inference.embedder import GGUFEmbedder

        emb = GGUFEmbedder.__new__(GGUFEmbedder)
        emb._api = MagicMock()
        emb._vocab = MagicMock()
        emb._llama_token = MagicMock()
        emb._effective_seq_ctx = 512
        emb._pre_type = "granite-embed-multi-97m"
        with pytest.raises(PretokenizerUnsafeInputError):
            emb._tokenize(_letters(400))
        emb._api.llama_tokenize.assert_not_called()

    def test_an_unaffected_embedding_model_is_untouched(self):
        from localm.inference.embedder import GGUFEmbedder

        emb = GGUFEmbedder.__new__(GGUFEmbedder)
        emb._api = MagicMock()
        emb._api.llama_tokenize.return_value = 2
        emb._vocab = MagicMock()
        emb._llama_token = MagicMock()
        emb._effective_seq_ctx = 512
        emb._pre_type = "bert-bge"
        emb._tokenize(_letters(400))
        emb._api.llama_tokenize.assert_called_once()


class TestScanCost:
    def test_the_scan_is_linear_enough_to_run_on_every_request(self):
        # The scanner is one bounded character-class quantifier; a pattern that
        # backtracked here would reintroduce the cost this guard exists to stop.
        import time

        text = ("word word word. " * 4000)[:guard._TOTAL_LIMIT]
        start = time.perf_counter()
        for _ in range(10):
            guard.check_text("llama4", text)
        assert time.perf_counter() - start < 2.0


class TestVisionPathIsGuardedToo:
    """mtmd_tokenize runs the same pre-tokenizer over the text parts of a
    vision prompt, and llama4 is both an affected pre-tokenizer and a
    multimodal model, so the image path needs the same check."""

    def _vision_llama(self, pre_type, prompt):
        import threading

        from localm.inference.backends.llamacpp.llama import LlamaCpp

        obj = LlamaCpp.__new__(LlamaCpp)
        obj._inference_lock = threading.RLock()
        obj._model_ptr = object()
        obj._mtmd = MagicMock()
        obj._mtmd.marker = "<__image__>"
        obj._tokenizer = _tokenizer(pre_type)
        obj._messages_with_markers = MagicMock(return_value=([{"role": "user",
                                                              "content": prompt}], [object()]))
        obj.mtp_active_this_call = None
        obj.chat_template_fallback_reason = None
        # Enough of the object for the vision path to run PAST the guard site and
        # reach mtmd, so assert_not_called below is load-bearing rather than
        # satisfied by the stub falling over first.
        obj._verbose = False
        obj._gen_lock = threading.RLock()
        obj._stop = threading.Event()
        obj._ctx_ptr = object()
        obj._reset_kv_for_image = MagicMock()
        obj._mtmd.eval_into.return_value = 0
        return obj

    def _drive(self, obj, prompt):
        with patch("localm.inference.backends.llamacpp.llama._apply_model_template",
                   return_value=(prompt, None)):
            return list(obj._generate_image(
                [{"role": "user", "content": prompt}],
                max_new_tokens=1, temperature=0.0, top_k=1, top_p=1.0,
                repeat_penalty=1.0))

    def test_a_long_run_is_refused_before_mtmd_is_touched(self):
        prompt = _letters(400)
        obj = self._vision_llama("llama4", prompt)
        with pytest.raises(PretokenizerUnsafeInputError):
            self._drive(obj, prompt)
        obj._mtmd.eval_into.assert_not_called()

    def test_an_unaffected_vision_model_is_not_blocked_by_the_guard(self):
        # The guard must not be what stops an ordinary vision request; this one
        # gets past it and fails later, on the native work these stubs lack.
        prompt = _letters(400)
        obj = self._vision_llama("qwen2", prompt)
        with pytest.raises(Exception) as exc:
            self._drive(obj, prompt)
        assert not isinstance(exc.value, PretokenizerUnsafeInputError)


class TestAFailedPreTypeReadIsVisible:
    """A read that fails leaves the guard OFF for that model, so it must not be
    passed over in silence."""

    def test_a_raising_read_warns(self, caplog):
        import logging

        mock_api = MagicMock()
        mock_api.has_model_meta_api.side_effect = OSError("no runtime")
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert guard.read_pre_type(object(), mock_api) is None
        assert any("tokenizer.ggml.pre" in r.getMessage() for r in caplog.records), \
            "a failed pre-type read must be reported, not swallowed"

    def test_an_absent_metadata_api_does_not_warn(self):
        # An older runtime without the metadata API is the ordinary case, not a
        # failure, so it must not produce a warning on every load.
        import logging

        mock_api = MagicMock()
        mock_api.has_model_meta_api.return_value = False
        logger = logging.getLogger("localm")
        seen = []
        handler = logging.Handler()
        handler.emit = lambda r: seen.append(r)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            assert guard.read_pre_type(object(), mock_api) is None
        finally:
            logger.removeHandler(handler)
        assert not [r for r in seen if r.levelno >= logging.WARNING]


class TestScannerCacheKeysAreDistinct:
    def test_every_policy_resolves_to_its_own_scanner(self):
        # A shared key would silently apply one policy's scanner to another's
        # text, which reads as the guard working.
        pairs = {(p.char_class, p.max_run) for p in guard.UNSAFE_PRE_TYPES.values()}
        assert len(guard._SCANNERS) == len(pairs)
        for p in guard.UNSAFE_PRE_TYPES.values():
            scanner = guard._SCANNERS[(p.char_class, p.max_run)]
            assert scanner.pattern.startswith(p.char_class)
            assert str(p.max_run + 1) in scanner.pattern


# --------------------------------------------------------------------------- #
#  The refusal has to REACH the caller, not merely fire                        #
# --------------------------------------------------------------------------- #

_REFUSAL = "This model's pre-tokenizer (llama4) crashes on an unbroken run"
_MSG = [{"role": "user", "content": "hi"}]


def _engine(*, count_exc=None):
    """An engine whose token COUNTING can refuse.

    That is what an ordinary chat request hits first: both chat routes count
    prompt tokens before they generate, and counting tokenizes. A fixture that
    injects only at the generation call cannot reach this path at all.
    """
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.supports_grammar = True
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = None
    type(engine).loaded = property(lambda self: True)
    if count_exc is not None:
        engine.count_messages_tokens.side_effect = count_exc
        engine.count_tokens.side_effect = count_exc
    else:
        engine.count_messages_tokens.return_value = 3
        engine.count_tokens.return_value = 2
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    return engine


def _post(engine, payload, path="/v1/chat/completions"):
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    # raise_server_exceptions=False so an unhandled error is OBSERVED as the 500
    # a real client would get, instead of being re-raised into the test.
    with TestClient(create_app(engine), raise_server_exceptions=False) as client:
        return client.post(path, json=payload)


class TestARefusalDuringTokenCountingReachesTheCaller:
    """Unhandled, the refusal falls through to the generic handler and the
    caller gets an opaque 500 rather than the reason."""

    def test_chat_non_streaming_reports_the_reason_with_400(self):
        r = _post(_engine(count_exc=PretokenizerUnsafeInputError(_REFUSAL)),
                  {"model": "test-model", "messages": _MSG, "stream": False})
        assert "unbroken run" in r.json()["detail"]
        assert r.json()["detail"] != "Internal server error"
        assert r.status_code == 400

    def test_chat_streaming_reports_the_reason_with_400(self):
        r = _post(_engine(count_exc=PretokenizerUnsafeInputError(_REFUSAL)),
                  {"model": "test-model", "messages": _MSG, "stream": True})
        assert "unbroken run" in r.json()["detail"]
        assert r.status_code == 400

    def test_completions_reports_the_reason_with_400(self):
        r = _post(_engine(count_exc=PretokenizerUnsafeInputError(_REFUSAL)),
                  {"model": "test-model", "prompt": "hi", "stream": False},
                  path="/v1/completions")
        assert "unbroken run" in r.json()["detail"]
        assert r.status_code == 400

    def test_an_ordinary_request_is_untouched(self):
        r = _post(_engine(), {"model": "test-model", "messages": _MSG,
                              "stream": False})
        assert r.status_code == 200


class TestACountRefusalIsNotAWorkerFault:
    """GgufBackend.count_messages_tokens estimates and latches a
    once-per-process degradation notice when the counting RPC FAILS. A refusal
    is not a failure: estimating answers a request that must be rejected, and
    the notice reports a permanent degradation caused by one request."""

    def test_the_refusal_propagates_instead_of_being_estimated(self, monkeypatch):
        import localm.inference.backends.gguf as gguf

        # A real instance with _loaded set, NOT a __new__ shell with `loaded`
        # patched onto the class: type(be) IS GgufBackend, so assigning or
        # deleting an attribute there mutates the class for the whole process.
        monkeypatch.setattr(gguf, "_count_messages_tokens_rpc_warned", False)
        be = gguf.GgufBackend("does-not-exist.gguf", n_ctx=512)
        be._loaded = True
        be._runner = MagicMock()
        be._runner.count_messages_tokens.side_effect = \
            PretokenizerUnsafeInputError(_REFUSAL)
        assert be.loaded, "test premise: the real `loaded` property must answer True"
        with pytest.raises(PretokenizerUnsafeInputError):
            be.count_messages_tokens(_MSG)
        assert gguf._count_messages_tokens_rpc_warned is False, \
            "a per-request refusal must not spend the worker-fault latch"

    def test_this_module_leaves_gguf_backend_unmutated(self):
        # The check that would have caught the class-level mutation this test
        # class used to do, which deleted GgufBackend.loaded for every test that
        # ran after it in the same process.
        import localm.inference.backends.gguf as gguf

        # Identity, not presence: ASSIGNING `loaded` on the class replaces the
        # real property process-wide, and DELETING it drops resolution to
        # BaseBackend's abstract stub, which returns None. Presence alone catches
        # only the second.
        assert vars(gguf.GgufBackend).get("loaded") is _ORIGINAL_GGUF_LOADED, \
            "GgufBackend.loaded is not the property the class was defined with"


class TestTheEmbedderWorkerCarriesTheTypedRefusal:
    """GGUFEmbedder runs only inside the embedder child, so an untagged refusal
    becomes RuntimeError in the parent - which the embeddings route reports as a
    temporary 503 and which RAG reads as the embedder being broken."""

    def _runner_with(self, result):
        import multiprocessing as mp

        from localm.inference._embedder_runner import EmbedderRunner

        class _AliveProc:
            def is_alive(self):
                return True

        r = EmbedderRunner.__new__(EmbedderRunner)
        q = mp.get_context("spawn").Queue()
        q.put(result)
        r._resp_q = q
        r._proc = _AliveProc()
        r.shutdown = MagicMock()
        return r

    def test_the_worker_tags_the_refusal(self):
        import inspect

        import localm.inference._embedder_runner as runner
        assert '"error", str(e), "PretokenizerUnsafeInputError"' in \
            inspect.getsource(runner)

    def test_the_parent_re_raises_it_typed_and_keeps_the_worker(self):
        r = self._runner_with(("error", _REFUSAL, "PretokenizerUnsafeInputError"))
        with pytest.raises(PretokenizerUnsafeInputError):
            r._wait(5.0, "embed", shutdown_on_error=True)
        r.shutdown.assert_not_called()

    def test_an_untagged_error_still_reads_as_a_fault(self):
        r = self._runner_with(("error", "the worker died"))
        with pytest.raises(RuntimeError) as exc:
            r._wait(5.0, "embed", shutdown_on_error=True)
        assert not isinstance(exc.value, PretokenizerUnsafeInputError)
        r.shutdown.assert_called_once()


class TestTheMtmdAbiProbeIsNotItselfFatal:
    """mtmd's layout probe sends a fixed string through mtmd_tokenize at vision
    load, before any caller text exists for the guard to check. A long
    single-class run there aborts the process on exactly the models this guard
    is about, and the probe's own except-Exception cannot catch a native abort."""

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_the_probe_strings_pass_the_guard(self, pre_type):
        from localm.inference.backends.llamacpp import mtmd

        for raw in (mtmd._PROBE_CONTROL, mtmd._PROBE_EMBEDDED_NUL):
            guard.check_text(pre_type, raw.decode("utf-8", "replace"))

    def test_the_probe_keeps_the_properties_it_measures(self):
        from localm.inference.backends.llamacpp import mtmd

        assert len(mtmd._PROBE_CONTROL) == len(mtmd._PROBE_EMBEDDED_NUL) == 256
        assert mtmd._PROBE_EMBEDDED_NUL[:1] == b"\x00"
        assert b"\x00" not in mtmd._PROBE_CONTROL
        assert mtmd._PROBE_CONTROL.strip()


class TestGeneratedTextDoesNotTruncateTheStream:
    """Usage is counted on text ALREADY delivered. A model can emit a run the
    pre-tokenizer refuses just as a caller can send one, and raising there ends
    the response with no terminal chunk, no usage and no [DONE]."""

    def test_a_refused_count_falls_back_instead_of_raising(self):
        import asyncio

        from localm.inference.http_server import _count_streamed_tokens

        engine = MagicMock()
        engine.count_tokens.side_effect = PretokenizerUnsafeInputError(_REFUSAL)
        assert asyncio.run(_count_streamed_tokens(engine, "x" * 400)) == 100

    def test_an_ordinary_count_is_unchanged(self):
        import asyncio

        from localm.inference.http_server import _count_streamed_tokens

        engine = MagicMock()
        engine.count_tokens.return_value = 42
        assert asyncio.run(_count_streamed_tokens(engine, "hello")) == 42


class TestTheRefusalQuotesNoCallerText:
    """The refusal becomes an HTTP detail, and _log_http_exception writes details
    to the debug log gated on debug_enabled() rather than debug_content_enabled(),
    because a detail is server-authored operational text. Quoting any of the
    caller's text here would put chat content in the debug log, privacy mode
    included."""

    SECRET = "SECRETPROJECTCODENAMEZEPHYR"

    def _refusal_for(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        if policy.char_class == guard._CLASS_DIGIT:
            run = "9" * (policy.max_run + 40)
        else:
            run = (self.SECRET * 10)[:policy.max_run + 40]
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.check_text(pre_type, f"please summarise. {run}. thanks")
        return str(exc.value), run

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_no_substring_of_the_offending_run_is_echoed(self, pre_type):
        msg, run = self._refusal_for(pre_type)
        # Any 8-character window of the caller's run would be chat content.
        windows = {run[i:i + 8] for i in range(0, len(run) - 8)}
        leaked = [w for w in windows if w in msg]
        assert not leaked, f"the refusal echoed caller text: {leaked[:3]}"

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_no_surrounding_prompt_text_is_echoed(self, pre_type):
        msg, _ = self._refusal_for(pre_type)
        for word in ("please", "summarise", "thanks", self.SECRET):
            assert word not in msg

    @pytest.mark.parametrize("pre_type", _AFFECTED)
    def test_it_still_says_enough_to_act_on(self, pre_type):
        # Withholding the text must not leave the message useless: the run's
        # length, the allowed length and the class are all caller-actionable.
        import re as _re

        msg, run = self._refusal_for(pre_type)
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        assert str(policy.max_run) in msg
        found = _re.search(r"run of (\d+)", msg)
        assert found, msg
        # At least what was built: for exaone-moe a SPACE is part of the run
        # class, so the space before the run is legitimately counted into it.
        assert int(found.group(1)) >= len(run)
        assert ("digits" in msg) or ("letters" in msg)

    def test_the_total_length_refusal_quotes_nothing_either(self):
        policy = guard.UNSAFE_PRE_TYPES["llama4"]
        body = (self.SECRET + ". ") * (policy.max_chars // 10)
        assert len(body) > policy.max_chars
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.check_text("llama4", body)
        assert self.SECRET not in str(exc.value)


class TestTheStreamStillTerminatesWhenUsageIsRefused:
    """The helper test above is blind to the property that actually broke: that
    the SSE body still carries its terminal chunk, its usage block and [DONE].
    The helper was extracted to fix a bug that lives in the stream, so the
    regression guard belongs at the stream layer."""

    def _sse(self, generated):
        engine = _engine()
        engine.chat_stream.side_effect = lambda messages, **kw: iter([generated])
        # Counting the PROMPT must succeed (that gate is a different contract);
        # only counting the GENERATED text refuses, which is the usage count.
        def _count(text):
            if text == generated:
                raise PretokenizerUnsafeInputError(_REFUSAL)
            return 2
        engine.count_tokens.side_effect = _count
        engine.count_messages_tokens.return_value = 3
        return _post(engine, {"model": "test-model", "messages": _MSG,
                              "stream": True})

    def test_the_body_is_complete(self):
        r = self._sse("a" * 400)
        assert r.status_code == 200
        body = r.text
        assert "[DONE]" in body, "the stream ended without its [DONE] sentinel"
        assert "usage" in body, "the stream ended without its usage block"

    def test_the_estimate_is_reported_rather_than_nothing(self):
        import json

        body = self._sse("a" * 400).text
        usage = None
        for line in body.splitlines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            obj = json.loads(line[len("data: "):])
            if obj.get("usage"):
                usage = obj["usage"]
        assert usage is not None, f"no usage block in the stream: {body[-300:]!r}"
        assert usage["completion_tokens"] == 100, usage

    def test_an_ordinary_stream_is_unchanged(self):
        engine = _engine()
        r = _post(engine, {"model": "test-model", "messages": _MSG,
                           "stream": True})
        assert r.status_code == 200
        assert "[DONE]" in r.text


class TestTheGatesRefuseAndTheReportsEstimate:
    """The partition itself: a count that GATES the work refuses, a count that
    REPORTS ON FINISHED WORK estimates. Completeness over the call sites is a
    different claim and is checked by enumeration in
    TestEveryCountingCallIsClassified, not here."""

    def test_the_post_compaction_recount_refuses_rather_than_500(self, monkeypatch):
        """The re-count after compaction reads a model-WRITTEN summary, which can
        carry a run of its own. It gates the generation, so it must answer 400 -
        and it sits in a different block from the first count, so wrapping that
        one did not cover it."""
        import localm.inference.compact as compact_mod

        engine = _engine()
        # The first count clears the wrapped call and lands inside the compaction
        # window; the RE-count, on the compacted messages, refuses.
        counts = iter([3000])

        def _count_messages(_messages):
            try:
                return next(counts)
            except StopIteration:
                raise PretokenizerUnsafeInputError(_REFUSAL)

        engine.count_messages_tokens.side_effect = _count_messages
        # capacity - prompt_tokens (4096 - 3000) < buffer (2048), so it compacts.
        engine.context_capacity.return_value = 4096
        monkeypatch.setattr(compact_mod, "compact_messages",
                            lambda ms, gen: (list(ms), True))

        thread = [{"role": "user", "content": "a"},
                  {"role": "assistant", "content": "b"},
                  {"role": "user", "content": "c"},
                  {"role": "assistant", "content": "d"},
                  {"role": "user", "content": "e"}]
        r = _post(engine, {"model": "test-model", "messages": thread,
                           "stream": False})
        assert r.json()["detail"] != "Internal server error"
        assert "unbroken run" in r.json()["detail"]
        assert r.status_code == 400

    def test_only_this_refusal_is_estimated(self):
        from localm.inference.pretokenizer_guard import count_tokens_or_estimate

        def _boom(_text):
            raise RuntimeError("the worker died")
        with pytest.raises(RuntimeError):
            count_tokens_or_estimate(_boom, "hi", "the generated text")


class TestTheEmbeddingsUsageCountIsSafeAndUnchanged:
    """The /v1/embeddings usage total is counted with the CHAT engine's
    tokenizer, AFTER the vectors exist, so a refusal there would fail a request
    that already succeeded.

    DEFENSIVE RATHER THAN REACHABLE TODAY, matching the comment at the site: the
    only backend whose ``count_tokens`` can refuse is ``GgufBackend``, whose
    ``embed()`` raises ``NotImplementedError`` and is refused earlier in the
    route. These tests drive the arithmetic with a mock engine, which is what
    lets them exercise a path the real backends cannot currently reach.

    Both halves matter: unchanged when nothing refuses, and degrading rather
    than failing when something does."""

    def _client(self, count_side_effect):
        from fastapi.testclient import TestClient

        from localm.inference.http_server import create_app
        engine = _engine()
        engine.embed.return_value = [[0.1, 0.2, 0.3]]
        engine.count_tokens.side_effect = count_side_effect
        return TestClient(create_app(engine), raise_server_exceptions=False)

    def test_the_total_is_the_real_sum_when_nothing_refuses(self):
        # len() as the token count, so the expected total is checkable.
        with self._client(lambda t: len(t)) as c:
            r = c.post("/v1/embeddings",
                       json={"model": "test-model", "input": ["hello", "worldly"]})
        assert r.status_code == 200
        assert r.json()["usage"]["total_tokens"] == len("hello") + len("worldly")

    def test_a_refused_count_does_not_fail_the_completed_embedding(self):
        def _refuse(_t):
            raise PretokenizerUnsafeInputError(_REFUSAL)

        with self._client(_refuse) as c:
            r = c.post("/v1/embeddings",
                       json={"model": "test-model", "input": ["x" * 400]})
        assert r.json() != {"detail": "Internal server error"}
        assert r.status_code == 200, r.json()
        # The vectors are still returned; only the count degraded.
        assert r.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert r.json()["usage"]["total_tokens"] == 100


class _CountingCallScanner:
    """Finds counting calls and classifies each one's exception handling.

    Parses with ``ast`` rather than scanning indentation. Successive reviews
    defeated the indentation version five ways - a window that ran to end of
    file, a trailing comment after ``try:``, a comment at column 0 inside a try
    body, a multi-line ``except`` tuple, and a multi-line ``def`` signature whose
    closing paren sits at column 0. None of those are expressible against a
    parse tree.
    """

    import ast as _ast

    ATTRS = {"count_tokens", "count_messages_tokens"}
    NAMED = "PretokenizerUnsafeInputError"
    # Handlers that would catch the refusal. It is a ValueError (pinned by
    # TestTheRefusalIsActionable), so a ValueError or broader clause swallows it.
    CATCHES = {NAMED, "ValueError", "Exception", "BaseException"}
    ROUTER = "count_tokens_or_estimate"
    DEAD_TEST = "prompt_tokens is None"

    def __init__(self, source: str):
        self.src = source
        self.tree = self._ast.parse(source)
        self.parent = {}
        for node in self._ast.walk(self.tree):
            for child in self._ast.iter_child_nodes(node):
                self.parent[child] = node

    def _ancestors(self, node):
        while node in self.parent:
            node = self.parent[node]
            yield node

    def calls(self):
        """(lineno, enclosing function, node) for each counting call."""
        out = []
        for node in self._ast.walk(self.tree):
            hit = None
            if isinstance(node, self._ast.Attribute) and node.attr in self.ATTRS:
                hit = node
            elif (isinstance(node, self._ast.Call)
                  and isinstance(node.func, self._ast.Name)
                  and node.func.id == "getattr"
                  and len(node.args) >= 2
                  and isinstance(node.args[1], self._ast.Constant)
                  and node.args[1].value in self.ATTRS):
                hit = node
            if hit is not None:
                out.append((hit.lineno, self._enclosing_def(hit), hit))
        return sorted(out, key=lambda t: t[0])

    def _enclosing_def(self, node):
        for anc in self._ancestors(node):
            if isinstance(anc, (self._ast.FunctionDef, self._ast.AsyncFunctionDef)):
                return anc.name
        return "<module>"

    def is_routed(self, node):
        """True when the reference is handed to count_tokens_or_estimate, either
        called directly or passed to run_in_executor alongside it."""
        for anc in self._ancestors(node):
            if not isinstance(anc, self._ast.Call):
                continue
            names = []
            if isinstance(anc.func, self._ast.Name):
                names.append(anc.func.id)
            elif isinstance(anc.func, self._ast.Attribute):
                names.append(anc.func.attr)
            names += [a.id for a in anc.args if isinstance(a, self._ast.Name)]
            if self.ROUTER in names:
                return True
        return False

    def _handler_names(self, handler):
        t = handler.type
        if t is None:
            return {"BaseException"}                      # bare except
        parts = t.elts if isinstance(t, self._ast.Tuple) else [t]
        names = set()
        for p in parts:
            if isinstance(p, self._ast.Name):
                names.add(p.id)
            elif isinstance(p, self._ast.Attribute):
                names.add(p.attr)
        return names

    def _in_body_of(self, node, try_node):
        """True when *node* is in the try's BODY rather than in a handler."""
        for stmt in try_node.body:
            for inner in self._ast.walk(stmt):
                if inner is node:
                    return True
        return False

    def guard_verdict(self, node):
        """``'named'``, ``'swallowed'`` or ``'unguarded'``.

        Walks enclosing ``try`` statements innermost-first and returns the
        verdict of the FIRST one that would catch the refusal, because Python
        stops there. Within that try the handlers are read IN ORDER and the
        first clause that would match decides, so a named handler listed after
        a broad one is correctly reported as the dead code it is.
        """
        for anc in self._ancestors(node):
            if isinstance(anc, (self._ast.FunctionDef, self._ast.AsyncFunctionDef)):
                return "unguarded"                        # left the function
            if not isinstance(anc, self._ast.Try):
                continue
            if not self._in_body_of(node, anc):
                continue                                  # sits in a handler, not the body
            for handler in anc.handlers:
                names = self._handler_names(handler)
                if not (names & self.CATCHES):
                    continue                              # this clause cannot catch it
                return "named" if self.NAMED in names else "swallowed"
            # No clause on this try catches it: it propagates further out.
        return "unguarded"

    def in_dead_branch(self, node):
        for anc in self._ancestors(node):
            if isinstance(anc, self._ast.If):
                if self.DEAD_TEST in self._ast.unparse(anc.test):
                    return True
        return False


class TestTheClassifierCanRejectAndIsNotFooled:
    """The classifier is itself a claim, so it gets the same treatment as the
    code it guards.

    Every case below is a shape a previous version got WRONG, most of them in
    the dangerous direction - answering "guarded" for code that would let the
    refusal through. A classifier that answers guarded for everything passes
    every other assertion in this file.
    """

    def _verdict(self, body):
        import textwrap
        scanner = _CountingCallScanner(textwrap.dedent(body))
        calls = scanner.calls()
        assert len(calls) == 1, f"fixture should hold one call, got {len(calls)}"
        return scanner.guard_verdict(calls[0][2])

    def test_a_bare_call_is_unguarded(self):
        assert self._verdict("""
            def f():
                n = engine.count_tokens(t)
        """) == "unguarded"

    def test_a_named_handler_guards(self):
        assert self._verdict("""
            def f():
                try:
                    n = engine.count_tokens(t)
                except PretokenizerUnsafeInputError:
                    raise
        """) == "named"

    def test_an_inner_broad_handler_swallows_it_before_an_outer_named_one(self):
        # The shape of the original GgufBackend.count_messages_tokens bug: a
        # broad except swallowed the refusal and estimated instead.
        assert self._verdict("""
            def f():
                try:
                    try:
                        n = engine.count_tokens(t)
                    except Exception:
                        n = fallback()
                except PretokenizerUnsafeInputError:
                    raise
        """) == "swallowed"

    def test_an_inner_value_error_handler_swallows_it(self):
        # PretokenizerUnsafeInputError IS a ValueError, so this catches it.
        assert self._verdict("""
            def f():
                try:
                    try:
                        n = engine.count_tokens(t)
                    except ValueError:
                        n = 0
                except PretokenizerUnsafeInputError:
                    raise
        """) == "swallowed"

    def test_a_named_handler_after_a_broad_one_is_dead_code(self):
        # Python takes the FIRST matching clause, so the named one never runs.
        assert self._verdict("""
            def f():
                try:
                    n = engine.count_tokens(t)
                except Exception:
                    n = 0
                except PretokenizerUnsafeInputError:
                    raise
        """) == "swallowed"

    def test_a_bare_except_swallows_it(self):
        assert self._verdict("""
            def f():
                try:
                    n = engine.count_tokens(t)
                except:
                    n = 0
        """) == "swallowed"

    def test_a_handler_that_cannot_catch_it_is_transparent(self):
        # KeyError does not catch it, so it propagates to the named handler.
        assert self._verdict("""
            def f():
                try:
                    try:
                        n = engine.count_tokens(t)
                    except KeyError:
                        n = 0
                except PretokenizerUnsafeInputError:
                    raise
        """) == "named"

    def test_a_multi_line_except_tuple_is_read(self):
        assert self._verdict("""
            def f():
                try:
                    n = engine.count_tokens(t)
                except (KeyError,
                        PretokenizerUnsafeInputError):
                    raise
        """) == "named"

    def test_a_trailing_comment_after_try_does_not_defeat_it(self):
        assert self._verdict("""
            def f():
                try:  # count the prompt
                    n = engine.count_tokens(t)
                except PretokenizerUnsafeInputError:
                    raise
        """) == "named"

    def test_a_column_zero_comment_inside_the_body_does_not_defeat_it(self):
        assert self._verdict("""
            def f():
                try:
            # a comment at column 0
                    n = engine.count_tokens(t)
                except PretokenizerUnsafeInputError:
                    raise
        """) == "named"

    def test_a_try_in_an_outer_function_does_not_guard_a_nested_def(self):
        assert self._verdict("""
            def outer():
                try:
                    def inner():
                        n = engine.count_tokens(t)
                except PretokenizerUnsafeInputError:
                    raise
        """) == "unguarded"

    def test_a_handler_body_is_not_the_try_body(self):
        assert self._verdict("""
            def f():
                try:
                    something()
                except PretokenizerUnsafeInputError:
                    n = engine.count_tokens(t)
        """) == "unguarded"

    def test_a_multi_line_def_signature_does_not_end_the_walk(self):
        # The indentation version reported "?" here, because the signature's
        # closing paren sits at column 0.
        import textwrap
        scanner = _CountingCallScanner(textwrap.dedent("""
            async def handler(
                engine,
                text,
            ) -> str:
                n = engine.count_tokens(text)
        """))
        _, func, node = scanner.calls()[0]
        assert func == "handler"
        assert scanner.guard_verdict(node) == "unguarded"

    def test_a_reference_split_across_lines_is_seen(self):
        import textwrap
        scanner = _CountingCallScanner(textwrap.dedent("""
            def f():
                n = run_in_executor(None, engine
                                    .count_tokens, text)
        """))
        assert len(scanner.calls()) == 1

    def test_routing_is_recognised_in_both_shapes(self):
        import textwrap
        direct = _CountingCallScanner(textwrap.dedent("""
            def f():
                n = count_tokens_or_estimate(engine.count_tokens, t, "x")
        """))
        assert direct.is_routed(direct.calls()[0][2])
        executor = _CountingCallScanner(textwrap.dedent("""
            def f():
                n = await loop.run_in_executor(
                    None, count_tokens_or_estimate, engine.count_tokens, text, "x")
        """))
        assert executor.is_routed(executor.calls()[0][2])
        raw = _CountingCallScanner(textwrap.dedent("""
            def f():
                n = await loop.run_in_executor(None, engine.count_tokens, text)
        """))
        assert not raw.is_routed(raw.calls()[0][2])

    def test_a_name_that_merely_contains_the_words_is_not_a_call(self):
        import textwrap
        scanner = _CountingCallScanner(textwrap.dedent("""
            def f():
                self.count_tokens_total = 0
                x = count_tokens(t)
        """))
        assert scanner.calls() == []


class TestEveryCountingCallIsClassified:
    """ENUMERATES the counting calls in the server files instead of sampling.

    Each must be routed through ``count_tokens_or_estimate`` (it REPORTS ON
    FINISHED WORK, so a refusal degrades to an estimate), guarded by a ``try``
    whose first matching handler names the refusal (it GATES the work, so a
    refusal answers 400), or inside an ``if prompt_tokens is None:`` branch that
    ``test_the_dead_branches_really_are_dead`` pins as unreachable.

    A call that is none of those reaches the generic handler as an opaque 500,
    which is how the non-streaming chat usage count shipped broken.
    """

    EXPECTED_CALLS = 11

    @staticmethod
    def _server_files():
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        files = ["localm/inference/http_server.py"]
        files += sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in (root / "localm/inference/routes").glob("*.py")
            if p.name != "__init__.py")
        return files

    def _scan(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in self._server_files():
            yield rel, _CountingCallScanner((root / rel).read_text(encoding="utf-8"))

    def test_the_enumeration_finds_exactly_what_is_there(self):
        found = [(rel, ln, fn) for rel, sc in self._scan() for ln, fn, _ in sc.calls()]
        assert len(found) == self.EXPECTED_CALLS, (
            f"expected {self.EXPECTED_CALLS} counting calls, found {len(found)}. "
            "This is a tripwire, not the gate: classify the new call against the "
            "test below, then update the number.\n  "
            + "\n  ".join(f"{r}:{n} in {f}()" for r, n, f in found))

    def test_every_counting_call_is_routed_guarded_or_dead(self):
        unclassified = []
        for rel, sc in self._scan():
            for lineno, func, node in sc.calls():
                if sc.is_routed(node):
                    continue
                if sc.guard_verdict(node) == "named":
                    continue
                if sc.in_dead_branch(node):
                    continue
                unclassified.append(
                    f"{rel}:{lineno} in {func}(): {sc.guard_verdict(node)}")
        assert not unclassified, (
            "these counting calls would reach the generic handler as a 500 on a "
            "pre-tokenizer refusal:\n  " + "\n  ".join(unclassified))

    def test_the_dead_branches_really_are_dead(self):
        """The `if prompt_tokens is None:` fallbacks are unreachable only because
        every caller passes a computed prompt_tokens the route already guarded.
        Asserted rather than assumed, so a caller that stops passing it fails
        here instead of quietly reviving an unguarded count."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (root / "localm/inference/routes/chat.py").read_text(encoding="utf-8"))
        handlers = {"_stream_sse", "_complete", "_stream_sse_completion"}
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in handlers:
                continue
            found += 1
            assert "prompt_tokens" in {k.arg for k in node.keywords}, (
                f"{name}() at line {node.lineno} is called without passing "
                f"prompt_tokens, which revives an unguarded counting call in "
                f"http_server.py")
        assert found >= 3, f"expected the three handler call sites, found {found}"


@pytest.fixture(autouse=True)
def _gguf_backend_class_is_not_mutated():
    """Runs after EVERY test in this file, not just at one collected moment.

    A test that assigns or deletes an attribute on the real GgufBackend class
    changes it for the rest of the process; deleting `loaded` in particular drops
    resolution to BaseBackend's abstract stub, which returns None and silently
    turns every later token count into the chars/4 heuristic. That escapes this
    file, so it is checked per test rather than once.
    """
    yield
    assert vars(_GgufBackend).get("loaded") is _ORIGINAL_GGUF_LOADED, \
        "a test in this file mutated GgufBackend.loaded, which leaks into every " \
        "test that runs after it in this process"


class TestNonStreamingChatSurvivesARefusedUsageCount:
    """`stream: false` is the default shape for most API clients, and its usage
    count sits in http_server.py rather than routes/chat.py - which is why the
    first sweep, scoped to the routes file, missed it. A refusal there discarded
    a generation that had fully succeeded."""

    GENERATED = "a" * 400

    def _client(self):
        from fastapi.testclient import TestClient

        from localm.inference.http_server import create_app
        engine = _engine()
        engine.chat_stream.side_effect = lambda m, **kw: iter([self.GENERATED])

        # Only counting the GENERATED text refuses; counting the PROMPT must
        # succeed, or this would test the gating path instead of the usage path.
        def _count(text):
            if text == self.GENERATED:
                raise PretokenizerUnsafeInputError(_REFUSAL)
            return 2
        engine.count_tokens.side_effect = _count
        engine.count_messages_tokens.return_value = 3
        return TestClient(create_app(engine), raise_server_exceptions=False)

    def _post(self):
        with self._client() as c:
            return c.post("/v1/chat/completions",
                          json={"model": "test-model", "messages": _MSG,
                                "stream": False})

    def test_the_generation_is_returned_rather_than_discarded(self):
        r = self._post()
        assert r.json() != {"detail": "Internal server error"}
        assert r.status_code == 200, r.json()
        # The answer is the property; the token count is a report about it.
        content = r.json()["choices"][0]["message"]["content"]
        assert content, "a successful generation was thrown away over its usage count"

    def test_the_count_degrades_to_an_estimate(self):
        usage = self._post().json()["usage"]
        assert usage["completion_tokens"] == len(self.GENERATED) // 4
