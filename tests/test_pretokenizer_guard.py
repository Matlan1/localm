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
