# SPDX-License-Identifier: AGPL-3.0-or-later
"""A text longer than a pre-tokenizer takes in one piece is tokenized in chunks.

llama.cpp's pre-tokenizer regexes abort the process on some long inputs, so the
guard cuts a text at offsets where the pre-tokenizer ends a match whatever
follows, tokenizes the chunks one by one and joins the tokens. The join must
equal the tokens of the whole text, and no chunk may be one the pre-tokenizer
must not see.

``TestChunkedTokensEqualWholeTokens`` runs the real native tokenizer, with a
hand-built vocabulary-only GGUF per pre-type, and compares whole against chunked
tokenization. The rest is the planning logic and how ``_Tokenizer`` drives it.
"""

import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference import pretokenizer_guard as guard
from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.inference.backends.llamacpp.llama import _Tokenizer

_LLAMA_API = "localm.inference.backends.llamacpp.llama.api"
_AFFECTED = sorted(guard.UNSAFE_PRE_TYPES)
_CHUNKABLE = [p for p in _AFFECTED if not guard.UNSAFE_PRE_TYPES[p].unusable]
_CAP = guard.UNSAFE_PRE_TYPES["gpt-4o"].max_chars


@pytest.fixture(autouse=True)
def _windows_limits(monkeypatch):
    """The limits this file pins are the table's (measured on Windows); the
    Linux limits have their own tests."""
    monkeypatch.setattr(guard, "_LINUX", False)


def _prose(n: int) -> str:
    return ("The quick brown fox 12 jumps over the lazy dog, then rests.\n" * (n // 59 + 1))[:n]


def _fake_tokenizer(pre_type):
    """A _Tokenizer with __init__ bypassed whose native tokenize writes the
    UTF-8 bytes of its input as tokens, so a join is checkable by eye."""
    tok = _Tokenizer.__new__(_Tokenizer)
    tok._vocab = MagicMock()
    tok._ctx = None
    tok._pre_type = pre_type
    tok._chunkable = True
    return tok


def _fake_llama_tokenize(calls):
    def fake(vocab, raw, n, buf, n_max, add_special=False, parse_special=False):
        calls.append((raw.decode("utf-8"), add_special, parse_special))
        ids = ([1] if add_special else []) + list(raw)
        for i, t in enumerate(ids):
            buf[i] = t
        return len(ids)
    return fake


class TestCutPoints:
    def test_a_space_after_visible_text_is_a_cut(self):
        assert guard.chunk_cuts("ab cd") == [2]

    def test_no_cut_after_whitespace_or_at_the_end(self):
        assert guard.chunk_cuts("ab  cd") == [2]
        assert guard.chunk_cuts("ab ") == []

    def test_a_line_break_run_is_cut_only_after_its_last_break(self):
        assert guard.chunk_cuts("a\n\nb") == [3]
        assert guard.chunk_cuts("a\r\nb") == [3]
        assert guard.chunk_cuts("a\n  \nb") == [5]

    def test_a_slash_after_a_line_break_is_never_split_from_it(self):
        assert guard.chunk_cuts("a\n/b") == []
        assert guard.chunk_cuts("a\n  /b") == []

    def test_digit_cuts_are_the_edges_of_a_number(self):
        assert guard.chunk_cuts("a 12 b", style="digits") == [2, 4]
        assert guard.chunk_cuts("a b", style="digits") == []

    def test_newline_cuts_are_after_a_line_break_only(self):
        assert guard.chunk_cuts("a b\nc d", style="newlines") == [4]
        assert guard.chunk_cuts("a b\n", style="newlines") == []

    def test_every_policy_names_a_known_cut_style(self):
        assert {p.cuts for p in guard.UNSAFE_PRE_TYPES.values()} == {
            "words", "digits", "newlines"}
        assert guard.UNSAFE_PRE_TYPES["superbpe"].cuts == "digits"
        assert guard.UNSAFE_PRE_TYPES["deepseek-llm"].cuts == "newlines"

    def test_digit_cuts_use_only_ascii_digits(self):
        assert guard.chunk_cuts("a \u0663 b", style="digits") == []

    def test_text_without_spaces_or_breaks_has_no_cut(self):
        assert guard.chunk_cuts("x" * 500) == []

    def test_cuts_never_touch_a_special_token(self):
        text = "aa <|x|> bb cc"
        assert guard.chunk_cuts(text) == [2, 8, 11]
        assert guard.chunk_cuts(text, ("<|x|>",)) == [11]

    def test_a_cut_is_not_made_inside_a_special_token_or_overlapping_ones(self):
        text = "aa <|x y|> bb"
        assert guard.chunk_cuts(text) == [2, 6, 10]
        assert guard.chunk_cuts(text, ("<|x y|>",)) == []
        assert guard.chunk_cuts(text, ("<|x", "y|>")) == []

    def test_a_cut_is_not_made_in_whitespace_a_special_token_may_strip(self):
        text = "a<|x|>\n  abc d"
        assert guard.chunk_cuts(text) == [7, 12]
        assert guard.chunk_cuts(text, ("<|x|>",)) == [12]
        text = "abc  \n <|x|>d e"
        assert guard.chunk_cuts(text) == [3, 6, 13]
        assert guard.chunk_cuts(text, ("<|x|>",)) == [13]


def _cuts_by_walking(text, separators, style):
    """The cut rule of ``guard.chunk_cuts`` written out one cut at a time: a cut
    is dropped when it lies inside or at the edge of a separator, or when the
    run of blank characters around it reaches a separator's edge."""
    cuts = guard.chunk_cuts(text, (), style)
    spans = []
    for sep in separators:
        pos = text.find(sep)
        while pos != -1:
            spans.append((pos, pos + len(sep)))
            pos = text.find(sep, pos + 1)
    if not spans:
        return cuts
    blank = guard.regex.compile(r"[\s\p{Z}\p{C}]")
    edges = {edge for span in spans for edge in span}
    out = []
    for c in cuts:
        if any(s <= c <= e for s, e in spans):
            continue
        lo = hi = c
        while lo > 0 and blank.match(text, lo - 1):
            lo -= 1
        while hi < len(text) and blank.match(text, hi):
            hi += 1
        if not any(lo <= edge <= hi for edge in edges):
            out.append(c)
    return out


def _calls_in_chunk_cuts(text, separators, style):
    """How many Python and C calls ``guard.chunk_cuts`` makes on *text*."""
    calls = [0]

    def count(frame, event, arg):
        if event in ("call", "c_call"):
            calls[0] += 1

    previous = sys.getprofile()
    sys.setprofile(count)
    try:
        guard.chunk_cuts(text, separators, style)
    finally:
        sys.setprofile(previous)
    return calls[0]


class TestCutPointsAgainstTheRule:
    ALPHABET = ["a", "b ", " ", "  ", "\n", "\n\n", "\r\n", "\t", "/", "1", "22", "\u00a0",
                "\u200b", "\u3000", "<s>", "<s>", "<|x y|>", "\n<s>\n", "\n "]

    @pytest.mark.parametrize("style", ["words", "digits", "newlines"])
    def test_the_cuts_equal_the_rule_applied_one_cut_at_a_time(self, style):
        rnd = random.Random(29)
        separators = ("<s>", "<|x y|>", "\n\n")
        compared_with_a_dropped_cut = 0
        for _ in range(3000):
            text = "".join(rnd.choice(self.ALPHABET) for _ in range(rnd.randint(0, 24)))
            want = _cuts_by_walking(text, separators, style)
            assert guard.chunk_cuts(text, separators, style) == want, repr(text)
            compared_with_a_dropped_cut += want != guard.chunk_cuts(text, (), style)
        assert compared_with_a_dropped_cut > 100

    @pytest.mark.parametrize("style", ["words", "digits", "newlines"])
    def test_the_work_grows_in_line_with_a_blank_run_beside_a_separator(self, style):
        small = _calls_in_chunk_cuts("\n" * 600 + "<sep>", ("<sep>",), style)
        large = _calls_in_chunk_cuts("\n" * 1200 + "<sep>", ("<sep>",), style)
        assert large <= small * 2 + 50

    def test_the_work_stays_in_line_with_many_runs_and_separators(self):
        block = "\n\n\n<sep>\n\nab\n"
        small = _calls_in_chunk_cuts(block * 300, ("<sep>",), "newlines")
        large = _calls_in_chunk_cuts(block * 600, ("<sep>",), "newlines")
        assert large <= small * 2 + 50

    def test_a_long_line_break_run_beside_a_separator_has_no_cut_in_it(self):
        text = "a\n" + "\n" * 2000 + "<sep>\nb"
        assert guard.chunk_cuts(text, ("<sep>",), "newlines") == []

    def test_a_line_break_run_away_from_a_separator_keeps_its_cuts(self):
        text = "<sep>a\n\n\nb"
        assert guard.chunk_cuts(text, ("<sep>",), "newlines") == [7, 8, 9]


class TestSplitText:
    def test_a_text_that_fits_is_one_chunk(self):
        assert guard.split_text("ab cd", 10) == ["ab cd"]

    @pytest.mark.parametrize("cap", [80, 100, 1000])
    def test_chunks_join_to_the_text_and_fit(self, cap):
        text = _prose(5000)
        chunks = guard.split_text(text, cap)
        assert "".join(chunks) == text
        assert all(len(c) <= cap for c in chunks)
        assert len(chunks) > 1

    def test_a_stretch_with_no_cut_stays_one_oversized_chunk(self):
        text = "x" * 50 + " " + "y" * 200 + " " + "z" * 10
        chunks = guard.split_text(text, 60)
        assert "".join(chunks) == text
        assert max(len(c) for c in chunks) == 201

    def test_every_cut_is_one_of_the_cut_points(self):
        text = _prose(3000)
        cuts = set(guard.chunk_cuts(text))
        pos = 0
        for chunk in guard.split_text(text, 40)[:-1]:
            pos += len(chunk)
            assert pos in cuts


class TestPlanChunks:
    @pytest.mark.parametrize("pre_type", _CHUNKABLE)
    def test_a_long_text_with_breaks_becomes_chunks_within_the_limit(self, pre_type):
        policy = guard.UNSAFE_PRE_TYPES[pre_type]
        text = _prose(policy.max_chars * 3 + 123)
        chunks = guard.plan_chunks(pre_type, text)
        assert len(chunks) >= 3
        assert "".join(chunks) == text
        assert all(len(c) <= policy.max_chars for c in chunks)

    def test_a_cost_bounded_pre_type_gets_smaller_chunks(self):
        policy = guard.UNSAFE_PRE_TYPES["deepseek-llm"]
        text = _prose(40000) + "x" * 3000 + "\n" + _prose(40000)
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.check_text("deepseek-llm", text)
        chunks = guard.plan_chunks("deepseek-llm", text)
        assert "".join(chunks) == text
        assert max(len(c) for c in chunks) <= policy.max_chars // 8

    def test_a_text_that_passes_is_one_piece(self):
        text = _prose(1000)
        assert guard.plan_chunks("gpt-4o", text) == [text]

    def test_an_unaffected_pre_tokenizer_is_one_piece(self):
        text = _prose(_CAP * 3)
        assert guard.plan_chunks("qwen2", text) == [text]
        assert guard.plan_chunks(None, text) == [text]

    def test_a_long_run_is_still_refused_in_a_text_that_is_chunked(self):
        text = _prose(_CAP * 2) + "a" * 100 + _prose(100)
        with pytest.raises(PretokenizerUnsafeInputError, match="unbroken run of more than 64"):
            guard.plan_chunks("gpt-4o", text)

    def test_a_long_line_break_run_is_refused_past_its_limit(self):
        limit = guard.UNSAFE_PRE_TYPES["gpt-4o"].newline_run
        guard.plan_chunks("gpt-4o", "a" + "\n" * limit + "b")
        with pytest.raises(PretokenizerUnsafeInputError, match="line breaks"):
            guard.plan_chunks("gpt-4o", "a" + "\n" * (limit + 1) + "b")

    def test_a_stretch_with_nothing_to_cut_at_is_refused_and_says_so(self):
        text = ("ab-" * ((_CAP + 5) // 3))
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.plan_chunks("gpt-4o", text)
        assert "in a row without a space or line break" in str(exc.value)
        assert str(len(text)) in str(exc.value)
        parsed = guard.parse_refusal(str(exc.value))
        assert parsed == guard.Refusal("chars", _CAP, len(text), part=True)

    def test_superbpe_refuses_long_prose_without_a_number(self):
        text = "word " * 20000
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.plan_chunks("superbpe", text, ("<s>",))
        assert "in a row without a number" in str(exc.value)
        assert str(len(text)) in str(exc.value)

    def test_superbpe_takes_the_same_prose_with_numbers_in_it(self):
        text = "word " * 4000 + "12 " + "word " * 4000 + "34 " + "word " * 4000
        chunks = guard.plan_chunks("superbpe", text * 2, ("<s>",))
        assert len(chunks) > 1
        assert "".join(chunks) == text * 2

    def test_deepseek_llm_refuses_long_prose_without_a_line_break(self):
        text = "word " * 20000
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.plan_chunks("deepseek-llm", text)
        assert "in a row without a line break" in str(exc.value)
        assert str(len(text)) in str(exc.value)

    def test_deepseek_llm_takes_the_same_prose_with_line_breaks_in_it(self):
        text = "word\n" * 20000
        chunks = guard.plan_chunks("deepseek-llm", text)
        assert len(chunks) > 1
        assert "".join(chunks) == text

    def test_a_words_pre_tokenizer_takes_the_same_prose(self):
        text = "word " * 20000
        chunks = guard.plan_chunks("gpt-4o", text)
        assert "".join(chunks) == text

    def test_the_refusal_quotes_none_of_the_text(self):
        text = "secret-" * ((_CAP + 5) // 7 + 1)
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.plan_chunks("gpt-4o", text)
        assert "secret" not in str(exc.value)

    def test_a_vocabulary_that_cannot_join_chunks_keeps_the_refusal(self):
        with pytest.raises(PretokenizerUnsafeInputError, match="at once"):
            guard.plan_chunks("gpt-4o", _prose(_CAP + 10), chunkable=False)

    def test_an_unusable_pre_tokenizer_is_never_chunked(self):
        with pytest.raises(PretokenizerUnsafeInputError):
            guard.plan_chunks("exaone-moe", _prose(_CAP + 10))

    def test_special_token_pieces_each_get_their_own_length_limit(self):
        msg = _prose(_CAP - 100)
        text = "<s>" + msg + "<sep>" + msg + "<sep>" + msg
        chunks = guard.plan_chunks("gpt-4o", text, ("<s>", "<sep>"))
        assert "".join(chunks) == text

    def test_an_oversized_piece_between_special_tokens_is_refused(self):
        text = "<s>" + "ab-" * (_CAP // 3 + 5) + "<sep>ok"
        with pytest.raises(PretokenizerUnsafeInputError, match="in a row"):
            guard.plan_chunks("gpt-4o", text, ("<s>", "<sep>"))


class TestTokenizerDrivesTheChunks:
    def _run(self, pre_type, text, **kw):
        calls = []
        tok = _fake_tokenizer(pre_type)
        with patch(_LLAMA_API) as api:
            api.llama_tokenize.side_effect = _fake_llama_tokenize(calls)
            ids = tok.encode(text, **kw)
        return ids, calls

    def test_a_long_text_reaches_native_code_in_safe_chunks_and_joins(self):
        text = _prose(_CAP * 2 + 77)
        ids, calls = self._run("gpt-4o", text)
        assert len(calls) >= 3
        assert all(len(raw) <= _CAP for raw, _, _ in calls)
        assert "".join(raw for raw, _, _ in calls) == text
        assert ids == [1] + list(text.encode("utf-8"))

    def test_only_the_first_chunk_asks_for_special_tokens(self):
        _, calls = self._run("gpt-4o", _prose(_CAP * 2))
        assert [add for _, add, _ in calls] == [True] + [False] * (len(calls) - 1)

    def test_no_bos_is_added_when_not_asked(self):
        text = _prose(_CAP * 2)
        ids, calls = self._run("gpt-4o", text, add_bos=False)
        assert not any(add for _, add, _ in calls)
        assert ids == list(text.encode("utf-8"))

    def test_a_short_text_is_still_the_single_native_call(self):
        ids, calls = self._run("gpt-4o", "hello there")
        assert calls == [("hello there", True, True)]

    def test_an_unaffected_model_is_one_call_for_any_length(self):
        text = _prose(_CAP * 3)
        _, calls = self._run("qwen2", text)
        assert len(calls) == 1

    def test_a_refused_text_never_reaches_native_code(self):
        calls = []
        tok = _fake_tokenizer("gpt-4o")
        with patch(_LLAMA_API) as api:
            api.llama_tokenize.side_effect = _fake_llama_tokenize(calls)
            with pytest.raises(PretokenizerUnsafeInputError):
                tok.encode(_prose(_CAP * 2) + "a" * 80)
        assert calls == []

    def test_untrusted_ranges_are_parsed_without_specials_in_every_chunk(self):
        left = _prose(_CAP + 500)
        middle = _prose(_CAP + 500)
        text = left + middle
        ids, calls = self._run("gpt-4o", text,
                               untrusted_ranges=[(len(left), len(text))])
        assert "".join(raw for raw, _, _ in calls) == text
        trusted = "".join(raw for raw, _, parse in calls if parse)
        untrusted = "".join(raw for raw, _, parse in calls if not parse)
        assert trusted == left
        assert untrusted == middle
        assert all(len(raw) <= _CAP for raw, _, _ in calls)

    def test_a_vocabulary_that_cannot_join_chunks_is_refused_as_before(self):
        tok = _fake_tokenizer("gpt-4o")
        tok._chunkable = False
        with patch(_LLAMA_API) as api:
            with pytest.raises(PretokenizerUnsafeInputError, match="at once"):
                tok.encode(_prose(_CAP + 10))
        api.llama_tokenize.assert_not_called()

    def test_check_still_refuses_a_text_over_the_limit(self):
        tok = _fake_tokenizer("gpt-4o")
        with pytest.raises(PretokenizerUnsafeInputError):
            tok.check(_prose(_CAP + 10))


class TestChunkableVocabulary:
    def _decide(self, vocab_type, add_eos, meta=None):
        tok = _Tokenizer.__new__(_Tokenizer)
        tok._vocab = MagicMock()
        meta = meta or {}
        with patch(_LLAMA_API) as api:
            api.llama_vocab_type.return_value = vocab_type
            api.llama_vocab_get_add_eos.return_value = add_eos
            api.has_model_meta_api.return_value = True
            api.llama_model_meta_val_str.side_effect = lambda m, key: meta.get(key)
            return tok._tokens_join_across_cuts(object())

    def test_a_bpe_vocabulary_without_an_end_token_is_chunkable(self):
        assert self._decide(2, False) is True

    def test_a_vocabulary_that_adds_an_end_token_is_not(self):
        assert self._decide(2, True) is False

    def test_a_runtime_that_cannot_say_is_not(self):
        assert self._decide(2, None) is False

    @pytest.mark.parametrize("vocab_type", [0, 1, 3, 4, 5])
    def test_another_vocabulary_type_is_not(self, vocab_type):
        assert self._decide(vocab_type, False) is False

    def test_a_vocabulary_that_prepends_a_space_or_metaspace_is_not(self):
        assert self._decide(2, False, {"tokenizer.ggml.add_space_prefix": "true"}) is False
        assert self._decide(2, False, {"tokenizer.ggml.metaspace_prepend_scheme": "first"}) is False
        assert self._decide(
            2, False, {"tokenizer.ggml.metaspace_prepend_scheme": "never"}) is True

    def test_a_tokenizer_that_cannot_list_its_special_tokens_is_not_chunked(self):
        tok = _Tokenizer.__new__(_Tokenizer)
        model = MagicMock()
        with patch(_LLAMA_API) as api, \
                patch.object(guard, "read_pre_type", return_value="gpt-4o"), \
                patch.object(guard, "read_separators", return_value=((), ())), \
                patch.object(_Tokenizer, "_tokens_join_across_cuts", return_value=True):
            api.llama_model_get_vocab.return_value = MagicMock()
            tok.__init__(model, None)
        assert tok._chunkable is False


def _mini_gguf(pre_type: str) -> bytes:
    """A vocabulary-only GGUF: a byte-level BPE with merges over spaces, line
    breaks, digits and a few letters, so a cut that changes a merge changes the
    tokens."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD))
          + list(range(0xAE, 0x100)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs, strict=True)}
    tokens = [b2u[i] for i in range(256)]
    merges = []
    sp, nl = b2u[32], b2u[10]
    pairs = [("t", "h"), ("th", "e"), (sp, "t"), (sp + "t", "h"), (sp + "th", "e"),
             ("a", "b"), ("ab", "c"), (nl, nl), (nl + nl, nl + nl), (sp, sp),
             (sp + sp, sp + sp), ("a", "a"), ("aa", "aa"), ("1", "2"), ("12", "3"),
             (nl, sp), (b2u[13], nl),
             (sp, b2u[0xF0]), (b2u[0xF0], b2u[0x9F]), (sp + b2u[0xF0], b2u[0x9F]),
             (sp, b2u[0xD7]), (b2u[0xD7], b2u[0x90]), (sp + b2u[0xD7], b2u[0x90]),
             (b2u[0xE2], b2u[0x82]), (sp, b2u[0xE2]), (sp + b2u[0xE2], b2u[0x82]),
             (b2u[0xAC], sp), (b2u[0x80], sp), (b2u[0x90], sp)]
    for a, b in pairs:
        merges.append(a + " " + b)
        if a + b not in tokens:
            tokens.append(a + b)
    tokens.append("<|endoftext|>")
    types = [1] * (len(tokens) - 1) + [3]

    def s(x):
        raw = x.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def kv_str(k, v):
        return s(k) + struct.pack("<I", 8) + s(v)

    def kv_u32(k, v):
        return s(k) + struct.pack("<II", 4, v)

    def kv_arr_str(k, vs):
        return (s(k) + struct.pack("<IIQ", 9, 8, len(vs))
                + b"".join(s(v) for v in vs))

    def kv_arr_i32(k, vs):
        return (s(k) + struct.pack("<IIQ", 9, 5, len(vs))
                + struct.pack("<%di" % len(vs), *vs))

    kvs = [kv_str("general.architecture", "llama"), kv_str("general.name", "mini"),
           kv_u32("llama.context_length", 4096), kv_u32("llama.embedding_length", 64),
           kv_u32("llama.block_count", 1), kv_u32("llama.feed_forward_length", 128),
           kv_u32("llama.attention.head_count", 2),
           kv_u32("llama.attention.head_count_kv", 2),
           kv_u32("llama.vocab_size", len(tokens)),
           kv_u32("llama.rope.dimension_count", 32),
           kv_str("tokenizer.ggml.model", "gpt2"), kv_str("tokenizer.ggml.pre", pre_type),
           kv_arr_str("tokenizer.ggml.tokens", tokens),
           kv_arr_i32("tokenizer.ggml.token_type", types),
           kv_arr_str("tokenizer.ggml.merges", merges),
           kv_u32("tokenizer.ggml.eos_token_id", len(tokens) - 1)]
    return (b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))


# Runs in its own process: it loads the native library and tokenizes. Every text
# it hands the tokenizer whole is one the guard accepts, so nothing here can
# trip the abort the guard exists to prevent.
_ORACLE = r'''
import json, random, sys
from localm.inference import pretokenizer_guard as g
from localm.inference.backends.llamacpp import _api as api
from localm.inference.backends.llamacpp._structs import llama_token

api.llama_backend_init()
ALPHA = ["a", "b", "Z", "x", " ", " ", " ", "  ", "\n", "\n", "\r\n", "\r", "\t", "/",
         "!", "1", "2", "3", "45", "'", "s", "'s", "-", ".", "\u4e2d", "\u00e9",
         "\u00a0", "\u3000", "\u200b", "\u0301", "_", "<", "$", "th", "the", "abc",
         "<|endoftext|>", "<|endoftext|>", "\U0001F600", "\u05d0", "\u20ac", "\u05d0\u05d1"]
SEPARATORS = ("<|endoftext|>",)
PROSE = ("Der schnelle braune Fuchs springt \u00fcber den Hund.\n\nZweite Zeile  mit "
         "Leerzeichen.\n  indented line 123 and the cat's toy/path\r\nend\n")


def tokenizer(path):
    p = api.llama_model_default_params()
    p.vocab_only = True
    model = api.llama_load_model_from_file(path, p)
    assert model, "vocabulary-only load failed"
    vocab = api.llama_model_get_vocab(model)

    def tok(text):
        raw = text.encode("utf-8")
        n_max = len(raw) + 128
        buf = (llama_token * n_max)()
        n = api.llama_tokenize(vocab, raw, len(raw), buf, n_max, False, True)
        if n < 0:
            n_max = -n + 64
            buf = (llama_token * n_max)()
            n = api.llama_tokenize(vocab, raw, len(raw), buf, n_max, False, True)
        assert n >= 0
        return [buf[i] for i in range(n)]
    return tok


def compare(tok, text, cuts):
    pts = [0] + list(cuts) + [len(text)]
    joined = []
    for a, b in zip(pts, pts[1:]):
        joined += tok(text[a:b])
    return joined == tok(text)


def run(directory, pre_types, n_fuzz):
    result = {}
    rnd = random.Random(11)
    for pre in pre_types:
        tok = tokenizer(f"{directory}/{pre}.gguf")
        compared = mismatched = bad_compared = bad_mismatched = cuts_made = 0
        texts = [PROSE * 5, PROSE.replace("\n", "\r\n") * 5]
        texts += ["".join(rnd.choice(ALPHA) for _ in range(rnd.randint(1, 14)))
                  for _ in range(n_fuzz)]
        for text in texts:
            cuts = g.chunk_cuts(text, SEPARATORS, g.UNSAFE_PRE_TYPES[pre].cuts)
            if not cuts:
                continue
            for pick in (cuts, [c for c in cuts if rnd.random() < 0.5]):
                if pick:
                    compared += 1
                    cuts_made += len(pick)
                    mismatched += not compare(tok, text, pick)
            every = list(range(1, len(text)))
            if every and len(text) < 40:
                bad_compared += 1
                bad_mismatched += not compare(tok, text, every)
        result[pre] = [compared, mismatched, cuts_made, bad_compared, bad_mismatched]
    return result


print(json.dumps(run(sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]))))
'''


@pytest.mark.real_gguf
class TestChunkedTokensEqualWholeTokens:
    """Whole against chunked, through llama.cpp's own tokenizer, for every
    pre-type the guard bounds that a model can be loaded with."""

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        from tests._real_gguf import require_native_runtime
        require_native_runtime()
        directory = tmp_path_factory.mktemp("minivocab")
        for pre in _CHUNKABLE:
            (directory / f"{pre}.gguf").write_bytes(_mini_gguf(pre))
        repo = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONIOENCODING": "utf-8"}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [sys.executable, "-c", _ORACLE, str(directory), json.dumps(_CHUNKABLE), "400"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
            timeout=600, creationflags=flags)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return json.loads(proc.stdout.strip().splitlines()[-1])

    @pytest.mark.parametrize("pre_type", _CHUNKABLE)
    def test_chunked_tokens_equal_whole_tokens(self, result, pre_type):
        compared, mismatched, cuts_made = result[pre_type][:3]
        assert compared > 100 and cuts_made > 200
        assert mismatched == 0

    @pytest.mark.parametrize("pre_type", _CHUNKABLE)
    def test_the_comparison_fails_for_a_cut_that_is_not_safe(self, result, pre_type):
        # The same comparison with a cut at every offset must find differences,
        # or "no mismatches" above says nothing.
        bad_compared, bad_mismatched = result[pre_type][3:]
        assert bad_compared > 20
        assert bad_mismatched > 0


class TestTheGuardTableStillBoundsEveryRun:
    def test_every_affected_policy_has_a_total_limit_for_the_chunk_size(self):
        for name, policy in guard.UNSAFE_PRE_TYPES.items():
            assert policy.max_chars, name

    def test_chunks_are_never_longer_than_the_limit_measured_safe(self):
        rnd = random.Random(3)
        for pre_type in _CHUNKABLE:
            policy = guard.UNSAFE_PRE_TYPES[pre_type]
            words = ["alpha", "beta", "gamma", "12", "x/y", "\n", "\n\n", " ", ".", "z"]
            text = "".join(rnd.choice(words) + " " for _ in range(policy.max_chars // 3))
            chunks = guard.plan_chunks(pre_type, text)
            assert max(len(c) for c in chunks) <= policy.max_chars
