# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query / document pair prompts for reranker models (localm.inference.rerank_pairs).

The layout mirrors llama.cpp's llama-server /v1/rerank: a model with a ``rerank``
chat template gets that template filled in and tokenised with special tokens
parsed; any other model gets ``[BOS] query [EOS] [SEP] document [EOS]`` with each
marker present only when the vocabulary asks for it.
"""

import pytest

from localm.inference import rerank_pairs as rp
from localm.inference.backends.base import RerankInputError

BOS, EOS, SEP = 1000, 1001, 1002
EOS_TEXT = "<E>"
EOS_TEXT_TOKEN = 2001


def make_tokenizer(calls=None):
    """One token per character; with parse_special the text ``<E>`` is ONE token."""
    def tokenize(text, add_special, parse_special):
        if calls is not None:
            calls.append((text, add_special, parse_special))
        out, i = [], 0
        while i < len(text):
            if parse_special and text.startswith(EOS_TEXT, i):
                out.append(EOS_TEXT_TOKEN)
                i += len(EOS_TEXT)
                continue
            out.append(ord(text[i]))
            i += 1
        return out
    return tokenize


def chars(text):
    return [ord(c) for c in text]


ALL = rp.VocabSpecials(bos=BOS, eos=EOS, sep=SEP, add_bos=True, add_eos=True, add_sep=True)


class TestPlainLayout:
    def test_every_marker_present_when_the_vocabulary_asks_for_all(self):
        pair = rp.build_pair(make_tokenizer(), ALL, None, 64, "ab", "cd")
        assert pair.tokens == [BOS, *chars("ab"), EOS, SEP, *chars("cd"), EOS]
        assert pair.truncated is False

    def test_xlmr_style_vocabulary_has_no_second_separator(self):
        sp = rp.VocabSpecials(bos=BOS, eos=EOS, sep=EOS, add_bos=True, add_eos=True,
                              add_sep=False)
        pair = rp.build_pair(make_tokenizer(), sp, None, 64, "ab", "cd")
        assert pair.tokens == [BOS, *chars("ab"), EOS, *chars("cd"), EOS]

    def test_markers_the_vocabulary_does_not_add_are_left_out(self):
        sp = rp.VocabSpecials(bos=BOS, eos=EOS, sep=SEP, add_bos=False, add_eos=False,
                              add_sep=False)
        pair = rp.build_pair(make_tokenizer(), sp, None, 64, "ab", "cd")
        assert pair.tokens == chars("abcd")

    def test_a_missing_eos_falls_back_to_the_separator(self):
        sp = rp.VocabSpecials(bos=BOS, eos=-1, sep=SEP, add_bos=True, add_eos=True,
                              add_sep=True)
        pair = rp.build_pair(make_tokenizer(), sp, None, 64, "ab", "cd")
        assert pair.tokens == [BOS, *chars("ab"), SEP, SEP, *chars("cd"), SEP]

    def test_a_token_the_vocabulary_lacks_is_never_emitted(self):
        sp = rp.VocabSpecials(bos=-1, eos=EOS, sep=-1, add_bos=True, add_eos=True,
                              add_sep=True)
        pair = rp.build_pair(make_tokenizer(), sp, None, 64, "ab", "cd")
        assert pair.tokens == [*chars("ab"), EOS, *chars("cd"), EOS]
        assert all(t >= 0 for t in pair.tokens)

    def test_query_and_document_are_tokenised_as_plain_text(self):
        calls = []
        rp.build_pair(make_tokenizer(calls), ALL, None, 64, "ab", "cd")
        assert calls == [("ab", False, False), ("cd", False, False)]

    def test_a_document_that_fits_exactly_is_not_marked_truncated(self):
        # head = BOS + 2 query + EOS + SEP = 5, tail = EOS = 1, so 6 document tokens fit in 12.
        pair = rp.build_pair(make_tokenizer(), ALL, None, 12, "ab", "cdefgh")
        assert len(pair.tokens) == 12
        assert pair.truncated is False

    def test_an_over_long_document_is_cut_and_keeps_the_closing_marker(self):
        pair = rp.build_pair(make_tokenizer(), ALL, None, 12, "ab", "cdefghijkl")
        assert len(pair.tokens) == 12
        assert pair.tokens[-1] == EOS
        assert pair.tokens[:5] == [BOS, *chars("ab"), EOS, SEP]
        assert pair.tokens[5:-1] == chars("cdefgh")
        assert pair.truncated is True

    def test_a_document_far_beyond_the_window_is_cut_before_it_is_tokenised(self):
        calls = []
        pair = rp.build_pair(make_tokenizer(calls), ALL, None, 10, "ab", "x" * 100_000)
        document_calls = [c for c in calls if c[0] != "ab"]
        assert len(document_calls) == 1
        assert len(document_calls[0][0]) <= 10 * rp._CHARS_PER_WINDOW_TOKEN
        assert len(pair.tokens) == 10
        assert pair.truncated is True

    def test_a_query_that_leaves_no_room_for_a_document_is_refused(self):
        with pytest.raises(RerankInputError, match="leaves no room"):
            rp.build_pair(make_tokenizer(), ALL, None, 6, "abcdefgh", "cd")

    def test_a_query_that_leaves_room_for_one_token_still_scores(self):
        pair = rp.build_pair(make_tokenizer(), ALL, None, 7, "ab", "cdef")
        assert len(pair.tokens) == 7
        assert pair.truncated is True

    def test_an_empty_document_still_makes_a_pair(self):
        pair = rp.build_pair(make_tokenizer(), ALL, None, 64, "ab", "")
        assert pair.tokens == [BOS, *chars("ab"), EOS, SEP, EOS]
        assert pair.truncated is False


TEMPLATE = "Q:{query}|D:{document}|<E>"


class TestFillTemplate:
    def test_both_fields_are_replaced(self):
        assert rp.fill_template(TEMPLATE, "qq", "dd") == "Q:qq|D:dd|<E>"

    def test_a_query_containing_the_document_field_is_left_as_written(self):
        assert rp.fill_template(TEMPLATE, "{document}", "dd") == "Q:{document}|D:dd|<E>"

    def test_a_field_appearing_twice_is_replaced_everywhere(self):
        assert rp.fill_template("{query}{query}{document}", "a", "b") == "aab"


class TestTemplatedLayout:
    def test_the_filled_template_is_tokenised_with_special_tokens_parsed(self):
        calls = []
        pair = rp.build_pair(make_tokenizer(calls), ALL, TEMPLATE, 64, "qq", "dd")
        assert calls == [("Q:qq|D:dd|<E>", False, True)]
        assert pair.tokens == [*chars("Q:qq|D:dd|"), EOS_TEXT_TOKEN]
        assert pair.truncated is False

    def test_the_vocabulary_markers_are_not_added_around_a_template(self):
        pair = rp.build_pair(make_tokenizer(), ALL, TEMPLATE, 64, "qq", "dd")
        assert BOS not in pair.tokens and SEP not in pair.tokens

    def test_a_cut_document_keeps_the_template_suffix(self):
        pair = rp.build_pair(make_tokenizer(), ALL, TEMPLATE, 20, "qq", "d" * 40)
        assert len(pair.tokens) <= 20
        assert pair.tokens[-1] == EOS_TEXT_TOKEN
        assert pair.tokens[:8] == chars("Q:qq|D:d")
        assert pair.truncated is True

    def test_the_cut_keeps_as_much_document_as_fits(self):
        # prefix "Q:qq|D:" is 7 tokens, suffix "|<E>" is 2 tokens: 11 document tokens fit in 20.
        pair = rp.build_pair(make_tokenizer(), ALL, TEMPLATE, 20, "qq", "d" * 40)
        assert len(pair.tokens) == 20
        assert pair.tokens.count(ord("d")) == 11

    def test_a_document_that_fits_is_not_marked_truncated(self):
        pair = rp.build_pair(make_tokenizer(), ALL, TEMPLATE, 20, "qq", "d" * 11)
        assert len(pair.tokens) == 20
        assert pair.truncated is False

    def test_a_query_that_fills_the_window_is_refused(self):
        with pytest.raises(RerankInputError, match="leaves no room"):
            rp.build_pair(make_tokenizer(), ALL, TEMPLATE, 12, "q" * 30, "dd")

    def test_a_document_far_beyond_the_window_is_cut_before_it_is_tokenised(self):
        calls = []
        pair = rp.build_pair(make_tokenizer(calls), ALL, TEMPLATE, 20, "qq", "d" * 100_000)
        assert max(len(c[0]) for c in calls) <= 20 * rp._CHARS_PER_WINDOW_TOKEN + len(TEMPLATE)
        assert pair.truncated is True
        assert len(pair.tokens) == 20


class TestVocabSpecialsRead:
    def test_reads_every_special_and_flag_through_the_native_binding(self):
        class Api:
            def llama_vocab_bos(self, v): return 5
            def llama_vocab_eos(self, v): return 6
            def llama_vocab_sep(self, v): return -1
            def llama_vocab_get_add_bos(self, v): return True
            def llama_vocab_get_add_eos(self, v): return 1
            def llama_vocab_get_add_sep(self, v): return False

        assert rp.VocabSpecials.read(Api(), object()) == rp.VocabSpecials(
            bos=5, eos=6, sep=-1, add_bos=True, add_eos=True, add_sep=False)
