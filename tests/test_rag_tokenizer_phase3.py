# SPDX-License-Identifier: AGPL-3.0-or-later
"""The BM25 tokenizer must be unicode-aware.

These tests pin the unicode behavior for CJK, Cyrillic, Arabic and accented
latin corpora while guarding the existing ASCII contract.
"""

from __future__ import annotations

from localm.rag.bm25 import BM25, tokenize


class TestUnicodeTokenize:
    def test_ascii_unchanged(self):
        assert tokenize("Hello, World! x2") == ["hello", "world", "x2"]

    def test_cyrillic_tokenizes_nonempty(self):
        toks = tokenize("Привет мир")
        assert toks == ["привет", "мир"]

    def test_accented_latin_preserved(self):
        # Accented latin must survive (not be stripped to ASCII fragments).
        toks = tokenize("Café déjà vu")
        assert toks == ["café", "déjà", "vu"]

    def test_arabic_tokenizes_nonempty(self):
        toks = tokenize("مرحبا بالعالم")
        assert toks == ["مرحبا", "بالعالم"]

    def test_cjk_split_per_character(self):
        # CJK has no spaces, so a run of ideographs must become per-char tokens.
        toks = tokenize("机器学习")
        assert toks == ["机", "器", "学", "习"]

    def test_mixed_cjk_and_latin(self):
        # Latin words stay whole; CJK runs split per character.
        toks = tokenize("hello 世界 world")
        assert toks == ["hello", "世", "界", "world"]


class TestUnicodeBM25:
    def test_cjk_query_returns_hit(self):
        docs = [
            "机器学习模型",          # machine learning model
            "今天天气很好",          # the weather is nice today
            "我喜欢喝咖啡",          # I like drinking coffee
        ]
        scores = BM25(docs).scores("机器学习")
        # The first doc shares the query characters; it must outscore the rest.
        assert max(scores) > 0.0
        assert scores.index(max(scores)) == 0

    def test_cyrillic_query_returns_hit(self):
        docs = [
            "машинное обучение модели",
            "сегодня хорошая погода",
        ]
        scores = BM25(docs).scores("машинное обучение")
        assert max(scores) > 0.0
        assert scores.index(max(scores)) == 0

    def test_non_latin_corpus_not_silently_empty(self):
        # An all-CJK corpus must not score uniformly zero against a relevant CJK
        # query.
        docs = ["数据库索引", "网络协议栈"]
        scores = BM25(docs).scores("数据库")
        assert scores != [0.0, 0.0]
