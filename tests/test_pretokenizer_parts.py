# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pre-tokenizer guard bounds what llama.cpp's pre-tokenizer sees in ONE
piece: llama.cpp cuts a text at its special tokens first and runs the
pre-tokenizer on each piece, so a conversation longer than the length bound is
fine as long as no single piece is. A refusal names what was too long, in a
form a client can read back."""

from __future__ import annotations

import random

import pytest

from localm.inference import pretokenizer_guard as guard
from localm.inference.backends.base import PretokenizerUnsafeInputError

_CAP = guard.UNSAFE_PRE_TYPES["tekken"].max_chars


def _prose(n: int) -> str:
    return ("the quick brown fox jumps over it. " * (n // 34 + 1))[:n]


def _llama_cpp_pieces(text: str, specials: list[str]) -> list[tuple[int, int]]:
    """Python port of llama_vocab::impl::tokenizer_st_partition: for each special
    token in *specials* order, cut every raw fragment at each leftmost
    occurrence, left to right. Returns the raw fragments as (start, end)."""
    frags: list = [("raw", 0, len(text))]
    for sp in specials:
        out: list = []
        for frag in frags:
            if frag[0] != "raw":
                out.append(frag)
                continue
            _, start, end = frag
            base = start
            while True:
                match = text.find(sp, base, end)
                if match == -1 or match + len(sp) > end:
                    if base < end:
                        out.append(("raw", base, end))
                    break
                if match > base:
                    out.append(("raw", base, match))
                out.append(("tok", match, match + len(sp)))
                base = match + len(sp)
        frags = out
    return [(s, e) for kind, s, e in frags if kind == "raw"]


def _piece_spans(text: str, separators) -> list[tuple[int, int]]:
    """Where split_at_separators' pieces sit in *text*."""
    pieces = guard.split_at_separators(text, separators)
    spans, pos = [], 0
    for p in pieces:
        at = text.find(p, pos)
        spans.append((at, at + len(p)))
        pos = at + len(p)
    return spans


def test_pieces_are_never_shorter_than_llama_cpp_cuts():
    rng = random.Random(7)
    alphabet = "ab<>|[] \n"
    for _ in range(400):
        specials = list({"".join(rng.choice("ab<>|[]") for _ in range(rng.randint(1, 4)))
                         for _ in range(rng.randint(1, 5))})
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        mine = _piece_spans(text, specials)
        for order in (sorted(specials, key=len, reverse=True),
                      sorted(specials, key=lambda s: (-len(s), s[::-1]))):
            for s, e in _llama_cpp_pieces(text, order):
                assert any(ms <= s and e <= me for ms, me in mine), (text, specials, order)


def test_a_conversation_longer_than_the_cap_passes_when_each_message_fits():
    msg = _prose(20000)
    text = "<s>" + "".join(f"[INST]{msg}[/INST]{msg}</s>" for _ in range(3))
    assert len(text) > _CAP
    with pytest.raises(PretokenizerUnsafeInputError):
        guard.check_text("tekken", text)
    guard.check_parts("tekken", text, ("[INST]", "[/INST]", "<s>", "</s>"))


def test_one_message_over_the_cap_is_refused_and_named_as_one_message():
    text = "[INST]" + _prose(_CAP + 10) + "[/INST]ok"
    with pytest.raises(PretokenizerUnsafeInputError) as exc:
        guard.check_parts("tekken", text, ("[INST]", "[/INST]"))
    assert "in one message" in str(exc.value)
    assert str(_CAP + 10) in str(exc.value)


def test_without_separators_the_whole_text_is_the_piece():
    text = _prose(_CAP + 1)
    with pytest.raises(PretokenizerUnsafeInputError) as exc:
        guard.check_parts("tekken", text, ())
    assert "at once" in str(exc.value)


def test_an_unaffected_pre_tokenizer_is_not_checked():
    guard.check_parts("qwen2", _prose(_CAP * 3), ())


def test_overlapping_separator_occurrences_are_not_cut():
    assert guard.split_at_separators("x<ab>y", ("<ab", "ab>")) == ["x<ab>y"]
    assert guard.split_at_separators("x<ab>y<ab>z", ("<ab>",)) == ["x", "y", "z"]


def test_a_separator_with_edge_whitespace_is_never_cut_at():
    assert guard.split_at_separators("a <m> b", (" <m>",)) == ["a <m> b"]


def test_a_run_inside_one_piece_is_still_refused():
    text = "[INST]" + "a" * 80 + "[/INST]"
    with pytest.raises(PretokenizerUnsafeInputError):
        guard.check_parts("tekken", text, ("[INST]", "[/INST]"))


@pytest.mark.parametrize("pre_type", sorted(guard.UNSAFE_PRE_TYPES))
def test_every_refusal_reads_back(pre_type):
    policy = guard.UNSAFE_PRE_TYPES[pre_type]
    cases = [(_prose(policy.max_chars + 5), False), (_prose(policy.max_chars + 5), True)]
    for text, part in cases:
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.check_text(pre_type, text, part=part)
        got = guard.parse_refusal("400 Bad Request: " + str(exc.value))
        assert got is not None, str(exc.value)
        if got.kind == "chars":
            assert (got.limit, got.size, got.part) == (policy.max_chars, len(text), part)
    if policy.char_class is not None and not policy.unusable:
        run = {"letters": "a", "digits": "7", "whitespace characters": " ",
               "letters and spaces": "a"}[guard._RUN_KIND[policy.char_class]]
        text = "x." + run * (policy.max_run + 3) + ".x"
        with pytest.raises(PretokenizerUnsafeInputError) as exc:
            guard.check_text(pre_type, text)
        got = guard.parse_refusal(str(exc.value))
        assert got is not None and got.kind == "run", str(exc.value)
        assert got.limit == policy.max_run and got.size == policy.max_run + 3


def test_a_block_refusal_reads_back():
    policy = guard.UNSAFE_PRE_TYPES["deepseek-llm"]
    block = "x" * (policy.cost_budget // 2000 + 10)
    text = (block + " ") + _prose(2000 - 1)
    with pytest.raises(PretokenizerUnsafeInputError) as exc:
        guard.check_text("deepseek-llm", text)
    got = guard.parse_refusal(str(exc.value))
    assert got is not None and got.kind == "block"
    assert got.size == len(block)


def test_unrelated_text_is_no_refusal():
    assert guard.parse_refusal("503 Service Unavailable: model loading") is None


class _FakeApi:
    def __init__(self, tokens):
        self._tokens = tokens

    def has_vocab_attr_api(self):
        return True

    def llama_vocab_n_tokens(self, vocab):
        return len(self._tokens)

    def llama_vocab_get_attr(self, vocab, tok):
        return self._tokens[tok][1]

    def llama_vocab_get_text(self, vocab, tok):
        return self._tokens[tok][0].encode()


def test_separators_are_the_control_user_defined_and_unknown_tokens():
    api = _FakeApi([
        ("hello", 1 << 2),
        ("[INST]", guard.TOKEN_ATTR_CONTROL),
        ("<unk>", guard.TOKEN_ATTR_UNKNOWN),
        ("<|tool|>", guard.TOKEN_ATTR_USER_DEFINED),
        ("", guard.TOKEN_ATTR_CONTROL),
    ])
    special, untrusted = guard.read_separators(None, api)
    assert special == ("[INST]", "<unk>", "<|tool|>")
    assert untrusted == ("<|tool|>",)


def test_a_runtime_without_token_attributes_has_no_separators():
    api = _FakeApi([])
    api.has_vocab_attr_api = lambda: False
    assert guard.read_separators(None, api) == ((), ())


def test_the_tokenizer_cuts_untrusted_text_only_at_user_defined_tokens():
    from localm.inference.backends.llamacpp.llama import _Tokenizer
    tok = object.__new__(_Tokenizer)
    tok._pre_type = "tekken"
    tok._separators = ("[INST]",)
    tok._separators_untrusted = ()
    msg = _prose(30000)
    trusted = f"[INST]{msg}[INST]{msg}[INST]{msg}"
    tok.check(trusted)
    # The same text inside an untrusted range is one piece: [INST] there is not
    # a token, so it is not a cut.
    with pytest.raises(PretokenizerUnsafeInputError):
        tok.check(trusted, untrusted_ranges=((0, len(trusted)),))
