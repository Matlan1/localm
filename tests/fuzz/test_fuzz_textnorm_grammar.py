# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the model-output normaliser and the GBNF pre-validators.

Model output and request grammars are untrusted text. The normaliser must give
the same answer for a reply however the token stream chunked it, and must run in
linear time; the grammar pre-validators must reject with ``InvalidGrammarError``
and nothing else."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, example, given, strategies as st  # noqa: E402

from localm import textnorm  # noqa: E402
from localm.inference import gbnf  # noqa: E402
from localm.inference.backends.base import InvalidGrammarError  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402

_FRAGMENTS = [
    "<think>", "</think>", "<", ">", "|", "/", "[", "]", "channel", "<|channel|>",
    "<|channel>", "<channel|>", "analysis", "thought", "final", "<|message|>", "\n", " ",
    "  ", "[TOOL_CALLS]", "<|turn>", "<turn|>", "model", "assistant", "<unused", "12",
    "x", "hello", "<start_of_turn>", "<|im_start|>", "<|start_header_id|>",
    "<|end_header_id|>", "<tool_call>", "<reasoning>", "</reasoning>", "<|\"|>",
    "<|think|>", "<|start|>", "<|return|>", "é", "\x00",
]

_text = st.lists(st.sampled_from(_FRAGMENTS), max_size=30).map("".join)


@st.composite
def chunked(draw):
    text = draw(_text)
    cuts = sorted(draw(st.lists(st.integers(0, len(text)), max_size=8)))
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(text[prev:c])
        prev = c
    pieces.append(text[prev:])
    return text, pieces


@given(data=chunked())
def test_scrub_stream_matches_one_shot_scrub_for_any_chunking(data):
    text, pieces = data
    assume(text.count("[TOOL_CALLS]") <= textnorm._MARKER_FLOOD_LIMIT)
    streamed = "".join(textnorm.scrub_stream(iter(pieces)))
    assert streamed == textnorm.scrub_text(text)


@given(data=chunked(), marker=st.sampled_from([None, "<tool_call>", "<"]))
def test_think_splitter_matches_split_think_for_any_chunking(data, marker):
    text, pieces = data
    text = textnorm.scrub_text(text)
    pieces = [textnorm.scrub_text(p) for p in pieces]
    assume("".join(pieces) == text)
    splitter = textnorm.ThinkSplitter(marker)
    content, reasoning = [], []
    for p in pieces:
        c, r = splitter.feed(p)
        content.append(c)
        reasoning.append(r)
    c, r = splitter.flush()
    content.append(c)
    reasoning.append(r)
    assert ("".join(content), "".join(reasoning)) == textnorm.split_think(text, marker)


@given(text=_text)
def test_scrub_text_is_idempotent_and_strip_think_never_raises(text):
    once = textnorm.scrub_text(text)
    assert textnorm.scrub_text(once) == once
    textnorm.strip_think(text)


_PATHOLOGICAL_UNIT = ["<", "< ", "<|", "<|    channel", "<unused", "<unused9", "[TOOL_CALLS",
                      "<think>", "</think>", "<start_of_turn", "<|channel|>analysis"]


@given(unit=st.sampled_from(_PATHOLOGICAL_UNIT))
def test_scrub_and_split_scale_linearly(unit):
    small, large = unit * 2_000, unit * 40_000

    def cost(text):
        t0 = time.perf_counter()
        textnorm.strip_think(text)
        return time.perf_counter() - t0

    cost(small)
    t_small = max(cost(small), 1e-4)
    t_large = cost(large)
    assert t_large < max(20 * 20 * t_small, 2.0), (
        f"{unit!r}: 2k reps {t_small:.4f}s, 40k reps {t_large:.4f}s")


@given(text=st.text(max_size=4000, alphabet=st.sampled_from("([{}])*+?|^$\\.,0123456789abc :=\"'-<>\n")))
@example(text="root ::= \"a\"{" + "9" * 5000 + "}")
@example(text="root ::= " + "(" * 200 + "x" + ")" * 200)
@example(text="a{1,99999999999999999999}")
def test_check_grammar_structure_raises_only_invalid_grammar_error(text):
    try:
        _bounds.returns_within(gbnf.check_grammar_structure, text)
    except InvalidGrammarError:
        pass


@given(extra=st.integers(1, 4096), filler=st.sampled_from(["a", "(", "{9", " "]))
def test_oversized_grammar_is_rejected_before_any_scan(extra, filler):
    grammar = filler * (gbnf.MAX_GRAMMAR_BYTES // len(filler) + extra)
    with pytest.raises(InvalidGrammarError):
        _bounds.returns_within(gbnf.check_grammar_structure, grammar)


@given(pattern=st.text(max_size=300, alphabet=st.sampled_from("()[]{}*+?|^$\\.,0123456789abc")))
@example(pattern="(" * 150 + "a" + ")*" * 150)
@example(pattern="[" * 3000)
def test_static_shape_rejection_is_total_and_fast(pattern):
    out = _bounds.returns_within(gbnf._static_shape_rejection, pattern)
    assert out is None or isinstance(out, str)
