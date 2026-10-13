# SPDX-License-Identifier: AGPL-3.0-or-later
"""The HF backend's grammar mask is applied with a backend that compiles nothing
at run time (never xgrammar's "auto", which resolves to Triton for CUDA/ROCm
logits), and the processor applying it follows xgrammar's token protocol.
A stand-in xgrammar module records what the processor asks of it, so xgrammar
need not be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from localm.inference.backends import _hf_worker

_GRAMMAR = 'root ::= "red" | "green" | "blue"'
_COMPILER_BACKENDS = {"auto", "triton", "torch_compile", "cuda"}


@pytest.mark.parametrize("device_type,expected", [
    ("cuda", "torch_native"), ("cpu", "cpu"), ("xpu", "cpu"), ("mps", "cpu"),
    ("npu", "cpu"), ("hpu", "cpu"), ("privateuseone", "cpu"), ("meta", "cpu")])
def test_bitmask_backend_never_needs_a_compiler(device_type, expected):
    backend = _hf_worker._bitmask_backend(device_type)
    assert backend == expected
    assert backend not in _COMPILER_BACKENDS


class _Device:
    def __init__(self, type_):
        self.type = type_


class _Scores:
    def __init__(self, device_type, moves=None):
        self.device = _Device(device_type)
        self.moves = [] if moves is None else moves

    def to(self, device):
        self.moves.append(device)
        return _Scores(device.type if isinstance(device, _Device) else device, self.moves)


class _InputIds:
    def __init__(self, rows):
        self._rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        i, j = key
        return self._rows[i][j]


class _Bitmask:
    def __init__(self):
        self.filled = []
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return self


@pytest.fixture
def fake_xgrammar(monkeypatch):
    """A stand-in xgrammar recording matcher traffic and every mask application."""
    rec = types.SimpleNamespace(applied=[], matchers=[], reject=set())
    xgr = types.ModuleType("xgrammar")

    class _Info:
        vocab_size = 100

        @staticmethod
        def from_huggingface(tokenizer, vocab_size=None):
            return _Info()

    class _Compiled:
        tokenizer_info = _Info()

    class _Compiler:
        def __init__(self, info):
            pass

        def compile_grammar(self, grammar):
            return _Compiled()

    class _Matcher:
        def __init__(self, compiled):
            self.accepted = []
            rec.matchers.append(self)

        def is_terminated(self):
            return False

        def accept_token(self, token):
            self.accepted.append(token)
            return token not in rec.reject

        def fill_next_token_bitmask(self, bitmask, index):
            bitmask.filled.append(index)

    def _apply(scores, bitmask, *, vocab_size=None, indices=None, backend="auto"):
        rec.applied.append((scores, backend))

    xgr.TokenizerInfo = _Info
    xgr.GrammarCompiler = _Compiler
    xgr.GrammarMatcher = _Matcher
    xgr.allocate_token_bitmask = lambda batch, vocab: _Bitmask()
    xgr.apply_token_bitmask_inplace = _apply
    monkeypatch.setitem(sys.modules, "xgrammar", xgr)
    tfm = types.ModuleType("transformers")
    tfm.LogitsProcessorList = list
    monkeypatch.setitem(sys.modules, "transformers", tfm)
    return rec


def _processor():
    procs = _hf_worker._grammar_processor(_GRAMMAR, object(), object())
    assert len(procs) == 1
    return procs[0]


def test_gpu_logits_are_masked_with_a_compiler_free_backend(fake_xgrammar):
    proc = _processor()
    for step in range(3):
        scores = _Scores("cuda")
        assert proc(_InputIds([[1, 2 + step]]), scores) is scores
    assert [b for _, b in fake_xgrammar.applied] == ["torch_native"] * 3
    assert proc.failed is False


def test_cpu_logits_are_masked_with_the_bundled_kernel(fake_xgrammar):
    proc = _processor()
    proc(_InputIds([[1]]), _Scores("cpu"))
    assert [b for _, b in fake_xgrammar.applied] == ["cpu"]


def test_other_device_logits_are_masked_on_a_cpu_copy(fake_xgrammar):
    proc = _processor()
    scores = _Scores("xpu")
    out = proc(_InputIds([[1]]), scores)
    (masked, backend), = fake_xgrammar.applied
    assert (masked.device.type, backend) == ("cpu", "cpu")
    assert scores.moves == ["cpu", scores.device]
    assert out.device.type == "xpu"


def test_sampled_tokens_reach_the_matcher_after_the_first_call(fake_xgrammar):
    proc = _processor()
    proc(_InputIds([[5, 6], [7, 8]]), _Scores("cuda"))
    assert [m.accepted for m in fake_xgrammar.matchers] == [[], []]
    proc(_InputIds([[5, 6, 9], [7, 8, 10]]), _Scores("cuda"))
    assert [m.accepted for m in fake_xgrammar.matchers] == [[9], [10]]
    assert proc.failed is False


def test_a_rejected_token_drops_the_constraint_and_reports_it(fake_xgrammar):
    fake_xgrammar.reject.add(13)
    proc = _processor()
    proc(_InputIds([[1]]), _Scores("cuda"))
    scores = _Scores("cuda")
    assert proc(_InputIds([[1, 13]]), scores) is scores
    assert proc.failed is True
    assert len(fake_xgrammar.applied) == 1


def test_a_batch_size_change_drops_the_constraint_and_reports_it(fake_xgrammar):
    proc = _processor()
    proc(_InputIds([[1]]), _Scores("cuda"))
    proc(_InputIds([[1, 2], [3, 4]]), _Scores("cuda"))
    assert proc.failed is True
    assert len(fake_xgrammar.applied) == 1
