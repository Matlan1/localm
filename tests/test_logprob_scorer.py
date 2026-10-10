# SPDX-License-Identifier: AGPL-3.0-or-later
"""``LogprobScorer``: a sampled token's log probability and the most likely
alternatives of its logits row.

With the native runtime provisioned, the scorer runs llama.cpp's own ``dist``
and ``top_k`` samplers over a real logits array and is checked against a
softmax computed here. Without it, a stand-in for those two samplers checks
the scorer's own bookkeeping."""
from __future__ import annotations

import ctypes
import math
import random
from types import SimpleNamespace

import pytest

from localm.inference.backends.llamacpp._logprobs import (
    FLOOR_LOGPROB, LogprobScorer, ScoredToken, tap_records)
from localm.inference.backends.llamacpp._structs import LlamaTokenData
from tests._real_gguf import native_runtime_lib_path, require_native_runtime


def reference(logits):
    m = max(logits)
    lse = m + math.log(math.fsum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


class _Row:
    """A logits row a fake context returns from llama_get_logits_ith."""

    def __init__(self, logits):
        self.array = (ctypes.c_float * len(logits))(*logits)
        self.reads = []

    def get(self, ctx, idx):
        self.reads.append((ctx, idx))
        return ctypes.cast(self.array, ctypes.POINTER(ctypes.c_float))


def _native_api(row):
    from localm.inference.backends.llamacpp import _api
    return SimpleNamespace(
        llama_get_logits_ith=row.get,
        llama_sampler_chain_default_params=_api.llama_sampler_chain_default_params,
        llama_sampler_chain_init=_api.llama_sampler_chain_init,
        llama_sampler_chain_add=_api.llama_sampler_chain_add,
        llama_sampler_init_dist=_api.llama_sampler_init_dist,
        llama_sampler_init_top_k=_api.llama_sampler_init_top_k,
        llama_sampler_apply=_api.llama_sampler_apply,
        llama_sampler_free=_api.llama_sampler_free)


class _FakeChain:
    def __init__(self):
        self.samplers = []
        self.freed = False


def _fake_api(row):
    """dist fills normalised p; top_k sorts by logit and truncates."""
    def apply(chain, ref):
        arr = ref._obj
        n = int(arr.size)
        view = (LlamaTokenData * n).from_address(ctypes.cast(arr.data, ctypes.c_void_p).value)
        for kind, k in chain.samplers:
            if kind == "dist":
                m = max(view[i].logit for i in range(n))
                z = sum(math.exp(view[i].logit - m) for i in range(n))
                for i in range(n):
                    view[i].p = math.exp(view[i].logit - m) / z
            else:
                items = sorted(((view[i].id, view[i].logit, view[i].p) for i in range(n)),
                               key=lambda c: -c[1])[:k]
                for i, (tid, logit, p) in enumerate(items):
                    view[i].id, view[i].logit, view[i].p = tid, logit, p
                arr.size = len(items)
                n = len(items)
    return SimpleNamespace(
        llama_get_logits_ith=row.get,
        llama_sampler_chain_default_params=lambda: SimpleNamespace(no_perf=False),
        llama_sampler_chain_init=lambda params: _FakeChain(),
        llama_sampler_chain_add=lambda chain, s: chain.samplers.append(s),
        llama_sampler_init_dist=lambda seed: ("dist", None),
        llama_sampler_init_top_k=lambda k: ("top_k", k),
        llama_sampler_apply=apply,
        llama_sampler_free=lambda chain: setattr(chain, "freed", True))


def _apis():
    yield pytest.param(_fake_api, id="stand-in")
    yield pytest.param(_native_api, id="native", marks=pytest.mark.skipif(
        native_runtime_lib_path() is None, reason="native llama runtime not provisioned"))


@pytest.mark.parametrize("make_api", list(_apis()))
@pytest.mark.parametrize("n_vocab", [1, 7, 5000])
def test_scores_match_a_softmax_of_the_row(make_api, n_vocab):
    if make_api is _native_api:
        require_native_runtime()
    rng = random.Random(n_vocab)
    logits = [rng.gauss(0, 5) for _ in range(n_vocab)]
    row = _Row(logits)
    want = reference(logits)
    order = sorted(range(n_vocab), key=lambda i: logits[i], reverse=True)
    for n_top in (0, 1, 5, 20):
        scorer = LogprobScorer(make_api(row), n_vocab, n_top)
        for token in {order[0], order[-1], n_vocab // 2}:
            scored = scorer.score("ctx", 3, token)
            assert isinstance(scored, ScoredToken) and scored == token
            assert scored.logprob == pytest.approx(max(want[token], FLOOR_LOGPROB), abs=2e-4)
            assert [t for t, _lp in scored.top] == order[:min(n_top, n_vocab)]
            assert [lp for _t, lp in scored.top] == pytest.approx(
                [want[t] for t in order[:min(n_top, n_vocab)]], abs=2e-4)
        scorer.close()
        scorer.close()
    assert row.reads and all(r == ("ctx", 3) for r in row.reads)


@pytest.mark.parametrize("make_api", list(_apis()))
def test_a_vanishing_probability_reports_the_floor(make_api):
    if make_api is _native_api:
        require_native_runtime()
    row = _Row([0.0, -1e5, 50.0])
    scorer = LogprobScorer(make_api(row), 3, 3)
    assert scorer.score(None, -1, 1).logprob == FLOOR_LOGPROB
    assert scorer.score(None, -1, 2).logprob == pytest.approx(0.0, abs=1e-6)
    scorer.close()


@pytest.mark.parametrize("make_api", list(_apis()))
@pytest.mark.parametrize("row", [
    [0.0, 1.0, 2.0, float("inf")], [float("inf"), 1.0, 2.0, 3.0],
    [0.0, float("nan"), 2.0, 3.0], [float("nan"), 1.0, 2.0, 3.0]],
    ids=["inf-last", "inf-first", "nan-mid", "nan-first"])
def test_a_row_that_is_not_finite_reports_the_floor_and_warns_once(make_api, row, caplog):
    if make_api is _native_api:
        require_native_runtime()
    scorer = LogprobScorer(make_api(_Row(row)), 4, 2)
    with caplog.at_level("WARNING", logger="localm"):
        first = scorer.score(None, 1, 2)
        second = scorer.score(None, 1, 0)
    scorer.close()
    assert (first.logprob, first.top) == (FLOOR_LOGPROB, ())
    assert (second.logprob, second.top) == (FLOOR_LOGPROB, ())
    assert first == 2 and second == 0
    assert caplog.text.count("not a finite number") == 1


@pytest.mark.parametrize("make_api", list(_apis()))
def test_a_minus_infinity_logit_is_a_token_that_cannot_occur(make_api):
    if make_api is _native_api:
        require_native_runtime()
    scorer = LogprobScorer(make_api(_Row([0.0, float("-inf"), 2.0, 3.0])), 4, 2)
    assert scorer.score(None, -1, 1).logprob == FLOOR_LOGPROB
    got = scorer.score(None, -1, 3)
    want = reference([0.0, 2.0, 3.0])
    assert got.logprob == pytest.approx(want[2], abs=1e-5)
    assert got.top[0][0] == 3
    scorer.close()


def test_a_missing_row_and_a_closed_scorer_raise():
    row = _Row([1.0, 2.0])
    api = _fake_api(row)
    api.llama_get_logits_ith = lambda ctx, idx: ctypes.POINTER(ctypes.c_float)()
    scorer = LogprobScorer(api, 2, 1)
    with pytest.raises(RuntimeError, match="no logits for output row 4"):
        scorer.score(None, 4, 0)
    scorer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scorer.score(None, 0, 0)


@pytest.mark.parametrize("n_top", [-1, 21])
def test_an_out_of_range_alternative_count_is_refused(n_top):
    with pytest.raises(ValueError, match="n_top"):
        LogprobScorer(_fake_api(_Row([0.0])), 1, n_top)


def test_tap_records_appends_each_record_before_its_token_passes():
    sink: list = []
    tokens = [ScoredToken(5, -0.5, ((5, -0.5), (6, -1.5))), ScoredToken(6, -0.25, ())]
    gen = tap_records(iter(tokens), lambda t: f"<{t}>".encode(), sink)
    assert next(gen) == 5
    assert sink == [(b"<5>", -0.5, ((b"<5>", -0.5), (b"<6>", -1.5)))]
    assert next(gen) == 6 and len(sink) == 2
    with pytest.raises(RuntimeError, match="did not score"):
        list(tap_records(iter([7]), lambda t: b"", []))


# ------------------------------------------------------------------ plumbing


def _llm(monkeypatch, **overrides):
    from localm.inference.backends.llamacpp import llama as L
    from tests._bare_llama import make_bare_llama
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1, **overrides)
    monkeypatch.setattr(L, "_apply_model_template", lambda model, messages: ("hi", None))
    monkeypatch.setattr(L, "_untrusted_prompt_ranges", lambda *a: ())
    llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): [1, 2]
    llm._tokenizer.token_to_piece_bytes = lambda t: {5: b"Hel", 6: b"lo", 7: b"!"}[t]
    return llm


def _capture(llm, monkeypatch, name, tokens=()):
    seen: dict = {}

    def fake(*args, **kw):
        seen.update(kw)
        yield from tokens
    monkeypatch.setattr(llm, name, fake)
    return seen


@pytest.mark.parametrize("path, overrides, extra", [
    ("_generate", {}, {}),
    ("_generate_encoder_decoder", {"is_encoder_decoder": True}, {}),
    ("_generate_image", {"_mtmd": object()}, {"image": True}),
])
def test_every_generation_path_gets_the_alternative_count(monkeypatch, path, overrides, extra):
    llm = _llm(monkeypatch, **overrides)
    seen = _capture(llm, monkeypatch, path)
    content = ([{"type": "text", "text": "what"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
               if extra else "hi")
    llm.create_chat_completion([{"role": "user", "content": content}], logprobs=3,
                               logprob_sink=[])
    assert seen["logprobs"] == 3
    seen.clear()
    llm.create_chat_completion([{"role": "user", "content": content}])
    assert seen["logprobs"] is None


def test_logprobs_and_their_sink_come_together(monkeypatch):
    llm = _llm(monkeypatch)
    _capture(llm, monkeypatch, "_generate")
    with pytest.raises(ValueError, match="together"):
        llm.create_chat_completion([{"role": "user", "content": "hi"}], logprobs=1)
    with pytest.raises(ValueError, match="together"):
        llm.create_chat_completion([{"role": "user", "content": "hi"}], logprob_sink=[])


def test_a_diffusion_model_refuses_logprobs(monkeypatch):
    from localm.inference.backends.base import UnsupportedInputError
    llm = _llm(monkeypatch, is_diffusion=True)
    seen = _capture(llm, monkeypatch, "_generate_diffusion")
    with pytest.raises(UnsupportedInputError, match="diffusion"):
        llm.create_chat_completion([{"role": "user", "content": "hi"}], logprobs=0,
                                   logprob_sink=[])
    assert seen == {}


def test_a_streamed_completion_fills_the_sink_before_each_text_chunk(monkeypatch):
    llm = _llm(monkeypatch)
    tokens = [ScoredToken(5, -0.1, ((5, -0.1),)), ScoredToken(6, -0.2, ()),
              ScoredToken(7, -0.3, ())]
    _capture(llm, monkeypatch, "_generate", tokens)
    sink: list = []
    seen = []
    for chunk in llm.create_chat_completion([{"role": "user", "content": "hi"}], stream=True,
                                            logprobs=1, logprob_sink=sink):
        text = chunk["choices"][0]["delta"].get("content")
        if text:
            seen.append((text, len(sink)))
    spelled = "".join(t for t, _ in seen)
    assert spelled == "Hello!"
    assert sink == [(b"Hel", -0.1, ((b"Hel", -0.1),)), (b"lo", -0.2, ()), (b"!", -0.3, ())]
    for i, (_text, n) in enumerate(seen):
        assert len("".join(t for t, _ in seen[:i + 1])) <= len(
            b"".join(r[0] for r in sink[:n]).decode())


def test_gguf_backend_reports_logprobs_except_for_a_diffusion_model():
    from unittest.mock import patch

    from localm.inference.backends.base import BaseBackend, UnsupportedInputError
    from localm.inference.backends.gguf import GgufBackend
    from localm.inference.backends.hf import HFBackend
    backend = GgufBackend.__new__(GgufBackend)
    with patch.object(GgufBackend, "is_diffusion", property(lambda self: False)):
        assert backend.supports_logprobs is True
    with patch.object(GgufBackend, "is_diffusion", property(lambda self: True)):
        assert backend.supports_logprobs is False
        backend._grammar_unsupported = False
        with pytest.raises(UnsupportedInputError, match="diffusion"):
            list(backend.chat_stream([{"role": "user", "content": "hi"}], logprobs=0,
                                     on_logprobs=lambda r: None))
    assert BaseBackend.supports_logprobs.fget(object()) is False
    assert HFBackend.supports_logprobs.fget(object()) is False


def test_the_engine_passes_logprobs_only_when_asked():
    from unittest.mock import MagicMock, patch

    from localm.inference.engine import Engine
    backend = MagicMock()
    backend.loaded = True
    backend.supports_logprobs = True
    backend.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    engine = object.__new__(Engine)
    engine.model_path = "m.gguf"
    engine.display_name = "m"
    engine._backend = backend
    cfg = {"max_tokens": 5, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
           "repeat_penalty": 1.1}
    with patch("localm.inference.engine.load_config", return_value=cfg):
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))
        assert not {"logprobs", "on_logprobs"} & set(backend.chat_stream.call_args.kwargs)
        sink = []
        list(engine.chat_stream([{"role": "user", "content": "hi"}], logprobs=0,
                                on_logprobs=sink.extend))
        kw = backend.chat_stream.call_args.kwargs
        assert kw["logprobs"] == 0 and kw["on_logprobs"] == sink.extend
    assert engine.supports_logprobs is True


def test_the_vision_path_scores_every_emitted_token_and_closes_its_scorer():
    from unittest.mock import patch

    from tests.test_generation_boundary_logging import (
        _VISION_MESSAGES, _bare_llama_vision, _mock_native_api)
    llm = _bare_llama_vision()
    calls = []
    llm._tokenizer.is_eog.side_effect = lambda t: len(calls) >= 4

    class _Scorer:
        closed = False

        def score(self, ctx, idx, token):
            calls.append((ctx, idx, token))
            return ScoredToken(token, -float(len(calls)), ())

        def close(self):
            self.closed = True

    scorer = _Scorer()
    llm._logprob_scorer = lambda n: scorer if n is not None else None
    with patch("localm.inference.backends.llamacpp.llama.api", _mock_native_api()), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
        tokens = list(llm._generate_image(
            _VISION_MESSAGES, max_new_tokens=10, temperature=0.8, top_k=40, top_p=0.95,
            repeat_penalty=1.1, logprobs=2))
    assert [t.logprob for t in tokens] == [-1.0, -2.0, -3.0, -4.0]
    assert all(isinstance(t, ScoredToken) for t in tokens)
    assert all(idx == -1 and ctx is llm._ctx_ptr for ctx, idx, _ in calls)
    assert scorer.closed
