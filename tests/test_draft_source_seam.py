# SPDX-License-Identifier: AGPL-3.0-or-later
"""The decode loop drives any DraftSource, not only MTP.

A scripted source proposes a fixed number of tokens per step, some of them
wrong, through the FakeNative runtime of test_mtp_drafting: the reply must be
the target's own reply, the request sampler must see only emitted tokens, and
the loop must call the source's hooks in the documented order.
"""

import ctypes

import pytest

from localm.inference.backends.llamacpp._drafting import DraftSource, MtpSource
from tests._bare_llama import make_bare_llama
from tests.test_mtp_drafting import EOG, PROMPT, FakeNative, _generate, _reference, next_token


class ScriptedSource(DraftSource):
    """Proposes the target's continuation, n_max tokens long, with the token
    for every position in *wrong* moved by one."""

    name = "scripted"

    def __init__(self, llm, *, n=3, wrong=(), raise_at=None, ready=True):
        self.llm, self.n, self.wrong, self.raise_at = llm, n, set(wrong), raise_at
        self._ready = ready
        self.calls = []
        self.on = False
        self.drafted = self.accepted = self.steps = self.paused = 0
        self.status = ""

    def begin_call(self):
        self.calls.append("begin_call")
        self.on = True
        return True

    def end_call(self):
        self.calls.append("end_call")

    def drafting(self):
        return self.on

    def ready(self, pos):
        return self._ready

    def budget(self, pos, tokens_left):
        n = self.n if tokens_left is None else min(self.n, tokens_left)
        return max(0, min(n, self.llm._ctx_capacity - pos - 1))

    def propose(self, token, pos, n_max):
        self.calls.append(("propose", pos))
        if self.raise_at is not None and pos >= self.raise_at:
            raise ValueError("scripted failure")
        out, t = [], token
        for i in range(n_max):
            t = next_token(t)
            out.append(t + 1 if (pos + 1 + i) in self.wrong else t)
        return out

    def after_verify(self, accepted, pos):
        self.calls.append(("after_verify", list(accepted), pos))

    def after_single_token(self, token, pos):
        self.calls.append(("after_single_token", token, pos))

    def finish(self):
        self.calls.append("finish")

    def stop_this_call(self, status):
        self.on = False
        self.status = status

    def on_verify(self, drafted, accepted):
        self.steps += 1
        self.drafted += drafted
        self.accepted += accepted

    def on_paused_step(self):
        self.paused += 1


def _llama_with(source_factory):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2))
    llm._tokenizer.is_eog.side_effect = lambda t: t == EOG
    llm._fit_generation_budget = lambda n_prompt, max_new: max_new
    llm._can_reuse_kv = lambda needed: True
    llm._source = source_factory(llm)
    return llm


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("wrong", [(), (8, 9, 14), tuple(range(7, 40, 3)), tuple(range(7, 40))])
def test_any_source_gives_the_target_alone_reply(n, wrong):
    llm = _llama_with(lambda llm: ScriptedSource(llm, n=n, wrong=wrong))
    fake = FakeNative(llm)

    tokens, mock_api = _generate(llm, fake, max_new_tokens=24)

    assert tokens == _reference(PROMPT, 24)
    assert fake.main_accepted == tokens
    main = [fake.main_cache[p] for p in sorted(fake.main_cache)]
    assert main == (PROMPT + tokens)[:len(main)]
    src = llm._source
    assert src.steps > 0 and src.drafted >= src.accepted
    mock_api.llama_sampler_accept.assert_not_called()


def test_the_hooks_run_in_the_documented_order():
    llm = _llama_with(lambda llm: ScriptedSource(llm, n=2, wrong=(8,)))
    fake = FakeNative(llm)

    _generate(llm, fake, max_new_tokens=6)

    calls = llm._source.calls
    assert calls[0] == "begin_call"
    assert calls[-2:] == ["finish", "end_call"]
    # Step at 6 drafts 7, 8: 7 accepted, 8 rejected.
    assert calls[1] == ("propose", len(PROMPT))
    assert calls[2] == ("after_verify", [_reference(PROMPT, 2)[1]], len(PROMPT))


def test_a_source_that_raises_stops_drafting_and_the_reply_completes():
    llm = _llama_with(lambda llm: ScriptedSource(llm, n=2, raise_at=10))
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=20)

    assert tokens == _reference(PROMPT, 20)
    src = llm._source
    assert src.status == "draft-decode-error:ValueError"
    late = [c for c in src.calls if isinstance(c, tuple) and c[0] == "propose" and c[1] >= 10]
    assert len(late) == 1, "the source was asked again after it failed"
    assert src.calls[-1] == "end_call"


def test_a_source_that_is_never_ready_decodes_one_token_at_a_time():
    llm = _llama_with(lambda llm: ScriptedSource(llm, n=3, ready=False))
    fake = FakeNative(llm)

    tokens, _ = _generate(llm, fake, max_new_tokens=10)

    assert tokens == _reference(PROMPT, 10)
    assert all(len(toks) == 1 for _p, toks in fake.main_decodes[1:])
    singles = [c for c in llm._source.calls if isinstance(c, tuple) and c[0] == "after_single_token"]
    assert len(singles) == 10


def test_end_call_runs_when_the_consumer_abandons_the_reply():
    from unittest.mock import patch
    llm = _llama_with(lambda llm: ScriptedSource(llm, n=2))
    fake = FakeNative(llm)
    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=fake.main_sampler):
        fake.install(mock_api)
        gen = llm._generate(prompt_tokens=list(PROMPT), max_new_tokens=20,
                            temperature=0.0, top_k=40, top_p=0.95, repeat_penalty=1.0)
        next(gen)
        next(gen)
        gen.close()

    assert llm._source.calls[-1] == "end_call"
    assert "finish" not in llm._source.calls


def test_the_base_source_never_drafts():
    src = DraftSource()
    assert src.begin_call() is False
    assert src.drafting() is False
    assert src.propose(1, 0, 4) == []
    assert src.extra_vram_bytes() == 0


def test_a_model_holding_its_mtp_source_is_finalized_when_dropped():
    import gc
    import weakref
    from localm.inference.backends.llamacpp.llama import LlamaCpp

    llm = LlamaCpp.__new__(LlamaCpp)
    llm.close = lambda: None
    llm._draft_source()
    gone = weakref.ref(llm)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del llm
        assert gone() is None, "the model and its draft source form a reference cycle"
    finally:
        if was_enabled:
            gc.enable()


def test_an_mtp_model_gets_the_mtp_source_once():
    llm = make_bare_llama()
    first = llm._draft_source()
    assert isinstance(first, MtpSource)
    assert llm._draft_source() is first
