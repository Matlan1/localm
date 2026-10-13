# SPDX-License-Identifier: AGPL-3.0-or-later
"""``cache_prompt: false``: from the request down to the KV cache. The reply is
computed from a full prefill into an emptied cache (decode positions start at 0,
no prefix is kept), and on a model with parallel slots it runs inside the slot
scheduler's exclusive section, never in a decode beside another reply."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import WAITING_FOR_MODEL_STATUS
from localm.inference.backends.llamacpp import llama as L
from localm.inference.backends.llamacpp._slots import SlotScheduler
from tests._bare_llama import make_bare_llama
from tests._fake_batch import batch_backing, fake_batch_init
from tests.test_openai_chat_conformance import MODEL, Served, home  # noqa: F401
from tests.test_slot_scheduler import PROMPTS, _drain, _reference, _Sampler, _SimContext

MEM = 333


def _api(decode_log: list, delay: float = 0.0) -> MagicMock:
    """A mocked native api whose decodes record each batch's positions and seq ids."""
    api = MagicMock()
    api.llama_get_memory.return_value = MEM
    api.llama_memory_seq_rm.return_value = True
    api.llama_batch_init.side_effect = fake_batch_init
    api.llama_sampler_sample.return_value = 42

    def decode(_ctx, batch):
        b = batch_backing(batch)
        if delay:
            time.sleep(delay)
        decode_log.append(("native", b["pos"], b["token"]))
        return 0
    api.llama_decode.side_effect = decode
    return api


def _llm(**kw):
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, **kw)
    llm._kv_supported = True
    llm._tokenizer.is_eog.return_value = False
    return llm


# ------------------------------------------------------------------ one slot


@pytest.mark.parametrize("keep, first_pos", [(True, 4), (False, 0)])
def test_prefill_keeps_the_shared_prefix_only_when_asked(keep, first_pos):
    llm = _llm()
    llm._cached_tokens = [1, 2, 3, 4]
    log: list = []
    api = _api(log)
    with patch.object(L, "api", api):
        llm._prefill_with_reuse([1, 2, 3, 4, 5, 6], keep_prefix=keep)
    assert log[0][1][0] == first_pos
    assert [p for _, pos, _ in log for p in pos] == list(range(first_pos, 6))
    if keep:
        api.llama_memory_clear.assert_not_called()
    else:
        api.llama_memory_clear.assert_called_once_with(MEM, True)
    api.llama_memory_seq_rm.assert_not_called()
    assert llm._cached_tokens == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("cache_prompt", [True, False])
def test_generate_without_the_cache_prefills_the_whole_prompt_from_zero(cache_prompt):
    llm = _llm()
    llm._cached_tokens = [1, 2, 3, 9, 9]
    log: list = []
    api = _api(log)
    with patch.object(L, "api", api), patch.object(L, "_build_sampler", return_value=999):
        out = list(llm._generate([1, 2, 3, 4, 5], 2, 0.0, 40, 0.9, 1.0,
                                 cache_prompt=cache_prompt))
    assert out == [42, 42]
    prefill_pos, prefill_tokens = log[0][1], log[0][2]
    if cache_prompt:
        assert prefill_pos == [3, 4] and prefill_tokens == [4, 5]
        api.llama_memory_seq_rm.assert_called_once_with(MEM, 0, 3, -1)
    else:
        assert prefill_pos == [0, 1, 2, 3, 4] and prefill_tokens == [1, 2, 3, 4, 5]
        api.llama_memory_clear.assert_called_once_with(MEM, True)
        assert not [c for c in api.llama_memory_seq_rm.call_args_list if c.args[2] != 0]
    assert [pos for _, pos, _ in log[1:]] == [[5]]


# ------------------------------------------------------------------ parallel slots


class _LoggedSim(_SimContext):
    def __init__(self, log: list, **kw):
        super().__init__(**kw)
        self._log = log

    def decode(self, entries):
        rc = super().decode(entries)
        self._log.append(("slot", sorted({e[2] for e in entries}),
                          [(e[0], e[1]) for e in entries]))
        return rc


def test_slot_mode_runs_the_reply_alone_in_the_exclusive_section():
    log: list = []
    ctx = _LoggedSim(log)
    ctx.step_delay = 0.003
    sched = SlotScheduler(ctx, 4)
    llm = _llm()
    llm._slots = sched
    llm._cached_tokens = [7, 8]
    api = _api(log, delay=0.01)
    try:
        running = sched.submit(PROMPTS[0], 40, _Sampler())
        next(running)
        statuses: list = []
        result: dict = {}

        def alone():
            result["out"] = list(llm._generate(
                [7, 8, 9], 3, 0.0, 40, 0.9, 1.0, on_status=statuses.append,
                cache_prompt=False))

        late: dict = {}
        with patch.object(L, "api", api), patch.object(L, "_build_sampler", return_value=999):
            t = threading.Thread(target=alone, daemon=True)
            t.start()
            deadline = time.monotonic() + 5
            while WAITING_FOR_MODEL_STATUS not in statuses and time.monotonic() < deadline:
                time.sleep(0.005)
            late_t = threading.Thread(
                target=lambda: late.update(r=_drain(sched.submit(PROMPTS[2], 5, _Sampler()))),
                daemon=True)
            late_t.start()
            first = _drain(running)
            t.join(10)
            late_t.join(10)
        assert not t.is_alive() and not late_t.is_alive()
    finally:
        sched.close(timeout=5)

    assert result["out"] == [42, 42, 42]
    assert WAITING_FOR_MODEL_STATUS in statuses
    native = [i for i, e in enumerate(log) if e[0] == "native"]
    assert native == list(range(native[0], native[-1] + 1)), \
        "a slot decode ran inside the exclusive reply"

    def prefill_of(prompt):
        want = [(tok, pos) for pos, tok in enumerate(prompt)]
        return next(i for i, e in enumerate(log)
                    if e[0] == "slot" and all(x in e[2] for x in want))
    assert prefill_of(PROMPTS[0]) < native[0]
    assert sum(1 for e in log[:native[0]] if e[0] == "slot") == 40, \
        "the reply in flight had not ended when the exclusive reply started"
    assert prefill_of(PROMPTS[2]) > native[-1], \
        "a reply admitted while the exclusive reply waited started before it ended"
    assert log[native[0]][1] == [0, 1, 2]
    api.llama_memory_clear.assert_called_with(MEM, True)
    assert first == _reference(PROMPTS[0], 40)
    assert late["r"] == _reference(PROMPTS[2], 5)


def test_slot_mode_with_the_cache_still_goes_to_the_scheduler():
    calls = []

    class _Stream:
        finish_reason = "stop"

        def __iter__(self):
            return iter([5])

        def close(self):
            pass

    class _Slots:
        def submit(self, prompt, budget, sampler, **kw):
            calls.append("submit")
            return _Stream()

        def exclusive(self, on_wait=None):
            raise AssertionError("a cached reply needs no exclusive section")

    llm = _llm()
    llm._slots = _Slots()
    with patch.object(L, "_build_sampler", return_value=object()), patch.object(L, "api"):
        assert list(llm._generate([1, 2], 3, 0.0, 40, 0.9, 1.0)) == [5]
    assert calls == ["submit"]


# ------------------------------------------------------------------ image path


def test_the_image_path_drops_the_vision_record_without_the_cache(monkeypatch):
    llm = _llm()
    seen = {}

    def fake_prefill(vprompt, needed, on_status=None):
        seen["vision_kv"] = llm._vision_kv
        raise RuntimeError("stop here")
    llm._vision_kv = [("x", 1)]
    llm._mtmd = MagicMock()
    llm._mtmd.encode_count = 0
    monkeypatch.setattr(llm, "_prefill_vision", fake_prefill)
    monkeypatch.setattr(L, "_apply_model_template", lambda model, messages: ("p", None))
    monkeypatch.setattr(L.pretokenizer_guard, "check_text", lambda *a: None)
    msgs = [{"role": "user", "content": "hi"}]
    with patch.object(L, "api"), pytest.raises(RuntimeError, match="stop here"):
        list(llm._generate_image(msgs, 3, 0.0, 40, 0.9, 1.0, cache_prompt=False))
    assert seen["vision_kv"] is None


# ------------------------------------------------------------------ the field travels


def _capture(llm, monkeypatch, name):
    seen: dict = {}

    def fake(*args, **kw):
        seen.update(kw)
        yield from ()
    monkeypatch.setattr(llm, name, fake)
    return seen


@pytest.mark.parametrize("given, want", [({}, True), ({"cache_prompt": False}, False)])
def test_create_chat_completion_passes_it_to_the_text_path(monkeypatch, given, want):
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1)
    monkeypatch.setattr(L, "_apply_model_template", lambda model, messages: ("hi", None))
    monkeypatch.setattr(L, "_untrusted_prompt_ranges", lambda *a: ())
    llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): [1, 2]
    seen = _capture(llm, monkeypatch, "_generate")
    llm.create_chat_completion([{"role": "user", "content": "hi"}], **given)
    assert seen["cache_prompt"] is want


def test_create_chat_completion_passes_it_to_the_image_path(monkeypatch):
    llm = make_bare_llama(_model_ptr=1, _ctx_ptr=1)
    llm._mtmd = object()
    monkeypatch.setattr(L, "_apply_model_template", lambda model, messages: ("hi", None))
    monkeypatch.setattr(L, "_untrusted_prompt_ranges", lambda *a: ())
    llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): [1, 2]
    seen = _capture(llm, monkeypatch, "_generate_image")
    image = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    llm.create_chat_completion([image], cache_prompt=False)
    assert seen["cache_prompt"] is False


@pytest.mark.parametrize("given", [{}, {"cache_prompt": True}, {"cache_prompt": False}])
def test_the_gguf_worker_passes_only_a_false_value_to_the_model(given):
    from localm.inference.backends.llamacpp._worker import GgufWorker
    w = GgufWorker("m.gguf", None, 512, 0, None, 512)
    captured: dict = {}

    class _Llm:
        def create_chat_completion(self, **kw):
            captured.update(kw)
            return iter(())
    w._llm = _Llm()
    list(w.chat_stream([{"role": "user", "content": "hi"}], **given))
    if given.get("cache_prompt") is False:
        assert captured["cache_prompt"] is False
    else:
        assert "cache_prompt" not in captured


@pytest.mark.parametrize("given", [{}, {"cache_prompt": True}, {"cache_prompt": False}])
def test_the_gguf_backend_sends_only_a_false_value_to_the_worker(tmp_path, given):
    from localm.inference.backends.gguf import GgufBackend
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    b = GgufBackend(str(path))
    runner = MagicMock()
    runner.chat_stream.side_effect = lambda **kw: iter(["x"])
    b._runner = runner
    list(b.chat_stream([{"role": "user", "content": "hi"}], **given))
    kw = runner.chat_stream.call_args.kwargs
    if given.get("cache_prompt") is False:
        assert kw["cache_prompt"] is False
    else:
        assert "cache_prompt" not in kw


@pytest.mark.parametrize("given, sent", [({}, False), ({"cache_prompt": None}, False),
                                         ({"cache_prompt": True}, False),
                                         ({"cache_prompt": False}, True)])
def test_the_engine_passes_only_a_false_value_to_the_backend(given, sent):
    from localm.inference.engine import Engine
    engine = object.__new__(Engine)
    engine.model_path = "m.gguf"
    engine.display_name = "m"
    backend = MagicMock()
    backend.loaded = True
    backend.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    engine._backend = backend
    cfg = {"max_tokens": 5, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
           "repeat_penalty": 1.1}
    with patch("localm.inference.engine.load_config", return_value=cfg):
        list(engine.chat_stream([{"role": "user", "content": "hi"}], **given))
    kw = backend.chat_stream.call_args.kwargs
    assert ("cache_prompt" in kw) is sent
    if sent:
        assert kw["cache_prompt"] is False


def test_the_hf_backend_accepts_it():
    import inspect

    from localm.inference.backends.hf import HFBackend
    assert inspect.signature(HFBackend.chat_stream).parameters["cache_prompt"].default is True


@pytest.mark.parametrize("given", [{}, {"cache_prompt": False}])
def test_the_http_engine_forwards_it_when_set(given):
    from localm.inference.http_engine import HttpEngine
    engine = object.__new__(HttpEngine)
    engine._model = "m"
    engine.display_name = "m"
    engine._base = "http://127.0.0.1:1/v1"
    engine._token = None
    engine.pin_model = True
    sent = {}

    class _Resp:
        status_code = 500
        text = "no"

        def json(self):
            return {"detail": "no"}

    def post(url, json=None, **kw):
        sent.update(json)
        return _Resp()
    with patch("requests.post", post), pytest.raises(RuntimeError):
        list(engine.chat_stream([{"role": "user", "content": "hi"}], **given))
    assert sent.get("cache_prompt") == given.get("cache_prompt")


# ------------------------------------------------------------------ routes


@pytest.mark.parametrize("route", ["chat", "complete"])
@pytest.mark.parametrize("given, want", [({}, None), ({"cache_prompt": True}, None),
                                         ({"cache_prompt": False}, False)])
def test_the_routes_pass_only_a_false_value_to_the_engine(home, route, given, want):  # noqa: F811
    served = Served()
    r = getattr(served, route)(**given)
    assert r.status_code == 200, r.text
    assert served.kwargs.get("cache_prompt") is want
    if want is None:
        assert "cache_prompt" not in served.kwargs


@pytest.mark.parametrize("route", ["chat", "complete"])
@pytest.mark.parametrize("bad", ["nope", "false", "true", 0, 1, 0.0, [], {}, None, 2])
def test_a_cache_prompt_that_is_not_a_boolean_is_a_422(home, route, bad):  # noqa: F811
    served = Served()
    r = getattr(served, route)(cache_prompt=bad)
    assert r.status_code == 422, r.text
    assert "cache_prompt" in r.text
    assert not served.calls


def test_the_responses_route_passes_it_to_the_chat_route(home):  # noqa: F811
    served = Served()
    r = served.client.post("/v1/responses", json={"model": MODEL, "input": "hi",
                                                  "cache_prompt": False})
    assert r.status_code == 200, r.text
    assert served.kwargs["cache_prompt"] is False


def test_the_messages_route_passes_it_to_the_chat_route(home):  # noqa: F811
    served = Served()
    r = served.client.post("/v1/messages", json={
        "model": MODEL, "max_tokens": 5, "cache_prompt": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert served.kwargs["cache_prompt"] is False


@pytest.mark.parametrize("path, body", [
    ("/v1/responses", {"model": MODEL, "input": "hi", "cache_prompt": "false"}),
    ("/v1/messages", {"model": MODEL, "max_tokens": 5, "cache_prompt": 0,
                      "messages": [{"role": "user", "content": "hi"}]}),
])
def test_the_other_wire_formats_refuse_a_wrong_type(home, path, body):  # noqa: F811
    served = Served()
    r = served.client.post(path, json=body)
    assert r.status_code == 400, r.text
    assert "cache_prompt" in r.text
    assert not served.calls
