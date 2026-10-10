# SPDX-License-Identifier: AGPL-3.0-or-later
"""How many parallel slots a GGUF load asks for and gets: the parallel_slots
setting, the parent's request (auto sizing against VRAM for recurrent state),
what LlamaCpp holds for a given model, and how the load reaches the worker."""
from __future__ import annotations

import ctypes
import struct
from unittest.mock import PropertyMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp._sizing import _AutoLayerBudget
from localm.inference.parallel_setting import (
    PARALLEL_AUTO, PARALLEL_AUTO_SLOTS, coerce_parallel_slots, resolve_parallel_slots)
from tests._bare_llama import make_bare_llama

_T_STRING = 8


def _gguf(path, arch="qwen3"):
    raw = arch.encode()
    key = b"general.architecture"
    body = (struct.pack("<Q", len(key)) + key + struct.pack("<I", _T_STRING)
            + struct.pack("<Q", len(raw)) + raw)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 1) + body + b"\0" * 4096)
    return path


# ------------------------------------------------------------------ setting


@pytest.mark.parametrize("raw, want", [
    ("auto", PARALLEL_AUTO), (" AUTO ", PARALLEL_AUTO), (1, 1), (16, 16), ("4", 4),
    (0, None), (17, None), (True, None), ("four", None), (None, None), (2.0, None)])
def test_coerce_parallel_slots(raw, want):
    assert coerce_parallel_slots(raw) == want


def test_an_invalid_setting_reads_as_auto_and_says_so(caplog):
    assert resolve_parallel_slots({"parallel_slots": "lots"}) == PARALLEL_AUTO
    assert "parallel_slots" in caplog.text
    assert resolve_parallel_slots({}) == PARALLEL_AUTO
    assert resolve_parallel_slots({"parallel_slots": ""}) == PARALLEL_AUTO
    assert resolve_parallel_slots({"parallel_slots": 8}) == 8


def test_the_backend_refuses_an_invalid_setting(tmp_path):
    with pytest.raises(ValueError):
        GgufBackend(str(_gguf(tmp_path / "m.gguf")), parallel_slots=0)


# ------------------------------------------------------------------ parent request


def _backend(tmp_path, **kw):
    return GgufBackend(str(_gguf(tmp_path / "m.gguf")), **kw)


def _with_state(per_copy):
    return patch("localm.model_manager.gguf.gguf_recurrent_state_bytes",
                 return_value=per_copy)


def _budget(layers=99, free=10_000, model=4_000, kv=1_000, overhead=1_000):
    return _AutoLayerBudget(layers, free, free * 2, model, kv, overhead, 1)


def test_auto_asks_for_four_slots_for_a_model_without_recurrent_state(tmp_path):
    b = _backend(tmp_path, parallel_slots="auto")
    with _with_state(0):
        assert b._requested_parallel_slots() == PARALLEL_AUTO_SLOTS


@pytest.mark.parametrize("source", ["mtp", "ngram"])
def test_a_draft_source_keeps_one_slot_even_when_asked_for_more(tmp_path, source):
    b = _backend(tmp_path, parallel_slots=8, spec_source=source)
    assert b._requested_parallel_slots() == 1


def test_a_diffusion_model_keeps_one_slot(tmp_path):
    b = _backend(tmp_path, parallel_slots=8)
    with patch.object(GgufBackend, "is_diffusion", new_callable=PropertyMock,
                      return_value=True):
        assert b._requested_parallel_slots() == 1


def test_an_explicit_count_is_asked_for_as_given(tmp_path):
    with _with_state(10**12):
        assert _backend(tmp_path, parallel_slots=8)._requested_parallel_slots() == 8
        assert _backend(tmp_path, parallel_slots=1)._requested_parallel_slots() == 1


@pytest.mark.parametrize("headroom_copies, want", [(3, 4), (2, 2), (1, 2), (0, 1)])
def test_auto_fits_extra_recurrent_state_copies_in_the_vram_left(tmp_path,
                                                                headroom_copies, want):
    per_copy = 500
    b = _backend(tmp_path, parallel_slots="auto")
    # model + kv + overhead (the overhead already holds one copy) leave
    # headroom_copies copies free.
    budget = _budget(free=6_000 + headroom_copies * per_copy, model=4_000, kv=1_000,
                     overhead=1_000)
    with _with_state(per_copy), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None), \
         patch.object(GgufBackend, "_auto_gpu_layers_budget", return_value=budget):
        assert b._requested_parallel_slots() == want


@pytest.mark.parametrize("budget", [None, _budget(layers=20)])
def test_auto_keeps_one_slot_for_recurrent_state_when_vram_is_short_or_unknown(tmp_path,
                                                                              budget):
    b = _backend(tmp_path, parallel_slots="auto")
    with _with_state(500), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None), \
         patch.object(GgufBackend, "_auto_gpu_layers_budget", return_value=budget):
        assert b._requested_parallel_slots() == 1


def test_a_cpu_only_load_counts_as_fitting(tmp_path):
    b = _backend(tmp_path, parallel_slots="auto", n_gpu_layers=0)
    with _with_state(500), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None), \
         patch.object(GgufBackend, "_auto_gpu_layers_budget", side_effect=AssertionError):
        assert b._requested_parallel_slots() == PARALLEL_AUTO_SLOTS


def test_the_recurrent_state_charge_counts_one_copy_per_slot(tmp_path):
    b = _backend(tmp_path)
    with _with_state(1000), \
         patch.object(GgufBackend, "_gguf_parsed_tensor_entries", return_value=None):
        assert b._recurrent_state_vram_bytes() == 1000
        b._set_parallel_copies(4)
        assert b._recurrent_state_vram_bytes() == 4000
        b._set_parallel_copies(1)
        assert b._recurrent_state_vram_bytes() == 1000


class TestLoadReachesTheWorker:
    def _load(self, b, meta):
        from localm.config import DEFAULT_CONFIG
        with patch("localm.config.load_config", return_value=dict(DEFAULT_CONFIG)), \
             patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load", return_value=meta) as spawn:
            b.effective_gpu_layers = 0
            b._load_native()
        return spawn.call_args[0][0]

    def test_slots_are_sent_and_the_reported_count_is_kept(self, tmp_path):
        b = _backend(tmp_path)
        b._set_parallel_copies(4)
        params = self._load(b, {"parallel_slots": 2, "parallel_note": "why"})
        assert params["n_parallel"] == 4
        assert (b.parallel_slots, b.parallel_note) == (2, "why")

    def test_one_slot_sends_nothing_and_an_old_worker_reads_as_one(self, tmp_path):
        b = _backend(tmp_path)
        params = self._load(b, {})
        assert "n_parallel" not in params
        assert b.parallel_slots == 1


# ------------------------------------------------------------------ LlamaCpp


def _llm(**kw):
    llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), **kw)
    llm.is_encoder_decoder = False
    llm.is_diffusion = False
    llm._spec_source_name = "off"
    llm._kv_supported = True
    return llm


def test_llamacpp_holds_the_requested_slots():
    assert _llm()._resolve_parallel(4, 4096) == (4, "")


def test_llamacpp_holds_one_slot_unless_asked():
    assert _llm()._resolve_parallel(1, 4096) == (1, "")
    assert _llm()._resolve_parallel("x", 4096) == (1, "")


@pytest.mark.parametrize("attr, value, word", [
    ("is_encoder_decoder", True, "encoder-decoder"),
    ("is_diffusion", True, "diffusion"),
    ("_spec_source_name", "mtp", "drafting"),
    ("_kv_supported", False, "sequences")])
def test_llamacpp_keeps_one_slot_for_models_that_cannot_share(attr, value, word):
    llm = _llm()
    setattr(llm, attr, value)
    slots, note = llm._resolve_parallel(4, 4096)
    assert slots == 1 and word in note


def test_llamacpp_lowers_slots_until_they_divide_the_batch():
    slots, note = _llm()._resolve_parallel(3, 4096)
    assert slots == 2 and "3" in note
    assert _llm()._resolve_parallel(5, 1000) == (5, "")


def test_context_params_get_one_shared_cache_only_with_slots():
    class _Cp:
        n_seq_max = 1
        kv_unified = False

    llm = _llm()
    llm.n_parallel = 4
    cp = _Cp()
    llm._apply_parallel_params(cp)
    assert (cp.n_seq_max, cp.kv_unified) == (4, True)
    llm.n_parallel = 1
    cp = _Cp()
    llm._apply_parallel_params(cp)
    assert (cp.n_seq_max, cp.kv_unified) == (1, False)


def test_the_finish_reason_is_kept_per_thread_with_slots():
    import threading
    llm = _llm()
    llm._slots = object()
    llm.last_finish_reason = "length"
    seen = {}

    def other():
        seen["before"] = llm.last_finish_reason
        llm.last_finish_reason = "stop"
        seen["after"] = llm.last_finish_reason

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert (seen["before"], seen["after"]) == ("length", "stop")
    assert llm.last_finish_reason == "length"


def test_without_slots_the_finish_reason_is_one_shared_value():
    import threading
    llm = _llm()
    llm.last_finish_reason = "length"
    t = threading.Thread(target=lambda: setattr(llm, "last_finish_reason", "error"))
    t.start()
    t.join()
    assert llm.last_finish_reason == "error"


def test_text_generation_goes_to_the_slot_scheduler_when_there_is_one():
    calls = []

    class _Stream:
        finish_reason = "length"

        def __iter__(self):
            return iter([5, 6])

        def close(self):
            calls.append("close")

    class _Slots:
        def submit(self, prompt, budget, sampler, **kw):
            calls.append(("submit", list(prompt), budget))
            return _Stream()

    llm = _llm()
    llm._slots = _Slots()
    llm._n_ctx_max = None
    with patch.object(llama_mod, "_build_sampler", return_value=object()) as build,          patch.object(llama_mod, "api") as api:
        api.llama_decode.return_value = 1
        try:
            out = list(llm._generate([1, 2, 3], 10, 0.0, 40, 0.9, 1.0,
                                     sampling={"min_p": 0.2, "penalty_freq": 0.5}))
        except Exception as exc:
            out = exc
    assert calls == [("submit", [1, 2, 3], 10), "close"]
    assert out == [5, 6]
    assert build.call_args.kwargs["min_p"] == 0.2
    assert build.call_args.kwargs["penalty_freq"] == 0.5
    assert llm.last_finish_reason == "length"


def test_the_backends_reply_results_are_kept_per_thread(tmp_path):
    import threading
    b = _backend(tmp_path)
    b.parallel_slots = 2
    b.last_finish_reason = "length"
    b.last_mtp_drafted = 5
    b.last_speculation = {"source": "ngram"}
    seen = {}

    def other():
        seen["latest"] = (b.last_finish_reason, b.last_mtp_drafted, b.last_speculation)
        b.last_finish_reason = "stop"
        b.last_mtp_drafted = 0
        b.last_speculation = None

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen["latest"] == ("length", 5, {"source": "ngram"})
    assert (b.last_finish_reason, b.last_mtp_drafted, b.last_speculation) == (
        "length", 5, {"source": "ngram"})
    assert _backend(tmp_path).last_finish_reason == "stop"


def test_the_engine_reports_slots_only_from_a_positive_int():
    from localm.inference.engine import Engine

    class _B:
        parallel_slots = 4

    eng = Engine.__new__(Engine)
    eng._backend = _B()
    assert eng.parallel_slots == 4
    for bad in (0, True, "4", None):
        _B.parallel_slots = bad
        assert eng.parallel_slots == 1


def test_a_load_payload_names_the_slots_only_when_there_are_several():
    from localm.inference.http_server import _gpu_placement_fields

    class _Eng:
        gpu_placement = None
        mmap_state = None
        applied_adapters = None
        parallel_slots = 1

    assert "parallel_slots" not in _gpu_placement_fields(_Eng())
    _Eng.parallel_slots = 4
    assert _gpu_placement_fields(_Eng())["parallel_slots"] == 4


def test_a_failed_stream_leaves_a_newer_runner_alone(tmp_path):
    b = _backend(tmp_path)
    shut = []

    class _Old:
        def chat_stream(self, **kw):
            b._runner = new          # another request reloaded meanwhile
            raise RuntimeError("worker died")
            yield

        def shutdown(self, grace=5.0):
            shut.append("old")

    class _New:
        def shutdown(self, grace=5.0):
            shut.append("new")

        def is_alive(self):
            return True

    new = _New()
    b._runner = _Old()
    b._loaded = True
    with pytest.raises(RuntimeError):
        list(b.chat_stream([{"role": "user", "content": "hi"}]))
    assert b._runner is new and b._loaded
    assert shut == ["old"]
