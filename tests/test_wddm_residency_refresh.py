# SPDX-License-Identifier: AGPL-3.0-or-later
"""On Windows, a model fully on a GPU whose process the WDDM counters show
holding a large share of its GPU memory in shared system memory, while enough
VRAM is free to hold it, gets the context recreated, same size and placement,
before the next request."""

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from localm import gpu_usage
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init

API = "localm.inference.backends.llamacpp.llama.api"
READ = "localm.gpu_usage.process_gpu_memory"
GB = 1024 ** 3


def _llm(capacity=4096, *, offload_kqv=True, free=6 * GB):
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _n_ctx=4096,
                          _n_ctx_max=65536, _n_ctx_grow=4096,
                          _ctx_capacity=capacity, _offload_kqv=offload_kqv,
                          _kv_supported=True)
    llm._tokenizer.is_eog.return_value = False
    llm._vram_free = MagicMock(return_value=free)
    return llm


def _api():
    api = MagicMock()
    cp = MagicMock()
    api.llama_context_default_params.return_value = cp
    api.llama_init_from_model.return_value = 444
    api.llama_decode.return_value = 0
    api.llama_batch_init.side_effect = fake_batch_init
    api.llama_sampler_sample.return_value = 42
    api.llama_memory_seq_rm.return_value = True
    return api, cp


def _generate(llm, api, reading, prompt=(1, 2, 3), n=4):
    read = MagicMock(return_value=reading)
    with patch(API, api), patch(READ, read), patch.object(sys, "platform", "win32"), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
        out = list(llm._generate(prompt_tokens=list(prompt), max_new_tokens=n,
                                 temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0))
    return out, read.call_count


DEMOTED = (int(9.78 * GB), int(3.90 * GB))
HEALTHY = (int(11.11 * GB), int(0.78 * GB))


def _inits(api):
    return api.llama_init_from_model.call_count


class TestDetection:
    def test_a_demoted_process_gets_its_context_recreated_before_the_request(self, caplog):
        llm = _llm(free=int(4.2 * GB))
        api, cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            out, reads = _generate(llm, api, DEMOTED)
        assert out == [42] * 4
        assert reads == 1
        assert _inits(api) == 1
        assert cp.n_ctx == 4096 and cp.offload_kqv is True
        assert llm._ctx_ptr == 444 and llm._residency_refresh is None
        lines = [r.getMessage() for r in caplog.records if "recreating the" in r.getMessage()]
        assert len(lines) == 1
        assert ("Windows holds 3.90 GB of this process's GPU memory in shared system "
                "memory (9.78 GB in VRAM) while 4.20 GB of VRAM is free") in lines[0]

    def test_the_reading_after_the_refresh_is_logged_with_the_one_before(self, caplog):
        llm = _llm()
        api, _cp = _api()
        _generate(llm, api, DEMOTED)
        with caplog.at_level(logging.INFO, logger="localm"):
            _out, reads = _generate(llm, api, HEALTHY)
        assert reads == 1
        assert _inits(api) == 1
        lines = [r.getMessage() for r in caplog.records if "after the refresh" in r.getMessage()]
        assert lines == ["gguf context: after the refresh Windows holds 0.78 GB of this "
                         "process's GPU memory in shared system memory (3.90 GB before)"]

    def test_a_healthy_process_keeps_its_context(self):
        llm = _llm()
        api, _cp = _api()
        _out, reads = _generate(llm, api, HEALTHY)
        assert reads == 1 and _inits(api) == 0
        llm._vram_free.assert_not_called()

    def test_shared_under_the_minimum_is_left_alone(self):
        llm = _llm()
        api, _cp = _api()
        _generate(llm, api, (int(2 * GB), llm._RESIDENCY_SHARED_MIN - 1))
        assert _inits(api) == 0

    def test_shared_under_the_share_of_dedicated_is_left_alone(self):
        llm = _llm()
        api, _cp = _api()
        dedicated = 20 * GB
        shared = int(llm._RESIDENCY_SHARED_RATIO * dedicated) - 1
        assert shared > llm._RESIDENCY_SHARED_MIN
        _generate(llm, api, (dedicated, shared))
        assert _inits(api) == 0

    def test_too_little_free_vram_is_left_alone(self):
        llm = _llm(free=int(3.89 * GB))
        api, _cp = _api()
        _generate(llm, api, DEMOTED)
        assert _inits(api) == 0
        assert llm._residency_refresh is None

    @pytest.mark.parametrize("free, said", [
        (None, "free VRAM is unmeasurable"),
        (RuntimeError("probe failed"), "the free VRAM reading raised RuntimeError")])
    def test_unmeasurable_free_vram_is_left_alone(self, free, said, caplog):
        llm = _llm()
        if isinstance(free, Exception):
            llm._vram_free.side_effect = free
        else:
            llm._vram_free.return_value = None
        api, _cp = _api()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            out, _reads = _generate(llm, api, DEMOTED)
        assert out == [42] * 4
        assert _inits(api) == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any(said in m for m in messages)
        assert not any("raised" in m and "keeping the live context" in m for m in messages)

    def test_no_free_vram_source_is_left_alone(self):
        llm = _llm()
        llm._vram_free = None
        api, _cp = _api()
        _generate(llm, api, DEMOTED)
        assert _inits(api) == 0

    def test_unavailable_counters_are_said_once_at_debug(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _generate(llm, api, None)
            llm._residency_check_after = 0.0
            _generate(llm, api, None)
        assert _inits(api) == 0
        lines = [r for r in caplog.records if "counters are unavailable" in r.getMessage()]
        assert len(lines) == 1 and lines[0].levelno == logging.DEBUG


class TestGating:
    def test_other_platforms_never_read_the_counters(self):
        llm = _llm()
        api, _cp = _api()
        read = MagicMock(return_value=DEMOTED)
        with patch(API, api), patch(READ, read), patch.object(sys, "platform", "linux"), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
            list(llm._generate(prompt_tokens=[1, 2, 3], max_new_tokens=4, temperature=0.0,
                               top_k=1, top_p=1.0, repeat_penalty=1.0))
        assert read.call_count == 0 and _inits(api) == 0

    def test_a_load_not_fully_on_a_gpu_is_not_checked(self):
        llm = _llm()
        llm._decode_judged = False
        api, _cp = _api()
        _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 0 and _inits(api) == 0

    def test_a_kv_cache_in_ram_is_not_checked(self):
        llm = _llm(offload_kqv=False)
        api, _cp = _api()
        _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 0

    def test_the_counters_are_read_at_most_once_a_minute(self):
        llm = _llm()
        api, _cp = _api()
        _out, first = _generate(llm, api, HEALTHY)
        _out, second = _generate(llm, api, HEALTHY)
        assert (first, second) == (1, 0)
        llm._residency_check_after -= llm._RESIDENCY_RECHECK_S + 1
        _out, third = _generate(llm, api, HEALTHY)
        assert third == 1

    def test_a_recreated_context_makes_the_check_due_again(self):
        llm = _llm()
        api, _cp = _api()
        _generate(llm, api, HEALTHY)
        _out, reads = _generate(llm, api, HEALTHY, prompt=range(1, 5000))
        assert _inits(api) == 1 and reads == 0
        _out, reads = _generate(llm, api, HEALTHY)
        assert reads == 1


class TestBackoff:
    def test_a_second_refresh_waits_out_the_backoff(self):
        llm = _llm()
        api, _cp = _api()
        _generate(llm, api, DEMOTED)
        assert _inits(api) == 1
        _generate(llm, api, DEMOTED)
        llm._residency_check_after = 0.0
        _generate(llm, api, DEMOTED)
        assert _inits(api) == 1
        llm._decode_refresh_after -= llm._DECODE_REFRESH_BACKOFF_S + 1
        llm._residency_check_after = 0.0
        _generate(llm, api, DEMOTED)
        assert _inits(api) == 2
        assert llm._decode_refresh_backoff == 2 * llm._DECODE_REFRESH_BACKOFF_S

    def test_a_pending_decode_rate_refresh_is_not_doubled(self, caplog):
        llm = _llm()
        llm._decode_refresh = (5.0, 50.0)
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 0 and _inits(api) == 1
        assert any("5.0 steps/s where 50.0 was expected" in r.getMessage()
                   for r in caplog.records)


class TestSlotSeam:
    def test_the_slot_seam_plans_a_same_size_refresh(self):
        from localm.inference.backends.llamacpp._slots import LlamaSlotOps
        llm = _llm(capacity=8192)
        with patch(READ, return_value=DEMOTED), patch.object(sys, "platform", "win32"):
            assert LlamaSlotOps(llm).rightsize(100, 5000) == (8192, True)
        assert llm._residency_refresh is not None


def _fake_pdh(arrays, fail=None):
    pdh = types.ModuleType("win32pdh")
    pdh.PDH_FMT_LARGE = 0x400
    pdh.opened = []
    pdh.closed = []
    pdh.paths = []

    def open_query():
        q = object()
        pdh.opened.append(q)
        return q
    pdh.OpenQuery = open_query
    pdh.CloseQuery = pdh.closed.append

    def add(query, path):
        pdh.paths.append(path)
        return path.rsplit("\\", 1)[1]
    pdh.AddEnglishCounter = add

    def collect(query):
        if fail:
            raise fail.pop(0)
    pdh.CollectQueryData = collect
    pdh.GetFormattedCounterArray = lambda h, fmt: dict(arrays[h])
    return pdh


@pytest.fixture
def fresh_proc_state():
    saved = gpu_usage._pdh_proc_state
    gpu_usage._pdh_proc_state = None
    yield
    gpu_usage._pdh_proc_state = saved


@pytest.mark.usefixtures("fresh_proc_state")
class TestProcessGpuMemory:
    def test_sums_the_adapters_holding_vram_and_skips_the_rest(self):
        pdh = _fake_pdh({
            "Dedicated Usage": {"pid_7_luid_a_phys_0": 9 * GB, "pid_7_luid_b_phys_0": 2 * GB,
                                "pid_7_luid_c_phys_0": 1024},
            "Shared Usage": {"pid_7_luid_a_phys_0": 3 * GB, "pid_7_luid_b_phys_0": GB,
                             "pid_7_luid_c_phys_0": 5 * GB},
        })
        with patch.dict(sys.modules, {"win32pdh": pdh}), patch.object(sys, "platform", "win32"):
            assert gpu_usage.process_gpu_memory(7) == (11 * GB, 4 * GB)
            assert gpu_usage.process_gpu_memory(7) == (11 * GB, 4 * GB)
        assert len(pdh.opened) == 1
        assert pdh.paths == ["\\GPU Process Memory(pid_7_*)\\Dedicated Usage",
                             "\\GPU Process Memory(pid_7_*)\\Shared Usage"]

    def test_no_adapter_holding_vram_is_none(self):
        pdh = _fake_pdh({"Dedicated Usage": {"pid_7_luid_c_phys_0": 1024},
                         "Shared Usage": {"pid_7_luid_c_phys_0": 5 * GB}})
        with patch.dict(sys.modules, {"win32pdh": pdh}), patch.object(sys, "platform", "win32"):
            assert gpu_usage.process_gpu_memory(7) is None

    def test_a_failed_query_is_none_and_reopened_next_call(self, caplog):
        pdh = _fake_pdh({"Dedicated Usage": {"pid_7_l": 9 * GB}, "Shared Usage": {"pid_7_l": GB}},
                        fail=[RuntimeError("No data to return.")])
        with patch.dict(sys.modules, {"win32pdh": pdh}), patch.object(sys, "platform", "win32"), \
             caplog.at_level(logging.DEBUG, logger="localm"):
            assert gpu_usage.process_gpu_memory(7) is None
            assert gpu_usage.process_gpu_memory(7) == (9 * GB, GB)
        assert len(pdh.opened) == 2 and len(pdh.closed) == 1
        assert any("GPU Process Memory query for pid 7 failed" in r.getMessage()
                   for r in caplog.records)

    def test_another_pid_opens_its_own_query(self):
        pdh = _fake_pdh({"Dedicated Usage": {"x": 9 * GB}, "Shared Usage": {"x": GB}})
        with patch.dict(sys.modules, {"win32pdh": pdh}), patch.object(sys, "platform", "win32"):
            gpu_usage.process_gpu_memory(7)
            gpu_usage.process_gpu_memory(8)
        assert len(pdh.opened) == 2 and len(pdh.closed) == 1
        assert pdh.paths[-1] == "\\GPU Process Memory(pid_8_*)\\Shared Usage"

    def test_other_platforms_are_none_without_importing_pdh(self):
        with patch.dict(sys.modules, {"win32pdh": None}), patch.object(sys, "platform", "linux"):
            assert gpu_usage.process_gpu_memory(7) is None
        assert gpu_usage._pdh_proc_state is None

    def test_missing_win32pdh_is_none_and_sticky(self):
        with patch.dict(sys.modules, {"win32pdh": None}), patch.object(sys, "platform", "win32"):
            assert gpu_usage.process_gpu_memory(7) is None
        assert gpu_usage._pdh_proc_state == {}

    @pytest.mark.skipif(sys.platform != "win32", reason="WDDM counters exist on Windows only")
    def test_the_real_counters_answer_without_raising(self):
        pytest.importorskip("win32pdh")
        reading = gpu_usage.process_gpu_memory()
        assert reading is None or (isinstance(reading, tuple) and len(reading) == 2
                                   and all(isinstance(v, int) and v >= 0 for v in reading))
