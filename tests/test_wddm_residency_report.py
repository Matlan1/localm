# SPDX-License-Identifier: AGPL-3.0-or-later
"""On Windows, a model fully on a GPU whose process the WDDM counters show
holding a large share of its GPU memory in shared system memory gets one INFO
line with the amounts, and one more when Windows moves it back. The live
context is never recreated for it."""

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
SAID = "of this model's GPU memory in shared system memory"
BACK = "moved this model's GPU memory back to VRAM"


def _llm(capacity=4096, *, offload_kqv=True):
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _n_ctx=4096,
                          _n_ctx_max=65536, _n_ctx_grow=4096,
                          _ctx_capacity=capacity, _offload_kqv=offload_kqv,
                          _kv_supported=True)
    llm._tokenizer.is_eog.return_value = False
    llm._residency_watch = True
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


def _generate(llm, api, reading, prompt=(1, 2, 3), n=4, platform="win32"):
    read = MagicMock(return_value=reading)
    with patch(API, api), patch(READ, read), patch.object(sys, "platform", platform), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
        out = list(llm._generate(prompt_tokens=list(prompt), max_new_tokens=n,
                                 temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0))
    return out, read.call_count


def _due(llm):
    llm._residency_check_after = 0.0


def _lines(caplog, text):
    return [r for r in caplog.records if text in r.getMessage()]


DEMOTED = (int(9.64 * GB), int(3.30 * GB))
HEALTHY = (int(11.11 * GB), int(0.48 * GB))


def _inits(api):
    return api.llama_init_from_model.call_count


class TestReport:
    def test_a_demoted_process_is_reported_once_with_the_amounts(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            out, reads = _generate(llm, api, DEMOTED)
            _due(llm)
            _generate(llm, api, DEMOTED)
        assert out == [42] * 4 and reads == 1
        lines = _lines(caplog, SAID)
        assert len(lines) == 1 and lines[0].levelno == logging.INFO
        assert ("Windows holds 3.30 GB of this model's GPU memory in shared system memory "
                "(9.64 GB in VRAM)") in lines[0].getMessage()

    def test_the_context_is_never_recreated_for_it(self):
        llm = _llm()
        api, _cp = _api()
        for _ in range(3):
            _due(llm)
            _generate(llm, api, DEMOTED)
        assert _inits(api) == 0
        assert llm._ctx_ptr == 222 and llm._decode_refresh is None

    def test_the_move_back_is_reported_and_a_new_spell_reports_again(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _generate(llm, api, DEMOTED)
            _due(llm)
            _generate(llm, api, HEALTHY)
            _due(llm)
            _generate(llm, api, HEALTHY)
            _due(llm)
            _generate(llm, api, DEMOTED)
        back = _lines(caplog, BACK)
        assert len(back) == 1
        assert "(11.11 GB in VRAM, 0.48 GB in shared system memory)" in back[0].getMessage()
        assert len(_lines(caplog, SAID)) == 2

    def test_a_healthy_process_says_nothing(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _out, reads = _generate(llm, api, HEALTHY)
        assert reads == 1
        assert not _lines(caplog, SAID) and not _lines(caplog, BACK)

    def test_shared_under_the_minimum_says_nothing(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _generate(llm, api, (int(2 * GB), llm._RESIDENCY_SHARED_MIN - 1))
        assert not _lines(caplog, SAID)

    def test_shared_at_the_minimum_is_reported(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _generate(llm, api, (int(2 * GB), llm._RESIDENCY_SHARED_MIN))
        assert len(_lines(caplog, SAID)) == 1

    def test_shared_under_the_share_of_dedicated_says_nothing(self, caplog):
        llm = _llm()
        api, _cp = _api()
        dedicated = 20 * GB
        shared = int(llm._RESIDENCY_SHARED_RATIO * dedicated) - 1
        assert shared > llm._RESIDENCY_SHARED_MIN
        with caplog.at_level(logging.INFO, logger="localm"):
            _generate(llm, api, (dedicated, shared))
        assert not _lines(caplog, SAID)

    def test_unavailable_counters_are_said_once_at_debug(self, caplog):
        llm = _llm()
        api, _cp = _api()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            _generate(llm, api, None)
            _due(llm)
            _generate(llm, api, None)
        lines = _lines(caplog, "counters are unavailable")
        assert len(lines) == 1 and lines[0].levelno == logging.DEBUG


class TestGating:
    def test_other_platforms_never_read_the_counters(self):
        llm = _llm()
        api, _cp = _api()
        _out, reads = _generate(llm, api, DEMOTED, platform="linux")
        assert reads == 0

    def test_an_instance_init_did_not_build_never_reads_the_counters(self):
        llm = _llm()
        llm._residency_watch = False
        api, _cp = _api()
        _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 0

    def test_init_turns_the_check_on(self):
        import inspect
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        assert LlamaCpp._residency_watch is False
        assert "self._residency_watch = True" in inspect.getsource(LlamaCpp.__init__)

    def test_a_load_not_fully_on_a_gpu_is_not_checked(self):
        llm = _llm()
        llm._decode_judged = False
        api, _cp = _api()
        _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 0

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

    def test_a_pending_decode_rate_refresh_still_runs(self, caplog):
        llm = _llm()
        llm._decode_refresh = (5.0, 50.0)
        api, _cp = _api()
        with caplog.at_level(logging.INFO, logger="localm"):
            _out, reads = _generate(llm, api, DEMOTED)
        assert reads == 1 and _inits(api) == 1
        assert _lines(caplog, "5.0 steps/s where 50.0 was expected")


class TestSlotSeam:
    def test_the_slot_seam_checks_and_plans_nothing_for_it(self):
        from localm.inference.backends.llamacpp._slots import LlamaSlotOps
        llm = _llm(capacity=8192)
        read = MagicMock(return_value=DEMOTED)
        with patch(READ, read), patch.object(sys, "platform", "win32"):
            assert LlamaSlotOps(llm).rightsize(100, 5000) is None
        assert read.call_count == 1 and llm._residency_reported is True


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

    @pytest.mark.skipif(sys.platform != "win32", reason="WDDM counters exist on Windows only")
    def test_the_real_counters_read_a_process_that_holds_vram(self):
        pdh = pytest.importorskip("win32pdh")
        try:
            _c, instances = pdh.EnumObjectItems(None, None, "GPU Process Memory",
                                                pdh.PERF_DETAIL_WIZARD)
        except Exception as exc:
            pytest.skip(f"GPU Process Memory is not enumerable here: {exc}")
        pids = sorted({int(i.split("_")[1]) for i in instances if i.startswith("pid_")})
        held = [r for r in (gpu_usage.process_gpu_memory(pid) for pid in pids) if r is not None]
        if not held:
            pytest.skip("no process holds dedicated VRAM on this machine")
        for dedicated, shared in held:
            assert dedicated >= gpu_usage._PROC_ADAPTER_MIN_DEDICATED and shared >= 0


@pytest.mark.skipif(sys.platform != "win32", reason="pywin32 exists on Windows only")
class TestWin32pdhWithoutSite:
    def _run(self, code):
        import os
        import subprocess
        from pathlib import Path
        pdh = pytest.importorskip("win32pdh")
        site_packages = Path(pdh.__file__).resolve().parent.parent
        base = Path(sys.base_prefix) / "python.exe"
        if not base.is_file():
            pytest.skip("no base interpreter next to sys.base_prefix")
        root = Path(__file__).resolve().parent.parent
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(root), str(site_packages), str(site_packages / "win32"),
             str(site_packages / "win32" / "lib")]))
        return subprocess.run([str(base), "-S", "-c", code], env=env, capture_output=True,
                              text=True, timeout=120)

    def test_the_plain_import_fails_there(self):
        r = self._run("import win32pdh")
        assert r.returncode != 0 and "DLL load failed" in r.stderr

    def test_win32pdh_imports_without_site(self):
        r = self._run("from localm.gpu_usage import _import_win32pdh\n"
                      "print(_import_win32pdh().__name__)")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "win32pdh"
