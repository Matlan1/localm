# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``moe_cache_mib`` setting: llama.cpp's VRAM cache for the MoE experts kept
in system RAM. Off by default; a load gets it only when experts were placed in
system RAM and read into memory (not memory-mapped); the VRAM budget sets it
aside for any model with routed experts."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _structs
from tests._bare_llama import make_bare_llama
from tests.test_kv_bytes_from_gguf import _gguf, _shape

MIB = 1024 ** 2


class TestSetting:
    def test_it_is_off_by_default(self):
        assert DEFAULT_CONFIG["moe_cache_mib"] == 0

    def test_the_schema_accepts_a_size_and_refuses_a_negative_one(self):
        assert ss.validate_update({"moe_cache_mib": 3072})["moe_cache_mib"] == 3072
        with pytest.raises(ValueError):
            ss.validate_update({"moe_cache_mib": -1})

    def test_a_change_reloads_the_model(self):
        from localm.inference.routing_latch import LOAD_CONFIG_KEYS
        assert "moe_cache_mib" in LOAD_CONFIG_KEYS

    def test_it_is_in_the_bug_report_config_keys(self):
        from localm.bugreport import _SAFE_CONFIG_KEYS
        assert "moe_cache_mib" in _SAFE_CONFIG_KEYS

    @pytest.mark.parametrize("raw,mib", [(3072, 3072), (None, 0), (-5, 0), ("many", 0)])
    def test_the_engine_reads_it(self, raw, mib):
        from localm.inference import engine
        assert engine._resolve_moe_cache_mib({"moe_cache_mib": raw}) == mib


class TestPlumbing:
    def test_create_backend_reads_the_setting(self, tmp_path):
        from localm.inference import engine
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF" + b"\0" * 64)
        cfg = dict(DEFAULT_CONFIG, moe_cache_mib=2048)
        with patch.object(engine, "load_config", return_value=cfg):
            backend = engine.create_backend(str(f))
        assert backend._MOE_CACHE_BYTES == 2048 * MIB

    @pytest.mark.parametrize("mib,sent", [(1024, 1024 * MIB), (0, None)])
    def test_the_backend_sends_the_size_to_the_worker(self, tmp_path, mib, sent):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        b = GgufBackend(str(f), moe_cache_mib=mib)
        with patch("localm.config.load_config", return_value=dict(DEFAULT_CONFIG)), \
             patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load", return_value={}) as spawn:
            b.effective_gpu_layers = 0
            b._load_native()
        assert spawn.call_args[0][0].get("moe_cache_size") == sent

    @pytest.mark.parametrize("size", [512 * MIB, 0])
    def test_the_worker_hands_the_size_to_llamacpp(self, tmp_path, size):
        from tests.test_gguf_worker import _StubLlm, _worker
        w = _worker(str(tmp_path / "m.gguf"), moe_cache_size=size)
        fake = MagicMock()
        fake.LlamaCpp.return_value = _StubLlm()
        with patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.dict(sys.modules, {"localm.inference.backends.llamacpp": fake}):
            w.load()
        _, kwargs = fake.LlamaCpp.call_args
        assert kwargs.get("moe_cache_size") == (size or None)


class TestWhenTheCacheApplies:
    def _llm(self, mapped, skip=None):
        llm = make_bare_llama()
        llm.mmap_mapped = mapped
        llm.moe_skip_reason = skip
        return llm

    def test_experts_read_into_system_ram_get_the_cache(self):
        llm = self._llm(mapped=False)
        assert llm._resolve_moe_cache(1024 * MIB, 8) == 1024 * MIB

    @pytest.mark.parametrize("mapped", [True, None])
    def test_memory_mapped_experts_do_not(self, mapped, caplog):
        llm = self._llm(mapped=mapped)
        with caplog.at_level(logging.INFO, logger="localm"):
            assert llm._resolve_moe_cache(1024 * MIB, 8) == 0
        assert "memory-mapped" in caplog.text

    def test_a_load_with_no_experts_in_system_ram_does_not(self):
        assert self._llm(mapped=False)._resolve_moe_cache(1024 * MIB, 0) == 0
        assert self._llm(mapped=False, skip="no experts")._resolve_moe_cache(1024 * MIB, 8) == 0

    def test_off_is_off(self):
        assert self._llm(mapped=False)._resolve_moe_cache(0, 8) == 0

    def test_every_context_gets_the_size(self):
        llm = make_bare_llama()
        llm.moe_cache_bytes = 256 * MIB
        cp = _structs.LlamaContextParamsV3()
        llm._apply_moe_cache(cp)
        assert cp.moe_cache_size == 256 * MIB

    def test_a_runtime_without_the_cache_turns_it_off(self, caplog):
        llm = make_bare_llama()
        llm.moe_cache_bytes = 256 * MIB
        cp = _structs.LlamaContextParamsV2()
        with caplog.at_level(logging.WARNING, logger="localm"):
            llm._apply_moe_cache(cp)
        assert llm.moe_cache_bytes == 0
        assert "no MoE expert cache" in caplog.text
        assert not hasattr(cp, "moe_cache_size")


class TestVramBudget:
    def _backend(self, tmp_path, mib, experts):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        b = GgufBackend(str(f), moe_cache_mib=mib, vram_overhead_bytes=1000)
        b._moe_expert_bytes_by_layer = lambda: experts
        return b

    def test_a_moe_model_sets_the_cache_aside_once(self, tmp_path):
        b = self._backend(tmp_path, 512, {0: 10, 1: 10})
        assert b._split_overhead_bytes(1) == 1000 + 512 * MIB
        assert b._split_overhead_bytes(2) == 2000 + 512 * MIB

    def test_a_dense_model_does_not(self, tmp_path):
        assert self._backend(tmp_path, 512, {})._split_overhead_bytes(1) == 1000

    def test_off_sets_nothing_aside(self, tmp_path):
        assert self._backend(tmp_path, 0, {0: 10})._split_overhead_bytes(1) == 1000
