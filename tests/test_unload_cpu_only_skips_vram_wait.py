# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unloading or evicting a model loaded with 0 GPU layers frees no VRAM, so it
takes no free-VRAM reading and waits for no VRAM release; a model that holds
VRAM still does both."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from localm.inference import http_server as hs
from localm.inference import switch_admission
from localm.inference.engine import Engine

_GB = 1024 ** 3


class _FakeEngine:
    def __init__(self, name, *, cpu_only):
        self.display_name = name
        self.model_path = f"Z:/models/{name}.gguf"
        self.cpu_only_load = cpu_only
        self.active_requests = 0
        self.unloading = False
        self._loaded = True

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self._loaded = False


class _VramSpy:
    """Counts the free-VRAM readings and release waits the unload paths take."""

    def __init__(self):
        self.readings = 0
        self.waits = 0

    def install(self, monkeypatch):
        import localm.vram as vram

        def _reading():
            self.readings += 1
            return 8 * _GB, True, None

        def _wait(read_free, *, before_bytes, **kwargs):
            self.waits += 1
            return True, before_bytes + _GB

        monkeypatch.setattr(vram, "_vram_free_reading", _reading)
        monkeypatch.setattr(vram, "wait_for_vram_release", _wait)
        return self


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(hs, "_engines", {})
    monkeypatch.setattr(hs, "_engines_lru", [])
    monkeypatch.setattr(hs, "_inference_sems", {})
    monkeypatch.setattr(hs, "_last_activity_per_model", {})
    monkeypatch.setattr(hs, "_active_model_name", None)
    monkeypatch.setattr(hs, "_last_active_model_name", None)
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(hs, "_inference_sem", None)
    return _VramSpy().install(monkeypatch)


def _put(*engines):
    for e in engines:
        hs._engines[e.display_name] = e
        hs._engines_lru.append(e.display_name)


class TestEngineCpuOnlyLoad:
    @pytest.mark.parametrize("offloaded, expected", [
        (0, True), (1, False), (30, False), (None, False), (False, False)])
    def test_reports_only_a_zero_gpu_layer_load(self, offloaded, expected):
        engine = object.__new__(Engine)
        engine._backend = SimpleNamespace(gpu_layers_offloaded=offloaded)
        assert engine.cpu_only_load is expected

    def test_a_backend_that_reports_no_gpu_layers_is_not_cpu_only(self):
        engine = object.__new__(Engine)
        engine._backend = SimpleNamespace()
        assert engine.cpu_only_load is False


class TestUnloadOne:
    def test_cpu_only_model_takes_no_reading_and_waits_for_nothing(self, spy):
        _put(_FakeEngine("cpu", cpu_only=True))
        result = asyncio.run(hs.unload_one_model("cpu"))
        assert result["status"] == "unloaded"
        assert not hs._engines["cpu"].loaded
        assert (spy.readings, spy.waits) == (0, 0)

    def test_model_holding_vram_still_waits_for_the_release(self, spy):
        _put(_FakeEngine("gpu", cpu_only=False))
        result = asyncio.run(hs.unload_one_model("gpu"))
        assert result["status"] == "unloaded"
        assert result["vram_freed"] is True
        assert (spy.readings, spy.waits) == (1, 1)


class TestUnloadAll:
    @pytest.fixture(autouse=True)
    def _no_other_models(self, monkeypatch):
        from localm.inference import embedder, reranker, speech
        monkeypatch.setattr(embedder, "loaded_dim", lambda: None)
        monkeypatch.setattr(reranker, "is_loaded", lambda: False)
        monkeypatch.setattr(speech, "is_loaded", lambda: False)

    def test_only_cpu_only_models_take_no_reading_and_wait_for_nothing(self, spy):
        _put(_FakeEngine("cpu-a", cpu_only=True), _FakeEngine("cpu-b", cpu_only=True))
        result = asyncio.run(hs.unload_all_models())
        assert sorted(result["unloaded_models"]) == ["cpu-a", "cpu-b"]
        assert (spy.readings, spy.waits) == (0, 0)

    def test_one_model_holding_vram_among_cpu_only_ones_still_waits(self, spy):
        _put(_FakeEngine("cpu", cpu_only=True), _FakeEngine("gpu", cpu_only=False))
        result = asyncio.run(hs.unload_all_models())
        assert sorted(result["unloaded_models"]) == ["cpu", "gpu"]
        assert result["vram_freed"] is True
        assert (spy.readings, spy.waits) == (1, 1)

    def test_a_loaded_embedder_beside_a_cpu_only_model_still_waits(self, spy, monkeypatch):
        from localm.inference import embedder
        monkeypatch.setattr(embedder, "loaded_dim", lambda: 384)
        monkeypatch.setattr(embedder, "reset_embedder", lambda force=False: True)
        _put(_FakeEngine("cpu", cpu_only=True))
        result = asyncio.run(hs.unload_all_models())
        assert result["unloaded_other"] == ["embedding model"]
        assert (spy.readings, spy.waits) == (1, 1)

    def test_nothing_loaded_still_reports_the_reading(self, spy):
        result = asyncio.run(hs.unload_all_models())
        assert result["status"] == "already_unloaded"
        assert (spy.readings, spy.waits) == (1, 0)


class TestEviction:
    @staticmethod
    def _probe():
        return switch_admission.VramProbe(
            free=8 * _GB, probe_ok=True, process_scoped=False, shortfall=[],
            shares_adaptive=False)

    def _evict(self, engine):
        async def _run():
            return await hs._switch_free_victim(
                asyncio.get_running_loop(), engine.display_name, engine, self._probe())
        return asyncio.run(_run())

    def test_cpu_only_victim_waits_for_no_release(self, spy):
        engine = _FakeEngine("cpu", cpu_only=True)
        release = self._evict(engine)
        assert not engine.loaded
        assert release.held_nothing is True
        assert release.released is None
        assert "held no VRAM" in release.describe()
        assert spy.waits == 0
        assert not hs._evicting_names

    def test_victim_holding_vram_still_waits_for_the_release(self, spy):
        engine = _FakeEngine("gpu", cpu_only=False)
        release = self._evict(engine)
        assert release.held_nothing is False
        assert release.released is True
        assert spy.waits == 1


_TINY_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_TINY_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.mark.integration
@pytest.mark.real_gguf
class TestRealCpuOnlyLoad:
    def test_unloading_a_real_zero_gpu_layer_model_takes_no_vram_wait(self, spy):
        from tests._real_gguf import fetch_gguf, require_native_runtime
        require_native_runtime()
        local = os.environ.get("LOCALM_TEST_TINY_GGUF")
        path = local if local and os.path.isfile(local) else fetch_gguf(_TINY_REPO, _TINY_FILE)

        engine = Engine(path, n_ctx=1024, n_gpu_layers=0, display_name="tiny-cpu")
        engine.load()
        try:
            assert engine.loaded
            assert engine.cpu_only_load is True
            _put(engine)
            started = time.monotonic()
            result = asyncio.run(hs.unload_one_model("tiny-cpu"))
            elapsed = time.monotonic() - started
            assert result["status"] == "unloaded"
            assert not engine.loaded
            assert (spy.readings, spy.waits) == (0, 0)
            assert elapsed < 5.0, f"unload took {elapsed:.1f}s"
        finally:
            engine.unload()
