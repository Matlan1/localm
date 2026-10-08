# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-GPU placement on a box whose integrated GPU llama.cpp may leave out:
the main GPU's messages and numbering on every path, a runtime that keeps the
integrated GPU, probes that did not finish, probe counts, HF loads on one
chosen GPU, the worker and native classes writing a parent-resolved main GPU,
and the peers asked to free VRAM.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest

from localm import discover
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader
from tests.test_implicit_split_fit import (
    _config, _registry, _tiny_model, _torch_readings)
from tests.test_multigpu_config_placement import (
    _backend, _box, _load, _worker_writes)

GiB = 1024 ** 3


def _registry_keeping_igpu(gpus):
    """A runtime registry that types every GPU, the integrated one included,
    as a plain GPU: llama.cpp then keeps the integrated GPU."""
    return [{"index": i, "name": f"CUDA{i}", "description": "",
             "type": _loader.GGML_DEV_TYPE_GPU, "free": g["free"], "total": g["total"]}
            for i, g in enumerate(gpus)]


def _mb_readings(*specs):
    """Torch readings from ``(free_bytes, integrated)`` pairs, total = free + 10."""
    return [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
             "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": ig}
            for i, (f, ig) in enumerate(specs)]


def _fit_backend(tmp_path):
    b = GgufBackend(str(_tiny_model(tmp_path, vocab=5000)), n_ctx=4096,
                    n_gpu_layers=99)
    b._VRAM_OVERHEAD_BYTES = 1000
    b._gguf_kv_bpt = 10
    return b


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


class TestMainGpuMessagesNameTheConfiguredGpu:
    """A main GPU decided in the parent is named by its torch index, and the
    worker writes the parent's llama.cpp device without warning about it."""

    def _run(self, tmp_path, gpus, cfg, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg)
            writes = _worker_writes(params, cfg, gpus)
        return params, writes, _warnings(caplog)

    def test_a_one_gpu_choice_other_than_the_main_gpu(self, tmp_path, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[2], gpu_split_ratios=None, main_gpu_index=1)
        params, writes, warned = self._run(tmp_path, gpus, cfg, caplog)
        assert writes == (discover._LLAMA_SPLIT_MODE_NONE, 1, None)
        assert params["main_gpu"] == 1
        assert warned == ["main_gpu_index=1 is not one of the GPUs this load uses "
                          "(2); GPU 2 is the primary instead"]

    def test_a_split_that_leaves_the_main_gpu_out(self, tmp_path, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[2, 3], gpu_split_ratios=None, main_gpu_index=1)
        params, writes, warned = self._run(tmp_path, gpus, cfg, caplog)
        assert sorted(params["gpu_split_ratios"]) == [1, 2]
        assert writes[1] == 1
        assert warned == ["main_gpu_index=1 is not one of the GPUs this load uses "
                          "(2, 3); GPU 2 is the primary instead"]

    def test_a_stale_main_gpu_is_warned_once_by_its_own_number(self, tmp_path, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=None, main_gpu_index=9)
        params, writes, warned = self._run(tmp_path, gpus, cfg, caplog)
        assert params["main_gpu"] == 0 and writes[1] == 0
        assert len(warned) == 1 and warned[0].startswith("main_gpu_index=9 ")
        assert not any("integrated" in w for w in warned)


class TestMainGpuOnEveryRenumberedPath:
    """With no gpu_split_indices on a box whose integrated GPU llama.cpp leaves
    out, main_gpu_index reaches the worker as a llama.cpp device, whether the
    implicit fit narrows the load or llama.cpp's default split stands."""

    def test_a_one_gpu_fit_on_the_configured_main_gpu_is_quiet(self, tmp_path, caplog):
        gpus = _mb_readings((2_000_000_000, True), (60_000_000, False),
                            (30_000_000, False))
        cfg = _config(gpu_split_indices=None, main_gpu_index=1)
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_fit_backend(tmp_path), gpus, _registry(gpus), cfg)
            writes = _worker_writes(params, cfg, gpus)
        assert params["gpu_split_ratios"] == {0: 1.0}
        assert writes == (discover._LLAMA_SPLIT_MODE_NONE, 0, None)
        assert _warnings(caplog) == []

    def test_a_fit_split_keeps_the_configured_main_gpu(self, tmp_path, caplog):
        gpus = _mb_readings((2_000_000_000, True), (50_000_000, False),
                            (50_000_000, False), (30_000_000, False))
        cfg = _config(gpu_split_indices=None, main_gpu_index=2)
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_fit_backend(tmp_path), gpus, _registry(gpus), cfg)
            writes = _worker_writes(params, cfg, gpus)
        assert sorted(params["gpu_split_ratios"]) == [0, 1]
        assert writes[0] == discover._LLAMA_SPLIT_MODE_LAYER
        assert writes[1] == 1, "main_gpu_index=2 is llama.cpp device 1"
        assert _warnings(caplog) == []

    def test_a_fit_onto_another_gpu_says_so_in_torch_numbering(
            self, tmp_path, caplog, capsys):
        # GPU 1 has room for nothing; the default split still gives it the
        # first layer, so the fit loads on GPU 2 alone.
        gpus = _mb_readings((2_000_000_000, True), (2_000, False),
                            (60_000_000, False))
        cfg = _config(gpu_split_indices=None, main_gpu_index=1)
        b = _fit_backend(tmp_path)
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(b, gpus, _registry(gpus), cfg)
            writes = _worker_writes(params, cfg, gpus)
        assert params["gpu_split_ratios"] == {1: 1.0}
        assert writes == (discover._LLAMA_SPLIT_MODE_NONE, 1, None)
        assert b.load_gpu_index == 2
        assert _warnings(caplog) == ["main_gpu_index=1 is not one of the GPUs this "
                                     "load uses (2); GPU 2 is the primary instead"]
        assert "loading on device 2 only" in " ".join(capsys.readouterr().out.split())

    def test_the_default_split_carries_the_main_gpu_too(self, tmp_path):
        gpus = _torch_readings((0.45, True), (23.0, False), (23.0, False))
        cfg = _config(gpu_split_indices=None, main_gpu_index=2)
        params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg)
        assert params["gpu_split_ratios"] is None
        assert params["main_gpu"] == 1
        assert _worker_writes(params, cfg, gpus) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 1, None)

    def test_without_an_integrated_gpu_main_gpu_is_left_to_the_worker(self, tmp_path):
        gpus = _torch_readings((23.0, False), (23.0, False))
        cfg = _config(gpu_split_indices=None, main_gpu_index=1)
        reads = []
        params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg,
                       registry_reads=reads)
        assert "main_gpu" not in params and reads == [0]
        assert _worker_writes(params, cfg, gpus)[1] == 1


class TestARuntimeThatKeepsTheIntegratedGpu:
    """A runtime that types the integrated GPU as a plain GPU keeps it in
    llama.cpp's device list, so torch's numbering is llama.cpp's."""

    def test_a_configured_split_applies_unchanged(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[3.0, 1.0],
                      main_gpu_index=None)
        params = _load(_backend(tmp_path), gpus, _registry_keeping_igpu(gpus), cfg)
        assert params["gpu_split_ratios"] is None
        assert _worker_writes(params, cfg, gpus)[2] == pytest.approx(
            [0, 3.0, 1.0, 0, 0, 0, 0, 0])

    def test_a_one_gpu_choice_of_the_integrated_gpu_loads_there(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False))
        cfg = _config(gpu_split_indices=[0], gpu_split_ratios=None)
        params = _load(_backend(tmp_path), gpus, _registry_keeping_igpu(gpus), cfg)
        assert params["gpu_split_ratios"] == {0: 1.0}

    def test_the_implicit_fit_plans_over_every_gpu(self, tmp_path):
        # Over the discrete GPUs alone the 30 MB one would take the output
        # layer and be left out; over all three the 30 MB GPU is left out and
        # the load splits over the integrated GPU and the 50 MB discrete GPU.
        gpus = _mb_readings((50_000_000, True), (50_000_000, False),
                            (30_000_000, False))
        b = _fit_backend(tmp_path)
        params = _load(b, gpus, _registry_keeping_igpu(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None))
        assert sorted(params["gpu_split_ratios"]) == [0, 1]
        assert b._fit_source_index == {0: 0, 1: 1, 2: 2}


class TestATrailingIntegratedGpu:
    """An integrated GPU torch numbers after the discrete ones is still left
    out of llama.cpp's device list; the numbering is not mistaken for one in
    which llama.cpp keeps every GPU."""

    def test_a_trailing_integrated_gpu_is_still_left_out(self, tmp_path):
        gpus = _mb_readings((2_000, False), (60_000_000, True))
        params = _load(_fit_backend(tmp_path), gpus, _registry(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None))
        assert params["gpu_split_ratios"] is None
        assert _worker_writes(params, _config(gpu_split_indices=None), gpus) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 0, None)

    def test_a_split_naming_a_trailing_integrated_gpu_is_refused(self, tmp_path):
        gpus = _torch_readings((20.0, False), (20.0, False), (2.0, True))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=None)
        with pytest.raises(discover.GpuSplitConfigError, match="GPU 2"):
            _load(_backend(tmp_path), gpus, _registry(gpus), cfg)

    def test_a_main_gpu_naming_a_trailing_integrated_gpu_is_warned(
            self, tmp_path, caplog):
        gpus = _torch_readings((20.0, False), (20.0, False), (2.0, True))
        cfg = _config(gpu_split_indices=None, main_gpu_index=2)
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg)
        assert params["main_gpu"] == 0
        assert _warnings(caplog) == ["main_gpu_index=2 is an integrated GPU, which "
                                     "llama.cpp does not use beside a discrete GPU; "
                                     "GPU 0 is the primary instead"]


class TestAProbeThatDidNotFinish:
    """A timed-out or busy probe never sends the configured split to the
    worker in torch's numbering."""

    def test_a_served_reading_is_still_renumbered(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[3.0, 1.0],
                      main_gpu_index=2)
        params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg,
                       status=discover.GPU_PROBE_TIMEOUT, last=False)
        assert _worker_writes(params, cfg, gpus) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 1,
            pytest.approx([3.0, 1.0, 0, 0, 0, 0, 0, 0]))

    def test_no_reading_keeps_llamacpps_default_split(self, tmp_path, caplog):
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[3.0, 1.0],
                      main_gpu_index=None)
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_backend(tmp_path), [], [], cfg,
                           status=discover.GPU_PROBE_BUSY, last=False)
        assert params["gpu_split_ratios"] == {}
        assert _worker_writes(params, cfg, []) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 0, None)
        assert any("keeping llama.cpp's default split" in w for w in _warnings(caplog))


class TestNoExtraProbes:
    """The placement reuses a reading the caller already took, and the GPU
    registry heartbeat never probes."""

    @pytest.mark.parametrize("ratios", [[3.0, 1.0], None])
    def test_a_load_without_an_integrated_gpu_probes_no_more(self, tmp_path, ratios):
        gpus = _torch_readings((20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[0, 1], gpu_split_ratios=ratios)
        with_placement, without = [], []
        _load(_backend(tmp_path), gpus, _registry(gpus), cfg, last=False,
              calls=with_placement)
        with mock.patch.object(discover, "configured_split_placement",
                               lambda *a, **k: None):
            _load(_backend(tmp_path), gpus, _registry(gpus), cfg, last=False,
                  calls=without)
        assert len(with_placement) == len(without)

    def test_an_embedder_load_probes_no_more(self, monkeypatch, tmp_path):
        class _Runner:
            def spawn_and_load(self, params, timeout=None):
                return {"dim": 768, "n_ctx": 512}

        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"\0" * (2 * 1024 * 1024))
        gpus = _torch_readings((20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[0, 1], gpu_split_ratios=[1.0, 1.0])
        monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _Runner)
        from localm.inference.embedder import IsolatedEmbedder
        counts = []
        for placement in (True, False):
            calls = []
            patch = (contextlib.nullcontext() if placement else
                     mock.patch.object(discover, "configured_split_placement",
                                       lambda *a, **k: None))
            with _box(gpus, _registry(gpus), calls=calls), patch, \
                    mock.patch("localm.config.load_config", return_value=cfg):
                IsolatedEmbedder(str(model_file))
            counts.append(len(calls))
        assert counts[0] == counts[1]

    def _sync(self, monkeypatch, cfg, gpus, *, engines=None, active=None, last=True):
        from localm.inference import http_server
        written, calls = {}, []
        monkeypatch.setattr(http_server, "_gpu_coord",
                            {"instance_id": "i", "port": 1})
        monkeypatch.setattr(http_server, "_engines", engines or {})
        monkeypatch.setattr(http_server, "_active_model_name", active)
        monkeypatch.setattr(http_server, "_loaded_model_identities", lambda: [])
        monkeypatch.setattr(http_server, "_model_file_size", lambda n: None)
        with _box(gpus, _registry(gpus), calls=calls, last=last), \
                mock.patch("localm.config.load_config", return_value=cfg):
            written.update(http_server._gpu_status() or {})
        return written["gpu_index"], len(calls)

    @pytest.mark.parametrize("split,main,last,expected", [
        ([1], None, True, 1), ([1], None, False, 1), ([5], 1, True, 1),
        ([0], 1, False, 0)])
    def test_the_registry_heartbeat_never_probes(self, monkeypatch, split, main,
                                                 last, expected):
        gpus = _torch_readings((8.0, False), (20.0, False))
        index, probes = self._sync(
            monkeypatch, _config(gpu_split_indices=split, main_gpu_index=main),
            gpus, last=last)
        assert probes == 0
        assert index == expected

    def test_a_loaded_model_on_one_gpu_needs_no_probe(self, monkeypatch):
        engine = SimpleNamespace(loaded=True,
                                 _backend=SimpleNamespace(load_gpu_index=1))
        index, probes = self._sync(
            monkeypatch, _config(gpu_split_indices=[1], main_gpu_index=None),
            _torch_readings((8.0, False), (20.0, False)), engines={"m": engine},
            active="m")
        assert (index, probes) == (1, 0)


class TestHfLoadsHonourOneChosenGpu:
    """A 1-entry gpu_split_indices confines an HF (transformers) load to that
    GPU, and the check before the load and the registry describe that GPU."""

    @staticmethod
    def _gpus():
        return [{"index": 0, "name": "small", "free": 8 * GiB, "total": 12 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False},
                {"index": 1, "name": "big", "free": 20 * GiB, "total": 24 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}]

    @pytest.mark.parametrize("main", [0, None])
    def test_the_device_map_admission_and_registry_agree(self, monkeypatch, main):
        from localm.inference import http_server, switch_admission
        from localm.inference.backends._hf_worker import _cuda_device_map
        gpus = self._gpus()
        cfg = _config(gpu_split_indices=[1], gpu_split_ratios=None, main_gpu_index=main)
        torch = mock.MagicMock()
        torch.cuda.mem_get_info.side_effect = lambda i: (gpus[i]["free"],
                                                         gpus[i]["total"])
        budget = switch_admission.LoadBudget(
            name="hf", vram_required=10 * GiB, headroom=GiB, resident_cap=None,
            pinned=frozenset(), check_split_fit=False)

        async def _probe():
            return await http_server._switch_probe_vram(
                asyncio.get_running_loop(), budget)

        written = {}
        monkeypatch.setattr(http_server, "_gpu_coord",
                            {"instance_id": "i", "port": 1})
        monkeypatch.setattr(http_server, "_engines", {})
        monkeypatch.setattr(http_server, "_active_model_name", None)
        monkeypatch.setattr(http_server, "_loaded_model_identities", lambda: [])
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=cfg):
            device_map = _cuda_device_map(torch, cfg)
            probe = asyncio.run(_probe())
            written.update(http_server._gpu_status() or {})
        assert set(device_map["max_memory"]) == {1, "cpu"}
        assert probe.free == 20 * GiB
        assert written["gpu_index"] == 1


class TestAnExplicitMainGpuBesideTheOnlyDiscreteGpu:
    """main_gpu_index naming an integrated GPU beside one discrete GPU: an HF
    load runs on that GPU and is judged by it, while a GGUF load (which
    llama.cpp places on the discrete GPU) is judged by the discrete GPU."""

    @staticmethod
    def _gpus():
        return [{"index": 0, "name": "igpu", "free": 6 * GiB, "total": 8 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": True},
                {"index": 1, "name": "dgpu", "free": 20 * GiB, "total": 24 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}]

    @pytest.mark.parametrize("gguf,free", [(False, 6 * GiB), (True, 20 * GiB)])
    def test_each_load_is_judged_by_the_gpu_it_uses(self, monkeypatch, gguf, free):
        from localm.inference import http_server, switch_admission
        gpus = self._gpus()
        budget = switch_admission.LoadBudget(
            name="m", vram_required=4 * GiB, headroom=GiB, resident_cap=None,
            pinned=frozenset(), check_split_fit=gguf)

        async def _probe():
            return await http_server._switch_probe_vram(
                asyncio.get_running_loop(), budget)

        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=_config(
                    gpu_split_indices=None, main_gpu_index=0)):
            probe = asyncio.run(_probe())
            index = http_server._current_gpu_index()
        assert probe.free == free
        assert index == 0


class TestMainGpuReachesTheNativeParams:
    """A parent-resolved main_gpu reaches llama_model_params through the real
    worker and native-load classes."""

    def test_llamacpp_writes_it(self, monkeypatch):
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        from tests.test_main_gpu_wiring import _mock_llama_api
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        mock_api = _mock_llama_api()
        with mock.patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True,
                           main_gpu=1)
            llm.close()
        assert mock_api.llama_model_default_params.return_value.main_gpu == 1
        assert llm._main_gpu_index == 1

    def test_the_worker_forwards_it(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        seen = {}

        class _FakeLlama:
            supports_images = False

            def __init__(self, **kw):
                seen.update(kw)

        with mock.patch("localm.inference.backends.llamacpp._loader.load_lib"), \
                mock.patch("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama):
            GgufWorker("m.gguf", None, 512, 99, None, 512, main_gpu=1).load()
        assert seen["main_gpu"] == 1

    def test_the_embedder_writes_it(self, monkeypatch):
        from tests.test_main_gpu_wiring import TestGgufEmbedderMainGpuWiring
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        mock_api = TestGgufEmbedderMainGpuWiring()._mock_embed_api()
        with mock.patch("localm.inference.backends.llamacpp._api", mock_api):
            from localm.inference.embedder import GGUFEmbedder
            GGUFEmbedder("embed.gguf", main_gpu=1)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 1


class TestCooperativeUnloadAsksDefaultSplitPeers:
    """With no configured split, the peers asked to free VRAM are those on the
    GPUs whose free VRAM the check before a load sums."""

    def _asked(self, monkeypatch, gpus, peer_gpu):
        from localm.inference import http_server
        asked = []
        peers = [{"instance_id": "p", "port": 2, "model": "x", "gpu_index": peer_gpu}]
        monkeypatch.setattr(http_server, "_gpu_coord",
                            {"instance_id": "i", "port": 1})
        monkeypatch.setattr("localm.gpu_registry.list_gpu_peers",
                            lambda exclude_self_id=None: peers)
        monkeypatch.setattr("localm.gpu_registry.request_cooperative_unload",
                            lambda peer: asked.append(peer["gpu_index"]) or False)
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=_config(
                    gpu_split_indices=None, main_gpu_index=None)):
            http_server._attempt_cooperative_unload(asked=set())
        return asked

    def test_a_peer_on_the_second_default_split_gpu_is_asked(self, monkeypatch):
        gpus = _torch_readings((20.0, False), (20.0, False))
        assert self._asked(monkeypatch, gpus, 1) == [1]

    def test_a_peer_on_an_integrated_gpu_left_out_is_not_asked(self, monkeypatch):
        gpus = _torch_readings((1.0, True), (20.0, False), (20.0, False))
        assert self._asked(monkeypatch, gpus, 0) == []
        assert self._asked(monkeypatch, gpus, 2) == [2]

    def test_one_gpu_is_unchanged(self, monkeypatch):
        gpus = _torch_readings((20.0, False))
        assert self._asked(monkeypatch, gpus, 0) == [0]
        assert self._asked(monkeypatch, gpus, 1) == []
