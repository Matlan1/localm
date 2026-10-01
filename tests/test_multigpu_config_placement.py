# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configured multi-GPU settings land on the GPUs they name.

On a CUDA/HIP build torch numbers every GPU, integrated ones included, while
llama.cpp's device list leaves out an integrated GPU whenever a discrete one
is present. A configured split, its main GPU, the VRAM readout, the check
before a load, the GPU registry entry and the context ceiling all have to
describe the devices the load actually uses.
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

GiB = 1024 ** 3


@contextlib.contextmanager
def _box(gpus, registry):
    """A CUDA/HIP build whose torch probe reports *gpus* (through every probe
    entry point, and as the last completed reading) and whose native registry
    reports *registry*."""
    def _list(*_a, **kw):
        return (list(gpus), discover.GPU_PROBE_OK) if kw.get("return_status") else list(gpus)

    with mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                           return_value=False), \
            mock.patch.object(discover, "list_gpus", _list), \
            mock.patch.object(discover, "last_gpu_reading", lambda: list(gpus)), \
            mock.patch.object(_loader, "gpu_devices_isolated",
                              return_value=registry) as reg, \
            mock.patch.object(_loader, "probe_daemon_running", return_value=True), \
            mock.patch.object(_loader, "stop_probe_daemon"):
        yield reg


def _load(b, gpus, registry, cfg, *, ctx_max=4096, registry_reads=None):
    """Run the backend's real ``_load_native`` on a torch box with *cfg* as the
    config and return the params handed to the worker."""
    captured = {}

    def _fake_spawn(self_runner, params, cancel_event=None, timeout=None,
                    on_progress=None):
        captured.update(params)
        return {"n_layers": 4}

    ctx_patch = (mock.patch.object(GgufBackend, "_effective_ctx_max",
                                   return_value=ctx_max)
                 if ctx_max is not None else contextlib.nullcontext())
    with _box(gpus, registry) as reg, ctx_patch, \
            mock.patch("localm.config.load_config", return_value=cfg), \
            mock.patch.object(_loader, "native_lib_loaded", return_value=False), \
            mock.patch("localm.inference.backends.llamacpp._sizing."
                       "embedder_ctx_reservation_bytes", return_value=0), \
            mock.patch("localm.inference.backends.llamacpp._runner."
                       "ModelRunner.spawn_and_load", _fake_spawn), \
            mock.patch("localm.model_meta.store_n_layers"):
        b.effective_gpu_layers = 99
        b._load_native()
    if registry_reads is not None:
        registry_reads.append(reg.call_count)
    return captured


def _worker_writes(params, cfg, gpus):
    """``(split_mode, main_gpu, tensor_split values or None)`` the worker writes
    into llama_model_params from *params*, the way llama.py calls
    ``apply_main_gpu`` and ``apply_gpu_split``, with *cfg* as the worker's
    config and *gpus* as its device reading."""
    class _MP:
        tensor_split = None
        split_mode = discover._LLAMA_SPLIT_MODE_LAYER
        main_gpu = 0
    mp = _MP()

    def _list(*_a, **kw):
        return (list(gpus), discover.GPU_PROBE_OK) if kw.get("return_status") else list(gpus)

    with mock.patch.object(discover, "_tensor_split_capacity", return_value=8), \
            mock.patch.object(discover, "_native_gpu_index_space_is_opaque",
                              return_value=False), \
            mock.patch.object(discover, "list_gpus", _list), \
            mock.patch("localm.config.load_config", return_value=cfg):
        if "main_gpu" in params:
            discover.apply_main_gpu(mp, slot=params["main_gpu"])
        else:
            discover.apply_main_gpu(mp)
        arr = discover.apply_gpu_split(mp, ratios_override=params["gpu_split_ratios"])
    values = None if arr is None else [arr[i] for i in range(8)]
    return mp.split_mode, mp.main_gpu, values


def _backend(tmp_path, **kw):
    b = GgufBackend(str(_tiny_model(tmp_path)), n_ctx=4096, n_gpu_layers=99, **kw)
    b._VRAM_OVERHEAD_BYTES = int(1.5e9)
    b._gguf_kv_bpt = 10
    return b


class TestConfiguredSplitInLlamaCppNumbering:
    """Gap 1: a configured split of 2+ GPUs is written in llama.cpp's device
    numbering when torch numbers an integrated GPU llama.cpp leaves out."""

    def test_a_split_after_an_integrated_gpu_lands_on_the_gpus_it_names(self, tmp_path):
        # Torch: 0 = iGPU, 1 and 2 = discrete. llama.cpp's devices 0 and 1 are
        # torch's 1 and 2, so the split [1, 2] is llama.cpp's [0, 1].
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[3.0, 1.0],
                      main_gpu_index=2)
        params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg)
        split_mode, main_gpu, values = _worker_writes(params, cfg, gpus)
        assert values == pytest.approx([3.0, 1.0, 0, 0, 0, 0, 0, 0]), (
            f"tensor_split written in the wrong numbering: {values}")
        assert split_mode == discover._LLAMA_SPLIT_MODE_LAYER
        assert main_gpu == 1, "main_gpu_index=2 is llama.cpp device 1"
        assert params["gpu_split_ratios"] == {0: 3.0, 1: 1.0}

    def test_automatic_shares_follow_the_renumbered_devices(self, tmp_path):
        gpus = _torch_readings((2.0, True), (30.0, False), (10.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=None,
                      main_gpu_index=None)
        b = _backend(tmp_path)
        params = _load(b, gpus, _registry(gpus), cfg)
        split = params["gpu_split_ratios"]
        assert sorted(split) == [0, 1]
        assert split[0] == pytest.approx(3 * split[1])
        assert "main_gpu" not in params
        assert b.applied_gpu_split["source"] == "auto"
        assert [d["index"] for d in b.applied_gpu_split["devices"]] == [1, 2]

    def test_a_split_naming_the_integrated_gpu_is_refused(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[0, 1], gpu_split_ratios=None)
        params, exc = None, None
        try:
            params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg)
        except discover.GpuSplitConfigError as e:
            exc = e
        assert params is None, f"a refused split reached the worker: {params}"
        assert exc is not None
        assert "GPU 0" in str(exc) and "integrated" in str(exc)

    def test_the_refusal_reaches_the_caller_without_runtime_advice(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[0, 1, 2], gpu_split_ratios=None)
        b = _backend(tmp_path)
        spawned, exc = [], None
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=cfg), \
                mock.patch.object(_loader, "native_lib_loaded", return_value=False), \
                mock.patch.object(GgufBackend, "_effective_gpu_layers", return_value=99), \
                mock.patch.object(GgufBackend, "_check_vram"), \
                mock.patch("localm.inference.backends.llamacpp._runner."
                           "ModelRunner.spawn_and_load",
                           side_effect=lambda *a, **k: spawned.append(1) or {}), \
                mock.patch("localm.model_meta.store_n_layers"):
            try:
                b.load()
            except RuntimeError as e:
                exc = e
        assert spawned == [], "a refused split must not reach the worker"
        assert isinstance(exc, discover.GpuSplitConfigError)
        assert "setup-llama" not in str(exc)

    def test_an_unproven_numbering_keeps_llamacpps_default_split(self, tmp_path, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        registry = _registry(gpus)
        registry.append(dict(registry[2], index=3))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[3.0, 1.0])
        with caplog.at_level("WARNING", logger="localm"):
            params = _load(_backend(tmp_path), gpus, registry, cfg)
        assert params["gpu_split_ratios"] == {}
        assert _worker_writes(params, cfg, gpus) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 0, None)
        assert any("keeping llama.cpp's default split" in r.getMessage()
                   for r in caplog.records)

    def test_without_an_integrated_gpu_a_configured_split_is_unchanged(self, tmp_path):
        gpus = _torch_readings((20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[0, 1], gpu_split_ratios=[3.0, 1.0],
                      main_gpu_index=1)
        reads = []
        params = _load(_backend(tmp_path), gpus, _registry(gpus), cfg,
                       registry_reads=reads)
        assert params["gpu_split_ratios"] is None
        assert "main_gpu" not in params
        assert reads == [0], "a box without an integrated GPU needs no registry read"
        assert _worker_writes(params, cfg, gpus) == (
            discover._LLAMA_SPLIT_MODE_LAYER, 1,
            pytest.approx([3.0, 1.0, 0, 0, 0, 0, 0, 0]))

    def test_a_one_entry_list_naming_the_integrated_gpu_is_refused(self, tmp_path):
        gpus = _torch_readings((2.0, True), (20.0, False))
        cfg = _config(gpu_split_indices=[0], gpu_split_ratios=None)
        with pytest.raises(discover.GpuSplitConfigError):
            _load(_backend(tmp_path), gpus, _registry(gpus), cfg)

    def test_a_one_entry_list_carries_its_main_gpu_in_llamacpps_numbering(
            self, tmp_path, caplog):
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[2], gpu_split_ratios=None, main_gpu_index=2)
        b = _backend(tmp_path)
        params = _load(b, gpus, _registry(gpus), cfg)
        with caplog.at_level("WARNING", logger="localm"):
            assert _worker_writes(params, cfg, gpus) == (
                discover._LLAMA_SPLIT_MODE_NONE, 1, None)
        assert not any("is not the device this load runs on" in r.getMessage()
                       for r in caplog.records)
        assert b.load_gpu_index == 2

    def test_the_embedder_gets_the_same_placement(self, monkeypatch, tmp_path):
        captured = {}

        class _Runner:
            def spawn_and_load(self, params, timeout=None):
                captured.update(params)
                return {"dim": 768, "n_ctx": 512}

        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"\0" * (2 * 1024 * 1024))
        gpus = _torch_readings((2.0, True), (20.0, False), (20.0, False))
        cfg = _config(gpu_split_indices=[1, 2], gpu_split_ratios=[1.0, 1.0],
                      main_gpu_index=2)
        monkeypatch.setattr("localm.inference._embedder_runner.EmbedderRunner", _Runner)
        from localm.inference.embedder import IsolatedEmbedder
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=cfg):
            IsolatedEmbedder(str(model_file))
        assert captured["gpu_split_ratios"] == {0: 1.0, 1: 1.0}
        assert captured["main_gpu"] == 1


class TestReadingsFollowTheLoadDevice:
    """Gap 3: the VRAM readout, the check before a load and the GPU registry
    describe the device the load runs on."""

    @staticmethod
    def _gpus():
        return [{"index": 0, "name": "small", "free": 4 * GiB, "total": 8 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE},
                {"index": 1, "name": "big", "free": 20 * GiB, "total": 24 * GiB,
                 "free_scope": discover.FREE_SCOPE_DEVICE}]

    def test_vram_info_reads_the_gpu_a_one_entry_list_names(self):
        gpus = self._gpus()
        cfg = _config(gpu_split_indices=[1], main_gpu_index=None)
        with _box(gpus, _registry([dict(g, integrated=False) for g in gpus])), \
                mock.patch("localm.config.load_config", return_value=cfg):
            info = discover.vram_info()
            capacity = discover.vram_capacity()
        assert info["total"] == 24 * GiB and info["free"] == 20 * GiB
        assert capacity["free"] == 20 * GiB

    @pytest.mark.parametrize("igpu_first", [True, False])
    def test_vram_info_reads_the_only_discrete_gpu_beside_an_integrated_one(
            self, igpu_first):
        specs = [(1.0, True), (20.0, False)]
        if not igpu_first:
            specs.reverse()
        gpus = _torch_readings(*specs)
        dgpu = next(g for g in gpus if not g["integrated"])
        cfg = _config(gpu_split_indices=None, main_gpu_index=None)
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=cfg):
            assert discover.vram_info()["total"] == dgpu["total"]

    def test_the_check_before_a_load_admits_a_model_the_chosen_gpu_holds(self):
        from localm.inference import http_server, switch_admission
        gpus = self._gpus()
        cfg = _config(gpu_split_indices=[1], main_gpu_index=None)
        budget = switch_admission.LoadBudget(
            name="m", vram_required=10 * GiB, headroom=GiB, resident_cap=None,
            pinned=frozenset(), check_split_fit=True)

        async def _probe():
            return await http_server._switch_probe_vram(
                asyncio.get_running_loop(), budget)

        with _box(gpus, _registry([dict(g, integrated=False) for g in gpus])), \
                mock.patch("localm.config.load_config", return_value=cfg):
            probe = asyncio.run(_probe())
        decision = switch_admission.decide_admission(probe, budget, [], {})
        assert probe.free == 20 * GiB
        assert decision.action == switch_admission.ADMIT

    def _registry_entry(self, monkeypatch, cfg, gpus, engines, active):
        from localm.inference import http_server
        written = {}
        monkeypatch.setattr(http_server, "_gpu_coord",
                            {"instance_id": "i", "token": "t", "port": 1})
        monkeypatch.setattr(http_server, "_engines", engines)
        monkeypatch.setattr(http_server, "_active_model_name", active)
        monkeypatch.setattr(http_server, "_loaded_model_identities", lambda: [])
        monkeypatch.setattr(http_server, "_model_file_size", lambda n: None)
        monkeypatch.setattr("localm.gpu_registry.registry_dir", lambda: "unused")
        monkeypatch.setattr("localm.gpu_registry.write_entry",
                            lambda _d, **kw: written.update(kw))
        with _box(gpus, _registry([dict(g, integrated=False) for g in gpus])), \
                mock.patch("localm.config.load_config", return_value=cfg):
            http_server._gpu_registry_sync()
        return written

    def test_the_registry_names_the_gpu_a_one_entry_list_names(self, monkeypatch):
        written = self._registry_entry(
            monkeypatch, _config(gpu_split_indices=[1], main_gpu_index=None),
            self._gpus(), {}, None)
        assert written["gpu_index"] == 1

    def test_the_registry_names_the_gpu_a_one_device_fit_loaded_on(self, monkeypatch):
        engine = SimpleNamespace(loaded=True,
                                 _backend=SimpleNamespace(load_gpu_index=1))
        written = self._registry_entry(
            monkeypatch, _config(gpu_split_indices=None, main_gpu_index=None),
            self._gpus(), {"m": engine}, "m")
        assert written["gpu_index"] == 1

    def test_a_one_device_fit_records_its_gpu_in_torch_numbering(self, tmp_path):
        # Torch 0 is an iGPU. The default split over the discrete GPUs puts the
        # output layer and a 41 MB logits buffer on the 30 MB device, so the
        # load runs on llama.cpp device 0, which is torch GPU 1.
        b = GgufBackend(str(_tiny_model(tmp_path, vocab=5000)), n_ctx=4096,
                        n_gpu_layers=99)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        gpus = [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": ig}
                for i, (f, ig) in enumerate([(2_000_000_000, True), (60_000_000, False),
                                             (30_000_000, False)])]
        params = _load(b, gpus, _registry(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None))
        assert params["gpu_split_ratios"] == {0: 1.0}
        assert b.load_gpu_index == 1


class TestAdmissionJudgesTheDefaultSplit:
    """With no configured split, llama.cpp spreads a GGUF load over every
    discrete GPU, and the backend sizes it against their summed free VRAM. The
    check before the load judges the same budget, so a model that fits across
    the GPUs, or on one GPU other than GPU 0, does not trigger evictions or a
    degraded-load prompt."""

    @staticmethod
    def _budget(gguf=True):
        from localm.inference import switch_admission
        return switch_admission.LoadBudget(
            name="m", vram_required=10 * GiB, headroom=GiB, resident_cap=None,
            pinned=frozenset(), check_split_fit=gguf)

    @staticmethod
    def _gpus(*specs):
        return [{"index": i, "name": f"gpu{i}", "free": int(f * GiB),
                 "total": int((f + 4) * GiB), "free_scope": discover.FREE_SCOPE_DEVICE,
                 "integrated": ig}
                for i, (f, ig) in enumerate(specs)]

    def _probe(self, gpus, budget):
        from localm.inference import http_server

        async def _run():
            return await http_server._switch_probe_vram(
                asyncio.get_running_loop(), budget)

        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=_config(
                    gpu_split_indices=None, main_gpu_index=None)):
            return asyncio.run(_run())

    def test_a_model_that_fits_across_the_gpus_is_admitted(self):
        from localm.inference import switch_admission
        budget = self._budget()
        gpus = self._gpus((4.0, False), (20.0, False))
        probe = self._probe(gpus, budget)
        decision = switch_admission.decide_admission(probe, budget, [], {})
        assert decision.action == switch_admission.ADMIT, (
            f"judged against one GPU: {probe.free / GiB:.1f} GB free")
        assert probe.free == 24 * GiB and probe.implicit_split

    def test_integrated_gpus_are_not_summed(self):
        budget = self._budget()
        gpus = self._gpus((30.0, True), (4.0, False), (4.0, False))
        probe = self._probe(gpus, budget)
        assert probe.free == 8 * GiB and probe.implicit_split

    def test_an_hf_load_is_still_judged_by_one_gpu(self):
        budget = self._budget(gguf=False)
        gpus = self._gpus((4.0, False), (20.0, False))
        probe = self._probe(gpus, budget)
        assert probe.free == 4 * GiB and not probe.implicit_split

    def test_one_gpu_is_judged_by_its_own_reading(self):
        budget = self._budget()
        gpus = self._gpus((20.0, False))
        probe = self._probe(gpus, budget)
        assert probe.free == 20 * GiB and not probe.implicit_split

    def test_the_release_wait_reads_the_same_sum(self):
        from localm.inference import http_server, switch_admission
        gpus = self._gpus((4.0, False), (20.0, False))
        probe = switch_admission.VramProbe(
            free=24 * GiB, probe_ok=True, process_scoped=False, shortfall=[],
            shares_adaptive=False, implicit_split=True)
        with _box(gpus, _registry(gpus)), \
                mock.patch("localm.config.load_config", return_value=_config(
                    gpu_split_indices=None)):
            assert http_server._probe_free_reader(probe)() == 24 * GiB


class TestContextCeilingFollowsTheFit:
    """Gap 4: the auto context ceiling counts only the devices the implicit
    split fit keeps."""

    _VOCAB = 5000

    def _fit_backend(self, tmp_path):
        b = GgufBackend(str(_tiny_model(tmp_path, vocab=self._VOCAB)), n_ctx=4096,
                        n_gpu_layers=99, ctx_auto=True, n_ctx_max=0)
        b._VRAM_OVERHEAD_BYTES = 1000
        b._gguf_kv_bpt = 10
        return b

    @staticmethod
    def _readings(frees):
        return [{"index": i, "name": f"gpu{i}", "free": f, "total": f + 10,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}
                for i, f in enumerate(frees)]

    def _expected(self, b, free, devices):
        budget = free - b._effective_model_bytes_for_vram() - 1000 * devices
        return max(b._AUTO_CTX_MIN, (budget // 10 // 1024) * 1024)

    def test_a_left_out_gpu_is_not_counted_in_the_context_ceiling(self, tmp_path):
        # The 30 MB device is left out; devices 0 and 1 hold 100 MB between them.
        b = self._fit_backend(tmp_path)
        gpus = self._readings([50_000_000, 50_000_000, 30_000_000])
        params = _load(b, gpus, _registry(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None),
                       ctx_max=None)
        assert sorted(params["gpu_split_ratios"]) == [0, 1]
        assert params["n_ctx_max"] == self._expected(b, 100_000_000, 2)

    def test_a_one_gpu_fit_sizes_the_context_from_that_gpu(self, tmp_path):
        b = self._fit_backend(tmp_path)
        gpus = self._readings([60_000_000, 30_000_000])
        params = _load(b, gpus, _registry(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None),
                       ctx_max=None)
        assert params["gpu_split_ratios"] == {0: 1.0}
        assert params["n_ctx_max"] == self._expected(b, 60_000_000, 1)

    def test_a_default_split_that_fits_still_counts_every_gpu(self, tmp_path):
        b = self._fit_backend(tmp_path)
        gpus = self._readings([10 ** 9, 10 ** 9])
        params = _load(b, gpus, _registry(gpus),
                       _config(gpu_split_indices=None, main_gpu_index=None),
                       ctx_max=None)
        assert params["gpu_split_ratios"] is None
        assert params["n_ctx_max"] == self._expected(b, 2 * 10 ** 9, 2)
