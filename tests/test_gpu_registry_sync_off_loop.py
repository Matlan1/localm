# SPDX-License-Identifier: AGPL-3.0-or-later
"""_gpu_registry_sync() must never run on the server event loop.

It does blocking work on every model load/unload: a registry temp-file
write + os.replace, a _model_file_size() stat/rglob walk, and - when a non-zero
main_gpu_index is configured - _current_gpu_index() -> resolve_main_gpu_index()
-> discover.list_gpus(), a real torch/nvidia-smi hardware probe bounded to a 4s
deadline. On the single event loop that stalls EVERY concurrent request and
stream for up to that deadline.

These tests assert the property directly - the sync work runs on a thread OTHER
than the event-loop thread - rather than trying to measure a stall.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localm.inference import http_server as hs


class _ThreadProbe:
    """Stands in for _gpu_registry_sync, recording which thread it ran on."""

    def __init__(self):
        self.threads = []

    def __call__(self):
        self.threads.append(threading.get_ident())


@pytest.fixture
def probe(monkeypatch):
    p = _ThreadProbe()
    monkeypatch.setattr(hs, "_gpu_registry_sync", p)
    # No real hardware probe / VRAM wait in a unit test.
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": 32 * 1024 ** 3,
                                             "total": 32 * 1024 ** 3})
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return p


class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


def _install_loaded(name):
    eng = FakeEngine(name)
    eng.load()
    hs._engines[name] = eng
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    hs._active_model_name = name
    hs._engine = eng
    hs._inference_sem = hs._inference_sems[name]
    return eng


def _assert_off_loop(probe, loop_thread, where):
    assert probe.threads, f"_gpu_registry_sync never ran in {where}"
    assert all(t != loop_thread for t in probe.threads), (
        f"{where} ran the gpu-registry sync (registry file I/O + a GPU driver "
        f"probe) ON the event loop thread, stalling every concurrent request")


def test_switch_engine_syncs_registry_off_the_loop(probe):
    """Every successful load hits this path."""

    async def scenario():
        engines = {"A": FakeEngine("A")}
        await hs.switch_engine("A", engines.__getitem__)
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "switch_engine")


def test_unload_all_models_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_all_models()
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_all_models")


def test_unload_one_model_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_one_model("A")
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_one_model")


def test_unload_embedder_if_matches_syncs_registry_off_the_loop(probe, monkeypatch, tmp_path):
    """The targeted-unload counterpart: loaded_path(), the active_requests()
    precheck, reset_embedder(force=False) and the VRAM wait are already
    offloaded, and the registry sync must be too."""
    model = tmp_path / "emb.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: str(model))
    monkeypatch.setattr("localm.inference.embedder.active_requests", lambda: 0)
    monkeypatch.setattr("localm.inference.embedder.reset_embedder",
                        lambda force=True: True)
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb": {"path": str(model), "source": "local"}})
    monkeypatch.setattr("localm.model_manager._entry_path", lambda entry: str(model))

    async def scenario():
        loop = asyncio.get_running_loop()
        res = await hs._unload_embedder_if_matches("emb", loop)
        assert res is not None and res["status"] == "unloaded", res
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "_unload_embedder_if_matches")
