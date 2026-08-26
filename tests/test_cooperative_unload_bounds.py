# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cooperative cross-instance unload must be bounded and worth its cost.

When two localm instances share a GPU and this one's local eviction candidates
are all busy, switch_engine asks a sibling to release VRAM instead of 503-ing at
once. Cooperating costs the sibling EVERY model it has loaded (the peer runs its
own unload_all_models), and the caller drives it from a `while True` that
re-probes VRAM and `continue`s on success without changing any local state. Two
consequences this pins down:

1. NON-PROGRESSING LOOP. request_cooperative_unload reports success for
   "already_unloaded" too, and a peer's own post-unload registry write is
   best-effort (a failure is logged, not raised), so its entry can keep
   advertising a model it no longer holds. The loop then re-asks the same peer
   forever, holding the per-model semaphore and re-probing VRAM, never
   progressing. Each peer must be asked at most once per load attempt.

2. YANKING A SIBLING FOR NOTHING. Freeing the peer must be checked to actually
   make this model fit: if the model needs far more than the card has, or a
   third-party app (ComfyUI) holds the bulk of VRAM, the sibling loses every
   model and this load still 503s.

_attempt_cooperative_unload returns False immediately unless the module global
_gpu_coord is set, and _gpu_coord is populated ONLY in the lifespan startup of a
real, non-isolated, advertised server, so under pytest the entire cooperative
branch is dead code. These tests set _gpu_coord explicitly to reach it.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from localm.inference import http_server as hs
from tests.conftest import probe_double


GB = 1024 ** 3

# The peer stops cooperating after this many requests, so an UNBOUNDED retry
# loop terminates instead of hanging the suite; the recorded call list shows how
# many times the peer was actually asked.
_MAX_COOPERATE = 4


class FakeEngine:
    def __init__(self, name, *, fails_to_fit=False):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0
        # Simulates a REAL backend's own final sizing decision (GgufBackend's
        # _check_vram, llamacpp/_sizing.py). switch_engine defers to the backend
        # rather than refusing on its own estimate, so the fake reproduces that
        # final refusal when the model cannot fit even at 0 GPU layers.
        self.fails_to_fit = fails_to_fit

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self.fails_to_fit:
            raise RuntimeError(
                "Context too large for available VRAM: this load needs more "
                "than this GPU has in total - freeing other VRAM will not "
                "help, it cannot fit regardless.")
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


@pytest.fixture
def coordinated(monkeypatch):
    """A registered, coordination-enabled instance whose GPU is too full for the
    incoming model, with no local eviction candidate (nothing else loaded).

    The model path does not exist on disk, so switch_engine falls back to its
    own 4 GB estimate: the load needs 4 GB * 1.2 + 1 GB headroom = ~5.8 GB
    against the 2 GB free below, and can never fit locally."""
    monkeypatch.setattr(hs, "_gpu_coord", {
        "instance_id": "self-1", "port": 8081, "host": "127.0.0.1",
        "scheme": "http", "token": "t",
    })
    monkeypatch.setattr(hs, "_current_gpu_index", lambda: 0)
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    # 2 GB free; incoming model is 10 GB * 1.2 + 1 GB headroom -> never fits.
    monkeypatch.setattr("localm.discover.vram_capacity",
                        probe_double({"free": 2 * GB, "total": 16 * GB}))
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    reg = {"incoming": {"path": "models/incoming.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "hint"))
    monkeypatch.setattr("localm.inference.engine._is_gguf", lambda p: False)
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None


def _install_peers(monkeypatch, peers, on_request):
    calls = []

    def _list(**kw):
        return list(peers)

    def _request(peer, **kw):
        calls.append(peer.get("instance_id"))
        if len(calls) > _MAX_COOPERATE:
            return False        # bound the loop
        return on_request(peer)

    monkeypatch.setattr("localm.gpu_registry.list_gpu_peers", _list)
    monkeypatch.setattr("localm.gpu_registry.request_cooperative_unload", _request)
    return calls


def test_peer_is_not_re_asked_forever_when_it_keeps_advertising_a_model(
        coordinated, monkeypatch):
    """The non-progressing loop: a peer whose post-unload registry write failed
    keeps advertising a model and keeps answering 'already_unloaded' (True), so
    the caller must NOT keep asking it. One ask, then an honest 503. (Without
    the bound the real loop never returns at all; the fake peer stops
    cooperating after _MAX_COOPERATE so this test reports the count instead of
    hanging.)"""
    peer = {"instance_id": "peer-1", "port": 8082, "model": "other",
            "vram_estimate_bytes": 12 * GB, "gpu_index": 0,
            "coordination_token": "x"}
    calls = _install_peers(monkeypatch, [peer], lambda p: True)  # always "freed"

    engine = FakeEngine("incoming", fails_to_fit=True)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(hs.switch_engine("incoming", {"incoming": engine}.__getitem__))
    assert ei.value.status_code == 503
    assert calls == ["peer-1"], (
        f"the peer must be asked exactly once per load attempt, got {calls}")


def test_peer_models_are_not_yanked_when_freeing_them_cannot_help(
        coordinated, monkeypatch):
    """The fit check: the peer holds only 1 GB, so freeing it (2 + 1 = 3 GB)
    still leaves a 13 GB requirement short. Destroying the sibling's models
    would not make this load succeed, so it must not be asked at all."""
    peer = {"instance_id": "peer-1", "port": 8082, "model": "small",
            "vram_estimate_bytes": 1 * GB, "gpu_index": 0,
            "coordination_token": "x"}
    calls = _install_peers(monkeypatch, [peer], lambda p: True)

    engine = FakeEngine("incoming", fails_to_fit=True)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(hs.switch_engine("incoming", {"incoming": engine}.__getitem__))
    assert ei.value.status_code == 503
    assert calls == [], (
        "a sibling's models were yanked even though freeing them could not "
        f"make this load fit: {calls}")


def test_peer_on_a_different_gpu_is_not_yanked(coordinated, monkeypatch):
    """Freeing another card's VRAM cannot make this load fit."""
    peer = {"instance_id": "peer-1", "port": 8082, "model": "other",
            "vram_estimate_bytes": 12 * GB, "gpu_index": 1,
            "coordination_token": "x"}
    calls = _install_peers(monkeypatch, [peer], lambda p: True)

    engine = FakeEngine("incoming", fails_to_fit=True)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(hs.switch_engine("incoming", {"incoming": engine}.__getitem__))
    assert ei.value.status_code == 503
    assert calls == [], f"a peer on another GPU was yanked: {calls}"


def test_split_instance_still_asks_a_peer_on_its_other_split_device(
        coordinated, monkeypatch):
    """The same-GPU filter must offer EVERY device this load can actually use,
    not one scalar index.

    With gpu_split_indices=[0,1] and main_gpu_index unset, _current_gpu_index()
    resolves to 0 (resolve_main_gpu_index(None) -> device 0, no detection). A
    sibling pinned to GPU 1 advertises gpu_index=1. But the free figure this
    decision weighs is vram_capacity()'s COMBINED split total, so part of the
    shortfall IS on GPU 1 and freeing that peer is exactly what makes the load
    fit. Comparing peers against a single index while comparing VRAM against the
    combined total contradicts itself: the peer is dropped, holders go empty, and
    the load 503s.
    """
    from localm.config import load_config as real_load_config
    base = real_load_config()
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {**base, "gpu_split_indices": [0, 1]})
    monkeypatch.setattr("localm.discover.list_gpus", probe_double([
        {"index": 0, "name": "A", "total": 16 * GB, "free": 1 * GB},
        {"index": 1, "name": "B", "total": 16 * GB, "free": 1 * GB},
    ]))
    peer = {"instance_id": "peer-1", "port": 8082, "model": "big",
            "vram_estimate_bytes": 12 * GB, "gpu_index": 1,
            "coordination_token": "x"}
    freed = {"yes": False}

    def _on_request(p):
        freed["yes"] = True
        return True

    calls = _install_peers(monkeypatch, [peer], _on_request)
    monkeypatch.setattr(
        "localm.discover.vram_capacity",
        probe_double(lambda: {"free": (14 * GB) if freed["yes"] else (2 * GB),
                              "total": 32 * GB}))

    engines = {"incoming": FakeEngine("incoming")}
    res = asyncio.run(hs.switch_engine("incoming", engines.__getitem__))
    assert calls == ["peer-1"], (
        "a peer holding VRAM on one of THIS instance's own split devices was "
        f"dropped by the same-GPU filter, so the split load never cooperated: {calls}")
    assert res["status"] == "loaded", res


def test_cooperation_still_happens_when_it_can_actually_free_enough(
        coordinated, monkeypatch):
    """The feature must still work: a peer holding enough IS asked, and once it
    frees the VRAM the load proceeds. Guards against 'fixing' the loop by
    disabling cooperation altogether."""
    peer = {"instance_id": "peer-1", "port": 8082, "model": "big",
            "vram_estimate_bytes": 12 * GB, "gpu_index": 0,
            "coordination_token": "x"}
    freed = {"yes": False}

    def _on_request(p):
        freed["yes"] = True
        return True

    calls = _install_peers(monkeypatch, [peer], _on_request)
    # After the peer frees, the next VRAM probe sees room.
    monkeypatch.setattr(
        "localm.discover.vram_capacity",
        probe_double(lambda: {"free": (14 * GB) if freed["yes"] else (2 * GB),
                              "total": 16 * GB}))

    engines = {"incoming": FakeEngine("incoming")}
    res = asyncio.run(hs.switch_engine("incoming", engines.__getitem__))
    assert res["status"] == "loaded", res
    assert calls == ["peer-1"]
    assert engines["incoming"].loaded
