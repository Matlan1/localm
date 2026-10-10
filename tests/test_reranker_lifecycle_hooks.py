# SPDX-License-Identifier: AGPL-3.0-or-later
"""The resident reranker is released by the same server paths that release the
shared embedder: unload-all (honouring an in-flight request), the VRAM eviction
that makes room for a chat model, and the shutdown and restart exits (lock-free,
so a worker never outlives the server holding its model in VRAM).
"""

import asyncio
import os
from types import SimpleNamespace

import pytest

from localm.inference import embedder as emb
from localm.inference import http_server as hs
from localm.inference import reranker as rr


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    yield


class TestUnloadAll:
    def test_an_idle_reranker_is_released_with_the_pin_check_made_atomically(self, isolated, monkeypatch):
        calls = []
        monkeypatch.setattr(rr, "is_loaded", lambda: True)
        monkeypatch.setattr(rr, "reset_reranker", lambda force=True: (calls.append(force), True)[1])
        res = asyncio.run(hs.unload_all_models())
        assert calls == [False]
        assert res["embedder_unloaded"] is True and res["status"] == "unloaded"

    def test_a_pinned_reranker_is_skipped_and_reported(self, isolated, monkeypatch):
        monkeypatch.setattr(rr, "is_loaded", lambda: True)
        monkeypatch.setattr(rr, "reset_reranker", lambda force=True: False)
        res = asyncio.run(hs.unload_all_models())
        assert "reranker model" in res["skipped_in_use"]
        assert res["status"] == "in_use" and res["embedder_unloaded"] is False

    def test_nothing_is_released_when_no_reranker_is_loaded(self, isolated, monkeypatch):
        calls = []
        monkeypatch.setattr(rr, "is_loaded", lambda: False)
        monkeypatch.setattr(rr, "reset_reranker", lambda force=True: calls.append(force))
        asyncio.run(hs.unload_all_models())
        assert calls == []


class TestUnloadOne:
    RERANK_PATH = "Z:/models/rerank.gguf"

    def _resident(self, monkeypatch, *, active=0, clears=True):
        monkeypatch.setattr("localm.config.load_registry", lambda: {
            "rr-model": {"path": self.RERANK_PATH},
            "other": {"path": "Z:/models/other.gguf"}})
        monkeypatch.setattr(emb, "loaded_path", lambda: None)
        monkeypatch.setattr(rr, "reranker_info",
                            lambda: {"path": self.RERANK_PATH, "labels": []})
        monkeypatch.setattr(rr, "active_requests", lambda: active)
        resets = []
        monkeypatch.setattr(rr, "reset_reranker",
                            lambda force=True: (resets.append(force), clears)[1])
        return resets

    def test_an_idle_resident_reranker_is_released_by_its_registered_name(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch)
        res = asyncio.run(hs.unload_one_model("rr-model"))
        assert resets == [False]
        assert res["status"] == "unloaded" and res["model"] == "rr-model"

    def test_a_pin_arriving_after_the_precheck_is_reported_in_use(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch, active=0, clears=False)
        res = asyncio.run(hs.unload_one_model("rr-model"))
        assert resets == [False]
        assert res["status"] == "in_use"

    def test_a_busy_reranker_is_rejected_before_the_vram_probe(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch, active=1)
        probes = []
        monkeypatch.setattr("localm.vram._vram_free_reading",
                            lambda: (probes.append(1), (None, True, None))[1])
        res = asyncio.run(hs.unload_one_model("rr-model"))
        assert res["status"] == "in_use"
        assert probes == [] and resets == []

    def test_another_registered_model_leaves_the_reranker_alone(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch)
        res = asyncio.run(hs.unload_one_model("other"))
        assert res["status"] == "already_unloaded"
        assert resets == []

    def test_nothing_is_released_when_no_reranker_is_resident(self, isolated, monkeypatch):
        resets = self._resident(monkeypatch)
        monkeypatch.setattr(rr, "reranker_info", lambda: None)
        res = asyncio.run(hs.unload_one_model("rr-model"))
        assert res["status"] == "already_unloaded"
        assert resets == []


def _evict(monkeypatch, *, embedder_loaded, reranker_loaded, embedder_clears=True, reranker_clears=True):
    resets = []
    embedder_mod = SimpleNamespace(
        loaded_dim=lambda: 384 if embedder_loaded else None,
        reset_embedder=lambda force=True: (resets.append("embedder"), embedder_clears)[1])
    monkeypatch.setattr(rr, "is_loaded", lambda: reranker_loaded)
    monkeypatch.setattr(rr, "reset_reranker",
                        lambda force=True: (resets.append("reranker"), reranker_clears)[1])
    probe = SimpleNamespace(measurable=False, free=0)
    attempt = SimpleNamespace(embedder_attempted=False)
    loop = asyncio.new_event_loop()
    try:
        freed = loop.run_until_complete(hs._switch_evict_embedder(loop, probe, attempt, embedder_mod))
    finally:
        loop.close()
    return freed, attempt, resets


class TestVramEviction:
    def test_an_idle_reranker_alone_is_freed_to_make_room(self, monkeypatch):
        freed, attempt, resets = _evict(monkeypatch, embedder_loaded=False, reranker_loaded=True)
        assert freed is True and resets == ["reranker"] and attempt.embedder_attempted is True

    def test_both_are_freed_when_both_are_resident(self, monkeypatch):
        freed, _attempt, resets = _evict(monkeypatch, embedder_loaded=True, reranker_loaded=True)
        assert freed is True and resets == ["embedder", "reranker"]

    def test_a_reranker_with_a_request_in_flight_is_left_alone(self, monkeypatch):
        freed, attempt, resets = _evict(monkeypatch, embedder_loaded=False, reranker_loaded=True,
                                        reranker_clears=False)
        assert freed is False and resets == ["reranker"] and attempt.embedder_attempted is True

    def test_the_embedder_is_still_freed_when_the_reranker_is_busy(self, monkeypatch):
        freed, _attempt, resets = _evict(monkeypatch, embedder_loaded=True, reranker_loaded=True,
                                         reranker_clears=False)
        assert freed is True and resets == ["embedder", "reranker"]

    def test_nothing_resident_frees_nothing_and_does_not_mark_an_attempt(self, monkeypatch):
        freed, attempt, resets = _evict(monkeypatch, embedder_loaded=False, reranker_loaded=False)
        assert freed is False and resets == [] and attempt.embedder_attempted is False


def _spy_exit_release(monkeypatch):
    reset_calls, released = [], []
    monkeypatch.setattr(rr, "reset_reranker", lambda *a, **k: reset_calls.append(1))
    monkeypatch.setattr(rr, "release_for_exit", lambda: (released.append(1), True)[1])
    monkeypatch.setattr(emb, "release_for_exit", lambda: True)
    monkeypatch.setattr(hs, "_engine", None)
    return reset_calls, released


class TestExitPaths:
    def test_shutdown_releases_the_reranker_worker_without_the_load_lock(self, monkeypatch):
        reset_calls, released = _spy_exit_release(monkeypatch)
        monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
        try:
            hs._do_shutdown()
        except SystemExit:
            pass
        assert released == [1] and reset_calls == []

    def test_restart_releases_the_reranker_worker_without_the_load_lock(self, monkeypatch):
        reset_calls, released = _spy_exit_release(monkeypatch)
        monkeypatch.setattr(os, "execv", lambda exe, argv: (_ for _ in ()).throw(SystemExit(0)))
        try:
            hs._do_restart()
        except SystemExit:
            pass
        assert released == [1] and reset_calls == []
